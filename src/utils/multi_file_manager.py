"""
Multi-file management utilities for the data analytics system
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
import time
import numpy as np

class MultiFileDataManager:
    """Enhanced data manager for handling multiple files with intelligent merging and analysis"""
    
    @staticmethod
    def find_common_columns(df_manager) -> Dict[str, List[str]]:
        """Find common columns across all loaded files"""
        file_info = df_manager.get_file_info()
        if len(file_info) < 2:
            return {}
        
        # Get all column sets
        column_sets = {name: set(info['columns']) for name, info in file_info.items()}
        
        # Find intersections between all pairs
        common_columns = {}
        file_names = list(column_sets.keys())
        
        for i in range(len(file_names)):
            for j in range(i + 1, len(file_names)):
                file1, file2 = file_names[i], file_names[j]
                intersection = column_sets[file1] & column_sets[file2]
                if intersection:
                    pair_key = f"{file1} & {file2}"
                    common_columns[pair_key] = list(intersection)
        
        # Find columns common to ALL files
        if len(file_names) > 2:
            all_common = set.intersection(*column_sets.values())
            if all_common:
                common_columns["All Files"] = list(all_common)
        
        return common_columns
    
    @staticmethod
    def attempt_smart_merge(df_manager) -> Tuple[Optional[pd.DataFrame], str, List[str]]:
        """Create ONE unified merged file from ALL uploaded files"""
        file_info = df_manager.get_file_info()
        if len(file_info) < 2:
            return None, "Need at least 2 files for merging", []
        
        files_to_merge = [(name, df_manager.get_dataframe(name)) 
                         for name in file_info.keys() 
                         if name != 'merged_data' and df_manager.get_dataframe(name) is not None]
        
        if len(files_to_merge) < 2:
            return None, "Not enough valid files to merge", []
        
        print(f"[MultiFileManager] Creating ONE unified merged file from {len(files_to_merge)} files")
        
        # ALWAYS use union merge to create ONE file with ALL data
        try:
            # Get all unique columns across ALL files
            all_columns = set()
            for _, df in files_to_merge:
                all_columns.update(df.columns)
            
            all_columns = sorted(list(all_columns))
            print(f"[MultiFileManager] Total unique columns across all files: {len(all_columns)}")
            
            # Standardize ALL dataframes to have the same column structure
            standardized_dfs = []
            file_names = []
            
            for file_name, df in files_to_merge:
                print(f"[MultiFileManager] Processing {file_name} with {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Create a copy and convert all columns to compatible types
                standardized_df = df.copy()
                
                # Convert all columns to string to avoid type conflicts
                for col in standardized_df.columns:
                    try:
                        standardized_df[col] = standardized_df[col].astype(str)
                    except:
                        pass
                
                # Add missing columns with empty strings for columns not in this file
                for col in all_columns:
                    if col not in standardized_df.columns:
                        standardized_df[col] = ''
                
                # Reorder columns to match the standard order
                standardized_df = standardized_df[all_columns]
                
                # Add source file identifier to track which file each row came from
                standardized_df['_source_file'] = file_name
                
                standardized_dfs.append(standardized_df)
                file_names.append(file_name)
                print(f"[MultiFileManager] Standardized {file_name}: {standardized_df.shape[0]} rows, {standardized_df.shape[1]} columns")
            
            # Concatenate ALL standardized dataframes into ONE unified file
            merged_df = pd.concat(standardized_dfs, ignore_index=True, sort=False)
            
            total_rows = merged_df.shape[0]
            total_cols = merged_df.shape[1]
            
            merge_info = f"UNIFIED MERGE: Combined ALL {len(file_names)} files into one dataset ({total_rows} rows, {total_cols} columns)"
            print(f"[MultiFileManager] {merge_info}")
            
            return merged_df, merge_info, file_names
            
        except Exception as e:
            print(f"[MultiFileManager] CRITICAL ERROR - Union merge failed: {e}")
            # Emergency fallback - just concatenate without column standardization
            try:
                all_dfs = [df for _, df in files_to_merge]
                merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
                return merged_df, f"Emergency concatenation of {len(files_to_merge)} files", [name for name, _ in files_to_merge]
            except:
                pass
        
        return None, "CRITICAL: All merge strategies failed", [name for name, _ in files_to_merge]
    
    @staticmethod
    def get_file_summary(df_manager) -> str:
        """Get a formatted summary of all available files with merge analysis"""
        file_info = df_manager.get_file_info()
        if not file_info:
            return "No files currently loaded."
        
        summary = f"📊 **Available Datasets ({len(file_info)} files)**\n\n"
        
        for name, info in file_info.items():
            shape = info['shape']
            columns = info['columns']
            metadata = info.get('metadata', {})
            
            summary += f"🗂️ **{name}**\n"
            summary += f"   • Size: {shape[0]:,} rows × {shape[1]} columns\n"
            summary += f"   • Key columns: {', '.join(columns[:4])}{'...' if len(columns) > 4 else ''}\n"
            
            if metadata.get('file_type'):
                summary += f"   • Type: {metadata['file_type']}\n"
            
            summary += "\n"
        
        # Add merge analysis if multiple files
        if len(file_info) > 1:
            common_cols = MultiFileDataManager.find_common_columns(df_manager)
            if common_cols:
                summary += "🔗 **Merge Possibilities:**\n"
                for pair, cols in common_cols.items():
                    summary += f"   • {pair}: {', '.join(cols)}\n"
            else:
                summary += "⚠️ **No common columns found for merging**\n"
            summary += "\n"
        
        return summary
    
    @staticmethod
    def suggest_relevant_files(query: str, df_manager) -> List[str]:
        """Suggest which files might be relevant for a given query"""
        file_info = df_manager.get_file_info()
        query_lower = query.lower()
        
        suggestions = []
        
        for name, info in file_info.items():
            relevance_score = 0
            
            # Check filename relevance
            if any(word in name.lower() for word in query_lower.split()):
                relevance_score += 5
            
            # Check column relevance
            for col in info['columns']:
                if col.lower() in query_lower:
                    relevance_score += 10
                elif any(word in col.lower() for word in query_lower.split()):
                    relevance_score += 3
            
            if relevance_score > 0:
                suggestions.append((name, relevance_score))
        
        # Sort by relevance and return names
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in suggestions]
    
    @staticmethod
    def create_table_output(df: pd.DataFrame, file_name: str, max_rows: int = 10) -> str:
        """Create a formatted table output for display"""
        if df is None or df.empty:
            return f"[{file_name}] No data available"
        
        # Limit rows and columns for display
        display_df = df.head(max_rows)
        
        if display_df.shape[1] > 8:
            # Show first 7 columns plus last column
            cols_to_show = list(display_df.columns[:7]) + [display_df.columns[-1]]
            display_df = display_df[cols_to_show]
            col_note = f" (showing 8 of {df.shape[1]} columns)"
        else:
            col_note = ""
        
        result = f"📋 **Data from {file_name}**{col_note}\n\n"
        result += "DATAFRAME_START\n"
        result += display_df.to_string(index=True)
        result += "\nDATAFRAME_END\n"
        result += f"\n*Total: {df.shape[0]:,} rows × {df.shape[1]} columns*"
        
        return result
    
    @staticmethod
    def create_combined_analysis(df_manager, query: str) -> str:
        """Create analysis combining data from multiple files when appropriate"""
        file_info = df_manager.get_file_info()
        
        if len(file_info) < 2:
            return "Single file analysis - no combination needed"
        
        # Check if query suggests cross-file analysis
        cross_file_indicators = ['compare', 'across', 'between', 'all files', 'combined', 'merge']
        if not any(indicator in query.lower() for indicator in cross_file_indicators):
            return "Query doesn't indicate cross-file analysis needed"
        
        # Attempt smart merge
        merged_df, merge_info, merged_files = MultiFileDataManager.attempt_smart_merge(df_manager)
        
        if merged_df is not None:
            result = f"🔗 **Combined Analysis Result**\n\n"
            result += f"{merge_info}\n\n"
            
            # Show merge statistics
            if '_source_file' in merged_df.columns:
                source_counts = merged_df['_source_file'].value_counts()
                result += f"**Data Distribution by Source:**\n"
                for source, count in source_counts.items():
                    result += f"  • {source}: {count:,} rows\n"
                result += "\n"
            
            result += MultiFileDataManager.create_table_output(merged_df, "Merged Data", 15)
            return result
        else:
            # Provide individual file summaries
            result = f"📊 **Individual File Analysis** (merge not possible: {merge_info})\n\n"
            for name, info in file_info.items():
                df = df_manager.get_dataframe(name)
                if df is not None:
                    result += MultiFileDataManager.create_table_output(df, name, 5)
                    result += "\n\n"
            return result
    
    @staticmethod
    def get_merged_dataframe(df_manager) -> Tuple[Optional[pd.DataFrame], str]:
        """Get or create a merged dataframe from all available files"""
        # Check if merged_data already exists
        merged_df = df_manager.get_dataframe('merged_data')
        if merged_df is not None:
            return merged_df, "merged_data"
        
        # Attempt to create merged dataframe
        merged_df, merge_info, merged_files = MultiFileDataManager.attempt_smart_merge(df_manager)
        
        if merged_df is not None:
            # Store the merged dataframe
            df_manager.store_dataframe('merged_data', merged_df, {
                'merge_info': merge_info,
                'source_files': merged_files,
                'file_type': 'merged'
            })
            return merged_df, "merged_data"
        
        return None, "merge_failed"