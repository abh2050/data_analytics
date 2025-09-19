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
        """Attempt to intelligently merge ALL files based on common columns"""
        file_info = df_manager.get_file_info()
        if len(file_info) < 2:
            return None, "Need at least 2 files for merging", []
        
        common_cols = MultiFileDataManager.find_common_columns(df_manager)
        
        if not common_cols:
            return None, "No common columns found for merging", []
        
        # Check if there are columns common to ALL files
        if "All Files" in common_cols:
            all_common_cols = common_cols["All Files"]
            
            # Find the best merge column from all-common columns
            id_patterns = ['id', 'key', 'code', 'number', 'ref']
            best_merge_col = None
            best_priority = 0
            
            for col in all_common_cols:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in id_patterns):
                    priority = 10
                elif col_lower in ['date', 'time', 'timestamp']:
                    priority = 8
                else:
                    priority = 5
                
                if priority > best_priority:
                    best_merge_col = col
                    best_priority = priority
            
            if best_merge_col:
                try:
                    # Get all files except merged_data
                    files_to_merge = [(name, df_manager.get_dataframe(name)) 
                                    for name in file_info.keys() 
                                    if name != 'merged_data' and df_manager.get_dataframe(name) is not None]
                    
                    if len(files_to_merge) < 2:
                        return None, "Not enough valid files to merge", []
                    
                    # Start with first file
                    merged_df = files_to_merge[0][1].copy()
                    merged_files = [files_to_merge[0][0]]
                    
                    # Merge with each subsequent file
                    for file_name, df in files_to_merge[1:]:
                        # Check overlap before merging
                        overlap = set(merged_df[best_merge_col].dropna()) & set(df[best_merge_col].dropna())
                        if len(overlap) > 0:
                            # Create unique suffixes for each file
                            suffix_num = len(merged_files)
                            merged_df = pd.merge(merged_df, df, on=best_merge_col, how='inner', 
                                               suffixes=(f'_{merged_files[0]}' if suffix_num == 1 else '', f'_{file_name}'))
                            merged_files.append(file_name)
                    
                    if len(merged_df) > 0:
                        overlap_info = f"Successfully merged {len(merged_files)} files: {', '.join(merged_files)} on '{best_merge_col}'"
                        return merged_df, overlap_info, merged_files
                        
                except Exception as e:
                    print(f"Error merging all files: {e}")
        
        # Fallback: try pairwise merging with best overlap
        best_merge = None
        best_overlap_ratio = 0
        
        for pair, cols in common_cols.items():
            if pair == "All Files":
                continue
                
            try:
                file1, file2 = pair.split(' & ')
                df1 = df_manager.get_dataframe(file1)
                df2 = df_manager.get_dataframe(file2)
                
                if df1 is None or df2 is None:
                    continue
                
                # Try each common column
                for merge_col in cols:
                    overlap = set(df1[merge_col].dropna()) & set(df2[merge_col].dropna())
                    if len(overlap) > 0:
                        overlap_ratio = len(overlap) / min(df1[merge_col].nunique(), df2[merge_col].nunique())
                        
                        if overlap_ratio > best_overlap_ratio:
                            merged_df = pd.merge(df1, df2, on=merge_col, how='inner', suffixes=('_1', '_2'))
                            if len(merged_df) > 0:
                                best_merge = (merged_df, f"Successfully merged {file1} and {file2} on '{merge_col}' (overlap: {overlap_ratio:.1%})", [file1, file2])
                                best_overlap_ratio = overlap_ratio
                        
            except Exception as e:
                continue
        
        if best_merge:
            return best_merge
        
        return None, "Merge attempts failed - no sufficient data overlap", list(common_cols.keys())
    
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