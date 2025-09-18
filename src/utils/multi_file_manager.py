"""
Multi-file management utilities for the data analytics system
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
import time

class MultiFileDataManager:
    """Enhanced data manager for handling multiple files with intelligent selection"""
    
    @staticmethod
    def get_file_summary(df_manager) -> str:
        """Get a formatted summary of all available files"""
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