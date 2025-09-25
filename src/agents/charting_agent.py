from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langgraph.prebuilt import create_react_agent
from typing import Optional, Type, Dict, Any, List, TypedDict
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import base64
from io import BytesIO
import json
import ast
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import matplotlib.pyplot as plt
# Import the shared DataFrameManager singleton
try:
    from .pandas_agent import df_manager
except ImportError:
    # Fallback with proper import path for singleton
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pandas_agent import df_manager

def safe_parse_action_input(action_input):
    """Safely parse action input from various formats (dict, JSON string, Python dict string)"""
    # If already a dict, return as is
    if isinstance(action_input, dict):
        return action_input
    # Try to parse as dict from string
    try:
        return ast.literal_eval(action_input)
    except Exception:
        # Optionally, try to parse as JSON
        try:
            return json.loads(action_input)
        except Exception:
            return {}

class GenerateChartInput(BaseModel):
    data: str = Field(description="JSON string representing the data")
    chart_type: str = Field(description="Type of chart to generate")
    x_axis: str = Field(description="Column name for x-axis")
    y_axis: str = Field(description="Column name for y-axis")
    title: str = Field(description="Chart title")

class LoadAndChartCSVInput(BaseModel):
    file_path: str = Field(description="Path to the CSV file")
    chart_type: str = Field(description="Type of chart to generate")
    x_axis: str = Field(description="Column name for x-axis")
    y_axis: str = Field(description="Column name for y-axis")
    title: str = Field(description="Chart title")

class RobustGenerateChartTool(BaseTool):
    name: str = "generate_chart"
    description: str = """Generates various types of charts from provided data and returns it as a base64 encoded PNG image.
    Supported chart types: 'line', 'bar', 'scatter', 'histogram', 'boxplot', 'heatmap', 'pie'
    The `data` parameter should be a JSON string representing the data.
    `x_axis` and `y_axis` are the column names for the axes, and `title` is the chart title."""
    args_schema: Type[BaseModel] = GenerateChartInput
    
    def _run(self, data: str, chart_type: str, x_axis: str, y_axis: str, title: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return generate_chart_impl(data, chart_type, x_axis, y_axis, title)

class RobustLoadAndChartCSVTool(BaseTool):
    name: str = "load_and_chart_csv"
    description: str = """Loads data from a CSV file and generates a chart directly.
    This tool combines data loading and charting for convenience."""
    args_schema: Type[BaseModel] = LoadAndChartCSVInput
    
    def _run(self, file_path: str, chart_type: str, x_axis: str, y_axis: str, title: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        return load_and_chart_csv_impl(file_path, chart_type, x_axis, y_axis, title)

def generate_chart_impl(data: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Implementation of chart generation with robust error handling"""
    try:
        df = pd.read_json(data)
        
        # Limit dataframe size for large datasets to save tokens
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
            sample_notice = "(Using 1000 sample rows)"
        else:
            sample_notice = ""
        
        # For wide dataframes, remove unnecessary columns to save memory
        if len(df.columns) > 5:
            needed_cols = [x_axis, y_axis]
            df = df[needed_cols]
        
        # Truncate long strings in axis labels
        x_label = x_axis[:25] + "..." if len(x_axis) > 25 else x_axis
        y_label = y_axis[:25] + "..." if len(y_axis) > 25 else y_axis
        
        # Truncate long title
        if len(title) > 50:
            title = title[:47] + "..."
            
        plt.figure(figsize=(10, 6))  # Larger figure for better spacing
        plt.style.use('seaborn-v0_8')
        
        if chart_type == "line":
            # Limit data points for line charts to prevent overlap
            if len(df) > 50:
                df_sample = df.sample(50, random_state=42).sort_values(x_axis)
                plt.plot(df_sample[x_axis], df_sample[y_axis], marker='o', linewidth=2, markersize=6)
            else:
                plt.plot(df[x_axis], df[y_axis], marker='o', linewidth=2, markersize=6)
        elif chart_type == "bar":
            # Smart bar chart handling
            if len(df) > 15:
                # Group and show top categories
                counts = df.groupby(x_axis)[y_axis].mean().nlargest(12)
                bars = plt.bar(range(len(counts)), counts.values, alpha=0.8, color='skyblue', edgecolor='navy')
                # Truncate labels smartly
                labels = [str(x)[:12] + '...' if len(str(x)) > 12 else str(x) for x in counts.index]
                plt.xticks(range(len(counts)), labels)
            else:
                bars = plt.bar(df[x_axis], df[y_axis], alpha=0.8, color='skyblue', edgecolor='navy')
                # Keep original labels but truncate if too long
                current_labels = [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in df[x_axis]]
                plt.xticks(range(len(df)), current_labels)
        elif chart_type == "scatter":
            # Limit scatter points to prevent overcrowding
            if len(df) > 200:
                df_sample = df.sample(200, random_state=42)
                plt.scatter(df_sample[x_axis], df_sample[y_axis], alpha=0.7, s=40, color='coral')
            else:
                plt.scatter(df[x_axis], df[y_axis], alpha=0.7, s=40, color='coral')
        elif chart_type == "histogram":
            plt.hist(df[y_axis], bins=8, alpha=0.7, color='lightgreen', edgecolor='black')  # Reduced bins
            plt.xlabel(y_label)
            plt.ylabel('Frequency')
        elif chart_type == "boxplot":
            df.boxplot(column=y_axis, by=x_axis)
            plt.suptitle('')  # Remove default title
            plt.xticks(rotation=45, ha='right')
        elif chart_type == "pie":
            # For pie charts, limit to top 6 categories
            if len(df) > 6:
                counts = df[y_axis].value_counts().nlargest(6)
                plt.pie(counts.values, labels=[str(x)[:10] for x in counts.index], autopct='%1.1f%%', startangle=90)
            else:
                plt.pie(df[y_axis], labels=[str(x)[:10] for x in df[x_axis]], autopct='%1.1f%%', startangle=90)
        else:
            return f"Unsupported chart type: {chart_type}. Supported types: line, bar, scatter, histogram, boxplot, pie"
            
        if chart_type not in ["histogram", "boxplot", "pie"]:
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            
            # Dynamic axis handling based on data type and size
            if chart_type == "bar":
                # For bar charts, show all labels but rotate and truncate if needed
                current_labels = plt.gca().get_xticklabels()
                if len(current_labels) > 15:
                    # Show every nth label to prevent overlap
                    step = len(current_labels) // 10
                    for i, label in enumerate(current_labels):
                        if i % step != 0:
                            label.set_visible(False)
                plt.xticks(rotation=45, ha='right')
            else:
                # For line/scatter, use smart tick selection
                n_points = len(df)
                if n_points > 20:
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=8))
                elif n_points > 10:
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
                else:
                    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=n_points))
                
                plt.xticks(rotation=30, ha='right')
            
            # Y-axis always readable
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        
        plt.title(f"{title} {sample_notice}", fontsize=12, fontweight='bold')
        # Add extra space for rotated labels
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI from 100 to 80
        buf.seek(0)
        img_data = buf.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Ensure proper base64 padding
        missing_padding = len(img_base64) % 4
        if missing_padding:
            img_base64 += '=' * (4 - missing_padding)
            
        plt.close()
        return f"Chart generated successfully: data:image/png;base64,{img_base64}"
    except Exception as e:
        return f"Error generating chart: {e}"

def load_and_chart_csv_impl(file_path: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Implementation of CSV loading and charting with robust error handling"""
    try:
        # Read only the necessary columns to save memory
        try:
            if x_axis != y_axis:
                df = pd.read_csv(file_path, usecols=[x_axis, y_axis])
            else:
                df = pd.read_csv(file_path, usecols=[x_axis])
        except:
            # If specific columns can't be read, read all and then filter
            df = pd.read_csv(file_path)
            
        # Convert to JSON with only the necessary data
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
            data_json = df_sample.to_json(orient='records')
        else:
            data_json = df.to_json(orient='records')
            
        return generate_chart_impl(data_json, chart_type, x_axis, y_axis, title)
    except Exception as e:
        return f"Error loading CSV and generating chart: {e}"

@tool
def generate_chart(data: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Generates various types of charts from provided data and returns it as a base64 encoded PNG image.
    Supported chart types: 'line', 'bar', 'scatter', 'histogram', 'boxplot', 'heatmap', 'pie'
    The `data` parameter should be a JSON string representing the data.
    `x_axis` and `y_axis` are the column names for the axes, and `title` is the chart title.
    """
    return generate_chart_impl(data, chart_type, x_axis, y_axis, title)

@tool
def load_and_chart_csv(file_path: str, chart_type: str, x_axis: str, y_axis: str, title: str) -> str:
    """Loads data from a CSV file and generates a chart directly.
    This tool combines data loading and charting for convenience.
    """
    return load_and_chart_csv_impl(file_path, chart_type, x_axis, y_axis, title)

@tool
def create_chart_from_uploaded_data(chart_type: str, x_column: str, y_column: str = "", title: str = "Chart", top_n: int = 10) -> str:
    """Create a chart from the currently uploaded dataframe.
    
    chart_type: 'line', 'bar', 'scatter', 'histogram', 'box', 'heatmap', 'pie'
    x_column: column name for x-axis  
    y_column: column name for y-axis (optional for some chart types)
    title: chart title
    top_n: number of top values to show (for bar charts, pie charts)
    """
    try:
        # Get dataframe from the shared df_manager with intelligent file selection
        query_for_selection = f"chart {chart_type} {x_column} {y_column or ''}"
        df, file_name = df_manager.get_relevant_dataframe(query_for_selection)
        print(f"[create_chart_from_uploaded_data] Selected file: {file_name} for query: {query_for_selection}")
            
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        if x_column not in df.columns:
            return f"Column '{x_column}' not found. Available columns: {list(df.columns)}"
        
        if y_column and y_column != "" and y_column not in df.columns:
            return f"Column '{y_column}' not found. Available columns: {list(df.columns)}"
        
        # Sample large dataframes to reduce processing time
        sample_notice = ""
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
            sample_notice = " (Using 1000 sample rows)"
        
        plt.figure(figsize=(12, 8))  # Larger figure for better spacing
        plt.style.use('default')
        
        if chart_type == "bar":
            if y_column and y_column != "":
                y_column = y_column if y_column != "" else None
                # Validate that y_column is numeric for aggregation
                if not pd.api.types.is_numeric_dtype(df[y_column]):
                    # Try to convert to numeric, if it fails, use value counts instead
                    try:
                        df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
                        # Remove rows where conversion failed (NaN values)
                        df = df.dropna(subset=[y_column])
                        if df.empty:
                            return f"Column '{y_column}' contains no valid numeric data for charting."
                    except:
                        return f"Column '{y_column}' is not numeric and cannot be used for bar chart aggregation. Try using it as x-axis instead."
                
                # Group and aggregate for bar chart
                if df[x_column].dtype == 'object':
                    # Categorical x-axis, aggregate y values
                    grouped = df.groupby(x_column)[y_column].sum().nlargest(top_n)
                    bars = plt.bar(range(len(grouped)), grouped.values)
                    # Truncate long labels and rotate
                    labels = [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in grouped.index]
                    plt.xticks(range(len(grouped)), labels, rotation=45, ha='right')
                    plt.ylabel(y_column)
                    # Add space for rotated labels
                    plt.subplots_adjust(bottom=0.25)
                else:
                    # Numeric x-axis
                    sorted_df = df.nlargest(top_n, y_column)
                    plt.bar(sorted_df[x_column], sorted_df[y_column])
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
            else:
                # Value counts bar chart
                value_counts = df[x_column].value_counts().head(top_n)
                bars = plt.bar(range(len(value_counts)), value_counts.values)
                # Truncate long labels and rotate
                labels = [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) for x in value_counts.index]
                plt.xticks(range(len(value_counts)), labels, rotation=45, ha='right')
                plt.ylabel('Count')
                # Add space for rotated labels
                plt.subplots_adjust(bottom=0.25)
            plt.xlabel(x_column)
        
        elif chart_type == "scatter":
            if y_column and y_column != "":
                y_column = y_column if y_column != "" else None
                # Limit scatter points to prevent overcrowding
                if len(df) > 500:
                    df_sample = df.sample(500, random_state=42)
                    plt.scatter(df_sample[x_column], df_sample[y_column], alpha=0.7, s=30)
                else:
                    plt.scatter(df[x_column], df[y_column], alpha=0.7, s=30)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            else:
                return "Scatter plot requires both x_column and y_column"
        
        elif chart_type == "line":
            if y_column and y_column != "":
                y_column = y_column if y_column != "" else None
                plt.plot(df[x_column], df[y_column], marker='o')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
            else:
                plt.plot(df[x_column], marker='o')
                plt.ylabel(x_column)
        
        elif chart_type == "histogram":
            plt.hist(df[x_column], bins=15, alpha=0.7, edgecolor='black')  # Reduced bins
            plt.xlabel(x_column)
            plt.ylabel('Frequency')
            # Limit x-axis ticks to prevent overlap
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=8))
        
        elif chart_type == "box":
            if y_column and y_column != "":
                y_column = y_column if y_column != "" else None
                # Box plot by category
                if df[x_column].nunique() > 10:
                    # Limit categories for readability
                    top_cats = df[x_column].value_counts().head(10).index
                    df_filtered = df[df[x_column].isin(top_cats)]
                    df_filtered.boxplot(column=y_column, by=x_column)
                else:
                    df.boxplot(column=y_column, by=x_column)
                plt.suptitle('')
            else:
                plt.boxplot(df[x_column])
                plt.ylabel(x_column)
        
        elif chart_type == "pie":
            value_counts = df[x_column].value_counts().head(min(top_n, 8))  # Max 8 slices
            # Truncate long labels
            labels = [str(x)[:12] + '...' if len(str(x)) > 12 else str(x) for x in value_counts.index]
            plt.pie(value_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
        
        elif chart_type == "heatmap":
            # Correlation heatmap for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            else:
                return "Heatmap requires multiple numeric columns"
        
        else:
            return f"Unknown chart type: {chart_type}. Supported types: line, bar, scatter, histogram, box, heatmap, pie"
        
        plt.title(f"{title} - {file_name}{sample_notice}")
        # Add extra spacing for better label visibility
        plt.subplots_adjust(bottom=0.15, left=0.1)
        plt.tight_layout()
        
        # Save chart to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = buf.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Ensure proper base64 padding
        missing_padding = len(img_base64) % 4
        if missing_padding:
            img_base64 += '=' * (4 - missing_padding)
            
        plt.close()
        
        return f"[Analyzing: {file_name}] Chart generated successfully: data:image/png;base64,{img_base64}"
    
    except Exception as e:
        return f"Error creating chart: {e}"

class ChartingAgent:
    def __init__(self, llm):
        if llm is None:
            raise ValueError("ChartingAgent requires a valid LLM instance")
        self.llm = llm
        # Include the new dynamic code generation tool as the primary tool
        self.tools = [
            generate_and_execute_chart_code,  # Primary tool for dynamic code generation
            create_chart_from_uploaded_data,
            RobustGenerateChartTool(),
            RobustLoadAndChartCSVTool()
        ]
        self.prompt = PromptTemplate.from_template(
            """You are an expert data visualization agent that generates custom Python plotting code on demand.

AVAILABLE TOOLS:
- generate_and_execute_chart_code: PREFERRED - Generates custom Python code for any visualization request
- create_chart_from_uploaded_data: Fallback for simple predefined chart types
- generate_chart: For JSON data visualization
- load_and_chart_csv: For CSV file visualization

INTELLIGENT APPROACH:
1. ALWAYS try generate_and_execute_chart_code FIRST for maximum flexibility
2. This tool can create any type of visualization by generating custom Python code
3. It can handle complex requests like multiple variables, subplots, custom styling
4. Use fallback tools only if dynamic generation fails

DYNAMIC CODE CAPABILITIES:
- Custom chart types (violin plots, swarm plots, 3D plots, etc.)
- Multiple subplots and complex layouts
- Advanced statistical visualizations
- Custom color schemes and styling
- Interactive elements and annotations
- Data transformations and aggregations

EXAMPLES OF DYNAMIC REQUESTS:
- "Create a violin plot showing distribution by category"
- "Make a subplot with revenue and expenses side by side"
- "Show correlation matrix as a heatmap with annotations"
- "Create a stacked bar chart with percentages"
- "Plot time series with trend lines"

USER REQUEST: {input}

Use generate_and_execute_chart_code to create a custom visualization that perfectly matches the user's request."""
        )
        # Create the LangGraph React agent
        self.agent = create_react_agent(self.llm, self.tools)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently generate data visualizations based on user requests
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with chart generation results
        """
        # Create a copy of the state to modify and return
        updated_state = state.copy()
        
        # Extract query from either state or messages
        query = state.get("query", "")
        if not query and state.get("messages"):
            # Get the last user message if query isn't directly available
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
                elif hasattr(msg, 'type') and msg.type == 'human':
                    query = msg.content
                    break
        
        # Default to empty string if we still don't have a query
        query = query or ""
        
        # Set current agent in state
        updated_state["current_agent"] = "chart"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[ChartingAgent] Processing query: {query}")
        print("updated_state",updated_state)
        try:
            # Check if this is a multi-file request from query context
            query_context = updated_state.get("query_context", {})
            # Also check agent_outputs for query_context
            if not query_context and "agent_outputs" in updated_state and "query_context" in updated_state["agent_outputs"]:
                query_context = updated_state["agent_outputs"]["query_context"].get("result", {})
            print("query_context:", query_context)
            is_multi_file_request = query_context.get("multi_file_request", False)
            print("is_multi_file_request:", is_multi_file_request)
            # Check if this is a general chart request (regardless of multi-file detection)
            query_lower = query.lower()
            general_chart_keywords = ['generate all charts', 'create all charts', 'all charts', 'generate charts', 'create charts', 'show charts', 'make charts', 'charts and graphs', 'generate graphs', 'create graphs', 'generate all graphs', 'all graphs', 'generate all chart and graph', 'all chart and graph', 'create all chart and graph', 'show all chart and graph', 'generate all charts and graphs', 'create all charts and graphs', 'all charts and graphs']
            is_general_request = any(keyword in query_lower for keyword in general_chart_keywords)
            
            if is_multi_file_request or is_general_request:
                print("General chart request detected - using comprehensive analysis")
                # Store query for use in the method
                self._current_query = query
                result = self._generate_individual_file_charts()
                has_data = True
                current_df = None  # Not needed for comprehensive analysis
            else:
                print("Specific chart request - using single file")
                # Single file selection
                current_df, file_name = df_manager.get_relevant_dataframe(query)
                print(f"[ChartingAgent] Selected file for query '{query}': {file_name}")
                has_data = current_df is not None
                
                if has_data:
                    result = self._intelligent_chart_creation(query, current_df)
                else:
                    result = ("I'd be happy to create visualizations for you! However, I don't see any dataset uploaded yet. "
                             "Please upload a CSV or Excel file using the file uploader in the sidebar, and then I can "
                             "create charts and visualizations from your data.")
                    has_data = False  # Ensure has_data is properly set
            
            print(f"[ChartingAgent] Has data: {has_data}")
            if has_data and not (is_multi_file_request or is_general_request) and current_df is not None:
                print(f"[ChartingAgent] Data shape: {current_df.shape}")
                print(f"[ChartingAgent] Columns: {list(current_df.columns)[:5]}")
            

            
            print(f"[ChartingAgent] Result: {result[:200]}...")
            
            # Update state with chart generation results
            updated_state["agent_outputs"]["chart"] = {
                "status": "completed",
                "result": result,
                "reasoning": "Completed intelligent chart generation"
            }
            
            return updated_state
            
        except Exception as e:
            print(f"[ChartingAgent] Error: {e}")
            error_message = f"Error in charting agent: {e}"
            
            # Update state with error information
            updated_state["agent_outputs"]["chart"] = {
                "status": "error",
                "result": error_message,
                "error": str(e)
            }
            
            return updated_state
    
    def _generate_individual_file_charts(self) -> str:
        """Generate comprehensive charts from all available files"""
        try:
            # Get query from the calling method's context
            query = getattr(self, '_current_query', '')
            
            # Get all available files
            all_files_info = df_manager.get_file_info()
            available_files = [name for name in all_files_info.keys() if name != 'merged_data']
            total_files = len(available_files)
            
            print(f"[ChartingAgent] Generating charts for all {total_files} files: {available_files}")
            
            # Try merged data first for comprehensive cross-file analysis
            merged_df = df_manager.get_dataframe('merged_data')
            if merged_df is not None:
                print(f"[ChartingAgent] Using merged data with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
                
                # Use dynamic code generation for merged data with explicit file count
                result = generate_and_execute_chart_code.invoke({
                    "user_request": f"{query} - Generate comprehensive charts covering all {total_files} files",
                    "chart_description": f"Create comprehensive visualizations from merged data representing all {total_files} files with cross-file relationships"
                })
                
                if "Error" not in result and "data:image/png;base64," in result:
                    return result
                else:
                    return f"Generated charts from merged data covering all {total_files} files with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns."
            else:
                # No merged data - generate charts for each file individually
                if total_files > 1:
                    print(f"[ChartingAgent] No merged data, generating individual charts for {total_files} files")
                    all_results = []
                    
                    for file_name in available_files:
                        file_df = df_manager.get_dataframe(file_name)
                        if file_df is not None:
                            # Generate charts for this specific file
                            result = generate_and_execute_chart_code.invoke({
                                "user_request": f"{query} - Generate charts for {file_name}",
                                "chart_description": f"Create visualizations from {file_name} (file {available_files.index(file_name)+1} of {total_files})"
                            })
                            
                            if "Error" not in result and "data:image/png;base64," in result:
                                all_results.append(result)
                    
                    if all_results:
                        return "\n\n".join(all_results)
                    else:
                        return f"Generated individual charts for all {total_files} files."
                else:
                    # Single file case
                    current_df, file_name = df_manager.get_relevant_dataframe(query)
                    if current_df is None:
                        return "No dataframes loaded. Please upload files first."
                    
                    result = generate_and_execute_chart_code.invoke({
                        "user_request": query,
                        "chart_description": f"Create comprehensive visualizations from {file_name}"
                    })
                    
                    if "Error" not in result and "data:image/png;base64," in result:
                        return result
                    else:
                        return f"Generated comprehensive charts from {file_name}."
                
        except Exception as e:
            print(f"[ChartingAgent] Error generating charts: {e}")
            return f"Error generating charts: {e}"
    
    def _generate_cross_file_comparison(self, file_list) -> str:
        """Generate comparison charts between files with matching columns"""
        try:
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            if len(file_list) < 2:
                return None
            
            # Find common columns across files
            common_cols = set(file_list[0][1].columns)
            for name, df in file_list[1:]:
                common_cols = common_cols.intersection(set(df.columns))
            
            if not common_cols:
                return None
            
            print(f"[ChartingAgent] Found common columns: {list(common_cols)}")
            
            # Filter to numeric common columns for meaningful comparisons
            numeric_common = []
            for col in common_cols:
                if all(df[col].dtype in ['int64', 'float64'] for name, df in file_list):
                    numeric_common.append(col)
            
            if not numeric_common:
                return None
            
            plt.figure(figsize=(15, 10))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            # Generate up to 6 comparison charts
            chart_count = 0
            max_charts = 6
            
            for i, col in enumerate(numeric_common[:3]):  # Max 3 columns
                if chart_count >= max_charts:
                    break
                
                # Bar comparison chart
                chart_count += 1
                plt.subplot(2, 3, chart_count)
                file_means = [df[col].mean() for name, df in file_list]
                file_names = [name.replace('.csv', '') for name, df in file_list]
                bars = plt.bar(file_names, file_means, color=colors[:len(file_list)], 
                              edgecolor='black', alpha=0.7)
                plt.title(f'{col} - Average Comparison')
                plt.ylabel(f'Average {col}')
                plt.xticks(rotation=45)
                
                # Box plot comparison
                if chart_count < max_charts:
                    chart_count += 1
                    plt.subplot(2, 3, chart_count)
                    data_for_box = [df[col].dropna() for name, df in file_list]
                    box_plot = plt.boxplot(data_for_box, labels=file_names, patch_artist=True)
                    for patch, color in zip(box_plot['boxes'], colors[:len(file_list)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    plt.title(f'{col} - Distribution Comparison')
                    plt.ylabel(col)
                    plt.xticks(rotation=45)
            
            # Overall summary comparison
            if chart_count < max_charts:
                chart_count += 1
                plt.subplot(2, 3, chart_count)
                total_rows = [len(df) for name, df in file_list]
                plt.pie(total_rows, labels=file_names, colors=colors[:len(file_list)], 
                       autopct='%1.1f%%')
                plt.title('Data Volume Distribution')
            
            plt.suptitle('Cross-File Comparison Analysis', fontsize=16)
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return f"**Cross-File Comparison Analysis**:\ndata:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"[ChartingAgent] Error in cross-file comparison: {e}")
            return None
    
    def _generate_multi_dataset_pie_chart(self, file_list) -> str:
        """Generate a single pie chart comparing data across multiple datasets"""
        try:
            import matplotlib.pyplot as plt
            import base64
            from io import BytesIO
            
            if len(file_list) < 2:
                return None
            
            plt.figure(figsize=(12, 8))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            # Create pie chart showing data volume distribution across datasets
            file_names = [name.replace('.csv', '') for name, df in file_list]
            data_sizes = [len(df) for name, df in file_list]
            
            plt.pie(data_sizes, labels=file_names, colors=colors[:len(file_list)], 
                   autopct='%1.1f%%', startangle=90, explode=[0.05]*len(file_list))
            plt.title('Data Distribution Across 5 Datasets', fontsize=16, fontweight='bold')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return f"**Multi-Dataset Pie Chart**:\ndata:image/png;base64,{img_base64}"
            
        except Exception as e:
            print(f"[ChartingAgent] Error in multi-dataset pie chart: {e}")
            return None
        
    def _intelligent_chart_creation(self, query: str, df) -> str:
        """
        Use LLM intelligence to analyze the query and create appropriate charts
        """
        print(f"[ChartingAgent] Intelligent processing: {query}")
        
        # Check if df is None first
        if df is None:
            return ("I'd be happy to create visualizations for you! However, I don't see any dataset uploaded yet. "
                   "Please upload a CSV or Excel file using the file uploader in the sidebar, and then I can "
                   "create charts and visualizations from your data.")
        
        try:
            safe_imports = """
            import matplotlib
            matplotlib.use("Agg")  # Safe backend for Streamlit
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            import numpy as np
            import base64
            from io import BytesIO
            """
            # First, try dynamic code generation for maximum flexibility
            result = generate_and_execute_chart_code.invoke({
                "user_request": query,
                "chart_description": f"Create a visualization based on the user request: {query}"
            })
            print(f"[ChartingAgent] Dynamic generation result: {result[:200]}...")
            if "Error" not in result:
                if isinstance(result, list) and any(r.startswith("data:image/png;base64,") for r in result):
                    # Join all charts into one message
                    joined = "\n\n".join(result)
                    return f"I generated custom Python code to create exactly what you requested:\n\n{joined}"
                elif isinstance(result, str) and "data:image/png;base64," in result:
                    return f"I generated custom Python code to create exactly what you requested:\n\n{result}"
            else:
                print(f"[ChartingAgent] Dynamic generation failed, trying LLM analysis fallback")
                # Fallback to LLM analysis approach if dynamic generation fails
                return self._llm_analysis_fallback(query, df)
                
        except Exception as e:
            print(f"[ChartingAgent] Dynamic code generation error: {e}")
            # Fallback to LLM analysis approach
            return self._llm_analysis_fallback(query, df)
    
    def _llm_analysis_fallback(self, query: str, df) -> str:
        """
        LLM analysis fallback when dynamic code generation fails
        """
        # Check if df is None
        if df is None:
            return ("I'd be happy to create visualizations for you! However, I don't see any dataset uploaded yet. "
                   "Please upload a CSV or Excel file using the file uploader in the sidebar, and then I can "
                   "create charts and visualizations from your data.")
        
        query_lower = query.lower()
        columns = list(df.columns)
        
        print(f"[ChartingAgent] LLM analysis fallback: {query}")
        print(f"[ChartingAgent] Available columns: {columns[:10]}")
        
        # Let the LLM analyze the query and suggest the approach
        try:
            analysis_prompt = f"""Analyze this data visualization request and suggest the best approach:

Query: "{query}"
Available columns: {columns}

Task: Determine the most appropriate:
1. Chart type (bar, line, scatter, histogram, pie, box)
2. X-axis column 
3. Y-axis column (if needed)
4. Number of items to show (for top-N queries)

Guidelines:
- For "top N" or "highest/lowest": use bar charts with appropriate top_n
- For "over time" or "trends": use line charts with time on x-axis
- For "distribution": use histograms
- For "relationship between": use scatter plots
- For "proportion" or "percentage": use pie charts

Respond in this exact format:
CHART_TYPE: [type]
X_COLUMN: [column name]
Y_COLUMN: [column name or None]
TOP_N: [number or None]
TITLE: [descriptive title]
REASONING: [brief explanation]"""

            # Get LLM analysis
            llm_response = self.llm.invoke(analysis_prompt)
            analysis = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            print(f"[ChartingAgent] LLM Analysis: {analysis[:200]}...")
            
            # Parse the LLM response
            chart_params = self._parse_llm_analysis(analysis, df)
            
            if chart_params:
                # Use the LLM-determined parameters to create the chart
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": chart_params["chart_type"],
                    "x_column": chart_params["x_column"],
                    "y_column": chart_params["y_column"],
                    "title": chart_params["title"],
                    "top_n": chart_params["top_n"]
                })
                
                reasoning = chart_params.get("reasoning", "LLM-driven analysis")
                return f"I analyzed your request and created a {chart_params['chart_type']} chart. {reasoning}\n\n{result}"
            else:
                # Final fallback to rule-based approach
                return self._direct_chart_fallback(query, df)
                
        except Exception as e:
            print(f"[ChartingAgent] LLM analysis error: {e}")
            # Final fallback to rule-based approach
            return self._direct_chart_fallback(query, df)
    
    def _parse_llm_analysis(self, analysis: str, df) -> dict:
        """
        Parse the LLM analysis response and validate parameters
        """
        try:
            lines = analysis.split('\n')
            params = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'chart_type':
                        valid_types = ['bar', 'line', 'scatter', 'histogram', 'pie', 'box', 'heatmap']
                        if value.lower() in valid_types:
                            params['chart_type'] = value.lower()
                    elif key == 'x_column':
                        if value in df.columns:
                            params['x_column'] = value
                        else:
                            # Try to find similar column
                            similar = [col for col in df.columns if value.lower() in col.lower()]
                            if similar:
                                params['x_column'] = similar[0]
                    elif key == 'y_column':
                        if value.lower() == 'none':
                            params['y_column'] = None
                        elif value in df.columns:
                            # Validate that y_column is numeric for aggregation charts
                            if pd.api.types.is_numeric_dtype(df[value]):
                                params['y_column'] = value
                            else:
                                # Try to find a similar numeric column
                                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                                similar = [col for col in numeric_cols if value.lower() in col.lower()]
                                if similar:
                                    params['y_column'] = similar[0]
                                else:
                                    # If no similar numeric column, set to None for value counts chart
                                    params['y_column'] = None
                        else:
                            # Try to find similar numeric column
                            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                            similar = [col for col in numeric_cols if value.lower() in col.lower()]
                            if similar:
                                params['y_column'] = similar[0]
                            else:
                                params['y_column'] = None
                    elif key == 'top_n':
                        if value.lower() == 'none':
                            params['top_n'] = 10  # Default
                        else:
                            try:
                                params['top_n'] = int(value)
                            except:
                                params['top_n'] = 10
                    elif key == 'title':
                        params['title'] = value
                    elif key == 'reasoning':
                        params['reasoning'] = value
            
            # Validate required parameters
            if 'chart_type' in params and 'x_column' in params:
                return params
            else:
                return None
                
        except Exception as e:
            print(f"[ChartingAgent] Error parsing LLM analysis: {e}")
            return None

    def _analyze_data_structure(self, df) -> dict:
        """
        Generically analyze the dataset structure to understand what types of visualizations are possible
        """
        analysis = {
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'string']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
            'suitable_for_pie': [],
            'suitable_for_bar': [],
            'suitable_for_histogram': [],
            'suitable_for_scatter': [],
            'high_cardinality': [],
            'low_cardinality': []
        }
        
        # Analyze categorical columns for chart suitability
        for col in analysis['categorical_columns']:
            unique_count = df[col].nunique()
            non_null_count = df[col].count()
            
            if unique_count <= 10 and non_null_count > 0:
                analysis['suitable_for_pie'].append(col)
                analysis['low_cardinality'].append(col)
            elif unique_count <= 20 and non_null_count > 0:
                analysis['suitable_for_bar'].append(col)
                analysis['low_cardinality'].append(col)
            elif unique_count > 50:
                analysis['high_cardinality'].append(col)
        
        # Analyze numeric columns
        for col in analysis['numeric_columns']:
            if df[col].count() > 0:  # Has non-null values
                analysis['suitable_for_histogram'].append(col)
                if len(analysis['numeric_columns']) > 1:
                    analysis['suitable_for_scatter'].append(col)
        
        return analysis

    def _create_generic_chart(self, query: str, df) -> str:
        """
        Create charts based on generic data analysis, not specific to any domain
        """
        data_analysis = self._analyze_data_structure(df)
        query_lower = query.lower()
        
        print(f"[ChartingAgent] Data analysis: {data_analysis}")
        
        # Look for distribution/status related queries
        distribution_keywords = ['status', 'distribution', 'breakdown', 'categories', 'types', 'different']
        if any(keyword in query_lower for keyword in distribution_keywords):
            # Find the best categorical column for distribution
            if data_analysis['suitable_for_pie']:
                best_col = data_analysis['suitable_for_pie'][0]
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": "pie",
                    "x_column": best_col,
                    "title": f"Distribution of {best_col}",
                    "top_n": 10
                })
                return f"I found a categorical column '{best_col}' and created a distribution chart:\n\n{result}"
            elif data_analysis['suitable_for_bar']:
                best_col = data_analysis['suitable_for_bar'][0]
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": "bar",
                    "x_column": best_col,
                    "title": f"Distribution of {best_col}",
                    "top_n": 15
                })
                return f"I found a categorical column '{best_col}' and created a distribution chart:\n\n{result}"
        
        # Look for top/highest/lowest queries
        ranking_keywords = ['top', 'highest', 'lowest', 'best', 'worst', 'largest', 'smallest']
        if any(keyword in query_lower for keyword in ranking_keywords):
            if data_analysis['numeric_columns'] and data_analysis['categorical_columns']:
                numeric_col = data_analysis['numeric_columns'][0]
                cat_col = data_analysis['categorical_columns'][0]
                result = create_chart_from_uploaded_data.invoke({
                    "chart_type": "bar",
                    "x_column": cat_col,
                    "y_column": numeric_col,
                    "title": f"Top Values: {cat_col} by {numeric_col}",
                    "top_n": 10
                })
                return f"I created a ranking chart using '{cat_col}' and '{numeric_col}':\n\n{result}"
        
        # Fallback: suggest what's available in the dataset
        suggestions = []
        if data_analysis['suitable_for_pie']:
            suggestions.append(f"Pie charts for: {data_analysis['suitable_for_pie'][:3]}")
        if data_analysis['suitable_for_histogram']:
            suggestions.append(f"Histograms for: {data_analysis['suitable_for_histogram'][:3]}")
        if data_analysis['numeric_columns'] and data_analysis['categorical_columns']:
            suggestions.append(f"Bar charts comparing {data_analysis['categorical_columns'][0]} vs {data_analysis['numeric_columns'][0]}")
        
        if suggestions:
            return f"I analyzed your dataset and found these visualization possibilities:\n\n" + "\n".join(f"• {s}" for s in suggestions) + f"\n\nPlease specify which type of chart you'd like to see, or I can create a default visualization based on your data structure."
        else:
            return f"This dataset has {len(df.columns)} columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}. Please specify which columns you'd like to visualize and what type of chart you prefer."
    
    def _analyze_file_requirement(self, user_request: str, available_files: list) -> dict:
        """Use LLM to analyze which file(s) the user actually wants based on their request"""
        try:
            analysis_prompt = f"""Analyze this user request to determine which file(s) they want to work with:

User Request: "{user_request}"
Available Files: {available_files}

Analyze the request and determine:
1. Does the user mention a specific file name or content that matches one file?
2. Do they want data from all files combined?
3. Are they asking for a specific type of analysis that would require one particular file?

Respond in this exact JSON format:
{{
    "specific_file": "filename.csv" or null,
    "multi_file": true or false,
    "reasoning": "explanation of decision"
}}

Guidelines:
- If user mentions "railways", "pharma", "it" or similar, match to corresponding file
- If user says "all files", "compare", "merged" set multi_file: true
- If user asks for general charts without specifying, use intelligent matching
- If unclear, default to single most relevant file"""

            response = self.llm.invoke(analysis_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                print(f"[ChartingAgent] File analysis: {analysis}")
                return analysis
            else:
                return {'specific_file': None, 'multi_file': False, 'reasoning': 'Failed to parse LLM response'}
                
        except Exception as e:
            print(f"[ChartingAgent] Error in file analysis: {e}")
            return {'specific_file': None, 'multi_file': False, 'reasoning': f'Error: {e}'}
    
    def _direct_chart_fallback(self, query: str, df) -> str:
        """
        Direct chart creation fallback when React agent hits token limits
        """
        # Check if df is None
        if df is None:
            return ("I'd be happy to create visualizations for you! However, I don't see any dataset uploaded yet. "
                   "Please upload a CSV or Excel file using the file uploader in the sidebar, and then I can "
                   "create charts and visualizations from your data.")
        
        query_lower = query.lower()
        columns = list(df.columns)
        
        print(f"[ChartingAgent] Fallback processing: {query}")
        print(f"[ChartingAgent] Available columns: {columns[:10]}")
        
        # Use generic data analysis instead of hardcoded keywords
        data_analysis = self._analyze_data_structure(df)
        
        try:
            # Handle distribution/status queries generically
            if any(word in query_lower for word in ['status', 'distribution', 'different', 'breakdown', 'categories']):
                return self._create_generic_chart(query, df)
            
            # Handle ranking queries generically  
            elif any(word in query_lower for word in ['top', 'highest', 'lowest', 'best', 'worst', 'ranking', 'compare']):
                if data_analysis['numeric_columns'] and data_analysis['categorical_columns']:
                    result = create_chart_from_uploaded_data.invoke({
                        "chart_type": "bar",
                        "x_column": data_analysis['categorical_columns'][0],
                        "y_column": data_analysis['numeric_columns'][0],
                        "title": f"Top Values: {data_analysis['categorical_columns'][0]} by {data_analysis['numeric_columns'][0]}",
                        "top_n": 10
                    })
                    return f"I created a ranking chart:\n\n{result}"
                else:
                    return "I need both categorical and numeric columns to create ranking charts."
            
            # Handle correlation/relationship queries
            elif any(word in query_lower for word in ['correlation', 'relationship', 'scatter', 'compare']):
                if len(data_analysis['numeric_columns']) >= 2:
                    result = create_chart_from_uploaded_data.invoke({
                        "chart_type": "scatter",
                        "x_column": data_analysis['numeric_columns'][0],
                        "y_column": data_analysis['numeric_columns'][1],
                        "title": f"Relationship: {data_analysis['numeric_columns'][0]} vs {data_analysis['numeric_columns'][1]}"
                    })
                    return f"I created a scatter plot to show relationships:\n\n{result}"
                else:
                    return "I need at least 2 numeric columns to create scatter plots or correlation charts."
            
            # Handle distribution queries for numeric data
            elif any(word in query_lower for word in ['distribution', 'histogram', 'spread']):
                if data_analysis['suitable_for_histogram']:
                    result = create_chart_from_uploaded_data.invoke({
                        "chart_type": "histogram",
                        "x_column": data_analysis['suitable_for_histogram'][0],
                        "title": f"Distribution of {data_analysis['suitable_for_histogram'][0]}"
                    })
                    return f"I created a histogram showing the distribution:\n\n{result}"
                else:
                    return "I need numeric columns to create distribution charts."
            
            # Handle general chart/graph requests
            elif any(word in query_lower for word in ['chart', 'graph', 'plot', 'visualize', 'show']):
                return self._create_generic_chart(query, df)
            
            # Default: suggest available columns and chart types
            else:
                return self._create_generic_chart(query, df)
                       
        except Exception as e:
            return f"Error creating chart: {e}. Available columns: {columns[:10]}"

def _analyze_file_requirement_standalone(user_request: str, available_files: list) -> dict:
    """Standalone function to analyze which file(s) the user wants based on their request"""
    try:
        from langchain_openai import ChatOpenAI
        import os
        llm = ChatOpenAI(temperature=0, model='gpt-4.1', api_key=os.environ.get('OPENAI_API_KEY'))

        
        analysis_prompt = f"""Analyze this user request to determine which file(s) they want to work with:

User Request: "{user_request}"
Available Files: {available_files}

Respond in this exact JSON format:
{{
    "specific_file": "filename.csv" or null,
    "multi_file": true or false,
    "reasoning": "explanation of decision"
}}

Analyze the request intelligently based on context and keywords.
"""

        response = llm.invoke(analysis_prompt)
        analysis_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            print(f"[FileAnalysis] {analysis}")
            return analysis
        else:
            return {'specific_file': None, 'multi_file': False, 'reasoning': 'Failed to parse LLM response'}
            
    except Exception as e:
        print(f"[FileAnalysis] Error: {e}")
        return {'specific_file': None, 'multi_file': False, 'reasoning': f'Error: {e}'}

@tool
def generate_and_execute_chart_code(user_request: str, chart_description: str = "") -> str:
    """Generate Python plotting code on the fly based on user request and execute it.
    
    user_request: The original user request for the chart
    chart_description: Optional description of what chart to create
    """
    try:
        from src.utils.multi_file_manager import MultiFileDataManager
        
        # Check if this is a multi-file request or general chart request
        request_lower = user_request.lower()
        general_chart_keywords = ['generate charts', 'create charts', 'show charts', 'make charts', 'charts and graphs', 'generate graphs', 'create graphs', 'generate all charts', 'create all charts', 'all charts', 'generate all graphs', 'all graphs']
        multi_file_keywords = ['all files', 'both files', 'from both', 'from all', 'compare files', 'merged data', 'combined data']
        
        # Check if it's an explicit multi-file request OR general chart request
        is_multi_file_request = any(keyword in request_lower for keyword in multi_file_keywords)
        is_general_chart_request = any(keyword in request_lower for keyword in general_chart_keywords)
        
        # First do dynamic keyword matching
        file_info = df_manager.get_file_info()
        mentioned_file = None
        requires_multi_file = False
        
        request_lower = user_request.lower()
        for file_name_check in file_info.keys():
            if file_name_check == 'merged_data':
                continue
            # Extract keywords from filename (remove extensions, split by underscore/space)
            file_keywords = file_name_check.lower().replace('.csv', '').replace('_', ' ').replace('(', '').replace(')', '').split()
            # Check if any keyword from filename appears in user request
            for keyword in file_keywords:
                if len(keyword) > 2 and keyword in request_lower:  # Skip short words like 'a', 'of'
                    mentioned_file = file_name_check
                    requires_multi_file = False
                    print(f"[ChartingAgent] Dynamic match: '{keyword}' found in request, using file: {file_name_check}")
                    break
            if mentioned_file:
                break
        
        # If no dynamic match found, use LLM analysis as fallback
        if not mentioned_file:
            file_selection_analysis = _analyze_file_requirement_standalone(user_request, list(file_info.keys()))
            mentioned_file = file_selection_analysis.get('specific_file')
            requires_multi_file = file_selection_analysis.get('multi_file', False)
        
        # Determine which data to use
        if mentioned_file:
            # User mentioned a specific file - use ONLY that file
            df = df_manager.get_dataframe(mentioned_file)
            file_name = mentioned_file
            print(f"[ChartingAgent] Using specific file: {mentioned_file} for request: {user_request}")
            if df is None:
                return f"File '{mentioned_file}' not found. Available files: {list(file_info.keys())}"
        elif requires_multi_file or is_multi_file_request or (is_general_chart_request and len([name for name in file_info.keys() if name != 'merged_data']) > 1):
            # Only use merged data for explicit multi-file requests
            merged_df = df_manager.get_dataframe('merged_data')
            if merged_df is not None:
                df = merged_df
                file_name = "merged_data"
                print(f"[ChartingAgent] Using merged data for multi-file request with {merged_df.shape[0]} rows and {merged_df.shape[1]} columns")
            else:
                merged_df, merge_status = MultiFileDataManager.get_merged_dataframe(df_manager)
                if merged_df is not None:
                    df = merged_df
                    file_name = "merged_data"
                else:
                    return "Could not create merged data for multi-file analysis."
        else:
            # Single file selection using intelligent matching
            df, file_name = df_manager.get_relevant_dataframe(user_request)
            if df is None:
                return "No dataframes loaded. Please upload files first."
        
        print(f"[ChartingAgent] Using data from: {file_name} for request: {user_request}")
        
        # Analyze the dataframe structure
        columns = list(df.columns)
        dtypes = df.dtypes.to_dict()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Use full data for analysis - no sampling to avoid null data issues
        df_sample = df
        sample_note = ""
            
        
        # Get all available files for comprehensive chart generation
        all_files_info = df_manager.get_file_info()
        available_files = [name for name in all_files_info.keys() if name != 'merged_data']
        total_files = len(available_files)
        
        # Use LLM to intelligently analyze the request type
        analysis_prompt = f"""Analyze this user request to determine if they want:
1. A specific single chart type (histogram, bar chart, pie chart, etc.)
2. Multiple comprehensive charts covering all data

User Request: "{user_request}"

Respond with either:
- "SPECIFIC" if they want one specific chart type
- "COMPREHENSIVE" if they want multiple charts or general analysis

Examples:
- "histogram" → SPECIFIC
- "bar chart" → SPECIFIC  
- "show me charts" → COMPREHENSIVE
- "analyze data" → COMPREHENSIVE
- "pie chart of sectors" → SPECIFIC
- "visualize everything" → COMPREHENSIVE"""
        
        from langchain_openai import ChatOpenAI
        import os
        llm = ChatOpenAI(temperature=0, model='gpt-4.1', api_key=os.environ.get('OPENAI_API_KEY'))
        analysis_response = llm.invoke(analysis_prompt)
        analysis_result = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
        
        is_comprehensive = "COMPREHENSIVE" in analysis_result.upper()
        
        # Enhanced code generation prompt based on request type
        if is_comprehensive:
            chart_instruction = f"Generate 12-15 different charts covering all columns and relationships"
        else:
            chart_instruction = f"Generate ONLY the specific chart type requested by the user. Create just ONE chart that matches their exact request."
        
        code_generation_prompt = f"""
You are an expert Python data visualization programmer. Generate matplotlib/seaborn code based on the user's request.

USER REQUEST: "{user_request}"
FILE: "{file_name}" | TOTAL FILES: {total_files} | ALL FILES: {available_files}

DATA SUMMARY:
- Shape: {df.shape}
- Columns: {columns}
- Numeric: {numeric_cols}
- Categorical: {categorical_cols}
- Sample: {df.head(2).to_dict() if len(df) > 0 else 'No data'}

CHART INSTRUCTION:
{chart_instruction}

CRITICAL REQUIREMENTS:
1. NEVER READ FILES: Use only 'df'; do NOT use pd.read_csv or file ops.
2. COLUMN CHECK: Always verify column exists: if 'col' in df.columns
3. ERROR HANDLING: Wrap chart ops in try-except
4. FORMAT CHARTS: 
   - Figure size: plt.figure(figsize=(12,8))
   - Limit categories for bars: top 8 (.value_counts().head(8))
   - Limit axis ticks: x-max 8, y-max 6
   - Rotate long labels: plt.xticks(rotation=45, ha='right')
   - Adaptive spacing: plt.subplots_adjust(bottom=0.25)
   - Truncate long labels: str(x)[:12]
5. DATA CLEANING: Convert numeric before groupby: pd.to_numeric(df[col], errors='coerce')
6. RETURN: Dict with chart names as keys and base64 images as values

CODE TEMPLATE:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO

plt.style.use('default')
sns.set_palette("husl")

def save_chart_as_base64():
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    img_base64 += '=' * ((4 - len(img_base64) % 4) % 4)
    plt.close('all'); plt.clf()
    return img_base64

def fix_labels(ax, max_labels=6):
    labels = ax.get_xticklabels()
    if len(labels) > max_labels:
        step = len(labels) // max_labels + 1
        for i, l in enumerate(labels):
            if i % step != 0: l.set_visible(False)
    max_len = max([len(str(l.get_text())) for l in labels]+[0])
    if max_len > 8 or len(labels) > 6: plt.xticks(rotation=45, ha='right'); plt.subplots_adjust(bottom=0.2)
    else: plt.xticks(rotation=0); plt.subplots_adjust(bottom=0.1)
    plt.yticks(rotation=0)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    return ax

result ={ {}}
# Use df directly; follow formatting, label, and numeric conversion guidelines

WARNING: DO NOT include any lines like 'merged_data = pd.read_csv(...)' or similar file reading operations. The data is already available in the 'df' variable.

IMPORTANT: Generate ONLY executable Python code that uses the existing 'df' dataframe. Always add proper spacing and label rotation to prevent overlapping.
"""

        # Get the LLM to generate the code
        from langchain_openai import ChatOpenAI
        import os
        llm = ChatOpenAI(temperature=0, model='gpt-4.1', api_key=os.environ.get('OPENAI_API_KEY'))
        
        code_response = llm.invoke(code_generation_prompt)
        generated_code = code_response.content if hasattr(code_response, 'content') else str(code_response)
        
        # Clean the code (remove markdown formatting)
        if "```python" in generated_code:
            generated_code = generated_code.split("```python")[1].split("```")[0]
        elif "```" in generated_code:
            generated_code = generated_code.split("```")[1].split("```")[0]
            
        generated_code = generated_code.strip()
        
        print(f"[ChartingAgent] Generated code length: {len(generated_code)} characters")
        # print(f"[ChartingAgent] Code preview: {generated_code[:200]}...")
        print("generated_code", generated_code)
        request_lower = user_request.lower()
        # Force multi-chart for comprehensive requests or when multiple files are available
        all_files_info = df_manager.get_file_info()
        total_files = len([name for name in all_files_info.keys() if name != 'merged_data'])
        # Only force comprehensive charts based on LLM analysis
        force_multi_chart = is_comprehensive  
        # Import required libraries
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from io import BytesIO
        import base64
        
        # Helper function for saving charts as base64
        def save_chart_as_base64(fig=None, dpi=150):
            if fig is None:
                fig = plt.gcf()
            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
            buf.seek(0)
            img_data = buf.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            missing_padding = len(img_base64) % 4
            if missing_padding:
                img_base64 += '=' * (4 - missing_padding)
            plt.close(fig)
            return img_base64

        # Execute the generated code
        safe_globals = {
            'df': df_sample,
            "plt": plt,
            "sns": sns,
            "pd": pd,
            "np": np,
            "BytesIO": BytesIO,
            "base64": base64,
            "__builtins__": __builtins__,
            'save_chart_as_base64': save_chart_as_base64,
            'result': {},
            'results': {},
            'charts': {},
            'chart_images': {},
            
        }

        # Add automatic function call if code contains commented usage
        if "# result =" in generated_code and "(df)" in generated_code:
            lines = generated_code.split('\n')
            for line in lines:
                if line.strip().startswith('# result =') and '(df)' in line:
                    # Extract function call and add it
                    func_call = line.strip()[2:].strip()  # Remove '# '
                    generated_code += f"\n{func_call}"
                    break
        
        local_vars = {
            'result': {},
            'results': {},
            'charts': {},
            'chart_images': {},
            'save_chart_as_base64': save_chart_as_base64
        }
        exec(generated_code, safe_globals, local_vars)

        # Check if a function was defined and call it if no result exists
        if 'result' not in local_vars:
            # Only look for user-defined functions, exclude built-in classes
            builtin_names = {'BytesIO', 'plt', 'sns', 'pd', 'np', 'base64'}
            function_names = [name for name in local_vars 
                            if callable(local_vars[name]) 
                            and not name.startswith('_') 
                            and name not in builtin_names
                            and hasattr(local_vars[name], '__code__')]
            if function_names:
                func_name = function_names[0]
                print(f"[ChartingAgent] Calling generated function: {func_name}")
                try:
                    result = local_vars[func_name](safe_globals['df'])
                    local_vars['result'] = result
                except Exception as e:
                    print(f"[ChartingAgent] Error calling function {func_name}: {e}")
                    raise e

        # Now capture results
        possible_keys = ["result", "results", "charts", "chart_images", "images", "base64_images","result_images",
            "charts_base64" ,"chart_base64","chart_image","chart","charts_base64"]
        found_result = None
        for key in possible_keys:
            if key in local_vars and local_vars[key]:
                found_result = local_vars[key]
                break
        
        # If nothing found, fallback
        if not found_result:
            print("⚠️ No chart result variable found in executed code")
            return "Chart code was executed but did not return any images."

        # --- Normalize output ---
        if isinstance(found_result, str):
            # Check if result contains an error
            if "Error" in found_result or "error" in found_result:
                print(f"[ChartingAgent] Generated code returned error: {found_result}")
                raise Exception(f"Chart generation failed: {found_result}")
            if not found_result.startswith("data:image"):
                found_result = f"data:image/png;base64,{found_result}"
            return f"[Analyzing: {file_name}] I generated and executed custom plotting code for your request. {sample_note}\n\n{found_result}"

        elif isinstance(found_result, list):
            normalized = [
                f"data:image/png;base64,{img}" if not img.startswith("data:image") else img
                for img in found_result
            ]
            return f"[Analyzing: {file_name}] I generated and executed custom plotting code for your request. {sample_note}\n\n" + "\n\n".join(normalized)

        elif isinstance(found_result, dict):
            normalized = []
            for key, val in found_result.items():
                if not str(val).startswith("data:image"):
                    val = f"data:image/png;base64,{val}"
                normalized.append(f"**{key}**:\n{val}")

            # Check if this is a comprehensive chart request
            all_files_info = df_manager.get_file_info()
            total_files = len([name for name in all_files_info.keys() if name != 'merged_data'])
            
            if force_multi_chart or total_files > 1:
                # Always return all charts for multi-file scenarios
                return (
                    f"[Analyzing: {file_name}] I generated and executed custom plotting code for your request covering all {total_files} files. {sample_note}\n\n"
                    + "\n\n".join(normalized)
                )
            else:
                # Single file - return first chart only
                first_key, first_val = next(iter(found_result.items()))
                if not str(first_val).startswith("data:image"):
                    first_val = f"data:image/png;base64,{first_val}"
                return (
                    f"[Analyzing: {file_name}] I generated and executed custom plotting code for your request. {sample_note}\n\n"
                    f"**{first_key}**:\n{first_val}"
                )
        else:
            return f"Chart code executed, but returned unexpected type: {type(found_result)}"

        
            
    except Exception as e:
        print(f"[ChartingAgent] Dynamic code generation error: {e}")
        import traceback
        print(traceback.print_exc())
        return f"Error generating dynamic chart code: {e}"

@tool
def generate_dynamic_chart(query: str, chart_requirements: str = "") -> str:
    """Generate and execute custom plotting code based on user requirements.
    
    query: The user's visualization request
    chart_requirements: Additional specific requirements or chart type preferences
    """
    try:
        # Get the current dataframe
        df, file_name = df_manager.get_relevant_dataframe(query)
            
        if df is None:
            return "No dataframe loaded. Please upload a file first."
        
        # Create a safe execution environment
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        from io import BytesIO
        import base64
        
        # Sample large dataframes
        if len(df) > 1000:
            df_sample = df.sample(1000, random_state=42)
            sample_notice = " (Using 1000 sample rows)"
        else:
            df_sample = df.copy()
            sample_notice = ""
        
        # Create the code generation prompt
        columns_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'shape': df.shape,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'string']).columns.tolist(),
            'sample_data': df.head(3).to_dict()
        }
        
        code_prompt = f"""Generate Python code to create a visualization for this request: "{query}"

Dataset Information:
- Shape: {columns_info['shape']}
- Columns: {columns_info['columns']}
- Numeric columns: {columns_info['numeric_columns']}
- Categorical columns: {columns_info['categorical_columns']}

Requirements:
{chart_requirements}

Generate Python code that:
1. Uses the variable 'df_sample' (already available)
2. Creates an appropriate matplotlib/seaborn visualization
3. Sets a descriptive title
4. Handles data preprocessing if needed (filtering, grouping, etc.)
5. Uses plt.figure(figsize=(10, 6)) for consistent sizing
6. Includes proper labels and formatting
7. Uses plt.tight_layout() before saving

Important:
- Only return the Python code, no explanations
- Use df_sample as the dataframe variable
- The code should be ready to execute
- Handle any data type conversions needed
- For top-N queries, use .nlargest() or .nsmallest()
- For categorical data, consider using .value_counts()

Example structure:
```python
plt.figure(figsize=(10, 6))
# Your visualization code here
plt.title('Your Title Here')
plt.tight_layout()
```"""

        # Get the generated code from LLM
        code_response = df_manager._ChartingAgent__llm.invoke(code_prompt)
        
        if hasattr(code_response, 'content'):
            generated_code = code_response.content
        else:
            generated_code = str(code_response)
        
        # Extract Python code from response (remove markdown formatting if present)
        import re
        code_match = re.search(r'```python\n(.*?)\n```', generated_code, re.DOTALL)
        if code_match:
            code_to_execute = code_match.group(1)
        else:
            # If no markdown formatting, use the entire response
            code_to_execute = generated_code.strip()
        
        print(f"[ChartingAgent] Generated code: {code_to_execute[:200]}...")
        
        # Create execution environment
        exec_globals = {
            'df_sample': df_sample,
            'df': df_sample,  # Alias for convenience
            'plt': plt,
            'sns': sns,
            'pd': pd,
            'np': np
        }
        
        # Execute the generated code
        exec_globals = {"__builtins__": __builtins__}
        exec(code_to_execute, exec_globals)
        
        # Save the plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_data = buf.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        # Ensure proper base64 padding
        missing_padding = len(img_base64) % 4
        if missing_padding:
            img_base64 += '=' * (4 - missing_padding)
            
        plt.close()
        
        return f"[Analyzing: {file_name}] I generated custom code to create your visualization{sample_notice}:\n\n```python\n{code_to_execute}\n```\n\ndata:image/png;base64,{img_base64}"
        
    except Exception as e:
        print(f"[ChartingAgent] Dynamic code generation error: {e}")
        return f"Error generating dynamic chart: {e}. Let me try a different approach."


