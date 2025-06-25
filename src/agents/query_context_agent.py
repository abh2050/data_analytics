from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import pandas as pd

# Import the shared DataFrameManager
try:
    from .pandas_agent import df_manager
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pandas_agent import df_manager

class QueryContextAgent:
    """
    Agent that provides context understanding for user queries by:
    1. Analyzing the current dataset structure and content
    2. Expanding abbreviations and domain-specific terms
    3. Suggesting better query formulations
    4. Providing column mapping and context hints
    """
    
    def __init__(self, llm):
        if llm is None:
            raise ValueError("QueryContextAgent requires a valid LLM instance")
        self.llm = llm
        
        # Common abbreviations and mappings that can be expanded based on context
        self.common_abbreviations = {
            'hp': ['horsepower', 'health_points', 'hit_points'],
            'mpg': ['miles_per_gallon', 'fuel_efficiency'],
            'rpm': ['revolutions_per_minute'],
            'temp': ['temperature'],
            'qty': ['quantity'],
            'amt': ['amount'],
            'pct': ['percent', 'percentage'],
            'min': ['minimum', 'minutes'],
            'max': ['maximum'],
            'avg': ['average', 'mean'],
            'std': ['standard_deviation'],
            'id': ['identifier', 'identification'],
            'num': ['number'],
            'vol': ['volume'],
            'wt': ['weight'],
            'ht': ['height'],
            'len': ['length'],
            'yr': ['year'],
            'mth': ['month'],
            'addr': ['address'],
            'tel': ['telephone', 'phone'],
            'email': ['email_address'],
            'url': ['website', 'link']
        }

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze user query and provide context enrichment
        
        Args:
            state: The current state dict containing messages, query, etc.
            
        Returns:
            Updated state with query context analysis
        """
        # Create a copy of the state to modify and return
        updated_state = state.copy()
        
        # Extract query from either state or messages
        query = state.get("query", "")
        if not query and state.get("messages"):
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
                elif hasattr(msg, 'type') and msg.type == 'human':
                    query = msg.content
                    break
        
        query = query or ""
        
        # Set current agent in state
        updated_state["current_agent"] = "query_context"
        
        # Initialize agent_outputs if not present
        if "agent_outputs" not in updated_state:
            updated_state["agent_outputs"] = {}
        
        print(f"[QueryContextAgent] Processing query: {query}")
        
        try:
            # Get current dataframe for context
            current_df = df_manager.get_current_dataframe()
            
            if current_df is None:
                result = {
                    "original_query": query,
                    "expanded_query": query,
                    "context_hints": [],
                    "column_suggestions": [],
                    "reasoning": "No dataset loaded for context analysis"
                }
            else:
                result = self._analyze_query_context(query, current_df)
            
            # Update state with context analysis
            updated_state["agent_outputs"]["query_context"] = {
                "status": "completed",
                "result": result,
                "reasoning": "Completed query context analysis"
            }
            
            # Also add the expanded query to the state for other agents to use
            updated_state["expanded_query"] = result.get("expanded_query", query)
            updated_state["query_context"] = result
            
            return updated_state
            
        except Exception as e:
            print(f"[QueryContextAgent] Error: {e}")
            error_result = {
                "original_query": query,
                "expanded_query": query,
                "context_hints": [],
                "column_suggestions": [],
                "reasoning": f"Error in context analysis: {e}"
            }
            
            updated_state["agent_outputs"]["query_context"] = {
                "status": "error",
                "result": error_result,
                "error": str(e)
            }
            
            return updated_state

    def _analyze_query_context(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze query in the context of the available dataset
        """
        print(f"[QueryContextAgent] Analyzing query: {query}")
        print(f"[QueryContextAgent] Dataset shape: {df.shape}")
        print(f"[QueryContextAgent] Columns: {list(df.columns)[:10]}")
        
        # Analyze dataset structure
        columns = list(df.columns)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Get sample of data for better context understanding
        sample_data = {}
        for col in columns[:10]:  # Limit to first 10 columns
            try:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:
                    sample_data[col] = unique_vals.tolist()
                else:
                    sample_data[col] = unique_vals[:5].tolist() + ["..."]
            except:
                sample_data[col] = ["<analysis_error>"]
        
        # Create context analysis prompt
        context_prompt = f"""You are a query context analyzer. Analyze the user's query in the context of the available dataset and provide expanded understanding.

USER QUERY: "{query}"

DATASET CONTEXT:
- Total columns: {len(columns)}
- Column names: {columns}
- Numeric columns: {numeric_cols}
- Categorical columns: {categorical_cols}
- Sample data: {sample_data}

TASK: Provide query context analysis including:
1. Identify any abbreviations or unclear terms in the query
2. Map abbreviations to likely column names in the dataset (ONLY if they exist)
3. Suggest the most relevant columns for the query
4. Provide an expanded/clarified version of the query
5. Give context hints that would help other agents

ANALYSIS GUIDELINES:
- Look for common abbreviations (hp=horsepower, mpg=miles per gallon, etc.)
- Consider domain context based on column names and data
- If an abbreviation doesn't match any column in the dataset, DO NOT map it
- Instead, explain what the abbreviation likely means and suggest what's available
- If query mentions "highest", "lowest", "top", identify relevant numeric columns
- If query asks about categories/types, identify relevant categorical columns
- If the query is about a domain that doesn't match the dataset, explain the mismatch

IMPORTANT: If the query contains terms that don't exist in this dataset (e.g., asking about cars when the dataset is about business metrics), clearly indicate this mismatch.

Respond in this exact format:
ABBREVIATIONS_FOUND: [list any abbreviations or unclear terms]
COLUMN_MAPPING: [map terms to actual column names - only if columns exist]
RELEVANT_COLUMNS: [list most relevant columns for this query - only existing ones]
EXPANDED_QUERY: [clearer version of the query with full terms]
CONTEXT_HINTS: [helpful context for other agents, including domain mismatches]
QUERY_TYPE: [classification like "ranking", "filtering", "comparison", "aggregation", "domain_mismatch"]
DOMAIN_MATCH: [yes/no - does the query domain match the dataset]
"""

        # Get LLM analysis
        try:
            response = self.llm.invoke(context_prompt)
            analysis_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            parsed_result = self._parse_context_analysis(analysis_text, query, df)
            
            print(f"[QueryContextAgent] Analysis result: {parsed_result}")
            return parsed_result
            
        except Exception as e:
            print(f"[QueryContextAgent] LLM analysis error: {e}")
            # Fallback to simple analysis
            return self._simple_context_analysis(query, df)

    def _parse_context_analysis(self, analysis_text: str, original_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Parse the LLM context analysis response
        """
        result = {
            "original_query": original_query,
            "expanded_query": original_query,
            "abbreviations_found": [],
            "column_mapping": {},
            "relevant_columns": [],
            "context_hints": [],
            "query_type": "general",
            "domain_match": True,  # New field
            "reasoning": "LLM-based context analysis"
        }
        
        try:
            lines = analysis_text.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == 'ABBREVIATIONS_FOUND':
                        # Parse list format
                        if value and value != 'None' and '[' in value:
                            try:
                                result["abbreviations_found"] = eval(value)
                            except:
                                result["abbreviations_found"] = [v.strip() for v in value.replace('[', '').replace(']', '').split(',') if v.strip()]
                    
                    elif key == 'COLUMN_MAPPING':
                        # Parse mapping
                        if value and value != 'None':
                            try:
                                # Handle different formats
                                if '{' in value:
                                    result["column_mapping"] = eval(value)
                                else:
                                    # Simple format: term->column
                                    mappings = {}
                                    parts = value.split(',')
                                    for part in parts:
                                        if '->' in part or '=' in part:
                                            separator = '->' if '->' in part else '='
                                            k, v = part.split(separator, 1)
                                            mappings[k.strip()] = v.strip()
                                    result["column_mapping"] = mappings
                            except:
                                result["column_mapping"] = {}
                    
                    elif key == 'RELEVANT_COLUMNS':
                        if value and value != 'None':
                            try:
                                if '[' in value:
                                    result["relevant_columns"] = eval(value)
                                else:
                                    result["relevant_columns"] = [v.strip() for v in value.split(',') if v.strip()]
                            except:
                                result["relevant_columns"] = [v.strip() for v in value.split(',') if v.strip()]
                    
                    elif key == 'EXPANDED_QUERY':
                        if value and value != 'None':
                            result["expanded_query"] = value
                    
                    elif key == 'CONTEXT_HINTS':
                        if value and value != 'None':
                            try:
                                if '[' in value:
                                    result["context_hints"] = eval(value)
                                else:
                                    result["context_hints"] = [v.strip() for v in value.split(',') if v.strip()]
                            except:
                                result["context_hints"] = [v.strip() for v in value.split(',') if v.strip()]
                    
                    elif key == 'QUERY_TYPE':
                        if value and value != 'None':
                            result["query_type"] = value.lower()
            
            # Validate columns exist in dataset
            valid_columns = []
            for col in result["relevant_columns"]:
                if col in df.columns:
                    valid_columns.append(col)
                else:
                    # Try fuzzy matching
                    similar = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                    if similar:
                        valid_columns.append(similar[0])
            
            result["relevant_columns"] = valid_columns
            
        except Exception as e:
            print(f"[QueryContextAgent] Parse error: {e}")
            result["reasoning"] = f"Parse error, using fallback: {e}"
        
        return result

    def _simple_context_analysis(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple fallback context analysis without LLM
        """
        result = {
            "original_query": query,
            "expanded_query": query,
            "abbreviations_found": [],
            "column_mapping": {},
            "relevant_columns": [],
            "context_hints": [],
            "query_type": "general",
            "reasoning": "Simple fallback analysis"
        }
        
        query_lower = query.lower()
        columns = list(df.columns)
        columns_lower = [col.lower() for col in columns]
        
        # Check for abbreviations
        for abbrev, expansions in self.common_abbreviations.items():
            if abbrev in query_lower:
                result["abbreviations_found"].append(abbrev)
                
                # Try to find matching columns
                found_match = False
                for expansion in expansions:
                    matching_cols = [col for col in columns if expansion in col.lower() or col.lower() in expansion]
                    if matching_cols:
                        result["column_mapping"][abbrev] = matching_cols[0]
                        result["relevant_columns"].extend(matching_cols)
                        # Expand the query
                        result["expanded_query"] = result["expanded_query"].replace(abbrev, expansion)
                        found_match = True
                        break
                
                # If no match found, add context hint about the mismatch
                if not found_match:
                    result["context_hints"].append(f"'{abbrev}' (likely '{expansions[0]}') not found in dataset")
                    result["context_hints"].append(f"Available columns: {', '.join(columns)}")
                    # Don't expand the query if we can't map it
                    result["expanded_query"] = query
        
        # Determine query type
        if any(word in query_lower for word in ['highest', 'maximum', 'top', 'largest', 'most']):
            result["query_type"] = "ranking_max"
        elif any(word in query_lower for word in ['lowest', 'minimum', 'bottom', 'smallest', 'least']):
            result["query_type"] = "ranking_min"
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            result["query_type"] = "comparison"
        elif any(word in query_lower for word in ['average', 'mean', 'sum', 'total', 'count']):
            result["query_type"] = "aggregation"
        
        # Add context hints
        if result["query_type"].startswith("ranking"):
            result["context_hints"].append("This is a ranking query - look for numeric columns to sort by")
        
        if result["relevant_columns"]:
            result["context_hints"].append(f"Focus analysis on columns: {result['relevant_columns']}")
        
        return result

@tool
def expand_query_context(query: str) -> str:
    """
    Expand a user query with better context and column mapping
    
    query: The original user query that might contain abbreviations or unclear terms
    """
    try:
        # Get current dataframe
        current_df = df_manager.get_current_dataframe()
        
        if current_df is None:
            return f"Original query: {query}\nNote: No dataset loaded for context expansion."
        
        # Create a temporary QueryContextAgent for this analysis
        from langchain_openai import ChatOpenAI
        import os
        
        llm = ChatOpenAI(temperature=0, model='gpt-4o-mini', api_key=os.environ.get('OPENAI_API_KEY'))
        context_agent = QueryContextAgent(llm)
        
        # Analyze the query
        result = context_agent._analyze_query_context(query, current_df)
        
        # Format the response
        response = f"Original query: {query}\n"
        response += f"Expanded query: {result['expanded_query']}\n"
        
        if result['abbreviations_found']:
            response += f"Abbreviations found: {result['abbreviations_found']}\n"
        
        if result['column_mapping']:
            response += f"Column mapping: {result['column_mapping']}\n"
        
        if result['relevant_columns']:
            response += f"Relevant columns: {result['relevant_columns']}\n"
        
        if result['context_hints']:
            response += f"Context hints: {'; '.join(result['context_hints'])}\n"
        
        response += f"Query type: {result['query_type']}"
        
        return response
        
    except Exception as e:
        return f"Error expanding query context: {e}\nOriginal query: {query}"
