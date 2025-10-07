"""
Agent tracking wrapper for LangSmith integration
"""

import time
from typing import Dict, Any, Callable
from functools import wraps
from .langsmith_config import get_langsmith_tracker

def track_agent_execution(agent_name: str):
    """Decorator to track agent execution with LangSmith"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
            tracker = get_langsmith_tracker()
            
            # Extract tracking info from state
            run_id = state.get("langsmith_run_id", "unknown")
            session_id = state.get("langsmith_session_id", "unknown")
            
            # Log agent start
            start_time = time.time()
            
            try:
                # Execute the agent
                result = func(self, state)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Extract relevant data for logging
                input_data = {
                    "query": state.get("query", ""),
                    "current_agent": state.get("current_agent", ""),
                    "has_data": state.get("has_data", False),
                    "iteration_count": state.get("iteration_count", 0)
                }
                
                output_data = {
                    "status": result.get("agent_outputs", {}).get(agent_name, {}).get("status", "unknown"),
                    "next_agent": result.get("next_agent", ""),
                    "execution_time": execution_time
                }
                
                # Log to LangSmith
                tracker.log_agent_execution(
                    agent_name=agent_name,
                    input_data=input_data,
                    output_data=output_data,
                    run_id=run_id,
                    metadata={
                        "session_id": session_id,
                        "execution_time": execution_time,
                        "success": True
                    }
                )
                
                print(f"[AgentTracker] {agent_name} executed successfully in {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Log error to LangSmith
                tracker.log_error(
                    error=e,
                    context={
                        "agent": agent_name,
                        "session_id": session_id,
                        "execution_time": execution_time,
                        "query": state.get("query", "")
                    },
                    run_id=run_id
                )
                
                print(f"[AgentTracker] {agent_name} failed after {execution_time:.2f}s: {e}")
                
                # Re-raise the exception
                raise e
        
        return wrapper
    return decorator

def initialize_tracking_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize LangSmith tracking fields in state"""
    tracker = get_langsmith_tracker()
    
    # Create session if not exists
    session_id = state.get("session_id", "default")
    langsmith_session_id = f"analytics-{session_id}"
    
    # Create LangSmith session
    tracker.create_session(
        session_id=langsmith_session_id,
        metadata={
            "system": "multi-agent-data-analytics",
            "has_data": state.get("has_data", False),
            "query": state.get("query", "")
        }
    )
    
    # Start query tracking
    run_id = tracker.log_query_start(
        query=state.get("query", ""),
        session_id=langsmith_session_id,
        metadata={
            "dataframe_info": state.get("dataframe_info", {}),
            "iteration_count": state.get("iteration_count", 0)
        }
    )
    
    # Update state with tracking info
    updated_state = state.copy()
    updated_state["langsmith_run_id"] = run_id
    updated_state["langsmith_session_id"] = langsmith_session_id
    updated_state["query_start_time"] = time.time()
    
    return updated_state

def finalize_tracking(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize LangSmith tracking for completed query"""
    tracker = get_langsmith_tracker()
    
    run_id = state.get("langsmith_run_id", "unknown")
    final_result = state.get("final_result", "")
    agent_outputs = state.get("agent_outputs", {})
    start_time = state.get("query_start_time", time.time())
    
    total_time = time.time() - start_time
    
    # Log query completion
    tracker.log_query_complete(
        run_id=run_id,
        final_result=final_result,
        agent_outputs=agent_outputs,
        metadata={
            "total_execution_time": total_time,
            "session_id": state.get("langsmith_session_id", "unknown"),
            "iteration_count": state.get("iteration_count", 0),
            "agents_executed": len([k for k, v in agent_outputs.items() 
                                 if v.get("status") == "completed"])
        }
    )
    
    print(f"[AgentTracker] Query completed in {total_time:.2f}s with {len(agent_outputs)} agents")
    
    return state
