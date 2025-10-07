"""
LangSmith configuration and tracking utilities for the multi-agent data analytics systems.
"""

import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

# Load environment variables
load_dotenv()

class LangSmithTracker:
    """Centralized LangSmith tracking for all agents and queries"""
    
    def __init__(self):
        self.client = None
        self.tracer = None
        self.project_name = os.getenv("LANGCHAIN_PROJECT", "data-analytics-agents")
        self.enabled = self._check_langsmith_config()
        
        if self.enabled:
            self._initialize_client()
    
    def _check_langsmith_config(self) -> bool:
        """Check if LangSmith is properly configured"""
        required_vars = ["LANGCHAIN_API_KEY", "LANGCHAIN_TRACING_V2"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            return False
        
        if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
            return False
        
        return True
    
    def _initialize_client(self):
        """Initialize LangSmith client and tracer"""
        try:
            self.client = Client()
            self.tracer = LangChainTracer(project_name=self.project_name)
            print(f"[LangSmith] Initialized tracking for project: {self.project_name}")
        except Exception as e:
            print(f"[LangSmith] Failed to initialize: {e}")
            self.enabled = False
    
    def get_callback_manager(self) -> Optional[CallbackManager]:
        """Get callback manager with LangSmith tracer"""
        if not self.enabled or not self.tracer:
            return None
        
        return CallbackManager([self.tracer])
    
    def create_session(self, session_id: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new tracking session"""
        if not self.enabled:
            return session_id
        
        try:
            print(f"[LangSmith] Session initialized: {session_id}")
            return session_id
            
        except Exception as e:
            print(f"[LangSmith] Failed to create session: {e}")
            return session_id
    
    def log_query_start(self, query: str, session_id: str, metadata: Dict[str, Any] = None) -> str:
        """Log the start of a query processing"""
        if not self.enabled:
            return str(uuid.uuid4())
        
        try:
            run_id = str(uuid.uuid4())
            query_metadata = {
                "query": query,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "type": "query_start",
                **(metadata or {})
            }
            
            print(f"[LangSmith] Query started - Run ID: {run_id}")
            return run_id
            
        except Exception as e:
            print(f"[LangSmith] Failed to log query start: {e}")
            return str(uuid.uuid4())
    
    def log_agent_execution(self, agent_name: str, input_data: Dict[str, Any], 
                          output_data: Dict[str, Any], run_id: str, 
                          metadata: Dict[str, Any] = None):
        """Log individual agent execution"""
        if not self.enabled:
            return
        
        try:
            agent_metadata = {
                "agent": agent_name,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "input_size": len(str(input_data)),
                "output_size": len(str(output_data)),
                "status": output_data.get("status", "unknown"),
                **(metadata or {})
            }
            
            print(f"[LangSmith] Agent executed: {agent_name} (Run: {run_id})")
            
        except Exception as e:
            print(f"[LangSmith] Failed to log agent execution: {e}")
    
    def log_query_complete(self, run_id: str, final_result: str, 
                          agent_outputs: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Log query completion with final results"""
        if not self.enabled:
            return
        
        try:
            completion_metadata = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "type": "query_complete",
                "result_size": len(final_result),
                "agents_used": list(agent_outputs.keys()),
                "total_agents": len(agent_outputs),
                **(metadata or {})
            }
            
            print(f"[LangSmith] Query completed - Run ID: {run_id}")
            
        except Exception as e:
            print(f"[LangSmith] Failed to log query completion: {e}")
    
    def log_error(self, error: Exception, context: Dict[str, Any], run_id: str = None):
        """Log errors with context"""
        if not self.enabled:
            return
        
        try:
            error_metadata = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id or "unknown"
            }
            
            print(f"[LangSmith] Error logged: {type(error).__name__}")
            
        except Exception as e:
            print(f"[LangSmith] Failed to log error: {e}")

# Global tracker instance
_tracker = None

def get_langsmith_tracker() -> LangSmithTracker:
    """Get the global LangSmith tracker instance"""
    global _tracker
    if _tracker is None:
        _tracker = LangSmithTracker()
    return _tracker

def setup_langsmith_for_llm(llm):
    """Setup LangSmith tracking for an LLM instance"""
    tracker = get_langsmith_tracker()
    if tracker.enabled and tracker.tracer:
        # Add the tracer to the LLM's callbacks
        if hasattr(llm, 'callbacks'):
            if llm.callbacks is None:
                llm.callbacks = []
            llm.callbacks.append(tracker.tracer)
        else:
            # For newer versions of langchain
            callback_manager = tracker.get_callback_manager()
            if callback_manager:
                llm.callbacks = callback_manager
    
    return llm
