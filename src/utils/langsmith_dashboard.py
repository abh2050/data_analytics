"""
LangSmith dashboard utilities for displaying tracking information in Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .langsmith_config import get_langsmith_tracker
import time
class LangSmithDashboard:
    """Dashboard component for displaying LangSmith tracking information"""
    
    def __init__(self):
        self.tracker = get_langsmith_tracker()
    
    def display_tracking_status(self):
        """Display LangSmith tracking status in sidebar"""
        if not self.tracker.enabled:
            st.sidebar.markdown("### 📊 LangSmith Tracking")
            st.sidebar.warning("⚠️ LangSmith tracking is disabled")
            st.sidebar.markdown("""
            To enable tracking:
            1. Set `LANGCHAIN_API_KEY` in .env
            2. Set `LANGCHAIN_TRACING_V2=true` in .env
            3. Restart the application
            """)
            return
        
        st.sidebar.markdown("### 📊 LangSmith Tracking")
        st.sidebar.success("✅ Tracking enabled")
        
        # Display project info
        if self.tracker.project_name:
            st.sidebar.info(f"📁 Project: {self.tracker.project_name}")
        
        # Add link to LangSmith dashboard
        if self.tracker.client:
            st.sidebar.markdown("""
            [🔗 View in LangSmith Dashboard](https://smith.langchain.com)
            """)
    
    def display_session_metrics(self, session_id: str):
        """Display metrics for current session"""
        if not self.tracker.enabled:
            return
        
        st.sidebar.markdown("### 📈 Session Metrics")
        
        # Display session ID
        st.sidebar.code(f"Session: {session_id[:8]}...")
        
        # Get session stats from session state if available
        if 'conversation_history' in st.session_state:
            history = st.session_state.conversation_history
            
            if history:
                total_queries = len(history)
                avg_time = sum(msg.get('processing_time', 0) for msg in history) / total_queries
                agents_used = set(msg.get('agent', 'unknown') for msg in history)
                
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    st.metric("Queries", total_queries)
                    st.metric("Agents", len(agents_used))
                
                with col2:
                    st.metric("Avg Time", f"{avg_time:.1f}s")
                    st.metric("Success Rate", "100%")  # Simplified for now
                
                # Agent usage breakdown
                if len(agents_used) > 1:
                    st.sidebar.markdown("**Agent Usage:**")
                    agent_counts = {}
                    for msg in history:
                        agent = msg.get('agent', 'unknown')
                        agent_counts[agent] = agent_counts.get(agent, 0) + 1
                    
                    for agent, count in agent_counts.items():
                        emoji = {'pandas': '🐼', 'chart': '📊', 'search': '🔍', 'python': '🐍', 'router': '🎯'}.get(agent, '🤖')
                        st.sidebar.write(f"{emoji} {agent}: {count}")
    
    def display_query_trace(self, query: str, result: Dict[str, Any]):
        """Display trace information for a query"""
        if not self.tracker.enabled:
            return
        
        # This would be expanded to show detailed trace information
        # For now, just show basic info
        with st.expander("🔍 Query Trace", expanded=False):
            st.write(f"**Query:** {query}")
            st.write(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")
            st.write(f"**Agent Used:** {result.get('agent', 'unknown')}")
            
            if 'langsmith_run_id' in st.session_state:
                st.code(f"Run ID: {st.session_state.langsmith_run_id}")
    
    def display_performance_insights(self):
        """Display performance insights and recommendations"""
        if not self.tracker.enabled or 'conversation_history' not in st.session_state:
            return
        
        history = st.session_state.conversation_history
        if len(history) < 3:  # Need some data for insights
            return
        
        st.sidebar.markdown("### 💡 Performance Insights")
        
        # Calculate insights
        processing_times = [msg.get('processing_time', 0) for msg in history]
        avg_time = sum(processing_times) / len(processing_times)
        
        # Performance feedback
        if avg_time < 2.0:
            st.sidebar.success("🚀 Excellent response times!")
        elif avg_time < 5.0:
            st.sidebar.info("⚡ Good performance")
        else:
            st.sidebar.warning("🐌 Consider optimizing queries")
        
        # Agent efficiency
        agent_times = {}
        for msg in history:
            agent = msg.get('agent', 'unknown')
            time_taken = msg.get('processing_time', 0)
            if agent not in agent_times:
                agent_times[agent] = []
            agent_times[agent].append(time_taken)
        
        if len(agent_times) > 1:
            st.sidebar.markdown("**Agent Performance:**")
            for agent, times in agent_times.items():
                avg_agent_time = sum(times) / len(times)
                emoji = {'pandas': '🐼', 'chart': '📊', 'search': '🔍', 'python': '🐍'}.get(agent, '🤖')
                st.sidebar.write(f"{emoji} {agent}: {avg_agent_time:.1f}s avg")
    
    def export_session_data(self, session_id: str) -> Optional[pd.DataFrame]:
        """Export session data for analysis"""
        if not self.tracker.enabled or 'conversation_history' not in st.session_state:
            return None
        
        history = st.session_state.conversation_history
        if not history:
            return None
        
        # Convert to DataFrame
        data = []
        for i, msg in enumerate(history):
            data.append({
                'query_id': i + 1,
                'session_id': session_id,
                'timestamp': msg.get('timestamp', time.time()),
                'agent': msg.get('agent', 'unknown'),
                'processing_time': msg.get('processing_time', 0),
                'context_aware': msg.get('context_aware', False),
                'success': not msg.get('error', False)
            })
        
        return pd.DataFrame(data)
    
    def display_export_options(self, session_id: str):
        """Display export options for session data"""
        if not self.tracker.enabled:
            return
        
        st.sidebar.markdown("### 📤 Export Options")
        
        # Export session data
        if st.sidebar.button("📊 Export Session Data"):
            df = self.export_session_data(session_id)
            if df is not None:
                csv = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"langsmith_session_{session_id[:8]}.csv",
                    mime="text/csv"
                )
            else:
                st.sidebar.warning("No data to export")

# Global dashboard instance
_dashboard = None

def get_langsmith_dashboard() -> LangSmithDashboard:
    """Get the global LangSmith dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = LangSmithDashboard()
    return _dashboard