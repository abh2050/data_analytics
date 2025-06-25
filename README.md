# Multi-Agent Data Analytics System

A robust, intelligent multi-agent system for comprehensive data analytics with context-aware query routing, dynamic chart generation, and flexible data exploration. Built with LangGraph, LangChain, and Streamlit.

## Overview

This system uses a sophisticated swarm of specialized agents with intelligent orchestration to handle complex data analytics tasks:

- **Router Agent**: Intelligently classifies and routes queries to appropriate specialist agents
- **Query Context Agent**: Expands abbreviations, maps terms to columns, and provides context hints
- **Memory Agent**: Maintains conversation context and session management
- **Pandas Agent**: Performs dataframe operations, statistics, and data analysis
- **Charting Agent**: Creates dynamic visualizations with LLM-driven code generation
- **Data Search Agent**: Context-aware searching, filtering, and data exploration
- **Python IDE Agent**: Executes custom Python code for advanced analysis
- **Coordinator Agent**: Orchestrates multi-agent workflows and result aggregation

## Key Features

### 🧠 Intelligent Query Understanding
- **Context-Aware Routing**: Automatically routes queries to the most appropriate agent(s)
- **Abbreviation Expansion**: Understands "hp" → "horsepower", "mpg" → "miles per gallon", etc.
- **Column Mapping**: Maps query terms to actual dataset columns intelligently
- **Domain Agnostic**: Works with any CSV/Excel dataset structure

### 📊 Dynamic Visualization
- **LLM-Driven Chart Generation**: Creates custom Python code for any visualization request
- **Adaptive Chart Types**: Analyzes data structure to suggest appropriate charts
- **Base64 Image Output**: Seamless integration with web interfaces
- **Chart Types**: Bar, line, scatter, histogram, box, pie, heatmap, violin plots, and custom visualizations

### 🔍 Powerful Data Exploration
- **Context-Enhanced Search**: Uses query context for more accurate results
- **Flexible Filtering**: Supports all comparison operators and text matching
- **Statistical Analysis**: Comprehensive dataframe operations and insights
- **Multi-Step Reasoning**: Chains multiple operations for complex analysis

### 💭 Conversation Memory
- **Session Persistence**: Maintains context across interactions
- **Chat-Style Interface**: Natural conversation flow with the system
- **Query History**: Learns from previous interactions
- **Multi-Turn Conversations**: Supports follow-up questions and refinements

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run the Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

### 4. Or Use Terminal Chat Interface
```bash
python chat_interface.py
```

### 5. Or Run Demo Conversations
```bash
python demo_chat.py
```

## App
https://langgraphdataanalyticsagents.streamlit.app/
![](https://github.com/abh2050/langgraph_data_analytics_agents/blob/main/assets/1.png)
![](https://github.com/abh2050/langgraph_data_analytics_agents/blob/main/assets/2.png)

## Agent Architecture & Capabilities

### 🎯 Router Agent
- **Intent Classification**: Analyzes query to determine appropriate routing
- **Multi-Agent Coordination**: Can route to multiple agents for complex queries
- **Fallback Handling**: Graceful handling of ambiguous or unclear requests

### 🧠 Query Context Agent  
- **Abbreviation Expansion**: hp → horsepower, mpg → miles_per_gallon, etc.
- **Column Mapping**: Maps query terms to actual dataset columns
- **Context Hints**: Provides guidance for downstream agents
- **Domain Analysis**: Understands dataset structure and content

### 💾 Memory & Session Management
- **Conversation Context**: Maintains chat history and context
- **Session Persistence**: Tracks user interactions across sessions
- **Context Formatting**: Prepares responses for UI display

### 🐼 Pandas Agent
- **DataFrame Operations**: Advanced pandas operations and transformations
- **Statistical Analysis**: Descriptive statistics, correlations, aggregations
- **Data Quality**: Missing value analysis, data type validation
- **Performance Optimization**: Efficient operations on large datasets

### 📈 Charting Agent
- **Dynamic Code Generation**: LLM creates custom Python plotting code
- **Intelligent Chart Selection**: Analyzes data to suggest appropriate visualizations
- **Fallback Charts**: Predefined chart types for reliability
- **Custom Styling**: Professional styling and formatting

### 🔍 Data Search Agent
- **Context-Enhanced Search**: Uses QueryContext for improved accuracy
- **Advanced Filtering**: Multiple operators and complex conditions
- **Smart Summaries**: Intelligent data summaries and insights
- **JSON Responses**: Structured output for programmatic use

### 🐍 Python IDE Agent
- **Code Execution**: Safe execution of custom Python code
- **Library Access**: pandas, numpy, matplotlib, seaborn pre-loaded
- **Debugging Support**: Error handling and code analysis
- **Custom Analysis**: User-defined data transformations and calculations

## Example Queries

### Smart Context Understanding
- "Which car has the highest hp?" → System understands "hp" = "horsepower" and handles domain mismatch gracefully
- "Show me top revenue" → Maps to revenue column and creates ranking visualization
- "What's the avg customer count by region?" → Aggregates and groups data appropriately

### Dynamic Visualizations  
- "Create a violin plot of customer distribution" → Generates custom matplotlib code
- "Show revenue vs expenses with trend line" → Creates scatter plot with regression
- "Make a stacked bar chart of products by region" → Complex multi-dimensional visualization

### Data Exploration
- "Find records where revenue > 1400 and region = 'North'" → Complex filtering
- "Show me data summary and missing values" → Comprehensive dataset analysis
- "What are the correlations between all numeric columns?" → Statistical relationships

### Conversation Flow
- User: "Show me revenue trends"
- System: *Creates line chart*
- User: "Now filter for just Product_A" 
- System: *Remembers context and filters previous analysis*

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB INTERFACE                 │
│         File Upload | Chat Input | Results Display         │
└─────────────────────┬───────────────────────────────────────┘
                      │ User Query
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORCHESTRATION LAYER                       │
│    ┌─────────────┐              ┌─────────────────┐        │
│    │ROUTER AGENT │◄────────────►│COORDINATOR AGENT│        │
│    │Query Intent │              │Workflow Mgmt    │        │
│    │Classification│              │Result Aggregation│        │
│    └─────────────┘              └─────────────────┘        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                CONTEXT & MEMORY LAYER                      │
│ ┌─────────────┐┌─────────────┐┌─────────────┐┌───────────┐ │
│ │QUERY CONTEXT││MEMORY AGENT ││CHAT FORMAT  ││TOOL EXEC  │ │
│ │Abbreviation ││Conversation ││Response     ││Tool Route │ │
│ │Expansion    ││Context      ││Formatting   ││Execution  │ │
│ └─────────────┘└─────────────┘└─────────────┘└───────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               SPECIALIZED AGENTS LAYER                     │
│┌────────────┐┌────────────┐┌────────────┐┌──────────────┐  │
││DATA SEARCH ││PANDAS AGENT││CHARTING    ││PYTHON IDE    │  │
││Text Search ││DataFrame   ││Dynamic     ││Code Execution│  │
││Filtering   ││Operations  ││Charts      ││Analysis      │  │
││Context-    ││Statistics  ││LLM Code    ││Debugging     │  │
││Aware       ││Analysis    ││Generation  ││              │  │
│└────────────┘└────────────┘└────────────┘└──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA & OUTPUT LAYER                      │
│ ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐ │
│ │ DataFrame Mgmt  │ │ Visualizations  │ │ LLM Backend   │ │
│ │ CSV/Excel       │ │ Base64 Images   │ │ GPT-4o-mini   │ │
│ │ Session State   │ │ JSON Results    │ │ Intelligence  │ │
│ └─────────────────┘ └─────────────────┘ └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
1. **User Input** → Streamlit UI or Terminal Chat
2. **Query Routing** → Router classifies intent
3. **Context Enhancement** → QueryContext expands abbreviations 
4. **Memory Integration** → Conversation context added
5. **Agent Processing** → Specialized agents handle requests
6. **Coordination** → Results aggregated and formatted
7. **Response Delivery** → Charts, data, analysis displayed

## Major Improvements & Features

### 🎯 **Context-Aware Query Understanding**
- **QueryContextAgent**: New agent that expands abbreviations (hp → horsepower) and maps query terms to dataset columns
- **Domain Agnostic**: Works with any CSV structure, not limited to specific domains
- **Intelligent Routing**: Enhanced router with better query classification

### 📊 **Dynamic Chart Generation** 
- **LLM-Driven Visualization**: ChartingAgent generates custom Python code for any chart request
- **Adaptive Charts**: Analyzes dataset structure to suggest appropriate visualizations
- **Fallback System**: Robust error handling with predefined chart types

### 🧠 **Multi-Agent Orchestration**
- **LangGraph Integration**: Sophisticated workflow management with conditional routing
- **Memory Management**: Conversation context maintained across interactions
- **Coordinator Agent**: Orchestrates complex multi-agent workflows

### 🔧 **Enhanced Data Operations**
- **Context-Enhanced Search**: DataSearchAgent uses query context for better accuracy
- **Advanced Pandas Operations**: Comprehensive dataframe manipulation and analysis
- **Error Recovery**: Graceful handling of domain mismatches and data issues

### 💻 **Multiple Interfaces**
- **Streamlit Web App**: Modern web interface with file upload and chat
- **Terminal Chat**: Command-line interface for quick interactions  
- **Demo Scripts**: Automated demonstrations of system capabilities

## Technology Stack

- **Frontend**: Streamlit (Web), Terminal (CLI)
- **Agent Framework**: LangGraph + LangChain  
- **LLM Backend**: OpenAI GPT-4o-mini
- **Data Processing**: Pandas + NumPy
- **Visualization**: Matplotlib + Seaborn
- **Language**: Python 3.8+

## Testing & Development

### Running Tests
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test categories  
pytest tests/test_router_agent.py
pytest tests/test_charting_agent.py
pytest tests/test_data_search_agent.py

# Use custom test runner
python run_tests.py
```

### Development Setup
```bash
# Clone and setup
git clone <repository>
cd agent_swarm_analytics
pip install -r requirements.txt

# Set environment variables
echo "OPENAI_API_KEY=your-key-here" > .env

# Run tests to verify setup
pytest

# Start development
streamlit run streamlit_app.py
```

### System Validation
```bash
# Test memory and conversation
python test_memory.py

# Test basic functionality  
python test_simple.py

# Interactive demo
python demo_chat.py
```

## File Structure

```
agent_swarm_analytics/
├── src/
│   ├── agents/                 # All agent implementations
│   │   ├── router_agent.py     # Query routing and classification
│   │   ├── query_context_agent.py  # Context analysis and expansion
│   │   ├── memory_agent.py     # Conversation memory management
│   │   ├── charting_agent.py   # Dynamic visualization generation
│   │   ├── data_search_agent.py # Context-aware data search
│   │   ├── pandas_agent.py     # DataFrame operations
│   │   └── python_ide_agent.py # Code execution
│   ├── langgraph_engine/       # Workflow orchestration
│   │   └── graph_builder.py    # Agent graph construction
│   ├── data/                   # Sample datasets
│   │   └── sample.csv          # Business sample data
│   └── api/                    # FastAPI backend (optional)
├── tests/                      # Comprehensive test suite
├── streamlit_app.py           # Web interface
├── chat_interface.py          # Terminal chat interface
├── demo_chat.py              # Demo conversations
├── architecture_ascii.txt     # System architecture diagram
└── requirements.txt          # Dependencies
```

## Future Roadmap

- **Database Integration**: PostgreSQL, MySQL, MongoDB connectors
- **Real-time Analytics**: Streaming data processing
- **ML Integration**: Scikit-learn, TensorFlow model training
- **Export Features**: PDF reports, Excel dashboards
- **API Extensions**: RESTful API for external integrations
- **Performance**: Caching, async processing, parallel execution

---

For detailed technical documentation, see [TESTING.md](TESTING.md) and the architecture diagram in [architecture_ascii.txt](architecture_ascii.txt).
