# LangSmith Integration Setup Guide

This guide explains how to set up and use LangSmith tracking for the Multi-Agent Data Analytics System.

## Overview

LangSmith provides comprehensive tracking and monitoring for LangChain applications. This integration tracks:

- **Query Processing**: Every user query and its routing decisions
- **Agent Execution**: Individual agent performance and outputs
- **Session Management**: Conversation context and user interactions
- **Performance Metrics**: Response times, success rates, and bottlenecks
- **Error Tracking**: Detailed error logs with context

## Quick Setup

### 1. Get LangSmith API Key

1. Visit [LangSmith](https://smith.langchain.com)
2. Sign up or log in to your account
3. Navigate to Settings → API Keys
4. Create a new API key
5. Copy the API key for configuration

### 2. Configure Environment Variables

Add these variables to your `.env` file:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=data-analytics-agents
```

### 3. Install Dependencies

The required `langsmith` package is already included in `requirements.txt`. If you need to install it separately:

```bash
pip install langsmith
```

### 4. Restart Application

After configuration, restart your Streamlit application:

```bash
streamlit run streamlit_app.py
```

## Features

### 🔍 Query Tracking

Every user query is tracked with:
- Query content and timestamp
- Routing decisions (which agent was selected)
- Processing time and success status
- Session context and conversation flow

### 🤖 Agent Monitoring

Individual agent executions are monitored:
- Input/output data sizes
- Execution time per agent
- Success/failure status
- Error details with context

### 📊 Performance Dashboard

The Streamlit sidebar displays:
- **Tracking Status**: Whether LangSmith is enabled
- **Session Metrics**: Query count, average response time, agents used
- **Performance Insights**: Response time analysis and recommendations
- **Agent Performance**: Individual agent efficiency metrics

### 📤 Data Export

Export session data for analysis:
- CSV export of all queries and metrics
- Session-specific data with timestamps
- Agent performance breakdown

## Architecture

### Components

1. **LangSmithTracker** (`src/utils/langsmith_config.py`)
   - Centralized tracking configuration
   - Session and query management
   - Error logging and context capture

2. **AgentTracker** (`src/utils/agent_tracker.py`)
   - Decorator-based agent monitoring
   - Automatic execution time tracking
   - State initialization and finalization

3. **LangSmithDashboard** (`src/utils/langsmith_dashboard.py`)
   - Streamlit dashboard components
   - Real-time metrics display
   - Export functionality

### Integration Points

- **Graph Builder**: Tracking initialization and finalization
- **Agent Decorators**: Automatic execution monitoring
- **Streamlit App**: Dashboard display and user interface
- **LLM Setup**: Callback manager integration

## Usage Examples

### Viewing Traces

1. Run a query in the Streamlit app
2. Check the sidebar for session metrics
3. Visit the LangSmith dashboard at https://smith.langchain.com
4. Navigate to your project: `data-analytics-agents`
5. View detailed traces for each query

### Analyzing Performance

The dashboard shows:
- **Response Times**: Average and per-agent timing
- **Agent Usage**: Which agents are used most frequently
- **Success Rates**: Query completion statistics
- **Error Patterns**: Common failure points

### Exporting Data

1. Use the "Export Session Data" button in the sidebar
2. Download CSV with query metrics
3. Analyze in Excel or other tools
4. Track performance trends over time

## Troubleshooting

### LangSmith Not Working

**Check Environment Variables:**
```bash
# Verify in your .env file
LANGCHAIN_TRACING_V2=true  # Must be exactly "true"
LANGCHAIN_API_KEY=ls-...   # Must start with "ls-"
```

**Check Sidebar Status:**
- Green checkmark: Tracking enabled
- Warning message: Configuration issue

**Common Issues:**
- API key not set or incorrect format
- `LANGCHAIN_TRACING_V2` not set to "true"
- Network connectivity issues

### Performance Impact

LangSmith tracking adds minimal overhead:
- ~10-50ms per query for logging
- Asynchronous data transmission
- Local caching for reliability

### Disabling Tracking

To disable LangSmith tracking:
1. Set `LANGCHAIN_TRACING_V2=false` in `.env`
2. Or remove the environment variable entirely
3. Restart the application

## Advanced Configuration

### Custom Project Names

Change the project name in `.env`:
```bash
LANGCHAIN_PROJECT=my-custom-project-name
```

### Session Management

Sessions are automatically created with:
- Unique session IDs per Streamlit session
- Conversation context and metadata
- User interaction tracking

### Custom Metadata

The system automatically tracks:
- Dataset information (file names, shapes)
- Query types and routing decisions
- Agent execution context
- Error details and stack traces

## Benefits

### For Development
- **Debugging**: Detailed execution traces
- **Performance**: Identify slow agents or queries
- **Reliability**: Error tracking and patterns
- **Optimization**: Data-driven improvements

### For Production
- **Monitoring**: Real-time system health
- **Analytics**: User behavior insights
- **Scaling**: Performance bottleneck identification
- **Quality**: Success rate tracking

### For Users
- **Transparency**: See how queries are processed
- **Performance**: Response time visibility
- **Insights**: Agent usage patterns
- **Export**: Data for further analysis

## Security Notes

- API keys are stored in environment variables
- No sensitive data is logged by default
- Traces can be configured to exclude PII
- Local development vs production configurations

## Next Steps

1. **Set up your LangSmith account**
2. **Configure environment variables**
3. **Restart the application**
4. **Run some queries to generate data**
5. **Explore the LangSmith dashboard**
6. **Use the Streamlit sidebar metrics**

For more advanced features and configuration options, visit the [LangSmith Documentation](https://docs.smith.langchain.com/).