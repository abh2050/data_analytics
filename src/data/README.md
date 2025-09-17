# Sample Datasets

This directory contains sample CSV files for testing and demonstrating the data analytics capabilities of the system.

## Available Datasets

### 1. sample.csv
**Business Sales Data**
- **Columns**: date, revenue, expenses, profit, region, category, customers
- **Records**: 50 rows of daily business metrics
- **Date Range**: January-February 2023
- **Use Cases**: 
  - Revenue analysis and trends
  - Regional performance comparison
  - Category-wise sales analysis
  - Profit margin calculations
  - Customer acquisition metrics

### 2. employees.csv
**Employee Information**
- **Columns**: employee_id, name, age, department, salary, years_experience, performance_score, last_promotion
- **Records**: 20 employee records
- **Use Cases**:
  - Salary analysis by department
  - Performance evaluation
  - Experience vs. salary correlation
  - Promotion pattern analysis
  - Department statistics

### 3. weather.csv
**Weather Station Data**
- **Columns**: timestamp, temperature, humidity, pressure, wind_speed, location
- **Records**: 50+ hourly measurements from 2 stations
- **Date Range**: January 1-2, 2023
- **Use Cases**:
  - Time series analysis
  - Weather pattern visualization
  - Station comparison
  - Environmental monitoring
  - Correlation analysis between weather variables

## How to Use

1. **Upload via Streamlit Interface**: Use the file uploader in the sidebar of the web app
2. **Direct Reference**: The system can automatically load these files for testing
3. **API Usage**: Reference these files in API calls for data analysis

## Example Queries

### For sample.csv:
- "Show me the revenue trends over time"
- "Which region has the highest profit margins?"
- "Create a chart comparing revenue by category"
- "What's the average profit per customer?"

### For employees.csv:
- "What's the average salary by department?"
- "Show the relationship between experience and salary"
- "Who are the top performers in each department?"
- "Create a visualization of salary distribution"

### For weather.csv:
- "Plot temperature trends over time"
- "Compare weather conditions between stations"
- "Show the correlation between temperature and humidity"
- "Create a time series chart of all weather variables"

## Data Quality

All sample datasets are:
- ✅ Clean and well-formatted
- ✅ No missing values
- ✅ Consistent data types
- ✅ Realistic data ranges
- ✅ Suitable for various analysis types

## File Formats

- All files are in CSV format with headers
- UTF-8 encoding
- Comma-separated values
- Standard date formats (YYYY-MM-DD for dates, YYYY-MM-DD HH:MM:SS for timestamps)