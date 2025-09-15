"""
Data Analysis Agent using LangGraph
A simple agent that can lookup sales data, analyze it, and create visualizations
"""

# ============================================================================
# IMPORTS AND SETUP
# ============================================================================

import os
import re
import pandas as pd
import duckdb
from typing import Annotated, List, Optional, Union, Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys and Settings
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
TRANSACTION_DATA_FILE_PATH = 'Store_Sales_Price_Elasticity_Promotions_Data.parquet'

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", 
    api_key=GOOGLE_API_KEY,
    temperature=0.1  # Lower temperature for more consistent outputs
)

# Initialize memory for conversation state
memory = InMemorySaver()

# ============================================================================
# TOOL 1: SALES DATA LOOKUP
# This tool converts natural language queries into SQL and executes them
# ============================================================================

def validate_sql_query(sql: str) -> bool:
    """
    Validate that the SQL query is safe (read-only).
    Only allows SELECT statements and WITH clauses (CTEs).
    
    Args:
        sql: SQL query string to validate
        
    Returns:
        Boolean indicating if query is safe
    """
    if not sql or not isinstance(sql, str):
        return False
    
    # Only allow SELECT or WITH...SELECT statements
    safe_pattern = re.compile(r"^\s*(?:WITH\b[\s\S]+?AS\s*\(|SELECT\b)", re.IGNORECASE)
    return bool(safe_pattern.match(sql.strip()))


def extract_sql_from_response(text: str) -> Optional[str]:
    """
    Extract SQL query from LLM response text.
    Handles various formats like code blocks or plain SQL.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Extracted SQL query or None if not found
    """
    if not text:
        return None
    
    # Try to extract from code blocks first (```sql ... ``` or ``` ... ```)
    code_block_match = re.search(r"```(?:sql)?\s*([\s\S]+?)\s*```", text, re.IGNORECASE)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Try to find SELECT statement
    select_match = re.search(r"(SELECT\b[\s\S]+?)(?:;|$)", text, re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return None


def generate_sql_query(
    user_request: str,
    table_columns: List[str],
    table_name: str = "sales",
    row_limit: int = 1000
) -> str:
    """
    Generate a SQL query from natural language using the LLM.
    
    Args:
        user_request: Natural language description of what data to fetch
        table_columns: List of available column names
        table_name: Name of the table to query
        row_limit: Maximum number of rows to return (for safety)
        
    Returns:
        Valid SQL query string
        
    Raises:
        ValueError: If generated SQL is invalid or unsafe
    """
    
    # Prepare the column list for the prompt
    columns_text = ", ".join(table_columns) if table_columns else "ALL_COLUMNS"
    
    # System prompt - instructs the LLM to only output SQL
    system_prompt = """You are a SQL expert assistant that ONLY outputs SQL queries.
    Rules:
    - Return ONLY a single SQL query, no explanations
    - Use the exact table name and column names provided
    - Only use SELECT statements (read-only queries)
    - Add appropriate LIMIT clause for safety
    """
    
    # User prompt with the specific request and table info
    user_prompt = f"""
    User request: {user_request}
    
    Table name: {table_name}
    Available columns: {columns_text}
    
    Generate a SQL query to fulfill this request.
    Add 'LIMIT {row_limit}' if not specified.
    """
    
    # Create messages for the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Get response from LLM
    response = llm.invoke(messages)
    
    # Extract the SQL from the response
    raw_text = response.content if hasattr(response, 'content') else str(response)
    sql_query = extract_sql_from_response(raw_text) or raw_text.strip()
    
    # Clean up the SQL query
    sql_query = sql_query.strip()
    
    # Add LIMIT if not present (for safety)
    if "limit" not in sql_query.lower():
        sql_query = sql_query.rstrip(";") + f" LIMIT {row_limit}"
    
    # Ensure it ends with semicolon
    if not sql_query.endswith(";"):
        sql_query += ";"
    
    # Validate the SQL for safety
    if not validate_sql_query(sql_query):
        raise ValueError(f"Generated SQL is not safe or valid: {sql_query}")
    
    return sql_query


@tool
def lookup_sales_data(query: str) -> str:
    """
    Look up sales data from the parquet file using natural language queries.
    Converts the query to SQL and executes it against the data.
    
    Args:
        query: Natural language query about sales data
        
    Returns:
        String representation of the query results
    """
    try:
        # Step 1: Load the parquet file into a pandas DataFrame
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        
        # Step 2: Create a DuckDB table from the DataFrame
        table_name = "sales"
        duckdb.sql(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        
        # Step 3: Generate SQL query from natural language
        sql_query = generate_sql_query(
            user_request=query,
            table_columns=list(df.columns),
            table_name=table_name
        )
        
        print(f"Generated SQL: {sql_query}")  # Debug output
        
        # Step 4: Execute the SQL query
        result_df = duckdb.sql(sql_query).df()
        
        # Step 5: Return results as formatted string
        if result_df.empty:
            return "No data found matching your query."
        
        return f"Query Results:\n{result_df.to_string()}"
        
    except Exception as e:
        return f"Error accessing sales data: {str(e)}"


# ============================================================================
# TOOL 2: DATA ANALYSIS
# This tool performs analytical insights on the data
# ============================================================================

def prepare_data_summary(df: pd.DataFrame, max_rows: int = 5) -> str:
    """
    Create a compact text summary of a DataFrame for LLM analysis.
    
    Args:
        df: DataFrame to summarize
        max_rows: Number of sample rows to include
        
    Returns:
        Formatted string summary of the data
    """
    summary_parts = []
    
    # Add basic info
    summary_parts.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    summary_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Add sample rows
    summary_parts.append(f"\nFirst {max_rows} rows:")
    summary_parts.append(df.head(max_rows).to_string())
    
    # Add statistical summary for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary_parts.append("\nNumeric Column Statistics:")
        summary_parts.append(df[numeric_cols].describe().to_string())
    
    # Add value counts for categorical columns (if any)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
        if df[col].nunique() <= 10:  # Only if few unique values
            summary_parts.append(f"\n{col} Value Counts:")
            summary_parts.append(df[col].value_counts().head(5).to_string())
    
    return "\n".join(summary_parts)


@tool
def analyze_sales_data(query: str, data: str) -> str:
    """
    Analyze sales data and provide insights based on the user's question.
    
    Args:
        query: The analytical question to answer
        data: String representation of the data to analyze
        
    Returns:
        Analytical insights as a formatted string
    """
    try:
        # System prompt for analysis
        system_prompt = """You are a expert data analyst for a retail store.
        Analyze the provided sales data and answer the user's question.
        
        Provide your analysis in this format:
        1. KEY INSIGHTS (3 bullet points)
        2. ANOMALIES OR PATTERNS (if any)
        3. RECOMMENDATIONS (2-3 actionable items)
        
        Be specific, use numbers, and base everything on the actual data provided.
        """
        
        # User prompt with data and question
        user_prompt = f"""
        Data to analyze:
        {data}
        
        Question: {query}
        
        Please provide a thorough but concise analysis.
        """
        
        # Create messages for the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # Get analysis from LLM
        response = llm.invoke(messages)
        
        # Extract and return the analysis
        analysis = response.content if hasattr(response, 'content') else str(response)
        return f"Analysis Results:\n{analysis}"
        
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


# ============================================================================
# TOOL 3: DATA VISUALIZATION
# This tool generates Python code for creating charts
# ============================================================================

# Add these imports at the top of your file
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

# ============================================================================
# TOOL 3: DATA VISUALIZATION
# This tool generates Python code for creating charts
# ============================================================================

def create_chart_from_data(data_str: str, config: Dict[str, Any]) -> str:
    """
    Actually create and save a chart from the data.
    
    Args:
        data_str: String representation of the data
        config: Chart configuration dictionary
        
    Returns:
        Path to saved chart or base64 encoded image
    """
    try:
        import pandas as pd
        from io import StringIO
        
        # Parse the data string back to DataFrame
        if "Query Results" in data_str:
            # Extract the data portion after the header
            lines = data_str.split('\n')
            data_lines = []
            in_data_section = False
            
            for line in lines:
                # Skip until we find the separator
                if '=' in line and len(line) > 20:
                    in_data_section = True
                    continue
                # Stop if we hit another separator or empty lines at the end
                if in_data_section:
                    if line.strip() and not line.startswith('...'):
                        data_lines.append(line)
            
            if data_lines:
                # Parse the table data
                # First line after separator might be headers with data
                data_text = '\n'.join(data_lines)
                
                # Try to parse as space-separated values
                try:
                    df = pd.read_csv(StringIO(data_text), sep=r'\s+', engine='python')
                except:
                    # If that fails, try to parse manually
                    rows = []
                    for line in data_lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            # Assume first column is index, second is ID/key, third is value
                            if len(parts) >= 3:
                                rows.append({'key': parts[1], 'value': float(parts[2])})
                            else:
                                rows.append({'key': parts[0], 'value': float(parts[1])})
                    df = pd.DataFrame(rows)
            else:
                raise ValueError("No data found in query results")
        else:
            # Try direct parsing
            df = pd.read_csv(StringIO(data_str), sep=r'\s+', engine='python')
        
        # Clean up DataFrame
        df.columns = [col.strip() for col in df.columns]
        
        # Detect the actual column names for charting
        if 'SKU_Coded' in df.columns:
            x_col = 'SKU_Coded'
        elif 'Store_Number' in df.columns:
            x_col = 'Store_Number'
        elif 'Sold_Date' in df.columns:
            x_col = 'Sold_Date'
        elif 'key' in df.columns:
            x_col = 'key'
        else:
            x_col = df.columns[0]
        
        if 'Total_Sales' in df.columns:
            y_col = 'Total_Sales'
        elif 'Total_Sale_Value' in df.columns:
            y_col = 'Total_Sale_Value'
        elif 'Daily_Sales' in df.columns:
            y_col = 'Daily_Sales'
        elif 'value' in df.columns:
            y_col = 'value'
        else:
            # Find a numeric column
            for col in df.columns:
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    y_col = col
                    break
            else:
                y_col = df.columns[-1]
        
        # Update config with detected columns
        config['x_axis'] = x_col
        config['y_axis'] = y_col
        
        # Set up the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Create chart based on type
        chart_type = config.get('chart_type', 'bar').lower()
        title = config.get('title', 'Sales Data Visualization')
        
        if chart_type == 'bar':
            # Limit to top entries for readability
            df_plot = df.head(20) if len(df) > 20 else df
            
            # Create bar chart
            bars = ax.bar(range(len(df_plot)), df_plot[y_col].values, color='steelblue', edgecolor='navy', linewidth=0.5)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.0f}', ha='center', va='bottom', fontsize=8)
            
            # Set x-axis labels
            ax.set_xticks(range(len(df_plot)))
            ax.set_xticklabels(df_plot[x_col].values, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
            ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
            
        elif chart_type == 'line':
            ax.plot(range(len(df)), df[y_col].values, marker='o', linewidth=2, markersize=4, color='darkblue')
            ax.set_xticks(range(0, len(df), max(1, len(df)//10)))  # Show every 10th label
            ax.set_xticklabels(df[x_col].values[::max(1, len(df)//10)], rotation=45, ha='right')
            ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
            ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif chart_type == 'pie':
            df_plot = df.head(10) if len(df) > 10 else df
            colors = plt.cm.Set3(range(len(df_plot)))
            wedges, texts, autotexts = ax.pie(df_plot[y_col].values, 
                                              labels=df_plot[x_col].values, 
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_weight('bold')
                
        else:
            # Default bar chart
            ax.bar(range(len(df)), df[y_col].values, color='steelblue')
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df[x_col].values, rotation=45, ha='right')
            ax.set_xlabel(x_col, fontsize=11, fontweight='bold')
            ax.set_ylabel(y_col, fontsize=11, fontweight='bold')
        
        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Improve layout
        plt.tight_layout()
        
        # Save the chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{chart_type}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        
        # Close plot
        plt.close()
        
        return f"""✅ Chart successfully created and saved!
📊 File: {filename}
📈 Type: {chart_type.title()} Chart
📉 Data: {len(df)} rows
🎨 Columns: {x_col} (x-axis) vs {y_col} (y-axis)"""
        
    except Exception as e:
        import traceback
        return f"❌ Error creating chart: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"


def extract_chart_configuration(data: str, visualization_goal: str) -> Dict[str, Any]:
    """
    Generate chart configuration from data and visualization requirements.
    
    Args:
        data: String representation of the data
        visualization_goal: Description of desired visualization
        
    Returns:
        Dictionary with chart configuration
    """
    # Analyze the visualization goal to determine chart type
    goal_lower = visualization_goal.lower()
    
    # Smart chart type detection based on keywords
    if any(word in goal_lower for word in ['bar', 'bars', 'column']):
        chart_type = 'bar'
    elif any(word in goal_lower for word in ['line', 'trend', 'time', 'series']):
        chart_type = 'line'
    elif any(word in goal_lower for word in ['scatter', 'correlation', 'relationship']):
        chart_type = 'scatter'
    elif any(word in goal_lower for word in ['pie', 'proportion', 'percentage', 'share']):
        chart_type = 'pie'
    elif any(word in goal_lower for word in ['histogram', 'distribution', 'frequency']):
        chart_type = 'histogram'
    else:
        chart_type = 'bar'  # default
    
    # Try to extract column names from the data
    x_axis = 'category'
    y_axis = 'value'
    
    # Parse data to get actual column names
    try:
        lines = data.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('Query Results') and '=' not in line:
                # Try to identify header line
                parts = line.split()
                if len(parts) >= 2 and not parts[0].isdigit():
                    x_axis = parts[0]
                    y_axis = parts[-1]
                    break
    except:
        pass
    
    # Generate title from goal
    title = visualization_goal if len(visualization_goal) < 50 else "Sales Data Visualization"
    
    return {
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "title": title,
        "description": f"Visualization of {visualization_goal}"
    }


@tool
def generate_visualization(data: str, visualization_goal: str) -> str:
    """
    Generate and create a data visualization chart.
    
    Args:
        data: String representation of the data to visualize
        visualization_goal: Description of what the visualization should show
        
    Returns:
        Success message with chart location or error message
    """
    try:
        # Step 1: Get chart configuration
        config = extract_chart_configuration(data, visualization_goal)
        
        # Step 2: Create the actual chart
        result = create_chart_from_data(data, config)
        
        # Step 3: Also generate the Python code for reference
        code = f"""
# Python code to recreate this visualization:
import pandas as pd
import matplotlib.pyplot as plt

# Your data would go here
# df = pd.read_csv('your_data.csv')

plt.figure(figsize=(12, 6))
plt.{config['chart_type']}(df['{config['x_axis']}'], df['{config['y_axis']}'])
plt.title('{config['title']}')
plt.xlabel('{config['x_axis']}')
plt.ylabel('{config['y_axis']}')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('chart.png')
plt.show()
"""
        
        return f"{result}\n\n💻 Code to recreate:\n{code}"
        
    except Exception as e:
        return f"Error generating visualization: {str(e)}"


# ============================================================================
# AGENT STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    State definition for the agent.
    Tracks messages and intermediate results throughout the conversation.
    """
    messages: Annotated[List, add_messages]
    current_data: Optional[str]  # Stores data from lookup for use in other tools


# ============================================================================
# AGENT ROUTER FUNCTION
# ============================================================================

def route_query(state: AgentState) -> str:
    """
    Router function that decides which tool to use based on the user's query.
    
    Args:
        state: Current agent state with messages
        
    Returns:
        Name of the next node to execute
    """
    # Get the last user message
    messages = state["messages"]
    last_message = messages[-1]
    
    # Extract the query text
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    query_lower = query.lower()
    
    # Routing logic based on keywords and intent
    
    # Check for data lookup requests
    lookup_keywords = ['show', 'get', 'fetch', 'find', 'lookup', 'retrieve', 'what is', 'list', 'display']
    if any(keyword in query_lower for keyword in lookup_keywords):
        return "lookup_data"
    
    # Check for analysis requests
    analysis_keywords = ['analyze', 'insight', 'trend', 'pattern', 'anomaly', 'explain', 'why', 'correlation']
    if any(keyword in query_lower for keyword in analysis_keywords):
        return "analyze_data"
    
    # Check for visualization requests
    viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'draw', 'diagram', 'histogram', 'bar', 'line']
    if any(keyword in query_lower for keyword in viz_keywords):
        return "visualize_data"
    
    # Default to lookup if unclear
    return "lookup_data"


# ============================================================================
# NODE FUNCTIONS FOR THE GRAPH
# ============================================================================

def lookup_node(state: AgentState) -> AgentState:
    """Node that handles data lookup"""
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Execute the lookup tool
    result = lookup_sales_data.invoke({"query": query})
    
    # Update state with the result
    state["current_data"] = result
    state["messages"].append(HumanMessage(content=result))
    
    return state


def analyze_node(state: AgentState) -> AgentState:
    """Node that handles data analysis"""
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Use current data if available, otherwise fetch it first
    data = state.get("current_data", "")
    if not data:
        # First lookup the data
        lookup_result = lookup_sales_data.invoke({"query": query})
        data = lookup_result
    
    # Execute the analysis tool
    result = analyze_sales_data.invoke({"query": query, "data": data})
    
    # Update state with the result
    state["messages"].append(HumanMessage(content=result))
    
    return state


def visualize_node(state: AgentState) -> AgentState:
    """Node that handles visualization generation"""
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # Use current data if available
    data = state.get("current_data", "")
    if not data:
        # Need to fetch data first
        lookup_result = lookup_sales_data.invoke({"query": "Show sample sales data"})
        data = lookup_result
    
    # Execute the visualization tool
    result = generate_visualization.invoke({"data": data, "visualization_goal": query})
    
    # Update state with the result
    state["messages"].append(HumanMessage(content=result))
    
    return state


# ============================================================================
# BUILD THE LANGGRAPH
# ============================================================================

def create_sales_agent():
    """
    Create and configure the LangGraph agent for sales data analysis.
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize the graph with our state schema
    workflow = StateGraph(AgentState)

    # --- router node must be a pass-through node (returns state) ---
    def router_node(state: AgentState) -> AgentState:
        return state

    # Add nodes to the graph
    workflow.add_node("router", router_node)
    workflow.add_node("lookup_data", lookup_node)
    workflow.add_node("analyze_data", analyze_node)
    workflow.add_node("visualize_data", visualize_node)

    # Define the edges (connections between nodes)
    workflow.add_edge(START, "router")

    # Add conditional edges from router to tools
    # the second arg must be a function that maps state -> key in mapping
    workflow.add_conditional_edges(
        "router",
        lambda st: route_query(st),   # route_query returns "lookup_data" / "analyze_data" / "visualize_data"
        {
            "lookup_data": "lookup_data",
            "analyze_data": "analyze_data",
            "visualize_data": "visualize_data"
        }
    )

    # All tool nodes lead to END
    workflow.add_edge("lookup_data", END)
    workflow.add_edge("analyze_data", END)
    workflow.add_edge("visualize_data", END)

    # Compile the graph
    app = workflow.compile(checkpointer=memory)

    return app


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(query: str):
    """
    Run the agent with a user query.
    
    Args:
        query: User's natural language query
        
    Returns:
        Agent's response
    """
    # Create the agent
    agent = create_sales_agent()
    
    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_data": None
    }
    
    # Run the agent
    result = agent.invoke(initial_state, {"configurable": {"thread_id": "1"}})
    
    # Extract and return the final response
    if result["messages"]:
        final_message = result["messages"][-1]
        return final_message.content if hasattr(final_message, 'content') else str(final_message)
    
    return "No response generated"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def inspect_data_structure():
    """
    Helper function to inspect the structure of your sales data.
    Useful for understanding columns and data types.
    """
    try:
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        print("=" * 80)
        print("DATA STRUCTURE INSPECTION")
        print("=" * 80)
        print(f"\n📊 File: {TRANSACTION_DATA_FILE_PATH}")
        print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print("\n📋 Column Information:")
        print("-" * 40)
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()
            print(f"  • {col}")
            print(f"    - Type: {dtype}")
            print(f"    - Unique values: {unique_count}")
            print(f"    - Null values: {null_count}")
            if dtype in ['object', 'datetime64[ns]']:
                sample_values = df[col].dropna().head(3).tolist()
                print(f"    - Sample: {sample_values}")
        
        print("\n📈 First 5 rows:")
        print("-" * 40)
        print(df.head())
        
        # Check for date columns
        print("\n📅 Date Column Detection:")
        print("-" * 40)
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                print(f"  • {col}: {df[col].dtype}")
                if df[col].dtype == 'object':
                    print(f"    ⚠️  String type - needs conversion to datetime")
                    # Try to parse a sample
                    try:
                        sample = pd.to_datetime(df[col].iloc[0])
                        print(f"    ✅ Can be converted: {sample}")
                    except:
                        print(f"    ❌ Cannot auto-convert, sample: {df[col].iloc[0]}")
        
        return df
    except Exception as e:
        print(f"Error inspecting data: {e}")
        return None


def test_date_queries():
    """
    Test function to verify date queries are working correctly.
    """
    print("\n" + "=" * 80)
    print("TESTING DATE QUERIES")
    print("=" * 80)
    
    try:
        # Load data
        df = pd.read_parquet(TRANSACTION_DATA_FILE_PATH)
        df['Sold_Date'] = pd.to_datetime(df['Sold_Date']).dt.date
        
        # Setup DuckDB
        conn = duckdb.connect(':memory:')
        conn.register('df_view', df)
        conn.execute("""
            CREATE TABLE sales AS 
            SELECT 
                Store_Number,
                SKU_Coded,
                Product_Class_Code,
                CAST(Sold_Date AS DATE) as Sold_Date,
                Qty_Sold,
                Total_Sale_Value,
                On_Promo
            FROM df_view
        """)
        
        # Test queries
        test_queries = [
            ("Date range in data", "SELECT MIN(Sold_Date) as min_date, MAX(Sold_Date) as max_date FROM sales"),
            ("Last 30 records", "SELECT * FROM sales ORDER BY Sold_Date DESC LIMIT 30"),
            ("November 2021 data", "SELECT COUNT(*) as count, SUM(Total_Sale_Value) as total FROM sales WHERE Sold_Date >= DATE '2021-11-01' AND Sold_Date < DATE '2021-12-01'"),
            ("Group by date", "SELECT Sold_Date, COUNT(*) as transactions, SUM(Total_Sale_Value) as daily_total FROM sales GROUP BY Sold_Date ORDER BY Sold_Date DESC LIMIT 10"),
        ]
        
        for desc, query in test_queries:
            print(f"\n📊 {desc}")
            print(f"Query: {query}")
            print("-" * 40)
            try:
                result = conn.execute(query).df()
                print(result.to_string())
            except Exception as e:
                print(f"Error: {e}")
        
        conn.close()
        print("\n✅ Date query testing complete!")
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")


if __name__ == "__main__":
    # First, inspect the data structure
    print("\n🔍 Inspecting your data file...")
    inspect_data_structure()
    
    # Test date queries
    test_date_queries()
    
    # Example queries to test the agent
    test_queries = [
        "Show me all sales from November 2021",
        "What are the top 5 products by total sales?",
        "Show me sales data for store 1320",
        "Calculate total sales by store"
    ]
    
    print("\n" + "=" * 80)
    print("SALES DATA ANALYSIS AGENT")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n📊 Query: {query}")
        print("-" * 40)
        
        try:
            response = run_agent(query)
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("=" * 80)
    
    # Interactive mode
    print("\n🤖 Agent is ready! Type 'quit' to exit.")
    while True:
        user_query = input("\n👤 Your query: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            response = run_agent(user_query)
            print(f"\n🤖 Agent: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")