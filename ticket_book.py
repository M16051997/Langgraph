from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from datetime import datetime, timedelta
import uuid, json, time, os

from dotenv import load_dotenv
load_dotenv()

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

#LLM and tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Small helper for date parsing
from dateutil import parser as dateparser


# Configure API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Basic sanity checks 
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. Set GEMINI_API_KEY in environment or .env to use the LLM.")
if not TAVILY_API_KEY:
    print("WARNING: TAVILY_API_KEY not set. Web search tool will fail without it.")


# ----------------------------
# Agent State
# ----------------------------
class AgentState(TypedDict):
    """
    AgentState: conversation memory shape stored by the graph.

    messages: list of messages (user/assistant/tool) - langgraph's add_messages decorator manages appending.
    """

    messages: Annotated[List[AnyMessage], add_messages]


# ----------------------------
# In-memory booking store (demo)
# ----------------------------

# For the demo we store tentative & confirmed bookings in a dict keyed by thread_id.
# Production: persistent DB (Postgres/Redis) + transactional semantics.

_bookings_store: Dict[str, List[Dict[str, Any]]] = {}

# ----------------------------
# Utility functions
# ----------------------------
def make_thread_id() -> str:
    '''Create a new unique thread id (UUID4).'''
    return str(uuid. uuid4())

def parse_date_safe(date_str: str) -> datetime:
    '''Try to parse a date string robustly using dateutil; raise ValueError on failure.'''
    if not date_str or not isinstance(date_str, str):
        raise ValueError("Invalid date input.")
    # Accept natural language like "tomorrow" or ISO dates
    parsed = dateparser.parse(date_str, fuzzy=True)
    if not parsed:
        raise ValueError(f"Could not parse date: {date_str}")
    return parsed

def sanitize_location(loc: str) -> str:
    """Simple sanitization of location strings to avoid injection in mock tools."""
    return loc.strip()

# ----------------------------
# Tools Definitions
# ----------------------------
@tool
def web_search(query: str) -> str:
    """
    Tool: web_search
    - Uses TavilySearchResults to fetch top results for a query and returns a summarized string.

    Returns:
        A string with formatted search results (title, snippet, url) joined.
    """
    # Defensive programming: fail gracefully if no API key
    if not TAVILY_API_KEY:
        return "ERROR: TAVILY_API_KEY not configured."

    try:
        search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
        results = search.invoke(query)
    except Exception as e:
        return f"ERROR: web_search failed: {e}"

    formatted = []
    for r in results:
        title = r.get("title", "No title")
        snippet = r.get("content") or r.get("snippet") or ""
        url = r.get("url", "")
        formatted.append(f"Title: {title}\nSnippet: {snippet}\nURL: {url}")
    return "\n\n---\n\n".join(formatted) if formatted else "No results found."


@tool
def check_availability(origin: str, destination: str, depart_date: str) -> str:
    """
    Tool: check_availability (mock)
    - Checks whether seats are available on the requested route+date.
    - For demo, availability logic is simplified.

    Returns:
        JSON string describing availability.
    """
    try:
        depart_dt = parse_date_safe(depart_date)
    except Exception as e:
        return json.dumps({"error": f"Invalid date: {e}"})

    origin = sanitize_location(origin)
    destination = sanitize_location(destination)

    # Mock logic: if date is within next 365 days and origin != destination -> available
    days_until = (depart_dt.date() - datetime.utcnow().date()).days
    if origin.lower() == destination.lower():
        return json.dumps({"available": False, "reason": "Origin and destination are the same."})
    if days_until < 0:
        return json.dumps({"available": False, "reason": "Date is in the past."})
    if days_until > 365:
        return json.dumps({"available": False, "reason": "Date too far in future for demo provider."})
    # Simulate limited capacity: if day-of-month is divisible by 13 -> low seats
    low_capacity = (depart_dt.day % 13 == 0)
    availability = {"available": True, "low_capacity": low_capacity, "days_until": days_until}
    return json.dumps(availability)

@tool
def make_booking(thread_id: str, user_name: str, origin: str, destination: str, depart_date: str) -> str:
    """
    Tool: make_booking (mock)
    - Creates a tentative booking and stores it in the in-memory _bookings_store.
    - Returns a JSON string with tentative booking details (including booking_id).
    - Note: this is 'tentative' and requires human approval in this demo.
    """
    try:
        depart_dt = parse_date_safe(depart_date)
    except Exception as e:
        return json.dumps({"error": f"Invalid date: {e}"})

    origin = sanitize_location(origin)
    destination = sanitize_location(destination)
    booking_id = str(uuid.uuid4())

    record = {
        "booking_id": booking_id,
        "user_name": user_name,
        "origin": origin,
        "destination": destination,
        "depart_date": depart_dt.isoformat(),
        "status": "tentative",
        "created_at": datetime.utcnow().isoformat(),
    }
    _bookings_store.setdefault(thread_id, []).append(record)

    # Return the tentative booking record
    return json.dumps({"result": "tentative_created", "booking": record})


@tool
def confirm_booking(thread_id: str, booking_id: str) -> str:
    """
    Tool: confirm_booking (mock)
    - Marks a tentative booking as 'confirmed'.
    - Returns JSON with confirmation details or an error message.
    """
    items = _bookings_store.get(thread_id, [])
    for item in items:
        if item["booking_id"] == booking_id:
            item["status"] = "confirmed"
            item["confirmed_at"] = datetime.utcnow().isoformat()
            return json.dumps({"result": "confirmed", "booking": item})
    return json.dumps({"error": "booking_id not found"})


@tool
def cancel_booking(thread_id: str, booking_id: str) -> str:
    """
    Tool: cancel_booking (mock)
    - Cancels a booking and records cancellation timestamp.
    """
    items = _bookings_store.get(thread_id, [])
    for item in items:
        if item["booking_id"] == booking_id:
            item["status"] = "cancelled"
            item["cancelled_at"] = datetime.utcnow().isoformat()
            return json.dumps({"result": "cancelled", "booking": item})
    return json.dumps({"error": "booking_id not found"})

# Tool list used by LLM
TOOLS = [web_search, check_availability, make_booking, confirm_booking, cancel_booking]

# ToolNode to execute tools within the LangGraph
tool_node = ToolNode(TOOLS)

# ----------------------------
# LLM Setup (Gemini)
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=GEMINI_API_KEY)

# ----------------------------
# Chatbot Node (decision maker)
# ----------------------------
def chatbot(state: AgentState):
    """
    Chatbot node: binds the available tools to the LLM and asks the LLM to respond based on the conversation history.

    Behavior:
    - The LLM is given a system prompt that describes available tools and expected behavior.
    - The chain returns a response. If the LLM decides to call a tool, the LLM's response will include a tool call in the structured format LangChain uses.
    """
    system_text = (
        "You are a travel assistant. You may use tools for:\n"
        "- web_search(query)\n"
        "- check_availability(origin, destination, depart_date)\n"
        "- make_booking(thread_id, user_name, origin, destination, depart_date)\n"
        "- confirm_booking(thread_id, booking_id)\n"
        "- cancel_booking(thread_id, booking_id)\n\n"
        "When you need up-to-date info, use web_search. When you want to create or confirm bookings, use the booking tools. "
        "Always be conversational and verify user details before booking. If you create a booking, it will be tentative and will require human approval."
    )

    # The prompt includes the conversation history injected as {messages}.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("placeholder", "{messages}")
    ])

    # Bind tools to the LLM so it can generate tool call actions.
    chain = prompt | llm.bind_tools(TOOLS)

    # Invoke with full message history
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


# ----------------------------
# Router: decide next step after chatbot
# ----------------------------
def should_continue(state: AgentState):
    """
    Determine where to route after chatbot:
    - If LLM requested tool calls -> go to "tools" node
    - If LLM asked for human approval -> go to "human" node
    - Otherwise -> END
    """
    last_msg = state["messages"][-1]
    # Chat message may have tool_calls attribute if LLM emitted a tool call action
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    # Quick heuristic: if assistant asks for "approval" or contains "please approve booking" -> human review
    # In production, use structured signals (e.g., message.metadata or special tokens)
    txt = getattr(last_msg, "content", "")
    if isinstance(txt, str) and ("please approve" in txt.lower() or "awaiting approval" in txt.lower()):
        return "human"

    return END

# ----------------------------
# Human-in-the-loop node
# ----------------------------
def human_approval_node(state: AgentState):
    """
    Pause the graph and return the state for human review.
    In this demo, we perform a CLI-based approval: print the last assistant message + tentative booking(s),
    then ask the operator to approve / modify / reject.

    In production:
      - Persist paused state, notify a human via UI, email, or internal dashboard.
      - Human edits state and resumes the graph via API.
    """
    # Extract context for display
    messages = state["messages"]
    last_assistant = messages[-1] if messages else None

    # Show tentative bookings for this thread (if any)
    print("\n=== HUMAN REVIEW REQUIRED ===")
    print("Assistant message (awaiting approval):")
    print(getattr(last_assistant, "content", str(last_assistant)))
    print("\nTentative bookings in store for this thread:")
    # Last message may contain thread id - we expect thread_id in the config of invocation
    # Here we just show all entries; in production, map via thread id
    for tid, recs in _bookings_store.items():
        for r in recs:
            if r["status"] == "tentative":
                print(json.dumps(r, indent=2))
    print("=============================\n")

    # CLI prompt: approve or reject
    while True:
        resp = input("Human action (approve <booking_id> / reject <booking_id> / skip): ").strip()
        if resp.lower() == "skip":
            print("Skipping human approval (agent will continue but booking stays tentative).")
            return state
        parts = resp.split()
        if len(parts) == 2 and parts[0].lower() in {"approve", "reject"}:
            action, booking_id = parts[0].lower(), parts[1]
            if action == "approve":
                print(f"Approving booking {booking_id} ...")
                confirm_booking("", booking_id)  # thread_id not used in mock confirm; in prod supply correct thread
                print("Booking approved.")
            else:
                print(f"Cancelling booking {booking_id} ...")
                cancel_booking("", booking_id)
                print("Booking cancelled.")
            return state
        print("Invalid input. Example commands: 'approve <booking_id>', 'reject <booking_id>', or 'skip'.")


# ----------------------------
# Graph building & memory
# ----------------------------
graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("human", human_approval_node)

# Start -> chatbot
graph_builder.add_edge(START, "chatbot")

# Conditional routing after chatbot
graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", "human": "human", END: END})

# After tools, go back to chatbot for further reasoning
graph_builder.add_edge("tools", "chatbot")

# After human review, go back to chatbot
graph_builder.add_edge("human", "chatbot")

# Memory: small MemorySaver checkpointing; replace with SQLite/Postgres savers in production
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ----------------------------
# Helper: invoke with thread_id management
# ----------------------------
def invoke_with_thread(messages: List[Dict[str, str]], thread_id: str = None) -> Dict[str, Any]:
    """
    Helper to invoke the compiled graph with message history and thread_id.
    If thread_id is None, a new one is generated (stateless chat).
    """
    if thread_id is None:
        thread_id = make_thread_id()

    config = {"configurable": {"thread_id": thread_id}}
    # Graph.invoke expects the initial partial state; we provide messages and config
    result = graph.invoke({"messages": messages}, config=config)
    return {"thread_id": thread_id, "result": result}


# ----------------------------
# CLI demo
# ----------------------------
def cli_demo():
    print("Travel Booking Agent - Demo")
    print("Start a new conversation or resume an existing thread.")
    thread_id = None
    while True:
        if not thread_id:
            choice = input("\nCommands:\n [n] new conversation\n [r] resume with thread_id\n [q] quit\nChoose: ").strip().lower()
            if choice == "n":
                thread_id = None
            elif choice == "r":
                tid = input("Enter thread_id to resume: ").strip()
                if tid:
                    thread_id = tid
                else:
                    print("Invalid thread_id.")
                    continue
            elif choice == "q":
                print("Exiting.")
                return
            else:
                print("Unknown command.")
                continue

        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Ending session.")
            return

        # Prepare message payload and call graph
        messages = [{"role": "user", "content": user_input}]
        out = invoke_with_thread(messages, thread_id)
        thread_id = out["thread_id"]
        result_state = out["result"]

        # Print the assistant's last message
        final_msg = result_state["messages"][-1]
        print("\nAssistant:", getattr(final_msg, "content", final_msg))

        print(f"[Thread ID: {thread_id}] (save this if you want to resume the conversation later)")

# Run CLI if executed
if __name__ == "__main__":
    cli_demo()