# main.py
import os
from datetime import datetime, time
import pytz
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot, Command
from langchain_core.messages import AIMessage, HumanMessage
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import json
# Import the calendar tools from our event_handler module.
from event_handler import create_event, get_events, update_event, delete_event, init_google_calendar, edit_todo

# ------------------------------------------------------------------------------
# 1. Load environment variables and initialize the LLM model
# ------------------------------------------------------------------------------
load_dotenv()
TOKEN_FILE = "token.json"
CLIENT_SECRET_FILE = "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def init_model() -> ChatOpenAI:
    try:
        MODEL_NAME = os.getenv("MODEL_NAME")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=0.3,
            api_key=OPENAI_API_KEY,
        )
        print("Model initialized successfully:", llm)
        return llm
    except Exception as e:
        print(f"Model cannot be initialized: {e}")

llm = init_model()

# ------------------------------------------------------------------------------
# 2. Define the agent node
#
#    If the user's message appears to be a to‑do list (detected via keywords), the prompt instructs
#    the LLM to extract individual tasks, estimate durations, and call the tool "create_event"
#    for each scheduled task. Otherwise, it falls back to normal calendar operations.
# ------------------------------------------------------------------------------
def calendar_agent(state: MessagesState) -> MessagesState:
    try:
        user_message = state["messages"][-1].content
        toronto_tz = pytz.timezone('America/Toronto')
        now = datetime.now(toronto_tz)
        today_str = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # First fetch today's events and todos
        today_start = f"{today_str}T00:00:00-04:00"  # Toronto timezone offset
        today_end = f"{today_str}T23:59:59-04:00"
        try:
            today_events = get_events.invoke({
                "startDateTime": today_start,
                "endDateTime": today_end
            })
            events_context = "\nToday's events:\n" + "\n".join([
                f"- {event['summary']}: {event['start']['dateTime']} to {event['end']['dateTime']}"
                for event in today_events.get('items', [])
            ])
        except Exception as e:
            events_context = f"\nNo events found for today ({today_str} Toronto time)."
            
        try:
            with open("todos.json", "r") as f:
                todos = json.load(f)
            todos_context = "\nCurrent todos:\n" + "\n".join([f"- {todo}" for todo in todos])
        except (FileNotFoundError, json.JSONDecodeError):
            todos_context = "\nNo existing todos."
# 3. If the user provides a todo list for scheduling:
#    - Parse and extract each task
#    - If times aren't specified, set needs_deep_analysis to True
#    - If times are specified, create events with:
#      * summary: task description
#      * location: empty string
#      * description: "Scheduled from to-do list"
#      * start_time and end_time in ISO 8601 format
#      * no attendees

        prompt = f"""
You are an intelligent assistant that manages a Google Calendar and todo lists.
{events_context}
{todos_context}

Your task is to process the user's request appropriately:

1. If the user wants to add items to the todo list:
   - Use the edit_todo tool to add the tasks
   - Return a response confirming the addition

2. If the user wants to schedule events or manage calendar:
   - For calendar queries: Use get_events to fetch relevant events
   - For creating events: Use create_event (one at a time)
   - For updating/deleting: Use update_event/delete_event with the event ID
   

User input: "{user_message}"
Today's date: {today_str}
Current time: {current_time}

Output must be a valid JSON with this structure:
{{
    "message": "Message describing the action taken",
    "needs_deep_analysis": False,
    "scheduling_context": {{ "tasks": [] }},
    "response_for_user": "User-friendly response (empty if needs_deep_analysis is true)"
}}
"""
        graph_agent = create_react_agent(
            llm,
            tools=[create_event, get_events, update_event, delete_event, edit_todo],
            state_modifier=prompt
        )
        result = graph_agent.invoke(state)
        
        # Ensure the response is proper JSON
        try:
            response_content = json.loads(result['messages'][-1].content)
        except json.JSONDecodeError:
            response_content = {
                "message": result['messages'][-1].content,
                "needs_deep_analysis": False,
                "scheduling_context": {},
                "response_for_user": result['messages'][-1].content
            }
        
        result["messages"][-1] = HumanMessage(content=json.dumps(response_content), name="calendar")
        state["messages"].extend(result["messages"])
        return state

    except Exception as e:
        print(f"Error in calendar_agent: {e}")
        # Return a valid JSON response even in case of error
        error_response = {
            "message": f"Error: {str(e)}",
            "needs_deep_analysis": False,
            "scheduling_context": {},
            "response_for_user": "I encountered an error processing your request."
        }
        state["messages"].append(HumanMessage(content=json.dumps(error_response), name="calendar"))
        return state
    

def scheduling_agent(state: MessagesState) -> MessagesState:
    try:
        agent_message = state["messages"][-1].content
        date = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        
        # Ensure we're working with parsed JSON
        try:
            parsed_message = json.loads(agent_message)
        except json.JSONDecodeError:
            parsed_message = {"scheduling_context": {}, "message": agent_message}

        prompt = f"""
        You are an intelligent task scheduling that schedules user's tasks or events at reasonable times by analysing user's schedule for the day. You need to think how much time will each task take and what order should to schedule the tasks in.
        Remember that today's date and time {date}. Schedule events only after the current time without overlap with existing events.
        Your input: {json.dumps(parsed_message)}
        output date/time values in ISO 8601/RFC3339 format including the time zone information.
        Output all user's tasks with the scheduled start time and end time and all other information you received. Respond only in valid json format.
        """
    
        deepseek = ChatOllama(model='deepseek-r1:8b')
        graph_agent = create_react_agent(model=deepseek, tools=[], state_modifier=prompt)
        result = graph_agent.invoke(state)
        
        # Ensure the response is proper JSON
        try:
            response_content = json.loads(result['messages'][-1].content.split('</think>')[-1])
        except json.JSONDecodeError:
            response_content = {
                "message": "Scheduled tasks",
                "needs_deep_analysis": False,
                "scheduling_context": {},
                "response_for_user": result['messages'][-1].content
            }
        
        result["messages"][-1] = HumanMessage(content=json.dumps(response_content), name="calendar")
        state["messages"].extend(result["messages"])
        return state
    
    except Exception as e:
        print(f"Error in scheduling_agent: {e}")
        error_response = {
            "message": f"Error: {str(e)}",
            "needs_deep_analysis": False,
            "scheduling_context": {},
            "response_for_user": "I encountered an error while scheduling tasks."
        }
        state["messages"].append(HumanMessage(content=json.dumps(error_response), name="calendar"))
        return state

# ------------------------------------------------------------------------------
# 3. (Optional) A helper to print streaming output for debugging.
# ------------------------------------------------------------------------------
def print_stream(stream):
    try:
        for s in stream:
            if isinstance(s, dict):
                if "branch" in s:
                    print(f"Branch condition met: {s['branch']}")
                elif "agent" in s and "messages" in s["agent"]:
                    message = s["agent"]["messages"][-1]
                    if "AIMessage" in str(type(message)):
                        print(message.content)
                else:
                    print("Other stream output:", s)
            else:
                print("Unexpected stream format:", s)
    except Exception as e:
        print(f"Error in print_stream: {e}")

def schedule_decision(state: dict):
    if json.loads(state['messages'][-1].content)['needs_deep_analysis']:
        return "scheduler"
    else: 
        return END

# ------------------------------------------------------------------------------
# 4. Build the workflow graph
# ------------------------------------------------------------------------------
def get_workflow() -> CompiledStateGraph:
    workflow = StateGraph(MessagesState)
    workflow.add_node("calendar", calendar_agent)
    workflow.add_node("scheduler", scheduling_agent)
    workflow.add_edge(START, "calendar")
    workflow.add_conditional_edges("calendar", schedule_decision)
    workflow.add_edge("scheduler", "calendar")
    # workflow.add_edge("calendar", END)
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph

# ------------------------------------------------------------------------------
# 5. Main runner: Initialize Google Calendar and run the agent workflow.
# ------------------------------------------------------------------------------
def run_chatbot(graph: CompiledStateGraph, state: MessagesState, creds) -> StateSnapshot:
    init_google_calendar(creds)
    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream(state, config=config):
        print("--------------------------------------------------------------------")
        print(chunk)
    return graph.get_state(config=config)

# ------------------------------------------------------------------------------
# 6. For local testing: simulate a to-do list input.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Replace the below with your actual Google Calendar credentials.
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=8080)
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    # creds = {"dummy": "credentials"}
    
    # Example user message containing a to-do list.
    initial_message = HumanMessage(
        content="List tomorrows events"
    )
    state = MessagesState(messages=[initial_message])
    
    workflow_graph = get_workflow()
    final_state = run_chatbot(workflow_graph, state, creds)
    print("Final state:", final_state.values['messages'][-1].content)
