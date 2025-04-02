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
        today_str = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if message is a todo list addition
        if user_message.lower().startswith("add") and "to the todo list" in user_message.lower():
            # Extract the todo item between "add" and "to the todo list"
            todo_item = user_message[3:].lower().split("to the todo list")[0].strip()
            if todo_item:
                updated_todos = edit_todo(todo_item)
                response = {
                    "message": f"Added '{todo_item}' to todo list",
                    "needs_deep_analysis": False,
                    "scheduling_context": {},
                    "response_for_user": f"✅ Added to your todo list: {todo_item}"
                }
                result = {"messages": [AIMessage(content=json.dumps(response))]}
                result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="calendar")
                state["messages"].extend(result["messages"])
                return state
       
        # Original calendar handling prompt for other cases
        prompt = f"""
You are an intelligent assistant that manages a Google Calendar.
You can call only one tool at a time, once you create one event you have to call again if you want to create another event.
While updating or deletiang events, get all the events for the mentioned date from 12am to 11:59pm. Use the id of that particular event to perform the necessary action.
output date/time values in ISO 8601/RFC3339 format including the time zone information.
If the user says Add to to do list, you have to call the edit_todo tool to add the task to the to do list.
If the user has provided a to-do list. Your task is to:
  1. Parse the following to-do list input and extract each task.
  2. Fetch the events of the mentioned day using get_event tool as prescribed from 12am to 11:59pm.
  3. If you do not have times for each task set the boolean needs_deep_analysis as True for scheduling tasks and return the output in the mentioned format. There exists an agent that will provide you with the times for each events. You can create events only after that. 
  4. If you do have times, set the boolean needs_deep_analysis as False and move to the next step.
  5. For each scheduled task, call the tool "create_event" with these parameters:
     - summary: the task description.
     - location: an empty string if not provided.
     - description: "Scheduled from to-do list".
     - start_time: the scheduled start time.
     - end_time: the scheduled end time.
     - attendees: an empty list.
User input: "{user_message}"
Today's date is {today_str}.
current time is {current_time}.
Output must only be a valid JSON in the following format with no extra characters:
            - message: Message for the agent.
            - needs_deep_analysis: Boolean indicating need for deeper scheduling help if the user asks to schedule a task or gives a todo list. Once you have all the details the next time you call set this to False.
            - scheduling_context: Additional metadata with user input.
            - response_for_user: Response to the user for user input with all information (if any) formatted in a pretty way if needs_deep_analysis is False, else empty.
"""
        graph_agent = create_react_agent(
            llm,
            tools=[create_event, get_events, update_event, delete_event, edit_todo],
            state_modifier=prompt
        )
        result = graph_agent.invoke(state)
        print("Final state:", result['messages'][-1].content)
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content, name="calendar")
        state["messages"].extend(result["messages"])
        return state

    except Exception as e:
        print(f"Error in calendar_agent: {e}")
        return state
    

def scheduling_agent(state: MessagesState) -> MessagesState:
    try:
      
        agent_message = state["messages"][-1].content
        date = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        
        prompt = f"""
        You are an intellient task scheduling that schedules user's tasks or events at reasonable times by analysing user's schedule for the day. You need to think how much time will each task take and what order should to schedule the tasks in.
        Remember that today's date and time {date}. Schedule events only after the current time without overlap with existing events.
        Your input: {agent_message}
        output date/time values in ISO 8601/RFC3339 format including the time zone information.
        Output all user's tasks with the scheduled start time and end time and all other information you received. Respond only in valid json format.
        """
    
        deepseek = ChatOllama(model='deepseek-r1:8b')
        print("----------------------------", deepseek)

        graph_agent = create_react_agent(model=deepseek, tools=[], state_modifier=prompt)
        result = graph_agent.invoke(state)
        print("Scheduling agent result:", result)  # Debugging
        result["messages"][-1] = HumanMessage(content=result["messages"][-1].content.split('</think>')[1], name="calendar")
        state["messages"].extend(result["messages"])
        
        return state
    
    except Exception as e:
        print(f"Error in scheduling_agent: {e}")
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