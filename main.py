import os
import json
import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timedelta, timezone
import pytz
import re
import chromadb
from langchain_ollama import ChatOllama
from event_handler import get_events
import tempfile
from openai import OpenAI
import queue
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
from PIL import Image
import io
from chatbot_with_todo import get_workflow, run_chatbot
from langchain_core.messages import AIMessage, HumanMessage
import pickle
import json
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from streamlit_extras.stylable_container import stylable_container
from event_handler import create_event, get_events, init_google_calendar, edit_todo, delete_event
TOKEN_FILE = "token.json"
CLIENT_SECRET_FILE = "credentials.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

class AudioRecorder:
    def __init__(self):
        self.fs = 44100  # Sample rate
        self.recording = False
        self.audio_queue = queue.Queue()
        
    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            callback=self.callback
        )
        self.stream.start()
    
    def stop_recording(self):
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        # Combine all audio data
        while not self.audio_queue.empty():
            self.audio_data.append(self.audio_queue.get())
        
        if not self.audio_data:
            return None
            
        audio_data = np.concatenate(self.audio_data, axis=0)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav.write(temp_file.name, self.fs, audio_data)
        return temp_file.name

def transcribe_audio(audio_file_path):
    client = OpenAI()
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text

def speak_text(text):
    client = OpenAI()
    speech_file_path = "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
    )
    response = response.read()
    st.audio(response, format="audio/wav", autoplay=True)

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

TODO_FILE = "todos.json"

def load_todos():
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, "r") as f:
            return json.load(f)
    return []

def save_todos(todos):
    with open(TODO_FILE, "w") as f:
        json.dump(todos, f, indent=2)

def schedule_todos(todos, events, energy_level):
    try:
        # Get today's date and time in PST
        pst = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pst)
        today_str = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Format existing events for context
        events_context = "Today's existing events:\n"
        for event in events:
            start = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00')).astimezone(pst)
            end = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00')).astimezone(pst)
            events_context += f"- {event['summary']}: {start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}\n"

        # Format todos for context
        todos_context = "Tasks to schedule:\n"
        for todo in todos:
            todos_context += f"- {todo}\n"

        # Add energy level context
        energy_context = f"\nCurrent energy level: {energy_level}/10"
        if energy_level <= 3:
            energy_context += " (Low energy - prefer shorter, easier tasks)"
        elif energy_level <= 6:
            energy_context += " (Medium energy - moderate task complexity ok)"
        else:
            energy_context += " (High energy - can handle challenging tasks)"

        prompt = f"""
You are an intelligent task scheduler that helps users with ADHD manage their time effectively.
Current date: {today_str}
Current time: {current_time}

{events_context}
{todos_context}
{energy_context}

Your task is to:
1. Analyze today's existing events and available time slots
2. Consider the user's current energy level when scheduling tasks
3. For each todo item:
   - Estimate reasonable duration (20-45 minutes for most tasks)
   - Find suitable time slot that doesn't conflict with existing events
   - Consider task complexity and user's energy level
   - Add 15-minute buffer time between tasks for breaks
4. Create calendar events for each task using the create_event tool

Rules:
- Only schedule for today ({today_str})
- Respect existing events - no overlaps
- Include 15-minute breaks between tasks
- Start scheduling after current time ({current_time})
- Use ISO 8601 format for dates/times with timezone
- Set location as empty string
- Set description as "Scheduled from todo list"
- No attendees needed
- End times must be after start times
"""

        # Create messages list for the agent
        messages = [
            HumanMessage(content=prompt)
        ]

        # Create React agent with calendar tools
        agent = create_react_agent(
            llm,
            tools=[create_event, get_events],
            state_modifier=prompt
        )

        # Invoke agent
        result = agent.invoke({"messages": messages})
        
        # Extract the response content
        if isinstance(result, dict) and 'messages' in result:
            response_content = result['messages'][-1].content
        else:
            response_content = str(result)

        # Create a formatted JSON response
        formatted_response = {
            "message": "Scheduling completed successfully",
            "needs_deep_analysis": False,
            "scheduling_context": {
                "date": today_str,
                "time": current_time,
                "energy_level": energy_level
            },
            "response_for_user": response_content
        }

        return json.dumps(formatted_response)

    except Exception as e:
        print(f"Error in schedule_todos: {e}")
        error_response = {
            "message": "Error scheduling todos",
            "needs_deep_analysis": False,
            "scheduling_context": {},
            "response_for_user": f"Failed to schedule todos: {str(e)}"
        }
        return json.dumps(error_response)

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = get_workflow() 
    if "config" not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": "1"}}
    if "state" not in st.session_state:
        st.session_state.state = st.session_state.graph.get_state(config=st.session_state.config)
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "creds" not in st.session_state:
        st.session_state.creds = None
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = None
    if "recording_icon" not in st.session_state:
        st.session_state.recording_icon = ":material/mic:"
    if "using_gemma" not in st.session_state:
        st.session_state.using_gemma = False
    if "todos" not in st.session_state:
        st.session_state.todos = load_todos()

def process_message(message, creds, energy_level=5) -> str:
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db_persistent")
    collection = client.get_collection("pdf_semantic_chunks")
    
    # Search for relevant context
    results = collection.query(
        query_texts=[message],
        n_results=3  # Get top 3 most relevant chunks
    )
    
    # Format the retrieved context
    context_chunks = []
    if results and results['documents']:
        context_chunks = results['documents'][0]  # Get documents from first query
    context_text = "\n\n".join(context_chunks)
    
    # Create energy level context
    energy_context = f"\nUser's current energy level is {energy_level}/10. "
    if energy_level <= 3:
        energy_context += "Please suggest smaller, more manageable tasks and be extra supportive."
    elif energy_level <= 6:
        energy_context += "Consider breaking down complex tasks into medium-sized chunks."
    else:
        energy_context += "User has good energy, can handle more challenging tasks."

    if st.session_state.state.values == {}:
        # Include both the retrieved context and energy level in the message
        enhanced_message = f"""User Query: {message}

Relevant Context from Knowledge Base:
{context_text}

Energy Context:
{energy_context}

Please consider both the above context and user's energy level while responding to the query."""
        
        st.session_state.state.values["messages"] = [HumanMessage(enhanced_message)]
    else:
        enhanced_message = f"""User Query: {message}

Relevant Context from Knowledge Base:
{context_text}

Energy Context:
{energy_context}

Please consider both the above context and user's energy level while responding to the query."""
        st.session_state.state.values["messages"].append(HumanMessage(enhanced_message))
    
    updated_state = run_chatbot(st.session_state.graph, st.session_state.state.values, creds)
    st.session_state.state.values['messages'].append(updated_state.values["messages"][-1])
    response = st.session_state.state.values["messages"][-1].content
    return response

def process_overwhelmed(message, creds) -> str:
    """
    Process a user message that contains the word "overwhelmed" by first fetching upcoming events
    and then starting a conversation with ChatOllama (gemma3:latest) using all previous messages.
    """
    # Initialize Google Calendar with credentials
    from event_handler import init_google_calendar
    init_google_calendar(creds)
    
    # Get the current UTC time and one day ahead
    now = datetime.now(timezone.utc)
    startDateTime = now.isoformat()
    endDateTime = (now + timedelta(days=1)).isoformat()
    
    # Retrieve upcoming events from the calendar
    try:
        events = get_events.invoke({
            "startDateTime": startDateTime,
            "endDateTime": endDateTime
        })
    except Exception as e:
        print(f"Error fetching events: {e}")
        events = []
    
    # Prepare context by appending event information and previous messages
    conversation_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages
    )
    context = (
        "The user mentioned feeling overwhelmed. the user suffers from adhd and has trouble starting with tasks which has intertia. You can break down the tasks into smaller tasks of 20-30 mins each. Here are the upcoming events from your calendar: \n"
        f"{json.dumps(events, indent=2)}\n\n"
        "Previous conversation history:\n" + conversation_history + "\n\n"
        f"User message: {message}\n"
        "Please provide a supportive response that takes into account the calendar events."
    )
    
    try:
        gemma = ChatOllama(model="gemma3:latest")
        response = gemma.invoke(context)
        return response.content
    except Exception as e:
        print(f"Error processing overwhelmed message: {e}")
        return "I'm sorry, I encountered an error while processing your request."
def process_with_gemma(message, creds) -> str:
    """
    Handle messages with Gemma, including context from past conversation, calendar events, and todo list.
    """
    # Initialize Google Calendar
    from event_handler import init_google_calendar
    init_google_calendar(creds)

    # Get upcoming events for the next 24 hours
    now = datetime.now(timezone.utc)
    startDateTime = now.isoformat()
    endDateTime = (now + timedelta(days=1)).isoformat()

    try:
        events = get_events.invoke({
            "startDateTime": startDateTime,
            "endDateTime": endDateTime
        })
    except Exception as e:
        print(f"Error fetching events: {e}")
        events = []

    # Get todo list from session state
    todos = st.session_state.todos if "todos" in st.session_state else []
    
    # Prepare conversation history
    conversation_history = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages
    )

    # Format todo list for context
    todo_list_context = "\n".join(f"- {todo}" for todo in todos)

    # Compose prompt for Gemma
    context = (
        "You're a helpful calendar and task management assistant. "
        "The user has ADHD and needs help breaking down tasks and managing their schedule.\n\n"
        f"Current Todo List:\n{todo_list_context}\n\n"
        f"Upcoming calendar events:\n{json.dumps(events, indent=2)}\n\n"
        "Recent conversation history:\n"
        f"{conversation_history}\n\n"
        f"User message: {message}\n"
        "Please respond supportively and helpfully, taking into account both the calendar events "
        "and todo list. If appropriate, suggest ways to break down tasks into smaller, manageable chunks."
    )

    # Call Gemma
    try:
        gemma = ChatOllama(model="gemma3:latest")
        response = gemma.invoke(context)
        return response.content
    except Exception as e:
        print(f"Error processing message with Gemma: {e}")
        return "I'm sorry, I encountered an error while processing your request."
# def process_with_gemma(message, creds) -> str:
#     """
#     Continue the conversation with Gemma by appending the conversation history.
#     """
#     conversation_history = "\n".join(
#         f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages
#     )
#     context = (
#         "Continuing conversation with Gemma.\n"
#         "Previous conversation history:\n" + conversation_history + "\n\n"
#         f"User message: {message}\n"
#         "Please provide a response."
#     )
#     try:
#         gemma = ChatOllama(model="gemma3:latest")
#         response = gemma.invoke(context)
#         return response.content
#     except Exception as e:
#         print(f"Error processing message with Gemma: {e}")
#         return "I'm sorry, I encountered an error while processing your request."

def authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CLIENT_SECRET_FILE, SCOPES
                )
                creds = flow.run_local_server(port=8080)
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                return
    
    st.session_state.authenticated = True
    st.session_state.creds = creds


# Configure page settings with dark theme support
st.set_page_config(
    page_title="Calendar Assistant",
    page_icon="📅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS for styling the page and the recording button
st.markdown("""
    <style>
    .stApp header {
        background-color: transparent !important;
    }
    [data-testid="stHeader"] {
        color: #ffffff !important;
    }
    h1 {
        color: #ffffff !important;
        font-weight: 500;
    }
    .record-button {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        background-color: #ff4b4b;
        color: white;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .record-button.recording {
        background-color: #ff3333;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

def main():
    try:
        initialize_session_state()
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return
    
    st.title("Executive Fuinction Assistant")
    if not st.session_state.authenticated:
        st.markdown("""
            <div class="welcome-container">
                <h2>👋 Welcome to Calendar Assistant!</h2>
                <p style="color: #666666; margin: 1rem 0;">Connect your Google Calendar to get started</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔗 Connect Calendar", use_container_width=True):
                authenticate()
    if st.session_state.authenticated:
        st.markdown("### 💡 Assistant Settings")
        feeling_toggle = st.toggle("Feeling Overwhelmed")
        
        # Add energy level slider in the sidebar
        with st.sidebar:
            st.markdown("### 🔋 Energy Level")
            energy_level = st.slider(
                "Current Energy Level",
                min_value=1,
                max_value=10,
                value=5,
                help="1 = Very low energy, 10 = Very high energy"
            )
        
        if feeling_toggle:
            st.session_state.using_gemma = True
        else:
            st.session_state.using_gemma = False
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("image"):
                st.image(message["image"])
            st.markdown(message["content"])
    
    # Chat input with audio recording support
    if st.session_state.authenticated:
        with stylable_container(
            key="bottom_content",
            css_styles="""
                {
                    position: fixed;
                    bottom: 0;
                    opacity: 1;
                    padding: 30px;
                    background-color: #0e1117;
                }
            """,
        ):
            with st.sidebar:
                # Today's Events Section
                st.markdown("### 📅 Today's Events")
                if st.session_state.creds:
                    from event_handler import init_google_calendar
                    init_google_calendar(st.session_state.creds)
                    
                    today_start = datetime.now(timezone.utc).replace(
                        hour=0, minute=0, second=0, microsecond=0
                    )
                    today_end = today_start + timedelta(days=1)
                    
                    try:
                        events_response = get_events.invoke({
                            "startDateTime": today_start.isoformat(),
                            "endDateTime": today_end.isoformat()
                        })
                        
                        # Directly use the events list if it's already in the expected format
                        if isinstance(events_response, list):
                            events = events_response
                        elif isinstance(events_response, dict):
                            # Handle the dictionary case as before
                            if 'data' in events_response and 'items' in events_response['data']:
                                events = events_response['data']['items']
                            elif 'items' in events_response:
                                events = events_response['items']
                            else:
                                events = []
                        else:
                            events = []
                        
                        if events:
                            for event in events:
                                col1, col2 = st.columns([8, 2])
                                with col1:
                                    # Extract start and end times from the event
                                    start = event.get('start', {}).get('dateTime')
                                    end = event.get('end', {}).get('dateTime')
                                    
                                    if start and end:
                                        start_time = datetime.fromisoformat(start).strftime("%I:%M %p")
                                        end_time = datetime.fromisoformat(end).strftime("%I:%M %p")
                                        
                                        st.markdown(
                                            f"**{event.get('summary', 'Untitled Event')}**\n"
                                            f"🕒 {start_time} - {end_time}"
                                        )
                                with col2:
                                    if st.button("❌", key=f"delete_event_{event.get('eventId')}"):
                                        try:
                                            delete_event.invoke({"eventId": event.get('eventId')})
                                            st.success("Event deleted!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Failed to delete event: {str(e)}")
                                st.divider()
                        else:
                            st.info("No events scheduled for today")
                    except Exception as e:
                        st.error(f"Could not fetch events: {str(e)}")
                        print(f"Events fetch error details: {e}")  # For debugging
                
                # Todo List Section
                st.markdown("### 📝 Your To-Do List")
                todos = st.session_state.todos
                updated_todos = []

                for i, todo in enumerate(todos):
                    col1, col2 = st.columns([8, 2])
                    with col1:
                        edited = st.text_input(f"todo_{i}", todo, key=f"todo_input_{i}")
                    with col2:
                        if st.button("❌", key=f"delete_{i}"):
                            # Skip this todo by not adding it to updated_todos
                            continue
                        else:
                            # Only add todos that weren't deleted
                            updated_todos.append(edited)

                # Update session state and save to file
                st.session_state.todos = updated_todos
                save_todos(updated_todos)

                # Create a form for adding new todos
                with st.form(key="add_todo_form"):
                    new_todo = st.text_input("Add new task", key="new_todo_input")
                    submit_button = st.form_submit_button("Add Task")
                    if submit_button and new_todo:
                        st.session_state.todos.append(new_todo)
                        save_todos(st.session_state.todos)
                        st.rerun()

                # Schedule Todo button
                st.markdown("### 📋 Task Scheduling")
                if st.button("🗓️ Schedule Todos", use_container_width=True):
                    if not st.session_state.todos:
                        st.warning("No todos to schedule!")
                    else:
                        with st.spinner("Scheduling todos..."):
                            try:
                                # Get current events
                                today_start = datetime.now(timezone.utc).replace(
                                    hour=0, minute=0, second=0, microsecond=0
                                )
                                today_end = today_start + timedelta(days=1)
                                
                                events = get_events.invoke({
                                    "startDateTime": today_start.isoformat(),
                                    "endDateTime": today_end.isoformat()
                                })
                                
                                # Get energy level (assuming it's defined somewhere)
                                energy_level = 5  # Default value if not set elsewhere
                                
                                # Schedule todos
                                result = schedule_todos(
                                    st.session_state.todos, 
                                    events,
                                    energy_level
                                )
                                
                                schedule_result = json.loads(result)
                                st.success("Tasks scheduled successfully!")
                                st.markdown(schedule_result["response_for_user"])
                                
                                # Clear todos after successful scheduling
                                st.session_state.todos = []
                                save_todos([])
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Failed to schedule todos: {str(e)}")
                                print(f"Scheduling error details: {e}")
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button(icon=st.session_state.recording_icon, label="", key="mic", type='primary'):
                    if not st.session_state.recording:
                        # Start recording
                        st.session_state.recording = True
                        st.session_state.recording_icon = ":material/stop_circle:"
                        st.session_state.audio_recorder = AudioRecorder()
                        st.session_state.audio_recorder.start_recording()
                        st.rerun()
                    else:
                        # Stop recording
                        audio_file = st.session_state.audio_recorder.stop_recording()
                        st.session_state.recording = False
                        st.session_state.recording_icon = ":material/mic:"
                        if audio_file:
                            with st.spinner("Transcribing..."):
                                transcription = transcribe_audio(audio_file)
                                print(transcription)
                                st.session_state.transcribed_text = transcription
                                os.unlink(audio_file)  # Clean up temporary file
                            st.rerun()
            with col2:
                user_input = st.chat_input("Ask about your calendar...")       
            
        if user_input or st.session_state.transcribed_text:
            if user_input:
                text = user_input
                audio = False
            if st.session_state.transcribed_text:
                text = st.session_state.transcribed_text
                st.session_state.transcribed_text = None
                audio = True
            
            with st.chat_message("user"):
                st.markdown(text)
                st.session_state.messages.append({
                    "role": "user",
                    "content": text
                })
            
            # If already using Gemma for conversation, continue with Gemma.
            if st.session_state.using_gemma:
                with st.chat_message("assistant"):
                    response = process_with_gemma(text, st.session_state.creds)
                    if audio:
                        speak_text(response)
                    st.markdown(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
            else:
                # Check if the user message contains "overwhelmed"
                if re.search(r"\boverwhelmed\b", text, re.IGNORECASE):
                    with st.chat_message("assistant"):
                        response = process_overwhelmed(text, st.session_state.creds)
                        # Set flag to continue conversation with Gemma
                        st.session_state.using_gemma = True
                        if audio:
                            speak_text(response)
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                else:
                    # Process normally using the default chatbot workflow
                    with st.chat_message("assistant"):
                        response = json.loads(process_message(
                            text, 
                            st.session_state.creds,
                            energy_level=energy_level  # Pass the energy level
                        ))['response_for_user']
                        if audio:
                            speak_text(response)
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

if __name__ == "__main__":
    main()
