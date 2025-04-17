# Agendly - Executive Function Support Assistant 📅

Agendly is an AI-powered calendar and task management assistant specifically designed for individuals with Executive Function difficulties. It combines calendar management, task scheduling, and voice interaction capabilities to provide comprehensive support for daily planning and organization.

## Features

- 🗣️ Voice and text-based interaction
- 📅 Google Calendar integration
- ✅ Smart task scheduling
- ⏲️ Built-in timer for task management
- 🔋 Energy level-aware task planning
- 📚 Context-aware responses using RAG (Retrieval-Augmented Generation)
- 🎯 Automatic task breakdown and scheduling
- 🔄 Real-time calendar updates
- 🗣️ Text-to-speech responses

## Prerequisites

- Python 3.12+
- Google Calendar API credentials
(Create from Google Cloud Console → Enable Calendar API → Create OAuth 2.0 Client ID → Download credentials.json)
- OpenAI API key
- Ollama (for local LLM support)
- ChromaDB (for document storage)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agendly
```

2. Create and activate a Conda environment:
```bash
conda create -n agendly python=3.12
conda activate agendly
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
MODEL_NAME=<your-openai-model>
OPENAI_API_KEY=<your-openai-api-key>
```

5. Set up Google Calendar Authentication:
- Place your `credentials.json` file in the root directory
- On first run, you'll be prompted to authenticate with Google Calendar

## Database Setup

1. Create a ChromaDB database for context-aware responses:
```bash
jupyter notebook create_database.ipynb
```
2. Run all cells to initialize the database with your PDF documents

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Connect your Google Calendar when prompted

3. The interface provides:
   - Energy level slider (1-10)
   - Voice recording capability
   - Task input and management
   - Calendar event viewing and scheduling
   - Timer functionality

## Project Structure

- `main.py`: Main Streamlit application
- `chatbot_with_todo.py`: Core chatbot and task management logic
- `event_handler.py`: Google Calendar integration and event management
- `create_database.ipynb`: ChromaDB setup for document storage
- `requirements.txt`: Project dependencies

## Features in Detail

### Task Scheduling
- Automatically schedules tasks based on:
  - Current energy level
  - Task complexity
  - Available time slots
  - Existing calendar events
- Includes 15-minute breaks between tasks
- Considers user's energy levels for optimal task placement

### Voice Interaction
- Records voice input using sounddevice
- Transcribes audio to text
- Provides text-to-speech responses using OpenAI's TTS

### Energy-Aware Planning
- Tracks user's energy levels (1-10)
- Adjusts task scheduling based on energy:
  - Low energy (1-3): Shorter, easier tasks
  - Medium energy (4-6): Moderate complexity
  - High energy (7-10): Challenging tasks

### Context-Aware Responses
- Uses ChromaDB for document storage
- Implements RAG for informed responses
- Maintains conversation history
- Switches between different LLM models based on needs

## Models Used

- OpenAI GPT for main interactions
- Gemma (local) for overwhelm support
- Deepseek for specialized planning
- All-MiniLM-L6-v2 for embeddings

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]