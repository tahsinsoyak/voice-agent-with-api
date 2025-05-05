# voice-agent-with-api

## Overview

`voice-agent-with-api` is a Python-based voice assistant designed to act as a virtual tour guide for Istanbul. It listens to user voice input, transcribes it, generates a relevant response using the Groq LLaMA-4 model, and speaks the response back using ElevenLabs text-to-speech.

## Features

*   **Voice Input:** Records audio from the microphone using `sounddevice`.
*   **Speech-to-Text:** Transcribes Turkish speech to text using Groq's Whisper API (`whisper-large-v3`).
*   **Conversational AI:** Generates context-aware responses using Groq's LLaMA-4 model (`meta-llama/llama-4-maverick-17b-128e-instruct`), maintaining a short conversation history.
*   **Text-to-Speech:** Converts the generated text response into natural-sounding Turkish speech using the ElevenLabs API.
*   **Audio Playback:** Plays the generated speech audio using `pygame`.
*   **Environment Variable Management:** Securely loads API keys using `python-dotenv`.
*   **Fallback Responses:** Provides default responses for greetings, misunderstandings, and errors.
*   **Response Truncation:** Limits response length for concise interactions.

## Technologies Used

*   **Python 3.x**
*   **Libraries:**
    *   `groq`: For interacting with the Groq API (Transcription & LLM).
    *   `elevenlabs`: For text-to-speech conversion.
    *   `sounddevice` & `numpy`: For audio recording.
    *   `pygame`: For audio playback.
    *   `python-dotenv`: For managing environment variables.
*   **APIs:**
    *   Groq Cloud API
    *   ElevenLabs API

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd voice-agent-with-api
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r c:\Users\tahsinsoyak\Desktop\project_github_clone\voice-agent-with-api\requirements.txt
    ```

4.  **Set up environment variables:**
    *   Create a file named `.env` in the project root directory (`c:\Users\tahsinsoyak\Desktop\project_github_clone\voice-agent-with-api`).
    *   Add your API keys to the `.env` file:
        ```dotenv
        GROQ_API_KEY=your_groq_api_key_here
        ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
        ```
    *   Obtain API keys from:
        *   Groq Cloud
        *   ElevenLabs

## Usage

Run the main application script:

```bash
python c:\Users\tahsinsoyak\Desktop\project_github_clone\voice-agent-with-api\app.py
```

The application will greet you and then enter listening mode. Speak your query regarding Istanbul within the 5-second recording window. The assistant will process your request and respond verbally.

To exit the application, say "çıkış", "kapat", or "programı kapat" during the listening phase.

## Configuration

Key parameters can be adjusted within `c:\Users\tahsinsoyak\Desktop\project_github_clone\voice-agent-with-api\app.py`:
*   `record_audio(duration=...)`: Change the recording duration (default is 5 seconds).
*   `transcribe_audio(model=...)`: Change the Whisper model used by Groq.
*   `generate_response(model=...)`: Change the LLaMA model used by Groq.
*   `generate_response(max_tokens=...)`: Adjust the maximum response length from the LLM.
*   `speak_text(voice_id=...)`: Change the ElevenLabs voice ID.
*   `speak_text(model_id=...)`: Change the ElevenLabs TTS model.