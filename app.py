import os
import sys
import time
import queue
import tempfile
import numpy as np
import sounddevice as sd
import pygame
import wave
from groq import Groq
from elevenlabs import ElevenLabs, VoiceSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize pygame for audio playback
pygame.mixer.init()

# Initialize Groq client
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    sys.exit(1)

# Initialize ElevenLabs client
try:
    elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
except Exception as e:
    print(f"Error initializing ElevenLabs client: {e}")
    sys.exit(1)

# Fallback responses
fallback_responses = {
    "greeting": "Merhaba! İstanbul tur rehberinizim. Size nasıl yardımcı olabilirim?",
    "not_understood": "Üzgünüm, sizi anlayamadım. Lütfen tekrar eder misiniz?",
    "default": "İstanbul'da gezilecek yerler çok! Önerdiğim bir yeri görmek ister misiniz?"
}

# Conversation history
conversation_history = []

# Audio recording queue
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

def record_audio(duration=5, samplerate=16000):
    """Record audio for specified duration and save to a temporary WAV file."""
    audio_data = []
    with sd.RawInputStream(samplerate=samplerate, blocksize=4000, dtype='int16',
                           channels=1, callback=audio_callback):
        start_time = time.time()
        while time.time() - start_time < duration:
            if not audio_queue.empty():
                audio_data.append(audio_queue.get())
    audio_np = np.frombuffer(b''.join(audio_data), dtype=np.int16)
    
    # Save to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        with wave.open(temp_wav.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(samplerate)
            wav_file.writeframes(audio_np.tobytes())
        return temp_wav.name

def transcribe_audio(audio_file):
    """Convert recorded audio to text using Groq's Whisper API."""
    try:
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_file, file.read()),
                model="whisper-large-v3",
                language="tr",
                response_format="text",
                temperature=0.0
            )
        os.remove(audio_file)  # Clean up temporary file
        return transcription.strip() if transcription else None
    except Exception as e:
        print(f"Transcription error: {e}")
        os.remove(audio_file)  # Clean up even if error occurs
        return None

def truncate_response(text, max_words=35):
    """Truncate response to max_words, stopping at the last complete sentence or word."""
    words = text.split()
    if len(words) <= max_words:
        return text
    
    # Take up to max_words and find the last sentence boundary
    truncated = words[:max_words]
    truncated_text = " ".join(truncated)
    
    # Try to end at a sentence boundary (e.g., period)
    last_period = truncated_text.rfind(".")
    if last_period != -1 and last_period > len(truncated_text) // 2:
        return truncated_text[:last_period + 1]
    
    # If no period, end cleanly with a prompt for more
    return truncated_text + "... Daha fazla bilgi için sorabilirsiniz."

def generate_response(user_text):
    """Generate a meaningful response using Groq's LLaMA-4 model with conversation history."""
    if not user_text:
        return fallback_responses["not_understood"]
    
    # Add user input to history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Keep only the last 5 exchanges to avoid token overload
    if len(conversation_history) > 10:  # 5 user + 5 assistant messages
        conversation_history[:] = conversation_history[-10:]
    
    # Prepare messages with system prompt and history
    messages = [
        {
            "role": "system",
            "content": (
                "Sen İstanbul'da profesyonel bir tur rehberisin. Turistlere doğru, net ve 30 kelimeyi geçmeyen yanıtlar ver. "
                "Önceki konuşma bağlamını dikkate al ve yanıtlarını buna göre şekillendir."
            )
        }
    ] + conversation_history
    
    try:
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=50,  # Increased for complete but concise responses
            temperature=0.7,
            top_p=0.9
        )
        response_text = response.choices[0].message.content.strip()
        
        # Truncate to ensure TTS fits in ~15 seconds
        response_text = truncate_response(response_text, max_words=35)
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        if len(response_text.split()) < 3:
            return fallback_responses["default"]
        return response_text
    except Exception as e:
        print(f"Response generation error: {e}")
        return fallback_responses["default"]

def speak_text(text):
    """Convert text to speech using ElevenLabs and play it with timeout."""
    if not text:
        return
    try:
        # Generate audio using ElevenLabs
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id="IuRRIAcbQK5AQk1XevPj",  # Voice ID for Doga
            optimize_streaming_latency=0,
            output_format="mp3_44100_128",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )
        
        # Save audio to temporary MP3 file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            for chunk in audio_stream:
                if chunk:
                    temp_mp3.write(chunk)
            audio_file = temp_mp3.name

        # Play audio with pygame and timeout
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        start_time = time.time()
        while pygame.mixer.music.get_busy() and (time.time() - start_time) < 20:  # 20-second timeout
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()  # Stop if timeout reached
        pygame.mixer.music.unload()
        os.remove(audio_file)
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        if 'audio_file' in locals():
            os.remove(audio_file)

def main():
    # Initial greeting
    speak_text("Merhaba! İstanbul tur rehberinizim. Size nasıl yardımcı olabilirim?")
    
    while True:
        print("\nListening mode: Please speak for 5 seconds...")
        audio_file = record_audio(duration=5)
        user_text = transcribe_audio(audio_file)
        
        if user_text:
            print(f"User: {user_text}")
            if user_text.lower().strip() in ["çıkış", "kapat", "programı kapat"]:
                speak_text("Görüşmek üzere! İyi günler.")
                break
            response_text = generate_response(user_text)
            print(f"\nGuide: {response_text}")
            speak_text(response_text)
        else:
            print("No speech detected or too short.")
            
    print("Program terminated.")

if __name__ == "__main__":
    main()