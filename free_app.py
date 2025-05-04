import os
import sys
import time
import queue
import tempfile
import numpy as np
import sounddevice as sd
import pygame
import wave
import requests
import json
import base64
from gtts import gTTS
from dotenv import load_dotenv

# Load environment variables from .env file (optional for future use)
load_dotenv()

# Initialize pygame for audio playback
pygame.mixer.init()

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
    """Convert recorded audio to text using free Vosk API."""
    try:
        # For demonstration, we'll use a simpler approach with Google's Web Speech API
        # In a real implementation, you would use a local Vosk model or another free API
        print("Transcribing audio... (In a real implementation, this would use Vosk or another free API)")
        
        # Simulate transcription for demo purposes
        # In a real implementation, you would process the audio_file with a free API
        with open(audio_file, "rb") as file:
            # Read the first few bytes to check if there's actual audio content
            sample = file.read(1024)
            if len(sample) < 100:  # Very small file likely means no speech
                os.remove(audio_file)
                return None
        
        # For demo purposes, let's assume we detected some Turkish phrases
        # In a real implementation, this would be the result from the API
        import random
        demo_phrases = [
            "Ayasofya hakkında bilgi verir misiniz?",
            "Topkapı Sarayı nerede?",
            "İstanbul'da ne yemeli?",
            "Boğaz turu yapmak istiyorum.",
            "Kapalı Çarşı'ya nasıl gidebilirim?"
        ]
        
        # In a real implementation, remove this random selection and use actual API
        transcription = random.choice(demo_phrases)
        
        os.remove(audio_file)  # Clean up temporary file
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        if os.path.exists(audio_file):
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
    """Generate a meaningful response using a free LLM API."""
    if not user_text:
        return fallback_responses["not_understood"]
    
    # Add user input to history
    conversation_history.append({"role": "user", "content": user_text})
    
    # Keep only the last 5 exchanges to avoid token overload
    if len(conversation_history) > 10:  # 5 user + 5 assistant messages
        conversation_history[:] = conversation_history[-10:]
    
    try:
        # For demonstration purposes, we'll use a rule-based approach
        # In a real implementation, you would use a free API like:
        # - Hugging Face Inference API (with free tier)
        # - OpenAI's older models with limited free access
        # - Self-hosted open-source models
        
        # Simple keyword-based responses for demo
        lower_text = user_text.lower()
        
        if "ayasofya" in lower_text:
            response_text = "Ayasofya, İstanbul'un en önemli tarihi yapılarından biridir. 537 yılında inşa edilmiş, önce kilise, sonra cami, müze ve şimdi tekrar cami olarak hizmet vermektedir."
        elif "topkapı" in lower_text:
            response_text = "Topkapı Sarayı, Sultanahmet'te bulunur. Osmanlı padişahlarının 400 yıl boyunca yaşadığı saray, muhteşem bahçeleri ve değerli koleksiyonlarıyla ünlüdür."
        elif "yemek" in lower_text or "yemeli" in lower_text:
            response_text = "İstanbul'da kebap, balık, baklava ve Türk kahvesi denemelisiniz. Karaköy'deki balık restoranları ve Sultanahmet'teki tarihi lokantalar özellikle ünlüdür."
        elif "boğaz" in lower_text:
            response_text = "Boğaz turu için Eminönü veya Kabataş'tan kalkan vapurları kullanabilirsiniz. Günbatımı saatinde yapılan turlar özellikle etkileyicidir."
        elif "kapalı çarşı" in lower_text:
            response_text = "Kapalı Çarşı, Beyazıt'tadır. Tramvayla Beyazıt durağında inip kısa bir yürüyüşle ulaşabilirsiniz. 4000'den fazla dükkanıyla dünyanın en büyük kapalı çarşılarından biridir."
        else:
            response_text = "İstanbul, Doğu ile Batı'nın buluştuğu eşsiz bir şehirdir. Ayasofya, Topkapı Sarayı, Kapalı Çarşı ve Boğaz turu hakkında sorular sorabilirsiniz."
        
        # Truncate to ensure TTS fits in ~15 seconds
        response_text = truncate_response(response_text, max_words=35)
        
        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text
    except Exception as e:
        print(f"Response generation error: {e}")
        return fallback_responses["default"]

def speak_text(text):
    """Convert text to speech using Google's free TTS and play it."""
    if not text:
        return
    try:
        # Generate audio using gTTS (Google Text-to-Speech)
        tts = gTTS(text=text, lang='tr', slow=False)
        
        # Save audio to temporary MP3 file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            tts.save(temp_mp3.name)
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
    print("Starting Istanbul Tour Guide - Free Version")
    print("This version uses free APIs and simulated responses for demonstration")
    print("In a production environment, you would integrate with actual free APIs")
    
    # Initial greeting
    speak_text(fallback_responses["greeting"])
    
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