"""
Emos - Your Voice Healer
A pure Python implementation with Streamlit UI, Gemini AI, and ElevenLabs TTS.
"""

import streamlit as st
import speech_recognition as sr
import requests
import google.generativeai as genai
import os
import logging
import tempfile
import base64
from dotenv import load_dotenv

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emos.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- LOAD ENVIRONMENT ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice

# Validate API keys
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found in .env")
if not ELEVEN_API_KEY:
    logger.error("ELEVEN_API_KEY not found in .env")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# --- CORE FUNCTIONS ---

def get_healing_advice(user_input: str) -> str:
    """Call Gemini API to get healing advice with fallback models."""
    logger.info(f"Calling Gemini API with input: {user_input[:50]}...")
    
    # Fallback chain of models to try
    models_to_try = [
        'gemini-2.0-flash',
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        'gemini-1.0-pro',
    ]
    
    system_prompt = """You are Emos, an empathetic emotional healer. 
    Your role is to provide comforting, positive, and healing advice.
    Be warm, understanding, and supportive. Keep responses concise but meaningful.
    Start with acknowledging the user's feelings, then offer gentle guidance."""
    
    full_prompt = f"{system_prompt}\n\nUser's situation: {user_input}"
    
    last_error = None
    for model_name in models_to_try:
        try:
            logger.info(f"Trying model: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(full_prompt)
            
            logger.info(f"Success with model: {model_name}")
            logger.debug(f"Response text: {response.text[:100]}...")
            
            return response.text
            
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Model {model_name} failed: {error_str[:100]}")
            last_error = e
            
            # If it's a quota error, try next model
            if "429" in error_str or "quota" in error_str.lower():
                continue
            else:
                # For other errors, don't try more models
                break
    
    # All models failed
    logger.error(f"All Gemini models failed. Last error: {last_error}", exc_info=True)
    return f"I'm having trouble connecting. Your API quota may be exhausted. Please try again later or check https://ai.google.dev/gemini-api/docs/rate-limits"


def text_to_speech(text: str) -> bytes | None:
    """Call ElevenLabs API to convert text to speech."""
    logger.info(f"Calling ElevenLabs TTS for text: {text[:50]}...")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 200:
            logger.info(f"ElevenLabs TTS success: {len(response.content)} bytes received")
            return response.content
        else:
            logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"ElevenLabs request failed: {str(e)}", exc_info=True)
        return None


def autoplay_audio(audio_bytes: bytes):
    """Create an autoplay audio element in Streamlit."""
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# --- STREAMLIT UI ---
st.set_page_config(page_title="Emos Voice Healer", page_icon="🎙️", layout="centered")
st.title("🎙️ Emos - Your Voice Healer")
st.write("Click the button and speak your mind. Emos will listen and respond with healing words.")

# Check API keys
if not GEMINI_API_KEY or not ELEVEN_API_KEY:
    st.error("⚠️ API keys not configured. Please check your .env file.")
    st.stop()

# Initialize speech recognizer
r = sr.Recognizer()
r.energy_threshold = 300
r.dynamic_energy_threshold = True

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Voice input button
if st.button("🎤 Speak to Emos", type="primary", use_container_width=True):
    logger.info("Voice input button pressed")
    
    with st.spinner("🎧 Calibrating microphone..."):
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=1.5)
                logger.info("Microphone calibrated")
                
            st.info("🎤 Speak now... (listening for up to 20 seconds)")
            
            with sr.Microphone() as source:
                audio_data = r.listen(source, timeout=10, phrase_time_limit=20)
                logger.info("Audio captured successfully")
            
            with st.spinner("🔄 Processing your speech..."):
                user_text = r.recognize_google(audio_data, language="en-US")
                logger.info(f"Speech recognized: {user_text}")
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_text})
            with st.chat_message("user"):
                st.write(user_text)
            
            # Get healing response
            with st.spinner("💭 Emos is thinking..."):
                healing_response = get_healing_advice(user_text)
            
            # Add Emos response to chat
            st.session_state.messages.append({"role": "assistant", "content": healing_response})
            with st.chat_message("assistant"):
                st.write(healing_response)
            
            # Text-to-speech
            with st.spinner("🔊 Generating voice response..."):
                audio_bytes = text_to_speech(healing_response)
            
            if audio_bytes:
                st.success("🔈 Playing audio response...")
                autoplay_audio(audio_bytes)
                logger.info("Audio playback initiated")
            else:
                st.warning("Could not generate audio response. Check logs for details.")
                
        except sr.WaitTimeoutError:
            logger.warning("Speech recognition timeout - no speech detected")
            st.warning("⏱️ No speech detected. Please try again and speak clearly.")
        except sr.UnknownValueError:
            logger.warning("Speech recognition failed - could not understand audio")
            st.error("❓ Could not understand audio. Please speak clearly and try again.")
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            st.error(f"🌐 Speech recognition service error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            st.error(f"❌ An error occurred: {e}")

# Text input fallback
st.markdown("---")
st.caption("Or type your message:")
text_input = st.text_input("Type here...", key="text_input", label_visibility="collapsed")

if text_input:
    logger.info(f"Text input received: {text_input}")
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": text_input})
    
    # Get response
    with st.spinner("💭 Emos is thinking..."):
        healing_response = get_healing_advice(text_input)
    
    st.session_state.messages.append({"role": "assistant", "content": healing_response})
    
    # TTS
    with st.spinner("🔊 Generating voice..."):
        audio_bytes = text_to_speech(healing_response)
    
    if audio_bytes:
        autoplay_audio(audio_bytes)
    
    st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by Python, Gemini 2.0 Flash, ElevenLabs, and Streamlit.")
st.caption("📋 Check `emos.log` for detailed debugging information.")
