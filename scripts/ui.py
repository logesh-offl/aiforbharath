import streamlit as st
import subprocess
import speech_recognition as sr
import os
import re

st.set_page_config(page_title="Emos Voice Healer", page_icon="🎙️")
st.title("🎙️ Emos - Your Voice Healer")
st.write("Click the button and speak your mind. Emos will listen and respond with healing words.")

# Initialize recognizer with improved settings
r = sr.Recognizer()
r.energy_threshold = 300  # Lower threshold for quieter environments
r.dynamic_energy_threshold = True  # Automatically adjust for ambient noise

if st.button("🎤 Speak to Emos"):
    with st.spinner("Listening..."):
        try:
            # Use the default microphone
            with sr.Microphone() as source:
                # Adjust for ambient noise (longer duration for better calibration)
                st.info("Calibrating for background noise... please wait")
                r.adjust_for_ambient_noise(source, duration=1.5)
                st.info("🎤 Speak now... (listening for up to 15 seconds)")
                
                # Listen for input with longer timeout
                audio_data = r.listen(source, timeout=10, phrase_time_limit=20)
                
            st.success("Processing audio...")
            
            # Recognize speech using Google's free STT
            user_text = r.recognize_google(audio_data, language="en-US")
            st.write(f"**You said:** {user_text}")
            
            # Send to Rust Backend
            with st.spinner("Consulting the healer (and synthesizing voice)..."):
                 # Compile and run the Rust binary with the user's input
                result = subprocess.run(
                    ["cargo", "run", "--quiet", "--", user_text],
                    capture_output=True,
                    text=True,
                    cwd="." 
                )
                
                output = result.stdout
                
                # Check for errors in Rust execution
                if result.returncode != 0:
                     st.error(f"Healer Error: {result.stderr}")
                else:
                    # Extract Gemini Response for display
                    # Look for content between "Gemini Suggests:" and "Synthesizing voice..."
                    gemini_match = re.search(r"Gemini Suggests:\s*(.*)\s*Synthesizing voice...", output, re.DOTALL)
                    if gemini_match:
                        divine_advice = gemini_match.group(1).strip()
                        st.markdown(f"**Healer:** {divine_advice}")
                    else:
                        st.warning("Healer spoke, but text was hard to read from the console.")
                        
                    if "Audio received! Playing stream..." in output:
                        st.toast("Voice synthesis playing on server speakers...", icon="🔊")
                    
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Speech Recognition service; {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Powered by Rust, Gemini 2.0 Flash, ElevenLabs and Streamlit.")
