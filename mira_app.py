import streamlit as st
import threading
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import av
import numpy as np

# --- Core Module Imports ---
from dotenv import load_dotenv
load_dotenv()

# IMPORTANT: Ensure these files exist in your project structure
from voice_module import VoiceModule
from ai_stylist_module import AIStylistModule
from vision_module import VisionProcessor
from wardrobe_db import get_wardrobe_summary

# --- GLOBAL LOCK FOR THREAD SAFETY ---
SESSION_LOCK = threading.Lock()

# --- 1. INITIALIZATION FUNCTION (Guaranteed to run once per session) ---
def initialize_session_state():
    """Initializes all necessary modules and state variables once."""
    
    # 1. Initialize Voice Module
    if 'voice_module' not in st.session_state:
        print("Initializing Voice Module...")
        st.session_state.voice_module = VoiceModule()
        print("Voice Module Ready.")

    # 2. Initialize Vision Processor
    if 'vision_processor' not in st.session_state:
        print("Initializing Vision Processor...")
        st.session_state.vision_processor = VisionProcessor(
            item_save_callback=st.session_state.voice_module.speak_response
        )
        print("Vision Processor Ready.")

    # 3. Initialize AI Stylist Module
    if 'ai_stylist' not in st.session_state:
        print("Initializing AI Stylist Module...")
        st.session_state.ai_stylist = AIStylistModule(
            vision_processor=st.session_state.vision_processor 
        )
        print("AI Stylist Module Ready.")


    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'command_trigger' not in st.session_state:
        st.session_state.command_trigger = False
    
    # Initialize the audio queue variable
    if 'mira_audio_to_play' not in st.session_state:
        st.session_state.mira_audio_to_play = None
    
    # FINAL CRITICAL FIX: Ensure the resource delay runs ONLY ONCE
    if 'initial_delay_done' not in st.session_state:
        st.session_state.initial_delay_done = True
        print("Waiting 3 seconds for system to stabilize after model load...")
        time.sleep(3) # Increased delay to 3s for certainty
        print("Startup delay complete.")

    # Send initial welcome message only on first run (after the delay check)
    if len(st.session_state.chat_history) == 0:
        welcome_text = "Hello! I'm MiraAI, your personal AI fashion stylist. What fashion question do you have for me?"
        
        # Queue the initial response for playback
        st.session_state.mira_audio_to_play = welcome_text
        st.session_state.chat_history.append({"role": "mira", "content": welcome_text})
        
    print("Application ready.")


# --- 2. VIDEO PROCESSOR CLASS (Fixed __init__ method) ---
class MiraAITransformer(VideoProcessorBase):
    """Handles real-time video processing and returns the annotated frame."""

    def __init__(self):
        pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Safely access the processor from st.session_state inside recv
        processor = st.session_state.vision_processor
        img = frame.to_ndarray(format="bgr24")
        annotated_img = processor.process_frame(img)
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- 3. AI & CHAT LOGIC (Running in Background Thread) ---
def process_user_command(user_command):
    """Handles user input, calls Gemini, and queues the result."""

    user_message = {"role": "user", "content": user_command}
    
    try:
        # Generate the response using the AI stylist module
        response_text = st.session_state.ai_stylist.generate_outfit_suggestion(user_command)
    except Exception as e:
        response_text = f"My styling brain failed: An error occurred during AI processing. Please check the console."
        print(f"AI Stylist Error: {e}")

    mira_message = {"role": "mira", "content": response_text}

    # FINAL SAFE UPDATE: Update all state variables ONCE under the global lock
    with SESSION_LOCK:
        st.session_state.chat_history.append(user_message)
        st.session_state.chat_history.append(mira_message)
        
        # Queue the audio text to be played non-blockingly
        st.session_state.mira_audio_to_play = response_text
        
        # Set trigger flag to tell the main thread to rerun and update the UI
        st.session_state.command_trigger = True


# --- 4. MAIN UI FUNCTION ---
def main():
    # --- Check for Rerun Trigger ---
    if st.session_state.command_trigger:
        st.session_state.command_trigger = False
        st.rerun()

    # Play queued audio in a separate thread to prevent blocking the UI
    if st.session_state.mira_audio_to_play:
        audio_text = st.session_state.mira_audio_to_play
        st.session_state.mira_audio_to_play = None # Clear the queue immediately
        
        # LAUNCH AUDIO PLAYBACK IN A SEPARATE THREAD (NON-BLOCKING)
        threading.Thread(
            target=st.session_state.voice_module.speak_response,
            args=(audio_text,)
        ).start()

    st.title("✨ MiraAI: Live Personal Stylist")
    st.markdown("Your AI assistant for real-time fashion advice.")

    # 4.1 Layout Columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Live Mirror & Detection")
        st.markdown("**(Hold clothing up to the camera)**")

        # Start the webcam stream with the VideoProcessor
        webrtc_streamer(
            key="mira_webcam",
            video_processor_factory=MiraAITransformer,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.header("Wardrobe & Status")

        # Display Wardrobe Summary
        wardrobe_summary = get_wardrobe_summary()
        st.text_area("Virtual Wardrobe Summary", wardrobe_summary, height=200)

        # Display Chat History
        st.subheader("Mira's Responses")

        # Acquire lock for reading chat history safely
        with SESSION_LOCK:
            for message in st.session_state.chat_history[-5:]: # Show last 5 messages
                if message["role"] == "mira":
                    st.info(message["content"])
                else:
                    st.text(f"You: {message['content']}")


    # 4.2 Command Input
    st.markdown("---")
    st.subheader("Ask Mira for Advice")

    user_input = st.text_input(
        label="Your Question:",
        placeholder="e.g., What should I wear to a party?",
        key="user_text_input"
    )

    if st.button("Send Command"):
        if user_input:
            # Launch the processor in a thread to prevent the UI from locking
            threading.Thread(target=process_user_command, args=(user_input,)).start()

# --- Execute Main Function ---
if __name__ == '__main__':
    initialize_session_state()
    main()