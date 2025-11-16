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
        # Initialize Vision Processor and pass the voice speaker as a callback
        st.session_state.vision_processor = VisionProcessor(
            item_save_callback=st.session_state.voice_module.speak_response
        )
        print("Vision Processor Ready.")

    # 3. Initialize AI Stylist Module LAST
    if 'ai_stylist' not in st.session_state:
        print("Initializing AI Stylist Module...")
        # FIX FOR STEP 1: Pass the vision processor instance to the AI Stylist
        st.session_state.ai_stylist = AIStylistModule(
            vision_processor=st.session_state.vision_processor 
        )
        print("AI Stylist Module Ready.")


    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'command_trigger' not in st.session_state:
        st.session_state.command_trigger = False

    # Send initial welcome message only on first run
    if len(st.session_state.chat_history) == 0:
        welcome_text = "Hello! I'm MiraAI, your personal AI fashion stylist. What fashion question do you have for me?"
        # Speak the response using the voice module
        st.session_state.voice_module.speak_response(welcome_text)
        st.session_state.chat_history.append({"role": "mira", "content": welcome_text})


# --- 2. VIDEO PROCESSOR CLASS (Fixed __init__ method) ---
class MiraAITransformer(VideoProcessorBase):
    """Handles real-time video processing and returns the annotated frame."""

    # The __init__ is kept empty for safety, the processor is accessed in recv
    def __init__(self):
        pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Safely access the processor from st.session_state inside recv
        processor = st.session_state.vision_processor
        
        img = frame.to_ndarray(format="bgr24")
        
        # Use the processor instance to process the frame
        annotated_img = processor.process_frame(img)
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- 3. AI & CHAT LOGIC (Running in Background Thread) ---
def process_user_command(user_command):
    """Handles user input, calls Gemini, and updates history (in background thread)."""

    # 1. Prepare messages for logging (DO NOT write to st.session_state yet)
    user_message = {"role": "user", "content": user_command}
    
    # Give immediate audible feedback (Safe, as voice module is initialized)
    st.session_state.voice_module.speak_response("One moment, let me check your style options...")

    try:
        # Generate the response using the AI stylist module
        response_text = st.session_state.ai_stylist.generate_outfit_suggestion(user_command)
    except Exception as e:
        response_text = f"My styling brain failed: An error occurred during AI processing. Please check the console."
        print(f"AI Stylist Error: {e}")

    mira_message = {"role": "mira", "content": response_text}

    # 2. FINAL SAFE UPDATE: Update all state variables ONCE under the global lock
    with SESSION_LOCK:
        # Append both messages at once
        st.session_state.chat_history.append(user_message)
        st.session_state.chat_history.append(mira_message)
        
        # Speak the final response (this uses the voice_module instance from session state)
        st.session_state.voice_module.speak_response(response_text)
        
        # Set trigger flag to tell the main thread to rerun and update the UI
        st.session_state.command_trigger = True


# --- 4. MAIN UI FUNCTION ---
def main():
    # --- Check for Rerun Trigger (st.rerun() confirmed) ---
    if st.session_state.command_trigger:
        st.session_state.command_trigger = False
        st.rerun() # Correctly uses st.rerun()

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
            # The thread will set 'command_trigger' to force the UI update.

# --- Execute Main Function ---
if __name__ == '__main__':
    # Initialize all session state variables before the main UI runs
    initialize_session_state()
    main()