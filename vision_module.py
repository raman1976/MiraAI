# vision_module.py (MediaPipe Hand Tracker)
import cv2
import mediapipe as mp
import time
import threading
import random

# --- Helper Function for Placeholder Color (Still needed for the prompt) ---
def extract_color(frame, bbox):
    """Placeholder for actual dominant color extraction logic."""
    colors = ["red", "black", "white", "navy blue", "beige", "gray", "green"]
    return random.choice(colors)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class VisionProcessor:
    """Handles MediaPipe initialization and non-blocking frame processing."""
    
    def __init__(self, item_save_callback=None):
        print("Initializing Vision Processor (MediaPipe)...")
        self.item_save_callback = item_save_callback
        self.last_save_time = time.time()
        self.save_debounce_period = 5 # seconds
        
        # New: MediaPipe hands model instance
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.5
        )
        
        # New: Variable to store the latest live status (e.g., "HAND DETECTED")
        self.latest_live_status = "No activity." 
        
        print("Vision Processor Ready.")

    def process_frame(self, frame):
        """Processes a single BGR frame from the webcam using MediaPipe."""
        
        # Convert the BGR frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(frame_rgb)
        annotated_frame = frame
        current_status = "No activity."

        if results.multi_hand_landmarks:
            current_status = "Hand detected."
            
            # Draw hand landmarks onto the frame
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

        # Update the latest live status for the AI Stylist
        self.latest_live_status = current_status
            
        # NOTE: Since MediaPipe is lighter than YOLO, we can keep the frame rate higher.
        
        return annotated_frame

    # FIX C: Method required by AIStylistModule
    def get_live_detections(self):
        """
        Returns a string summary of the current live status for the AI stylist.
        """
        # We will use this status to infer the user's current engagement
        return self.latest_live_status

    # NOTE: _process_yolo_results is removed as YOLO is gone.
    # We will rely on the AI stylist to understand the 'No activity' vs 'Hand detected' status.