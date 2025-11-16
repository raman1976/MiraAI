# vision_module.py (STREAMLIT READY - With Live Detection Tracking)
from ultralytics import YOLO
import numpy as np
import threading
import time
import os
import random
from wardrobe_db import add_item_to_wardrobe
import cv2 # Keep import for image processing utilities

# --- Configuration ---
YOLO_MODEL = 'yolov8n-seg.pt' 
CONFIDENCE_THRESHOLD = 0.5 
CLASS_MAP = {
    0: 'person', 
    24: 'bag', 
    27: 'backpack',
    31: 'handbag',
    41: 'tie', 
    42: 'suitcase',
    # Note: Use a dedicated fashion model for better results than the COCO classes below.
}

# --- Helper Function for Placeholder Color ---
def extract_color(frame, bbox):
    """Placeholder for actual dominant color extraction logic."""
    colors = ["red", "black", "white", "navy blue", "beige", "gray", "green"]
    return random.choice(colors)

class VisionProcessor:
    """Handles YOLO initialization and frame processing for Streamlit."""
    
    def __init__(self, item_save_callback=None):
        print("Initializing Vision Processor...")
        self.model = YOLO(YOLO_MODEL) 
        self.item_save_callback = item_save_callback
        self.last_save_time = time.time()
        self.save_debounce_period = 5 # seconds
        
        # FIX A: Variable to store the latest detections for the AI Stylist
        self.latest_live_outfit = [] 
        
        print("Vision Processor Ready.")

    def process_frame(self, frame):
        """Processes a single BGR frame from the webcam."""
        
        # Streamlit-webrtc often sends BGR or RGB; ensure BGR for standard processing
        BGR_frame = frame 

        results = self.model.predict(BGR_frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        detected_items = []
        
        if results and results[0] is not None:
            detected_items = self._process_yolo_results(results[0], BGR_frame)
            
            # FIX B: Update the latest live outfit detection
            # We filter out 'person' as it's not useful for styling feedback.
            self.latest_live_outfit = [
                f"{item['label']} ({item['color']})" 
                for item in detected_items 
                if item['label'] != 'person'
            ]
            
            # Draw bounding boxes and annotations onto the frame for display
            annotated_frame = results[0].plot()
        else:
            # If nothing is detected, clear the live outfit list
            self.latest_live_outfit = [] 
            annotated_frame = BGR_frame

        # Check for save condition (now runs per frame)
        if detected_items and (time.time() - self.last_save_time) > self.save_debounce_period:
            for item in detected_items:
                if item['label'] != 'person': # Only save clothing/accessories
                    print(f"NEW ITEM DETECTED & SAVED: {item['label']} ({item['color']})")
                    add_item_to_wardrobe(item)
                    if self.item_save_callback:
                        self.item_save_callback(f"I've saved the {item['label']} to your virtual wardrobe!")
            self.last_save_time = time.time()
            
        # Return the processed frame (in BGR format)
        return annotated_frame

    def _process_yolo_results(self, results, frame): 
        """Extracts and formats the clothing detection results."""
        processed_items = []
        
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            label = CLASS_MAP.get(class_id, self.model.names[class_id])

            if confidence >= CONFIDENCE_THRESHOLD: 
                item = {
                    'label': label,
                    'confidence': round(confidence, 2),
                    'bbox': box.xyxy[0].tolist(),
                    'class_id': class_id,
                    'color': extract_color(frame, box.xyxy[0].tolist()) 
                }
                processed_items.append(item)
                
        return processed_items
    
    # FIX C: Add the method required by AIStylistModule
    def get_live_detections(self):
        """
        Returns a string summary of the currently detected items 
        that can be used as context for the AI stylist.
        """
        if self.latest_live_outfit:
            # Join the list into a comma-separated string
            return ", ".join(self.latest_live_outfit) 
        else:
            return "Nothing in focus."