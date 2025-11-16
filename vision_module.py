# vision_module.py (FINAL CORRECTED VERSION)
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
from wardrobe_db import add_item_to_wardrobe
import os
import random # Ensure random is imported for extract_color (moved from inside the function)

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

def extract_color(frame, bbox):
    """Placeholder for actual dominant color extraction logic."""
    # In a real app, this would use the bbox coordinates to crop the frame
    # and then analyze the dominant HSV/RGB colors.
    colors = ["red", "black", "white", "navy blue", "beige", "gray", "green"]
    # Removed 'import random' from here as it's defined globally above
    return random.choice(colors)

class VisionModule:
    def __init__(self, item_save_callback=None):
        print("Initializing Vision Module...")
        self.model = YOLO(YOLO_MODEL) 
        self.item_save_callback = item_save_callback
        self.running = threading.Event()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Check camera connection/permissions.")
        
        self.last_save_time = time.time()
        self.save_debounce_period = 5 # seconds
        
        print("Vision Module Ready.")

    def run_detection(self):
        """Runs the main video loop for real-time detection."""
        self.running.set()
        
        while self.running.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Perform detection with YOLOv8
            results = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Draw bounding boxes and process results
            annotated_frame = results[0].plot()
            
            # --- FIX 1: Pass the 'frame' argument when calling the processing method ---
            detected_items = self._process_yolo_results(results[0], frame) 
            
            # Display the frame
            cv2.imshow('MiraAI Smart Stylist', annotated_frame)

            # Check for save condition
            if detected_items and (time.time() - self.last_save_time) > self.save_debounce_period:
                for item in detected_items:
                    print(f"NEW ITEM DETECTED & SAVED: {item['label']}")
                    add_item_to_wardrobe(item)
                    if self.item_save_callback:
                        self.item_save_callback(f"I've saved the {item['label']} to your virtual wardrobe!")

                self.last_save_time = time.time() 
                
            # Check for user quit command (must be run for OpenCV window to refresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    # --- FIX 2: Accept the 'frame' argument in the function signature ---
    def _process_yolo_results(self, results, frame): 
        """Extracts and formats the clothing detection results."""
        processed_items = []
        
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            
            # --- FIX 3: Removed redundant and incorrectly placed line here ---
            # 'color': extract_color(frame, box.xyxy[0].tolist()) 
            
            # Map the detected class ID to a simple label
            label = CLASS_MAP.get(class_id, self.model.names[class_id])

            if confidence >= CONFIDENCE_THRESHOLD: 
                item = {
                    'label': label,
                    'confidence': round(confidence, 2),
                    'bbox': box.xyxy[0].tolist(),
                    'class_id': class_id,
                    # Call extract_color with the successfully passed frame
                    'color': extract_color(frame, box.xyxy[0].tolist()) 
                }
                processed_items.append(item)
                
        return processed_items

    def stop(self):
        """Releases the camera and closes all OpenCV windows."""
        self.running.clear()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Vision Module Stopped.")