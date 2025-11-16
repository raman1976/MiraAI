# vision_module.py
import cv2
from ultralytics import YOLO
import numpy as np
import threading
import time
from wardrobe_db import add_item_to_wardrobe
import os

# --- Configuration ---
# Uses a pre-trained segmentation model for better boundary detection
YOLO_MODEL = 'yolov8n-seg.pt' 
CONFIDENCE_THRESHOLD = 0.5 
# Mapping of COCO classes to our simple wardrobe labels (use COCO objects as proxies)
CLASS_MAP = {
    24: 'bag', 
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    # Placeholder for general clothing items in a standard COCO model
    39: 'bottle', # Not clothing, but for detection example
    41: 'tie', # Often in COCO
    42: 'suitcase',
    # NOTE: For proper clothing detection, you'd fine-tune a model on fashion datasets.
    # For this skeleton, we rely on the generic COCO classes that resemble items.
}

class VisionModule:
    def __init__(self, item_save_callback=None):
        print("Initializing Vision Module...")
        # Check if the YOLO model exists locally, if not, it will download on first use
        self.model = YOLO(YOLO_MODEL) 
        self.item_save_callback = item_save_callback
        self.running = threading.Event()
        self.cap = cv2.VideoCapture(0) # 0 is typically the default webcam
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam. Check camera connection/permissions.")
        
        # Simple debounce mechanism for saving items
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
            annotated_frame = results[0].plot() # YOLO handles drawing the boxes and masks
            
            detected_items = self._process_yolo_results(results[0])
            
            # Display the frame
            cv2.imshow('MiraAI Smart Stylist', annotated_frame)

            # Check for save condition
            if detected_items and (time.time() - self.last_save_time) > self.save_debounce_period:
                for item in detected_items:
                    # In a real app, you'd add color/texture extraction here
                    print(f"NEW ITEM DETECTED & SAVED: {item['label']}")
                    add_item_to_wardrobe(item)
                    if self.item_save_callback:
                         # Notify the user via voice 
                        self.item_save_callback(f"I've saved the {item['label']} to your virtual wardrobe!")

                self.last_save_time = time.time() # Reset debounce timer
                
            # Check for user quit command (must be run for OpenCV window to refresh)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def _process_yolo_results(self, results):
        """Extracts and formats the clothing detection results."""
        processed_items = []
        
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            
            # Map the detected class ID to a simple label
            label = CLASS_MAP.get(class_id, self.model.names[class_id])

            if confidence >= CONFIDENCE_THRESHOLD: 
                item = {
                    'label': label,
                    'confidence': round(confidence, 2),
                    'bbox': box.xyxy[0].tolist(),
                    'class_id': class_id,
                    'color': 'unknown' # Placeholder
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