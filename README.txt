MiraAI Project 
Technica 2025

**Project Requirements:**

1. **Live Wardrobe Scanning:**
   - Access webcam video using OpenCV.
   - Detect clothing items (tops, bottoms, jackets, shoes, accessories) in real time using YOLO (Ultralytics YOLOv8 or similar).
   - Draw bounding boxes for detected items (optional).
   - Automatically save detected items into a virtual wardrobe.

2. **Voice-Based Interaction:**
   - Recognize user speech using Python voice recognition (like `speech_recognition`).
   - Understand natural language commands (e.g., “Give me a casual outfit”, “What should I wear to a party?”).
   - Respond using **ElevenLabs API** for natural, expressive audio.

3. **Outfit Recommendation Engine:**
   - Generate outfit suggestions based on:
     - Detected wardrobe items
     - Color harmony rules
     - Style compatibility (formal, casual, sporty)
     - Event type (work, date, party, gym)
     - Weather (optional: integrate a weather API)
   - Return text descriptions suitable for **voice output**.
   - Optionally suggest multiple outfit combinations.

4. **Gesture Controls (Optional):**
   - Detect hand gestures (e.g., thumbs-up to save an outfit) using OpenCV or MediaPipe.

5. **Project Structure:**
   - Modular Python code with separate modules for:
     - Video capture & clothing detection
     - Voice recognition & response
     - Outfit recommendation
     - Wardrobe database management
     - Gesture controls (optional)
   - Include clear comments and instructions for setup and running.

6. **Extras:**
   - Lightweight “Smart Mirror” mode where the webcam acts like a mirror with live outfit suggestions overlayed.

**Output:**
- Generate full Python code with all modules wired together.
- Include instructions for installing dependencies.
- Make code extensible for future features.


/MiraAI_Project
├── mira_core.py             # Main application orchestrator
├── vision_module.py         # OpenCV, YOLOv8 Clothing Detection, and Wardrobe Storage
├── voice_module.py          # Speech Recognition (Input) and ElevenLabs (Output)
├── ai_stylist_module.py     # Outfit Recommendation Logic (Gemini API)
└── wardrobe_db.py           # Simple JSON/File-based Wardrobe Management