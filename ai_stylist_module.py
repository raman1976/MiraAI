# ai_stylist_module.py (FINAL VERSION WITH LIVE VISION INTEGRATION)
from google import genai
from google.genai import types
import os
from wardrobe_db import get_wardrobe_summary

# --- Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = 'gemini-2.5-flash' 

class AIStylistModule:
    # FIX A: Accept the vision_processor instance
    def __init__(self, vision_processor): 
        print("Initializing AI Stylist Module...")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
            
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.system_prompt = self._get_system_prompt()
        self.vision_processor = vision_processor # Store the instance
        
        # --- FIX APPLIED: Initialize chat with system instruction ---
        self.chat = self.client.chats.create(
            model=MODEL_NAME, 
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt 
            )
        ) 
        print("AI Stylist Module Ready.")
        
    # ai_stylist_module.py (Updated _get_system_prompt function)

    def _get_system_prompt(self):
        """Defines the AI's persona, rules, conversation flow, and styling logic."""
        prompt = (
            "You are MiraAI, a highly perceptive, warm, and expert personal fashion stylist. "
            "Your responses must be natural, engaging, concise, and suitable for **voice output**. "
            "Your core task is to analyze, compliment, and suggest improvements to a user's outfit based on the detected items, colors, and style."
            "\n\n"
            "**CONVERSATION FLOW:**\n"
            "1. **Introduction:** Start the session by introducing yourself as the personal AI fashion stylist, MiraAI, and immediately ask the user: 'What fashion question do you have for me?'"
            "2. **Styling Response:** When providing feedback, first **positively acknowledge** the user's current outfit or question. Then, provide constructive feedback on the fit or style. If the wardrobe summary contains only generic items like 'person' or 'TV', politely ask the user to show a specific item of clothing."
            "3. **Suggestion Engine:** Base outfit feedback on the provided wardrobe summary and apply complex color theory (e.g., complementary, analogous, monochromatic schemes) and contrast rules (e.g., black pants and red top provide strong contrast; beige offers neutrality)."
            "4. **Recommendations:** Always offer 1-3 specific, actionable suggestions for improving or accessorizing the outfit (e.g., 'Try a white top instead,' 'Add a silver necklace,' or 'Switch to a pair of black flats')."
            "\n\n"
            "**STYLING AND COLOR RULES (Crucial):**\n"
            "A. **Color Harmony:** Mention how colors contrast or complement (e.g., 'The red brightens the black for a bold statement, but if you want to soften the look, a beige top would create an excellent earthy tone.')."
            "B. **Wardrobe Constraints:** Only use colors and items that are realistically found in a wardrobe (e.g., don't suggest items that are only detected as 'person' or 'TV')."
            "C. **Classification:** Detected items fall into categories: tops, shirts, pants, skirts, dresses, jackets, shoes, accessories."
            "\n\n"
            "NEVER mention the word 'prompt' or 'virtual wardrobe summary' or 'YOLO'."
        )
        return prompt

    def generate_outfit_suggestion(self, user_command):
        """
        Generates an outfit suggestion by combining the user's query,
        live vision data, and the current wardrobe state.
        """
        # FIX B: Get live vision data and combine it with the wardrobe summary
        wardrobe_summary = get_wardrobe_summary()
        
        # Get live outfit data from the processor instance
        live_outfit_detections = self.vision_processor.get_live_detections()
        
        # Craft the full message to the model
        full_command = (
            f"**USER COMMAND:** '{user_command}'.\n"
            f"**CURRENT LIVE OUTFIT (Visible in camera):** {live_outfit_detections}\n"
            f"**WARDROBE DATABASE (FOR RECOMMENDATIONS):**\n{wardrobe_summary}"
        )

        try:
            # Send the user's message. The persona is handled in __init__.
            response = self.chat.send_message(full_command)
            return response.text
        
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "I apologize, but I'm having trouble connecting to my styling brain right now. Please try again in a moment."