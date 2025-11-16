# ai_stylist_module.py (FINAL VERSION WITH MEDIAPIPE CONTEXT)
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
            "Your core task is to analyze, compliment, and suggest improvements to a user's outfit based on the visual status and their question."
            "\n\n"
            "**CONVERSATION FLOW:**\n"
            "1. **Introduction:** Start the session by introducing yourself as the personal AI fashion stylist, MiraAI, and immediately ask the user: 'What fashion question do you have for me?'"
            "2. **Styling Response:** When providing feedback, first **positively acknowledge** the user's current outfit or question. Then, provide constructive feedback on the fit or style. If the visual status is 'No activity' or the wardrobe summary is generic, politely ask the user to show a specific item of clothing or make a gesture (like raising a hand)."
            "3. **Suggestion Engine:** Base outfit feedback on the provided wardrobe summary and apply complex color theory (e.g., complementary, analogous, monochromatic schemes) and contrast rules (e.g., black pants and red top provide strong contrast; beige offers neutrality)."
            "4. **Recommendations:** Always offer 1-3 specific, actionable suggestions for improving or accessorizing the outfit (e.g., 'Try a white top instead,' 'Add a silver necklace,' or 'Switch to a pair of black flats')."
            "\n\n"
            "**STYLING AND COLOR RULES (Crucial):**\n"
            "A. **Color Harmony:** Mention how colors contrast or complement (e.g., 'The red brightens the black for a bold statement, but if you want to soften the look, a beige top would create an excellent earthy tone.')."
            "B. **Wardrobe Constraints:** Only use colors and items that are realistically found in a wardrobe (e.g., don't suggest items that are only detected as 'person' or 'TV')."
            "C. **Visual Context:** The vision status will tell you if the user is engaged (e.g., 'Hand detected.'). Use this to make the response more conversational."
            "\n\n"
            "NEVER mention the word 'prompt', 'YOLO', 'MediaPipe', or 'virtual wardrobe summary'."
        )
        return prompt

    def generate_outfit_suggestion(self, user_command):
        """
        Generates an outfit suggestion by combining the user's query,
        live vision status, and the current wardrobe state.
        """
        # Get the full wardrobe database summary
        wardrobe_summary = get_wardrobe_summary()
        
        # FIX B: Get live vision status from the processor instance (e.g., "Hand detected.")
        live_vision_status = self.vision_processor.get_live_detections()
        
        # Craft the full message to the model
        full_command = (
            f"**USER COMMAND:** '{user_command}'.\n"
            f"**CURRENT LIVE VISION STATUS (Context):** {live_vision_status}\n"
            f"**WARDROBE DATABASE (FOR RECOMMENDATIONS):**\n{wardrobe_summary}"
        )

        try:
            # Send the user's message.
            response = self.chat.send_message(full_command)
            return response.text
        
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "I apologize, but I'm having trouble connecting to my styling brain right now. Please try again in a moment."