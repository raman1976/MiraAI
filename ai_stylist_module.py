# ai_stylist_module.py (FINAL VERSION)
from google import genai
from google.genai import types
import os
from wardrobe_db import get_wardrobe_summary

# --- Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = 'gemini-2.5-flash' 

class AIStylistModule:
    def __init__(self):
        print("Initializing AI Stylist Module...")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
            
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.system_prompt = self._get_system_prompt()
        
        # --- FIX APPLIED: Initialize chat with system instruction ---
        self.chat = self.client.chats.create(
            model=MODEL_NAME, 
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt 
            )
        ) 
        print("AI Stylist Module Ready.")
        
    def _get_system_prompt(self):
        """Defines the AI's persona, rules, and goal."""
        prompt = (
            "You are MiraAI, an expert personal stylist. Your goal is to give clear, friendly, and "
            "actionable outfit recommendations based on the user's request and their virtual wardrobe. "
            "You must respond naturally and conversationally, suitable for **voice output** (keep responses concise and engaging). "
            "NEVER mention the word 'prompt' or 'virtual wardrobe summary'. "
            "Your output must be a single, text-based response. \n\n"
            "**Styling Rules:**\n"
            "1. Base all suggestions ONLY on the available items in the wardrobe summary provided in the context.\n"
            "2. Consider color harmony, occasion, and style compatibility.\n"
            "3. If the wardrobe is empty or lacks suitable items, clearly state what's missing (e.g., 'You need a pair of casual shoes to complete that look.')."
        )
        return prompt

    def generate_outfit_suggestion(self, user_command):
        """
        Generates an outfit suggestion by combining the user's query 
        and the current wardrobe state, then calling the Gemini API.
        """
        wardrobe_summary = get_wardrobe_summary()
        
        # Craft the full message to the model
        full_command = (
            f"**USER COMMAND:** '{user_command}'. "
            f"**CURRENT WARDROBE (USE ONLY THESE ITEMS):**\n{wardrobe_summary}"
        )

        try:
            # Send the user's message. The persona is handled in __init__.
            response = self.chat.send_message(full_command)
            return response.text
        
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "I apologize, but I'm having trouble connecting to my styling brain right now. Please try again in a moment."