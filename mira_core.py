# mira_core.py
from dotenv import load_dotenv
load_dotenv() # Load API keys from the .env file first

import threading
import time
from vision_module import VisionModule
from voice_module import VoiceModule
from ai_stylist_module import AIStylistModule
import os

# Ensure environment variables are set before running
if not all([os.getenv('GEMINI_API_KEY'), os.getenv('ELEVEN_API_KEY')]):
    print("FATAL ERROR: Please set GEMINI_API_KEY and ELEVEN_API_KEY environment variables in the .env file.")
    exit(1)

class MiraAI_App:
    def __init__(self):
        print("--- Initializing MiraAI Personal Stylist ---")
        
        # 1. Initialize Modules
        self.voice_module = VoiceModule()
        self.ai_stylist = AIStylistModule()
        # Pass the voice module's speaker as a callback for the Vision Module
        # This allows the Vision module to "talk" when it saves a new item.
        self.vision_module = VisionModule(item_save_callback=self.voice_module.speak_response) 
        
        self.running = True

    def run_voice_loop(self):
        """Runs the main voice interaction loop in a separate thread."""
        
        # Give the user a welcome message (spoken)
        self.voice_module.speak_response(
            "Hello! I'm MiraAI, your live personal stylist. Look through your wardrobe, or just ask me for an outfit!"
        )
        
        while self.running:
            # 1. Listen for user command
            command = self.voice_module.listen_for_command()
            
            if command == "TIMEOUT":
                time.sleep(1)
                continue
            
            if command == "UNKNOWN" or command == "ERROR":
                self.voice_module.speak_response(
                    "I didn't quite catch that. Could you repeat your question, or look at a clothing item for me to scan?"
                )
                continue
                
            # Check for a "quit" command to stop the whole application
            if "quit" in command.lower() or "exit" in command.lower() or "stop" in command.lower():
                self.voice_module.speak_response("Goodbye! I hope you look great today.")
                self.running = False
                break
            
            # 2. Get AI Recommendation
            try:
                self.voice_module.speak_response("One moment, let me check your style options...")
                
                # The core logic: send the command to the AI stylist
                response_text = self.ai_stylist.generate_outfit_suggestion(command)
                
                # 3. Speak the AI response
                self.voice_module.speak_response(response_text)
                
            except Exception as e:
                print(f"An error occurred during AI interaction: {e}")
                self.voice_module.speak_response(
                    "I'm sorry, I encountered an internal error. Please check the console."
                )
                
            # Small pause before listening again
            time.sleep(0.5)

    def run(self):
        """Starts the Vision and Voice loops concurrently."""
        
        # Start the Voice loop in a separate thread (so the camera doesn't freeze)
        voice_thread = threading.Thread(target=self.run_voice_loop)
        voice_thread.start()

        try:
            # Run the Vision loop in the main thread (best practice for OpenCV)
            self.vision_module.run_detection()
            
        except IOError as e:
            print(f"FATAL VISION ERROR: {e}. Shutting down.")
        except KeyboardInterrupt:
            print("\nApplication closing from keyboard interrupt...")
        finally:
            # Cleanup
            self.running = False
            voice_thread.join(timeout=2) # Wait for the voice thread to finish
            self.vision_module.stop()
            print("--- MiraAI Application Closed ---")

if __name__ == '__main__':
    app = MiraAI_App()
    app.run()