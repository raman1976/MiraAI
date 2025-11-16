# voice_module.py (FINAL NON-BLOCKING VERSION)
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice 
import os
from io import BytesIO 
import sounddevice as sd
import numpy as np
from pydub import AudioSegment# <-- REQUIRES FFMPEG to decode the stream

# --- Configuration ---
ELEVEN_API_KEY = os.getenv('ELEVEN_API_KEY')
ELEVEN_VOICE_ID = "cgSgspJ2msm6clMCkdW9" 

class VoiceModule:
    def __init__(self, eleven_api_key=ELEVEN_API_KEY):
        print("Initializing Voice Module...")
        if eleven_api_key:
            self.elevenlabs_client = ElevenLabs(api_key=eleven_api_key)
            print("ElevenLabs Client initialized.")
        else:
            self.elevenlabs_client = None
            print("WARNING: ELEVEN_API_KEY not found. TTS will be disabled.")
            
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        print("Voice Module Ready.")

    def listen_for_command(self):
        """Listens for user speech and returns the transcribed text."""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("\nListening for a command...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                return "TIMEOUT"
            
        print("Processing audio...")
        try:
            command = self.recognizer.recognize_google(audio)
            print(f"User said: **{command}**")
            return command
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return "UNKNOWN"
        except sr.RequestError as e:
            print(f"Speech recognition service error; {e}")
            return "ERROR"

    def speak_response(self, text):
        """Generates audio and plays it back using pydub (FFMPEG) and sounddevice (non-blocking)."""
        if not self.elevenlabs_client:
            print(f"(TTS Disabled) MiraAI would say: {text}")
            return
            
        print(f"MiraAI response: {text}")
        try:
            # 1. Generate the audio (returns generator)
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=ELEVEN_VOICE_ID,
                model_id="eleven_multilingual_v2", 
                output_format="mp3_44100" 
            )
            
            # 2. Collect the audio chunks from the generator into one bytes object
            audio_bytes = b"".join(audio_generator)
            
            # 3. Use pydub to load and process the audio bytes (Requires FFMPEG installed and in PATH)
            audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            
            # 4. Extract parameters and convert to numpy array
            audio_segment = audio_segment.set_sample_width(2) 
            audio_np = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            
            # 5. Play the audio using sounddevice
            # sd.play() is non-blocking by default (blocking=False), and since we call
            # this from a background thread in mira_app.py, the main thread stays free.
            sd.play(audio_np, samplerate=audio_segment.frame_rate)
            
            # CRITICAL FIX: REMOVE THE BLOCKING CALL
            # sd.wait() would block the thread until playback finishes, which is what caused the UI freeze.
            # We rely on the threading.Thread in mira_app.py to handle the background execution.

        except Exception as e:
            print(f"ElevenLabs or Audio Playback error: {e}")
            # NOTE: The FileNotFoundError for 'ffmpeg' is common if FFMPEG is not in PATH.
            if "ffmpeg" in str(e).lower():
                 print("CRITICAL: FFMPEG NOT FOUND. Ensure FFMPEG is installed and in your system PATH.")
            return