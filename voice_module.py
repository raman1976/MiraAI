# voice_module.py (FINAL VERSION)
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import Voice 
import os
import wave               # For wave file specs (not decoding the stream)
from io import BytesIO   
import sounddevice as sd
import numpy as np
from pydub import AudioSegment  # <-- REQUIRES FFMPEG to decode the stream

# --- Configuration ---
ELEVEN_API_KEY = os.getenv('ELEVEN_API_KEY')
# Use your verified ID (e.g., the one that didn't return 404)
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
        """Generates audio and plays it back using pydub (FFMPEG) and sounddevice."""
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
                output_format="mp3_44100" # Requesting MP3 for pydub to decode
            )
            
            # 2. Collect the audio chunks from the generator into one bytes object
            audio_bytes = b"".join(audio_generator)
            
            # 3. Use pydub to load and process the audio bytes (Requires FFMPEG)
            audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
            
            # 4. Extract parameters and convert to numpy array (for sounddevice)
            # Ensure 16-bit PCM for sounddevice compatibility
            audio_segment = audio_segment.set_sample_width(2) 
            audio_np = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)
            
            # 5. Play the audio using sounddevice
            sd.play(audio_np, samplerate=audio_segment.frame_rate)
            sd.wait()

        except Exception as e:
            print(f"ElevenLabs error: {e}")
            print("Falling back to text-only output.")