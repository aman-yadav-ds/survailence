import asyncio
import queue
import numpy as np
import pyaudio
import torch
import wave
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class AudioInput:
    def __init__(self, stop_event):
        # This event is the "Shut Up" button. If it's set, we stop talking.
        self.stop_event = stop_event
        
        # The "Inbox" for the main loop. Final text goes here.
        self.text_queue = asyncio.Queue()
        
        # The "Raw Feed". Audio bytes from the mic get dumped here.
        self.audio_queue = queue.Queue()
        
        # --- Audio Settings ---
        self.CHUNKS = 512       # Chunk size. Too small = CPU burn, too big = lag.
        self.FORMAT = pyaudio.paInt16 # 16-bit audio. Good enough for voice.
        self.CHANNELS = 1       # Mono. We have one mouth, so one ear is fine.
        self.RATE = 16000       # 16kHz. The gold standard for speech models.
        
        # --- "Can you hear me now?" Thresholds ---
        # RMS = Volume. How loud are you screaming?
        self.RMS_THRESHOLD = 200      # Silence threshold. Don't transcribe breathing.
        self.BARGE_IN_RMS = 3000      # Interruption threshold. You gotta yell to stop the bot.
        self.BARGE_IN_CONFIDENCE = 0.8 # Higher confidence for barge-in to avoid false positives.
        
        # Are we currently blabbering?
        self.is_speaking = False
        
        # --- Brain Transplants ---
        print("Loading Gemini STT... (The ears)")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.stt_model = genai.GenerativeModel("gemini-2.5-flash")
        
        print("Loading Silero VAD... (The reflex)")
        # This thing is crazy fast at telling speech from a car horn.
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )

    def mic_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio calls this when it has data. 
        We treat it like a hot potato: grab it and throw it in the queue.
        Don't do heavy math here or the audio will glitch.
        """
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def start(self):
        """Starts the background listening and transcription tasks."""
        asyncio.create_task(self.listen_loop())
        asyncio.create_task(self.transcription_loop())

    async def listen_loop(self):
        """
        The 'Ear' Loop.
        It sits there, listening to the mic, filtering out the noise,
        and waiting for you to say something profound.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNKS,
                        stream_callback=self.mic_callback)
        
        stream.start_stream()
        print("\n🎤 Listening... (Say 'Exit' to stop)")

        audio_buffer = []
        is_recording = False
        silence_counter = 0
        # 16000 Hz / 512 samples per chunk ~= 31.25 chunks per second
        # 1.5 seconds of silence to stop recording
        silence_limit = int(1.5 * (self.RATE / self.CHUNKS)) 
        
        # The conveyor belt to the Brain (transcription_loop)
        self.transcription_queue = asyncio.Queue()

        while True:
            try:
                # Get the latest audio chunk from the mic
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                # If no data yet, take a tiny nap to let other tasks run
                await asyncio.sleep(0.01)
                continue

            # Convert raw bytes to numbers so we can do math on them
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Calculate volume (RMS)
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))
            
            # Ask the VAD model: "Is this speech?" (returns 0.0 to 1.0)
            voice_confidence = self.vad_model(torch.from_numpy(audio_float32), self.RATE).item()
            
            # --- Interruption & Speech Detection Logic ---
            
            # 1. Check for Interruption (Barge-In)
            if self.is_speaking:
                # If AI is talking, we need high confidence and volume to interrupt
                if (voice_confidence > self.BARGE_IN_CONFIDENCE) and (rms > self.BARGE_IN_RMS):
                    print("\n🛑 Interruption detected! Stopping AI...")
                    self.stop_event.set() # Kill the output
                    self.is_speaking = False
                    
                    # Start recording immediately
                    if not is_recording:
                        print(f"\n🗣️ User started speaking (Interruption)...")
                        is_recording = True
                        audio_buffer = []
                        silence_counter = 0
            
            # 2. Check for Normal Speech Start
            elif not is_recording:
                if (voice_confidence > 0.5) and (rms > self.RMS_THRESHOLD):
                    print(f"\n🗣️ User started speaking...")
                    is_recording = True
                    audio_buffer = []
                    silence_counter = 0

            # 3. Recording State
            if is_recording:
                audio_buffer.append(data)
                
                # Check for Silence (End of Turn)
                if (voice_confidence < 0.5) or (rms < self.RMS_THRESHOLD):
                    silence_counter += 1
                else:
                    silence_counter = 0 # Reset if we hear speech again
                
                if silence_counter > silence_limit:
                    print("🔇 End of speech detected.")
                    is_recording = False
                    
                    # Send the full buffer to be transcribed
                    await self.process_audio_input(audio_buffer, is_final=True)
                    audio_buffer = []
                    silence_counter = 0

    async def process_audio_input(self, audio_buffer, is_final=False):
        """Helper to prepare audio data for the transcriber."""
        if not audio_buffer:
            return
        # Combine all chunks into one byte stream
        audio_data = b''.join(audio_buffer)
        audio_int16 = np.frombuffer(audio_data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        # Send it to the transcription loop
        await self.transcription_queue.put((audio_float32, is_final))

    async def transcription_loop(self):
        """
        The 'Translator' Loop.
        Takes raw audio bytes and turns them into words using Gemini.
        """
        while True:
            audio_data, is_final = await self.transcription_queue.get()
            
            # We only care about final segments now
            if not is_final:
                self.transcription_queue.task_done()
                continue
                
            print(f"👂 Transcribing full turn...")
            
            try:
                # Convert float32 back to int16 for wav encoding
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Create a WAV file in memory
                import io
                wav_io = io.BytesIO()
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_int16.tobytes())
                
                wav_data = wav_io.getvalue()
                
                # Generate content using Gemini with lower temperature for accuracy
                response = self.stt_model.generate_content(
                    [
                        "You are an expert speech-to-text transcriber. Transcribe the following audio exactly as spoken. "
                        "Do not add any commentary, markdown, or extra text. If the audio is silent or unintelligible, return an empty string.",
                        {"mime_type": "audio/wav", "data": wav_data}
                    ],
                    generation_config={"temperature": 0.0}
                )
                
                text = response.text.strip() if response.text else ""
                
            except Exception as e:
                print(f"⚠️ Transcription error: {e}")
                text = ""

            # Filter out common "ghost" phrases
            if text.lower() in ["thank you.", "thank you", "you", "thanks."]:
                 print(f"⚠️ Ignored hallucination: '{text}'")
                 text = ""

            if text:
                print(f"📝 Final Transcript: {text}")
                await self.text_queue.put(text)
            
            self.transcription_queue.task_done()
