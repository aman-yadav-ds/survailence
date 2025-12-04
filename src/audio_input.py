import asyncio
import queue
import numpy as np
import pyaudio
import torch
import time
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

class AudioInput:
    """
    The 'Ear' of the operation. 👂
    Responsible for listening to the microphone, detecting voice activity (VAD),
    and transcribing speech to text using Faster-Whisper.
    """
    def __init__(self, stop_event, wake_word="friday", wake_word_enabled=True):
        # The "Kill Switch" event. If set, we stop any active output.
        self.stop_event = stop_event
        
        # The "Inbox" for the main loop. Final, polished text lands here.
        self.text_queue = asyncio.Queue()
        
        # The "Raw Feed". Raw audio bytes from the mic are dumped here.
        self.audio_queue = queue.Queue()
        
        # --- Wake Word Configuration ---
        self.wake_word = wake_word.lower()
        self.wake_word_enabled = wake_word_enabled
        self.is_awake = not wake_word_enabled  # If disabled, we're always listening.
        self.wake_word_timeout = 30  # Seconds of silence before we snooze.
        self.last_wake_time = 0
        self.wake_word_buffer_size = 3.0  # Rolling buffer size (in seconds) for wake detection.
        
        # Throttling for wake word checks (prevent CPU spam)
        self.last_wake_check_time = 0
        self.WAKE_CHECK_INTERVAL = 0.5 # Check at most every 0.5 seconds
        
        # --- Audio Stream Settings ---
        self.CHUNKS = 512       # Buffer size. Lower = less latency, Higher = less CPU load.
        self.FORMAT = pyaudio.paInt16 # 16-bit audio. Standard for speech recognition.
        self.CHANNELS = 1       # Mono audio. We only need one ear.
        self.RATE = 16000       # 16kHz sample rate. The native tongue of Whisper.
        
        # --- Voice Activity Detection (VAD) Thresholds ---
        # RMS (Root Mean Square) = Volume.
        self.RMS_THRESHOLD = 300      # Noise floor. Ignore anything quieter than this.
        self.BARGE_IN_RMS = 3000      # Interruption threshold. Yell louder than this to stop the bot.
        self.BARGE_IN_CONFIDENCE = 0.8 # VAD confidence required to interrupt.
        
        # State flag: Are we currently outputting audio?
        self.is_speaking = False
        
        # --- Model Initialization ---
        print("🎧 Initializing Audio Subsystems...")
        
        # Initialize Faster-Whisper
        # We use a tiny model for quick wake-word detection and a small model for accurate transcription.
        
        print("  - Loading 'tiny.en' model for wake word detection...")
        self.wake_word_model = WhisperModel(
            "tiny.en",
            device="cpu",
            compute_type="int8",  # Quantized for CPU efficiency
            num_workers=2,
            cpu_threads=4
        )
        
        print("  - Loading 'small.en' model for main transcription...")
        self.stt_model = WhisperModel(
            "small.en",
            device="cpu",
            compute_type="int8",
            num_workers=2,
            cpu_threads=4
        )
        
        print("  - Loading Silero VAD for voice detection...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        
        # --- Wake Word Buffer Setup ---
        if self.wake_word_enabled:
            print(f"✅ Wake word ready! Say '{self.wake_word.capitalize()}' to activate.")
            self.wake_word_buffer = []
            # Calculate max chunks for the rolling buffer
            self.max_wake_buffer_chunks = int(self.wake_word_buffer_size * (self.RATE / self.CHUNKS))

    def mic_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback.
        This needs to be lightning fast. Grab the data, shove it in the queue, and get out.
        """
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    async def start(self):
        """Ignites the background listening and transcription engines."""
        asyncio.create_task(self.listen_loop())
        asyncio.create_task(self.transcription_loop())
        if self.wake_word_enabled:
            asyncio.create_task(self.wake_word_timeout_loop())

    def _transcribe_wake_word(self, audio_float32):
        """
        Blocking transcription call to be run in a thread.
        """
        try:
            segments, _ = self.wake_word_model.transcribe(
                audio_float32,
                beam_size=1,
                language="en",
                vad_filter=False,
                without_timestamps=True
            )
            return " ".join([segment.text.lower().strip() for segment in segments])
        except Exception as e:
            print(f"⚠️ Wake word transcription failed: {e}")
            return ""

    async def check_for_wake_word(self, audio_buffer):
        """
        Checks the audio buffer for the wake word using the tiny model.
        Runs in a separate thread to avoid blocking the main loop.
        Returns True if detected, False otherwise.
        """
        try:
            # Flatten buffer to a single byte string
            audio_data = b''.join(audio_buffer)
            # Convert to normalized float32 array
            audio_int16 = np.frombuffer(audio_data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Run transcription in a thread
            text = await asyncio.to_thread(self._transcribe_wake_word, audio_float32)
            
            if self.wake_word in text:
                print(f"\n🎯 Wake word '{self.wake_word.capitalize()}' detected!")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Wake word check failed: {e}")
            return False

    async def wake_word_timeout_loop(self):
        """
        The Sandman. 😴
        Puts the agent back to sleep if no one talks to it for a while.
        """
        while True:
            await asyncio.sleep(1)
            
            if self.is_awake and self.last_wake_time > 0:
                elapsed = time.time() - self.last_wake_time
                if elapsed > self.wake_word_timeout:
                    self.is_awake = False
                    print(f"\n😴 Timeout reached. Going to sleep.")
                    print(f"   Say '{self.wake_word.capitalize()}' to wake me up.")

    async def listen_loop(self):
        """
        The Main Listening Loop.
        Captures audio, checks for voice activity, and manages recording state.
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNKS,
                        stream_callback=self.mic_callback)
        
        stream.start_stream()
        
        if self.wake_word_enabled:
            print(f"\n🎤 Listening for wake word '{self.wake_word.capitalize()}'...")
        else:
            print("\n🎤 Listening... (Say 'Exit' to stop)")

        audio_buffer = []
        is_recording = False
        silence_counter = 0
        # Calculate silence limit (0.8 seconds) - Reduced for lower latency
        silence_limit = int(0.8 * (self.RATE / self.CHUNKS)) 
        
        # Track when recording started for grace period
        recording_start_time = 0
        
        self.transcription_queue = asyncio.Queue()

        while True:
            try:
                # Non-blocking get from queue
                data = self.audio_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01) # Yield to other tasks
                continue

            # Convert for processing
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Compute metrics
            rms = np.sqrt(np.mean(audio_int16.astype(np.float32)**2))
            voice_confidence = self.vad_model(torch.from_numpy(audio_float32), self.RATE).item()
            
            # --- Wake Word Mode ---
            if self.wake_word_enabled and not self.is_awake:
                self.wake_word_buffer.append(data)
                if len(self.wake_word_buffer) > self.max_wake_buffer_chunks:
                    self.wake_word_buffer.pop(0)
                
                # Check for wake word only if we hear something resembling speech
                if voice_confidence > 0.6 and rms > self.RMS_THRESHOLD:
                    # Throttle checks
                    current_time = time.time()
                    if current_time - self.last_wake_check_time > self.WAKE_CHECK_INTERVAL:
                        self.last_wake_check_time = current_time
                        
                        if await self.check_for_wake_word(self.wake_word_buffer):
                            self.is_awake = True
                            self.last_wake_time = time.time()
                            print("✅ Awake and listening!")
                            
                            # Transition directly to recording state to capture the rest of the sentence
                            # We keep the wake buffer as the start of the recording
                            is_recording = True
                            recording_start_time = time.time()
                            audio_buffer = list(self.wake_word_buffer) 
                            self.wake_word_buffer = []
                            silence_counter = 0
                continue
            
            # --- Active Mode ---
            
            # Update activity timer
            if self.wake_word_enabled and voice_confidence > 0.5 and rms > self.RMS_THRESHOLD:
                self.last_wake_time = time.time()
            
            # 1. Interruption Logic (Barge-In)
            if self.is_speaking:
                if (voice_confidence > self.BARGE_IN_CONFIDENCE) and (rms > self.BARGE_IN_RMS):
                    print("\n🛑 Interruption detected! Cutting speech.")
                    self.stop_event.set()
                    self.is_speaking = False
                    
                    if not is_recording:
                        print(f"\n🗣️ User interrupted...")
                        is_recording = True
                        recording_start_time = time.time()
                        audio_buffer = []
                        silence_counter = 0
            
            # 2. Start Recording
            elif not is_recording:
                if (voice_confidence > 0.5) and (rms > self.RMS_THRESHOLD):
                    print(f"\n🗣️ Speech detected...")
                    is_recording = True
                    recording_start_time = time.time()
                    audio_buffer = []
                    silence_counter = 0

            # 3. Continue Recording
            if is_recording:
                audio_buffer.append(data)
                
                # Silence Detection
                if (voice_confidence < 0.5) or (rms < self.RMS_THRESHOLD):
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                # End of Turn
                # Grace Period: Don't cut off due to silence if we just started recording (e.g. < 2.0s)
                # This allows for a pause between "Friday" and the command.
                if silence_counter > silence_limit:
                    if (time.time() - recording_start_time > 2.0):
                        print("🔇 Silence detected. Processing turn.")
                        is_recording = False
                        await self.process_audio_input(audio_buffer, is_final=True)
                        audio_buffer = []
                        silence_counter = 0
                    # else: inside grace period, ignore silence

    async def process_audio_input(self, audio_buffer, is_final=False):
        """Prepares audio data for the transcription loop."""
        if not audio_buffer:
            return
        audio_data = b''.join(audio_buffer)
        audio_int16 = np.frombuffer(audio_data, np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        await self.transcription_queue.put((audio_float32, is_final))

    def detect_hallucination(self, text, audio_duration):
        """
        The BS Detector. 🕵️‍♂️
        Filters out common Whisper hallucinations like "Thank you for watching".
        """
        if not text:
            return True
        
        text_lower = text.lower().strip()
        
        # The Hall of Shame
        hallucination_phrases = [
            "thank you", "thanks", "thank you for watching",
            "please subscribe", "like and subscribe", "see you next time",
            "goodbye", "bye bye", "thanks for watching", "i hope you enjoyed",
            "don't forget to subscribe", "hit the like button",
            "thanks for listening", "music", "applause", "subtitle",
            "subtitles by", "transcribed by", "translated by"
        ]
        
        if text_lower in hallucination_phrases:
            return True
        
        for phrase in hallucination_phrases:
            if text_lower.startswith(phrase):
                return True
        
        # Repetition Check
        words = text_lower.split()
        if len(words) > 3:
            half = len(words) // 2
            if words[:half] == words[half:2*half]:
                return True
        
        # Speed Check (Words per Second)
        if audio_duration > 0:
            words_per_second = len(words) / audio_duration
            if words_per_second > 5:
                print(f"⚠️ Suspicious speech rate: {words_per_second:.1f} wps")
                return True
        
        return False

    async def transcription_loop(self):
        """
        The Scribe. ✍️
        Consumes raw audio and produces text.
        """
        while True:
            audio_data, is_final = await self.transcription_queue.get()
            
            if not is_final:
                self.transcription_queue.task_done()
                continue
                
            print(f"👂 Transcribing...")
            audio_duration = len(audio_data) / self.RATE
            
            try:
                segments, _ = self.stt_model.transcribe(
                    audio_data,
                    beam_size=1,
                    language="en",
                    vad_filter=True,
                    vad_parameters=dict(
                        threshold=0.6,
                        min_speech_duration_ms=250,
                        min_silence_duration_ms=1000
                    ),
                    condition_on_previous_text=False,
                    without_timestamps=False,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6
                )
                
                text = " ".join([segment.text.strip() for segment in segments])
                
            except Exception as e:
                print(f"⚠️ Transcription failed: {e}")
                text = ""

            # Hallucination Check
            if text and self.detect_hallucination(text, audio_duration):
                print(f"⚠️ Hallucination filtered: '{text}'")
                text = ""

            if text:
                print(f"📝 Transcript: {text}")
                await self.text_queue.put(text)
            
            self.transcription_queue.task_done()

# Installation instructions:
"""
Install Faster-Whisper:
pip install faster-whisper

Optional (for speed):
pip install ctranslate2>=4.0.0
"""