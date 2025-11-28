
# for LLM - gemini-2.5-flash
from google import genai

# for Speech Recognition
import torch
import pyaudio
import numpy as np
import wave
import time

# for STT
from faster_whisper import WhisperModel

# for TTS
import edge_tts
import pygame

import os
import asyncio
from dotenv import load_dotenv

load_dotenv() # Load environment variables

# =============== CONFIGURATION =================
client = genai.Client() # Assumes GEMINI_API_KEY is set in environment variables
# Configure Silero VAD
CHUNKS = 512 # Number of frames per buffer
FORMAT = pyaudio.paInt16 # 16 bits per sample
CHANNELS = 1 # Mono audio
RATE = 16000 # Sampling rate

# --- INITIALIZE MODELS (Load once to be fast) ---
print("Loading Whisper Model...")
# 'base' is a good balance. Use 'small' for better accuracy if your PC can handle it.
stt_model = WhisperModel("small", device="cpu", compute_type="int8")

print(f"Loading Silero VAD Model...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False, #cache the model
    onnx=False,
    trust_repo=True
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# print("Configuring Gemini...")
chat = client.chats.create(
    model="gemini-2.5-flash",
)

print("Initializing Audio...")
pygame.mixer.init()

# --- HELPER FUNCTIONS ---

async def text_to_speech(text, output_file="response.mp3"):
    """Converts text to audio using Edge-TTS"""
    communicate = edge_tts.Communicate(text, "en-US-AndrewMultilingualNeural")
    await communicate.save(output_file)

def play_audio(file_path):
    """Plays audio using Pygame"""
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    # Unload to release the file so we can overwrite it next time
    pygame.mixer.music.unload()

def record_audio(filename="input.wav"):
    """Records audio when speech is detected using Silero VAD"""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE, # Sampling rate
                    input=True, # Microphone
                    frames_per_buffer=CHUNKS) 
    print("\n🎤 Listening... ")

    audio_buffer = []
    started_talking = False
    speech_counter = 0
    silence_limit = 30  # Number of silent chunks to wait before stopping, adjust for responsiveness
    
    while True:
        try:
            data = stream.read(CHUNKS, exception_on_overflow=False) # exception_on_overflow to avoid crashes
            audio_int16 = np.frombuffer(data, np.int16) # NumPy array for VAD
            audio_float32 = audio_int16.astype(np.float32) / 32768.0 # Convert to float32 required by Silero VAD

            voice_confidence = vad_model(torch.from_numpy(audio_float32), RATE).item()
            if voice_confidence > 0.5:  # Threshold for detecting speech
                if not started_talking:
                    print(f"🗣️ Voice Detected. Recording...")
                    started_talking = True
                silence_counter = 0
                audio_buffer.append(data)
            elif started_talking:
                audio_buffer.append(data)
                silence_counter += 1
                if silence_counter > silence_limit:
                    print("🔇 Silence detected. Stopping recording.")
                    break
        except KeyboardInterrupt:
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    if not audio_buffer:
        return False

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(audio_buffer))
    wf.close()
    return True
# --- MAIN LOOP ---
async def main_loop():
    print("✅ Agent is ready! Say 'Exit' to stop.")
    
    while True:
        # 1. RECORD
        if not record_audio("input.wav"):
            continue

        # 2. TRANSCRIBE (STT)
        print("👂 Transcribing...")
        segments, _ = stt_model.transcribe(
                "input.wav", 
                beam_size=5, 
                language="en",  # Telling it 'Hindi' usually captures Hinglish better than 'English'
                task="transcribe"
            )
        user_text = "".join([segment.text for segment in segments]).strip()
        
        if not user_text:
            continue
            
        print(f"👤 You said: {user_text}")

        if "exit" in user_text.lower():
            print("Goodbye!")
            break

        # 3. THINK (LLM)
        print("🧠 Thinking...")
        response = chat.send_message(user_text)
        ai_text = response.text
        print(f"🤖 AI: {ai_text}")

        # 4. SPEAK (TTS)
        print("🗣️ Speaking...")
        # We need to await this because edge-tts is async
        await text_to_speech(ai_text, "response.mp3")
        # await text_to_speech(user_text, "response.mp3")

        # 5. PLAY
        play_audio("response.mp3")

        # Optional: Clean up files
        os.remove("input.wav")
        os.remove("response.mp3")

if __name__ == "__main__":
    asyncio.run(main_loop())