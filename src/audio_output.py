import asyncio
import edge_tts
import pygame
import io
import re

class AudioOutput:
    """
    The 'Mouth' of the operation. 🗣️
    Converts text to speech using Edge TTS and plays it back via Pygame.
    """
    def __init__(self, stop_event):
        self.stop_event = stop_event
        # Queue for holding audio streams. FIFO (First In, First Out).
        self.audio_queue = asyncio.Queue()
        # Initialize the DJ (Pygame Mixer)
        pygame.mixer.init()
        
    async def start(self):
        """Kicks off the background playback loop."""
        asyncio.create_task(self.playback_loop())

    async def generate_audio_stream(self, text):
        """
        The Voice Box. 🎙️
        Uses Edge TTS to generate audio bytes from text.
        Returns an in-memory BytesIO stream because disk I/O is for chumps.
        """
        communicate = edge_tts.Communicate(text, "en-US-EmmaNeural")
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return io.BytesIO(audio_data)

    async def play_audio_stream(self, audio_stream):
        """
        The Speaker. 🔊
        Plays a single audio stream. Monitors the 'stop_event' like a hawk
        to cut off speech instantly if interrupted.
        """
        audio_stream.seek(0)
        try:
            pygame.mixer.music.load(audio_stream)
            pygame.mixer.music.play()
            
            # Busy wait loop for playback
            while pygame.mixer.music.get_busy():
                # The "Shut Up" check
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                    return False
                # Low latency sleep for quick reaction time
                await asyncio.sleep(0.01)
            
            pygame.mixer.music.unload()
            return True
        except Exception as e:
            print(f"⚠️ Audio playback error: {e}")
            return True

    async def playback_loop(self):
        """
        The DJ Booth. 🎧
        Continuously pulls tracks from the queue and spins them.
        Allows for pipelining: generating the next sentence while playing the current one.
        """
        while True:
            audio_stream = await self.audio_queue.get()
            
            # Check if we should skip this track (interruption)
            if self.stop_event.is_set():
                self.audio_queue.task_done()
                continue
                
            await self.play_audio_stream(audio_stream)
            self.audio_queue.task_done()

    async def speak(self, text_stream, on_start_speaking=None):
        """
        The Orator. 📜
        Consumes a stream of text tokens, assembles them into sentences,
        and queues them for playback.
        """
        text_buffer = ""
        
        for chunk in text_stream:
            # Check for interruption
            if self.stop_event.is_set():
                break
            
            # Extract text from various chunk types
            text_chunk = ""
            if hasattr(chunk, "content"):
                text_chunk = chunk.content # LangChain
            elif hasattr(chunk, "text"):
                text_chunk = chunk.text # Google GenAI SDK
            elif isinstance(chunk, str):
                text_chunk = chunk # Raw string
            
            if text_chunk:
                text_buffer += text_chunk
                
                # Sentence boundary detection
                if any(punct in text_buffer for punct in [".", "?", "!"]):
                    # Split by sentence endings, keeping the delimiter
                    sentences = re.split(r'(?<=[.?!])\s+', text_buffer)
                    
                    # Process all complete sentences
                    for sentence in sentences[:-1]:
                        if self.stop_event.is_set():
                            break
                        
                        # Signal start of speech (e.g., to pause VAD)
                        if on_start_speaking:
                            on_start_speaking()
                            
                        print(f"🗣️ AI Speaking: {sentence}")
                        
                        # Generate and queue audio
                        audio_stream = await self.generate_audio_stream(sentence)
                        await self.audio_queue.put(audio_stream)
                        
                    text_buffer = sentences[-1]
        
        # Flush remaining text
        if text_buffer and not self.stop_event.is_set():
            if on_start_speaking:
                on_start_speaking()
            print(f"🗣️ AI Speaking: {text_buffer}")
            audio_stream = await self.generate_audio_stream(text_buffer)
            await self.audio_queue.put(audio_stream)
            
        # Wait for the queue to drain
        await self.audio_queue.join()
