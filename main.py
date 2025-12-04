
import asyncio
import os
from src.audio_input import AudioInput
from src.llm_engine import LLMEngine
from src.audio_output import AudioOutput

async def main_loop():
    """
    The Big Boss. The Conductor. The Main Loop. 🎩
    This function ties the Ear, Brain, and Mouth together into one somewhat cohesive unit.
    It's the puppet master pulling the strings.
    """
    print("✅ Agent is locked and loaded!")
    
    # The "Shut Up" Button
    # If this event is set, the Mouth stops talking immediately.
    stop_speaking_event = asyncio.Event()
    
    # --- The Crew ---
    # 1. The Ear: Handles the mic, VAD, and transcription.
    audio_input = AudioInput(stop_speaking_event)
    
    # 2. The Brain: The smart part (hopefully). Talks to Gemini.
    llm_engine = LLMEngine()
    
    # 3. The Mouth: The loud part. Turns text into noise.
    audio_output = AudioOutput(stop_speaking_event)
    
    # --- Spin up the background workers ---
    # These guys run forever, doing the heavy lifting in the background.
    await audio_input.start()
    await audio_output.start()
    
    while True:
        # 1. LISTEN: Wait for the Ear to hand us a complete thought.
        user_text = await audio_input.text_queue.get()
        print(f"👤 You said (Final): {user_text}")

        # Check for exit commands
        # Because sometimes you just need some peace and quiet.
        exit_phrases = ["exit", "shutdown", "you can sleep"]
        if any(phrase in user_text.lower() for phrase in exit_phrases):
            print("Catch you on the flip side!")
            os._exit(0)

        # 2. THINK: Send it to the Brain and get a stream of thoughts back.
        response_stream = llm_engine.generate_response_stream(user_text)
        
        # 3. SPEAK: Pipe those thoughts directly to the Mouth.
        
        # We need this callback so the Ear knows when to cover its ears.
        # Otherwise, the AI hears itself and we get an infinite echo loop of doom.
        def on_start_speaking():
            audio_input.is_speaking = True
            stop_speaking_event.clear() # Reset the kill switch
            
        await audio_output.speak(response_stream, on_start_speaking)
        
        # Done talking? Cool, tell the Ear it's safe to listen again.
        audio_input.is_speaking = False

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C
        pass