from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class LLMEngine:
    """
    The 'Brain' of the operation. 🧠
    This is where the magic happens (and occasionally some hallucinations).
    Wraps the Gemini API using LangChain to give our agent some semblance of intelligence.
    """
    def __init__(self):
        # Initialize the Gemini client via LangChain.
        # We're using 'ChatGoogleGenerativeAI' because it plays nice with the free API key.
        # No credit card required, just pure, unadulterated AI power.
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-flash-latest", 
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7 # A little creativity, but let's not get too crazy.
        )

        # The "Personality" of the AI.
        # We tell it to be a helpful voice assistant, not a novelist.
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a voice AI Agent named Friday. You are helpful, witty, and concise. "
                       "Your responses are spoken out loud, so keep them short and conversational. "
                       "Avoid markdown formatting like bold or bullet points unless absolutely necessary. "
                       "If asked about yourself, you can mention you are a brain in a jar powered by Gemini."),
            ("human", "{text}"),
        ])
        
        # Create the chain (Prompt -> LLM)
        self.chain = self.prompt | self.llm
        
    def generate_response_stream(self, text):
        """
        Sends your text to Gemini and returns a 'stream' of consciousness.
        Streaming is crucial here. It allows the 'Mouth' to start yapping while the 'Brain' 
        is still figuring out the end of the sentence. Makes it feel alive.
        """
        print("🧠 Thinking (with LangChain)...")
        # Stream the response so we can pipeline the audio generation.
        return self.chain.stream({"text": text})

if __name__ == "__main__":
    llm = LLMEngine()
    for chunk in llm.generate_response_stream("Explain Greed"):
        print(chunk.content)