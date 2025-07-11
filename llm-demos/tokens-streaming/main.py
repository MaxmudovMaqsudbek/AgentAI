import os
import time
from typing import Generator, List
import google.generativeai as genai
from dotenv import load_dotenv

USER_REQUEST = 'Give me the list of first 30 presidents of the United States of America.'


class GeminiStreamer:
    """Advanced streaming client for Google Gemini AI."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """Initialize the Gemini streamer with API key and model."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def stream_response(self, prompt: str, temperature: float = 0.0) -> Generator[str, None, None]:
        """
        Stream response from Gemini AI.
        
        Args:
            prompt: The user prompt to send
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            
        Yields:
            str: Content chunks as they arrive
        """
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=2048,
        )
        
        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def get_complete_response(self, prompt: str, temperature: float = 0.0) -> tuple[str, float]:
        """
        Get complete streamed response with timing information.
        
        Args:
            prompt: The user prompt to send
            temperature: Controls randomness
            
        Returns:
            tuple: (complete_response, total_time)
        """
        start_time = time.time()
        collected_chunks: List[str] = []
        
        print("🤖 Gemini AI is responding...\n")
        print("-" * 50)
        
        try:
            for chunk in self.stream_response(prompt, temperature):
                chunk_time = time.time() - start_time
                collected_chunks.append(chunk)
                
                # Print with real-time streaming effect
                print(chunk, end='', flush=True)
                
        except Exception as e:
            print(f"\n❌ Error during streaming: {e}")
            return "", time.time() - start_time
            
        total_time = time.time() - start_time
        full_response = ''.join(collected_chunks)
        
        print("\n" + "-" * 50)
        print(f"✅ Response completed in {total_time:.2f} seconds")
        print(f"📊 Total characters: {len(full_response)}")
        print(f"🔤 Total words: {len(full_response.split())}")
        
        return full_response, total_time


def main():
    """Main execution function."""
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize Gemini streamer
        streamer = GeminiStreamer(api_key)
        
        print(f"🚀 Starting Gemini AI streaming demo")
        print(f"📝 User Request: {USER_REQUEST}")
        print("=" * 60)
        
        # Get streamed response
        response, duration = streamer.get_complete_response(USER_REQUEST, temperature=0.1)
        
        if response:
            print(f"\n📋 Summary:")
            print(f"   • Processing time: {duration:.2f} seconds")
            print(f"   • Response length: {len(response)} characters")
            print(f"   • Average speed: {len(response)/duration:.1f} chars/second")
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == '__main__':
    main()

