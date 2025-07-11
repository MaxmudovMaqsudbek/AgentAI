"""
Advanced Multi-Provider AI Streaming Chat System.

This module provides an enterprise-grade streaming chat system supporting both OpenAI and Google Gemini AI
with real-time streaming, performance monitoring, error handling, and comprehensive analytics.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streaming_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face imports
try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install with: pip install huggingface_hub")


class StreamProvider(Enum):
    """Supported streaming providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"


@dataclass
class StreamChunk:
    """Represents a single streaming chunk."""
    content: str
    timestamp: float
    chunk_index: int
    provider: StreamProvider
    model: str
    finish_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "chunk_index": self.chunk_index,
            "provider": self.provider.value,
            "model": self.model,
            "finish_reason": self.finish_reason
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming requests."""
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: Optional[int] = 1000
    stream_delay: float = 0.0  # Artificial delay between chunks (for testing)
    show_timing: bool = True
    save_chunks: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class StreamingResponse:
    """Complete streaming response with analytics."""
    content: str
    provider: StreamProvider
    model: str
    chunks: List[StreamChunk]
    total_time: float
    first_chunk_time: float
    average_chunk_time: float
    total_chunks: int
    characters_per_second: float
    words_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "total_time": self.total_time,
            "first_chunk_time": self.first_chunk_time,
            "average_chunk_time": self.average_chunk_time,
            "total_chunks": self.total_chunks,
            "characters_per_second": self.characters_per_second,
            "words_per_second": self.words_per_second,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


class OpenAIStreamer:
    """Advanced OpenAI streaming client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI streaming client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI streaming client initialized")
    
    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        config: StreamingConfig
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream completion from OpenAI.
        
        Args:
            messages: List of conversation messages
            config: Streaming configuration
            
        Yields:
            StreamChunk objects as they arrive
        """
        try:
            start_time = time.time()
            chunk_index = 0
            
            # Create streaming request
            response = self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True
            )
            
            logger.info(f"Started OpenAI streaming: {config.model}")
            
            for chunk in response:
                current_time = time.time()
                
                # Extract content from chunk
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    finish_reason = chunk.choices[0].finish_reason
                    
                    stream_chunk = StreamChunk(
                        content=content,
                        timestamp=current_time - start_time,
                        chunk_index=chunk_index,
                        provider=StreamProvider.OPENAI,
                        model=config.model,
                        finish_reason=finish_reason
                    )
                    
                    chunk_index += 1
                    
                    # Optional delay for testing
                    if config.stream_delay > 0:
                        time.sleep(config.stream_delay)
                    
                    yield stream_chunk
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise


class GeminiStreamer:
    """Advanced Gemini AI streaming client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini streaming client."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        logger.info("Gemini streaming client initialized")
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI format messages to Gemini prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System Instructions: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def _map_model(self, model: str) -> str:
        """Map OpenAI model names to Gemini equivalents."""
        model_mapping = {
            "gpt-4o": "gemini-1.5-flash",
            "gpt-4": "gemini-1.5-flash",
            "gpt-3.5-turbo": "gemini-1.5-flash",
        }
        return model_mapping.get(model, "gemini-1.5-flash")
    
    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        config: StreamingConfig
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream completion from Gemini AI.
        
        Args:
            messages: List of conversation messages
            config: Streaming configuration
            
        Yields:
            StreamChunk objects as they arrive
        """
        try:
            start_time = time.time()
            chunk_index = 0
            
            # Map model and create config
            gemini_model = self._map_model(config.model)
            
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens or 1024,
            )
            
            model = genai.GenerativeModel(
                model_name=gemini_model,
                generation_config=generation_config
            )
            
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(messages)
            
            logger.info(f"Started Gemini streaming: {gemini_model}")
            
            # Generate streaming response
            response = model.generate_content(prompt, stream=True)
            
            for chunk in response:
                current_time = time.time()
                
                if chunk.text:
                    stream_chunk = StreamChunk(
                        content=chunk.text,
                        timestamp=current_time - start_time,
                        chunk_index=chunk_index,
                        provider=StreamProvider.GEMINI,
                        model=gemini_model,
                        finish_reason=None
                    )
                    
                    chunk_index += 1
                    
                    # Optional delay for testing
                    if config.stream_delay > 0:
                        time.sleep(config.stream_delay)
                    
                    yield stream_chunk
                    
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise


class HuggingFaceStreamer:
    """Advanced Hugging Face streaming client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Hugging Face streaming client."""
        if not HUGGINGFACE_AVAILABLE:
            raise ValueError("Hugging Face Hub not available. Install with: pip install huggingface_hub")
        
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("Hugging Face API key not found. Set HUGGINGFACE_API_KEY environment variable.")
        
        self.client = InferenceClient(token=self.api_key)
        logger.info("Hugging Face streaming client initialized")
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI format messages to HuggingFace prompt."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def _map_model(self, model: str) -> str:
        """Map OpenAI model names to HuggingFace equivalents."""
        model_mapping = {
            "gpt-4o": "microsoft/DialoGPT-large",
            "gpt-4": "microsoft/DialoGPT-large", 
            "gpt-3.5-turbo": "microsoft/DialoGPT-medium",
            "llama": "meta-llama/Llama-2-7b-chat-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1",
        }
        return model_mapping.get(model, "microsoft/DialoGPT-large")
    
    def stream_completion(
        self,
        messages: List[Dict[str, str]],
        config: StreamingConfig
    ) -> Generator[StreamChunk, None, None]:
        """Stream completion from Hugging Face."""
        try:
            start_time = time.time()
            chunk_index = 0
            
            hf_model = self._map_model(config.model)
            prompt = self._convert_messages_to_prompt(messages)
            
            logger.info(f"Started Hugging Face streaming: {hf_model}")
            
            # Try streaming first
            try:
                response = self.client.text_generation(
                    prompt=prompt,
                    model=hf_model,
                    max_new_tokens=config.max_tokens or 512,
                    temperature=config.temperature,
                    stream=True,
                    return_full_text=False
                )
                
                chunk_yielded = False
                for chunk in response:
                    current_time = time.time()
                    
                    if hasattr(chunk, 'token') and chunk.token.text:
                        content = chunk.token.text
                    elif isinstance(chunk, str):
                        content = chunk
                    else:
                        continue
                    
                    stream_chunk = StreamChunk(
                        content=content,
                        timestamp=current_time - start_time,
                        chunk_index=chunk_index,
                        provider=StreamProvider.HUGGINGFACE,
                        model=hf_model,
                        finish_reason=None
                    )
                    
                    chunk_index += 1
                    chunk_yielded = True
                    
                    if config.stream_delay > 0:
                        time.sleep(config.stream_delay)
                    
                    yield stream_chunk
                
                # If no chunks were yielded, fall through to fallback
                if not chunk_yielded:
                    raise Exception("No streaming chunks received")
                    
            except Exception as stream_error:
                logger.warning(f"Streaming failed, using fallback: {stream_error}")
                
                # Fallback to non-streaming
                response = self.client.text_generation(
                    prompt=prompt,
                    model=hf_model,
                    max_new_tokens=config.max_tokens or 512,
                    temperature=config.temperature,
                    return_full_text=False
                )
                
                # Simulate streaming by chunking words
                if isinstance(response, str):
                    words = response.split()
                    for i, word in enumerate(words):
                        content = word + " " if i < len(words) - 1 else word
                        
                        stream_chunk = StreamChunk(
                            content=content,
                            timestamp=time.time() - start_time,
                            chunk_index=i,
                            provider=StreamProvider.HUGGINGFACE,
                            model=hf_model,
                            finish_reason="stop" if i == len(words) - 1 else None
                        )
                        
                        yield stream_chunk
                        
                        if config.stream_delay > 0:
                            time.sleep(config.stream_delay)
                        
        except Exception as e:
            logger.error(f"Hugging Face streaming failed: {e}")
            raise


class AdvancedStreamingSystem:
    """Enterprise-grade multi-provider streaming system."""
    
    def _load_environment_variables(self):
        """Load environment variables from multiple potential locations."""
        # List of potential .env file locations
        potential_env_files = [
            Path(".env"),  # Current directory
            Path("../.env"),  # Parent directory
            Path("../tokens-streaming/.env"),  # Specific known location
            Path("../../.env"),  # Grandparent directory
            Path.home() / ".env",  # Home directory
        ]
        
        env_loaded = False
        for env_file in potential_env_files:
            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded environment variables from: {env_file}")
                env_loaded = True
                break
        
        if not env_loaded:
            logger.warning("No .env file found in standard locations")
            # Try loading from environment without .env file
            load_dotenv()
        
        # Log which environment variables are available (without showing values)
        available_vars = []
        if os.getenv("OPENAI_API_KEY"):
            available_vars.append("OPENAI_API_KEY")
        if os.getenv("GOOGLE_API_KEY"):
            available_vars.append("GOOGLE_API_KEY")
        if os.getenv("HUGGINGFACE_API_KEY"):
            available_vars.append("HUGGINGFACE_API_KEY")
        
        logger.info(f"Available environment variables: {available_vars}")
    
    def __init__(self):
        """Initialize streaming system."""
        # Load environment variables from multiple potential locations
        self._load_environment_variables()
        
        self.streamers = {}
        
        # Initialize available streamers
        try:
            self.streamers[StreamProvider.OPENAI] = OpenAIStreamer()
        except ValueError as e:
            logger.warning(f"OpenAI streamer not available: {e}")
        
        try:
            self.streamers[StreamProvider.GEMINI] = GeminiStreamer()
        except ValueError as e:
            logger.warning(f"Gemini streamer not available: {e}")
        
        try:
            if HUGGINGFACE_AVAILABLE:
                self.streamers[StreamProvider.HUGGINGFACE] = HuggingFaceStreamer()
        except ValueError as e:
            logger.warning(f"Hugging Face streamer not available: {e}")
        
        if not self.streamers:
            raise ValueError("No streaming providers available. Please set API keys.")
        
        logger.info(f"Streaming system initialized with providers: {list(self.streamers.keys())}")
    
    def stream_with_display(
        self,
        messages: List[Dict[str, str]],
        provider: StreamProvider,
        config: StreamingConfig
    ) -> StreamingResponse:
        """
        Stream completion with real-time display.
        
        Args:
            messages: Conversation messages
            provider: Streaming provider to use
            config: Streaming configuration
            
        Returns:
            Complete streaming response with analytics
        """
        if provider not in self.streamers:
            raise ValueError(f"Provider {provider} not available")
        
        start_time = time.time()
        chunks = []
        content_parts = []
        first_chunk_time = None
        
        print(f"\n🤖 Starting {provider.value.upper()} streaming...")
        print(f"📝 Model: {config.model}")
        print(f"🌡️ Temperature: {config.temperature}")
        print("-" * 60)
        
        try:
            # Stream the response
            for chunk in self.streamers[provider].stream_completion(messages, config):
                if first_chunk_time is None:
                    first_chunk_time = chunk.timestamp
                
                chunks.append(chunk)
                content_parts.append(chunk.content)
                
                # Display chunk with timing if enabled
                if config.show_timing:
                    print(f"[{chunk.timestamp:.3f}s] {chunk.content}", end='', flush=True)
                else:
                    print(chunk.content, end='', flush=True)
                
                # Optional delay
                if config.stream_delay > 0:
                    time.sleep(config.stream_delay)
            
            # Calculate analytics
            total_time = time.time() - start_time
            full_content = ''.join(content_parts)
            
            analytics = self._calculate_analytics(
                chunks, full_content, total_time, first_chunk_time or 0
            )
            
            print(f"\n{'-' * 60}")
            print(f"✅ Streaming completed in {total_time:.3f}s")
            print(f"📊 Analytics:")
            print(f"   • Total chunks: {analytics['total_chunks']}")
            print(f"   • First chunk: {analytics['first_chunk_time']:.3f}s")
            print(f"   • Avg chunk time: {analytics['average_chunk_time']:.3f}s")
            print(f"   • Characters/sec: {analytics['characters_per_second']:.1f}")
            print(f"   • Words/sec: {analytics['words_per_second']:.1f}")
            
            return StreamingResponse(
                content=full_content,
                provider=provider,
                model=config.model,
                chunks=chunks if config.save_chunks else [],
                **analytics
            )
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    def _calculate_analytics(
        self,
        chunks: List[StreamChunk],
        content: str,
        total_time: float,
        first_chunk_time: float
    ) -> Dict[str, Any]:
        """Calculate streaming analytics."""
        total_chunks = len(chunks)
        average_chunk_time = total_time / total_chunks if total_chunks > 0 else 0
        characters_per_second = len(content) / total_time if total_time > 0 else 0
        words_per_second = len(content.split()) / total_time if total_time > 0 else 0
        
        return {
            "total_time": total_time,
            "first_chunk_time": first_chunk_time,
            "average_chunk_time": average_chunk_time,
            "total_chunks": total_chunks,
            "characters_per_second": characters_per_second,
            "words_per_second": words_per_second
        }
    
    def compare_providers(
        self,
        messages: List[Dict[str, str]],
        config: StreamingConfig
    ) -> Dict[StreamProvider, StreamingResponse]:
        """
        Compare streaming performance across providers.
        
        Args:
            messages: Conversation messages
            config: Streaming configuration
            
        Returns:
            Dictionary of provider responses
        """
        results = {}
        
        for provider in self.streamers.keys():
            print(f"\n{'='*60}")
            print(f"🔄 Testing {provider.value.upper()} Streaming")
            print('='*60)
            
            try:
                response = self.stream_with_display(messages, provider, config)
                results[provider] = response
                
                # Save individual result
                result_file = f"streaming_result_{provider.value}.json"
                with open(result_file, "w") as f:
                    json.dump(response.to_dict(), f, indent=2)
                
                print(f"💾 Results saved to: {result_file}")
                
            except Exception as e:
                logger.error(f"Provider {provider} failed: {e}")
                print(f"❌ {provider.value} failed: {e}")
        
        return results
    
    def interactive_streaming(self):
        """Start interactive streaming chat."""
        print("🚀 Advanced Streaming Chat System")
        print("💡 Available providers:", [p.value for p in self.streamers.keys()])
        print("📝 Commands: 'quit', 'switch <provider>', 'config'")
        print("=" * 60)
        
        current_provider = list(self.streamers.keys())[0]
        config = StreamingConfig()
        conversation = []
        
        while True:
            try:
                user_input = input(f"\n[{current_provider.value}] You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower().startswith('switch '):
                    provider_name = user_input[7:].strip()
                    try:
                        new_provider = StreamProvider(provider_name)
                        if new_provider in self.streamers:
                            current_provider = new_provider
                            print(f"✅ Switched to {provider_name}")
                        else:
                            print(f"❌ Provider {provider_name} not available")
                    except ValueError:
                        print(f"❌ Unknown provider: {provider_name}")
                    continue
                
                if user_input.lower() == 'config':
                    print(f"Current config: {config}")
                    continue
                
                if not user_input:
                    continue
                
                # Add user message
                conversation.append({"role": "user", "content": user_input})
                
                # Stream response
                response = self.stream_with_display(
                    conversation, current_provider, config
                )
                
                # Add assistant response
                conversation.append({"role": "assistant", "content": response.content})
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive streaming error: {e}")
                print(f"❌ Error: {e}")


def demo_examples():
    """Demonstrate streaming capabilities."""
    try:
        # Initialize streaming system
        streaming = AdvancedStreamingSystem()
        
        print("🚀 Advanced AI Streaming System")
        print("=" * 60)
        
        # Example messages
        messages = [
            {"role": "user", "content": "Give me the list of first 10 presidents of the United States of America."}
        ]
        
        # Configuration
        config = StreamingConfig(
            model="gpt-4o",
            temperature=0,
            max_tokens=500,
            show_timing=True,
            save_chunks=True
        )
        
        # Single provider demo
        if StreamProvider.OPENAI in streaming.streamers:
            print("\n📝 Single Provider Demo (OpenAI)")
            response = streaming.stream_with_display(messages, StreamProvider.OPENAI, config)
            
            # Save results
            with open("streaming_demo.json", "w") as f:
                json.dump(response.to_dict(), f, indent=2)
            print("\n💾 Demo results saved to: streaming_demo.json")
        
        # Provider comparison
        if len(streaming.streamers) > 1:
            print(f"\n🔄 Provider Comparison Demo")
            results = streaming.compare_providers(messages, config)
            
            # Comparison summary
            print(f"\n📊 Performance Comparison:")
            for provider, response in results.items():
                print(f"  {provider.value}:")
                print(f"    • Total time: {response.total_time:.3f}s")
                print(f"    • First chunk: {response.first_chunk_time:.3f}s")
                print(f"    • Chars/sec: {response.characters_per_second:.1f}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo error: {e}")


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive streaming mode
        streaming = AdvancedStreamingSystem()
        streaming.interactive_streaming()
    else:
        # Demo mode
        demo_examples()


if __name__ == '__main__':
    main()
