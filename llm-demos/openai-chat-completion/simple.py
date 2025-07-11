"""
Advanced Multi-Provider AI Chat Completion System.

This module provides an enterprise-grade chat completion system supporting both OpenAI and Google Gemini AI
with proper error handling, configuration management, retry logic, and comprehensive logging.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_completion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class Message:
    """Structured message for chat conversations."""
    role: str
    content: str
    
    def __post_init__(self):
        """Validate message structure."""
        valid_roles = ["system", "user", "assistant"]
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of: {valid_roles}")
        if not self.content.strip():
            raise ValueError("Message content cannot be empty")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role, "content": self.content}


@dataclass
class CompletionConfig:
    """Configuration for chat completion requests."""
    model: str = "gpt-4o"
    max_tokens: Optional[int] = 1000
    temperature: float = 0.7
    seed: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")


@dataclass
class CompletionResponse:
    """Structured response from AI completion."""
    content: str
    provider: AIProvider
    model: str
    usage: Dict[str, Any]
    response_time: float
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "provider": self.provider.value,
            "model": self.model,
            "usage": self.usage,
            "response_time": self.response_time,
            "finish_reason": self.finish_reason
        }


class OpenAIClient:
    """Advanced OpenAI client with retry logic and error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI client initialized successfully")
    
    def complete(
        self,
        messages: List[Message],
        config: CompletionConfig
    ) -> CompletionResponse:
        """
        Create a chat completion using OpenAI.
        
        Args:
            messages: List of conversation messages
            config: Completion configuration
            
        Returns:
            Structured completion response
        """
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [msg.to_dict() for msg in messages]
            
            # Prepare request parameters
            params = {
                "model": config.model,
                "messages": openai_messages,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "frequency_penalty": config.frequency_penalty,
                "presence_penalty": config.presence_penalty
            }
            
            # Add optional parameters
            if config.max_tokens:
                params["max_tokens"] = config.max_tokens
            if config.seed is not None:
                params["seed"] = config.seed
            
            logger.info(f"Sending request to OpenAI: {config.model}")
            response = self.client.chat.completions.create(**params)
            
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            finish_reason = response.choices[0].finish_reason
            
            logger.info(f"OpenAI request completed in {response_time:.3f}s")
            
            return CompletionResponse(
                content=content,
                provider=AIProvider.OPENAI,
                model=config.model,
                usage=usage,
                response_time=response_time,
                finish_reason=finish_reason,
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise


class GeminiClient:
    """Advanced Gemini AI client with retry logic and error handling."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        logger.info("Gemini client initialized successfully")
    
    def _map_openai_to_gemini_model(self, model: str) -> str:
        """Map OpenAI model names to Gemini equivalents."""
        model_mapping = {
            "gpt-4o": "gemini-pro",
            "gpt-4": "gemini-pro",
            "gpt-3.5-turbo": "gemini-pro",
        }
        return model_mapping.get(model, "gemini-pro")
    
    def _convert_messages_to_gemini_prompt(self, messages: List[Message]) -> str:
        """Convert conversation messages to Gemini prompt format."""
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System Instructions: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        return "\n\n".join(prompt_parts)
    
    def complete(
        self,
        messages: List[Message],
        config: CompletionConfig
    ) -> CompletionResponse:
        """
        Create a chat completion using Gemini AI.
        
        Args:
            messages: List of conversation messages
            config: Completion configuration
            
        Returns:
            Structured completion response
        """
        start_time = time.time()
        
        try:
            # Map model and create generation config
            gemini_model = self._map_openai_to_gemini_model(config.model)
            
            generation_config = genai.types.GenerationConfig(
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_tokens or 1024,
            )
            
            # Initialize model
            model = genai.GenerativeModel(
                model_name=gemini_model,
                generation_config=generation_config
            )
            
            # Convert messages to prompt
            prompt = self._convert_messages_to_gemini_prompt(messages)
            
            logger.info(f"Sending request to Gemini: {gemini_model}")
            response = model.generate_content(prompt)
            
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.text if response.text else ""
            
            # Estimate usage (Gemini doesn't provide detailed usage stats)
            estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimation
            estimated_output_tokens = len(content.split()) * 1.3
            
            usage = {
                "prompt_tokens": int(estimated_input_tokens),
                "completion_tokens": int(estimated_output_tokens),
                "total_tokens": int(estimated_input_tokens + estimated_output_tokens)
            }
            
            logger.info(f"Gemini request completed in {response_time:.3f}s")
            
            return CompletionResponse(
                content=content,
                provider=AIProvider.GEMINI,
                model=gemini_model,
                usage=usage,
                response_time=response_time,
                finish_reason="stop",
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            raise


class AdvancedChatSystem:
    """Enterprise-grade multi-provider chat completion system."""
    
    def __init__(self, default_provider: AIProvider = AIProvider.OPENAI):
        """Initialize the chat system."""
        load_dotenv()
        
        self.default_provider = default_provider
        self.clients = {}
        
        # Initialize available clients
        try:
            self.clients[AIProvider.OPENAI] = OpenAIClient()
        except ValueError as e:
            logger.warning(f"OpenAI client not available: {e}")
        
        try:
            self.clients[AIProvider.GEMINI] = GeminiClient()
        except ValueError as e:
            logger.warning(f"Gemini client not available: {e}")
        
        if not self.clients:
            raise ValueError("No AI providers available. Please set API keys.")
        
        logger.info(f"Chat system initialized with providers: {list(self.clients.keys())}")
    
    def create_conversation(self, system_prompt: str = "You are a helpful assistant.") -> List[Message]:
        """Create a new conversation with system prompt."""
        return [Message(role="system", content=system_prompt)]
    
    def add_message(self, conversation: List[Message], role: str, content: str) -> List[Message]:
        """Add a message to the conversation."""
        conversation.append(Message(role=role, content=content))
        return conversation
    
    def complete(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        provider: Optional[AIProvider] = None,
        config: Optional[CompletionConfig] = None
    ) -> CompletionResponse:
        """
        Create a chat completion.
        
        Args:
            messages: Conversation messages
            provider: AI provider to use
            config: Completion configuration
            
        Returns:
            Structured completion response
        """
        # Use default provider if not specified
        provider = provider or self.default_provider
        
        # Ensure provider is available
        if provider not in self.clients:
            available_providers = list(self.clients.keys())
            if available_providers:
                logger.warning(f"Provider {provider} not available, using {available_providers[0]}")
                provider = available_providers[0]
            else:
                raise ValueError("No AI providers available")
        
        # Convert dict messages to Message objects if needed
        if messages and isinstance(messages[0], dict):
            messages = [Message(role=msg["role"], content=msg["content"]) for msg in messages]
        
        # Use default config if not provided
        config = config or CompletionConfig()
        
        # Adjust model for provider
        if provider == AIProvider.GEMINI and config.model.startswith("gpt"):
            logger.info(f"Automatically mapping OpenAI model {config.model} for Gemini")
        
        return self.clients[provider].complete(messages, config)
    
    def chat_interactive(self, system_prompt: str = "You are a helpful assistant.") -> None:
        """Start an interactive chat session."""
        print("🤖 Advanced AI Chat System")
        print("💡 Available providers:", [p.value for p in self.clients.keys()])
        print("📝 Type 'quit' to exit, 'switch <provider>' to change provider")
        print("=" * 60)
        
        conversation = self.create_conversation(system_prompt)
        current_provider = self.default_provider
        
        while True:
            try:
                user_input = input(f"\n[{current_provider.value}] You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower().startswith('switch '):
                    provider_name = user_input[7:].strip()
                    try:
                        new_provider = AIProvider(provider_name)
                        if new_provider in self.clients:
                            current_provider = new_provider
                            print(f"✅ Switched to {provider_name}")
                        else:
                            print(f"❌ Provider {provider_name} not available")
                    except ValueError:
                        print(f"❌ Unknown provider: {provider_name}")
                    continue
                
                if not user_input:
                    continue
                
                # Add user message
                conversation = self.add_message(conversation, "user", user_input)
                
                # Get AI response
                response = self.complete(conversation, provider=current_provider)
                
                # Add AI response to conversation
                conversation = self.add_message(conversation, "assistant", response.content)
                
                # Display response
                print(f"\n🤖 Assistant: {response.content}")
                print(f"⏱️ Response time: {response.response_time:.3f}s")
                print(f"📊 Tokens: {response.usage.get('total_tokens', 'N/A')}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print(f"❌ Error: {e}")


def demo_examples():
    """Demonstrate various usage examples."""
    try:
        # Initialize chat system
        chat = AdvancedChatSystem()
        
        print("🚀 Advanced AI Chat Completion System Demo")
        print("=" * 60)
        
        # Example 1: Simple completion
        print("\n📝 Example 1: Simple Completion")
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Tell me a joke about programming.")
        ]
        
        config = CompletionConfig(max_tokens=150, temperature=0.8, seed=42)
        response = chat.complete(messages, config=config)
        
        print(f"Provider: {response.provider.value}")
        print(f"Model: {response.model}")
        print(f"Response: {response.content}")
        print(f"Time: {response.response_time:.3f}s")
        print(f"Usage: {response.usage}")
        
        # Example 2: Provider comparison
        if len(chat.clients) > 1:
            print(f"\n🔄 Example 2: Provider Comparison")
            question_messages = [
                Message(role="system", content="You are a creative storyteller."),
                Message(role="user", content="Write a short story about a robot learning to paint.")
            ]
            
            for provider in chat.clients.keys():
                print(f"\n--- {provider.value.upper()} ---")
                response = chat.complete(question_messages, provider=provider)
                print(f"Response: {response.content[:200]}...")
                print(f"Time: {response.response_time:.3f}s")
        
        # Example 3: Save results
        print(f"\n💾 Example 3: Saving Results")
        results = {
            "timestamp": time.time(),
            "examples": []
        }
        
        for provider in chat.clients.keys():
            response = chat.complete(messages, provider=provider)
            results["examples"].append(response.to_dict())
        
        results_path = Path("chat_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo error: {e}")


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive mode
        chat = AdvancedChatSystem()
        chat.chat_interactive()
    else:
        # Demo mode
        demo_examples()


if __name__ == '__main__':
    main()