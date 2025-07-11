"""
Advanced Local AI Model Completion System.

This module provides an enterprise-grade system for interacting with local AI models
through OpenAI-compatible APIs (like LM Studio, Ollama, vLLM, etc.) with comprehensive
monitoring, error handling, model management, and performance analytics.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import openai
import requests
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_ai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported local AI providers."""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"
    VLLM = "vllm"
    TEXT_GENERATION_WEBUI = "text_generation_webui"
    GENERIC_OPENAI = "generic_openai"


@dataclass
class ServerConfig:
    """Configuration for local AI server."""
    base_url: str = "http://localhost:1234/v1/"
    api_key: Optional[str] = None
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    provider: ModelProvider = ModelProvider.LM_STUDIO
    
    def __post_init__(self):
        """Validate server configuration."""
        try:
            parsed_url = urlparse(self.base_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid base_url format")
        except Exception:
            raise ValueError("Invalid base_url format")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass
class CompletionConfig:
    """Configuration for completion requests."""
    model: str = "phi-4"
    temperature: float = 0.5
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False
    
    def __post_init__(self):
        """Validate completion configuration."""
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1")


@dataclass
class Message:
    """Structured message for conversations."""
    role: str
    content: str
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize message with timestamp."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        
        valid_roles = ["system", "user", "assistant", "function"]
        if self.role not in valid_roles:
            raise ValueError(f"Invalid role '{self.role}'. Must be one of: {valid_roles}")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to OpenAI API format."""
        return {"role": self.role, "content": self.content}


@dataclass
class CompletionResponse:
    """Structured response from completion."""
    content: str
    model: str
    usage: Dict[str, Any]
    response_time: float
    finish_reason: Optional[str] = None
    server_info: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "response_time": self.response_time,
            "finish_reason": self.finish_reason,
            "server_info": self.server_info,
            "timestamp": datetime.now().isoformat()
        }


class ServerHealthMonitor:
    """Monitor local AI server health and performance."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.health_history: List[Dict[str, Any]] = []
    
    def check_server_health(self) -> Dict[str, Any]:
        """Check if the server is healthy and responsive."""
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # Try to get models list as health check
            health_url = self.config.base_url.rstrip('/') + '/models'
            response = requests.get(health_url, timeout=10)
            
            response_time = time.time() - start_time
            health_data["response_time"] = response_time
            
            if response.status_code == 200:
                health_data["status"] = "healthy"
                try:
                    models_data = response.json()
                    health_data["available_models"] = len(models_data.get("data", []))
                except json.JSONDecodeError:
                    health_data["available_models"] = "unknown"
            else:
                health_data["status"] = "unhealthy"
                health_data["error"] = f"HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            health_data["status"] = "unreachable"
            health_data["error"] = "Connection refused"
        except requests.exceptions.Timeout:
            health_data["status"] = "timeout"
            health_data["error"] = "Request timeout"
        except Exception as e:
            health_data["status"] = "error"
            health_data["error"] = str(e)
        
        self.health_history.append(health_data)
        
        # Keep only last 100 health checks
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health_data
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics."""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_checks = self.health_history[-10:]  # Last 10 checks
        healthy_count = sum(1 for check in recent_checks if check["status"] == "healthy")
        
        avg_response_time = None
        response_times = [
            check["response_time"] for check in recent_checks 
            if check["response_time"] is not None
        ]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
        
        return {
            "overall_status": "healthy" if healthy_count >= 7 else "unstable",
            "success_rate": healthy_count / len(recent_checks),
            "average_response_time": avg_response_time,
            "last_check": recent_checks[-1] if recent_checks else None,
            "total_checks": len(self.health_history)
        }


class ModelManager:
    """Manage available models on the local server."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.cached_models: Optional[List[Dict[str, Any]]] = None
        self.cache_time: Optional[datetime] = None
        self.cache_duration = 300  # 5 minutes
    
    def get_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get list of available models from the server."""
        # Check cache
        if (not force_refresh and 
            self.cached_models is not None and 
            self.cache_time is not None and
            (datetime.now() - self.cache_time).seconds < self.cache_duration):
            return self.cached_models
        
        try:
            models_url = self.config.base_url.rstrip('/') + '/models'
            response = requests.get(models_url, timeout=30)
            response.raise_for_status()
            
            models_data = response.json()
            models = models_data.get("data", [])
            
            # Cache the results
            self.cached_models = models
            self.cache_time = datetime.now()
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"Failed to fetch models: {e}")
            return self.cached_models or []
    
    def find_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Find a specific model by name or partial match."""
        models = self.get_available_models()
        
        # Exact match first
        for model in models:
            if model.get("id") == model_name:
                return model
        
        # Partial match
        for model in models:
            if model_name.lower() in model.get("id", "").lower():
                return model
        
        return None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        model = self.find_model(model_name)
        if not model:
            return {"status": "not_found", "message": f"Model '{model_name}' not found"}
        
        return {
            "status": "found",
            "id": model.get("id"),
            "object": model.get("object"),
            "created": model.get("created"),
            "owned_by": model.get("owned_by"),
            "permission": model.get("permission", [])
        }


class LocalAIClient:
    """Advanced client for local AI model interactions."""
    
    def __init__(self, server_config: ServerConfig):
        """Initialize the local AI client."""
        self.server_config = server_config
        self.health_monitor = ServerHealthMonitor(server_config)
        self.model_manager = ModelManager(server_config)
        
        # Initialize OpenAI client
        self.client = openai.Client(
            base_url=server_config.base_url,
            api_key=server_config.api_key or "dummy-key"
        )
        
        logger.info(f"Local AI client initialized for {server_config.base_url}")
    
    def complete(
        self,
        messages: List[Message],
        config: CompletionConfig
    ) -> CompletionResponse:
        """
        Create a chat completion with retry logic.
        
        Args:
            messages: List of conversation messages
            config: Completion configuration
            
        Returns:
            Structured completion response
        """
        # Check server health first
        health = self.health_monitor.check_server_health()
        if health["status"] != "healthy":
            logger.warning(f"Server health check failed: {health}")
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(1, self.server_config.max_retries + 1):
            try:
                logger.info(f"Completion attempt {attempt}/{self.server_config.max_retries}")
                
                # Convert messages to OpenAI format
                openai_messages = [msg.to_dict() for msg in messages]
                
                # Prepare request parameters
                params = {
                    "model": config.model,
                    "messages": openai_messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "frequency_penalty": config.frequency_penalty,
                    "presence_penalty": config.presence_penalty,
                    "stream": config.stream
                }
                
                if config.stop:
                    params["stop"] = config.stop
                
                # Make the request
                response = self.client.chat.completions.create(**params)
                
                response_time = time.time() - start_time
                
                # Extract response data
                content = response.choices[0].message.content or ""
                usage = response.usage.model_dump() if response.usage else {}
                finish_reason = response.choices[0].finish_reason
                
                logger.info(f"Completion successful in {response_time:.3f}s")
                
                return CompletionResponse(
                    content=content,
                    model=config.model,
                    usage=usage,
                    response_time=response_time,
                    finish_reason=finish_reason,
                    server_info={"base_url": self.server_config.base_url},
                    raw_response=response
                )
                
            except Exception as e:
                last_error = e
                logger.error(f"Completion attempt {attempt} failed: {e}")
                
                if attempt < self.server_config.max_retries:
                    time.sleep(self.server_config.retry_delay)
                    continue
                else:
                    raise last_error
    
    def stream_complete(
        self,
        messages: List[Message],
        config: CompletionConfig
    ):
        """Stream completion with real-time output."""
        config.stream = True
        
        # Convert messages to OpenAI format
        openai_messages = [msg.to_dict() for msg in messages]
        
        params = {
            "model": config.model,
            "messages": openai_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "stream": True
        }
        
        start_time = time.time()
        collected_content = []
        
        try:
            response = self.client.chat.completions.create(**params)
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content.append(content)
                    print(content, end='', flush=True)
            
            print()  # New line after streaming
            
            total_time = time.time() - start_time
            full_content = ''.join(collected_content)
            
            return CompletionResponse(
                content=full_content,
                model=config.model,
                usage={"estimated_tokens": len(full_content.split())},
                response_time=total_time,
                server_info={"base_url": self.server_config.base_url}
            )
            
        except Exception as e:
            logger.error(f"Streaming completion failed: {e}")
            raise


class AdvancedLocalAISystem:
    """Enterprise-grade local AI system with comprehensive features."""
    
    def __init__(self, server_config: Optional[ServerConfig] = None):
        """Initialize the local AI system."""
        load_dotenv()
        
        self.server_config = server_config or ServerConfig()
        self.client = LocalAIClient(self.server_config)
        self.conversation_history: List[Message] = []
        
        logger.info("Advanced Local AI System initialized")
    
    def create_conversation(self, system_prompt: str = "You are a helpful assistant.") -> List[Message]:
        """Create a new conversation with system prompt."""
        return [Message(role="system", content=system_prompt)]
    
    def add_message(self, conversation: List[Message], role: str, content: str) -> List[Message]:
        """Add a message to the conversation."""
        conversation.append(Message(role=role, content=content))
        return conversation
    
    def quick_completion(
        self,
        prompt: str,
        model: str = "phi-4",
        temperature: float = 0.5,
        max_tokens: int = 1024
    ) -> str:
        """Quick completion for simple prompts."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=prompt)
        ]
        
        config = CompletionConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = self.client.complete(messages, config)
        return response.content
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        print("🤖 Advanced Local AI Chat System")
        print(f"🔗 Connected to: {self.server_config.base_url}")
        print("📝 Commands: 'quit', 'health', 'models', 'clear', 'save', 'stream on/off'")
        print("=" * 60)
        
        # Check initial health
        health = self.client.health_monitor.check_server_health()
        print(f"🏥 Server Status: {health['status']}")
        
        conversation = self.create_conversation()
        config = CompletionConfig()
        stream_mode = False
        
        while True:
            try:
                user_input = input(f"\n[{config.model}] You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'health':
                    health_summary = self.client.health_monitor.get_health_summary()
                    print(f"📊 Health Summary: {json.dumps(health_summary, indent=2)}")
                    continue
                
                if user_input.lower() == 'models':
                    models = self.client.model_manager.get_available_models(force_refresh=True)
                    print(f"🤖 Available Models:")
                    for model in models[:10]:  # Show first 10
                        print(f"  • {model.get('id', 'Unknown')}")
                    if len(models) > 10:
                        print(f"  ... and {len(models) - 10} more")
                    continue
                
                if user_input.lower() == 'clear':
                    conversation = self.create_conversation()
                    print("🗑️ Conversation cleared")
                    continue
                
                if user_input.lower().startswith('stream '):
                    mode = user_input[7:].strip().lower()
                    stream_mode = mode == 'on'
                    print(f"🌊 Streaming mode: {'ON' if stream_mode else 'OFF'}")
                    continue
                
                if user_input.lower() == 'save':
                    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    conversation_data = [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                        }
                        for msg in conversation
                    ]
                    with open(filename, "w") as f:
                        json.dump(conversation_data, f, indent=2)
                    print(f"💾 Conversation saved to: {filename}")
                    continue
                
                if not user_input:
                    continue
                
                # Add user message
                conversation = self.add_message(conversation, "user", user_input)
                
                # Get AI response
                if stream_mode:
                    print(f"\n🤖 Assistant: ", end='')
                    response = self.client.stream_complete(conversation, config)
                else:
                    response = self.client.complete(conversation, config)
                    print(f"\n🤖 Assistant: {response.content}")
                
                # Add AI response to conversation
                conversation = self.add_message(conversation, "assistant", response.content)
                
                print(f"⏱️ Response time: {response.response_time:.3f}s")
                if response.usage:
                    tokens = response.usage.get('total_tokens', 'N/A')
                    print(f"📊 Tokens: {tokens}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Chat error: {e}")
                print(f"❌ Error: {e}")
    
    def benchmark_models(self, test_prompt: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Benchmark available models with a test prompt."""
        models = self.client.model_manager.get_available_models()
        if not models:
            return {"error": "No models available"}
        
        results = {}
        config = CompletionConfig(max_tokens=200, temperature=0.1)
        
        print(f"🏁 Benchmarking {len(models)} models...")
        
        for model in models[:5]:  # Test first 5 models
            model_name = model.get("id", "unknown")
            print(f"Testing {model_name}...")
            
            try:
                config.model = model_name
                messages = [
                    Message(role="system", content="You are a helpful assistant."),
                    Message(role="user", content=test_prompt)
                ]
                
                response = self.client.complete(messages, config)
                
                results[model_name] = {
                    "success": True,
                    "response_time": response.response_time,
                    "content_length": len(response.content),
                    "finish_reason": response.finish_reason,
                    "usage": response.usage
                }
                
                print(f"✅ {model_name}: {response.response_time:.3f}s")
                
            except Exception as e:
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ {model_name}: {e}")
        
        return results


def demo_examples():
    """Demonstrate local AI system capabilities."""
    try:
        # Initialize system
        config = ServerConfig(
            base_url="http://localhost:1234/v1/",
            provider=ModelProvider.LM_STUDIO
        )
        
        system = AdvancedLocalAISystem(config)
        
        print("🚀 Advanced Local AI System Demo")
        print("=" * 60)
        
        # Example 1: Server health check
        print("\n🏥 Example 1: Server Health Check")
        health = system.client.health_monitor.check_server_health()
        print(f"Status: {health['status']}")
        print(f"Response time: {health.get('response_time', 'N/A')}")
        
        # Example 2: Available models
        print(f"\n🤖 Example 2: Available Models")
        models = system.client.model_manager.get_available_models()
        print(f"Found {len(models)} models")
        for model in models[:3]:
            print(f"  • {model.get('id', 'Unknown')}")
        
        # Example 3: Quick completion
        if models:
            print(f"\n💬 Example 3: Quick Completion")
            try:
                response = system.quick_completion(
                    "What is the fastest car in the world?",
                    model=models[0].get("id", "phi-4")
                )
                print(f"Response: {response[:200]}...")
            except Exception as e:
                print(f"❌ Completion failed: {e}")
        
        # Example 4: Model benchmark
        if len(models) > 1:
            print(f"\n🏁 Example 4: Model Benchmark")
            benchmark_results = system.benchmark_models("Explain quantum computing in one sentence.")
            
            # Save benchmark results
            with open("model_benchmark.json", "w") as f:
                json.dump(benchmark_results, f, indent=2)
            print("💾 Benchmark results saved to: model_benchmark.json")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo error: {e}")


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive chat mode
        system = AdvancedLocalAISystem()
        system.interactive_chat()
    elif len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Benchmark mode
        system = AdvancedLocalAISystem()
        results = system.benchmark_models()
        print(json.dumps(results, indent=2))
    else:
        # Demo mode
        demo_examples()


if __name__ == '__main__':
    main()