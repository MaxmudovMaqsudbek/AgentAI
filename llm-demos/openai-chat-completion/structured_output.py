"""
Advanced Multi-Provider AI Structured Output System.

This module provides an enterprise-grade structured output system supporting both OpenAI and Google Gemini AI
with Pydantic schema validation, error handling, retry logic, and comprehensive output formatting.
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, validator, ValidationError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('structured_output.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class OutputProvider(Enum):
    """Supported structured output providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class StructuredConfig:
    """Configuration for structured output requests."""
    model: str = "gpt-4o-2024-08-06"
    temperature: float = 0.1
    max_tokens: Optional[int] = 2000
    max_retries: int = 3
    retry_delay: float = 1.0
    strict_validation: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


# Enhanced Pydantic Models with Validation
class Person(BaseModel):
    """Enhanced person model with comprehensive validation."""
    name: str = Field(..., min_length=1, max_length=50, description="Person's first name")
    surname: str = Field(..., min_length=1, max_length=50, description="Person's last name")
    date_of_birth: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    favourite_movies: List[str] = Field(..., min_items=1, max_items=10, description="List of favorite movies")
    age: Optional[int] = Field(None, ge=0, le=150, description="Person's age (calculated or provided)")
    email: Optional[str] = Field(None, description="Person's email address")
    
    @validator('date_of_birth')
    def validate_date_of_birth(cls, v):
        """Validate date of birth format and reasonableness."""
        try:
            birth_date = datetime.strptime(v, '%Y-%m-%d').date()
            today = date.today()
            
            if birth_date > today:
                raise ValueError("Date of birth cannot be in the future")
            
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            if age > 150:
                raise ValueError("Age cannot exceed 150 years")
                
            return v
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError("Date must be in YYYY-MM-DD format")
            raise e
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email format if provided."""
        if v and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError("Invalid email format")
        return v
    
    @property
    def full_name(self) -> str:
        """Get full name."""
        return f"{self.name} {self.surname}"
    
    def calculate_age(self) -> int:
        """Calculate current age from date of birth."""
        birth_date = datetime.strptime(self.date_of_birth, '%Y-%m-%d').date()
        today = date.today()
        return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


class Company(BaseModel):
    """Company model for business-related structured output."""
    name: str = Field(..., min_length=1, max_length=100, description="Company name")
    industry: str = Field(..., description="Industry sector")
    founded_year: int = Field(..., ge=1800, le=2025, description="Year company was founded")
    employees: int = Field(..., ge=1, description="Number of employees")
    headquarters: str = Field(..., description="Company headquarters location")
    website: Optional[str] = Field(None, description="Company website URL")
    
    @validator('website')
    def validate_website(cls, v):
        """Validate website URL format."""
        if v and not re.match(r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', v):
            raise ValueError("Invalid website URL format")
        return v


class Product(BaseModel):
    """Product model for e-commerce structured output."""
    name: str = Field(..., min_length=1, max_length=100, description="Product name")
    category: str = Field(..., description="Product category")
    price: float = Field(..., ge=0, description="Product price")
    description: str = Field(..., min_length=10, description="Product description")
    in_stock: bool = Field(True, description="Whether product is in stock")
    rating: Optional[float] = Field(None, ge=1, le=5, description="Product rating (1-5)")
    
    @validator('price')
    def validate_price(cls, v):
        """Validate price is reasonable."""
        if v > 1000000:
            raise ValueError("Price seems unreasonably high")
        return round(v, 2)


class Clients(BaseModel):
    """Collection of persons."""
    persons: List[Person] = Field(..., min_items=1, description="List of persons")
    total_count: Optional[int] = Field(None, description="Total count of persons")
    
    @validator('total_count', always=True)
    def set_total_count(cls, v, values):
        """Automatically set total count if not provided."""
        if 'persons' in values:
            return len(values['persons'])
        return v


class Companies(BaseModel):
    """Collection of companies."""
    companies: List[Company] = Field(..., min_items=1, description="List of companies")
    total_count: Optional[int] = Field(None, description="Total count of companies")
    
    @validator('total_count', always=True)
    def set_total_count(cls, v, values):
        """Automatically set total count if not provided."""
        if 'companies' in values:
            return len(values['companies'])
        return v


class Products(BaseModel):
    """Collection of products."""
    products: List[Product] = Field(..., min_items=1, description="List of products")
    total_count: Optional[int] = Field(None, description="Total count of products")
    
    @validator('total_count', always=True)
    def set_total_count(cls, v, values):
        """Automatically set total count if not provided."""
        if 'products' in values:
            return len(values['products'])
        return v


@dataclass
class StructuredResponse:
    """Response from structured output generation."""
    data: BaseModel
    provider: OutputProvider
    model: str
    response_time: float
    attempt_count: int
    raw_response: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "data": self.data.dict(),
            "provider": self.provider.value,
            "model": self.model,
            "response_time": self.response_time,
            "attempt_count": self.attempt_count,
            "validation_errors": self.validation_errors
        }


class BaseStructuredClient(ABC):
    """Abstract base class for structured output clients."""
    
    @abstractmethod
    def generate_structured_output(
        self,
        prompt: str,
        response_model: Type[T],
        config: StructuredConfig
    ) -> T:
        """Generate structured output from prompt."""
        pass


class OpenAIStructuredClient(BaseStructuredClient):
    """Advanced OpenAI structured output client."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info("OpenAI structured client initialized")
    
    def generate_structured_output(
        self,
        prompt: str,
        response_model: Type[T],
        config: StructuredConfig
    ) -> T:
        """Generate structured output using OpenAI's beta parse API."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates accurate structured data."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            logger.info(f"Generating structured output with OpenAI: {response_model.__name__}")
            
            completion = self.client.beta.chat.completions.parse(
                model=config.model,
                messages=messages,
                response_format=response_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            parsed_data = completion.choices[0].message.parsed
            
            if parsed_data is None:
                raise ValueError("Failed to parse structured output from OpenAI")
            
            logger.info("OpenAI structured output generated successfully")
            return parsed_data
            
        except Exception as e:
            logger.error(f"OpenAI structured output failed: {e}")
            raise


class GeminiStructuredClient(BaseStructuredClient):
    """Advanced Gemini structured output client with JSON parsing."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        logger.info("Gemini structured client initialized")
    
    def _create_schema_prompt(self, response_model: Type[BaseModel]) -> str:
        """Create a detailed schema prompt for Gemini."""
        schema = response_model.schema()
        
        prompt_parts = [
            f"Generate a JSON response that strictly follows this schema for {response_model.__name__}:",
            f"\nSchema: {json.dumps(schema, indent=2)}",
            "\nIMPORTANT REQUIREMENTS:",
            "1. Return ONLY valid JSON that matches the schema exactly",
            "2. Include all required fields",
            "3. Follow all field constraints and validation rules",
            "4. Use realistic and diverse data",
            "5. Ensure dates are in YYYY-MM-DD format",
            "6. Do not include any text outside the JSON response"
        ]
        
        return "\n".join(prompt_parts)
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from Gemini response."""
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'\s*```', '', response_text)
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group().strip()
        
        # If no JSON found, try the entire response
        return response_text.strip()
    
    def _map_model(self, model: str) -> str:
        """Map OpenAI model names to Gemini equivalents."""
        model_mapping = {
            "gpt-4o-2024-08-06": "gemini-pro",
            "gpt-4o": "gemini-pro",
            "gpt-4": "gemini-pro",
        }
        return model_mapping.get(model, "gemini-pro")
    
    def generate_structured_output(
        self,
        prompt: str,
        response_model: Type[T],
        config: StructuredConfig
    ) -> T:
        """Generate structured output using Gemini with JSON parsing."""
        # Map model and create generation config
        gemini_model = self._map_model(config.model)
        
        generation_config = genai.types.GenerationConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens or 2000,
        )
        
        model = genai.GenerativeModel(
            model_name=gemini_model,
            generation_config=generation_config
        )
        
        # Create enhanced prompt with schema
        schema_prompt = self._create_schema_prompt(response_model)
        full_prompt = f"{schema_prompt}\n\nUser Request: {prompt}"
        
        try:
            logger.info(f"Generating structured output with Gemini: {response_model.__name__}")
            
            response = model.generate_content(full_prompt)
            
            if not response.text:
                raise ValueError("Empty response from Gemini")
            
            # Extract and parse JSON
            json_text = self._extract_json_from_response(response.text)
            
            try:
                json_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from Gemini response: {e}")
                logger.debug(f"Raw response: {response.text}")
                raise ValueError(f"Invalid JSON response from Gemini: {e}")
            
            # Validate with Pydantic
            parsed_data = response_model.parse_obj(json_data)
            
            logger.info("Gemini structured output generated successfully")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Gemini structured output failed: {e}")
            raise


class AdvancedStructuredSystem:
    """Enterprise-grade multi-provider structured output system."""
    
    def __init__(self):
        """Initialize structured output system."""
        load_dotenv()
        
        self.clients = {}
        
        # Initialize available clients
        try:
            self.clients[OutputProvider.OPENAI] = OpenAIStructuredClient()
        except ValueError as e:
            logger.warning(f"OpenAI client not available: {e}")
        
        try:
            self.clients[OutputProvider.GEMINI] = GeminiStructuredClient()
        except ValueError as e:
            logger.warning(f"Gemini client not available: {e}")
        
        if not self.clients:
            raise ValueError("No structured output providers available. Please set API keys.")
        
        logger.info(f"Structured system initialized with providers: {list(self.clients.keys())}")
    
    def generate_with_retry(
        self,
        prompt: str,
        response_model: Type[T],
        provider: OutputProvider,
        config: StructuredConfig
    ) -> StructuredResponse:
        """
        Generate structured output with retry logic.
        
        Args:
            prompt: Generation prompt
            response_model: Pydantic model class
            provider: Output provider to use
            config: Generation configuration
            
        Returns:
            Structured response with metadata
        """
        if provider not in self.clients:
            raise ValueError(f"Provider {provider} not available")
        
        start_time = time.time()
        validation_errors = []
        
        for attempt in range(1, config.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{config.max_retries} for {provider.value}")
                
                data = self.clients[provider].generate_structured_output(
                    prompt, response_model, config
                )
                
                response_time = time.time() - start_time
                
                return StructuredResponse(
                    data=data,
                    provider=provider,
                    model=config.model,
                    response_time=response_time,
                    attempt_count=attempt,
                    validation_errors=validation_errors
                )
                
            except ValidationError as e:
                error_msg = f"Validation failed on attempt {attempt}: {e}"
                logger.warning(error_msg)
                validation_errors.append(error_msg)
                
                if attempt < config.max_retries:
                    time.sleep(config.retry_delay)
                    continue
                else:
                    raise ValueError(f"All {config.max_retries} attempts failed validation")
                    
            except Exception as e:
                error_msg = f"Generation failed on attempt {attempt}: {e}"
                logger.error(error_msg)
                
                if attempt < config.max_retries:
                    time.sleep(config.retry_delay)
                    continue
                else:
                    raise
    
    def compare_providers(
        self,
        prompt: str,
        response_model: Type[T],
        config: StructuredConfig
    ) -> Dict[OutputProvider, StructuredResponse]:
        """Compare structured output across providers."""
        results = {}
        
        for provider in self.clients.keys():
            print(f"\n{'='*60}")
            print(f"🔄 Testing {provider.value.upper()} Structured Output")
            print('='*60)
            
            try:
                response = self.generate_with_retry(prompt, response_model, provider, config)
                results[provider] = response
                
                print(f"✅ {provider.value} completed in {response.response_time:.3f}s")
                print(f"📊 Attempts: {response.attempt_count}")
                print(f"📝 Generated {response_model.__name__} with {len(response.data.dict())} fields")
                
                # Save result
                result_file = f"structured_{provider.value}_{response_model.__name__.lower()}.json"
                with open(result_file, "w") as f:
                    json.dump(response.to_dict(), f, indent=2, default=str)
                
                print(f"💾 Saved to: {result_file}")
                
            except Exception as e:
                logger.error(f"Provider {provider} failed: {e}")
                print(f"❌ {provider.value} failed: {e}")
        
        return results
    
    def interactive_generator(self):
        """Interactive structured output generator."""
        print("🚀 Advanced Structured Output System")
        print("💡 Available providers:", [p.value for p in self.clients.keys()])
        print("📋 Available models: Clients, Companies, Products, Person")
        print("📝 Commands: 'quit', 'switch <provider>', 'model <name>'")
        print("=" * 60)
        
        current_provider = list(self.clients.keys())[0]
        current_model = Clients
        config = StructuredConfig()
        
        model_map = {
            "clients": Clients,
            "companies": Companies,
            "products": Products,
            "person": Person
        }
        
        while True:
            try:
                user_input = input(f"\n[{current_provider.value}:{current_model.__name__}] Request: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower().startswith('switch '):
                    provider_name = user_input[7:].strip()
                    try:
                        new_provider = OutputProvider(provider_name)
                        if new_provider in self.clients:
                            current_provider = new_provider
                            print(f"✅ Switched to {provider_name}")
                        else:
                            print(f"❌ Provider {provider_name} not available")
                    except ValueError:
                        print(f"❌ Unknown provider: {provider_name}")
                    continue
                
                if user_input.lower().startswith('model '):
                    model_name = user_input[6:].strip().lower()
                    if model_name in model_map:
                        current_model = model_map[model_name]
                        print(f"✅ Switched to {current_model.__name__} model")
                    else:
                        print(f"❌ Unknown model: {model_name}")
                        print(f"Available: {list(model_map.keys())}")
                    continue
                
                if not user_input:
                    continue
                
                # Generate structured output
                response = self.generate_with_retry(
                    user_input, current_model, current_provider, config
                )
                
                print(f"\n📋 Generated {current_model.__name__}:")
                print(json.dumps(response.data.dict(), indent=2, default=str))
                print(f"\n⏱️ Time: {response.response_time:.3f}s")
                print(f"🔄 Attempts: {response.attempt_count}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive generation error: {e}")
                print(f"❌ Error: {e}")


def demo_examples():
    """Demonstrate structured output capabilities."""
    try:
        # Initialize system
        system = AdvancedStructuredSystem()
        
        print("🚀 Advanced Structured Output System Demo")
        print("=" * 60)
        
        # Configuration
        config = StructuredConfig(
            temperature=0.1,
            max_retries=2,
            strict_validation=True
        )
        
        # Example 1: Generate clients
        print("\n📝 Example 1: Generate Clients")
        prompt = "Generate me a list of 5 persons with name, surname, date of birth and favourite movies."
        
        if OutputProvider.OPENAI in system.clients:
            response = system.generate_with_retry(prompt, Clients, OutputProvider.OPENAI, config)
            print(f"✅ Generated {len(response.data.persons)} persons")
            print(f"⏱️ Time: {response.response_time:.3f}s")
            
            # Display first person as example
            if response.data.persons:
                person = response.data.persons[0]
                print(f"👤 Example: {person.full_name}, Age: {person.calculate_age()}")
        
        # Example 2: Generate companies
        print(f"\n📝 Example 2: Generate Companies")
        companies_prompt = "Generate 3 technology companies with their details including name, industry, founding year, employees, and headquarters."
        
        if system.clients:
            provider = list(system.clients.keys())[0]
            response = system.generate_with_retry(companies_prompt, Companies, provider, config)
            print(f"✅ Generated {len(response.data.companies)} companies")
            
            # Save example
            with open("demo_companies.json", "w") as f:
                json.dump(response.to_dict(), f, indent=2, default=str)
            print("💾 Saved to: demo_companies.json")
        
        # Example 3: Provider comparison (if multiple available)
        if len(system.clients) > 1:
            print(f"\n🔄 Example 3: Provider Comparison")
            results = system.compare_providers(prompt, Clients, config)
            
            print(f"\n📊 Comparison Summary:")
            for provider, response in results.items():
                print(f"  {provider.value}: {response.response_time:.3f}s, {response.attempt_count} attempts")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo error: {e}")


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # Interactive mode
        system = AdvancedStructuredSystem()
        system.interactive_generator()
    else:
        # Demo mode
        demo_examples()


if __name__ == '__main__':
    main()
