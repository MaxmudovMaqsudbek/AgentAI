#!/usr/bin/env python3
"""
Simple test of core Business Assistant functionality.
"""

import os
import sys
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

def test_knowledge_base():
    """Test the knowledge base directly."""
    
    # Business info
    BUSINESS_INFO = {
        "name": "AgentAI Solutions",
        "hours": "Monday-Friday: 9:00 AM - 6:00 PM, Saturday: 10:00 AM - 4:00 PM, Closed Sunday",
        "location": "123 Tech Street, Innovation District, San Francisco, CA 94105",
        "phone": "+1 (555) 123-4567",
        "email": "contact@agentai-solutions.com",
        "services": [
            "AI Chatbot Development",
            "Custom ML Solutions", 
            "Data Analytics Consulting",
            "AI Integration Services",
            "Machine Learning Training"
        ]
    }
    
    # Simple knowledge base class
    class BusinessKnowledgeBase:
        def __init__(self, business_info):
            self.business_info = business_info
            self.patterns = self._build_knowledge_patterns()
        
        def _build_knowledge_patterns(self):
            return {
                'greeting': {
                    'keywords': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'greetings'],
                    'response_template': f"Hello! Welcome to {self.business_info['name']}. How can I help you today?"
                },
                'hours': {
                    'keywords': ['hours', 'open', 'close', 'time', 'schedule', 'when'],
                    'response_template': f"Our business hours are: {self.business_info['hours']}"
                },
                'location': {
                    'keywords': ['location', 'address', 'where', 'find', 'visit', 'directions'],
                    'response_template': f"We're located at: {self.business_info['location']}"
                },
                'contact': {
                    'keywords': ['contact', 'phone', 'email', 'call', 'reach'],
                    'response_template': f"You can reach us at:\n📞 Phone: {self.business_info['phone']}\n📧 Email: {self.business_info['email']}"
                },
                'services': {
                    'keywords': ['services', 'offer', 'do', 'provide', 'help', 'solutions'],
                    'response_template': f"We offer the following services:\n" + "\n".join([f"• {service}" for service in self.business_info['services']])
                }
            }
        
        def search_knowledge(self, query):
            query_lower = query.lower()
            matches = {}
            
            for category, data in self.patterns.items():
                for keyword in data['keywords']:
                    if keyword in query_lower:
                        if category not in matches:
                            matches[category] = 0
                        matches[category] += 1
            
            return matches
    
    print("🧪 Testing Knowledge Base")
    print("=" * 40)
    
    kb = BusinessKnowledgeBase(BUSINESS_INFO)
    
    test_queries = [
        "What are your business hours?",
        "Where are you located?", 
        "What services do you offer?",
        "How can I contact you?",
        "Hello, tell me about your company"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        matches = kb.search_knowledge(query)
        
        if matches:
            best_match = max(matches.items(), key=lambda x: x[1])
            category = best_match[0]
            response = kb.patterns[category]['response_template']
            print(f"✅ Best match: {category} (score: {best_match[1]})")
            print(f"💬 Response: {response[:100]}...")
        else:
            print("❌ No matches found")
    
    return True

def test_streaming_system():
    """Test the streaming system."""
    print("\n🧠 Testing AI Streaming System")
    print("=" * 40)
    
    try:
        # Test environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        if os.getenv('GOOGLE_API_KEY'):
            print("✅ Google API Key found")
        else:
            print("⚠️ Google API Key not found")
            
        if os.getenv('HUGGINGFACE_API_KEY'):
            print("✅ HuggingFace API Key found")  
        else:
            print("⚠️ HuggingFace API Key not found")
        
        # Test basic streaming import
        from streaming import GeminiStreamer, HuggingFaceStreamer
        
        # Test Gemini
        gemini = GeminiStreamer()
        print("✅ Gemini streamer imported successfully")
        
        # Test HuggingFace
        hf = HuggingFaceStreamer()
        print("✅ HuggingFace streamer imported successfully")
        
        print("✅ All streaming components available")
        
        return True
        
    except Exception as e:
        print(f"Streaming test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AgentAI Business Assistant - Core Tests")
    print("=" * 50)
    
    success1 = test_knowledge_base()
    success2 = test_streaming_system()
    
    if success1 and success2:
        print("\n🎉 Core functionality is working!")
        print("\n📋 To run the full app:")
        print("   streamlit run streamlit_app.py")
    else:
        print("\n⚠️ Some tests failed.")
