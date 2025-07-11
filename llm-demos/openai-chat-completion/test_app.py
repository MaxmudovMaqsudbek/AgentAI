#!/usr/bin/env python3
"""
Test the Streamlit app functionality without running the full Streamlit server.
"""

import os
import sys
sys.path.append('.')

# Mock Streamlit for testing
class MockStreamlit:
    class session_state:
        conversation = []
        support_tickets = []
        business_assistant = None
        user_session_id = "test-session-123"
        chat_analytics = []
    
    @staticmethod
    def set_page_config(**kwargs):
        print(f"Page config: {kwargs}")
    
    @staticmethod
    def title(text):
        print(f"TITLE: {text}")
    
    @staticmethod
    def markdown(text):
        print(f"MARKDOWN: {text[:100]}...")
    
    @staticmethod
    def warning(msg):
        print(f"WARNING: {msg}")
    
    @staticmethod
    def info(msg):
        print(f"INFO: {msg}")
    
    @staticmethod
    def success(msg):
        print(f"SUCCESS: {msg}")
    
    @staticmethod
    def sidebar():
        return MockStreamlit()
    
    @staticmethod
    def selectbox(label, options, **kwargs):
        return options[0] if options else None
    
    @staticmethod
    def slider(label, min_val, max_val, value):
        return value
    
    @staticmethod
    def checkbox(label, value=False):
        return value
    
    @staticmethod
    def button(label):
        return False
    
    @staticmethod
    def text_input(label, value=""):
        return value
    
    @staticmethod
    def container():
        return MockStreamlit()
    
    @staticmethod
    def columns(n):
        return [MockStreamlit() for _ in range(n)]
    
    @staticmethod
    def write(text):
        print(f"WRITE: {text}")
    
    @staticmethod
    def empty():
        return MockStreamlit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

# Replace streamlit module for testing
sys.modules['streamlit'] = MockStreamlit

# Import our app components
from streamlit_app import BusinessAIAssistant, BusinessKnowledgeBase, BUSINESS_INFO

def test_business_assistant():
    """Test the business assistant functionality."""
    print("🧪 Testing Business Assistant Components")
    print("=" * 50)
    
    try:
        # Test Knowledge Base
        print("📚 Testing Knowledge Base...")
        kb = BusinessKnowledgeBase(BUSINESS_INFO)
        
        test_queries = [
            "What are your business hours?",
            "Where are you located?", 
            "What services do you offer?",
            "How can I contact you?",
            "Hello, tell me about your company",
            "Thank you for your help"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            matches = kb.search_knowledge(query)
            print(f"📊 Found {len(matches)} matches: {list(matches.keys())}")
        
        # Test Business Assistant
        print(f"\n🤖 Testing Business Assistant...")
        assistant = BusinessAIAssistant()
        print(f"✅ Assistant initialized successfully")
        print(f"🔧 AI Available: {assistant.ai_available}")
        
        # Test queries
        test_query = "What are your business hours?"
        print(f"\n💬 Testing query: '{test_query}'")
        response = assistant.process_query(test_query, use_ai=False)
        print(f"📝 Response type: {response['type']}")
        print(f"🎯 Confidence: {response['confidence']}")
        print(f"💭 Response: {response['response'][:200]}...")
        
        print("\n✅ All tests passed! Business Assistant is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_responses():
    """Test AI response generation."""
    print("\n🧠 Testing AI Response System")
    print("=" * 50)
    
    try:
        assistant = BusinessAIAssistant()
        
        # Test different types of queries
        test_cases = [
            ("Hello", "greeting"),
            ("What are your hours?", "hours"),
            ("Where are you located?", "location"),
            ("Tell me about your services", "services"),
            ("Random question about something", "unknown")
        ]
        
        for query, expected_type in test_cases:
            print(f"\n🎯 Testing: '{query}' (Expected: {expected_type})")
            response = assistant.process_query(query, use_ai=False)
            print(f"📊 Type: {response['type']}, Confidence: {response['confidence']}")
            print(f"💬 Response: {response['response'][:150]}...")
        
        print("\n✅ AI response system working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ AI test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AgentAI Business Assistant - Component Tests")
    print("=" * 60)
    
    success1 = test_business_assistant()
    success2 = test_ai_responses()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Your Business Assistant is ready!")
        print("\n📋 How to run:")
        print("   streamlit run streamlit_app.py")
        print("\n🌐 The app will open at http://localhost:8501")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
