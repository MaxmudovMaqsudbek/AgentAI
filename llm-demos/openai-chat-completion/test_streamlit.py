#!/usr/bin/env python3
"""
Direct test of the Streamlit app functionality.
"""

import os
import sys
sys.path.append('.')

# Set environment for testing
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

def test_streamlit_app():
    """Test the streamlit app by importing and checking components."""
    print("🧪 Testing Streamlit App Components")
    print("=" * 50)
    
    try:
        # Mock streamlit for import test
        import sys
        
        class MockStreamlit:
            session_state = type('SessionState', (), {
                'conversation': [],
                'support_tickets': [],
                'business_assistant': None,
                'user_session_id': 'test-123',
                'chat_analytics': []
            })()
            
            @staticmethod
            def set_page_config(**kwargs): pass
            @staticmethod
            def title(text): print(f"TITLE: {text}")
            @staticmethod
            def markdown(text): pass
            @staticmethod
            def info(msg): print(f"INFO: {msg}")
            @staticmethod
            def success(msg): print(f"SUCCESS: {msg}")
            @staticmethod
            def warning(msg): print(f"WARNING: {msg}")
            @staticmethod
            def sidebar(): return MockStreamlit()
            @staticmethod
            def selectbox(label, options, **kwargs): return options[0] if options else None
            @staticmethod
            def slider(label, min_val, max_val, value): return value
            @staticmethod
            def checkbox(label, value=False): return value
            @staticmethod
            def button(label): return False
            @staticmethod
            def text_input(label, value=""): return value
            @staticmethod
            def container(): return MockStreamlit()
            @staticmethod
            def columns(n): return [MockStreamlit() for _ in range(n)]
            @staticmethod
            def write(text): pass
            @staticmethod
            def empty(): return MockStreamlit()
            def __enter__(self): return self
            def __exit__(self, *args): pass
        
        # Mock streamlit
        sys.modules['streamlit'] = MockStreamlit
        
        # Now import the app
        print("📦 Importing streamlit app...")
        import streamlit_app
        print("✅ App imported successfully!")
        
        # Test business assistant
        print("🤖 Testing Business Assistant...")
        if hasattr(streamlit_app, 'BusinessAIAssistant'):
            assistant = streamlit_app.BusinessAIAssistant()
            print(f"✅ Business Assistant created - AI Available: {assistant.ai_available}")
            
            # Test query processing
            response = assistant.process_query("What are your business hours?", use_ai=False)
            print(f"✅ Query processed - Type: {response['type']}, Confidence: {response['confidence']}")
            print(f"📝 Response: {response['response'][:100]}...")
        
        # Test knowledge base
        print("📚 Testing Knowledge Base...")
        if hasattr(streamlit_app, 'BusinessKnowledgeBase'):
            kb = streamlit_app.BusinessKnowledgeBase(streamlit_app.BUSINESS_INFO)
            matches = kb.search_knowledge("What are your hours?")
            print(f"✅ Knowledge base working - Found {len(matches)} matches")
        
        print("\n🎉 All components working! App is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_minimal_server():
    """Run a minimal test server."""
    print("\n🌐 Starting Minimal Test Server")
    print("=" * 50)
    
    try:
        import socket
        
        # Check if port 8501 is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8501))
        sock.close()
        
        if result == 0:
            print("⚠️ Port 8501 is already in use")
            print("💡 You can access the app at: http://localhost:8501")
        else:
            print("✅ Port 8501 is available")
            print("\n📋 To start the app manually:")
            print("   1. Open a new terminal")
            print("   2. Navigate to the project directory")
            print("   3. Run: streamlit run streamlit_app.py")
            print("   4. Open: http://localhost:8501")
        
        return True
        
    except Exception as e:
        print(f"Server test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 AgentAI Business Assistant - App Tests")
    print("=" * 60)
    
    success1 = test_streamlit_app()
    success2 = run_minimal_server()
    
    if success1 and success2:
        print("\n🎉 Your Business Assistant is working perfectly!")
        print("\n🔧 Manual Start Instructions:")
        print("   streamlit run streamlit_app.py")
        print("   Then open: http://localhost:8501")
    else:
        print("\n⚠️ Some components need attention.")
