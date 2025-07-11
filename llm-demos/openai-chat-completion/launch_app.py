#!/usr/bin/env python3
"""
Manual Streamlit App Launcher for AgentAI Business Assistant
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def launch_streamlit_app():
    """Launch the Streamlit app manually."""
    
    print("🚀 AgentAI Business Assistant - Manual Launcher")
    print("=" * 60)
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    app_file = current_dir / "streamlit_app.py"
    
    print(f"📁 Current directory: {current_dir}")
    print(f"📄 App file: {app_file}")
    print(f"✅ App file exists: {app_file.exists()}")
    
    if not app_file.exists():
        print("❌ streamlit_app.py not found!")
        return False
    
    try:
        print("\n🌐 Starting Streamlit server...")
        print("💡 This will open your default browser automatically")
        print("🔗 URL: http://localhost:8501")
        print("\n⚠️ To stop the server, press Ctrl+C in the terminal")
        print("-" * 60)
        
        # Change to the app directory
        os.chdir(current_dir)
        
        # Run streamlit with proper arguments
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port", "8501",
            "--server.headless", "false"
        ]
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor output
        for line in iter(process.stdout.readline, ''):
            print(line.strip())
            if "Local URL:" in line or "localhost:8501" in line:
                print("\n🎉 App is ready! Opening browser...")
                webbrowser.open("http://localhost:8501")
                break
        
        # Wait for process to complete
        process.wait()
        return True
        
    except FileNotFoundError:
        print("❌ Streamlit not found! Please install it:")
        print("   pip install streamlit")
        return False
        
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return False

def show_manual_instructions():
    """Show manual start instructions."""
    print("\n📋 Manual Start Instructions")
    print("=" * 40)
    print("1. Open a command prompt or PowerShell")
    print("2. Navigate to the project directory:")
    print(f"   cd \"{Path(__file__).parent.absolute()}\"")
    print("3. Run the Streamlit app:")
    print("   streamlit run streamlit_app.py")
    print("4. Open your browser to: http://localhost:8501")
    print("\n🎯 Features Available:")
    print("• Business hours and location info")
    print("• AI-powered customer support")
    print("• Support ticket creation")
    print("• Multiple AI providers (Gemini, HuggingFace)")

if __name__ == "__main__":
    try:
        if launch_streamlit_app():
            print("\n✅ App launched successfully!")
        else:
            show_manual_instructions()
    except KeyboardInterrupt:
        print("\n\n🛑 App stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        show_manual_instructions()
