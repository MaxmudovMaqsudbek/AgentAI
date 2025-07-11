@echo off
cd /d "c:\Users\user\Desktop\AgentAI\llm-demos\openai-chat-completion"
echo Starting AgentAI Business Assistant...
echo Current directory: %CD%
echo Checking if streamlit_app.py exists...
if exist streamlit_app.py (
    echo File found! Starting Streamlit...
    streamlit run streamlit_app.py --server.port 8501
) else (
    echo File not found!
    dir *.py
)
pause
