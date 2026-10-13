```bat
@echo off
title Voice2Voice AI
cd /d "C:\Users\User\Desktop\voice2voice"

call .venv\Scripts\activate.bat

ollama serve >nul 2>&1

python -m uvicorn app.main:app --host 127.0.0.1 --port 3000
pause
```