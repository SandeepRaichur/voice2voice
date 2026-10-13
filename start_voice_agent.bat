@echo off
title Voice2Voice AI Agent - Ultra-Low Latency Conversational Voice
color 0A

echo =====================================================================
echo           VOICE2VOICE AI AGENT - REAL-TIME CONVERSATIONAL SYSTEM
echo =====================================================================
echo.

cd /d "%~dp0"

:: 1. Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please create the virtual environment first:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: 2. Check and launch Ollama if installed
echo [*] Checking Ollama Local AI Brain service...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [*] Ollama daemon is not running. Attempting to start Ollama...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
        echo [*] Started Ollama serve in background.
        timeout /t 3 /nobreak >nul
    ) else (
        echo [!] Ollama binary not found in standard path.
        echo [!] If you intend to use local offline inference, ensure Ollama is installed and running.
    )
) else (
    echo [OK] Ollama service is active.
)

:: 3. Launch Web Application Server
echo.
echo [*] Starting Voice2Voice FastAPI Server on http://127.0.0.1:3000 ...
echo [*] Press CTRL+C to stop the application.
echo.

:: Open browser automatically after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:3000"

:: Start Uvicorn web server
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 3000

pause
