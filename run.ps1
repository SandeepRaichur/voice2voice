# Voice2Voice AI Agent Launcher & Setup Script
# PowerShell 5.1 / 7+ compatible

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "🎙️  VOICE2VOICE AI AGENT (PYTHON & OLLAMA) SETUP & RUNNER" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python 3.11
Write-Host "`n[1/7] Checking Python 3.11 installation..." -ForegroundColor Yellow
$PythonCmd = $null

if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCheck = py -3.11 --version 2>&1
    if ($pyCheck -match "Python 3\.11") {
        $PythonCmd = "py -3.11"
        Write-Host "Found Python 3.11 via py launcher: $pyCheck" -ForegroundColor Green
    }
}

if (-not $PythonCmd -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $ver = python --version 2>&1
    if ($ver -match "Python 3\.11") {
        $PythonCmd = "python"
        Write-Host "Found Python 3.11 in PATH: $ver" -ForegroundColor Green
    }
}

if (-not $PythonCmd) {
    Write-Host "Python 3.11 not found in PATH or py launcher." -ForegroundColor Red
    Write-Host "Please install Python 3.11 (64-bit) from python.org or run:" -ForegroundColor Yellow
    Write-Host "  winget install --id Python.Python.3.11" -ForegroundColor Yellow
    exit 1
}

# 2. Virtual Environment
Write-Host "`n[2/7] Checking virtual environment (.venv)..." -ForegroundColor Yellow
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at .venv..." -ForegroundColor Cyan
    Invoke-Expression "$PythonCmd -m venv .venv"
}
Write-Host "Virtual environment ready at .venv." -ForegroundColor Green

# 3. Install Dependencies
Write-Host "`n[3/7] Verifying Python dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install -r requirements.txt --quiet
Write-Host "Dependencies verified." -ForegroundColor Green

# 4. Check Ollama
Write-Host "`n[4/7] Checking local Ollama installation..." -ForegroundColor Yellow
$OllamaExe = $null

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    $OllamaExe = "ollama"
} elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
    $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
}

if (-not $OllamaExe) {
    Write-Host "Ollama is not installed. Please install Ollama from https://ollama.com" -ForegroundColor Red
    exit 1
}
Write-Host "Ollama found: $OllamaExe" -ForegroundColor Green

# Ensure Ollama daemon is running
$PortOpen = Test-NetConnection -ComputerName 127.0.0.1 -Port 11434 -InformationLevel Quiet
if (-not $PortOpen) {
    Write-Host "Starting Ollama background server..." -ForegroundColor Cyan
    Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# 5. Check/Pull qwen2.5:3b Model
Write-Host "`n[5/7] Verifying Ollama model (qwen2.5:3b)..." -ForegroundColor Yellow
$ModelsList = & $OllamaExe list
if ($ModelsList -notmatch "qwen2\.5:3b") {
    Write-Host "Pulling qwen2.5:3b model into Ollama (one-time download)..." -ForegroundColor Cyan
    & $OllamaExe pull qwen2.5:3b
} else {
    Write-Host "Model qwen2.5:3b is ready in Ollama." -ForegroundColor Green
}

# 6. Verify Offline TTS
Write-Host "`n[6/7] Verifying offline TTS engine..." -ForegroundColor Yellow
& $VenvPython -c "import pyttsx3; e=pyttsx3.init(); print(f'Detected {len(e.getProperty(\"voices\"))} system voices.')"
Write-Host "TTS engine verified." -ForegroundColor Green

# 7. Start FastAPI Application
Write-Host "`n[7/7] Launching Voice2Voice FastAPI Server..." -ForegroundColor Yellow
Write-Host "Local URL: http://127.0.0.1:3000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Cyan

& $VenvPython -m uvicorn app.main:app --host 127.0.0.1 --port 3000
