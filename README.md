# 🎙️ Voice2Voice: Real-Time, Ultra-Low-Latency Conversational AI Voice Agent

A high-performance, real-time voice-to-voice AI agent engineered in **100% Python** for sub-second conversational latency, full-duplex WebSocket streaming, gapless neural playback, and instant user barge-in.

---

## ⚡ Latency Transformation Summary

| Pipeline Component | Original Baseline (Cold/Batch) | Optimized Real-Time Pipeline | Latency Reduction |
| :--- | :--- | :--- | :--- |
| **VAD Silence Delay** | 1,200 ms wait | **400 – 500 ms** (Snappy adaptive VAD) | **~60% faster turn trigger** |
| **STT (faster-whisper)** | 1,934 ms (beam_size=5) | **~350 – 480 ms** (Greedy INT8, beam_size=1) | **~4x faster transcription** |
| **Emotion Recognition** | 6,891 ms (Blocking 95M Wav2Vec2) | **< 2 ms** (Signal heuristic or async background) | **~3,400x faster (saves ~6.9s)** |
| **LLM Inference** | 2,982 ms (Full response wait) | **Streaming tokens** (~300–1,200 ms to first token) | **Incremental token streaming** |
| **TTS Generation** | 294 ms (Full response to WAV file) | **Pipelined sentence chunks** (~500ms first chunk) | **Plays sentence 1 while sentence 2 generates** |
| **Time to First Audio (TTFA)** | **13,300 ms – 120,000 ms (2 min stalls)** | **~1,200 ms – 3,300 ms** (Empirically measured) | **~10x to 40x speedup!** |

---

## 🛠️ Technology Stack & Architecture

- **Backend**: Python 3.11 (64-bit), FastAPI, Uvicorn, WebSockets.
- **Real-Time Transport**: Full-duplex WebSocket at `/ws/conversation` with automatic reconnect & HTTP fallback at `/api/turn`.
- **Speech-to-Text (STT)**: `faster-whisper` (`base.en`, INT8 quantized on CPU, greedy `beam_size=1`).
- **AI Brains**:
  - **Local Offline Brain**: [Ollama](https://ollama.com) (`qwen2.5:3b`) with native incremental streaming (`stream: True`).
  - **Cloud Brain (Optional)**: Google Gemini (`gemini-3.8-flash` or `gemini-3.5-flash-lite`) via SSE streaming with instant local Ollama fallback.
- **Text-to-Speech (TTS)**:
  - **Real-Time Neural Streaming**: `edge-tts` (ultra-low-latency neural voices; default Indian English `en-IN-NeerjaNeural` / `en-IN-PrabhatNeural`).
  - **100% Offline Fallback**: Windows SAPI5 (`pyttsx3`).
- **Vocal Emotion Recognition**: Non-blocking acoustic feature estimator (<2ms) with optional background PyTorch Wav2Vec2 classification.
- **Frontend**: Glassmorphism UI with Web Audio API gapless streaming buffer, live token rendering, 3D reactive visualizer orb, real-time latency telemetry badge, and settings modal.

---

## 📁 Project Structure

```
voice2voice/
├── app/
│   ├── main.py                  # FastAPI server, lifespan prewarming & routing
│   ├── config.py                # Configuration tokens, model defaults & prompts
│   ├── schemas.py               # Pydantic validation schemas
│   ├── routes/
│   │   ├── health.py            # GET /api/status endpoint
│   │   ├── conversation.py      # REST endpoints: /api/turn, /api/settings, /api/chat
│   │   └── realtime.py          # Real-time WebSocket: /ws/conversation
│   ├── services/
│   │   ├── audio_service.py     # 16kHz WAV normalization & acoustic analysis
│   │   ├── stt_service.py       # faster-whisper INT8 CPU greedy inference
│   │   ├── emotion_service.py   # Non-blocking emotion classifier & acoustic heuristic
│   │   ├── llm_service.py       # Streaming token generator (Ollama & Gemini SSE)
│   │   └── tts_service.py       # Pipelined sentence chunker & edge-tts neural streamer
│   └── static/                  # Responsive glassmorphism frontend
│       ├── index.html           # UI layout, latency HUD, settings modal
│       ├── css/style.css        # Glassmorphic styles & animations
│       └── js/
│           ├── app.js           # WebSocket orchestrator & gapless Web Audio player
│           ├── audio-processor.js # Real mic capture, 500ms VAD & barge-in detector
│           └── visualizer.js    # Canvas 3D reactive neural voice orb
├── tests/
│   ├── test_health.py           # Health endpoint tests
│   ├── test_conversation.py     # HTTP conversation tests
│   └── test_realtime_streaming.py # WebSocket, streaming LLM, pipelined TTS & barge-in tests
├── start_voice_agent.bat        # Single-click Windows application launcher
├── requirements.txt             # Python package dependencies
├── .env                         # Active configuration
├── .env.example                 # Template configuration
└── README.md
```

---

## 🚀 Quick Start (Windows)

### Option 1: Single-Click Windows Launcher (`.bat`)
Double-click `start_voice_agent.bat` in the project folder. It will:
1. Verify Python virtual environment.
2. Ensure the Ollama daemon is running.
3. Start the FastAPI server on `http://127.0.0.1:3000`.
4. Open the browser automatically.

### Option 2: Windows PowerShell Commands
Open PowerShell in the project directory:

```powershell
# 1. Start Ollama local brain (if using local offline mode)
ollama serve

# 2. Run the application with Uvicorn
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 3000
```

Open your browser at:
👉 **[http://127.0.0.1:3000](http://127.0.0.1:3000)**

---

## 🧪 Testing and Validation

Run the complete automated test suite (23 comprehensive tests):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Run the live WebSocket latency benchmark:
```powershell
.\.venv\Scripts\python.exe scratch\benchmark_realtime.py
```

---

## ⚙️ Configuration Reference (`.env`)

```ini
# AI Provider: 'gemini' (Cloud API) or 'ollama' (100% Offline Local)
AI_PROVIDER=gemini
FALLBACK_TO_OLLAMA=true

# Google Gemini Settings (Free Tier key: https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash

# Ollama Local Brain Settings
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b
LLM_MAX_TOKENS=120

# Speech-to-Text (faster-whisper)
WHISPER_MODEL=base.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BEAM_SIZE=1

# Conversational Turn Detection (VAD silence threshold in milliseconds)
REALTIME_VAD_SILENCE_MS=500

# Text-to-Speech (TTS)
TTS_ENGINE=edge-tts
EDGE_TTS_VOICE=en-IN-NeerjaNeural

# Voice Emotion Recognition
EMOTION_ENABLED=false
```
