import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes.health import router as health_router
from app.routes.conversation import router as conversation_router
from app.routes.realtime import router as realtime_router
from app.services.stt_service import stt_service
from app.services.emotion_service import emotion_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("[Voice2Voice] Real-Time AI Voice Agent Backend Starting Up")
    print(f"[Voice2Voice] Host: {settings.HOST}:{settings.PORT}")
    print(f"[Voice2Voice] Provider: {settings.AI_PROVIDER.upper()} (Ollama: {settings.OLLAMA_MODEL})")
    print(f"[Voice2Voice] STT: faster-whisper ({settings.WHISPER_MODEL}, greedy decoding beam_size={getattr(settings, 'WHISPER_BEAM_SIZE', 1)})")
    print(f"[Voice2Voice] TTS: {settings.TTS_ENGINE} ({getattr(settings, 'EDGE_TTS_VOICE', 'default')})")
    print(f"[Voice2Voice] Real-Time VAD Turn Delay: {getattr(settings, 'REALTIME_VAD_SILENCE_MS', 500)} ms")
    print("=" * 60)

    # Pre-warm faster-whisper in background thread so turns start with zero cold-start delay
    print("[Voice2Voice] Pre-warming speech recognition model in RAM...")
    asyncio.create_task(asyncio.to_thread(stt_service.preload))

    if getattr(settings, "EMOTION_ENABLED", False):
        asyncio.create_task(asyncio.to_thread(emotion_service.preload))

    yield
    print("[Voice2Voice] Backend Shutting Down.")


app = FastAPI(
    title="Voice2Voice Real-Time Emotion-Aware AI Agent",
    description="Ultra-low-latency bidirectional real-time voice-to-voice AI agent backend with WebSockets, token streaming, and neural TTS.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount audio temp directory for synthesized audio playback
app.mount("/audio", StaticFiles(directory=str(settings.AUDIO_TEMP_DIR)), name="audio")

# Include API routers
app.include_router(health_router)
app.include_router(conversation_router)
app.include_router(realtime_router)

# Mount frontend static files
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
