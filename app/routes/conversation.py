"""Conversation and Audio Endpoints."""

import json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.config import settings, BASE_DIR
from app.schemas import (


    ChatRequest,
    ChatResponse,
    TranscribeResponse,
    VoiceRequest,
    VoiceResponse,
    VoiceTurnResponse,
    ChatMessage,
    EmotionEstimate,
    SettingsUpdateRequest
)
from app.services.audio_service import audio_service
from app.services.stt_service import stt_service
from app.services.emotion_service import emotion_service
from app.services.llm_service import llm_service
from app.services.tts_service import tts_service

router = APIRouter(prefix="/api", tags=["Conversation"])


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe microphone audio with faster-whisper."""
    try:
        content = await file.read()
        if len(content) < 100:
            return TranscribeResponse(
                transcript="",
                duration_seconds=0.0,
                success=False,
                error="Audio recording was too short or empty."
            )

        wav_path, _, _ = audio_service.save_upload_to_wav(content, "stt_input")
        transcript, duration, lang = stt_service.transcribe(wav_path)

        return TranscribeResponse(
            transcript=transcript,
            duration_seconds=duration,
            language=lang,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    """Generate conversational AI response using configured provider (Gemini or Ollama)."""
    if not payload.transcript or not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    try:
        reply, guidance, provider_used, model_used = await llm_service.generate_response(
            transcript=payload.transcript,
            history=payload.history,
            emotion=payload.emotion
        )
        return ChatResponse(
            reply=reply,
            emotion_guidance=guidance,
            provider=provider_used,
            model=model_used,
            success=True
        )
    except Exception as e:
        err_str = str(e)
        if "key" in err_str.lower() or "unsupported" in err_str.lower():
            status_code = 400
        elif "timed out" in err_str.lower() or "timeout" in err_str.lower():
            status_code = 504
        else:
            status_code = 503
        raise HTTPException(status_code=status_code, detail=err_str)


@router.post("/voice", response_model=VoiceResponse)
async def synthesize_speech(payload: VoiceRequest):
    """Synthesize text to speech using offline Python TTS engine."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    try:
        audio_url, duration = tts_service.synthesize(payload.text, payload.emotion)
        return VoiceResponse(
            audio_url=audio_url,
            duration_seconds=duration,
            engine=tts_service.engine_type,
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@router.post("/turn", response_model=VoiceTurnResponse)
async def process_full_voice_turn(
    file: UploadFile = File(...),
    history: Optional[str] = Form("[]")
):
    """Process complete end-to-end voice turn: STT -> Emotion -> LLM -> TTS."""
    content = await file.read()
    if len(content) < 100:
        raise HTTPException(status_code=400, detail="Audio recording payload was empty.")

    # Parse conversation history
    parsed_history = []
    if history:
        try:
            history_data = json.loads(history)
            parsed_history = [ChatMessage(**item) for item in history_data]
        except Exception:
            parsed_history = []

    try:
        # 1. Normalize uploaded audio to 16kHz WAV
        wav_path, _, _ = audio_service.save_upload_to_wav(content, "turn_input")

        # 2. Speech-to-Text via faster-whisper
        transcript, stt_duration, _ = stt_service.transcribe(wav_path)
        if not transcript.strip():
            transcript = "I didn't hear anything clearly."

        # 3. Voice Emotion Classification on real audio
        emotion_estimate = emotion_service.analyze_audio(wav_path)

        # 4. Generate AI response from configured provider
        try:
            reply, _, provider_used, model_used = await llm_service.generate_response(
                transcript=transcript,
                history=parsed_history,
                emotion=emotion_estimate
            )
        except Exception as llm_err:
            provider_used = "system"
            model_used = "recovery"
            clean_err = str(llm_err)
            if "503" in clean_err or "high demand" in clean_err.lower():
                reply = "Google's cloud service is currently experiencing high demand. Please try speaking again in a few moments."
            elif "key" in clean_err.lower():
                reply = "Your Gemini API key could not be verified. Please check your key in settings."
            else:
                reply = "I had trouble generating a response just now. Could you please repeat that?"

        # 5. Offline Text-to-Speech synthesis
        audio_url, tts_duration = tts_service.synthesize(reply, emotion_estimate.emotion)

        return VoiceTurnResponse(
            transcript=transcript,
            emotion=emotion_estimate,
            reply=reply,
            audio_url=audio_url,
            provider=provider_used,
            model=model_used,
            duration_seconds=round(stt_duration + tts_duration, 2),
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice turn processing failed: {str(e)}")



def save_settings_to_env(
    provider: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    gemini_model: Optional[str] = None,
    ollama_model: Optional[str] = None,
    fallback_to_ollama: Optional[bool] = None,
    tts_engine: Optional[str] = None,
    edge_voice: Optional[str] = None,
    vad_silence_ms: Optional[int] = None,
    emotion_enabled: Optional[bool] = None,
):
    """Safely persist updated settings to .env without exposing secrets."""
    env_path = BASE_DIR / ".env"
    lines = []

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    updates = {}
    if provider is not None:
        updates["AI_PROVIDER"] = provider
    if gemini_api_key is not None:
        updates["GEMINI_API_KEY"] = gemini_api_key.strip()
    if gemini_model is not None:
        updates["GEMINI_MODEL"] = gemini_model.strip().removeprefix("models/")
    if ollama_model is not None:
        updates["OLLAMA_MODEL"] = ollama_model.strip()
    if fallback_to_ollama is not None:
        updates["FALLBACK_TO_OLLAMA"] = str(fallback_to_ollama).lower()
    if tts_engine is not None:
        updates["TTS_ENGINE"] = tts_engine.strip()
    if edge_voice is not None:
        updates["EDGE_TTS_VOICE"] = edge_voice.strip()
    if vad_silence_ms is not None:
        updates["REALTIME_VAD_SILENCE_MS"] = str(vad_silence_ms)
    if emotion_enabled is not None:
        updates["EMOTION_ENABLED"] = str(emotion_enabled).lower()

    if not updates:
        return

    new_lines = []
    handled_keys = set()
    for line in lines:
        matched = False
        stripped = line.strip()
        for k, v in updates.items():
            if stripped.startswith(f"{k}=") or stripped.startswith(f"#{k}="):
                new_lines.append(f"{k}={v}\n")
                handled_keys.add(k)
                matched = True
                break
        if not matched:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in handled_keys:
            new_lines.append(f"{k}={v}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


@router.get("/settings")
async def get_settings():
    """Retrieve current runtime configuration."""
    return {
        "provider": settings.AI_PROVIDER,
        "gemini_api_key_set": bool(settings.get_gemini_api_key()),
        "gemini_model": settings.GEMINI_MODEL.removeprefix("models/").strip(),
        "ollama_model": settings.OLLAMA_MODEL,
        "fallback_to_ollama": settings.FALLBACK_TO_OLLAMA,
        "tts_engine": settings.TTS_ENGINE,
        "edge_voice": settings.EDGE_TTS_VOICE,
        "vad_silence_ms": settings.REALTIME_VAD_SILENCE_MS,
        "emotion_enabled": settings.EMOTION_ENABLED,
        "whisper_beam_size": settings.WHISPER_BEAM_SIZE,
    }


@router.post("/settings")
async def update_settings(payload: SettingsUpdateRequest):
    """Update runtime configuration dynamically and persist safely to .env."""
    if payload.provider:
        settings.AI_PROVIDER = payload.provider.lower()
    if payload.gemini_api_key is not None:
        settings.set_gemini_api_key(payload.gemini_api_key)
    if payload.gemini_model:
        clean_model = payload.gemini_model.strip().removeprefix("models/")
        if any(r in clean_model for r in ["2.0-flash", "2.0-flash-lite", "2.0-flash-exp"]):
            clean_model = "gemini-3.8-flash"
        settings.GEMINI_MODEL = clean_model
    if payload.ollama_model:
        settings.OLLAMA_MODEL = payload.ollama_model.strip()
    if payload.fallback_to_ollama is not None:
        settings.FALLBACK_TO_OLLAMA = payload.fallback_to_ollama
    if payload.tts_engine:
        settings.TTS_ENGINE = payload.tts_engine.strip().lower()
        tts_service.engine_type = settings.TTS_ENGINE
    if payload.edge_voice:
        settings.EDGE_TTS_VOICE = payload.edge_voice.strip()
    if payload.vad_silence_ms is not None:
        settings.REALTIME_VAD_SILENCE_MS = payload.vad_silence_ms
    if payload.emotion_enabled is not None:
        settings.EMOTION_ENABLED = payload.emotion_enabled

    try:
        save_settings_to_env(
            provider=payload.provider,
            gemini_api_key=payload.gemini_api_key,
            gemini_model=payload.gemini_model,
            ollama_model=payload.ollama_model,
            fallback_to_ollama=payload.fallback_to_ollama,
            tts_engine=payload.tts_engine,
            edge_voice=payload.edge_voice,
            vad_silence_ms=payload.vad_silence_ms,
            emotion_enabled=payload.emotion_enabled,
        )
    except Exception as e:
        print(f"Notice: Could not persist settings to .env: {e}")

    return {
        "success": True,
        "provider": settings.AI_PROVIDER,
        "model": llm_service.active_model,
        "gemini_api_key_set": bool(settings.get_gemini_api_key()),
        "fallback_to_ollama": settings.FALLBACK_TO_OLLAMA,
        "tts_engine": settings.TTS_ENGINE,
        "edge_voice": settings.EDGE_TTS_VOICE,
        "vad_silence_ms": settings.REALTIME_VAD_SILENCE_MS,
        "emotion_enabled": settings.EMOTION_ENABLED,
    }

