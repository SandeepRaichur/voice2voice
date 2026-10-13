"""Health and Service Readiness Endpoints."""

import sys
from fastapi import APIRouter
from app.config import settings
from app.schemas import HealthResponse
from app.services.llm_service import llm_service
from app.services.stt_service import stt_service
from app.services.emotion_service import emotion_service
from app.services.tts_service import tts_service

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/status", response_model=HealthResponse)
async def get_status():
    """Verify and report active operational readiness of all sub-services."""
    ollama_status = await llm_service.check_health()
    stt_status = stt_service.check_health()
    emotion_status = emotion_service.check_health()
    tts_status = tts_service.check_health()

    return HealthResponse(
        server="online",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        provider=llm_service.provider,
        ollama=ollama_status,
        stt=stt_status,
        emotion=emotion_status,
        tts=tts_status,
        selected_model=llm_service.active_model
    )
