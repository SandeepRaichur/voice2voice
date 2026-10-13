"""Pydantic Request and Response Schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    ready: bool = False
    details: str = ""
    error: Optional[str] = None


class HealthResponse(BaseModel):
    server: str = "online"
    python_version: str
    provider: str = "gemini"
    ollama: ServiceStatus
    stt: ServiceStatus
    emotion: ServiceStatus
    tts: ServiceStatus
    selected_model: str


class SettingsUpdateRequest(BaseModel):
    provider: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    ollama_model: Optional[str] = None
    fallback_to_ollama: Optional[bool] = None
    tts_engine: Optional[str] = None
    edge_voice: Optional[str] = None
    vad_silence_ms: Optional[int] = None
    emotion_enabled: Optional[bool] = None



class EmotionEstimate(BaseModel):
    emotion: str = "neutral"
    confidence: float = 0.60
    summary: str = "Neutral / Composed"
    pitch_hz: Optional[float] = None
    energy_db: Optional[float] = None
    is_vocal_prediction: bool = True
    note: str = "Tentative acoustic estimate, treated as unconfirmed"


class ChatMessage(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    transcript: str
    history: List[ChatMessage] = Field(default_factory=list)
    emotion: Optional[EmotionEstimate] = None


class ChatResponse(BaseModel):
    reply: str
    emotion_guidance: Optional[str] = None
    provider: Optional[str] = None
    model: str
    success: bool = True
    error: Optional[str] = None


class TranscribeResponse(BaseModel):
    transcript: str
    duration_seconds: float = 0.0
    language: str = "en"
    success: bool = True
    error: Optional[str] = None


class VoiceRequest(BaseModel):
    text: str
    emotion: Optional[str] = None


class VoiceResponse(BaseModel):
    audio_url: str
    duration_seconds: float = 0.0
    engine: str = "pyttsx3"
    success: bool = True
    error: Optional[str] = None


class VoiceTurnResponse(BaseModel):
    transcript: str
    emotion: EmotionEstimate
    reply: str
    audio_url: str
    provider: Optional[str] = None
    model: str
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None

