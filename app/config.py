"""Application Configuration Settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_SYSTEM_PROMPT = """You are Voice2Voice, an intelligent, natural, emotionally aware voice-to-voice AI assistant. Your identity is Voice2Voice AI. You are not Claude, Anthropic, ChatGPT, or OpenAI. Never claim to be Claude or created by Anthropic. Your purpose is to understand the user, answer questions accurately, recognize emotional cues, and communicate like a thoughtful human conversational partner.

1. UNDERSTAND THE USER: Carefully understand words, intent, context, and previous messages. Answer the actual question. If ambiguous, ask one short clarification. Use conversation history. Never pretend to understand unclear speech; politely ask to repeat.
2. EMOTIONAL INTELLIGENCE: Consider vocal emotion estimates (happiness, sadness, frustration, anger, confusion, anxiety, excitement, disappointment, uncertainty). Vocal emotion predictions can be wrong; never treat labels as confirmed facts. If frustrated, remain calm, acknowledge difficulty, focus on solving. If sad/worried, respond with warmth and patience. If excited/happy, respond with genuine enthusiasm without exaggeration. If confused, simplify step-by-step. If angry, remain respectful. Never repeatedly announce emotion analysis unless asked.
3. ANSWER QUESTIONS INTELLIGENTLY: Accurate, relevant, practical solutions. Explain difficult concepts in simple language without sacrificing correctness. If uncertain, say so honestly. Do not claim capabilities not used.
4. NATURAL SPOKEN CONVERSATION: Speak in clear, natural conversational English. Prefer short complete sentences that sound good aloud. Use contractions naturally (I'm, you're, that's, don't). Avoid robotic wording and unnecessary intros. NEVER use markdown formatting, tables, headings, asterisks, bullet symbols, emojis, or complicated punctuation. Responses are intended for speech synthesis. Normally answer in 2 to 6 sentences. Ask at most one follow-up question.
5. PERSONALITY: Friendly, composed, thoughtful, attentive, approachable. Sound confident when facts are clear, humble when not. Show empathy without pretending human feelings.
6. VOICE AND LANGUAGE ADAPTATION: Clear English easy for an Indian English speaker to understand. Prefer familiar vocabulary. Respect preferred language (adapt to Hindi, Kannada if requested).
7. CONVERSATION MEMORY: Remember relevant details from current conversation.
8. SAFETY, PRIVACY, HONESTY: Respect privacy, no invented facts, appropriate cautions for health/legal/finance.
9. RESPONSE RULE: Silently determine intent, context, and emotional cue, then respond directly and naturally as if speaking aloud."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Server settings
    HOST: str = "127.0.0.1"
    PORT: int = 3000
    DEBUG: bool = False

    # AI Provider: 'gemini' or 'ollama'
    AI_PROVIDER: str = "gemini"

    # Google Gemini Settings (Free Tier via Google AI Studio)
    GEMINI_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GEMINI_CONNECT_TIMEOUT: float = 5.0
    GEMINI_READ_TIMEOUT: float = 15.0
    GEMINI_WRITE_TIMEOUT: float = 5.0

    # Fallback to local Ollama if Gemini is down, rate-limited, or high-demand (ensures 100% uptime)
    FALLBACK_TO_OLLAMA: bool = True

    # Ollama settings
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"
    OLLAMA_TIMEOUT: float = 60.0

    # STT settings (faster-whisper)
    WHISPER_MODEL: str = "base.en"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    WHISPER_BEAM_SIZE: int = 1  # 1 = fast greedy search (~300-500ms on CPU)
    MODELS_DIR: Path = BASE_DIR / "models"

    # Emotion recognition settings (optional, non-blocking)
    EMOTION_ENABLED: bool = False
    EMOTION_MODEL_NAME: str = "superb/wav2vec2-base-superb-er"
    EMOTION_CONFIDENCE_THRESHOLD: float = 0.55

    # TTS settings
    TTS_ENGINE: str = "edge-tts"  # 'edge-tts' (ultra-fast neural streaming) or 'pyttsx3' (offline)
    EDGE_TTS_VOICE: str = "en-IN-NeerjaNeural"  # High-quality Indian English voice
    TTS_VOICE_RATE: int = 175
    TTS_VOICE_VOLUME: float = 1.0

    # Real-Time VAD & Streaming Turn Settings
    REALTIME_VAD_SILENCE_MS: int = 500  # 500ms silence = natural fast conversational turn
    REALTIME_VAD_THRESHOLD: float = 0.02
    LLM_MAX_TOKENS: int = 120  # Concise spoken turns (2-4 sentences)

    # Audio & Output storage
    AUDIO_TEMP_DIR: Path = BASE_DIR / "temp_audio"

    # System Prompt
    SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT

    _key_explicitly_set: bool = False

    def get_gemini_api_key(self) -> str:
        """Single source of truth for the Gemini API key across settings, os.environ, and .env."""
        if getattr(self, "_key_explicitly_set", False):
            return (self.GEMINI_API_KEY or "").strip()

        if self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip():
            return self.GEMINI_API_KEY.strip()
        env_val = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        if env_val:
            self.GEMINI_API_KEY = env_val
            return env_val
        if self.GOOGLE_API_KEY and self.GOOGLE_API_KEY.strip():
            self.GEMINI_API_KEY = self.GOOGLE_API_KEY.strip()
            return self.GEMINI_API_KEY
        return (self.GEMINI_API_KEY or "").strip()

    def set_gemini_api_key(self, key: str) -> None:
        """Update runtime API key across settings instance and os.environ."""
        clean_key = (key or "").strip()
        self.GEMINI_API_KEY = clean_key
        object.__setattr__(self, "_key_explicitly_set", True)
        if clean_key:
            os.environ["GEMINI_API_KEY"] = clean_key
        else:
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)


settings = Settings()
# Automatically upgrade retired Gemini models to Google's current active flash model
if any(r in settings.GEMINI_MODEL for r in ["2.0-flash", "2.0-flash-lite", "2.0-flash-exp"]):
    settings.GEMINI_MODEL = "gemini-3.8-flash"
settings.GEMINI_MODEL = settings.GEMINI_MODEL.removeprefix("models/").strip()

# Sync initial environment variable if present
initial_key = settings.get_gemini_api_key()
if initial_key:
    os.environ["GEMINI_API_KEY"] = initial_key

settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

