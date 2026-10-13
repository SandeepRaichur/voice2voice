"""Unit and integration tests for conversation, TTS, STT, and end-to-end pipeline."""

import io
import wave
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from app.main import app
from app.services.audio_service import audio_service
from app.services.tts_service import tts_service
from app.services.emotion_service import emotion_service
from app.services.llm_service import llm_service
from app.schemas import ChatMessage, EmotionEstimate

client = TestClient(app)


def test_audio_normalization_and_features():
    """Verify audio service converts and extracts acoustic features."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine_wave, sr, format="WAV")
    wav_bytes = buf.getvalue()

    wav_path, data, out_sr = audio_service.save_upload_to_wav(wav_bytes, "test_tone")
    assert wav_path.exists()
    assert out_sr == 16000
    assert len(data) > 0

    features = audio_service.extract_acoustic_features(data, out_sr)
    assert "rms_db" in features
    assert "pitch_hz" in features
    assert features["duration_s"] == pytest.approx(1.0, rel=0.1)


def test_emotion_recognition_on_audio():
    """Verify emotion service analyzes audio signal and produces an estimate."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    sine_wave = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    estimate = emotion_service.analyze_audio(sine_wave, sr)
    assert estimate.emotion in ["neutral", "calm", "excited", "frustrated", "sad", "confused", "worried"]
    assert 0.0 <= estimate.confidence <= 1.0
    assert estimate.is_vocal_prediction is True


def test_offline_tts_synthesis():
    """Verify offline Python TTS synthesizes audio file and returns duration."""
    response = client.post("/api/voice", json={"text": "Hello, this is a test of offline speech synthesis."})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["audio_url"].startswith("/audio/")
    assert data["duration_seconds"] > 0.0


def test_chat_empty_transcript_validation():
    """Verify empty transcript is rejected with 400 Bad Request."""
    response = client.post("/api/chat", json={"transcript": "   "})
    assert response.status_code == 400


def test_ollama_multiturn_context():
    """Verify Ollama qwen2.5:3b answers and maintains conversational context."""
    from app.config import settings
    orig_provider = settings.AI_PROVIDER
    settings.AI_PROVIDER = "ollama"
    try:
        # Turn 1
        res1 = client.post("/api/chat", json={
            "transcript": "My favorite color is emerald green.",
            "history": []
        })
        assert res1.status_code == 200
        data1 = res1.json()
        assert len(data1["reply"]) > 5

        # Turn 2 with history
        res2 = client.post("/api/chat", json={
            "transcript": "What color did I say was my favorite?",
            "history": [
                {"role": "user", "content": "My favorite color is emerald green."},
                {"role": "assistant", "content": data1["reply"]}
            ]
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert "green" in data2["reply"].lower()
    finally:
        settings.AI_PROVIDER = orig_provider



def test_webm_payload_decoding():
    """Verify audio service successfully decodes browser WebM Opus audio payloads."""
    import av
    out_buf = io.BytesIO()
    out_container = av.open(out_buf, mode="w", format="webm")
    out_stream = out_container.add_stream("opus", rate=48000)
    out_stream.layout = "mono"

    t = np.linspace(0, 0.5, 24000, endpoint=False)
    data = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    frame = av.AudioFrame.from_ndarray(data.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = 48000

    for packet in out_stream.encode(frame):
        out_container.mux(packet)
    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()

    webm_bytes = out_buf.getvalue()
    wav_path, decoded_data, sr = audio_service.save_upload_to_wav(webm_bytes, "test_webm")
    assert wav_path.exists()
    assert sr == 16000
    assert len(decoded_data) > 0


def test_settings_api_and_provider_toggle():
    """Verify settings GET and POST endpoints correctly configure runtime options without breaking disk state."""
    from unittest.mock import patch
    from app.config import settings

    orig_provider = settings.AI_PROVIDER
    orig_fallback = settings.FALLBACK_TO_OLLAMA
    orig_model = settings.GEMINI_MODEL
    try:
        with patch("app.routes.conversation.save_settings_to_env"):
            get_res = client.get("/api/settings")
            assert get_res.status_code == 200
            settings_data = get_res.json()
            assert "provider" in settings_data
            assert "gemini_model" in settings_data
            assert "gemini_api_key_set" in settings_data

            post_res = client.post("/api/settings", json={
                "provider": "gemini",
                "gemini_model": "gemini-3.8-flash",
                "fallback_to_ollama": True
            })
            assert post_res.status_code == 200
            updated_data = post_res.json()
            assert updated_data["provider"] == "gemini"
            assert updated_data["fallback_to_ollama"] is True
            assert settings.FALLBACK_TO_OLLAMA is True
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.FALLBACK_TO_OLLAMA = orig_fallback
        settings.GEMINI_MODEL = orig_model


def test_settings_key_consistency_and_propagation():
    """Verify settings and LLM service share single source of truth for API key and model updates."""
    from app.config import settings
    orig_key = settings.GEMINI_API_KEY
    orig_model = settings.OLLAMA_MODEL
    try:
        # 1. Update key via helper
        settings.set_gemini_api_key("test_runtime_sync_key")
        assert settings.get_gemini_api_key() == "test_runtime_sync_key"
        assert llm_service.gemini_api_key == "test_runtime_sync_key"

        # 2. Update model and verify llm_service has no stale attributes
        settings.OLLAMA_MODEL = "custom-test-model"
        assert llm_service.ollama_model == "custom-test-model"
        assert llm_service.active_model == "gemini-3.8-flash"  # provider is gemini

        # 3. Clear key
        settings.set_gemini_api_key("")
        assert settings.get_gemini_api_key() == ""
        assert llm_service.gemini_api_key == ""
    finally:
        settings.set_gemini_api_key(orig_key)
        settings.OLLAMA_MODEL = orig_model


def test_gemini_routing_when_selected():
    """Verify Gemini API is called with correct headers and payload when Gemini is selected."""
    from unittest.mock import patch
    from app.config import settings
    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_model = settings.GEMINI_MODEL
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("test_fake_gemini_key_123")
    settings.GEMINI_MODEL = "gemini-3.8-flash"
    settings.FALLBACK_TO_OLLAMA = False

    called_urls = []
    called_headers = []

    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello, I am Voice2Voice assistant."}]
                    }
                }]
            }

    async def mock_post(self_client, url, *args, **kwargs):
        called_urls.append(str(url))
        called_headers.append(kwargs.get("headers", {}))
        return MockResponse()

    try:
        with patch("httpx.AsyncClient.post", new=mock_post):
            res = client.post("/api/chat", json={
                "transcript": "Hello, who are you?",
                "history": []
            })
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["provider"] == "gemini"
            assert data["model"] == "gemini-3.8-flash"
            assert "Voice2Voice" in data["reply"]
            # Verify endpoint called was Google Generative Language
            assert any("generativelanguage.googleapis.com" in u for u in called_urls)

            # Verify Claude/Anthropic was NOT called
            assert not any("anthropic.com" in u for u in called_urls)
            # Verify API key was sent via x-goog-api-key header, NOT in URL
            assert any(h.get("x-goog-api-key") == "test_fake_gemini_key_123" for h in called_headers)
            assert not any("test_fake_gemini_key_123" in u for u in called_urls)
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.GEMINI_MODEL = orig_model
        settings.FALLBACK_TO_OLLAMA = orig_fallback


def test_gemini_missing_api_key_error():
    """Verify selecting Gemini without an API key surfaces an error when fallback is disabled."""
    from app.config import settings
    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("")
    settings.FALLBACK_TO_OLLAMA = False
    try:
        res = client.post("/api/chat", json={
            "transcript": "Hello!",
            "history": []
        })
        assert res.status_code == 400
        data = res.json()
        assert "key" in data["detail"].lower()
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.FALLBACK_TO_OLLAMA = orig_fallback


def test_gemini_timeout_handling_without_fallback():
    """Verify Gemini read timeout raises clear timeout error when fallback is disabled."""
    import httpx
    from unittest.mock import patch
    from app.config import settings

    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("test_key_for_timeout")
    settings.FALLBACK_TO_OLLAMA = False

    async def mock_post_timeout(self_client, url, *args, **kwargs):
        raise httpx.ReadTimeout("The read operation timed out")

    try:
        with patch("httpx.AsyncClient.post", new=mock_post_timeout):
            res = client.post("/api/chat", json={
                "transcript": "Hello, can you answer?",
                "history": []
            })
            assert res.status_code == 504
            data = res.json()
            assert "timed out" in data["detail"].lower()
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.FALLBACK_TO_OLLAMA = orig_fallback


def test_gemini_timeout_fallback_to_ollama():
    """Verify Gemini read timeout seamlessly delegates to Ollama when fallback is enabled."""
    import httpx
    from unittest.mock import patch
    from app.config import settings

    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("test_key_for_timeout")
    settings.FALLBACK_TO_OLLAMA = True

    class MockOllamaResponse:
        status_code = 200
        def json(self):
            return {"message": {"content": "I am Voice2Voice responding via local fallback."}}

    async def mock_post(self_client, url, *args, **kwargs):
        if "generativelanguage" in str(url):
            raise httpx.ReadTimeout("The read operation timed out")
        return MockOllamaResponse()

    try:
        with patch("httpx.AsyncClient.post", new=mock_post):
            res = client.post("/api/chat", json={
                "transcript": "Hello, how are you?",
                "history": []
            })
            assert res.status_code == 200
            data = res.json()
            assert data["provider"] == "ollama"
            assert data["model"] == settings.OLLAMA_MODEL
            assert "fallback" in data["reply"]
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.FALLBACK_TO_OLLAMA = orig_fallback


def test_gemini_503_high_demand_fallback():
    """Verify Gemini 503 High Demand triggers automatic Ollama fallback with truthful provider metadata."""
    from unittest.mock import patch
    from app.config import settings

    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("test_key_for_503")
    settings.FALLBACK_TO_OLLAMA = True

    class Mock503Response:
        status_code = 503
        text = '{"error": {"code": 503, "message": "This model is currently experiencing high demand."}}'
        def json(self):
            return {"error": {"code": 503, "message": "This model is currently experiencing high demand."}}

    class MockOllamaResponse:
        status_code = 200
        def json(self):
            return {"message": {"content": "I am Voice2Voice answering locally."}}

    async def mock_post(self_client, url, *args, **kwargs):
        if "generativelanguage" in str(url):
            return Mock503Response()
        return MockOllamaResponse()

    try:
        with patch("httpx.AsyncClient.post", new=mock_post):
            res = client.post("/api/chat", json={
                "transcript": "Hello!",
                "history": []
            })
            assert res.status_code == 200
            data = res.json()
            assert data["provider"] == "ollama"
            assert data["model"] == settings.OLLAMA_MODEL
            assert "answering locally" in data["reply"]
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.FALLBACK_TO_OLLAMA = orig_fallback


def test_gemini_unavailable_model_handling():
    """Verify Gemini 404 (model retired or unavailable) surfaces error when fallback is disabled."""
    from unittest.mock import patch
    from app.config import settings

    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("test_key_for_404")
    settings.FALLBACK_TO_OLLAMA = False

    class Mock404Response:
        status_code = 404
        text = '{"error": {"code": 404, "message": "models/gemini-old is no longer available."}}'
        def json(self):
            return {"error": {"code": 404, "message": "models/gemini-old is no longer available."}}

    async def mock_post(self_client, url, *args, **kwargs):
        return Mock404Response()

    try:
        with patch("httpx.AsyncClient.post", new=mock_post):
            res = client.post("/api/chat", json={
                "transcript": "Hello!",
                "history": []
            })
            assert res.status_code == 503
            data = res.json()
            assert "not available" in data["detail"].lower()
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.FALLBACK_TO_OLLAMA = orig_fallback


def test_explicit_fallback_updates_metadata():
    """Verify when FALLBACK_TO_OLLAMA=True, metadata reflects Ollama and not Gemini."""
    from unittest.mock import patch
    from app.config import settings
    orig_provider = settings.AI_PROVIDER
    orig_key = settings.GEMINI_API_KEY
    orig_fallback = settings.FALLBACK_TO_OLLAMA

    settings.AI_PROVIDER = "gemini"
    settings.set_gemini_api_key("failing_sample_key")
    settings.FALLBACK_TO_OLLAMA = True

    class MockErrorResponse:
        status_code = 400
        text = '{"error": {"code": 400, "message": "API key not valid."}}'
        def json(self):
            return {"error": {"code": 400, "message": "API key not valid."}}

    class MockOllamaResponse:
        status_code = 200
        def json(self):
            return {"message": {"content": "I am Voice2Voice assistant running on Ollama."}}

    async def mock_post(self_client, url, *args, **kwargs):
        if "generativelanguage" in str(url):
            return MockErrorResponse()
        return MockOllamaResponse()

    try:
        with patch("httpx.AsyncClient.post", new=mock_post):
            res = client.post("/api/chat", json={
                "transcript": "Hello!",
                "history": []
            })
            assert res.status_code == 200
            data = res.json()
            # Crucial check: Provider and model MUST be Ollama, not Gemini!
            assert data["provider"] == "ollama"
            assert data["model"] == settings.OLLAMA_MODEL
            assert "Claude" not in data["reply"]
            assert "Anthropic" not in data["reply"]
    finally:
        settings.AI_PROVIDER = orig_provider
        settings.set_gemini_api_key(orig_key)
        settings.FALLBACK_TO_OLLAMA = orig_fallback


