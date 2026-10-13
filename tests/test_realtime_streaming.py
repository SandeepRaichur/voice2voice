"""Tests for Real-Time Streaming, Pipelined Audio, Cancellation, and WebSockets."""

import asyncio
import base64
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.config import settings
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.schemas import ChatMessage, EmotionEstimate


@pytest.fixture
def test_client():
    return TestClient(app)


def test_llm_generate_response_stream():
    """Verify incremental token streaming from LLM generator."""
    async def _test():
        llm = LLMService()
        transcript = "Hello! Please tell me a quick one-sentence fact about stars."
        history = []
        emotion = EmotionEstimate(emotion="neutral", confidence=0.8)

        tokens_received = []
        async for token, provider, model in llm.generate_response_stream(transcript, history, emotion):
            assert isinstance(token, str)
            tokens_received.append(token)
            if len(tokens_received) > 5:
                break

        assert len(tokens_received) > 0
        full_text = "".join(tokens_received)
        assert len(full_text) > 0

    asyncio.run(_test())


def test_tts_sentence_chunker_and_pipelining():
    """Verify streaming sentence chunker splits incoming tokens and synthesizes audio."""
    async def _test():
        tts = TTSService()
        cancel_event = asyncio.Event()

        async def mock_token_stream():
            tokens = [
                "Hello ", "there! ", "This is ", "sentence one. ",
                "And here ", "comes sentence ", "two, which is also ", "complete."
            ]
            for t in tokens:
                await asyncio.sleep(0.01)
                yield t, "ollama", "qwen2.5:3b"

        events = []
        async for event in tts.stream_pipelined_tts(mock_token_stream(), cancel_event):
            events.append(event)

        types = [e["type"] for e in events]
        assert "sentence_start" in types
        assert "audio_chunk" in types
        assert "sentence_end" in types

        # Check that audio chunk contains non-empty binary bytes
        audio_chunks = [e for e in events if e["type"] == "audio_chunk"]
        assert len(audio_chunks) >= 1
        assert len(audio_chunks[0]["data"]) > 100

    asyncio.run(_test())


def test_tts_cancellation_on_barge_in():
    """Verify that setting cancel_event immediately halts downstream synthesis."""
    async def _test():
        tts = TTSService()
        cancel_event = asyncio.Event()

        async def mock_long_token_stream():
            yield "Sentence one is ready right now. ", "ollama", "qwen2.5:3b"
            await asyncio.sleep(0.05)
            cancel_event.set()  # Simulate user barge-in!
            yield "Sentence two should never finish because user interrupted. ", "ollama", "qwen2.5:3b"

        events = []
        async for event in tts.stream_pipelined_tts(mock_long_token_stream(), cancel_event):
            events.append(event)

        # Sentence 2 should not have completed
        completed_sentences = [e["index"] for e in events if e.get("type") == "sentence_end"]
        assert 1 not in completed_sentences

    asyncio.run(_test())


def test_realtime_websocket_handshake(test_client):
    """Test connecting to /ws/conversation WebSocket endpoint and receiving 'ready' message."""
    with test_client.websocket_connect("/ws/conversation") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "ready"
        assert "config" in data
        assert "sample_rate" in data["config"]
        assert data["config"]["sample_rate"] == 16000


def test_realtime_websocket_text_turn(test_client):
    """Test sending explicit text turn through the WebSocket and receiving tokens and audio."""
    with test_client.websocket_connect("/ws/conversation") as websocket:
        ready_msg = websocket.receive_json()
        assert ready_msg["type"] == "ready"

        # Send text turn
        websocket.send_json({
            "type": "text_turn",
            "text": "What is 2 plus 2? Answer in one word."
        })

        received_types = set()
        received_tokens = []
        timeout_turns = 0

        while timeout_turns < 50:
            msg = websocket.receive_json()
            mtype = msg.get("type")
            received_types.add(mtype)

            if mtype == "llm_token":
                received_tokens.append(msg.get("token", ""))
            elif mtype == "turn_complete":
                break
            timeout_turns += 1

        assert "transcript" in received_types
        assert "llm_token" in received_types
        assert len(received_tokens) > 0
        assert "turn_complete" in received_types


def test_realtime_websocket_barge_in(test_client):
    """Test sending interrupt signal during turn cancels generation immediately."""
    with test_client.websocket_connect("/ws/conversation") as websocket:
        ready_msg = websocket.receive_json()
        assert ready_msg["type"] == "ready"

        # Start a turn
        websocket.send_json({
            "type": "text_turn",
            "text": "Tell me a very long story about ancient castles."
        })

        # Receive at least the transcript message
        first_msg = websocket.receive_json()
        assert first_msg["type"] == "transcript"

        # Send cancel / barge-in interrupt
        websocket.send_json({"type": "interrupt"})

        # Expect interrupted message or clean termination
        interrupted_seen = False
        for _ in range(15):
            msg = websocket.receive_json()
            if msg.get("type") == "interrupted":
                interrupted_seen = True
                break

        assert interrupted_seen is True


def test_settings_streaming_persistence(test_client):
    """Verify that new streaming configuration options persist and update."""
    res = test_client.get("/api/settings")
    assert res.status_code == 200
    cfg = res.json()
    assert "tts_engine" in cfg
    assert "edge_voice" in cfg
    assert "vad_silence_ms" in cfg
    assert "emotion_enabled" in cfg

    # Update settings
    update_res = test_client.post("/api/settings", json={
        "tts_engine": "edge-tts",
        "edge_voice": "en-IN-PrabhatNeural",
        "vad_silence_ms": 400,
        "emotion_enabled": False
    })
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["tts_engine"] == "edge-tts"
    assert updated["edge_voice"] == "en-IN-PrabhatNeural"
    assert updated["vad_silence_ms"] == 400
    assert updated["emotion_enabled"] is False
