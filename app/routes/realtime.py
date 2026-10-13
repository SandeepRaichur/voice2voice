"""Real-Time WebSocket Endpoint for Ultra-Low-Latency Conversational Voice Agent."""

import asyncio
import base64
import json
import time
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.schemas import ChatMessage, EmotionEstimate
from app.services.stt_service import stt_service
from app.services.emotion_service import emotion_service
from app.services.llm_service import llm_service
from app.services.tts_service import tts_service

router = APIRouter(tags=["Realtime"])


@router.websocket("/ws/conversation")
async def websocket_conversation(websocket: WebSocket):
    """Full-duplex real-time conversational voice WebSocket.
    
    Streams live speech transcription, incremental LLM tokens, and pipelined TTS audio.
    Supports instant client barge-in / interruption.
    """
    await websocket.accept()

    conversation_history: List[ChatMessage] = []
    active_turn_task: Optional[asyncio.Task] = None
    cancel_event = asyncio.Event()

    audio_buffer = bytearray()
    turn_active = False

    # Send initial readiness handshake
    await websocket.send_json({
        "type": "ready",
        "config": {
            "sample_rate": 16000,
            "silence_ms": getattr(settings, "REALTIME_VAD_SILENCE_MS", 500),
            "tts_engine": tts_service.engine_type,
            "tts_voice": getattr(settings, "EDGE_TTS_VOICE", "en-IN-NeerjaNeural"),
            "provider": llm_service.provider,
            "model": llm_service.active_model,
            "emotion_enabled": getattr(settings, "EMOTION_ENABLED", False)
        }
    })

    async def cancel_current_turn():
        nonlocal active_turn_task
        cancel_event.set()
        if active_turn_task and not active_turn_task.done():
            active_turn_task.cancel()
            try:
                await active_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        cancel_event.clear()
        active_turn_task = None

    async def execute_voice_turn(pcm_bytes: bytes, explicit_text: Optional[str] = None):
        """Execute pipelined STT -> LLM stream -> TTS stream turn with sub-second TTFA."""
        t_start = time.perf_counter()
        first_token_time = None
        first_audio_time = None

        try:
            # 1. Speech-to-Text
            if explicit_text:
                transcript = explicit_text.strip()
                t_stt = 0.0
            else:
                t_stt_start = time.perf_counter()
                # Parse 16kHz 16-bit mono PCM into float32 array
                if len(pcm_bytes) >= 44 and pcm_bytes[:4] == b"RIFF":
                    # WAV payload: extract raw PCM
                    raw_samples = np.frombuffer(pcm_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    raw_samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                if len(raw_samples) < 1600:  # < 100ms
                    await websocket.send_json({"type": "turn_ignored", "reason": "audio_too_short"})
                    return

                transcript, _, _ = await stt_service.transcribe_async(raw_samples)
                t_stt = (time.perf_counter() - t_stt_start) * 1000

            if not transcript or not transcript.strip():
                await websocket.send_json({"type": "turn_ignored", "reason": "empty_transcript"})
                return

            await websocket.send_json({
                "type": "transcript",
                "text": transcript,
                "stt_ms": round(t_stt, 1)
            })

            # 2. Non-blocking emotion check (instant acoustic or async background)
            emotion_est = EmotionEstimate(emotion="neutral", confidence=0.7)
            if getattr(settings, "EMOTION_ENABLED", False) and not explicit_text:
                # Trigger parallel task for emotion without blocking response
                asyncio.create_task(run_background_emotion(pcm_bytes, websocket))

            # 3. LLM Token Stream + Sentence Pipelined TTS
            full_reply_text = []
            provider_used = llm_service.provider
            model_used = llm_service.active_model

            token_stream = llm_service.generate_response_stream(
                transcript=transcript,
                history=conversation_history[-10:],
                emotion=emotion_est
            )

            # Wrapper to intercept tokens for UI and metrics
            async def wrapped_token_stream():
                nonlocal first_token_time, provider_used, model_used
                async for token, prov, mod in token_stream:
                    if cancel_event.is_set():
                        break
                    if first_token_time is None:
                        first_token_time = (time.perf_counter() - t_start) * 1000
                    provider_used = prov
                    model_used = mod
                    full_reply_text.append(token)
                    await websocket.send_json({
                        "type": "llm_token",
                        "token": token
                    })
                    yield token, prov, mod

            # Stream audio chunks directly as each sentence is ready
            async for event in tts_service.stream_pipelined_tts(wrapped_token_stream(), cancel_event):
                if cancel_event.is_set():
                    break

                if event["type"] == "sentence_start":
                    await websocket.send_json({
                        "type": "sentence_start",
                        "index": event["index"],
                        "text": event["text"]
                    })
                elif event["type"] == "audio_chunk":
                    if first_audio_time is None:
                        first_audio_time = (time.perf_counter() - t_start) * 1000

                    b64_audio = base64.b64encode(event["data"]).decode("ascii")
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "index": event["index"],
                        "format": event["format"],
                        "data": b64_audio,
                        "ttfa_ms": round(first_audio_time, 1) if first_audio_time else None
                    })
                elif event["type"] == "sentence_end":
                    await websocket.send_json({
                        "type": "sentence_end",
                        "index": event["index"],
                        "text": event["text"]
                    })

            # Save to conversation memory
            final_reply = "".join(full_reply_text).strip()
            if final_reply and not cancel_event.is_set():
                conversation_history.append(ChatMessage(role="user", content=transcript))
                conversation_history.append(ChatMessage(role="assistant", content=final_reply))
                if len(conversation_history) > 20:
                    conversation_history[:] = conversation_history[-20:]

            t_total = (time.perf_counter() - t_start) * 1000
            await websocket.send_json({
                "type": "turn_complete",
                "reply": final_reply,
                "provider": provider_used,
                "model": model_used,
                "metrics": {
                    "stt_ms": round(t_stt, 1),
                    "ttft_ms": round(first_token_time or 0.0, 1),
                    "ttfa_ms": round(first_audio_time or 0.0, 1),
                    "total_ms": round(t_total, 1)
                }
            })

        except asyncio.CancelledError:
            await websocket.send_json({"type": "interrupted"})
        except Exception as err:
            await websocket.send_json({
                "type": "error",
                "message": str(err)
            })

    async def run_background_emotion(pcm: bytes, ws: WebSocket):
        try:
            raw_samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            est = await emotion_service.analyze_audio_async(raw_samples)
            await ws.send_json({
                "type": "emotion",
                "emotion": est.emotion,
                "confidence": est.confidence,
                "summary": est.summary
            })
        except Exception:
            pass

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Incoming binary PCM audio frame
                audio_buffer.extend(message["bytes"])

            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "audio_chunk":
                        raw_b64 = data.get("data", "")
                        if raw_b64:
                            audio_buffer.extend(base64.b64decode(raw_b64))

                    elif msg_type in ("audio_turn", "turn_audio"):
                        raw_b64 = data.get("data", "")
                        if raw_b64:
                            pcm_payload = base64.b64decode(raw_b64)
                            audio_buffer.clear()
                            await cancel_current_turn()
                            active_turn_task = asyncio.create_task(execute_voice_turn(pcm_payload))

                    elif msg_type == "speech_complete":
                        # User stopped speaking (VAD triggered)
                        if len(audio_buffer) > 1600:
                            pcm_payload = bytes(audio_buffer)
                            audio_buffer.clear()
                            await cancel_current_turn()
                            active_turn_task = asyncio.create_task(execute_voice_turn(pcm_payload))
                        else:
                            audio_buffer.clear()

                    elif msg_type in ("interrupt", "cancel"):
                        # Client detected user speech onset (barge-in): cancel assistant audio instantly!
                        await cancel_current_turn()
                        audio_buffer.clear()
                        await websocket.send_json({"type": "interrupted"})

                    elif msg_type == "text_turn":
                        # User sent direct text prompt
                        prompt_text = data.get("text", "")
                        if prompt_text.strip():
                            await cancel_current_turn()
                            active_turn_task = asyncio.create_task(execute_voice_turn(b"", explicit_text=prompt_text))

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

                except Exception as parse_err:
                    print(f"WS JSON parse error: {parse_err}")

    except WebSocketDisconnect:
        await cancel_current_turn()
    except Exception as e:
        await cancel_current_turn()
