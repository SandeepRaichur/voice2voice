"""Text-to-Speech (TTS) Service.

Provides real-time neural streaming speech synthesis via edge-tts (Indian English voices)
and genuinely offline Python-based speech synthesis using pyttsx3 (Windows SAPI5).
"""

import asyncio
import io
import os
import re
import uuid
import wave
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, AsyncIterator
from app.config import settings
from app.schemas import ServiceStatus


class TTSService:
    def __init__(self):
        self.temp_dir = settings.AUDIO_TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = settings.MODELS_DIR / "piper"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def engine_type(self) -> str:
        return getattr(settings, "TTS_ENGINE", "edge-tts")

    @engine_type.setter
    def engine_type(self, val: str):
        settings.TTS_ENGINE = val


    @property
    def edge_voice(self) -> str:
        return getattr(settings, "EDGE_TTS_VOICE", "en-IN-NeerjaNeural")

    def check_health(self) -> ServiceStatus:
        """Report TTS service readiness and available offline and streaming engines."""
        details_list = []
        try:
            import edge_tts
            details_list.append(f"edge-tts neural streaming ready ({self.edge_voice})")
        except ImportError:
            pass

        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            details_list.append(f"pyttsx3 offline ready ({len(voices)} Windows voices)")
            engine.stop()
        except Exception:
            pass

        if details_list:
            return ServiceStatus(
                ready=True,
                details=" | ".join(details_list)
            )
        return ServiceStatus(
            ready=False,
            details="No TTS engine initialized.",
            error="Install edge-tts or verify pyttsx3."
        )

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """Strip markdown, unpronounceable symbols, and excessive formatting for clean TTS."""
        if not text:
            return ""
        # Remove markdown formatting characters like *, _, #, `, ~, etc.
        text = re.sub(r'[*_#`~]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove emojis and non-standard unicode symbols
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def synthesize_sentence_edge_tts_bytes(self, sentence: str, voice: Optional[str] = None) -> bytes:
        """Synthesize a complete sentence into in-memory MP3 audio bytes using edge-tts (~300-500ms)."""
        clean_sentence = self.clean_text_for_speech(sentence)
        if not clean_sentence:
            return b""

        import edge_tts
        chosen_voice = voice or self.edge_voice
        communicate = edge_tts.Communicate(clean_sentence, chosen_voice)
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio" and chunk.get("data"):
                chunks.append(chunk["data"])
        return b"".join(chunks)

    def _synthesize_sentence_pyttsx3_bytes(self, sentence: str) -> bytes:
        """Synthesize a single sentence to in-memory WAV bytes using pyttsx3."""
        clean_sentence = self.clean_text_for_speech(sentence)
        if not clean_sentence:
            return b""

        import pyttsx3
        temp_file = self.temp_dir / f"temp_{uuid.uuid4().hex[:6]}.wav"
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', settings.TTS_VOICE_RATE)
            engine.setProperty('volume', settings.TTS_VOICE_VOLUME)
            voices = engine.getProperty('voices')
            selected_voice = None
            for v in voices:
                name_lower = v.name.lower()
                if "india" in name_lower or "heera" in name_lower or "neerja" in name_lower:
                    selected_voice = v.id
                    break
                elif "zira" in name_lower or "david" in name_lower or "natural" in name_lower:
                    if selected_voice is None:
                        selected_voice = v.id
            if selected_voice:
                engine.setProperty('voice', selected_voice)

            engine.save_to_file(clean_sentence, str(temp_file))
            engine.runAndWait()
            engine.stop()

            if temp_file.exists():
                with open(temp_file, "rb") as f:
                    return f.read()
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
        return b""

    async def stream_pipelined_tts(
        self,
        token_stream: AsyncIterator[Tuple[str, str, str]],
        cancel_event: Optional[asyncio.Event] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Sentence Chunker & Streaming Audio Pipeline.
        
        Consumes tokens from the LLM, groups them into sentences, and immediately begins
        streaming TTS audio for sentence N while sentence N+1 is still generating!
        Yields complete, high-fidelity audio chunks per sentence for crystal-clear playback.
        """
        buffer = ""
        sentence_index = -1
        preferred_engine = self.engine_type

        # Boundary regex: period, exclamation, question, newline, or comma/colon if clause has >= 8 words
        sentence_end_re = re.compile(r'([.!?\n]+)\s*')

        async for token_text, provider, model in token_stream:
            if cancel_event and cancel_event.is_set():
                break

            buffer += token_text

            # Check if we have formed a complete sentence
            match = sentence_end_re.search(buffer)
            should_split = False
            split_pos = -1

            if match:
                should_split = True
                split_pos = match.end()
            elif len(buffer.split()) >= 12 and any(p in buffer for p in [',', ';', ':', ' - ']):
                # Long clause without punctuation: split at comma to keep latency ultra-low
                for p in [',', ';', ':']:
                    if p in buffer:
                        split_pos = buffer.find(p) + 1
                        should_split = True
                        break

            if should_split and split_pos > 0:
                sentence = buffer[:split_pos].strip()
                buffer = buffer[split_pos:].strip()

                if len(sentence) >= 2:
                    sentence_index += 1
                    yield {
                        "type": "sentence_start",
                        "index": sentence_index,
                        "text": sentence,
                        "provider": provider,
                        "model": model
                    }

                    # Synthesize full high-fidelity audio for this sentence immediately
                    audio_streamed = False
                    if preferred_engine == "edge-tts":
                        try:
                            mp3_bytes = await self.synthesize_sentence_edge_tts_bytes(sentence)
                            if mp3_bytes and not (cancel_event and cancel_event.is_set()):
                                audio_streamed = True
                                yield {
                                    "type": "audio_chunk",
                                    "index": sentence_index,
                                    "format": "mp3",
                                    "data": mp3_bytes
                                }
                        except Exception as edge_err:
                            # Graceful offline fallback to pyttsx3
                            audio_streamed = False

                    if not audio_streamed:
                        # Fallback to pyttsx3
                        wav_bytes = await asyncio.to_thread(self._synthesize_sentence_pyttsx3_bytes, sentence)
                        if wav_bytes and not (cancel_event and cancel_event.is_set()):
                            yield {
                                "type": "audio_chunk",
                                "index": sentence_index,
                                "format": "wav",
                                "data": wav_bytes
                            }

                    yield {
                        "type": "sentence_end",
                        "index": sentence_index,
                        "text": sentence
                    }

        # Flush any remaining buffer
        remaining = buffer.strip()
        if remaining and len(remaining) >= 2 and not (cancel_event and cancel_event.is_set()):
            sentence_index += 1
            yield {
                "type": "sentence_start",
                "index": sentence_index,
                "text": remaining,
                "provider": "assistant",
                "model": "active"
            }

            audio_streamed = False
            if preferred_engine == "edge-tts":
                try:
                    mp3_bytes = await self.synthesize_sentence_edge_tts_bytes(remaining)
                    if mp3_bytes and not (cancel_event and cancel_event.is_set()):
                        audio_streamed = True
                        yield {
                            "type": "audio_chunk",
                            "index": sentence_index,
                            "format": "mp3",
                            "data": mp3_bytes
                        }
                except Exception:
                    audio_streamed = False

            if not audio_streamed:
                wav_bytes = await asyncio.to_thread(self._synthesize_sentence_pyttsx3_bytes, remaining)
                if wav_bytes and not (cancel_event and cancel_event.is_set()):
                    yield {
                        "type": "audio_chunk",
                        "index": sentence_index,
                        "format": "wav",
                        "data": wav_bytes
                    }

            yield {
                "type": "sentence_end",
                "index": sentence_index,
                "text": remaining
            }

    def synthesize(self, text: str, emotion_hint: Optional[str] = None) -> Tuple[str, float]:
        """Convert text into an offline WAV audio file for HTTP API backwards compatibility.
        
        Returns: (audio_url_path, duration_seconds)
        """
        if not text or not text.strip():
            raise ValueError("No text provided for synthesis.")

        unique_id = uuid.uuid4().hex[:8]
        wav_filename = f"tts_{unique_id}.wav"
        output_path = self.temp_dir / wav_filename

        rate = settings.TTS_VOICE_RATE
        if emotion_hint:
            emo = emotion_hint.lower()
            if emo in ["frustrated", "angry"]:
                rate = int(settings.TTS_VOICE_RATE * 0.95)
            elif emo in ["sad", "worried"]:
                rate = int(settings.TTS_VOICE_RATE * 0.90)
            elif emo in ["excited", "happy"]:
                rate = int(settings.TTS_VOICE_RATE * 1.08)
            elif emo in ["confused"]:
                rate = int(settings.TTS_VOICE_RATE * 0.92)

        self._synthesize_pyttsx3(text, output_path, rate)

        duration = 0.0
        try:
            with wave.open(str(output_path), "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
                duration = round(frames / float(sr), 2)
        except Exception:
            words = len(text.split())
            duration = round((words / (rate / 60.0)), 2)

        return f"/audio/{wav_filename}", duration

    def _synthesize_pyttsx3(self, text: str, output_path: Path, rate: int):
        """Synthesize using pyttsx3 in an isolated engine call."""
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', settings.TTS_VOICE_VOLUME)

        voices = engine.getProperty('voices')
        selected_voice = None
        for v in voices:
            name_lower = v.name.lower()
            if "india" in name_lower or "heera" in name_lower or "neerja" in name_lower:
                selected_voice = v.id
                break
            elif "zira" in name_lower or "david" in name_lower or "natural" in name_lower:
                if selected_voice is None:
                    selected_voice = v.id

        if selected_voice:
            engine.setProperty('voice', selected_voice)

        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        engine.stop()


tts_service = TTSService()
