"""Speech-to-Text Service using faster-whisper on CPU with INT8 quantization."""

import time
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np

from app.config import settings
from app.schemas import ServiceStatus


class STTService:
    def __init__(self):
        self.model_name = settings.WHISPER_MODEL
        self.device = settings.WHISPER_DEVICE
        self.compute_type = settings.WHISPER_COMPUTE_TYPE
        self.download_root = str(settings.MODELS_DIR / "whisper")
        self._model = None
        self._is_loading = False

    def get_model(self):
        """Lazy loader for faster-whisper model."""
        if self._model is None and not self._is_loading:
            self._is_loading = True
            try:
                from faster_whisper import WhisperModel
                print(f"Loading faster-whisper model '{self.model_name}' on {self.device} ({self.compute_type})...")
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=self.download_root
                )
                print(f"faster-whisper model '{self.model_name}' loaded successfully.")
            finally:
                self._is_loading = False
        return self._model

    def check_health(self) -> ServiceStatus:
        """Report STT service readiness."""
        try:
            import faster_whisper
            return ServiceStatus(
                ready=True,
                details=f"faster-whisper available. Model '{self.model_name}' (CPU, {self.compute_type})."
            )
        except ImportError as e:
            return ServiceStatus(
                ready=False,
                details="faster-whisper package is not installed.",
                error=str(e)
            )

    def preload(self):
        """Warm up faster-whisper model at application startup."""
        self.get_model()

    def transcribe(self, audio: Union[str, Path, np.ndarray]) -> Tuple[str, float, str]:
        """Transcribe audio file or float32 numpy array synchronously."""
        model = self.get_model()
        if model is None:
            raise RuntimeError("faster-whisper model could not be initialized.")

        start_time = time.time()
        
        # Audio can be file path or numpy array (16kHz float32)
        if isinstance(audio, Path):
            audio = str(audio)

        beam_size = getattr(settings, "WHISPER_BEAM_SIZE", 1)

        segments, info = model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400)
        )

        transcript_parts = []
        for segment in segments:
            transcript_parts.append(segment.text.strip())

        full_transcript = " ".join(transcript_parts).strip()
        duration = round(time.time() - start_time, 2)
        lang = getattr(info, "language", "en")

        return full_transcript, duration, lang

    async def transcribe_async(self, audio: Union[str, Path, np.ndarray]) -> Tuple[str, float, str]:
        """Asynchronously transcribe audio in a worker thread without blocking the event loop."""
        import asyncio
        return await asyncio.to_thread(self.transcribe, audio)


stt_service = STTService()
