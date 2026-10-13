"""Audio Processing Service.

Handles audio ingestion, format normalization, and conversion to 16kHz mono float32.
"""

import io
import math
import wave
import uuid
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple
from app.config import settings


class AudioService:
    def __init__(self):
        self.temp_dir = settings.AUDIO_TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def save_upload_to_wav(self, audio_bytes: bytes, filename_hint: str = "input") -> Tuple[Path, np.ndarray, int]:
        """Save raw audio upload (WebM, OGG, WAV, etc.) to 16kHz mono WAV and return path and numpy array."""
        unique_id = uuid.uuid4().hex[:8]
        out_path = self.temp_dir / f"{filename_hint}_{unique_id}.wav"

        # 1. Try reading with soundfile (supports standard WAV, FLAC, OGG)
        data = None
        samplerate = 16000
        try:
            data, samplerate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            pass

        # 2. Try decoding with PyAV (supports WebM, Opus, Ogg, MP3, AAC, and all browser containers)
        if data is None:
            try:
                data, samplerate = self._decode_with_av(audio_bytes)
            except Exception:
                pass

        # 3. Fallback for standard RIFF WAV via wave module
        if data is None:
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    samplerate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    
                    if sampwidth == 2:
                        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sampwidth == 4:
                        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

                    if n_channels > 1:
                        data = data.reshape(-1, n_channels)
            except Exception as e:
                raise ValueError(f"Could not decode audio payload (tried soundfile, PyAV, and wave): {e}")

        # Convert to mono if multi-channel
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        # Resample to 16kHz if needed
        target_sr = 16000
        if samplerate != target_sr:
            data = self._resample_simple(data, samplerate, target_sr)
            samplerate = target_sr

        # Write normalized WAV
        sf.write(str(out_path), data, target_sr, subtype="PCM_16")
        return out_path, data, target_sr

    def _decode_with_av(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Decode any audio container (WebM, Opus, Ogg, MP3, AAC) via PyAV."""
        import av
        container = av.open(io.BytesIO(audio_bytes))
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError("No audio stream found in media payload.")

        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks = []
        for packet in container.demux(audio_stream):
            for frame in packet.decode():
                resampled_list = resampler.resample(frame)
                if resampled_list:
                    for rf in resampled_list:
                        chunks.append(rf.to_ndarray()[0])

        if chunks:
            data = np.concatenate(chunks).astype(np.float32)
            return data, 16000
        return np.array([], dtype=np.float32), 16000

    def _resample_simple(self, data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling to avoid heavy external dependencies."""
        if orig_sr == target_sr:
            return data
        duration = len(data) / orig_sr
        target_len = int(duration * target_sr)
        if target_len <= 0:
            return data
        indices = np.linspace(0, len(data) - 1, target_len)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)

    def extract_acoustic_features(self, audio_data: np.ndarray, samplerate: int) -> dict:
        """Extract RMS energy, pitch estimate, and duration from audio data."""
        if len(audio_data) == 0:
            return {"rms_db": -100.0, "pitch_hz": 0.0, "duration_s": 0.0}

        duration_s = len(audio_data) / samplerate
        rms = float(np.sqrt(np.mean(audio_data ** 2)))
        rms_db = float(20 * math.log10(max(rms, 1e-5)))

        # Simple zero-crossing rate pitch approximation
        zero_crossings = np.nonzero(np.diff(audio_data > 0))[0]
        if len(zero_crossings) > 2 and duration_s > 0:
            pitch_hz = float((len(zero_crossings) / 2.0) / duration_s)
            pitch_hz = min(max(pitch_hz, 50.0), 500.0)
        else:
            pitch_hz = 0.0

        return {
            "rms_db": round(rms_db, 1),
            "pitch_hz": round(pitch_hz, 1),
            "duration_s": round(duration_s, 2)
        }


audio_service = AudioService()
