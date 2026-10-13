"""Voice Emotion Recognition Service.

Runs audio emotion classification locally on CPU using Transformers/PyTorch.
Treats all outputs as tentative estimates and provides graceful fallback if model is unavailable.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from app.config import settings
from app.schemas import EmotionEstimate, ServiceStatus
from app.services.audio_service import audio_service


class EmotionService:
    def __init__(self):
        self.model_name = settings.EMOTION_MODEL_NAME
        self.cache_dir = settings.MODELS_DIR / "emotion"
        self._classifier = None
        self._is_loading = False

    def get_classifier(self):
        """Lazy loader for audio classification pipeline."""
        if self._classifier is None and not self._is_loading:
            self._is_loading = True
            try:
                from transformers import pipeline
                print(f"Loading emotion classification pipeline '{self.model_name}' on CPU...")
                self._classifier = pipeline(
                    task="audio-classification",
                    model=self.model_name,
                    device=-1,  # CPU
                    model_kwargs={"cache_dir": str(self.cache_dir)}
                )
                print(f"Emotion classifier '{self.model_name}' ready.")
            except Exception as e:
                print(f"Notice: Pretrained model '{self.model_name}' could not be loaded ({e}).")
                print("Using acoustic signal feature estimator as graceful offline fallback.")
                self._classifier = None
            finally:
                self._is_loading = False
        return self._classifier

    def check_health(self) -> ServiceStatus:
        """Report emotion recognition readiness."""
        try:
            import torch
            import transformers
            return ServiceStatus(
                ready=True,
                details=f"Emotion service active on CPU (Model: {self.model_name} with acoustic fallback)."
            )
        except ImportError as e:
            return ServiceStatus(
                ready=False,
                details="Torch or Transformers not installed for emotion recognition.",
                error=str(e)
            )

    def preload(self):
        """Warm up classifier at application startup if emotion recognition is enabled."""
        if getattr(settings, "EMOTION_ENABLED", False):
            self.get_classifier()

    def analyze_audio(
        self,
        audio_path_or_data: Union[str, Path, np.ndarray],
        samplerate: int = 16000
    ) -> EmotionEstimate:
        """Analyze voice audio and produce a tentative emotion estimate.
        
        When EMOTION_ENABLED is False (default for ultra-low latency), uses instantaneous
        acoustic signal features (~2ms) rather than blocking on the 95M-parameter model.
        """
        # Calculate base acoustic metrics (RMS, Pitch, Duration)
        if isinstance(audio_path_or_data, np.ndarray):
            audio_array = audio_path_or_data
            file_path = None
        else:
            file_path = str(audio_path_or_data)
            try:
                import soundfile as sf
                audio_array, samplerate = sf.read(file_path, dtype="float32")
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)
            except Exception:
                audio_array = np.array([], dtype=np.float32)

        acoustic_metrics = audio_service.extract_acoustic_features(audio_array, samplerate)

        # 1. If deep emotion classification is explicitly enabled, run the model
        if getattr(settings, "EMOTION_ENABLED", False):
            classifier = self.get_classifier()
            if classifier is not None and len(audio_array) > 0:
                try:
                    results = classifier({"raw": audio_array, "sampling_rate": samplerate})
                    if results and len(results) > 0:
                        top_pred = results[0]
                        raw_label = str(top_pred.get("label", "")).lower()
                        score = float(top_pred.get("score", 0.6))
                        normalized_emotion, summary = self._normalize_label(raw_label)
                        
                        return EmotionEstimate(
                            emotion=normalized_emotion,
                            confidence=round(score, 2),
                            summary=summary,
                            pitch_hz=acoustic_metrics.get("pitch_hz"),
                            energy_db=acoustic_metrics.get("rms_db"),
                            is_vocal_prediction=True,
                            note="Deep acoustic model estimate; treated as tentative"
                        )
                except Exception as ex:
                    print(f"DL emotion classification notice: {ex}. Falling back to acoustic estimator.")

        # 2. Instantaneous Acoustic Signal Analysis (< 2ms CPU)
        return self._acoustic_fallback(acoustic_metrics)

    async def analyze_audio_async(
        self,
        audio_path_or_data: Union[str, Path, np.ndarray],
        samplerate: int = 16000
    ) -> EmotionEstimate:
        """Asynchronously analyze emotion in worker thread without blocking event loop."""
        import asyncio
        return await asyncio.to_thread(self.analyze_audio, audio_path_or_data, samplerate)

    def _normalize_label(self, raw_label: str) -> tuple[str, str]:
        """Map model-specific emotion classes to standardized labels."""
        raw = raw_label.lower()
        if any(k in raw for k in ["ang", "angry", "fru", "frustrated"]):
            return "frustrated", "Tense / Frustrated Acoustic Cues"
        elif any(k in raw for k in ["hap", "happy", "exc", "excited", "joy"]):
            return "excited", "Upbeat / Excited Acoustic Cues"
        elif any(k in raw for k in ["sad", "dep"]):
            return "sad", "Subdued / Sad Vocal Cues"
        elif any(k in raw for k in ["fea", "fear", "anx", "worried"]):
            return "worried", "Anxious / Hesitant Intonation"
        elif any(k in raw for k in ["sur", "surprise", "que", "question"]):
            return "confused", "Inquiring / Uncertain Intonation"
        return "neutral", "Composed / Neutral Tone"

    def _acoustic_fallback(self, metrics: dict) -> EmotionEstimate:
        """Acoustic heuristic fallback when model weights are not yet cached."""
        rms_db = metrics.get("rms_db", -50.0)
        pitch = metrics.get("pitch_hz", 0.0)

        if rms_db > -22.0 and pitch > 220.0:
            return EmotionEstimate(
                emotion="excited",
                confidence=0.72,
                summary="High-energy vocal cues",
                pitch_hz=pitch,
                energy_db=rms_db
            )
        elif rms_db > -24.0 and pitch > 200.0:
            return EmotionEstimate(
                emotion="frustrated",
                confidence=0.65,
                summary="Elevated acoustic energy / tension",
                pitch_hz=pitch,
                energy_db=rms_db
            )
        elif rms_db < -36.0 and pitch < 140.0:
            return EmotionEstimate(
                emotion="sad",
                confidence=0.68,
                summary="Quiet / subdued vocal cadence",
                pitch_hz=pitch,
                energy_db=rms_db
            )
        elif pitch > 210.0 and -35.0 <= rms_db <= -24.0:
            return EmotionEstimate(
                emotion="confused",
                confidence=0.62,
                summary="Questioning pitch contour",
                pitch_hz=pitch,
                energy_db=rms_db
            )
        else:
            return EmotionEstimate(
                emotion="neutral",
                confidence=0.75,
                summary="Composed & Neutral",
                pitch_hz=pitch,
                energy_db=rms_db
            )


emotion_service = EmotionService()
