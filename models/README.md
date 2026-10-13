# Cached Models Directory

This directory stores downloaded and cached model weights for the offline inference pipeline:

1. `models/whisper/`
   - faster-whisper quantized models (e.g. `base.en`).
   - Cached automatically by faster-whisper upon first run.

2. `models/emotion/`
   - Audio classification model weights (e.g. `superb/wav2vec2-base-superb-er`).
   - Cached automatically by HuggingFace Transformers.

3. `models/piper/`
   - Piper TTS ONNX models and JSON voice configs if using Piper TTS.

### Offline Preparation
Once models are loaded on initial startup, all files remain cached here. The system can then run completely offline without internet connectivity.
