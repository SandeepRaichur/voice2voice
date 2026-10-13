/**
 * Client-Side Audio Processor & WAV/VAD Recorder
 * Captures real microphone audio using Web Audio API.
 * Encodes directly into 16kHz mono RIFF WAV (guaranteed universal compatibility).
 * Feeds live energy/pitch to the visualizer and emits clean audio blobs to the Python backend.
 */

class AudioProcessor {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.analyser = null;
    this.sourceNode = null;
    this.scriptProcessor = null;
    this.recordedSamples = [];

    this.isRecording = false;
    this.isRunning = false;
    this.animationFrameId = null;

    // Real-Time VAD (Voice Activity Detection) settings
    this.vadThreshold = 0.02; // RMS threshold
    this.silenceDurationMs = 500; // Snappy 500ms silence detection for natural turn-taking
    this.lastSpeechTime = 0;
    this.hasSpokenInTurn = false;
    this.isAssistantSpeaking = false;

    // Live acoustic metrics
    this.currentMetrics = {
      isSpeaking: false,
      rms: 0,
      energyDb: -100,
      pitchHz: 0
    };

    // Callbacks
    this.onMetricsUpdate = null;
    this.onSpeechComplete = null; // Emitted when VAD detects end of user turn
    this.onAudioChunk = null; // Emitted continuously with raw 16kHz Int16 PCM for real-time WebSocket streaming
    this.onBargeIn = null; // Emitted when user speaks while assistant is speaking
  }

  async start() {
    if (this.isRunning) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContextClass();
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      this.analyser.smoothingTimeConstant = 0.6;

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.sourceNode.connect(this.analyser);

      // ScriptProcessor to capture raw PCM float32 samples directly
      const bufferSize = 4096;
      this.scriptProcessor = this.audioContext.createScriptProcessor(bufferSize, 1, 1);
      this.scriptProcessor.onaudioprocess = (e) => {
        if (!this.isRecording) return;
        const inputData = e.inputBuffer.getChannelData(0);
        // Clone input buffer samples for local WAV packaging
        this.recordedSamples.push(new Float32Array(inputData));

        // Stream downsampled 16kHz Int16 PCM chunk over WebSocket if handler attached
        if (this.onAudioChunk && this.audioContext) {
          const origSr = this.audioContext.sampleRate;
          const targetSr = 16000;
          const resampled = this.resampleAudio(inputData, origSr, targetSr);
          const pcm16 = new Int16Array(resampled.length);
          for (let i = 0; i < resampled.length; i++) {
            const s = Math.max(-1, Math.min(1, resampled[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          this.onAudioChunk(pcm16.buffer);
        }
      };

      this.sourceNode.connect(this.scriptProcessor);
      // Mute the scriptProcessor output so the mic does not loopback/echo through the speakers
      this.muteNode = this.audioContext.createGain();
      this.muteNode.gain.value = 0;
      this.scriptProcessor.connect(this.muteNode);
      this.muteNode.connect(this.audioContext.destination);

      this.isRunning = true;
      this.processLoop();
      console.log('Microphone audio stream initialized with real-time PCM streamer.');
    } catch (err) {
      console.error('Microphone access failed:', err);
      throw err;
    }
  }

  startRecording() {
    if (!this.mediaStream) return;
    this.recordedSamples = [];
    this.hasSpokenInTurn = false;
    this.lastSpeechTime = Date.now();
    this.isRecording = true;
  }

  async stopRecording() {
    this.isRecording = false;
    if (this.recordedSamples.length === 0) return null;

    // Merge float32 arrays
    let totalLength = 0;
    for (const chunk of this.recordedSamples) {
      totalLength += chunk.length;
    }
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.recordedSamples) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.recordedSamples = [];

    // Resample to 16kHz
    const origSr = this.audioContext ? this.audioContext.sampleRate : 48000;
    const targetSr = 16000;
    const resampled = this.resampleAudio(merged, origSr, targetSr);

    // Encode directly into RIFF WAV blob
    return this.encodeWAV(resampled, targetSr);
  }

  resampleAudio(audioBuffer, origSr, targetSr) {
    if (origSr === targetSr) return audioBuffer;
    const ratio = origSr / targetSr;
    const newLength = Math.round(audioBuffer.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const origIndex = i * ratio;
      const indexFloor = Math.floor(origIndex);
      const indexCeil = Math.min(indexFloor + 1, audioBuffer.length - 1);
      const fraction = origIndex - indexFloor;
      result[i] = audioBuffer[indexFloor] * (1 - fraction) + audioBuffer[indexCeil] * fraction;
    }
    return result;
  }

  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, 'WAVE');

    // fmt sub-chunk
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true);  // NumChannels (1 = Mono)
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    view.setUint16(32, 2, true);  // BlockAlign (NumChannels * BitsPerSample/8)
    view.setUint16(34, 16, true); // BitsPerSample (16 bits)

    // data sub-chunk
    this.writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    // Write 16-bit PCM samples
    let index = 44;
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(index, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      index += 2;
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  stop() {
    this.isRunning = false;
    this.isRecording = false;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
    if (this.scriptProcessor) {
      try { this.scriptProcessor.disconnect(); } catch (e) {}
      this.scriptProcessor = null;
    }
    if (this.muteNode) {
      try { this.muteNode.disconnect(); } catch (e) {}
      this.muteNode = null;
    }
    if (this.sourceNode) {
      try { this.sourceNode.disconnect(); } catch (e) {}
      this.sourceNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      try { this.audioContext.close(); } catch (e) {}
    }
  }

  processLoop() {
    if (!this.isRunning || !this.analyser) return;

    const buffer = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(buffer);

    let sumSquares = 0;
    for (let i = 0; i < buffer.length; i++) {
      sumSquares += buffer[i] * buffer[i];
    }
    const rms = Math.sqrt(sumSquares / buffer.length);
    const energyDb = 20 * Math.log10(Math.max(rms, 0.00001));

    const isSpeaking = rms > this.vadThreshold;
    const now = Date.now();

    if (isSpeaking) {
      this.lastSpeechTime = now;
      this.hasSpokenInTurn = true;

      // Instant Barge-In: if user begins speaking while assistant is outputting audio
      if (this.isAssistantSpeaking && this.onBargeIn) {
        this.isAssistantSpeaking = false;
        this.onBargeIn();
      }
    }

    // Real-Time VAD Trigger: silence after speech (500ms default)
    if (this.isRecording && this.hasSpokenInTurn && (now - this.lastSpeechTime > this.silenceDurationMs)) {
      this.hasSpokenInTurn = false;
      this.stopRecording().then(blob => {
        if (this.onSpeechComplete) {
          this.onSpeechComplete(blob);
        }
      });
    }

    this.currentMetrics = {
      isSpeaking,
      rms,
      energyDb: Math.round(energyDb),
      pitchHz: isSpeaking ? Math.round(this.detectPitch(buffer, this.audioContext.sampleRate)) : 0
    };

    if (this.onMetricsUpdate) {
      this.onMetricsUpdate(this.currentMetrics);
    }

    this.animationFrameId = requestAnimationFrame(() => this.processLoop());
  }

  detectPitch(buffer, sampleRate) {
    const SIZE = buffer.length;
    let r1 = 0, r2 = SIZE - 1;
    for (let i = 0; i < SIZE / 2; i++) {
      if (Math.abs(buffer[i]) < 0.2) { r1 = i; break; }
    }
    for (let i = 1; i < SIZE / 2; i++) {
      if (Math.abs(buffer[SIZE - i]) < 0.2) { r2 = SIZE - i; break; }
    }
    const trimmed = buffer.slice(r1, r2);
    const c = new Float32Array(trimmed.length);
    for (let i = 0; i < trimmed.length; i++) {
      for (let j = 0; j < trimmed.length - i; j++) {
        c[i] = c[i] + trimmed[j] * trimmed[j + i];
      }
    }
    let d = 0;
    while (c[d] > c[d + 1]) d++;
    let maxval = -1, maxpos = -1;
    for (let i = d; i < trimmed.length; i++) {
      if (c[i] > maxval) { maxval = c[i]; maxpos = i; }
    }
    let T0 = maxpos;
    if (T0 <= 0) return 0;
    const x1 = c[T0 - 1], x2 = c[T0], x3 = c[T0 + 1];
    const a = (x1 + x3 - 2 * x2) / 2;
    const b = (x3 - x1) / 2;
    if (a) T0 = T0 - b / (2 * a);
    return sampleRate / T0;
  }
}

window.AudioProcessor = AudioProcessor;
