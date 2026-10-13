/**
 * Voice2Voice Python AI Agent - Real-Time Ultra-Low-Latency Orchestrator
 * Full-duplex WebSocket streaming (/ws/conversation) with HTTP fallback (/api/turn)
 * Features:
 * - Sub-second Time-To-First-Audio (TTFA) via sentence pipelining
 * - Gapless Web Audio API streaming playback
 * - Zero-latency client barge-in & cancellation
 * - Live token-by-token text streaming
 * - Real-time latency telemetry HUD
 */

document.addEventListener('DOMContentLoaded', () => {
  const audioProcessor = new AudioProcessor();
  const visualizer = new VoiceVisualizer('voiceVisualizerCanvas');
  visualizer.attachAudio(audioProcessor, null);

  // DOM Elements
  const btnMainMic = document.getElementById('btnMainMic');
  const toggleHandsFree = document.getElementById('toggleHandsFree');
  const btnInterrupt = document.getElementById('btnInterrupt');
  const btnToggleDrawer = document.getElementById('btnToggleDrawer');
  const btnCloseDrawer = document.getElementById('btnCloseDrawer');
  const btnClearHistory = document.getElementById('btnClearHistory');
  const transcriptDrawer = document.getElementById('transcriptDrawer');
  const transcriptList = document.getElementById('transcriptList');

  const serverBadge = document.getElementById('serverBadge');
  const serverText = document.getElementById('serverText');
  const aiBadge = document.getElementById('aiBadge');
  const aiText = document.getElementById('aiText');
  const emotionBadge = document.getElementById('emotionBadge');
  const emotionText = document.getElementById('emotionText');
  const latencyBadge = document.getElementById('latencyBadge');
  const latencyText = document.getElementById('latencyText');

  const liveSpeechOverlay = document.getElementById('liveSpeechOverlay');
  const liveSpeechText = document.getElementById('liveSpeechText');
  const assistantResponseCard = document.getElementById('assistantResponseCard');
  const assistantSpokenText = document.getElementById('assistantSpokenText');
  const ttsAudioPlayer = document.getElementById('ttsAudioPlayer');

  // Telemetry HUD Elements
  const telemetryDb = document.getElementById('telemetryDb');
  const meterBarEnergy = document.getElementById('meterBarEnergy');
  const telemetryPitch = document.getElementById('telemetryPitch');
  const meterBarPitch = document.getElementById('meterBarPitch');
  const telemetryConfidence = document.getElementById('telemetryConfidence');
  const telemetryEmotionSummary = document.getElementById('telemetryEmotionSummary');
  const telemetryToneBadge = document.getElementById('telemetryToneBadge');
  const telemetryToneRule = document.getElementById('telemetryToneRule');

  // Settings Modal Elements
  const btnOpenSettings = document.getElementById('btnOpenSettings');
  const btnCloseSettings = document.getElementById('btnCloseSettings');
  const btnSaveSettings = document.getElementById('btnSaveSettings');
  const settingsModal = document.getElementById('settingsModal');
  const selectAiProvider = document.getElementById('selectAiProvider');
  const groupGeminiKey = document.getElementById('groupGeminiKey');
  const inputGeminiApiKey = document.getElementById('inputGeminiApiKey');
  const groupGeminiModel = document.getElementById('groupGeminiModel');
  const selectGeminiModel = document.getElementById('selectGeminiModel');
  const selectTtsEngine = document.getElementById('selectTtsEngine');
  const groupEdgeVoice = document.getElementById('groupEdgeVoice');
  const selectEdgeVoice = document.getElementById('selectEdgeVoice');
  const selectVadSilence = document.getElementById('selectVadSilence');
  const selectEmotionToggle = document.getElementById('selectEmotionToggle');
  const settingsAiDiag = document.getElementById('settingsAiDiag');

  // Application State
  let isSessionActive = false;
  let isProcessing = false;
  let conversationHistory = [];
  let ws = null;
  let wsConnected = false;
  let reconnectTimer = null;
  let currentStreamingText = '';

  // -------------------------------------------------------------
  // 1. Web Audio Gapless Streaming Player
  // -------------------------------------------------------------
  class StreamingAudioPlayer {
    constructor() {
      this.audioContext = null;
      this.masterGain = null;
      this.nextStartTime = 0;
      this.activeSources = [];
      this.isPlaying = false;
      this.chunkQueue = [];
      this.isProcessing = false;
      this.isTurnFinished = false;
    }

    initContext() {
      if (!this.audioContext || this.audioContext.state === 'closed') {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx();
        this.masterGain = this.audioContext.createGain();
        this.masterGain.gain.value = 1.0;
        this.masterGain.connect(this.audioContext.destination);
      }
      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }
    }

    reset() {
      this.stopAll();
      this.nextStartTime = 0;
      this.chunkQueue = [];
      this.isPlaying = false;
      this.isProcessing = false;
      this.isTurnFinished = false;
    }

    stopAll() {
      for (const src of this.activeSources) {
        try {
          src.stop();
          src.disconnect();
        } catch (e) {}
      }
      this.activeSources = [];
    }

    async enqueueChunk(arrayBuffer) {
      this.initContext();
      this.chunkQueue.push(arrayBuffer);
      if (!this.isProcessing) {
        this.processQueue();
      }
    }

    async processQueue() {
      if (this.chunkQueue.length === 0) {
        this.isProcessing = false;
        return;
      }

      this.isProcessing = true;
      const bufferToDecode = this.chunkQueue.shift();

      try {
        const copyBuffer = bufferToDecode.slice(0);
        const decoded = await this.audioContext.decodeAudioData(copyBuffer);

        const source = this.audioContext.createBufferSource();
        source.buffer = decoded;
        source.connect(this.masterGain || this.audioContext.destination);

        const now = this.audioContext.currentTime;
        // Schedule next sentence seamlessly with zero gap
        const scheduledStart = Math.max(now, this.nextStartTime);
        source.start(scheduledStart);
        this.nextStartTime = scheduledStart + decoded.duration;

        this.isPlaying = true;
        this.activeSources.push(source);
        audioProcessor.isAssistantSpeaking = true;
        visualizer.setState('speaking');
        btnInterrupt.disabled = false;
        assistantResponseCard.classList.add('speaking');

        source.onended = () => {
          const idx = this.activeSources.indexOf(source);
          if (idx !== -1) this.activeSources.splice(idx, 1);

          if (this.activeSources.length === 0 && this.chunkQueue.length === 0) {
            this.isPlaying = false;
            audioProcessor.isAssistantSpeaking = false;
            if (this.isTurnFinished) {
              onAssistantPlaybackFinished();
            }
          }
        };
      } catch (err) {
        console.warn('Audio sentence chunk decode notice:', err);
      }

      // Continue processing remaining queued sentence chunks
      this.processQueue();
    }
  }

  const streamPlayer = new StreamingAudioPlayer();

  // -------------------------------------------------------------
  // 2. Full-Duplex Real-Time WebSocket Connection
  // -------------------------------------------------------------
  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/conversation`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsConnected = true;
      console.log('Real-Time Voice WebSocket connected.');
      if (latencyText) {
        latencyText.textContent = 'Real-Time: Connected';
      }
      if (latencyBadge) {
        latencyBadge.querySelector('.status-dot').style.backgroundColor = '#10b981';
      }
    };

    ws.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      } catch (e) {
        console.error('WS message parse error:', e);
      }
    };

    ws.onerror = (err) => {
      console.warn('WS error notice:', err);
    };

    ws.onclose = () => {
      wsConnected = false;
      if (latencyBadge) {
        latencyBadge.querySelector('.status-dot').style.backgroundColor = '#f59e0b';
        latencyText.textContent = 'Real-Time: Reconnecting...';
      }
      clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectWebSocket, 2500);
    };
  }

  function handleServerMessage(msg) {
    switch (msg.type) {
      case 'ready':
        if (msg.config) {
          if (msg.config.silence_ms) {
            audioProcessor.silenceDurationMs = msg.config.silence_ms;
          }
        }
        break;

      case 'transcript':
        streamPlayer.isTurnFinished = false;
        liveSpeechText.textContent = `"${msg.text}"`;
        currentStreamingText = '';
        assistantSpokenText.textContent = '...';
        visualizer.setState('processing');
        break;

      case 'llm_token':
        currentStreamingText += msg.token;
        assistantSpokenText.textContent = `"${currentStreamingText}"`;
        assistantResponseCard.classList.add('speaking');
        break;

      case 'audio_chunk':
        if (msg.data) {
          const binaryStr = atob(msg.data);
          const bytes = new Uint8Array(binaryStr.length);
          for (let i = 0; i < binaryStr.length; i++) {
            bytes[i] = binaryStr.charCodeAt(i);
          }
          streamPlayer.enqueueChunk(bytes.buffer);

          if (msg.ttfa_ms && latencyText) {
            latencyText.textContent = `⚡ TTFA: ${Math.round(msg.ttfa_ms)}ms`;
            latencyBadge.querySelector('.status-dot').style.backgroundColor = '#10b981';
          }
        }
        break;

      case 'emotion':
        updateEmotionHUD(msg);
        break;

      case 'turn_complete':
        streamPlayer.isTurnFinished = true;
        if (!streamPlayer.isPlaying && streamPlayer.chunkQueue.length === 0) {
          onAssistantPlaybackFinished();
        }
        if (msg.reply) {
          assistantSpokenText.textContent = `"${msg.reply}"`;
          addTranscriptItem('user', liveSpeechText.textContent.replace(/^"|"$/g, ''), null);
          addTranscriptItem('assistant', msg.reply, null, msg.model, msg.provider);
        }

        if (msg.metrics) {
          const m = msg.metrics;
          latencyText.textContent = `⚡ TTFA: ${m.ttfa_ms}ms | STT: ${m.stt_ms}ms | Total: ${(m.total_ms / 1000).toFixed(1)}s`;
          latencyBadge.title = `STT: ${m.stt_ms}ms\nFirst Token: ${m.ttft_ms}ms\nFirst Audio: ${m.ttfa_ms}ms\nTotal: ${m.total_ms}ms`;
        }
        break;

      case 'interrupted':
        streamPlayer.reset();
        onAssistantPlaybackFinished();
        break;

      case 'turn_ignored':
        isProcessing = false;
        if (isSessionActive && toggleHandsFree.checked) {
          audioProcessor.startRecording();
          visualizer.setState('listening');
        }
        break;

      case 'error':
        console.error('Server turn error:', msg.message);
        liveSpeechText.textContent = `Notice: ${msg.message}`;
        isProcessing = false;
        onAssistantPlaybackFinished();
        break;
    }
  }

  // Initial WebSocket handshake
  connectWebSocket();

  // -------------------------------------------------------------
  // 3. Backend Health & Diagnostics Monitor
  // -------------------------------------------------------------
  async function checkBackendHealth() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        serverText.textContent = `Python: Online (${data.python_version})`;
        serverBadge.querySelector('.status-dot').style.backgroundColor = '#10b981';

        const isGemini = data.provider === 'gemini';
        if (data.ollama && data.ollama.ready) {
          aiText.textContent = isGemini ? `Gemini: Active (${data.selected_model})` : `Ollama: Ready (${data.selected_model})`;
          aiBadge.className = 'badge';
          aiBadge.querySelector('.status-dot').style.backgroundColor = '#10b981';
          settingsAiDiag.textContent = `${data.ollama.details} | Model: ${data.selected_model}`;
        } else {
          const errDetail = data.ollama?.error || data.ollama?.details || 'Offline';
          aiText.textContent = isGemini ? 'Gemini: Active' : 'Ollama: Offline';
          aiBadge.className = 'badge';
          aiBadge.querySelector('.status-dot').style.backgroundColor = isGemini ? '#10b981' : '#ef4444';
          settingsAiDiag.textContent = `${data.ollama?.details || ''} - ${errDetail}`;
        }
      }
    } catch (e) {
      serverText.textContent = 'Python: Offline';
      serverBadge.querySelector('.status-dot').style.backgroundColor = '#ef4444';
      aiText.textContent = 'AI: Unreachable';
    }
  }

  checkBackendHealth();
  setInterval(checkBackendHealth, 12000);

  // -------------------------------------------------------------
  // 4. Audio Processor Telemetry & VAD Handlers
  // -------------------------------------------------------------
  audioProcessor.onMetricsUpdate = (metrics) => {
    const db = metrics.energyDb;
    telemetryDb.textContent = `${db} dB`;
    const energyPercent = Math.min(Math.max((db + 60) * 1.8, 5), 100);
    meterBarEnergy.style.width = `${energyPercent}%`;

    if (metrics.pitchHz > 0) {
      telemetryPitch.textContent = `${metrics.pitchHz} Hz`;
      const pitchPercent = Math.min(Math.max((metrics.pitchHz - 80) / 3, 5), 100);
      meterBarPitch.style.width = `${pitchPercent}%`;
    }
  };

  // Instant Barge-In detection: user spoke while assistant was outputting speech
  audioProcessor.onBargeIn = () => {
    console.log('Instant barge-in detected! Stopping assistant output.');
    cancelAssistantSpeech();
  };

  // VAD Speech Complete: fired as soon as user pauses
  audioProcessor.onSpeechComplete = async (audioBlob) => {
    if (isProcessing) return;
    await processAudioTurn(audioBlob);
  };

  // -------------------------------------------------------------
  // 5. Voice Turn Execution (WebSocket Fast Path + HTTP Fallback)
  // -------------------------------------------------------------
  async function processAudioTurn(audioBlob) {
    if (!audioBlob || audioBlob.size < 800) {
      if (isSessionActive && toggleHandsFree.checked) {
        audioProcessor.startRecording();
      }
      return;
    }

    isProcessing = true;
    visualizer.setState('processing');
    liveSpeechOverlay.classList.add('active');
    liveSpeechText.textContent = 'Listening complete. Processing speech...';
    btnInterrupt.disabled = false;

    // Fast-path: WebSocket streaming
    if (wsConnected && ws && ws.readyState === WebSocket.OPEN) {
      try {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64Audio = reader.result.split(',')[1];
          ws.send(JSON.stringify({
            type: 'audio_turn',
            data: base64Audio
          }));
        };
        reader.readAsDataURL(audioBlob);
        return;
      } catch (wsErr) {
        console.warn('WS send failed, falling back to HTTP:', wsErr);
      }
    }

    // Fallback path: HTTP multipart POST /api/turn
    const formData = new FormData();
    const filename = audioBlob.type.includes('wav') ? 'recording.wav' : 'recording.webm';
    formData.append('file', audioBlob, filename);
    formData.append('history', JSON.stringify(conversationHistory));

    try {
      const res = await fetch('/api/turn', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}`);
      }

      const turn = await res.json();

      liveSpeechText.textContent = `"${turn.transcript}"`;
      addTranscriptItem('user', turn.transcript, turn.emotion);
      updateEmotionHUD(turn.emotion);

      assistantSpokenText.textContent = `"${turn.reply}"`;
      assistantResponseCard.classList.add('speaking');
      addTranscriptItem('assistant', turn.reply, null, turn.model, turn.provider);

      conversationHistory.push({ role: 'user', content: turn.transcript });
      conversationHistory.push({ role: 'assistant', content: turn.reply });
      if (conversationHistory.length > 20) conversationHistory = conversationHistory.slice(-20);

      playGeneratedAudioUrl(turn.audio_url);

    } catch (err) {
      console.error('Turn error:', err);
      liveSpeechText.textContent = `Notice: ${err.message}`;
      assistantSpokenText.textContent = `Error: ${err.message}`;
      visualizer.setState('idle');
      isProcessing = false;

      if (isSessionActive && toggleHandsFree.checked) {
        setTimeout(() => {
          if (isSessionActive && !isProcessing) {
            audioProcessor.startRecording();
            visualizer.setState('listening');
          }
        }, 1200);
      }
    }
  }

  // Play whole audio URL (HTTP turn fallback)
  function playGeneratedAudioUrl(audioUrl) {
    visualizer.setState('speaking');
    btnInterrupt.disabled = false;
    audioProcessor.isAssistantSpeaking = true;

    ttsAudioPlayer.src = audioUrl;
    ttsAudioPlayer.play().catch(e => {
      console.warn('Audio auto-play notice:', e);
      onAssistantPlaybackFinished();
    });

    ttsAudioPlayer.onended = () => {
      onAssistantPlaybackFinished();
    };
  }

  function onAssistantPlaybackFinished() {
    btnInterrupt.disabled = true;
    assistantResponseCard.classList.remove('speaking');
    audioProcessor.isAssistantSpeaking = false;
    isProcessing = false;

    if (isSessionActive && toggleHandsFree.checked) {
      visualizer.setState('listening');
      liveSpeechText.textContent = 'Listening to your voice...';
      audioProcessor.startRecording();
    } else {
      visualizer.setState('idle');
      liveSpeechText.textContent = 'Ready. Click microphone to speak.';
    }
  }

  // -------------------------------------------------------------
  // 6. User Barge-In & Cancellation
  // -------------------------------------------------------------
  function cancelAssistantSpeech() {
    // 1. Immediately kill client audio sources
    streamPlayer.reset();
    ttsAudioPlayer.pause();
    ttsAudioPlayer.currentTime = 0;
    audioProcessor.isAssistantSpeaking = false;

    // 2. Tell backend to abort running LLM/TTS generation
    if (wsConnected && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'cancel' }));
    }

    // 3. Reset visual state to listening
    assistantResponseCard.classList.remove('speaking');
    btnInterrupt.disabled = true;
    isProcessing = false;

    if (isSessionActive) {
      visualizer.setState('listening');
      liveSpeechText.textContent = 'Interrupted. Listening to you...';
      audioProcessor.startRecording();
    }
  }

  btnInterrupt.addEventListener('click', () => {
    cancelAssistantSpeech();
  });

  // -------------------------------------------------------------
  // 7. Microphone Controls
  // -------------------------------------------------------------
  btnMainMic.addEventListener('click', async () => {
    // Ensure Web Audio context is primed on user interaction gesture
    streamPlayer.initContext();

    if (!isSessionActive) {
      try {
        await audioProcessor.start();
        audioProcessor.startRecording();
        isSessionActive = true;
        btnMainMic.classList.add('active');
        visualizer.setState('listening');
        liveSpeechOverlay.classList.add('active');
        liveSpeechText.textContent = 'Listening to your voice... say anything!';
      } catch (err) {
        alert('Please allow microphone permissions to communicate with the assistant.');
      }
    } else {
      if (!toggleHandsFree.checked && audioProcessor.isRecording) {
        // Push-to-talk turn completion
        const blob = await audioProcessor.stopRecording();
        btnMainMic.classList.remove('active');
        isSessionActive = false;
        await processAudioTurn(blob);
      } else {
        // Stop session
        audioProcessor.stop();
        streamPlayer.reset();
        ttsAudioPlayer.pause();
        isSessionActive = false;
        btnMainMic.classList.remove('active');
        visualizer.setState('idle');
        liveSpeechText.textContent = 'Session paused. Click microphone to resume.';
      }
    }
  });

  // -------------------------------------------------------------
  // 8. Emotion HUD Rendering
  // -------------------------------------------------------------
  function updateEmotionHUD(emotionData) {
    if (!emotionData) return;
    const emo = emotionData.emotion || 'neutral';
    visualizer.setEmotion(emo);

    emotionBadge.className = `badge badge-emotion ${emo}`;
    const icons = {
      neutral: '🌿',
      calm: '🌊',
      excited: '⚡',
      frustrated: '🔥',
      sad: '🌧️',
      confused: '❓',
      worried: '🌧️'
    };
    emotionBadge.querySelector('.emotion-icon').textContent = icons[emo] || '🌿';
    emotionText.textContent = emotionData.summary || emo.toUpperCase();

    telemetryConfidence.textContent = `${Math.round(emotionData.confidence * 100)}%`;
    telemetryEmotionSummary.textContent = `${emotionData.summary} (Acoustic Estimate)`;

    const toneMap = {
      frustrated: { tone: 'Calm & Direct', rule: 'Clear, reassuring resolution' },
      sad: { tone: 'Warm & Patient', rule: 'Gentle empathy and calm clarity' },
      worried: { tone: 'Reassuring & Steady', rule: 'Steady comfort and clarity' },
      excited: { tone: 'Enthusiastic & Bright', rule: 'Engaging, upbeat response' },
      confused: { tone: 'Step-by-Step & Simple', rule: 'Clear, straightforward explanations' },
      neutral: { tone: 'Composed & Natural', rule: 'Conversational spoken English' }
    };
    const info = toneMap[emo] || toneMap.neutral;
    telemetryToneBadge.textContent = info.tone;
    telemetryToneRule.textContent = info.rule;
  }

  // -------------------------------------------------------------
  // 9. Transcript History Drawer
  // -------------------------------------------------------------
  btnToggleDrawer.addEventListener('click', () => {
    transcriptDrawer.classList.toggle('open');
  });

  btnCloseDrawer.addEventListener('click', () => {
    transcriptDrawer.classList.remove('open');
  });

  btnClearHistory.addEventListener('click', () => {
    conversationHistory = [];
    transcriptList.innerHTML = '<div class="empty-state"><p>Conversation cleared. Speak to start fresh.</p></div>';
  });

  function addTranscriptItem(role, text, emotionData = null, model = null, provider = null) {
    const emptyState = transcriptList.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const bubble = document.createElement('div');
    bubble.className = `transcript-bubble ${role}`;

    const contentP = document.createElement('p');
    contentP.textContent = text;
    bubble.appendChild(contentP);

    const meta = document.createElement('div');
    meta.className = 'bubble-meta';

    const timeSpan = document.createElement('span');
    const now = new Date();
    timeSpan.textContent = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
    meta.appendChild(timeSpan);

    if (role === 'user' && emotionData && emotionData.emotion) {
      const emoSpan = document.createElement('span');
      emoSpan.className = 'bubble-emotion-tag';
      emoSpan.textContent = `Acoustic cue: ${emotionData.emotion} (${Math.round(emotionData.confidence * 100)}%)`;
      meta.appendChild(emoSpan);
    } else if (role === 'assistant' && model) {
      const srcSpan = document.createElement('span');
      const provLabel = provider ? (provider === 'gemini' ? 'Gemini' : 'Ollama') : (model.includes('gemini') ? 'Gemini' : 'Ollama');
      srcSpan.textContent = `${provLabel}: ${model}`;
      meta.appendChild(srcSpan);
    }

    bubble.appendChild(meta);
    transcriptList.appendChild(bubble);
    transcriptList.scrollTop = transcriptList.scrollHeight;
  }

  // -------------------------------------------------------------
  // 10. Settings Configuration Modal
  // -------------------------------------------------------------
  btnOpenSettings.addEventListener('click', async () => {
    settingsModal.classList.add('open');
    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const cfg = await res.json();
        selectAiProvider.value = cfg.provider || 'gemini';
        selectGeminiModel.value = (cfg.gemini_model && !cfg.gemini_model.includes('2.0')) ? cfg.gemini_model : 'gemini-3.8-flash';

        if (cfg.gemini_api_key_set && !inputGeminiApiKey.value) {
          inputGeminiApiKey.placeholder = 'Key is saved (Enter new key to change)';
        }

        if (selectTtsEngine && cfg.tts_engine) {
          selectTtsEngine.value = cfg.tts_engine;
        }
        if (selectEdgeVoice && cfg.edge_voice) {
          selectEdgeVoice.value = cfg.edge_voice;
        }
        if (selectVadSilence && cfg.vad_silence_ms) {
          selectVadSilence.value = String(cfg.vad_silence_ms);
          audioProcessor.silenceDurationMs = parseInt(cfg.vad_silence_ms);
        }
        if (selectEmotionToggle && cfg.emotion_enabled !== undefined) {
          selectEmotionToggle.value = cfg.emotion_enabled ? 'true' : 'false';
        }

        updateSettingsFieldsVisibility();
      }
    } catch (e) {
      console.warn('Failed to load settings:', e);
    }
    checkBackendHealth();
  });

  selectAiProvider.addEventListener('change', updateSettingsFieldsVisibility);
  if (selectTtsEngine) {
    selectTtsEngine.addEventListener('change', updateSettingsFieldsVisibility);
  }

  function updateSettingsFieldsVisibility() {
    const isGemini = selectAiProvider.value === 'gemini';
    groupGeminiKey.style.display = isGemini ? 'flex' : 'none';
    groupGeminiModel.style.display = isGemini ? 'flex' : 'none';

    if (groupEdgeVoice && selectTtsEngine) {
      groupEdgeVoice.style.display = selectTtsEngine.value === 'edge-tts' ? 'flex' : 'none';
    }
  }

  btnCloseSettings.addEventListener('click', () => {
    settingsModal.classList.remove('open');
  });

  btnSaveSettings.addEventListener('click', async () => {
    btnSaveSettings.textContent = 'Saving...';
    try {
      const payload = {
        provider: selectAiProvider.value,
        gemini_model: selectGeminiModel.value,
        tts_engine: selectTtsEngine ? selectTtsEngine.value : 'edge-tts',
        edge_voice: selectEdgeVoice ? selectEdgeVoice.value : 'en-IN-NeerjaNeural',
        vad_silence_ms: selectVadSilence ? parseInt(selectVadSilence.value) : 500,
        emotion_enabled: selectEmotionToggle ? selectEmotionToggle.value === 'true' : false
      };

      if (inputGeminiApiKey.value.trim()) {
        payload.gemini_api_key = inputGeminiApiKey.value.trim();
      }

      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        if (payload.vad_silence_ms) {
          audioProcessor.silenceDurationMs = payload.vad_silence_ms;
        }

        btnSaveSettings.textContent = 'Saved!';
        setTimeout(() => {
          btnSaveSettings.textContent = 'Save Settings';
          settingsModal.classList.remove('open');
          checkBackendHealth();
        }, 500);
      } else {
        throw new Error('Server returned ' + res.status);
      }
    } catch (err) {
      alert('Failed to save settings: ' + err.message);
      btnSaveSettings.textContent = 'Save Settings';
    }
  });
});
