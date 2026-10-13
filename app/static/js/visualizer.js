/**
 * Futuristic Neural Voice Visualizer
 * Renders an organic, breathing 3D-styled luminous audio orb and fluid neural wave rings.
 * Color shifts dynamically with vocal emotional state and audio energy.
 */

class VoiceVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.state = 'idle'; // 'idle' | 'listening' | 'processing' | 'speaking'
    this.emotion = 'neutral';
    this.audioProcessor = null;
    this.speechService = null;
    this.animationId = null;

    this.time = 0;
    this.baseRadius = 90;
    this.pulseFactor = 0;

    // Emotion Color Palettes (Gradients & Glows)
    this.colorThemes = {
      neutral: {
        primary: 'rgba(0, 242, 254, 0.85)',
        secondary: 'rgba(79, 172, 254, 0.4)',
        glow: '#00f2fe',
        aura: 'rgba(0, 242, 254, 0.15)'
      },
      calm: {
        primary: 'rgba(56, 239, 125, 0.85)',
        secondary: 'rgba(17, 153, 142, 0.4)',
        glow: '#38ef7d',
        aura: 'rgba(56, 239, 125, 0.15)'
      },
      excited: {
        primary: 'rgba(255, 215, 0, 0.9)',
        secondary: 'rgba(255, 126, 95, 0.5)',
        glow: '#ffd700',
        aura: 'rgba(255, 215, 0, 0.22)'
      },
      frustrated: {
        primary: 'rgba(255, 110, 80, 0.9)',
        secondary: 'rgba(255, 175, 64, 0.5)',
        glow: '#ff6e50',
        aura: 'rgba(255, 110, 80, 0.22)'
      },
      sad: {
        primary: 'rgba(255, 126, 179, 0.85)',
        secondary: 'rgba(107, 72, 255, 0.4)',
        glow: '#ff7eb3',
        aura: 'rgba(255, 126, 179, 0.18)'
      },
      confused: {
        primary: 'rgba(178, 36, 239, 0.85)',
        secondary: 'rgba(117, 121, 255, 0.4)',
        glow: '#b224ef',
        aura: 'rgba(178, 36, 239, 0.2)'
      }
    };

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.animate();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = rect.height;
  }

  attachAudio(audioProcessor, speechService) {
    this.audioProcessor = audioProcessor;
    this.speechService = speechService;
  }

  setState(state) {
    this.state = state;
  }

  setEmotion(emotion) {
    this.emotion = this.colorThemes[emotion] ? emotion : 'neutral';
  }

  animate() {
    this.time += 0.03;
    this.render();
    this.animationId = requestAnimationFrame(() => this.animate());
  }

  render() {
    if (!this.ctx) return;
    const { ctx, width, height, time } = this;
    ctx.clearRect(0, 0, width, height);

    const centerX = width / 2;
    const centerY = height / 2;
    const theme = this.colorThemes[this.emotion] || this.colorThemes.neutral;

    // Get live audio energy
    let energy = 0;
    if (this.state === 'listening' && this.audioProcessor && this.audioProcessor.isRunning) {
      energy = this.audioProcessor.currentMetrics.rms * 3.5;
    } else if (this.state === 'speaking') {
      // Synthesize rhythmic speech energy modulation
      energy = 0.35 + Math.sin(time * 6) * 0.2 + Math.cos(time * 11) * 0.15;
    } else if (this.state === 'processing') {
      energy = 0.2 + Math.sin(time * 8) * 0.1;
    }

    energy = Math.min(Math.max(energy, 0), 1.2);
    this.pulseFactor += (energy - this.pulseFactor) * 0.15;

    // 1. Draw Outer Glowing Aura
    const auraRadius = this.baseRadius * (1.3 + this.pulseFactor * 0.8 + Math.sin(time * 1.5) * 0.05);
    const auraGradient = ctx.createRadialGradient(centerX, centerY, this.baseRadius * 0.4, centerX, centerY, auraRadius);
    auraGradient.addColorStop(0, theme.aura);
    auraGradient.addColorStop(0.6, theme.secondary);
    auraGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, auraRadius, 0, Math.PI * 2);
    ctx.fillStyle = auraGradient;
    ctx.fill();
    ctx.restore();

    // 2. Draw Fluid Organic Wave Rings
    const ringCount = 3;
    for (let r = 0; r < ringCount; r++) {
      ctx.save();
      ctx.beginPath();
      const points = 60;
      const ringRadius = this.baseRadius * (0.85 + r * 0.28) * (1 + this.pulseFactor * 0.4);

      for (let i = 0; i <= points; i++) {
        const angle = (i / points) * Math.PI * 2;
        // Multi-frequency harmonic wave deformation
        const wave1 = Math.sin(angle * 5 + time * 3 + r * 1.5) * (8 + this.pulseFactor * 25);
        const wave2 = Math.cos(angle * 3 - time * 2) * (5 + this.pulseFactor * 18);
        const rad = ringRadius + wave1 + wave2;

        const x = centerX + Math.cos(angle) * rad;
        const y = centerY + Math.sin(angle) * rad;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.closePath();
      ctx.strokeStyle = r === 0 ? theme.primary : theme.secondary;
      ctx.lineWidth = r === 0 ? 3.0 : 1.8;
      ctx.shadowColor = theme.glow;
      ctx.shadowBlur = 15 + r * 5;
      ctx.stroke();
      ctx.restore();
    }

    // 3. Draw Core Glowing Orb
    const coreRadius = this.baseRadius * (0.65 + this.pulseFactor * 0.25 + Math.sin(time * 2) * 0.03);
    const coreGrad = ctx.createRadialGradient(
      centerX - coreRadius * 0.2,
      centerY - coreRadius * 0.2,
      2,
      centerX,
      centerY,
      coreRadius
    );
    coreGrad.addColorStop(0, '#ffffff');
    coreGrad.addColorStop(0.3, theme.primary);
    coreGrad.addColorStop(0.85, theme.secondary);
    coreGrad.addColorStop(1, 'rgba(10, 15, 30, 0.9)');

    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, coreRadius, 0, Math.PI * 2);
    ctx.fillStyle = coreGrad;
    ctx.shadowColor = theme.glow;
    ctx.shadowBlur = 30;
    ctx.fill();
    ctx.restore();

    // 4. Subtle center core node
    ctx.save();
    ctx.beginPath();
    ctx.arc(centerX, centerY, 8 + this.pulseFactor * 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = '#ffffff';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.restore();
  }

  destroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}

window.VoiceVisualizer = VoiceVisualizer;
