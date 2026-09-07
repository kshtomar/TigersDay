/**
 * Tiger's Day – Procedural Web Audio Sound Engine
 * Zero external audio dependencies – synthesizes authentic 18th-century military soundscapes
 * entirely via the Web Audio API.
 */

(function(global) {
  'use strict';

  class SoundEngine {
    constructor() {
      this.ctx = null;
      this.masterGain = null;
      this.enabled = true;
      this.volume = 0.5;

      this._loadSettings();
    }

    _loadSettings() {
      try {
        const savedEnabled = localStorage.getItem('tigersday_sound_enabled');
        if (savedEnabled !== null) this.enabled = savedEnabled === 'true';

        const savedVol = localStorage.getItem('tigersday_sound_volume');
        if (savedVol !== null) this.volume = Math.max(0, Math.min(1, parseFloat(savedVol)));
      } catch (e) {}
    }

    _saveSettings() {
      try {
        localStorage.setItem('tigersday_sound_enabled', String(this.enabled));
        localStorage.setItem('tigersday_sound_volume', String(this.volume));
      } catch (e) {}
    }

    _initContext() {
      if (!this.ctx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return false;
        this.ctx = new AudioCtx();
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.setValueAtTime(this.volume, this.ctx.currentTime);
        this.masterGain.connect(this.ctx.destination);
      }
      if (this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
      return true;
    }

    setEnabled(enabled) {
      this.enabled = Boolean(enabled);
      this._saveSettings();
    }

    setVolume(vol) {
      this.volume = Math.max(0, Math.min(1, parseFloat(vol)));
      if (this.masterGain && this.ctx) {
        this.masterGain.gain.setTargetAtTime(this.volume, this.ctx.currentTime, 0.05);
      }
      this._saveSettings();
    }

    _createNoiseBuffer(duration = 1.0) {
      if (!this.ctx) return null;
      const bufferSize = this.ctx.sampleRate * duration;
      const buffer = this.ctx.createBuffer(1, bufferSize, this.ctx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = Math.random() * 2 - 1;
      }
      return buffer;
    }

    /**
     * Troop March: Rhythmic military snare & boots on dry Deccan turf
     */
    playMarch() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      // Two consecutive steps
      for (let step = 0; step < 2; step++) {
        const t = now + step * 0.16;

        // Snare/drum low body
        const osc = this.ctx.createOscillator();
        const oscGain = this.ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(110, t);
        osc.frequency.exponentialRampToValueAtTime(45, t + 0.12);

        oscGain.gain.setValueAtTime(0.35, t);
        oscGain.gain.exponentialRampToValueAtTime(0.001, t + 0.14);

        osc.connect(oscGain);
        oscGain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + 0.15);

        // Turf rustle noise
        const noise = this.ctx.createBufferSource();
        noise.buffer = this._createNoiseBuffer(0.15);
        const filter = this.ctx.createBiquadFilter();
        filter.type = 'bandpass';
        filter.frequency.setValueAtTime(700, t);
        filter.Q.setValueAtTime(2.0, t);

        const noiseGain = this.ctx.createGain();
        noiseGain.gain.setValueAtTime(0.2, t);
        noiseGain.gain.exponentialRampToValueAtTime(0.001, t + 0.12);

        noise.connect(filter);
        filter.connect(noiseGain);
        noiseGain.connect(this.masterGain);
        noise.start(t);
        noise.stop(t + 0.13);
      }
    }

    /**
     * Siege Clash: Deep cannon fire rumble with metallic steel clash transient
     */
    playSiegeClash() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      // 1. Low cannon boom
      const boomOsc = this.ctx.createOscillator();
      const boomGain = this.ctx.createGain();
      boomOsc.type = 'sine';
      boomOsc.frequency.setValueAtTime(150, now);
      boomOsc.frequency.exponentialRampToValueAtTime(28, now + 0.6);

      boomGain.gain.setValueAtTime(0.6, now);
      boomGain.gain.exponentialRampToValueAtTime(0.001, now + 0.7);

      boomOsc.connect(boomGain);
      boomGain.connect(this.masterGain);
      boomOsc.start(now);
      boomOsc.stop(now + 0.75);

      // 2. Heavy cannon explosive rumble (filtered brown noise)
      const noise = this.ctx.createBufferSource();
      noise.buffer = this._createNoiseBuffer(0.8);
      const lowFilter = this.ctx.createBiquadFilter();
      lowFilter.type = 'lowpass';
      lowFilter.frequency.setValueAtTime(260, now);
      lowFilter.frequency.exponentialRampToValueAtTime(60, now + 0.7);

      const noiseGain = this.ctx.createGain();
      noiseGain.gain.setValueAtTime(0.7, now);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.75);

      noise.connect(lowFilter);
      lowFilter.connect(noiseGain);
      noiseGain.connect(this.masterGain);
      noise.start(now);
      noise.stop(now + 0.8);

      // 3. Clashing steel transient
      const clashOsc = this.ctx.createOscillator();
      const clashGain = this.ctx.createGain();
      clashOsc.type = 'sawtooth';
      clashOsc.frequency.setValueAtTime(1250, now + 0.05);
      clashOsc.frequency.exponentialRampToValueAtTime(800, now + 0.3);

      clashGain.gain.setValueAtTime(0.18, now + 0.05);
      clashGain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

      clashOsc.connect(clashGain);
      clashGain.connect(this.masterGain);
      clashOsc.start(now + 0.05);
      clashOsc.stop(now + 0.36);
    }

    /**
     * Card Activation: Parchment rustle and wax seal snap
     */
    playCardPlay() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      // Parchment sweep
      const noise = this.ctx.createBufferSource();
      noise.buffer = this._createNoiseBuffer(0.2);
      const filter = this.ctx.createBiquadFilter();
      filter.type = 'bandpass';
      filter.frequency.setValueAtTime(1400, now);
      filter.frequency.linearRampToValueAtTime(800, now + 0.15);

      const gain = this.ctx.createGain();
      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);

      noise.connect(filter);
      filter.connect(gain);
      gain.connect(this.masterGain);
      noise.start(now);
      noise.stop(now + 0.2);

      // Wax seal click
      const snap = this.ctx.createOscillator();
      const snapGain = this.ctx.createGain();
      snap.type = 'triangle';
      snap.frequency.setValueAtTime(600, now + 0.06);
      snap.frequency.exponentialRampToValueAtTime(180, now + 0.12);

      snapGain.gain.setValueAtTime(0.3, now + 0.06);
      snapGain.gain.exponentialRampToValueAtTime(0.001, now + 0.13);

      snap.connect(snapGain);
      snapGain.connect(this.masterGain);
      snap.start(now + 0.06);
      snap.stop(now + 0.14);
    }

    /**
     * Luck Discard: Flintlock misfire / dice rattle clicks
     */
    playLuckDiscard() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      // 3 rapid dry clicks
      for (let i = 0; i < 3; i++) {
        const t = now + i * 0.05;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(1400 - i * 200, t);
        osc.frequency.exponentialRampToValueAtTime(300, t + 0.03);

        gain.gain.setValueAtTime(0.18, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + 0.04);

        osc.connect(gain);
        gain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + 0.045);
      }
    }

    /**
     * Victory: Regimental fanfare or Mysore royal nagara resonance
     */
    playVictory(isMysore = false) {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      if (isMysore) {
        // Mysore Royal Nagara Drums & Gong
        const notes = [65, 82, 98, 131];
        notes.forEach((freq, idx) => {
          const t = now + idx * 0.22;
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, t);
          osc.frequency.exponentialRampToValueAtTime(freq * 0.8, t + 0.6);

          gain.gain.setValueAtTime(0.4, t);
          gain.gain.exponentialRampToValueAtTime(0.001, t + 0.7);

          osc.connect(gain);
          gain.connect(this.masterGain);
          osc.start(t);
          osc.stop(t + 0.75);
        });
      } else {
        // British Regimental Bugle / Fanfare triad
        const fanfare = [261.63, 329.63, 392.00, 523.25]; // C4, E4, G4, C5
        fanfare.forEach((freq, idx) => {
          const t = now + idx * 0.16;
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(freq, t);

          const filter = this.ctx.createBiquadFilter();
          filter.type = 'lowpass';
          filter.frequency.setValueAtTime(1800, t);

          gain.gain.setValueAtTime(0.25, t);
          gain.gain.exponentialRampToValueAtTime(0.001, t + 0.35);

          osc.connect(filter);
          filter.connect(gain);
          gain.connect(this.masterGain);
          osc.start(t);
          osc.stop(t + 0.38);
        });
      }
    }

    /**
     * Subtle UI Click / Parchment tap
     */
    playClick() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(800, now);
      osc.frequency.exponentialRampToValueAtTime(200, now + 0.04);

      gain.gain.setValueAtTime(0.12, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.045);

      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(now);
      osc.stop(now + 0.05);
    }
  }

  // Global instance
  const TDSound = new SoundEngine();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDSound;
  } else {
    global.TDSound = TDSound;
  }
})(typeof window !== 'undefined' ? window : this);
