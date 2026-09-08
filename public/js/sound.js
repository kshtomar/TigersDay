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
        if (typeof window === 'undefined') return false;
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

    setMuted(muted) {
      this.setEnabled(!muted);
    }

    setVolume(vol) {
      let parsed = parseFloat(vol);
      this.volume = isNaN(parsed) ? 0.0 : Math.max(0, Math.min(1, parsed));
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
     * Winning: Triumphant celebratory fanfare (British) or Mysore Royal Nagara resonance
     */
    playWin(isMysore = false) {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      if (isMysore) {
        // Mysore Royal Nagara Drums & Triumphal Gong Flourish
        const drumNotes = [55, 73, 98, 131];
        drumNotes.forEach((freq, idx) => {
          const t = now + idx * 0.18;
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(freq, t);
          osc.frequency.exponentialRampToValueAtTime(freq * 0.75, t + 0.55);

          gain.gain.setValueAtTime(0.45, t);
          gain.gain.exponentialRampToValueAtTime(0.001, t + 0.65);

          osc.connect(gain);
          gain.connect(this.masterGain);
          osc.start(t);
          osc.stop(t + 0.7);
        });

        // Ascending pentatonic victory brass/gong chime
        const flourish = [196, 261.63, 329.63, 392, 523.25];
        flourish.forEach((freq, idx) => {
          const t = now + 0.35 + idx * 0.12;
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'triangle';
          osc.frequency.setValueAtTime(freq, t);

          gain.gain.setValueAtTime(0.28, t);
          gain.gain.exponentialRampToValueAtTime(0.001, t + 0.6);

          osc.connect(gain);
          gain.connect(this.masterGain);
          osc.start(t);
          osc.stop(t + 0.65);
        });
      } else {
        // British Regimental Brass Fanfare with Snare Roll
        for (let s = 0; s < 4; s++) {
          const st = now + s * 0.06;
          const snareNoise = this.ctx.createBufferSource();
          snareNoise.buffer = this._createNoiseBuffer(0.05);
          const sFilter = this.ctx.createBiquadFilter();
          sFilter.type = 'highpass';
          sFilter.frequency.setValueAtTime(1000, st);

          const sGain = this.ctx.createGain();
          sGain.gain.setValueAtTime(0.18, st);
          sGain.gain.exponentialRampToValueAtTime(0.001, st + 0.05);

          snareNoise.connect(sFilter);
          sFilter.connect(sGain);
          sGain.connect(this.masterGain);
          snareNoise.start(st);
          snareNoise.stop(st + 0.06);
        }

        // Triumphant bugle fanfare triad (C4, E4, G4, C5, E5)
        const fanfare = [261.63, 329.63, 392.00, 523.25, 659.25];
        fanfare.forEach((freq, idx) => {
          const t = now + 0.22 + idx * 0.14;
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(freq, t);

          const filter = this.ctx.createBiquadFilter();
          filter.type = 'lowpass';
          filter.frequency.setValueAtTime(2200, t);
          filter.frequency.exponentialRampToValueAtTime(1200, t + (idx === fanfare.length - 1 ? 0.8 : 0.28));

          const dur = (idx === fanfare.length - 1) ? 0.9 : 0.32;
          gain.gain.setValueAtTime(0.32, t);
          gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

          osc.connect(filter);
          filter.connect(gain);
          gain.connect(this.masterGain);
          osc.start(t);
          osc.stop(t + dur + 0.05);
        });
      }
    }

    /**
     * Victory alias for backward compatibility
     */
    playVictory(isMysore = false) {
      this.playWin(isMysore);
    }

    /**
     * Defeated: Somber, mournful descending minor military cadence
     */
    playDefeat() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      // Heavy muffled funeral drum thud
      const drumOsc = this.ctx.createOscillator();
      const drumGain = this.ctx.createGain();
      drumOsc.type = 'sine';
      drumOsc.frequency.setValueAtTime(80, now);
      drumOsc.frequency.exponentialRampToValueAtTime(32, now + 0.5);

      drumGain.gain.setValueAtTime(0.55, now);
      drumGain.gain.exponentialRampToValueAtTime(0.001, now + 0.65);

      drumOsc.connect(drumGain);
      drumGain.connect(this.masterGain);
      drumOsc.start(now);
      drumOsc.stop(now + 0.7);

      // Mournful descending minor brass motif: D4 -> Bb3 -> G3 -> Eb3 -> D3
      const minorNotes = [293.66, 233.08, 196.00, 155.56, 146.83];
      minorNotes.forEach((freq, idx) => {
        const t = now + 0.12 + idx * 0.24;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, t);

        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(900, t);
        filter.frequency.exponentialRampToValueAtTime(350, t + 0.45);

        const noteDur = idx === minorNotes.length - 1 ? 0.9 : 0.35;
        gain.gain.setValueAtTime(0.3, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + noteDur);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + noteDur + 0.05);
      });
    }

    /**
     * Resign: Tactical surrender / yielding weapons with ceasefire signal
     */
    playResign() {
      if (!this.enabled || !this._initContext()) return;
      const now = this.ctx.currentTime;

      // 1. Sword sheathing / weapon yielding metallic friction transient
      const noise = this.ctx.createBufferSource();
      noise.buffer = this._createNoiseBuffer(0.2);
      const bandpass = this.ctx.createBiquadFilter();
      bandpass.type = 'bandpass';
      bandpass.frequency.setValueAtTime(2400, now);
      bandpass.frequency.exponentialRampToValueAtTime(800, now + 0.16);
      bandpass.Q.setValueAtTime(4.0, now);

      const noiseGain = this.ctx.createGain();
      noiseGain.gain.setValueAtTime(0.35, now);
      noiseGain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);

      noise.connect(bandpass);
      bandpass.connect(noiseGain);
      noiseGain.connect(this.masterGain);
      noise.start(now);
      noise.stop(now + 0.2);

      // 2. Ceasefire white-flag bugle call: Descending solemn two-tone signal (A3 -> E3)
      const tones = [220.00, 164.81];
      tones.forEach((freq, idx) => {
        const t = now + 0.15 + idx * 0.28;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(freq, t);

        const filter = this.ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(1100, t);
        filter.frequency.exponentialRampToValueAtTime(400, t + 0.4);

        const dur = idx === 1 ? 0.75 : 0.35;
        gain.gain.setValueAtTime(0.25, t);
        gain.gain.exponentialRampToValueAtTime(0.001, t + dur);

        osc.connect(filter);
        filter.connect(gain);
        gain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + dur + 0.05);
      });

      // 3. Low hollow surrender rim stroke
      const rimOsc = this.ctx.createOscillator();
      const rimGain = this.ctx.createGain();
      rimOsc.type = 'triangle';
      rimOsc.frequency.setValueAtTime(140, now + 0.12);
      rimOsc.frequency.exponentialRampToValueAtTime(50, now + 0.25);

      rimGain.gain.setValueAtTime(0.28, now + 0.12);
      rimGain.gain.exponentialRampToValueAtTime(0.001, now + 0.26);

      rimOsc.connect(rimGain);
      rimGain.connect(this.masterGain);
      rimOsc.start(now + 0.12);
      rimOsc.stop(now + 0.28);
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
    module.exports.SoundEngine = SoundEngine;
  } else {
    global.TDSound = TDSound;
    global.SoundEngine = SoundEngine;
  }
})(typeof window !== 'undefined' ? window : this);
