const test = require('node:test');
const assert = require('node:assert/strict');

// Ensure mock localStorage exists in Node test runner
if (typeof global.localStorage === 'undefined') {
  const store = {};
  global.localStorage = {
    getItem(key) {
      return Object.prototype.hasOwnProperty.call(store, key) ? store[key] : null;
    },
    setItem(key, val) {
      store[key] = String(val);
    },
    removeItem(key) {
      delete store[key];
    },
    clear() {
      for (const k of Object.keys(store)) delete store[k];
    }
  };
}

const TDSound = require('../../public/js/sound.js');
const SoundEngine = TDSound.SoundEngine;

test('SoundEngine - Volume Range Clamping and NaN protection', () => {
  const engine = new SoundEngine();

  // Excessive value clamped to 1.0
  engine.setVolume(2.5);
  assert.strictEqual(engine.volume, 1.0, 'Volume > 1.0 must be clamped to 1.0');

  // Negative value clamped to 0.0
  engine.setVolume(-0.5);
  assert.strictEqual(engine.volume, 0.0, 'Volume < 0.0 must be clamped to 0.0');

  // Valid intermediate value preserved
  engine.setVolume(0.42);
  assert.strictEqual(Math.round(engine.volume * 100) / 100, 0.42, 'Valid volume must be set correctly');

  // String parsing
  engine.setVolume('0.85');
  assert.strictEqual(Math.round(engine.volume * 100) / 100, 0.85, 'String numeric volume must be parsed and clamped');

  // NaN input clamped to 0.0
  engine.setVolume(NaN);
  assert.strictEqual(engine.volume, 0.0, 'NaN volume input must safely clamp to 0.0');

  // Invalid string clamped to 0.0
  engine.setVolume('invalid');
  assert.strictEqual(engine.volume, 0.0, 'Invalid non-numeric string must clamp to 0.0');
});

test('SoundEngine - Storage Persistence Invariants', () => {
  const engine = new SoundEngine();

  // Test muting persistence
  engine.setMuted(true);
  assert.strictEqual(localStorage.getItem('tigersday_sound_enabled'), 'false');
  assert.strictEqual(engine.enabled, false);

  engine.setMuted(false);
  assert.strictEqual(localStorage.getItem('tigersday_sound_enabled'), 'true');
  assert.strictEqual(engine.enabled, true);

  // Test volume persistence
  engine.setVolume(0.35);
  assert.strictEqual(localStorage.getItem('tigersday_sound_volume'), '0.35');
  assert.strictEqual(engine.volume, 0.35);

  // Re-instantiate engine to verify _loadSettings restores values
  const restoredEngine = new SoundEngine();
  assert.strictEqual(restoredEngine.enabled, true);
  assert.strictEqual(restoredEngine.volume, 0.35);
});

test('SoundEngine - Headless Safe Audio Execution without AudioContext', () => {
  const engine = new SoundEngine();
  engine.setEnabled(true);

  // In headless Node.js without window.AudioContext, none of these should throw
  assert.doesNotThrow(() => engine.playMarch(), 'playMarch should execute safely without AudioContext');
  assert.doesNotThrow(() => engine.playSiegeClash(), 'playSiegeClash should execute safely without AudioContext');
  assert.doesNotThrow(() => engine.playCardPlay(), 'playCardPlay should execute safely without AudioContext');
  assert.doesNotThrow(() => engine.playLuckDiscard(), 'playLuckDiscard should execute safely without AudioContext');
  assert.doesNotThrow(() => engine.playVictory(true), 'playVictory (Mysore) should execute safely without AudioContext');
  assert.doesNotThrow(() => engine.playVictory(false), 'playVictory (British) should execute safely without AudioContext');
  assert.doesNotThrow(() => engine.playClick(), 'playClick should execute safely without AudioContext');
});

test('SoundEngine - Headless Safe Audio Execution with Mock AudioContext', () => {
  // Mock Web Audio Context
  class MockAudioParam {
    constructor() { this.value = 1.0; }
    setValueAtTime() {}
    exponentialRampToValueAtTime() {}
    linearRampToValueAtTime() {}
    setTargetAtTime() {}
  }

  class MockAudioNode {
    constructor() {
      this.frequency = new MockAudioParam();
      this.gain = new MockAudioParam();
      this.Q = new MockAudioParam();
    }
    connect() {}
    start() {}
    stop() {}
  }

  class MockAudioContext {
    constructor() {
      this.currentTime = 0.0;
      this.sampleRate = 44100;
      this.destination = new MockAudioNode();
    }
    createOscillator() { return new MockAudioNode(); }
    createGain() { return new MockAudioNode(); }
    createBiquadFilter() { return new MockAudioNode(); }
    createBufferSource() { return new MockAudioNode(); }
    createBuffer(channels, size, rate) {
      return {
        getChannelData: () => new Float32Array(size)
      };
    }
    resume() { return Promise.resolve(); }
  }

  global.window = global.window || {};
  global.window.AudioContext = MockAudioContext;

  const mockEngine = new SoundEngine();
  mockEngine.setEnabled(true);
  mockEngine.setVolume(0.6);

  assert.doesNotThrow(() => mockEngine.playMarch());
  assert.doesNotThrow(() => mockEngine.playSiegeClash());
  assert.doesNotThrow(() => mockEngine.playCardPlay());
  assert.doesNotThrow(() => mockEngine.playLuckDiscard());
  assert.doesNotThrow(() => mockEngine.playVictory(true));
  assert.doesNotThrow(() => mockEngine.playVictory(false));
  assert.doesNotThrow(() => mockEngine.playClick());
});
