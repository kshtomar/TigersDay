const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const { GameState, MOVE_SPACE, MOVE_VECTOR_LENGTH } = require('../../public/js/state.js');
const { TDEngine } = require('../../public/js/engine.js');

test('GameState default setup invariants', () => {
  const s = new GameState();
  s.default_setup();

  assert.strictEqual(s.turn, 1);
  assert.strictEqual(s.to_move, 0);
  assert.strictEqual(s.card_strength, 0);
  assert.strictEqual(s.state[94], 1);

  const str = s.toString();
  assert.strictEqual(str.length, 148);
  assert.strictEqual(str[94], '1');
});

test('GameState serialization roundtrip', () => {
  const s = new GameState();
  s.default_setup();
  const str = s.toString();

  const restored = new GameState();
  restored.read_str(str);

  assert.strictEqual(restored.toString(), str);
  assert.deepStrictEqual(Array.from(s.state), Array.from(restored.state));
});

test('Action space and dispatch table length is 959', () => {
  assert.strictEqual(MOVE_VECTOR_LENGTH, 959);
  assert.strictEqual(TDEngine.ACTION_DISPATCH.length, 959);

  // Check that dispatch items are properly defined
  const first = TDEngine.ACTION_DISPATCH[0];
  assert.strictEqual(first.name, 'Move');
  assert.strictEqual(first.idx, 0);

  const last = TDEngine.ACTION_DISPATCH[958];
  assert.strictEqual(last.name, 'Pass British');
});

test('Repetition check works as expected', () => {
  const s = new GameState();
  s.default_setup();
  const sStr = s.toString();

  const history = [sStr, 'other'];
  assert.strictEqual(TDEngine.checkRepetition(history, s), false);

  history.push(sStr);
  assert.strictEqual(TDEngine.checkRepetition(history, s), true);
});

test('Opening book JSON exists and has entries', () => {
  const bookPath = path.join(__dirname, '../../public/opening_book.json');
  assert.ok(fs.existsSync(bookPath), 'public/opening_book.json should exist');

  const content = JSON.parse(fs.readFileSync(bookPath, 'utf8'));
  assert.ok(Object.keys(content).length > 0, 'Opening book should contain entries');
});

test('GameState read_str input validation throws appropriately', () => {
  const s = new GameState();

  // Too short
  assert.throws(() => s.read_str('0'.repeat(147)), /Invalid bit-string length/);
  // Too long
  assert.throws(() => s.read_str('0'.repeat(149)), /Invalid bit-string length/);
  // Invalid characters
  assert.throws(() => s.read_str('0'.repeat(147) + 'x'), /Bit-string must contain only 1s and 0s/);

  // Conflicting territory ownership (node 0 controlled by both British and Mysore)
  const defaultState = new GameState();
  defaultState.default_setup();
  const bits = Array.from(defaultState.toString());
  bits[0] = '1';
  bits[25] = '1';
  assert.throws(() => s.read_str(bits.join('')), /Multiple units assigned to territory/);
});

test('TDEngine getNextState executes legal moves accurately', () => {
  const s = new GameState();
  s.default_setup();

  const legalMask = TDEngine.getLegalMoves(s);
  assert.ok(legalMask[77] === 1, 'Move 77 should be legal in initial setup');

  const nextState = TDEngine.getNextState(s, 77);
  assert.notStrictEqual(nextState.toString(), s.toString());
  assert.strictEqual(nextState.toString().length, 148);
});

test('TDSound audio engine API and controls', () => {
  const TDSound = require('../../public/js/sound.js');
  assert.ok(TDSound, 'TDSound module should load');
  assert.strictEqual(typeof TDSound.playMarch, 'function');
  assert.strictEqual(typeof TDSound.playSiegeClash, 'function');
  assert.strictEqual(typeof TDSound.playCardPlay, 'function');
  assert.strictEqual(typeof TDSound.playLuckDiscard, 'function');
  assert.strictEqual(typeof TDSound.playVictory, 'function');
  assert.strictEqual(typeof TDSound.playClick, 'function');
  assert.strictEqual(typeof TDSound.setVolume, 'function');
  assert.strictEqual(typeof TDSound.setMuted, 'function');

  TDSound.setVolume(0.8);
  assert.strictEqual(TDSound.volume, 0.8);
  TDSound.setMuted(true);
  assert.strictEqual(TDSound.enabled, false);
});

test('TDThemes and UNIT_STYLES module configuration', () => {
  const TDThemes = require('../../public/js/ui/themes.js');
  assert.ok(TDThemes.THEMES, 'THEMES should be exported');
  assert.ok(TDThemes.UNIT_STYLES, 'UNIT_STYLES should be exported');
  assert.ok(TDThemes.THEMES['deccan-imperial'], 'Default deccan-imperial theme must exist');
  assert.ok(TDThemes.THEMES['midnight-tiger'], 'Midnight tiger theme must exist');
  assert.strictEqual(Object.keys(TDThemes.THEMES).length, 10);
  assert.strictEqual(Object.keys(TDThemes.UNIT_STYLES).length, 5);
});
