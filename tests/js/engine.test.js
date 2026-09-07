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
