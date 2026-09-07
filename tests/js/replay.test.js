const test = require('node:test');
const assert = require('node:assert/strict');

const { TDReplay, ReplayManager } = require('../../public/js/replay.js');

test('TDReplay - parseReplay validates structure and move boundaries', () => {
  const replay = new ReplayManager();

  const validPayload = {
    format: "TigerDayReplay",
    version: "1.0",
    players: { british: "Wellesley", mysore: "Tipu" },
    winner: 1,
    moves: [77, 162, 485],
    algebraic: "trv>alw MS:hyd HL:cyl"
  };

  // Valid JSON string
  const parsed = replay.parseReplay(JSON.stringify(validPayload));
  assert.strictEqual(parsed.format, "TigerDayReplay");
  assert.deepStrictEqual(parsed.moves, [77, 162, 485]);

  // Reject non-replay format
  assert.throws(() => {
    replay.parseReplay(JSON.stringify({ format: "Unknown" }));
  }, /not a valid Tiger's Day Replay/);

  // Reject missing moves array
  assert.throws(() => {
    replay.parseReplay(JSON.stringify({ format: "TigerDayReplay" }));
  }, /missing moves array/);

  // Reject invalid move indices (< 0 or >= 959)
  assert.throws(() => {
    replay.parseReplay(JSON.stringify({ format: "TigerDayReplay", moves: [-1] }));
  }, /Invalid move index/);

  assert.throws(() => {
    replay.parseReplay(JSON.stringify({ format: "TigerDayReplay", moves: [959] }));
  }, /Invalid move index/);
});
