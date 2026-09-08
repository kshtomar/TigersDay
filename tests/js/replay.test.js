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

test('TDReplay - exportReplay and exportTDR payload generation & normalization', () => {
  const replay = new ReplayManager();

  // Test export with raw move numbers
  const payload1 = replay.exportReplay([77, 162], {
    british: "General Harris",
    mysore: "Tipu Sultan",
    winner: "british",
    filename: "match_001.tdr"
  });
  assert.strictEqual(payload1.format, "TigerDayReplay");
  assert.strictEqual(payload1.version, "1.0");
  assert.deepStrictEqual(payload1.moves, [77, 162]);
  assert.strictEqual(payload1.players.british, "General Harris");
  assert.strictEqual(payload1.winner, "british");

  // Test exportTDR alias with object moves and string filename
  const payload2 = replay.exportTDR([{ moveIdx: 77 }, { moveIdx: 162 }], "test_export.tdr");
  assert.strictEqual(payload2.format, "TigerDayReplay");
  assert.deepStrictEqual(payload2.moves, [77, 162]);

  // Test export with full data object wrapper
  const payload3 = replay.exportTDR({
    metadata: { match_mode: "human_vs_ai" },
    moves: [{ moveIdx: 12 }, { moveIdx: 34 }]
  });
  assert.deepStrictEqual(payload3.moves, [12, 34]);
  assert.strictEqual(payload3.metadata.match_mode, "human_vs_ai");
});

test('TDReplay - loadFromFile interface and HTML action button bindings', () => {
  const fs = require('node:fs');
  const path = require('node:path');

  // Verify TDReplay instance API
  assert.strictEqual(typeof TDReplay.exportReplay, 'function');
  assert.strictEqual(typeof TDReplay.exportTDR, 'function');
  assert.strictEqual(typeof TDReplay.parseReplay, 'function');
  assert.strictEqual(typeof TDReplay.readReplayFile, 'function');
  assert.strictEqual(typeof TDReplay.loadFromFile, 'function');

  // Verify HTML markup contains operational replay buttons
  const htmlPath = path.resolve(__dirname, '../../public/index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');

  assert.ok(html.includes('id="btn-export-tdr"'), 'Must have btn-export-tdr button');
  assert.ok(html.includes('onclick="handleExportTDR()"'), 'btn-export-tdr must trigger handleExportTDR()');
  assert.ok(html.includes('id="btn-import-tdr"'), 'Must have btn-import-tdr button');
  assert.ok(html.includes("document.getElementById('input-import-tdr').click()"), 'btn-import-tdr must trigger hidden file input');
  assert.ok(html.includes('id="input-import-tdr"'), 'Must have hidden input-import-tdr');
  assert.ok(html.includes('onchange="handleImportTDR(event)"'), 'input-import-tdr must trigger handleImportTDR');
});
