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

  // Verify engine getStateWinner safely handles undefined (preventing export crash)
  const { TDEngine } = require('../../public/js/engine.js');
  assert.strictEqual(TDEngine.getStateWinner(undefined), 0);

  // Verify scenarios.js has no unguarded browser require calls
  const scenariosCode = fs.readFileSync(path.resolve(__dirname, '../../public/js/scenarios.js'), 'utf8');
  assert.ok(!scenariosCode.includes('|| require('), 'Must not use unguarded || require() in browser scripts');
});

test('TDReplay - Clean Card Presentation and Absence of Clutter Markers', () => {
  const fs = require('node:fs');
  const path = require('node:path');

  const html = fs.readFileSync(path.resolve(__dirname, '../../public/index.html'), 'utf8');
  const script = fs.readFileSync(path.resolve(__dirname, '../../public/script.js'), 'utf8');
  const css = fs.readFileSync(path.resolve(__dirname, '../../public/style.css'), 'utf8');

  // 1. Verify HTML DOM does NOT contain hand count badges or review banner card lists
  assert.ok(!html.includes('id="mysore-hand-count"'), 'Must not have mysore-hand-count badge in Mysore column');
  assert.ok(!html.includes('id="british-hand-count"'), 'Must not have british-hand-count badge in British column');
  assert.ok(!html.includes('id="mobile-mysore-count"'), 'Must not have mobile-mysore-count badge in mobile tabs');
  assert.ok(!html.includes('id="mobile-british-count"'), 'Must not have mobile-british-count badge in mobile tabs');
  assert.ok(!html.includes('id="review-banner-cards-hud"'), 'Must not have review-banner-cards-hud in review banner');
  assert.ok(!html.includes('id="hist-mysore-cards-list"'), 'Must not have hist-mysore-cards-list text container');
  assert.ok(!html.includes('id="hist-british-cards-list"'), 'Must not have hist-british-cards-list text container');

  // 2. Verify script.js synchronizes lastUiState but does NOT generate redundant status pills or textual HUDs
  assert.ok(script.includes('lastUiState = data.ui_state;'), 'handleHistoricalRender must update lastUiState');
  assert.ok(!script.includes('updateHistoricalCardsHUD'), 'Must not have updateHistoricalCardsHUD');
  assert.ok(!script.includes('card-replay-status'), 'renderCardDeck must NOT generate card-replay-status pill');
  assert.ok(script.includes('card-exhausted-stamp'), 'renderCardDeck must provide natural card-exhausted-stamp');
  assert.ok(script.includes('card-historical-action'), 'renderCardDeck must highlight cards played in that historical move');

  // 3. Verify CSS does not retain cluttered status badge rules
  assert.ok(!css.includes('.review-banner-cards-hud'), 'CSS must not define .review-banner-cards-hud');
  assert.ok(!css.includes('.card-replay-status'), 'CSS must not define .card-replay-status');
  assert.ok(!css.includes('.hand-count-badge'), 'CSS must not define .hand-count-badge');
  assert.ok(css.includes('.card-exhausted-stamp'), 'CSS must style .card-exhausted-stamp');
  assert.ok(css.includes('.player-card.card-historical-action'), 'CSS must style .player-card.card-historical-action');
});

