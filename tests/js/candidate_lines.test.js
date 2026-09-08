const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const { GameState, MOVE_SPACE } = require('../../public/js/state.js');
const { TDEngine } = require('../../public/js/engine.js');
const { MCTS, MCTSNode, decodeMoveGeometry } = require('../../public/js/mcts.js');

test('decodeMoveGeometry decodes edge moves and captures', () => {
  const s = new GameState();
  s.default_setup();

  // Move 0: from edge 0 (Madras -> Vellore)
  const geom0 = decodeMoveGeometry(s, 0);
  assert.strictEqual(geom0.type, 'Move');
  assert.strictEqual(geom0.fromNode, 0); // Madras
  assert.strictEqual(geom0.toNode, 5);   // Vellore
  assert.strictEqual(typeof geom0.isAttack, 'boolean');

  // Coastal Naval operation (e.g. Royal Navy)
  const geomRN = decodeMoveGeometry(s, 487); // 487 is RN Madras
  assert.strictEqual(geomRN.type, 'Royal Navy');
  assert.strictEqual(geomRN.toNode, 0);

  // Pass move (958)
  const geomPass = decodeMoveGeometry(s, 958);
  assert.strictEqual(geomPass.type, 'Pass British');
  assert.strictEqual(geomPass.fromNode, null);
  assert.strictEqual(geomPass.toNode, null);

  // Fortress attack detection: place fort on node 5 (Satara)
  const sAttack = s.copy();
  sAttack.set_node_fort(5);
  const geomAttack = decodeMoveGeometry(sAttack, 0);
  assert.strictEqual(geomAttack.isAttack, true);

  // Card target: Highlanders (single node)
  const geomHighlanders = decodeMoveGeometry(s, 462); // 462 is Highlanders at node 0
  assert.strictEqual(geomHighlanders.type, 'Highlanders');
  assert.strictEqual(geomHighlanders.fromNode, 0);
  assert.strictEqual(geomHighlanders.toNode, 0);

  // Coastal: Sea Trade
  const geomST = decodeMoveGeometry(s, 187);
  assert.strictEqual(geomST.type, 'Sea Trade');
  assert.ok(geomST.fromNode !== null);
  assert.ok(geomST.toNode !== null);
});

test('MCTS getTopCandidateLines respects K limits and traces notation', () => {
  const s = new GameState();
  s.default_setup();

  const mcts = new MCTS(null);
  const root = new MCTSNode(s);
  root.is_expanded = true;

  // Add 5 children with descending visits
  for (let i = 0; i < 5; i++) {
    const child = new MCTSNode(null, root, i, 0.2);
    child.visit_count = 50 - i * 5;
    child.value_sum = (50 - i * 5) * 0.1;
    root.children.set(i, child);
  }
  mcts.root = root;

  // Test limit = 1
  const top1 = mcts.getTopCandidateLines(1);
  assert.strictEqual(top1.length, 1);
  assert.strictEqual(top1[0].rank, 1);
  assert.ok(top1[0].firstMoveNotation);
  assert.ok(top1[0].lineNotation);
  assert.strictEqual(top1[0].visits, 50);

  // Test limit = 3
  const top3 = mcts.getTopCandidateLines(3);
  assert.strictEqual(top3.length, 3);
  assert.strictEqual(top3[0].rank, 1);
  assert.strictEqual(top3[1].rank, 2);
  assert.strictEqual(top3[2].rank, 3);

  // Test limit = 5
  const top5 = mcts.getTopCandidateLines(5);
  assert.strictEqual(top5.length, 5);
  for (let i = 0; i < top5.length; i++) {
    assert.strictEqual(top5[i].rank, i + 1);
    assert.ok(typeof top5[i].eval === 'number');
    assert.ok(typeof top5[i].winrate === 'string');
    assert.ok(top5[i].actionType);
  }

  // Clamping when k exceeds available children (e.g. limit = 10 on 5 children)
  const topClamped = mcts.getTopCandidateLines(10);
  assert.strictEqual(topClamped.length, 5);
});

test('MCTS getTopCandidateLines handles empty or unsearched tree safely', () => {
  const mcts = new MCTS(null);
  assert.deepStrictEqual(mcts.getTopCandidateLines(3), []);

  const s = new GameState();
  s.default_setup();
  mcts.root = new MCTSNode(s);
  // Root exists but has 0 children
  assert.deepStrictEqual(mcts.getTopCandidateLines(3), []);
});

test('MCTS getTopCandidateLines traces multi-ply variation lines', () => {
  const s = new GameState();
  s.default_setup();

  const mcts = new MCTS(null);
  const root = new MCTSNode(s);
  root.is_expanded = true;

  // Move 0 leads to state1
  const s1 = s.copy();
  s1.turn = 1;
  s1.to_move = 1; // Mysore to move
  const child0 = new MCTSNode(s1, root, 0, 0.5);
  child0.visit_count = 100;
  child0.is_expanded = true;

  // Deep child: Mysore moves pass (move 461)
  const s2 = s1.copy();
  s2.to_move = 2; // British Card
  const grandChild = new MCTSNode(s2, child0, 461, 0.5);
  grandChild.visit_count = 80;
  child0.children.set(461, grandChild);

  root.children.set(0, child0);
  mcts.root = root;

  const lines = mcts.getTopCandidateLines(1);
  assert.strictEqual(lines.length, 1);
  // Should contain both plies
  assert.ok(lines[0].lineNotation.includes('bom>sat'));
  assert.ok(lines[0].lineNotation.includes('pass'));
});

test('MCTS getTopCandidateLines stops at luck states and adds luck indicator', () => {
  const s = new GameState();
  s.default_setup();

  const mcts = new MCTS(null);
  const root = new MCTSNode(s);
  root.is_expanded = true;

  // Create next state that is a luck state
  const nextState = s.copy();
  nextState.bluck = 1;
  nextState.attacker = 0;
  nextState.defender = 1; // Hyderabad

  const child = new MCTSNode(nextState, root, 0, 1.0);
  child.visit_count = 20;
  child.value_sum = 5;
  root.children.set(0, child);
  mcts.root = root;

  const candidates = mcts.getTopCandidateLines(1);
  assert.strictEqual(candidates.length, 1);
  const cand = candidates[0];
  assert.strictEqual(cand.reachedLuck, true);
  assert.ok(cand.lineNotation.includes('🎲 Battle: Hyderabad'));
});

test('HTML template includes candidate moves layer and UI settings controls', () => {
  const htmlPath = path.resolve(__dirname, '../../public/index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');

  // SVG candidate moves layer
  assert.ok(html.includes('id="candidate-moves-layer"'), 'Missing #candidate-moves-layer in SVG');
  assert.ok(html.includes('id="engine-lines-container"'), 'Missing #engine-lines-container in eval panel');

  // SVG marker defs
  for (let k = 1; k <= 5; k++) {
    assert.ok(html.includes(`id="ai-arrow-${k}"`), `Missing SVG marker definition #ai-arrow-${k}`);
  }

  // Settings dropdown and toggle
  assert.ok(html.includes('id="top-k-candidates-select"'), 'Missing #top-k-candidates-select');
  assert.ok(html.includes('id="show-candidate-arrows-checkbox"'), 'Missing #show-candidate-arrows-checkbox');
});

test('CSS includes styling rules for candidate lines and tactical arrows', () => {
  const cssPath = path.resolve(__dirname, '../../public/style.css');
  const css = fs.readFileSync(cssPath, 'utf8');

  assert.ok(css.includes('.engine-lines-container'), 'Missing .engine-lines-container in style.css');
  assert.ok(css.includes('.engine-candidate-card'), 'Missing .engine-candidate-card in style.css');
  assert.ok(css.includes('.candidate-arrow'), 'Missing .candidate-arrow in style.css');
  assert.ok(css.includes('.candidate-rank-badge'), 'Missing .candidate-rank-badge in style.css');
  assert.ok(css.includes('.luck-badge-pill'), 'Missing .luck-badge-pill in style.css');
});

test('Client script defines top-k settings and default values', () => {
  const scriptPath = path.resolve(__dirname, '../../public/script.js');
  const script = fs.readFileSync(scriptPath, 'utf8');

  assert.ok(script.includes('topKCandidates:'), 'Missing topKCandidates setting');
  assert.ok(script.includes('showCandidateArrows:'), 'Missing showCandidateArrows setting');
  assert.ok(script.includes('renderCandidateArrows'), 'Missing renderCandidateArrows function');
  assert.ok(script.includes('clearCandidateArrows'), 'Missing clearCandidateArrows function');
  assert.ok(script.includes('getRankColor'), 'Missing getRankColor function for arbitrary k');
  assert.ok(script.includes('ensureArrowMarker'), 'Missing ensureArrowMarker function for dynamic SVG markers');
});

test('MCTS getTopCandidateLines supports arbitrary positive integer K and handles fewer moves than K', () => {
  const mockModel = { predict: () => ({ policy: new Float32Array(959).fill(1 / 959), value: 0.0 }) };
  const mcts = new MCTS(mockModel, { simulations: 5 });

  const s = new GameState();
  const root = new MCTSNode(s, null, null, 1.0);

  // Add exactly 3 children
  for (let i = 0; i < 3; i++) {
    const nextState = s.copy();
    const child = new MCTSNode(nextState, root, i, 1.0);
    child.visit_count = 10 - i;
    child.value_sum = 2;
    root.children.set(i, child);
  }
  mcts.root = root;

  // When k is larger than available moves (e.g. k=10, k=50), it safely outputs all 3 valid lines
  const candidatesK10 = mcts.getTopCandidateLines(10);
  assert.strictEqual(candidatesK10.length, 3, 'Should output all 3 valid lines when k=10 > 3');

  const candidatesK50 = mcts.getTopCandidateLines(50);
  assert.strictEqual(candidatesK50.length, 3, 'Should output all 3 valid lines when k=50 > 3');

  // When k is smaller (e.g. k=1, k=2), it outputs exactly k lines
  const candidatesK1 = mcts.getTopCandidateLines(1);
  assert.strictEqual(candidatesK1.length, 1);
  assert.strictEqual(candidatesK1[0].rank, 1);

  const candidatesK2 = mcts.getTopCandidateLines(2);
  assert.strictEqual(candidatesK2.length, 2);
  assert.strictEqual(candidatesK2[1].rank, 2);
});

test('DOM Architecture places eval bar & candidate lines in bottom-analysis-dock, not notation panel', () => {
  const htmlPath = path.resolve(__dirname, '../../public/index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');

  // #bottom-analysis-dock must exist and contain #eval-panel and #engine-analysis-section
  assert.ok(html.includes('id="bottom-analysis-dock"'), 'Missing #bottom-analysis-dock');
  const dockIdx = html.indexOf('id="bottom-analysis-dock"');
  const evalIdx = html.indexOf('id="eval-panel"');
  const engineIdx = html.indexOf('id="engine-analysis-section"');
  const notationIdx = html.indexOf('id="notation-panel"');

  assert.ok(dockIdx < evalIdx, '#eval-panel must be inside #bottom-analysis-dock');
  assert.ok(dockIdx < engineIdx, '#engine-analysis-section must be inside #bottom-analysis-dock');
  assert.ok(engineIdx < notationIdx, '#engine-analysis-section must precede #notation-panel');

  // #notation-tab-moves-content must NOT contain #engine-analysis-section
  const notationContent = html.substring(notationIdx);
  assert.ok(!notationContent.includes('id="engine-analysis-section"'), '#engine-analysis-section must NOT be in notation panel');
});

test('MCTS getTopCandidateLines boundary values, string inputs, and monotonicity', () => {
  const mockModel = { predict: () => ({ policy: new Float32Array(959).fill(1 / 959), value: 0.0 }) };
  const mcts = new MCTS(mockModel, { simulations: 5 });
  const s = new GameState();
  const root = new MCTSNode(s, null, null, 1.0);

  // Add 4 children with distinct visit counts
  const visitCounts = [45, 30, 15, 8];
  for (let i = 0; i < visitCounts.length; i++) {
    const nextState = s.copy();
    const child = new MCTSNode(nextState, root, i, 1.0);
    child.visit_count = visitCounts[i];
    child.value_sum = visitCounts[i] * 0.2;
    root.children.set(i, child);
  }
  mcts.root = root;

  // String input limit e.g. "3"
  const candStr = mcts.getTopCandidateLines("3");
  assert.strictEqual(candStr.length, 3);

  // Negative or zero limit defaults safely to at least 1 / safe limit
  const candNeg = mcts.getTopCandidateLines(-5);
  assert.ok(candNeg.length >= 1 && candNeg.length <= 4);

  const candZero = mcts.getTopCandidateLines(0);
  assert.ok(candZero.length >= 1 && candZero.length <= 4);

  // Large limit e.g. 100 on 4 children outputs all 4
  const candLarge = mcts.getTopCandidateLines(100);
  assert.strictEqual(candLarge.length, 4);

  // Monotonicity invariants: visits must be non-increasing, ranks 1, 2, 3, 4
  for (let i = 0; i < candLarge.length; i++) {
    assert.strictEqual(candLarge[i].rank, i + 1, `Rank should be ${i + 1}`);
    assert.ok(typeof candLarge[i].eval === 'number' && !isNaN(candLarge[i].eval), 'eval must be a valid number');
    assert.ok(typeof candLarge[i].visits === 'number' && candLarge[i].visits > 0, 'visits must be positive');
    if (i > 0) {
      assert.ok(candLarge[i - 1].visits >= candLarge[i].visits, 'Visits must be descending/non-increasing');
    }
  }
});

test('Settings input attributes and client script parsing invariants', () => {
  const htmlPath = path.resolve(__dirname, '../../public/index.html');
  const html = fs.readFileSync(htmlPath, 'utf8');

  // Input attributes for positive integer entry
  assert.ok(html.includes('type="number"'), 'Top-k input must have type="number"');
  assert.ok(html.includes('id="top-k-candidates-select"'), 'Top-k input must retain id="top-k-candidates-select"');
  assert.ok(html.includes('min="1"'), 'Top-k input must have min="1" constraint');
  assert.ok(html.includes('step="1"'), 'Top-k input must have step="1" integer constraint');

  const scriptPath = path.resolve(__dirname, '../../public/script.js');
  const script = fs.readFileSync(scriptPath, 'utf8');

  // Verify handleTopKCandidatesChange parses with Math.max(1, ...)
  assert.ok(script.includes('Math.max(1,'), 'handleTopKCandidatesChange must clamp to minimum of 1');
});

test('CSS layout rules guarantee horizontal scrolling rail and responsive display contents', () => {
  const cssPath = path.resolve(__dirname, '../../public/style.css');
  const css = fs.readFileSync(cssPath, 'utf8');

  // Horizontal scrolling rail in dock
  assert.ok(css.includes('.bottom-analysis-dock'), 'Missing .bottom-analysis-dock in style.css');
  assert.ok(css.includes('overflow-x: auto'), 'Candidate lines container must have overflow-x: auto');
  assert.ok(css.includes('flex-direction: row'), 'Candidate lines container must be flex-direction: row');

  // Responsive tablet/mobile display: contents to preserve grid
  assert.ok(css.includes('display: contents'), 'Must use display: contents for mobile/tablet responsive layout');
  assert.ok(css.includes('.play-area'), 'Missing .play-area in style.css');
  assert.ok(css.includes('.game-middle-area'), 'Missing .game-middle-area in style.css');
});


