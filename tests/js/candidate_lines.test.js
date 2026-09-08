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
  }
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
