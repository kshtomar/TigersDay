/**
 * The Tiger's Day – Anglo-Mysore Wars
 * 100% Client-Side WebAssembly Game Engine & P2P Multiplayer UX Controller
 * 
 * Direct Point-and-Click Engine Interface (Zero Server Dependencies / Zero Modals)
 */

// ==========================================================================
// 1. CONSTANTS & GAME MAP GEOMETRY
// ==========================================================================
const SVG_NS = 'http://www.w3.org/2000/svg';

var NODES = {
  Bombay:        { x:110, y: 74,  owner:'british', armyType:'active',  key:true,  coast:true,  labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Hyderabad:     { x:515, y:100,  owner:'british', armyType:'active',  key:true,  coast:false, labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Madras:        { x:618, y:322,  owner:'british', armyType:'active',  key:true,  coast:true,  labelAnchor:{anchor:'end',    dx:-18, dy:-24} },
  Seringapatam:  { x:230, y:480,  owner:'mysore',  armyType:'fort',    key:true,  coast:false, labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Coimbatore:    { x:305, y:600,  owner:'mysore',  armyType:'fort',    key:true,  coast:false, labelAnchor:{anchor:'middle', dx:0,   dy:-24} },
  Satara:        { x:255, y:128,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Poona:         { x:345, y:70,   owner:'empty',   armyType:'empty',   key:false, coast:false },
  Raichur:       { x:390, y:178,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Masulipatam:   { x:656, y:162,  owner:'empty',   armyType:'empty',   key:false, coast:true,  labelAnchor:{anchor:'end',   dx:-12, dy:-16} },
  Goa:           { x: 94, y:262,  owner:'empty',   armyType:'empty',   key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Darwar:        { x:232, y:232,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Anantapur:     { x:470, y:228,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Chitaldoorg:   { x:250, y:350,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Mangalore:     { x:118, y:398,  owner:'mysore',  armyType:'fort',    key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Bangalore:     { x:350, y:400,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Vellore:       { x:460, y:340,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  'Mahé':        { x:145, y:586,  owner:'mysore',  armyType:'fort',    key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Pondicherry:   { x:610, y:446,  owner:'empty',   armyType:'empty',   key:false, coast:true,  labelAnchor:{anchor:'end',   dx:-12, dy:-16} },
  Erode:         { x:405, y:515,  owner:'mysore',  armyType:'fort',    key:false, coast:false },
  Trichy:        { x:506, y:580,  owner:'empty',   armyType:'empty',   key:false, coast:false },
  Alwaye:        { x:225, y:720,  owner:'mysore',  armyType:'empty',   key:false, coast:false },
  Dindigul:      { x:415, y:670,  owner:'empty',   armyType:'fort',    key:false, coast:false },
  Ramnad:        { x:415, y:770,  owner:'empty',   armyType:'empty',   key:false, coast:true },
  Travancore:    { x:260, y:830,  owner:'british', armyType:'active',  key:false, coast:true,  labelAnchor:{anchor:'start', dx: 12, dy:-16} },
  Ceylon:        { x:540, y:800,  owner:'empty',   armyType:'empty',   key:false, coast:true  },
};

const EDGES = [
  ['Bombay', 'Satara'],
  ['Bombay', 'Goa', {curve: -0.2}],
  ['Hyderabad', 'Raichur'],
  ['Hyderabad', 'Masulipatam'],
  ['Hyderabad', 'Anantapur'],
  ['Madras', 'Masulipatam', {curve: -0.2}],
  ['Madras', 'Anantapur'],
  ['Madras', 'Vellore'],
  ['Madras', 'Pondicherry'],
  ['Seringapatam', 'Mangalore'],
  ['Seringapatam', 'Bangalore'],
  ['Seringapatam', 'Mahé'],
  ['Seringapatam', 'Erode'],
  ['Coimbatore', 'Mahé'],
  ['Coimbatore', 'Erode'],
  ['Coimbatore', 'Alwaye'],
  ['Coimbatore', 'Dindigul'],
  ['Satara', 'Raichur'],
  ['Satara', 'Darwar'],
  ['Poona', 'Satara'],
  ['Poona', 'Hyderabad'],
  ['Poona', 'Bombay'],
  ['Raichur', 'Anantapur'],
  ['Raichur', 'Chitaldoorg'],
  ['Goa', 'Darwar'],
  ['Goa', 'Mangalore', {curve: -0.2}],
  ['Darwar', 'Chitaldoorg'],
  ['Anantapur', 'Vellore'],
  ['Chitaldoorg', 'Mangalore'],
  ['Chitaldoorg', 'Bangalore'],
  ['Bangalore', 'Vellore'],
  ['Vellore', 'Erode'],
  ['Pondicherry', 'Erode'],
  ['Pondicherry', 'Trichy'],
  ['Erode', 'Trichy'],
  ['Trichy', 'Dindigul'],
  ['Trichy', 'Ceylon', {curve: -0.2}],
  ['Alwaye', 'Travancore'],
  ['Alwaye', 'Ramnad'],
  ['Dindigul', 'Ramnad'],
  ['Ramnad', 'Ceylon'],
  ['Ramnad', 'Travancore'],
  ['Travancore', 'Ceylon']
];

const BRITISH_CARD_DATA = [
  { name: 'Wall Breach',      strength: 3, icon: '💥', desc: 'Powerful siege breach & battle power' },
  { name: 'Highlanders',      strength: 2, icon: '🟥', desc: 'Deploy a Fresh Army on any Coast' },
  { name: 'Royal Navy',       strength: 2, icon: '⚓', desc: 'Move an Army to any Coastal territory' },
  { name: 'Divide and Rule',  strength: 1, icon: '🤝', desc: 'Relocate a Fort not in a Key City' },
  { name: 'Force March',      strength: 1, icon: '🥾', desc: 'Move a Tired Army to adjacent territory' },
  { name: 'Princely States',  strength: 1, icon: '🏰', desc: 'Deploy a Tired Army in an empty Key City' },
];

const MYSORE_CARD_DATA = [
  { name: 'Iron Rockets',     strength: 3, icon: '🚀', desc: 'Devastating artillery & battle power' },
  { name: 'Sepoy Mutiny',     strength: 2, icon: '⚔️', desc: 'Remove an Army not in a Key City' },
  { name: 'French Alliance',  strength: 2, icon: '💠', desc: 'Deploy a Fort adjacent to another Fort' },
  { name: 'Monsoon',          strength: 1, icon: '🌧️', desc: 'Flip a Fresh Army to Tired' },
  { name: 'Cavalry Raid',     strength: 1, icon: '🏇', desc: 'Force British to discard a random card' },
  { name: 'Sea Trade',        strength: 1, icon: '🪙', desc: 'Move a Fort from Coast to any territory' },
];

const CARD_VALUE = [3, 2, 2, 1, 1, 1];

// ==========================================================================
// 2. RUNTIME STATE MACHINE & ENGINES
// ==========================================================================
let currentGameState = null;
let currentBitString = "";
let lastUiState = null;
let currentMoves = [];                 // Array of { idx, type, desc }
let selectedUnit = null;               // Selected territory string on map
let cardTargetingMode = null;          // { cardName, faction, step: 1|2, sourceNode, targetNodes, validSources, allMoves, isTwoStep }
let stagedTradeCard = null;            // { faction, cardIndex, cardName }

let matchMode = "human_vs_ai";        // "human_vs_ai" | "human" | "p2p_multiplayer" | "ai_vs_ai"
let humanPlayerSide = "british";       // "british" | "mysore"
let players = { british: 'human', mysore: 'ai' };
let currentEvalLoopState = null;

let settings = {
  showEval: false,
  showDebugMoves: false
};

// Client-Side AI & MCTS Singletons
const onnxModel = new TDMCTS.ONNXModelWrapper('./alphatiger.onnx');
const mctsEngine = new TDMCTS.MCTS(onnxModel, { simulations: 250, depsilon: 0.0 });
const multiplayerManager = new TDMultiplayer.MultiplayerManager();

// Preload ONNX model in background
onnxModel.init().catch(err => console.log("ONNX preload notice:", err));

/**
 * Central state reset — clears ALL interactive UI state (unit selection,
 * card targeting highlights, trade staging) and refreshes the map + cards.
 */
function clearAllInteractionState() {
  selectedUnit = null;
  cardTargetingMode = null;
  stagedTradeCard = null;
  refreshMapHighlights();
  renderAllCards();
  updateActionButtons();
  updateTurnHeaderInstruction();
}

// ==========================================================================
// 3. SVG MAP RENDERING & INITIALIZATION
// ==========================================================================
function splineControlPoint(ax, ay, bx, by, f = 0.18) {
  const mx = (ax + bx) / 2, my = (ay + by) / 2;
  const dx = bx - ax, dy = by - ay;
  const len = Math.sqrt(dx * dx + dy * dy);
  return { cx: mx + (-dy / len) * len * f, cy: my + (dx / len) * len * f };
}

function renderEdges() {
  const layer = document.getElementById('edge-layer');
  if (!layer) return;
  layer.innerHTML = '';
  for (const edge of EDGES) {
    const [aName, bName, opts = {}] = edge;
    const a = NODES[aName], b = NODES[bName];
    if (!a || !b) continue;
    const { cx, cy } = splineControlPoint(a.x, a.y, b.x, b.y, opts.curve || 0.15);
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', `M ${a.x},${a.y} Q ${cx},${cy} ${b.x},${b.y}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#6a4c1e');
    path.setAttribute('stroke-width', '1.8');
    path.setAttribute('stroke-linecap', 'round');
    path.setAttribute('opacity', '0.7');
    layer.appendChild(path);
  }
}

window.renderNodes = function renderNodes() {
  const layer = document.getElementById('node-layer');
  if (!layer) return;
  layer.innerHTML = '';

  for (const [name, data] of Object.entries(window.NODES)) {
    const { x, y, owner, key, coast } = data;
    const armyType = data.armyType || 'empty';
    const la = data.labelAnchor || { anchor: 'middle', dx: 0, dy: key ? -24 : -16 };
    const anchor = la.anchor || 'middle', ldx = la.dx || 0, ldy = la.dy || (key ? -24 : -16);

    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', 'node-group');
    g.setAttribute('id', `node-group-${name.replace(/[^a-zA-Z0-9]/g, '_')}`);
    g.setAttribute('transform', `translate(${x},${y})`);
    g.dataset.name = name;
    g.dataset.owner = owner;
    g.dataset.armyType = armyType;
    g.dataset.key = String(key);
    g.dataset.coast = String(coast);

    // Event listeners
    g.addEventListener('click', (e) => {
      e.stopPropagation();
      handleNodeClick(name);
    });
    g.addEventListener('mousemove', (e) => tooltipShow(e, name, data));
    g.addEventListener('mouseleave', tooltipHide);

    // Selection ring
    const ring = document.createElementNS(SVG_NS, 'circle');
    ring.setAttribute('class', 'sel-ring');
    ring.setAttribute('r', key ? '24' : '18');
    ring.setAttribute('fill', 'none');
    ring.setAttribute('stroke', '#d4a030');
    ring.setAttribute('stroke-width', '2.5');
    ring.setAttribute('opacity', '0');
    g.appendChild(ring);

    // Glowing target pulse ring
    const targetRing = document.createElementNS(SVG_NS, 'circle');
    targetRing.setAttribute('class', 'target-glow-ring');
    targetRing.setAttribute('r', '20');
    targetRing.setAttribute('fill', 'none');
    targetRing.setAttribute('stroke', '#ffdd44');
    targetRing.setAttribute('stroke-width', '3');
    targetRing.setAttribute('opacity', '0');
    g.appendChild(targetRing);

    // Territory shape
    if (key) {
      const sq = document.createElementNS(SVG_NS, 'rect');
      sq.setAttribute('x', '-18');
      sq.setAttribute('y', '-18');
      sq.setAttribute('width', '36');
      sq.setAttribute('height', '36');
      sq.setAttribute('fill', '#1c1814');
      sq.setAttribute('stroke', '#e0c896');
      sq.setAttribute('stroke-width', '1.6');
      g.appendChild(sq);
    } else {
      const circ = document.createElementNS(SVG_NS, 'circle');
      circ.setAttribute('r', '10');
      circ.setAttribute('fill', '#2a2420');
      circ.setAttribute('stroke', '#b0a080');
      circ.setAttribute('stroke-width', '1.4');
      g.appendChild(circ);
    }

    // Unit overlay
    if (armyType === 'fort') {
      const p = document.createElementNS(SVG_NS, 'polygon');
      p.setAttribute('points', '0,-22 22,0 0,22 -22,0');
      p.setAttribute('fill', '#2e7a2e');
      p.setAttribute('stroke', '#8fe08f');
      p.setAttribute('stroke-width', '1.2');
      p.setAttribute('filter', 'url(#nshadow)');
      g.appendChild(p);
    } else if (armyType === 'active' || armyType === 'tired') {
      const isTired = armyType === 'tired';
      const opacity = isTired ? '0.55' : '1';

      if (owner === 'british') {
        const p = document.createElementNS(SVG_NS, 'rect');
        p.setAttribute('x', '-18');
        p.setAttribute('y', '-18');
        p.setAttribute('width', '36');
        p.setAttribute('height', '36');
        p.setAttribute('fill', '#c0281a');
        p.setAttribute('stroke', '#ff9999');
        p.setAttribute('stroke-width', '1.2');
        p.setAttribute('opacity', opacity);
        p.setAttribute('filter', 'url(#nshadow)');
        g.appendChild(p);
      } else if (owner === 'mysore') {
        const p = document.createElementNS(SVG_NS, 'polygon');
        p.setAttribute('points', '0,-24 24,0 0,24 -24,0');
        p.setAttribute('fill', '#2e7a2e');
        p.setAttribute('stroke', '#8fe08f');
        p.setAttribute('stroke-width', '1.2');
        p.setAttribute('opacity', opacity);
        p.setAttribute('filter', 'url(#nshadow)');
        g.appendChild(p);
      }

      if (isTired) {
        const slash = document.createElementNS(SVG_NS, 'line');
        slash.setAttribute('x1', '-13');
        slash.setAttribute('y1', '-13');
        slash.setAttribute('x2', '13');
        slash.setAttribute('y2', '13');
        slash.setAttribute('stroke', 'rgba(255,255,255,0.9)');
        slash.setAttribute('stroke-width', '3');
        slash.setAttribute('stroke-linecap', 'round');
        g.appendChild(slash);
      }
    }

    // Labels
    const makeLabel = (isHalo) => {
      const t = document.createElementNS(SVG_NS, 'text');
      t.setAttribute('dy', String(ldy));
      t.setAttribute('dx', String(ldx));
      t.setAttribute('text-anchor', anchor);
      t.setAttribute('font-family', key ? 'Cinzel,serif' : 'Cormorant Garamond,serif');
      t.setAttribute('font-size', key ? '21' : '19');
      t.setAttribute('font-weight', '700');
      if (key) t.setAttribute('letter-spacing', '.06em');
      if (isHalo) {
        t.setAttribute('stroke', 'rgba(228,213,155,0.92)');
        t.setAttribute('stroke-width', '4');
        t.setAttribute('stroke-linejoin', 'round');
        t.setAttribute('fill', 'none');
        t.setAttribute('paint-order', 'stroke');
      } else {
        t.setAttribute('fill', '#1a1208');
      }
      t.textContent = name;
      return t;
    };
    g.appendChild(makeLabel(true));
    g.appendChild(makeLabel(false));

    layer.appendChild(g);
  }

  refreshMapHighlights();
};

// ==========================================================================
// 4. MAP HIGHLIGHTS & PULSE MANAGEMENT
// ==========================================================================
function clearMapHighlights() {
  document.querySelectorAll('.node-group').forEach(el => {
    el.classList.remove('selected-unit', 'valid-map-target');
    const ring = el.querySelector('.sel-ring');
    const glowRing = el.querySelector('.target-glow-ring');
    if (ring) ring.setAttribute('opacity', '0');
    if (glowRing) glowRing.setAttribute('opacity', '0');
  });
}

function refreshMapHighlights() {
  clearMapHighlights();

  if (selectedUnit) {
    const el = getNodeElement(selectedUnit);
    if (el) {
      el.classList.add('selected-unit');
      const ring = el.querySelector('.sel-ring');
      if (ring) ring.setAttribute('opacity', '1');
    }

    const validDests = getValidMoveDestinations(selectedUnit);
    validDests.forEach(destName => {
      const destEl = getNodeElement(destName);
      if (destEl) {
        destEl.classList.add('valid-map-target');
        const glowRing = destEl.querySelector('.target-glow-ring');
        if (glowRing) glowRing.setAttribute('opacity', '1');
      }
    });
  }

  if (cardTargetingMode) {
    if (!cardTargetingMode.isTwoStep) {
      cardTargetingMode.targetNodes.forEach(nodeName => {
        const el = getNodeElement(nodeName);
        if (el) {
          el.classList.add('valid-map-target');
          const glowRing = el.querySelector('.target-glow-ring');
          if (glowRing) glowRing.setAttribute('opacity', '1');
        }
      });
    } else {
      if (cardTargetingMode.step === 1) {
        cardTargetingMode.validSources.forEach(srcName => {
          const el = getNodeElement(srcName);
          if (el) {
            el.classList.add('valid-map-target');
            const glowRing = el.querySelector('.target-glow-ring');
            if (glowRing) glowRing.setAttribute('opacity', '1');
          }
        });
      } else if (cardTargetingMode.step === 2 && cardTargetingMode.sourceNode) {
        const srcEl = getNodeElement(cardTargetingMode.sourceNode);
        if (srcEl) {
          srcEl.classList.add('selected-unit');
          const ring = srcEl.querySelector('.sel-ring');
          if (ring) ring.setAttribute('opacity', '1');
        }

        const validDests = getValidTwoStepDestinations(cardTargetingMode.cardName, cardTargetingMode.sourceNode);
        validDests.forEach(destName => {
          const destEl = getNodeElement(destName);
          if (destEl) {
            destEl.classList.add('valid-map-target');
            const glowRing = destEl.querySelector('.target-glow-ring');
            if (glowRing) glowRing.setAttribute('opacity', '1');
          }
        });
      }
    }
  }
}

function getNodeElement(name) {
  return document.getElementById(`node-group-${name.replace(/[^a-zA-Z0-9]/g, '_')}`);
}

function getValidMoveDestinations(sourceName) {
  const dests = new Set();
  const prefix = sourceName + " -> ";
  currentMoves.forEach(m => {
    if (m.type === 'Move' && m.desc.startsWith(prefix)) {
      const dest = m.desc.substring(prefix.length).trim();
      dests.add(dest);
    }
  });
  return dests;
}

function getValidTwoStepDestinations(cardName, sourceName) {
  const dests = new Set();
  const prefix = sourceName + " -> ";
  currentMoves.forEach(m => {
    if (m.type === cardName && m.desc.startsWith(prefix)) {
      const dest = m.desc.substring(prefix.length).trim();
      dests.add(dest);
    }
  });
  return dests;
}

// ==========================================================================
// 5. DIRECT POINT-AND-CLICK MAP INTERACTIONS
// ==========================================================================
function handleNodeClick(nodeName) {
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  if (cardTargetingMode) {
    handleCardTargetNodeClick(nodeName);
    return;
  }

  if (stagedTradeCard) {
    stagedTradeCard = null;
    renderAllCards();
    updateTurnHeaderInstruction();
  }

  if (selectedUnit === nodeName) {
    selectedUnit = null;
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
    return;
  }

  if (selectedUnit !== null) {
    const targetMoveStr = `${selectedUnit} -> ${nodeName}`;
    const move = currentMoves.find(m => m.type === 'Move' && m.desc === targetMoveStr);

    if (move) {
      const fromUnit = selectedUnit;
      selectedUnit = null;
      refreshMapHighlights();
      updateActionButtons();
      showToast(`Moving ${fromUnit} → ${nodeName}`, 'info');
      window.applyMove(move.idx);
      return;
    }

    const hasMoveMoves = currentMoves.some(m => m.type === 'Move' && m.desc.startsWith(nodeName + " -> "));
    const hasTireMove = currentMoves.some(m => m.type === 'Tire' && m.desc === nodeName);

    if (hasMoveMoves || hasTireMove) {
      selectedUnit = nodeName;
      refreshMapHighlights();
      updateActionButtons();
      updateTurnHeaderInstruction();
      return;
    }

    selectedUnit = null;
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
    return;
  }

  const hasMoveMoves = currentMoves.some(m => m.type === 'Move' && m.desc.startsWith(nodeName + " -> "));
  const hasTireMove = currentMoves.some(m => m.type === 'Tire' && m.desc === nodeName);

  if (hasMoveMoves || hasTireMove) {
    selectedUnit = nodeName;
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
  } else {
    const node = NODES[nodeName];
    if (node && node.armyType !== 'empty') {
      showToast(`${nodeName} has no legal moves right now.`);
    }
  }
}

function handleCardTargetNodeClick(nodeName) {
  if (!cardTargetingMode) return;

  if (!cardTargetingMode.isTwoStep) {
    if (cardTargetingMode.targetNodes.has(nodeName)) {
      const move = currentMoves.find(m => m.type === cardTargetingMode.cardName && m.desc === nodeName);
      if (move) {
        const cardName = cardTargetingMode.cardName;
        exitCardTargetingMode();
        showToast(`Activated ${cardName} on ${nodeName}!`, 'success');
        window.applyMove(move.idx);
      }
    } else {
      showToast(`${nodeName} is not a valid target for ${cardTargetingMode.cardName}.`, 'error');
    }
  } else {
    if (cardTargetingMode.step === 1) {
      if (cardTargetingMode.validSources.has(nodeName)) {
        cardTargetingMode.sourceNode = nodeName;
        cardTargetingMode.step = 2;
        refreshMapHighlights();
        updateTurnHeaderInstruction();
      } else {
        showToast(`${nodeName} cannot be selected as a source.`, 'error');
      }
    } else if (cardTargetingMode.step === 2) {
      if (nodeName === cardTargetingMode.sourceNode) {
        cardTargetingMode.sourceNode = null;
        cardTargetingMode.step = 1;
        refreshMapHighlights();
        updateTurnHeaderInstruction();
        return;
      }

      const targetMoveStr = `${cardTargetingMode.sourceNode} -> ${nodeName}`;
      const move = currentMoves.find(m => m.type === cardTargetingMode.cardName && m.desc === targetMoveStr);

      if (move) {
        const cardName = cardTargetingMode.cardName;
        exitCardTargetingMode();
        showToast(`Executed ${cardName}: ${targetMoveStr}!`, 'success');
        window.applyMove(move.idx);
      } else {
        showToast(`${nodeName} is not a valid destination.`, 'error');
      }
    }
  }
}

function exitCardTargetingMode() {
  cardTargetingMode = null;
  refreshMapHighlights();
  renderAllCards();
  updateActionButtons();
  updateTurnHeaderInstruction();
}

// ==========================================================================
// 6. CARD INTERACTIONS
// ==========================================================================
function renderAllCards() {
  if (!lastUiState) return;
  renderCardDeck('mysore', lastUiState.mysore_cards);
  renderCardDeck('british', lastUiState.british_cards);

  const mAvail = lastUiState.mysore_cards.filter(Boolean).length;
  const bAvail = lastUiState.british_cards.filter(Boolean).length;
  const mBadge = document.getElementById('mysore-hand-count');
  const bBadge = document.getElementById('british-hand-count');
  if (mBadge) mBadge.textContent = `${mAvail}/6`;
  if (bBadge) bBadge.textContent = `${bAvail}/6`;
}

function renderCardDeck(faction, availArray) {
  const container = document.getElementById(`${faction}-cards-list`);
  if (!container) return;
  container.innerHTML = '';

  const cardDataList = faction === 'mysore' ? MYSORE_CARD_DATA : BRITISH_CARD_DATA;

  cardDataList.forEach((card, index) => {
    const isUsable = Boolean(availArray[index]);
    const cardDiv = document.createElement('div');

    let classNames = ['player-card', `${faction}-card`];
    if (!isUsable) classNames.push('card-activated', 'used');

    if (stagedTradeCard && stagedTradeCard.faction === faction && stagedTradeCard.cardIndex === index) {
      classNames.push('staged-trade');
    }

    if (stagedTradeCard && stagedTradeCard.faction === faction && !isUsable) {
      const tradeType = `Draw ${stagedTradeCard.cardName}`;
      const canReclaim = currentMoves.some(m => m.type === tradeType && m.desc === card.name);
      if (canReclaim) classNames.push('valid-trade-target');
    }

    if (cardTargetingMode && cardTargetingMode.faction === faction && cardTargetingMode.cardName === card.name) {
      classNames.push('targeting-active');
    }

    cardDiv.className = classNames.join(' ');

    // Seal
    const seal = document.createElement('div');
    seal.className = 'card-strength-seal';
    seal.textContent = card.strength;
    seal.title = `Commit +${card.strength} Combat Strength`;
    seal.addEventListener('click', (e) => {
      e.stopPropagation();
      handleCardStrengthClick(faction, index, card.name);
    });
    cardDiv.appendChild(seal);

    // Header
    const headerRow = document.createElement('div');
    headerRow.className = 'card-header-row';
    headerRow.innerHTML = `
      <span class="card-icon">${card.icon}</span>
      <span class="card-name">${card.name}</span>
    `;
    cardDiv.appendChild(headerRow);

    // Desc
    const desc = document.createElement('div');
    desc.className = 'card-desc';
    desc.textContent = card.desc;
    cardDiv.appendChild(desc);

    // Exhausted Stamp
    const stamp = document.createElement('div');
    stamp.className = 'used-stamp';
    stamp.textContent = 'EXHAUSTED';
    cardDiv.appendChild(stamp);

    cardDiv.addEventListener('click', (e) => {
      e.stopPropagation();
      handleCardBodyClick(faction, index, card.name, isUsable);
    });

    container.appendChild(cardDiv);
  });
}

function handleCardStrengthClick(faction, index, cardName) {
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  const powerType = faction === 'mysore' ? 'Mysore Power' : 'British Power';
  const move = currentMoves.find(m => m.type === powerType && m.desc === cardName);

  if (move) {
    showToast(`Committed ${cardName} (+${CARD_VALUE[index]} Strength) to battle!`, 'success');
    window.applyMove(move.idx);
  } else {
    showToast(`Cannot commit strength outside active combat.`, 'info');
  }
}

function handleCardBodyClick(faction, index, cardName, isUsable) {
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  if (!isUsable) {
    if (stagedTradeCard && stagedTradeCard.faction === faction) {
      const tradeType = `Draw ${stagedTradeCard.cardName}`;
      const move = currentMoves.find(m => m.type === tradeType && m.desc === cardName);

      if (move) {
        const tradedCard = stagedTradeCard.cardName;
        stagedTradeCard = null;
        renderAllCards();
        updateTurnHeaderInstruction();
        showToast(`Traded ${tradedCard} to reclaim ${cardName}!`, 'success');
        window.applyMove(move.idx);
        return;
      } else {
        showToast(`Cannot trade ${stagedTradeCard.cardName} for ${cardName}.`, 'error');
        return;
      }
    } else {
      showToast(`${cardName} is exhausted. Stage an active card to trade for it.`, 'info');
      return;
    }
  }

  const abilityCardNames = [
    'Cavalry Raid', 'Sepoy Mutiny', 'French Alliance', 'Monsoon',
    'Highlanders', 'Princely States',
    'Divide and Rule', 'Force March', 'Royal Navy', 'Sea Trade'
  ];

  if (abilityCardNames.includes(cardName)) {
    const hasAbilityMove = currentMoves.some(m => m.type === cardName);
    if (hasAbilityMove) {
      handleCardAbilityActivation(faction, index, cardName);
      return;
    }
  }

  handleCardTradeSelection(faction, index, cardName);
}

function handleCardAbilityActivation(faction, index, cardName) {
  clearAllInteractionState();

  if (cardName === 'Cavalry Raid') {
    const move = currentMoves.find(m => m.type === 'Cavalry Raid');
    if (move) {
      showToast(`Cavalry Raid launched! British must discard.`, 'success');
      window.applyMove(move.idx);
      return;
    }
  }

  const singleStepCards = ['Sepoy Mutiny', 'French Alliance', 'Monsoon', 'Highlanders', 'Princely States'];
  if (singleStepCards.includes(cardName)) {
    const moves = currentMoves.filter(m => m.type === cardName);
    if (moves.length === 0) {
      showToast(`No legal targets on the map for ${cardName}.`, 'error');
      return;
    }

    const targetNodes = new Set(moves.map(m => m.desc));
    cardTargetingMode = {
      cardName,
      faction,
      step: 1,
      targetNodes,
      isTwoStep: false
    };

    renderAllCards();
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
    showToast(`Targeting ${cardName}: Click a pulsing territory on the map.`, 'info');
    return;
  }

  const twoStepCards = ['Divide and Rule', 'Force March', 'Royal Navy', 'Sea Trade'];
  if (twoStepCards.includes(cardName)) {
    const moves = currentMoves.filter(m => m.type === cardName);
    if (moves.length === 0) {
      showToast(`No legal actions on the map for ${cardName}.`, 'error');
      return;
    }

    const validSources = new Set(moves.map(m => m.desc.split(' -> ')[0]));
    cardTargetingMode = {
      cardName,
      faction,
      step: 1,
      sourceNode: null,
      validSources,
      isTwoStep: true
    };

    renderAllCards();
    refreshMapHighlights();
    updateActionButtons();
    updateTurnHeaderInstruction();
    showToast(`Targeting ${cardName}: Select origin territory.`, 'info');
  }
}

function handleCardTradeSelection(faction, index, cardName) {
  if (stagedTradeCard && stagedTradeCard.cardName === cardName) {
    stagedTradeCard = null;
    renderAllCards();
    updateTurnHeaderInstruction();
    showToast(`Cancelled trade staging for ${cardName}.`);
    return;
  }

  const tradeType = `Draw ${cardName}`;
  const hasTradeMoves = currentMoves.some(m => m.type === tradeType);

  if (hasTradeMoves) {
    stagedTradeCard = { faction, cardIndex: index, cardName };
    renderAllCards();
    updateTurnHeaderInstruction();
    showToast(`Staged ${cardName} for trade! Click a greyed-out card to reclaim.`, 'info');
  } else {
    showToast(`No exhausted cards can currently be reclaimed with ${cardName}.`, 'info');
  }
}

// ==========================================================================
// 7. HEADER INSTRUCTIONS & ACTION BUTTONS
// ==========================================================================
function updateTurnHeader(uiState, winner) {
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');
  const icon = document.getElementById('turn-faction-icon');
  const battleBar = document.getElementById('battle-status-bar');

  if (!header || !uiState) return;

  if (winner === 1) {
    header.className = 'turn-header british-phase victory-state';
    counter.textContent = 'GAME OVER';
    title.textContent = 'BRITISH VICTORY! ALL KEY CITIES OCCUPIED';
    icon.textContent = '👑';
    showToast('The British East India Company has achieved total dominion!', 'success');
    return;
  } else if (winner === -1) {
    header.className = 'turn-header mysore-phase victory-state';
    counter.textContent = 'GAME OVER';
    title.textContent = 'MYSORE VICTORY! SULTANATE HAS WITHSTOOD SIEGE';
    icon.textContent = '🐅';
    showToast('The Sultanate of Mysore has repelled the British invasion!', 'success');
    return;
  }

  const whoToMove = uiState.who_to_move || 'British Move';
  counter.textContent = `TURN ${uiState.turn} OF 4`;

  if (whoToMove === 'British Move') {
    header.className = 'turn-header british-phase';
    title.textContent = 'BRITISH PHASE: MOVE AN ARMY';
    icon.textContent = '🦁';
  } else if (whoToMove === 'Mysore Card') {
    header.className = 'turn-header mysore-phase';
    title.textContent = 'MYSORE PHASE: PLAY A CARD';
    icon.textContent = '🐅';
  } else if (whoToMove === 'British Card') {
    header.className = 'turn-header british-phase';
    title.textContent = 'BRITISH PHASE: PLAY A CARD';
    icon.textContent = '🦁';
  }

  if (battleBar) {
    if (uiState.attacker !== 'None' && uiState.defender !== 'None') {
      battleBar.classList.remove('hidden');
      document.getElementById('battle-attacker-name').textContent = uiState.attacker;
      document.getElementById('battle-defender-name').textContent = uiState.defender;

      const netVal = uiState.net_strength !== undefined ? uiState.net_strength : (uiState.card_strength || 0);
      const sign = netVal > 0 ? '+' : '';
      const strengthEl = document.getElementById('battle-strength-val');
      if (strengthEl) {
        strengthEl.textContent = `${sign}${netVal}`;
        if (netVal > 0) {
          strengthEl.style.color = '#8fd48f'; // British attacker has the advantage
        } else if (netVal < 0) {
          strengthEl.style.color = '#e58f8f'; // Mysore defender is holding
        } else {
          strengthEl.style.color = '#f0c868'; // Tied (attacker needs > 0 to win)
        }
      }
    } else {
      battleBar.classList.add('hidden');
    }
  }

  updateActionButtons();
  updateTurnHeaderInstruction();
}

function updateTurnHeaderInstruction() {
  const instructionEl = document.getElementById('turn-instruction');
  if (!instructionEl || !lastUiState) return;

  if (cardTargetingMode) {
    if (!cardTargetingMode.isTwoStep) {
      instructionEl.innerHTML = `🎯 <strong>Targeting ${cardTargetingMode.cardName}:</strong> Click a pulsing territory on the map.`;
    } else if (cardTargetingMode.step === 1) {
      instructionEl.innerHTML = `🎯 <strong>${cardTargetingMode.cardName}:</strong> Select origin territory.`;
    } else if (cardTargetingMode.step === 2) {
      instructionEl.innerHTML = `🎯 <strong>${cardTargetingMode.cardName}:</strong> Move ${cardTargetingMode.sourceNode} → Select glowing destination.`;
    }
    return;
  }

  if (stagedTradeCard) {
    instructionEl.innerHTML = `🪙 <strong>Trading ${stagedTradeCard.cardName}:</strong> Click an exhausted (greyed-out) card in hand to reclaim.`;
    return;
  }

  if (selectedUnit) {
    instructionEl.innerHTML = `📍 <strong>${selectedUnit} selected:</strong> Click a glowing destination node to march, or click Rest.`;
    return;
  }

  const whoToMove = lastUiState.who_to_move || '';
  if (whoToMove === 'British Move') {
    instructionEl.textContent = 'Click an Army on the map to select, then click a pulsing destination.';
  } else if (whoToMove.includes('Card')) {
    instructionEl.textContent = 'Click a card in your hand to activate ability or commit battle strength.';
  }
}

function updateActionButtons() {
  const restBtn = document.getElementById('header-rest-btn');
  const passBtn = document.getElementById('header-pass-btn');
  const cancelBtn = document.getElementById('header-cancel-btn');

  if (cancelBtn) {
    if (selectedUnit || cardTargetingMode || stagedTradeCard) {
      cancelBtn.classList.remove('hidden');
    } else {
      cancelBtn.classList.add('hidden');
    }
  }

  if (restBtn) {
    if (selectedUnit) {
      const tireMove = currentMoves.find(m => m.type === 'Tire' && m.desc === selectedUnit);
      if (tireMove) {
        restBtn.classList.remove('hidden');
      } else {
        restBtn.classList.add('hidden');
      }
    } else {
      restBtn.classList.add('hidden');
    }
  }

  if (passBtn) {
    const hasPass = currentMoves.some(m => m.type === 'Pass Mysore' || m.type === 'Pass British');
    if (hasPass && !cardTargetingMode && !stagedTradeCard) {
      passBtn.classList.remove('hidden');
    } else {
      passBtn.classList.add('hidden');
    }
  }
}

function handleHeaderRestClick() {
  if (isTurnBlockedForLocalPlayer()) return;
  if (!selectedUnit) return;
  const tireMove = currentMoves.find(m => m.type === 'Tire' && m.desc === selectedUnit);
  if (tireMove) {
    showToast(`Rested army at ${selectedUnit}.`, 'info');
    clearAllInteractionState();
    window.applyMove(tireMove.idx);
  }
}

function handleHeaderPassClick() {
  if (isTurnBlockedForLocalPlayer()) return;
  const passMove = currentMoves.find(m => m.type === 'Pass Mysore' || m.type === 'Pass British');
  if (passMove) {
    showToast('Passed turn phase.', 'info');
    clearAllInteractionState();
    window.applyMove(passMove.idx);
  }
}

function handleHeaderResignClick() {
  if (!currentGameState || getStateWinner(currentGameState) !== 0) return;

  const confirmMsg = 'Are you sure you want to resign? This will concede the match.';
  if (!confirm(confirmMsg)) return;

  // Determine which side is resigning
  const uiState = lastUiState;
  let resigningSide = humanPlayerSide || 'british';

  // In P2P multiplayer, notify opponent
  if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendResign(resigningSide);
  }

  // Declare the opponent the winner
  const winnerVal = resigningSide === 'british' ? -1 : 1;
  const winnerName = winnerVal === 1 ? 'BRITISH' : 'MYSORE';
  showToast(`${resigningSide.toUpperCase()} has resigned. ${winnerName} wins!`, 'success');

  // Update the turn header to show victory state
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');
  const icon = document.getElementById('turn-phase-icon');

  if (header && counter && title && icon) {
    header.className = winnerVal === 1
      ? 'turn-header british-phase victory-state'
      : 'turn-header mysore-phase victory-state';
    counter.textContent = 'GAME OVER';
    title.textContent = `${resigningSide.toUpperCase()} RESIGNED — ${winnerName} VICTORY`;
    icon.textContent = winnerVal === 1 ? '🦁' : '🐅';
  }

  // Disable further interaction
  currentMoves = [];
  updateActionButtons();
  clearAllInteractionState();
}

function handleHeaderCancelClick() {
  clearAllInteractionState();
}

// Logarithmic MCTS Simulations Slider (250 – 1,000,000)
const SIMS_MIN = Math.log10(250);   // ~2.398
const SIMS_MAX = Math.log10(1000000); // 6.0

function simsSliderToValue(sliderPos) {
  // sliderPos is 0–100, map logarithmically to 250–1,000,000
  const logVal = SIMS_MIN + (sliderPos / 100) * (SIMS_MAX - SIMS_MIN);
  const raw = Math.round(Math.pow(10, logVal));
  // Snap to clean values
  if (raw <= 500) return Math.round(raw / 50) * 50;
  if (raw <= 5000) return Math.round(raw / 100) * 100;
  if (raw <= 50000) return Math.round(raw / 1000) * 1000;
  if (raw <= 500000) return Math.round(raw / 10000) * 10000;
  return Math.round(raw / 100000) * 100000;
}

function formatSimsLabel(val) {
  if (val >= 1000000) return (val / 1000000).toFixed(0) + 'M';
  if (val >= 1000) return (val / 1000).toFixed(val % 1000 === 0 ? 0 : 1) + 'K';
  return val.toString();
}

function handleSimsSliderInput(sliderPos) {
  const sims = simsSliderToValue(Number(sliderPos));
  mctsEngine.simulations = sims;
  const label = document.getElementById('mcts-sims-label');
  if (label) label.textContent = formatSimsLabel(sims);
}

// ==========================================================================
// 8. STOCKFISH-STYLE EVALUATION BAR
// ==========================================================================
function handleEvalToggle(enabled) {
  settings.showEval = enabled;
  const panel = document.getElementById('eval-panel');
  if (enabled) {
    panel.classList.remove('hidden');
    if (currentBitString) startProgressiveEval(currentBitString);
  } else {
    panel.classList.add('hidden');
  }
}

function setEvalBar(score, totalSims) {
  const bar = document.getElementById('eval-bar-mysore');
  const scoreLabel = document.getElementById('eval-bar-score');
  const simsLabel = document.getElementById('eval-sims-label');
  const winrateLabel = document.getElementById('eval-winrate-label');

  if (!bar || !scoreLabel) return;

  const mysorePercentage = Math.max(5, Math.min(95, ((1 - score) / 2) * 100));
  bar.style.width = `${mysorePercentage}%`;

  const sign = score > 0 ? '+' : '';
  scoreLabel.textContent = `${sign}${score.toFixed(2)}`;
  if (simsLabel) simsLabel.textContent = `Engine: ${totalSims} sims (Wasm)`;

  if (winrateLabel) {
    if (score > 0.3) winrateLabel.textContent = `British Advantage (${(100 - mysorePercentage).toFixed(0)}%)`;
    else if (score < -0.3) winrateLabel.textContent = `Mysore Advantage (${mysorePercentage.toFixed(0)}%)`;
    else winrateLabel.textContent = `Even Position`;
  }
}

async function startProgressiveEval(stateStr) {
  if (!settings.showEval || !currentGameState) return;
  currentEvalLoopState = stateStr;

  const linesContainer = document.getElementById('engine-lines-container');

  try {
    const rootNode = await mctsEngine.search(currentGameState, false);
    if (stateStr !== currentBitString || !settings.showEval) return;

    setEvalBar(rootNode.eval, mctsEngine.simulations);

    const topLines = mctsEngine.getTopCandidateLines(3);
    if (linesContainer && topLines.length > 0) {
      linesContainer.innerHTML = '';
      topLines.forEach(item => {
        const evalVal = item.eval;
        let evalStr = evalVal.toFixed(2);
        if (evalVal > 0) evalStr = '+' + evalStr;
        const squareClass = evalVal > 0.05 ? 'british-favored' : (evalVal < -0.05 ? 'mysore-favored' : 'neutral');

        const lineDiv = document.createElement('div');
        lineDiv.className = 'engine-line';
        lineDiv.innerHTML = `
          <span class="engine-eval-square ${squareClass}">${evalStr}</span>
          <span class="engine-move">${item.moveName} (${item.visits}v)</span>
        `;
        linesContainer.appendChild(lineDiv);
      });
    }
  } catch (err) {
    console.warn("Client eval error:", err);
  }
}

// ==========================================================================
// 9. DEBUG MOVE CONSOLE & DRAWER MENUS
// ==========================================================================
function handleDebugToggle(enabled) {
  settings.showDebugMoves = enabled;
  const consoleEl = document.getElementById('debug-move-console');
  if (enabled) {
    consoleEl.classList.remove('hidden');
    renderDebugMoveList();
  } else {
    consoleEl.classList.add('hidden');
  }
}

function renderDebugMoveList() {
  if (!settings.showDebugMoves) return;
  const list = document.getElementById('move-list');
  const badge = document.getElementById('move-count-badge');
  if (!list) return;

  badge.textContent = currentMoves.length;
  list.innerHTML = currentMoves.map(m => `
    <div class="move-entry" onclick="window.applyMove(${m.idx})">
      <span style="color:#bfa577; min-width:30px;">[${m.idx}]</span>
      <strong style="color:var(--parchment); min-width:130px;">${m.type}</strong>
      <span>${m.desc}</span>
    </div>
  `).join('');
}

function toggleTutorialMenu() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const settingsDrawer = document.getElementById('settings-drawer');
  if (settingsDrawer) settingsDrawer.classList.add('hidden');
  if (tutorialDrawer) tutorialDrawer.classList.toggle('hidden');
}

function toggleSettingsMenu() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const settingsDrawer = document.getElementById('settings-drawer');
  if (tutorialDrawer) tutorialDrawer.classList.add('hidden');
  if (settingsDrawer) settingsDrawer.classList.toggle('hidden');
}

document.addEventListener('click', (e) => {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const tutorialBtn = document.getElementById('tutorial-toggle-btn');
  const settingsDrawer = document.getElementById('settings-drawer');
  const settingsBtn = document.getElementById('settings-toggle-btn');

  if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
    if (!tutorialDrawer.contains(e.target) && !tutorialBtn.contains(e.target)) {
      tutorialDrawer.classList.add('hidden');
    }
  }

  if (settingsDrawer && !settingsDrawer.classList.contains('hidden')) {
    if (!settingsDrawer.contains(e.target) && !settingsBtn.contains(e.target)) {
      settingsDrawer.classList.add('hidden');
    }
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const tutorialDrawer = document.getElementById('tutorial-drawer');
    const settingsDrawer = document.getElementById('settings-drawer');
    if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
      tutorialDrawer.classList.add('hidden');
      return;
    }
    if (settingsDrawer && !settingsDrawer.classList.contains('hidden')) {
      settingsDrawer.classList.add('hidden');
      return;
    }
    handleHeaderCancelClick();
  }
});

// ==========================================================================
// 10. CLIENT-SIDE GAME ENGINE CONTROLLER
// ==========================================================================
function updateConnectionPill(status, customText = null) {
  const pill = document.getElementById('connection-pill');
  if (!pill) return;
  pill.className = `pill ${status}`;
  if (customText) {
    pill.textContent = customText;
  } else {
    pill.textContent = status === 'connected' ? '⬤ CLIENT READY (OFFLINE)' : (status === 'waiting' ? '⬤ THINKING…' : '⬤ OFFLINE');
  }
}

function buildPlayerMap(mode, humanSide) {
  if (mode === 'human') {
    players = { british: 'human', mysore: 'human' };
  } else if (mode === 'ai_vs_ai') {
    players = { british: 'ai', mysore: 'ai' };
  } else if (mode === 'p2p_multiplayer') {
    const mySide = (multiplayerManager.mySide || 'british').toLowerCase();
    players = {
      british: mySide === 'british' ? 'human' : 'remote',
      mysore: mySide === 'mysore' ? 'human' : 'remote'
    };
  } else {
    const human = (humanSide || 'british').toLowerCase();
    players = { british: 'ai', mysore: 'ai' };
    players[human] = 'human';
  }
}

function isCurrentSideAi(uiState) {
  const whoToMove = (uiState.who_to_move || '').toLowerCase();
  if (whoToMove.includes('british')) return players.british === 'ai';
  if (whoToMove.includes('mysore')) return players.mysore === 'ai';
  return false;
}

function isTurnBlockedForLocalPlayer() {
  if (!lastUiState) return true;
  const whoToMove = (lastUiState.who_to_move || '').toLowerCase();
  if (matchMode === 'p2p_multiplayer') {
    if (whoToMove.includes('british')) return players.british !== 'human';
    if (whoToMove.includes('mysore')) return players.mysore !== 'human';
  } else if (matchMode === 'human_vs_ai') {
    if (whoToMove.includes('british')) return players.british !== 'human';
    if (whoToMove.includes('mysore')) return players.mysore !== 'human';
  } else if (matchMode === 'ai_vs_ai') {
    return true;
  }
  return false;
}

function handleLocalGameUpdate(data) {
  clearAllInteractionState();

  currentBitString = data.state_str;
  lastUiState = data.ui_state;
  currentMoves = data.moves || [];

  if (data.ui_state && data.ui_state.nodes) {
    data.ui_state.nodes.forEach(nodeData => {
      const node = window.NODES[nodeData.name];
      if (!node) return;

      if (nodeData.armyType === 'fresh') {
        node.armyType = 'active';
        node.owner = 'british';
      } else if (nodeData.armyType === 'tired') {
        node.armyType = 'tired';
        node.owner = 'british';
      } else if (nodeData.armyType === 'fort') {
        node.armyType = 'fort';
        node.owner = 'mysore';
      } else {
        node.armyType = 'empty';
        node.owner = 'empty';
      }
    });
  }

  window.renderNodes();
  renderAllCards();
  updateTurnHeader(data.ui_state, data.winner);
  renderDebugMoveList();

  if (settings.showEval) {
    startProgressiveEval(data.state_str);
  }

  // Handle AI turn execution locally
  if (data.winner === 0 && isCurrentSideAi(data.ui_state)) {
    const delay = players.british === 'ai' && players.mysore === 'ai' ? 600 : 250;
    setTimeout(() => triggerAiMove(), delay);
  }
}

async function triggerAiMove() {
  if (!currentGameState || getStateWinner(currentGameState) !== 0) return;
  updateConnectionPill('waiting', '⬤ AI THINKING (WASM)…');

  try {
    const { bestMove } = await mctsEngine.findMove(currentGameState, 0.0);
    
    currentGameState = TDEngine.getNextState(currentGameState, bestMove);
    currentGameState = TDEngine.resolveLuck(currentGameState);

    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    updateConnectionPill('connected', '⬤ CLIENT READY (OFFLINE)');
    handleLocalGameUpdate(gameData);
  } catch (err) {
    console.error("AI execution error:", err);
    updateConnectionPill('connected', '⬤ CLIENT READY (OFFLINE)');
  }
}

window.applyMove = function(moveIdx) {
  if (!currentGameState) return;

  try {
    currentGameState = TDEngine.getNextState(currentGameState, moveIdx);
    const { finalState, trajectory } = TDEngine.resolveLuckWithTrajectory(currentGameState);
    currentGameState = finalState;

    if (matchMode === 'p2p_multiplayer') {
      multiplayerManager.sendMove(moveIdx, trajectory, currentGameState.toString());
    }

    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    handleLocalGameUpdate(gameData);
  } catch (err) {
    console.error("Error applying move:", err);
    showToast("Failed to apply move.", 'error');
  }
};

function initGame() {
  renderEdges();
  window.renderNodes();

  currentGameState = new GameState();
  currentGameState.default_setup();
  currentGameState = TDEngine.resolveLuck(currentGameState);

  buildPlayerMap(matchMode, humanPlayerSide);

  const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
  handleLocalGameUpdate(gameData);
  updateConnectionPill('connected', '⬤ CLIENT READY (100% OFFLINE)');
}

// ==========================================================================
// 11. GAME MODE & P2P MULTIPLAYER INTEGRATION
// ==========================================================================
function handleGameModeChange(newMode) {
  matchMode = newMode;
  const p2pPanel = document.getElementById('multiplayer-panel');
  const humanSideRow = document.getElementById('human-side-row');

  if (newMode === 'p2p_multiplayer') {
    if (p2pPanel) p2pPanel.classList.remove('hidden');
    if (humanSideRow) humanSideRow.classList.remove('hidden');
    setupMultiplayerCallbacks();
  } else {
    if (p2pPanel) p2pPanel.classList.add('hidden');
    if (humanSideRow) {
      if (newMode === 'human_vs_ai') humanSideRow.classList.remove('hidden');
      else humanSideRow.classList.add('hidden');
    }
    multiplayerManager.disconnect();
  }

  buildPlayerMap(matchMode, humanPlayerSide);
  initGame();
  showToast(`Switched mode: ${newMode.replace(/_/g, ' ').toUpperCase()}`, 'info');
}

function handleHumanSideChange(side) {
  humanPlayerSide = side;
  buildPlayerMap(matchMode, humanPlayerSide);
  initGame();
  showToast(`Playing as ${side.toUpperCase()}`, 'info');
}

function setupMultiplayerCallbacks() {
  const statusPill = document.getElementById('p2p-status-pill');

  multiplayerManager.onStatusChange = ({ status, isHost, mySide, roomCode }) => {
    if (statusPill) {
      statusPill.className = `p2p-status-pill ${status}`;
      statusPill.textContent = status.toUpperCase();
    }

    if (status === 'connected') {
      updateConnectionPill('connected', `⬤ P2P CONNECTED (${mySide.toUpperCase()})`);
      buildPlayerMap('p2p_multiplayer', mySide);
      showToast(`P2P Connected! You are playing as ${mySide.toUpperCase()}`, 'success');
      toggleSettingsMenu();
    } else if (status === 'hosting') {
      updateConnectionPill('waiting', `⬤ HOSTING (${roomCode})`);
    } else {
      updateConnectionPill('disconnected', '⬤ P2P OFFLINE');
    }
  };

  multiplayerManager.onMoveReceived = (moveIdx, luckTrajectory, stateStr) => {
    showToast("Opponent moved!", 'info');
    if (currentGameState) {
      if (stateStr) {
        try {
          currentGameState = new GameState().read_str(stateStr);
        } catch (e) {
          currentGameState = TDEngine.getNextState(currentGameState, moveIdx);
          currentGameState = TDEngine.applyLuckTrajectory(currentGameState, luckTrajectory);
        }
      } else {
        currentGameState = TDEngine.getNextState(currentGameState, moveIdx);
        currentGameState = TDEngine.applyLuckTrajectory(currentGameState, luckTrajectory);
      }
      const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
      handleLocalGameUpdate(gameData);
    }
  };

  multiplayerManager.onStateSyncReceived = (stateStr) => {
    if (stateStr && currentGameState) {
      try {
        currentGameState = new GameState().read_str(stateStr);
        const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
        handleLocalGameUpdate(gameData);
      } catch (e) {
        console.warn("State sync parse error:", e);
      }
    }
  };

  multiplayerManager.onGameResetReceived = () => {
    showToast("Host reset the board.", 'info');
    initGame();
  };

  multiplayerManager.onResignReceived = (resigningSide) => {
    const winnerVal = resigningSide === 'british' ? -1 : 1;
    const winnerName = winnerVal === 1 ? 'BRITISH' : 'MYSORE';
    showToast(`Opponent (${resigningSide.toUpperCase()}) resigned! ${winnerName} wins!`, 'success');

    const header = document.getElementById('turn-header');
    const counter = document.getElementById('turn-counter');
    const title = document.getElementById('turn-phase-title');
    const icon = document.getElementById('turn-phase-icon');

    if (header && counter && title && icon) {
      header.className = winnerVal === 1
        ? 'turn-header british-phase victory-state'
        : 'turn-header mysore-phase victory-state';
      counter.textContent = 'GAME OVER';
      title.textContent = `${resigningSide.toUpperCase()} RESIGNED — ${winnerName} VICTORY`;
      icon.textContent = winnerVal === 1 ? '🦁' : '🐅';
    }

    currentMoves = [];
    updateActionButtons();
    clearAllInteractionState();
  };

  multiplayerManager.onError = (err) => {
    showToast(`P2P Error: ${err}`, 'error');
  };
}

async function handleHostGameClick() {
  setupMultiplayerCallbacks();
  try {
    const code = await multiplayerManager.hostGame(humanPlayerSide);
    const box = document.getElementById('room-code-box');
    const val = document.getElementById('room-code-val');
    if (box && val) {
      box.classList.remove('hidden');
      val.textContent = code;
    }
    showToast(`Room created! Share code: ${code}`, 'success');
  } catch (err) {
    showToast("Failed to host P2P room.", 'error');
  }
}

async function handleJoinGameClick() {
  setupMultiplayerCallbacks();
  const input = document.getElementById('join-room-input');
  const code = input ? input.value.trim().toUpperCase() : '';
  if (!code) {
    showToast("Enter a room code.", 'error');
    return;
  }
  try {
    await multiplayerManager.joinGame(code);
    showToast(`Connecting to ${code}...`, 'info');
  } catch (err) {
    showToast("Failed to join room.", 'error');
  }
}

function copyRoomCode() {
  const val = document.getElementById('room-code-val');
  if (!val) return;
  navigator.clipboard.writeText(val.textContent).then(() => {
    showToast("Room code copied to clipboard!", 'success');
  });
}

// ==========================================================================
// 12. ZERO-POPUP STATE SAVE / LOAD & UTILITIES
// ==========================================================================
function saveBinaryState() {
  if (!currentBitString) {
    showToast("No active game state to save.", 'error');
    return;
  }
  navigator.clipboard.writeText(currentBitString).then(() => {
    showToast("Binary state copied to clipboard!", 'success');
  });
  toggleSettingsMenu();
}

function loadBinaryState() {
  const input = document.getElementById('binary-load-input');
  const rawInput = input ? input.value.trim() : '';
  if (!rawInput) {
    showToast("Please enter a valid binary state string.", 'error');
    return;
  }
  try {
    currentGameState = new GameState().read_str(rawInput);
    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    handleLocalGameUpdate(gameData);
    showToast("Game state loaded successfully!", 'success');
    toggleSettingsMenu();
  } catch (err) {
    showToast(`Error: ${err.message}`, 'error');
  }
}

function resetGamePrompt() {
  initGame();
  if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendReset();
  }
  showToast("Game reset to starting position.", 'info');
  toggleSettingsMenu();
}

// ==========================================================================
// 13. FLOATING TOOLTIPS & TOAST NOTIFICATIONS
// ==========================================================================
const tooltip = document.getElementById('tooltip');

function tooltipShow(e, name, data) {
  if (!tooltip) return;
  const armyType = data.armyType || 'empty';
  const owner = data.owner === 'empty' ? 'Unoccupied' : data.owner.charAt(0).toUpperCase() + data.owner.slice(1);
  const armyLabel = { active: '⚔ Fresh Army', tired: '😴 Tired Army', fort: '🏰 Fort', empty: 'Empty' };

  let extra = '';
  if (data.key) extra += ' · ⬛ Key City';
  if (data.coast) extra += ' · 🌊 Coastal';

  tooltip.innerHTML = `<b>${name}</b><span style="color:#6a4c1e;">${owner} · ${armyLabel[armyType] || ''}${extra}</span>`;
  tooltip.classList.add('show');
  tooltip.style.left = (e.clientX + 14) + 'px';
  tooltip.style.top = (e.clientY + 14) + 'px';
}

function tooltipHide() {
  if (tooltip) tooltip.classList.remove('show');
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  container.appendChild(toast);

  setTimeout(() => {
    if (toast.parentNode) toast.parentNode.removeChild(toast);
  }, 3000);
}

// Start game client on load
initGame();