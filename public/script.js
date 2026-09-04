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
  { name: 'Wall Breach',      strength: 3, icon: '💥', desc: 'Powerful' },
  { name: 'Highlanders',      strength: 2, icon: '🎖️', desc: 'Deploy a Fresh Army on Coast' },
  { name: 'Royal Navy',       strength: 2, icon: '⚓', desc: 'Move an Army to any Coast' },
  { name: 'Divide and Rule',  strength: 1, icon: '🤝', desc: 'Move a Fort not in a Key' },
  { name: 'Force March',      strength: 1, icon: '🥾', desc: 'Move a Tired Army' },
  { name: 'Princely States',  strength: 1, icon: '🏰', desc: 'Deploy a Tired Army in a Key' },
];

const MYSORE_CARD_DATA = [
  { name: 'Iron Rockets',     strength: 3, icon: '🚀', desc: 'Powerful' },
  { name: 'Sepoy Mutiny',     strength: 2, icon: '⚔️', desc: 'Remove an Army not in a Key' },
  { name: 'French Alliance',  strength: 2, icon: '🇫🇷', desc: 'Deploy a Fort adjacent to another Fort' },
  { name: 'Monsoon',          strength: 1, icon: '🌧️', desc: 'Flip a Fresh Army to Tired' },
  { name: 'Cavalry Raid',     strength: 1, icon: '🏇', desc: 'British discard' },
  { name: 'Sea Trade',        strength: 1, icon: '🪙', desc: 'Move a Fort from Coast to any' },
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
let activeSelection = null;            // Selected card name

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
const mctsEngine = new TDMCTS.MCTS(onnxModel, { simulations: 800, depsilon: 0.1 }); // Micro Dirichlet noise
const multiplayerManager = new TDMultiplayer.MultiplayerManager();

// Preload ONNX model in background
onnxModel.init().catch(err => console.log("ONNX preload notice:", err));

/**
 * Central state reset — clears ALL interactive UI state (unit selection,
 * card targeting highlights, trade staging) and refreshes the map + cards.
 */
function clearAllInteractionState() {
  selectedUnit = null;
  activeSelection = null; // Replaces all card modes
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

  if (activeSelection) {
    if (!activeSelection.isTwoStep) {
      Object.keys(activeSelection.mapTargets).forEach(nodeName => {
        const el = getNodeElement(nodeName);
        if (el) {
          el.classList.add('valid-map-target');
          const glowRing = el.querySelector('.target-glow-ring');
          if (glowRing) glowRing.setAttribute('opacity', '1');
        }
      });
    } else {
      if (activeSelection.step === 1) {
        Object.keys(activeSelection.validSources).forEach(srcName => {
          const el = getNodeElement(srcName);
          if (el) {
            el.classList.add('valid-map-target');
            const glowRing = el.querySelector('.target-glow-ring');
            if (glowRing) glowRing.setAttribute('opacity', '1');
          }
        });
      } else if (activeSelection.step === 2 && activeSelection.sourceNode) {
        const srcEl = getNodeElement(activeSelection.sourceNode);
        if (srcEl) {
          srcEl.classList.add('selected-unit');
          const ring = srcEl.querySelector('.sel-ring');
          if (ring) ring.setAttribute('opacity', '1');
        }

        const dests = activeSelection.validSources[activeSelection.sourceNode] || [];
        dests.forEach(d => {
          const destEl = getNodeElement(d.dest);
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

function handleNodeClick(nodeName) {
  if (isViewingHistory) return;
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  if (activeSelection) {
    if (!activeSelection.isTwoStep) {
      if (activeSelection.mapTargets[nodeName] !== undefined) {
        const moveIdx = activeSelection.mapTargets[nodeName];
        const cName = activeSelection.cardName;
        clearAllInteractionState();
        showToast(`Activated ${cName} on ${nodeName}!`, 'success');
        window.applyMove(moveIdx);
      } else {
        showToast("Invalid target.", "error");
      }
      return;
    } 
    
    if (activeSelection.step === 1) {
      if (activeSelection.validSources[nodeName]) {
        activeSelection.sourceNode = nodeName;
        activeSelection.step = 2;
        refreshMapHighlights();
        updateTurnHeaderInstruction();
      } else {
        showToast(`${nodeName} cannot be selected as a source.`, 'error');
      }
      return;
    } 
    
    if (activeSelection.step === 2) {
      if (nodeName === activeSelection.sourceNode) {
        activeSelection.sourceNode = null;
        activeSelection.step = 1;
        refreshMapHighlights();
        updateTurnHeaderInstruction();
        return;
      }

      const dests = activeSelection.validSources[activeSelection.sourceNode] || [];
      const destMove = dests.find(d => d.dest === nodeName);
      
      if (destMove) {
        const cName = activeSelection.cardName;
        const targetMoveStr = `${activeSelection.sourceNode} -> ${nodeName}`;
        clearAllInteractionState();
        showToast(`Executed ${cName}: ${targetMoveStr}!`, 'success');
        window.applyMove(destMove.idx);
      } else {
        showToast(`${nodeName} is not a valid destination.`, 'error');
      }
      return;
    }
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

// ==========================================================================
// 6. CARD INTERACTIONS
// ==========================================================================
let mobileActiveFaction = 'british';

function switchMobileFactionTab(faction) {
  mobileActiveFaction = faction;
  const mysoreTab = document.getElementById('mobile-tab-mysore');
  const britishTab = document.getElementById('mobile-tab-british');
  const mysoreCol = document.getElementById('mysore-column');
  const britishCol = document.getElementById('british-column');

  if (mysoreTab && britishTab) {
    if (faction === 'mysore') {
      mysoreTab.classList.add('active-tab');
      britishTab.classList.remove('active-tab');
      if (mysoreCol) mysoreCol.classList.remove('mobile-hidden');
      if (britishCol) britishCol.classList.add('mobile-hidden');
    } else {
      britishTab.classList.add('active-tab');
      mysoreTab.classList.remove('active-tab');
      if (britishCol) britishCol.classList.remove('mobile-hidden');
      if (mysoreCol) mysoreCol.classList.add('mobile-hidden');
    }
  }
}
window.switchMobileFactionTab = switchMobileFactionTab;

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

  const mMobileBadge = document.getElementById('mobile-mysore-count');
  const bMobileBadge = document.getElementById('mobile-british-count');
  if (mMobileBadge) mMobileBadge.textContent = `${mAvail}/6`;
  if (bMobileBadge) bMobileBadge.textContent = `${bAvail}/6`;
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

    if (activeSelection && activeSelection.faction === faction && activeSelection.cardName === card.name) {
      classNames.push('targeting-active');
    }

    if (activeSelection && activeSelection.faction === faction && !isUsable) {
      if (activeSelection.tradeTargets && activeSelection.tradeTargets[card.name] !== undefined) {
        classNames.push('valid-trade-target');
      }
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

    cardDiv.addEventListener('click', (e) => {
      e.stopPropagation();
      handleCardBodyClick(faction, index, card.name, isUsable);
    });

    container.appendChild(cardDiv);
  });
}

function handleCardStrengthClick(faction, index, cardName) {
  if (isViewingHistory) return;
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
  if (isViewingHistory) return;
  if (isTurnBlockedForLocalPlayer()) {
    showToast("It is not your turn.", 'error');
    return;
  }

  // 1. Resolve a Trade if clicking an exhausted card while a selection is active
  if (!isUsable) {
    if (activeSelection && activeSelection.tradeTargets && activeSelection.tradeTargets[cardName] !== undefined) {
      const moveIdx = activeSelection.tradeTargets[cardName];
      const tradedCard = activeSelection.cardName;
      clearAllInteractionState();
      showToast(`Traded ${tradedCard} to reclaim ${cardName}!`, 'success');
      window.applyMove(moveIdx);
      return;
    }
    showToast(`${cardName} is exhausted. Click an active card to stage a trade.`, 'info');
    return;
  }

  // 2. Toggle off if clicking the same card
  if (activeSelection && activeSelection.cardName === cardName) {
    clearAllInteractionState();
    return;
  }

  // 3. Generate Unified Selection Object
  clearAllInteractionState(); 

  const newSelection = {
    cardName: cardName,
    faction: faction,
    mapTargets: {},      // Format: { "NodeName": moveIdx }
    tradeTargets: {},    // Format: { "ExhaustedCardName": moveIdx }
    validSources: {},    // Format: { "SourceName": [{ dest: "DestName", idx: moveIdx }] } (For 2-step)
    isTwoStep: false,
    step: 1,
    sourceNode: null
  };

  // A. Scan for map targets (type: cardName)
  const abilityMoves = currentMoves.filter(m => m.type === cardName);
  const twoStepCards = ['Divide and Rule', 'Force March', 'Royal Navy', 'Sea Trade'];

  if (abilityMoves.length > 0) {
    // Immediate Execute (e.g., Cavalry Raid)
    if (cardName === 'Cavalry Raid') { 
      showToast(`Cavalry Raid launched! British must discard.`, 'success');
      window.applyMove(abilityMoves[0].idx);
      return;
    }

    if (twoStepCards.includes(cardName)) {
      newSelection.isTwoStep = true;
      abilityMoves.forEach(m => {
        const [src, dest] = m.desc.split(' -> ');
        if (!newSelection.validSources[src]) newSelection.validSources[src] = [];
        newSelection.validSources[src].push({ dest: dest, idx: m.idx });
      });
    } else {
      abilityMoves.forEach(m => {
        newSelection.mapTargets[m.desc] = m.idx;
      });
    }
  }

  // B. Scan for trade targets (type: "Draw cardName")
  const tradeType = `Draw ${cardName}`;
  const tradeMoves = currentMoves.filter(m => m.type === tradeType);
  tradeMoves.forEach(m => {
    newSelection.tradeTargets[m.desc] = m.idx;
  });

  // Validate state
  if (Object.keys(newSelection.mapTargets).length === 0 &&
      Object.keys(newSelection.tradeTargets).length === 0 &&
      Object.keys(newSelection.validSources).length === 0) {
      showToast(`No legal actions or trades available for ${cardName}.`, 'info');
      return;
  }

  // 4. Mount the Unified State & Render
  activeSelection = newSelection;
  renderAllCards();
  refreshMapHighlights();
  updateActionButtons();
  updateTurnHeaderInstruction();
  showToast(`Activated ${cardName}. Select a map target or card to trade.`, 'info');
}

function updateTurnHeaderInstruction() {
  // Turn instruction banner is retired in streamlined UI
}

function renderDebugMoveList() {
  // Debug move console is retired in streamlined UI
}

// ==========================================================================
// 7. HEADER INSTRUCTIONS & ACTION BUTTONS
// ==========================================================================
function updateTurnHeader(uiState, winner) {
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');

  if (!header || !uiState) return;

  if (winner === 1) {
    header.className = 'app-header british-phase victory-state';
    if (counter) counter.textContent = 'GAME OVER';
    if (title) title.textContent = 'BRITISH VICTORY';
    showToast('The British East India Company has achieved total dominion!', 'success');
    return;
  } else if (winner === -1) {
    header.className = 'app-header mysore-phase victory-state';
    if (counter) counter.textContent = 'GAME OVER';
    if (title) title.textContent = 'MYSORE VICTORY';
    showToast('The Sultanate of Mysore has repelled the British invasion!', 'success');
    return;
  }

  const whoToMove = uiState.who_to_move || 'British Move';
  if (counter) counter.textContent = `TURN ${uiState.turn} OF 4`;

  if (whoToMove === 'British Move') {
    header.className = 'app-header british-phase';
    if (title) title.textContent = 'BRITISH: MOVE ARMY';
    if (window.innerWidth <= 860 && mobileActiveFaction !== 'british') {
      switchMobileFactionTab('british');
    }
  } else if (whoToMove === 'Mysore Card') {
    header.className = 'app-header mysore-phase';
    if (title) title.textContent = 'MYSORE: PLAY CARD';
    if (window.innerWidth <= 860 && mobileActiveFaction !== 'mysore') {
      switchMobileFactionTab('mysore');
    }
  } else if (whoToMove === 'British Card') {
    header.className = 'app-header british-phase';
    if (title) title.textContent = 'BRITISH: PLAY CARD';
    if (window.innerWidth <= 860 && mobileActiveFaction !== 'british') {
      switchMobileFactionTab('british');
    }
  }

  renderBattleMarker(uiState);
  updateActionButtons();
}

function updateActionButtons() {
  const restBtn = document.getElementById('header-rest-btn');
  const passBtn = document.getElementById('header-pass-btn');
  const cancelBtn = document.getElementById('header-cancel-btn');

  if (cancelBtn) {
    if (selectedUnit || activeSelection) {
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
    if (hasPass && !activeSelection) {
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
  if (typeof adjustBoardDimensions === 'function') {
    setTimeout(adjustBoardDimensions, 10);
  }
}

function setEvalBar(score, totalSims) {
  const bar = document.getElementById('eval-bar-mysore');
  const scoreLabel = document.getElementById('eval-bar-score');
  const simsLabel = document.getElementById('eval-sims-label');
  const winrateLabel = document.getElementById('eval-winrate-label');

  if (!bar || !scoreLabel) return;

  const mysorePercentage = (1 - score) / 2 * 100;
  bar.style.width = `${mysorePercentage}%`;

  const sign = score > 0 ? '+' : '';
  scoreLabel.textContent = `${sign}${score.toFixed(2)}`;
  if (simsLabel) simsLabel.textContent = `Engine: ${totalSims} sims (Wasm)`;

  if (winrateLabel) {
    if (score > 0.05) winrateLabel.textContent = `British Win (${(100 - mysorePercentage).toFixed(0)}%)`;
    else if (score < -0.05) winrateLabel.textContent = `Mysore Win (${mysorePercentage.toFixed(0)}%)`;
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
// 9. DRAWER MENUS
// ==========================================================================
const TUTORIAL_DISMISSED_KEY = 'tigersday_tutorial_dismissed';

function isTutorialDismissed() {
  try {
    return localStorage.getItem(TUTORIAL_DISMISSED_KEY) === '1';
  } catch (err) {
    return false;
  }
}

function setTutorialDismissed() {
  try {
    localStorage.setItem(TUTORIAL_DISMISSED_KEY, '1');
  } catch (err) {
    // Gracefully handle storage quota or private browsing restrictions
  }
}

function openTutorialDrawer() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const settingsDrawer = document.getElementById('settings-drawer');
  if (settingsDrawer) settingsDrawer.classList.add('hidden');
  if (tutorialDrawer) tutorialDrawer.classList.remove('hidden');
}

function closeTutorialDrawer(markDismissed = true) {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
    tutorialDrawer.classList.add('hidden');
  }
  if (markDismissed) {
    setTutorialDismissed();
  }
}

function toggleTutorialMenu() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
    closeTutorialDrawer(true);
  } else {
    openTutorialDrawer();
  }
}

function toggleSettingsMenu() {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const settingsDrawer = document.getElementById('settings-drawer');
  if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
    closeTutorialDrawer(true);
  }
  if (settingsDrawer) settingsDrawer.classList.toggle('hidden');
}

document.addEventListener('click', (e) => {
  const tutorialDrawer = document.getElementById('tutorial-drawer');
  const tutorialBtn = document.getElementById('tutorial-toggle-btn');
  const settingsDrawer = document.getElementById('settings-drawer');
  const settingsBtn = document.getElementById('settings-toggle-btn');

  if (tutorialDrawer && !tutorialDrawer.classList.contains('hidden')) {
    if (!tutorialDrawer.contains(e.target) && !tutorialBtn.contains(e.target)) {
      closeTutorialDrawer(true);
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
      closeTutorialDrawer(true);
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

// ==========================================================================
// 8. CHESS.COM STYLE NOTATION, REPLAY HISTORY & GAME REVIEW ENGINE
// ==========================================================================
let gameHistory = [];
let isViewingHistory = false;
let browsingHistoryIndex = -1;
let liveGameState = null;

const BRITISH_CARD_NAMES = [
  "Iron Rockets", "Wall Breach", "Sepoy Mutiny", "French Help", "Maratha Alliance", "Chitaldoorg Defection"
];
const MYSORE_CARD_NAMES = [
  "Royal Navy", "Highlanders", "Force March", "Sea Trade", "Diplomatic Mission", "Cavalry Raid"
];

function isTurnBlockedForLocalPlayer() {
  if (isViewingHistory) return true;
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

function recordMoveInHistory(stateBefore, moveIdx, nextState, finalState) {
  if (!stateBefore) return;

  const actor = stateBefore.to_move === 0 ? 'british' : 'mysore';
  const turn = stateBefore.turn;
  let notation = TDEngine.notate(stateBefore, moveIdx);

  // Clean move description from currentMoves if found
  let moveDesc = "";
  const matched = currentMoves.find(m => m.idx === moveIdx);
  if (matched) {
    moveDesc = `${matched.type}${matched.desc ? ': ' + matched.desc : ''}`;
  } else {
    moveDesc = notation;
  }

  // Detect if random battle card discard occurred (stochastic luck edge case)
  let luckEvent = null;
  if (nextState && finalState) {
    // Check if British had a card randomly discarded
    for (let i = 0; i < 6; i++) {
      if (nextState.british_cards[i] && !finalState.british_cards[i]) {
        luckEvent = {
          faction: 'british',
          cardName: BRITISH_CARD_NAMES[i] || `Card #${i+1}`
        };
        break;
      }
    }
    // Check if Mysore had a card randomly discarded
    if (!luckEvent) {
      for (let i = 0; i < 6; i++) {
        if (nextState.mysore_cards[i] && !finalState.mysore_cards[i]) {
          luckEvent = {
            faction: 'mysore',
            cardName: MYSORE_CARD_NAMES[i] || `Card #${i+1}`
          };
          break;
        }
      }
    }
  }

  const historyEntry = {
    step: gameHistory.length + 1,
    turn: turn,
    actor: actor,
    moveIdx: moveIdx,
    notation: notation,
    desc: `${actor === 'british' ? 'British' : 'Mysore'}: ${moveDesc}`,
    stateBeforeStr: stateBefore.toString(),
    stateAfterStr: finalState ? finalState.toString() : stateBefore.toString(),
    hasLuck: !!luckEvent,
    luckDetail: luckEvent 
      ? `🎲 ${luckEvent.faction === 'british' ? 'British' : 'Mysore'} discarded '${luckEvent.cardName}' (battle loss)` 
      : null
  };

  gameHistory.push(historyEntry);
}

function renderNotationPanel() {
  const moveCount = gameHistory.length;

  // 1. Update badges
  const notBadge = document.getElementById('notation-move-badge');
  const rightColBadge = document.getElementById('right-col-moves-badge');
  const mobileBadge = document.getElementById('mobile-moves-count');
  if (notBadge) notBadge.textContent = moveCount;
  if (rightColBadge) rightColBadge.textContent = moveCount;
  if (mobileBadge) mobileBadge.textContent = moveCount;

  // 2. Update phase indicator
  const phaseLabel = document.getElementById('notation-phase-name');
  if (phaseLabel && lastUiState) {
    const who = (lastUiState.who_to_move || 'British').toUpperCase();
    phaseLabel.textContent = `Turn ${lastUiState.turn || 1} · ${who} PHASE`;
  }

  // 3. Populate 2-Column Move Notation Table (Chess.com Style)
  const list = document.getElementById('moves-history-list');
  if (list) {
    if (moveCount === 0) {
      list.innerHTML = `<div class="notation-empty-msg">No moves played yet</div>`;
    } else {
      let html = '';
      // Group impulses by round pairs (1. British Move | Mysore Move)
      let roundNum = 1;
      for (let i = 0; i < moveCount; i += 2) {
        const bEntry = gameHistory[i];
        const mEntry = i + 1 < moveCount ? gameHistory[i + 1] : null;

        const bActive = isViewingHistory && browsingHistoryIndex === i;
        const mActive = isViewingHistory && browsingHistoryIndex === (i + 1);

        const bLuck = bEntry && bEntry.hasLuck ? ' <span class="luck-badge" title="Battle Random Discard">🎲</span>' : '';
        const mLuck = mEntry && mEntry.hasLuck ? ' <span class="luck-badge" title="Battle Random Discard">🎲</span>' : '';

        html += `
          <div class="notation-row ${roundNum % 2 === 0 ? 'even-row' : 'odd-row'}">
            <span class="col-num">${roundNum}.</span>
            <span class="col-faction notation-cell ${bActive ? 'active-step' : ''}" onclick="viewHistoricalStep(${i})" title="${bEntry ? bEntry.desc : ''}">
              ${bEntry ? bEntry.notation + bLuck : '—'}
            </span>
            <span class="col-faction notation-cell ${mActive ? 'active-step' : ''} ${!mEntry ? 'empty-cell' : ''}" ${mEntry ? `onclick="viewHistoricalStep(${i + 1})" title="${mEntry.desc}"` : ''}>
              ${mEntry ? mEntry.notation + mLuck : '...'}
            </span>
          </div>
        `;
        roundNum++;
      }
      list.innerHTML = html;

      // Auto-scroll to active or bottom row
      const activeCell = list.querySelector('.active-step');
      if (activeCell) {
        activeCell.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        list.scrollTop = list.scrollHeight;
      }
    }
  }

  // 5. Update Stepping Buttons State
  const btnStart = document.getElementById('btn-step-start');
  const btnPrev = document.getElementById('btn-step-prev');
  const btnNext = document.getElementById('btn-step-next');
  const btnLive = document.getElementById('btn-step-live');

  if (moveCount === 0) {
    if (btnStart) btnStart.disabled = true;
    if (btnPrev) btnPrev.disabled = true;
    if (btnNext) btnNext.disabled = true;
    if (btnLive) {
      btnLive.disabled = true;
      btnLive.classList.remove('live-active');
    }
  } else if (!isViewingHistory) {
    if (btnStart) btnStart.disabled = false;
    if (btnPrev) btnPrev.disabled = false;
    if (btnNext) btnNext.disabled = true;
    if (btnLive) {
      btnLive.disabled = false;
      btnLive.classList.remove('live-active');
    }
  } else {
    if (btnStart) btnStart.disabled = browsingHistoryIndex <= 0;
    if (btnPrev) btnPrev.disabled = browsingHistoryIndex <= 0;
    if (btnNext) btnNext.disabled = false;
    if (btnLive) {
      btnLive.disabled = false;
      btnLive.classList.add('live-active');
    }
  }
}

// ──────────────────────────────────────────────────────────────────────────
// HISTORICAL STATE INSPECTION (READ-ONLY)
// ──────────────────────────────────────────────────────────────────────────
function viewHistoricalStep(stepIndex) {
  if (stepIndex < 0 || stepIndex >= gameHistory.length) return;

  if (!isViewingHistory) {
    liveGameState = currentGameState.copy();
  }

  isViewingHistory = true;
  browsingHistoryIndex = stepIndex;

  const entry = gameHistory[stepIndex];
  try {
    const histState = new GameState().read_str(entry.stateAfterStr);
    currentGameState = histState;

    const gameData = TDEngine.generateGameData(histState, matchMode, humanPlayerSide);
    gameData.moves = []; // Strictly lock interactions while reviewing

    handleHistoricalRender(gameData, entry);

    const banner = document.getElementById('historical-review-banner');
    const title = document.getElementById('review-banner-title');
    if (banner && title) {
      title.textContent = `Viewing Move #${entry.step} (${entry.notation}) · ${entry.actor === 'british' ? 'British' : 'Mysore'} Turn ${entry.turn}`;
      banner.classList.remove('hidden');
    }

    renderNotationPanel();
    adjustBoardDimensions();
  } catch (err) {
    console.error("Failed to render historical state:", err);
    showToast("Error inspecting historical state.", "error");
  }
}

function handleHistoricalRender(data, entry) {
  clearAllInteractionState();
  currentMoves = []; // Disallow moves in historical review mode

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

  // Update header in review mode
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');
  const icon = document.getElementById('turn-phase-icon');
  const instruction = document.getElementById('turn-instruction');

  if (header) {
    header.className = entry.actor === 'british'
      ? 'app-header british-phase historical-mode'
      : 'app-header mysore-phase historical-mode';
  }
  if (counter) counter.textContent = `HISTORY · MOVE #${entry.step}`;
  if (title) title.textContent = `${entry.actor.toUpperCase()}: ${entry.notation}`;
}

function returnToLiveGame() {
  if (!isViewingHistory) return;

  isViewingHistory = false;
  browsingHistoryIndex = -1;

  if (liveGameState) {
    currentGameState = liveGameState;
    liveGameState = null;
  }

  const banner = document.getElementById('historical-review-banner');
  if (banner) banner.classList.add('hidden');

  const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
  handleLocalGameUpdate(gameData);
  renderNotationPanel();
  adjustBoardDimensions();
  showToast("Returned to live game.", "info");
}

function stepHistoryPrev() {
  if (gameHistory.length === 0) return;
  if (!isViewingHistory) {
    viewHistoricalStep(gameHistory.length - 1);
  } else if (browsingHistoryIndex > 0) {
    viewHistoricalStep(browsingHistoryIndex - 1);
  }
}

function stepHistoryNext() {
  if (!isViewingHistory) return;
  if (browsingHistoryIndex < gameHistory.length - 1) {
    viewHistoricalStep(browsingHistoryIndex + 1);
  } else {
    returnToLiveGame();
  }
}

// ──────────────────────────────────────────────────────────────────────────
// OFFER DRAW & RESIGN ACTIONS
// ──────────────────────────────────────────────────────────────────────────
function handleOfferDraw() {
  if (isViewingHistory) {
    showToast("Return to live game before offering a draw.", "info");
    return;
  }
  if (!currentGameState || getStateWinner(currentGameState) !== 0) {
    showToast("Game is already finished.", "info");
    return;
  }

  if (matchMode === 'human_vs_ai') {
    // Evaluate if position is reasonably balanced for AI to accept
    let freshArmies = 0, forts = 0;
    for (let i = 0; i < currentGameState.fresh_armies.length; i++) {
      if (currentGameState.fresh_armies[i] || currentGameState.tired_armies[i]) freshArmies++;
      if (currentGameState.forts[i]) forts++;
    }

    if (forts >= 3 && freshArmies >= 3) {
      declareDraw("AI accepted your draw offer! Peaceful armistice concluded.");
    } else {
      showToast("AI declined the draw offer. The campaign continues!", "info");
    }
  } else if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendOfferDraw(humanPlayerSide);
    showToast("Draw offer sent to opponent...", "info");
  } else {
    const confirmDraw = confirm("Offer a Draw? If both players agree, the match will conclude in a draw.");
    if (confirmDraw) {
      declareDraw("Game drawn by mutual agreement.");
    }
  }
}

function declareDraw(reason) {
  showToast(reason, "info");
  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');

  if (header) header.className = 'app-header game-over-phase victory-state';
  if (counter) counter.textContent = 'MATCH DRAW';
  if (title) title.textContent = 'DRAW AGREED';

  gameHistory.push({
    step: gameHistory.length + 1,
    turn: currentGameState ? currentGameState.turn : 1,
    actor: 'system',
    moveIdx: -1,
    notation: '½–½',
    desc: reason,
    stateBeforeStr: currentGameState ? currentGameState.toString() : '',
    stateAfterStr: currentGameState ? currentGameState.toString() : '',
    hasLuck: false,
    luckDetail: null
  });

  currentMoves = [];
  updateActionButtons();
  clearAllInteractionState();
  renderNotationPanel();
}

function handleResignClick() {
  if (isViewingHistory) return;
  if (!currentGameState || getStateWinner(currentGameState) !== 0) {
    showToast("Game is already finished.", "info");
    return;
  }

  const resigningSide = (matchMode === 'human_vs_ai' || matchMode === 'p2p_multiplayer') 
    ? humanPlayerSide 
    : (currentGameState.to_move === 0 ? 'british' : 'mysore');

  const confirmResign = confirm(`Are you sure you want to resign as ${resigningSide.toUpperCase()}?`);
  if (!confirmResign) return;

  if (matchMode === 'p2p_multiplayer') {
    multiplayerManager.sendResign(resigningSide);
  }

  const winnerVal = resigningSide === 'british' ? -1 : 1;
  const winnerName = winnerVal === 1 ? 'BRITISH' : 'MYSORE';
  showToast(`${resigningSide.toUpperCase()} resigned. ${winnerName} wins!`, 'info');

  const header = document.getElementById('turn-header');
  const counter = document.getElementById('turn-counter');
  const title = document.getElementById('turn-phase-title');

  if (header) {
    header.className = winnerVal === 1
      ? 'app-header british-phase victory-state'
      : 'app-header mysore-phase victory-state';
  }
  if (counter) counter.textContent = 'GAME OVER';
  if (title) title.textContent = `${resigningSide.toUpperCase()} RESIGNED — ${winnerName} VICTORY`;

  gameHistory.push({
    step: gameHistory.length + 1,
    turn: currentGameState ? currentGameState.turn : 1,
    actor: resigningSide,
    moveIdx: -1,
    notation: resigningSide === 'british' ? '0–1' : '1–0',
    desc: `${resigningSide.toUpperCase()} resigned. ${winnerName} Victory.`,
    stateBeforeStr: currentGameState ? currentGameState.toString() : '',
    stateAfterStr: currentGameState ? currentGameState.toString() : '',
    hasLuck: false,
    luckDetail: null
  });

  currentMoves = [];
  updateActionButtons();
  clearAllInteractionState();
  renderNotationPanel();
}

// ──────────────────────────────────────────────────────────────────────────
// TAB SWITCHING HELPERS
// ──────────────────────────────────────────────────────────────────────────
function switchNotationTab(tab) {
  const movesTabBtn = document.getElementById('tab-btn-moves');
  const infoTabBtn = document.getElementById('tab-btn-info');
  const movesContent = document.getElementById('notation-tab-moves-content');
  const infoContent = document.getElementById('notation-tab-info-content');

  if (tab === 'moves') {
    if (movesTabBtn) movesTabBtn.classList.add('active');
    if (infoTabBtn) infoTabBtn.classList.remove('active');
    if (movesContent) movesContent.classList.remove('hidden');
    if (infoContent) infoContent.classList.add('hidden');
  } else {
    if (infoTabBtn) infoTabBtn.classList.add('active');
    if (movesTabBtn) movesTabBtn.classList.remove('active');
    if (infoContent) infoContent.classList.remove('hidden');
    if (movesContent) movesContent.classList.add('hidden');
  }
}

function switchRightColumnView(view) {
  const handTabBtn = document.getElementById('right-col-tab-hand');
  const movesTabBtn = document.getElementById('right-col-tab-moves');
  const handContainer = document.getElementById('british-hand-container');
  const notationPanel = document.getElementById('notation-panel');

  if (view === 'hand') {
    if (handTabBtn) handTabBtn.classList.add('active');
    if (movesTabBtn) movesTabBtn.classList.remove('active');
    if (handContainer) handContainer.classList.remove('hidden');
    if (notationPanel) notationPanel.classList.remove('active-in-right-col');
  } else {
    if (movesTabBtn) movesTabBtn.classList.add('active');
    if (handTabBtn) handTabBtn.classList.remove('active');
    if (handContainer) handContainer.classList.add('hidden');
    if (notationPanel) notationPanel.classList.add('active-in-right-col');
  }
}

function switchMobileFactionTab(tab) {
  const bTab = document.getElementById('mobile-tab-british');
  const mTab = document.getElementById('mobile-tab-mysore');
  const movesTab = document.getElementById('mobile-tab-moves');
  const bCol = document.getElementById('british-column');
  const mCol = document.getElementById('mysore-column');
  const notPanel = document.getElementById('notation-panel');

  if (bTab) bTab.classList.remove('active-tab');
  if (mTab) mTab.classList.remove('active-tab');
  if (movesTab) movesTab.classList.remove('active-tab');

  if (tab === 'british') {
    if (bTab) bTab.classList.add('active-tab');
    if (bCol) bCol.classList.remove('mobile-hidden');
    if (mCol) mCol.classList.add('mobile-hidden');
    if (notPanel) notPanel.classList.remove('mobile-active');
  } else if (tab === 'mysore') {
    if (mTab) mTab.classList.add('active-tab');
    if (mCol) mCol.classList.remove('mobile-hidden');
    if (bCol) bCol.classList.add('mobile-hidden');
    if (notPanel) notPanel.classList.remove('mobile-active');
  } else if (tab === 'moves') {
    if (movesTab) movesTab.classList.add('active-tab');
    if (bCol) bCol.classList.add('mobile-hidden');
    if (mCol) mCol.classList.add('mobile-hidden');
    if (notPanel) notPanel.classList.add('mobile-active');
  }
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
    const stateBefore = currentGameState.copy();
    const nextState = TDEngine.getNextState(currentGameState, bestMove);
    const { finalState } = TDEngine.resolveLuckWithTrajectory(nextState);
    currentGameState = finalState;

    recordMoveInHistory(stateBefore, bestMove, nextState, finalState);

    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    updateConnectionPill('connected', '⬤ CLIENT READY (OFFLINE)');
    handleLocalGameUpdate(gameData);
    renderNotationPanel();
  } catch (err) {
    console.error("AI execution error:", err);
    updateConnectionPill('connected', '⬤ CLIENT READY (OFFLINE)');
  }
}

window.applyMove = function(moveIdx) {
  if (!currentGameState) return;
  if (isViewingHistory) return;

  try {
    const stateBefore = currentGameState.copy();
    const nextState = TDEngine.getNextState(currentGameState, moveIdx);
    const { finalState, trajectory } = TDEngine.resolveLuckWithTrajectory(nextState);
    currentGameState = finalState;

    recordMoveInHistory(stateBefore, moveIdx, nextState, finalState);

    if (matchMode === 'p2p_multiplayer') {
      multiplayerManager.sendMove(moveIdx, trajectory, currentGameState.toString());
    }

    const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
    handleLocalGameUpdate(gameData);
    renderNotationPanel();
  } catch (err) {
    console.error("Error applying move:", err);
    showToast("Failed to apply move.", 'error');
  }
};

function initGame() {
  renderEdges();
  window.renderNodes();

  gameHistory = [];
  isViewingHistory = false;
  browsingHistoryIndex = -1;
  liveGameState = null;

  const banner = document.getElementById('historical-review-banner');
  if (banner) banner.classList.add('hidden');

  currentGameState = new GameState();
  currentGameState.default_setup();
  currentGameState = TDEngine.resolveLuck(currentGameState);

  buildPlayerMap(matchMode, humanPlayerSide);

  const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
  handleLocalGameUpdate(gameData);
  renderNotationPanel();
}

// ==========================================================================
// 11. GAME MODE & P2P MULTIPLAYER INTEGRATION
// ==========================================================================
function handleGameModeChange(newMode) {
  matchMode = newMode;
  const p2pPanel = document.getElementById('multiplayer-panel');
  const humanSideRow = document.getElementById('human-side-row');
  const infoMode = document.getElementById('info-mode-label');
  if (infoMode) infoMode.textContent = `Mode: ${newMode.replace(/_/g, ' ').toUpperCase()}`;

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
      const stateBefore = currentGameState.copy();
      let nextState = null;
      if (stateStr) {
        try {
          currentGameState = new GameState().read_str(stateStr);
        } catch (e) {
          nextState = TDEngine.getNextState(currentGameState, moveIdx);
          currentGameState = TDEngine.applyLuckTrajectory(nextState, luckTrajectory);
        }
      } else {
        nextState = TDEngine.getNextState(currentGameState, moveIdx);
        currentGameState = TDEngine.applyLuckTrajectory(nextState, luckTrajectory);
      }
      recordMoveInHistory(stateBefore, moveIdx, nextState, currentGameState);
      const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
      handleLocalGameUpdate(gameData);
      renderNotationPanel();
    }
  };

  multiplayerManager.onStateSyncReceived = (stateStr) => {
    if (stateStr && currentGameState) {
      try {
        currentGameState = new GameState().read_str(stateStr);
        const gameData = TDEngine.generateGameData(currentGameState, matchMode, humanPlayerSide);
        handleLocalGameUpdate(gameData);
        renderNotationPanel();
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

    if (header) {
      header.className = winnerVal === 1
        ? 'app-header british-phase victory-state'
        : 'app-header mysore-phase victory-state';
    }
    if (counter) counter.textContent = 'GAME OVER';
    if (title) title.textContent = `${resigningSide.toUpperCase()} RESIGNED — ${winnerName} VICTORY`;

    currentMoves = [];
    updateActionButtons();
    clearAllInteractionState();
    renderNotationPanel();
  };

  multiplayerManager.onDrawOfferReceived = (offeringSide) => {
    const accept = confirm(`Opponent (${offeringSide.toUpperCase()}) offered a draw. Accept peace treaty?`);
    if (accept) {
      multiplayerManager.sendAcceptDraw();
      declareDraw("Draw agreed mutually with opponent!");
    } else {
      multiplayerManager.sendDeclineDraw();
      showToast("You declined the draw offer.", "info");
    }
  };

  multiplayerManager.onDrawAcceptedReceived = () => {
    declareDraw("Opponent accepted your draw offer! Match drawn.");
  };

  multiplayerManager.onDrawDeclinedReceived = () => {
    showToast("Opponent declined your draw offer.", "info");
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
  if (data.key) extra += ' · ⬛ Key';
  if (data.coast) extra += ' · 🌊 Coast';

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

function renderBattleMarker(uiState) {
  const layer = document.getElementById('battle-layer');
  if (!layer) return;
  layer.innerHTML = '';

  if (!uiState || uiState.attacker === 'None' || uiState.defender === 'None') return;

  const attNode = window.NODES[uiState.attacker];
  const defNode = window.NODES[uiState.defender];
  if (!attNode || !defNode) return;

  const mx = (attNode.x + defNode.x) / 2;
  const my = (attNode.y + defNode.y) / 2;

  const netVal = Math.abs(uiState.net_strength);
  const textStr = `${netVal}`;

  const styles = getComputedStyle(document.body);
  const britishRed = styles.getPropertyValue('--british-red').trim();
  const mysoreGreen = styles.getPropertyValue('--mysore-green').trim();

  const battleColor = netVal > 0 ? britishRed : mysoreGreen;

  const g = document.createElementNS(SVG_NS, 'g');
  g.setAttribute('transform', `translate(${mx},${my})`);
  g.setAttribute('class', 'battle-map-marker'); 

  const icon = document.createElementNS(SVG_NS, 'text');
  icon.setAttribute('text-anchor', 'middle');
  icon.setAttribute('dominant-baseline', 'central'); // Precise vertical centering
  icon.setAttribute('font-size', '42'); // Larger icon as the text is centered *on* it
  // Apply minor drop shadow filter defined in svg defs for overall pop
  icon.setAttribute('filter', 'url(#nshadow)'); 
  icon.textContent = '⚔️';
  g.appendChild(icon);

  const makeText = (isHalo) => {
    const t = document.createElementNS(SVG_NS, 'text');
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('dominant-baseline', 'central'); // Align with icon center
    t.setAttribute('font-family', 'Cinzel, serif'); // Keeping historic font
    t.setAttribute('font-size', '28');
    t.setAttribute('font-weight', '700');

    if (isHalo) {
      t.setAttribute('stroke', '#1a1208'); 
      t.setAttribute('stroke-width', '3');
      t.setAttribute('fill', 'none');
    } else {
      t.setAttribute('fill', battleColor); 
    }
    t.textContent = textStr;
    return t;
  };

  // Add halo first (bottom), then filled text (top)
  g.appendChild(makeText(true)); 
  g.appendChild(makeText(false)); 

  layer.appendChild(g);
}

function syncUIStateOnLoad() {
  // 1. Sync MCTS Simulations
  const simsSlider = document.getElementById('mcts-sims-slider');
  if (simsSlider && typeof handleSimsSliderInput === 'function') {
    handleSimsSliderInput(simsSlider.value);
  }

  // 2. Sync Eval Toggle
  const evalToggle = document.getElementById('eval-toggle-checkbox');
  if (evalToggle && typeof handleEvalToggle === 'function') {
    handleEvalToggle(evalToggle.checked);
  }

  // 3. Sync Debug Toggle
  const debugToggle = document.getElementById('debug-toggle-checkbox');
  if (debugToggle && typeof handleDebugToggle === 'function') {
    handleDebugToggle(debugToggle.checked);
  }

  // 4. Sync Player Side
  const sideSelect = document.getElementById('human-side-select');
  if (sideSelect && typeof handleHumanSideChange === 'function') {
    handleHumanSideChange(sideSelect.value);
  }

  // 5. Sync Game Mode
  const modeSelect = document.getElementById('game-mode-select');
  if (modeSelect && typeof handleGameModeChange === 'function') {
    handleGameModeChange(modeSelect.value);
  }
}

// ==========================================================================
// 10. DYNAMIC ZERO-LETTERBOX BOARD AUTO-SIZER
// Strictly locks the map to 760:880 aspect ratio within available section space.
// Completely eliminates letterboxing and empty blue boxes on all screen sizes.
// ==========================================================================
function adjustBoardDimensions() {
  const boardSection = document.getElementById('board-section');
  const boardCard = document.getElementById('board-card');
  if (!boardSection || !boardCard) return;

  const evalPanel = document.getElementById('eval-panel');
  const debugConsole = document.getElementById('debug-move-console');

  let reservedHeight = 0;
  if (evalPanel && !evalPanel.classList.contains('hidden')) {
    reservedHeight += evalPanel.offsetHeight + 6;
  }
  if (debugConsole && !debugConsole.classList.contains('hidden')) {
    reservedHeight += debugConsole.offsetHeight + 6;
  }

  const availWidth = boardSection.clientWidth;
  const availHeight = Math.max(100, boardSection.clientHeight - reservedHeight);
  if (availWidth <= 0 || availHeight <= 0) return;

  const aspect = 760 / 880;

  let targetWidth, targetHeight;
  if (availWidth / availHeight > aspect) {
    // Section is wider than map -> height is the limiting constraint
    targetHeight = availHeight;
    targetWidth = targetHeight * aspect;
  } else {
    // Section is taller than map -> width is the limiting constraint
    targetWidth = availWidth;
    targetHeight = targetWidth / aspect;
  }

  boardCard.style.width = `${Math.floor(targetWidth)}px`;
  boardCard.style.height = `${Math.floor(targetHeight)}px`;
}

function initBoardResponsiveAutoSizer() {
  adjustBoardDimensions();
  const boardSection = document.getElementById('board-section');
  if (boardSection && window.ResizeObserver) {
    const observer = new ResizeObserver(() => {
      adjustBoardDimensions();
    });
    observer.observe(boardSection);
  }
  window.addEventListener('resize', adjustBoardDimensions);
  window.addEventListener('orientationchange', adjustBoardDimensions);
}

// Start game client on load
document.addEventListener('DOMContentLoaded', () => {
  syncUIStateOnLoad();
  initGame(); 
  initBoardResponsiveAutoSizer();
  if (!isTutorialDismissed()) {
    openTutorialDrawer();
  }
});