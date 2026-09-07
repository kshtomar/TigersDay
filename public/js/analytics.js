/**
 * Tiger's Day – Tactical Evaluation & Territory Heatmap Analytics Engine
 * Provides:
 * - Chess.com-style Move Classification (Brilliant, Best, Inaccuracy, Mistake, Blunder)
 * - Dynamic 25-Node Military Influence & Control Heatmap Calculation
 * - Interactive Evaluation Graph / Sparkline Generator
 */

(function(global) {
  'use strict';

  const BADGE_TYPES = {
    BRILLIANT: { label: 'Brilliant', glyph: '!!', color: '#10b981', cssClass: 'badge-brilliant' },
    BEST: { label: 'Best', glyph: '★', color: '#38bdf8', cssClass: 'badge-best' },
    EXCELLENT: { label: 'Excellent', glyph: '✓', color: '#6ee7b7', cssClass: 'badge-excellent' },
    GOOD: { label: 'Good', glyph: '👍', color: '#94a3b8', cssClass: 'badge-good' },
    INACCURACY: { label: 'Inaccuracy', glyph: '?!', color: '#facc15', cssClass: 'badge-inaccuracy' },
    MISTAKE: { label: 'Mistake', glyph: '?', color: '#fb923c', cssClass: 'badge-mistake' },
    BLUNDER: { label: 'Blunder', glyph: '??', color: '#ef4444', cssClass: 'badge-blunder' }
  };

  /**
   * Classifies a played move based on evaluation differential.
   * @param {number} prevEval - Evaluation before move (-1.0 to 1.0, from to_move perspective)
   * @param {number} postEval - Evaluation after move (-1.0 to 1.0, from same player perspective)
   * @param {boolean} isTopEngineMove - Whether this move matched the top MCTS engine recommendation
   * @param {boolean} isSacrifice - Whether the move sacrificed tactical material
   */
  function classifyMove(prevEval, postEval, isTopEngineMove = false, isSacrifice = false) {
    const delta = postEval - prevEval; // Positive = improved position, Negative = dropped eval

    if (isTopEngineMove && isSacrifice && delta >= 0.0) {
      return BADGE_TYPES.BRILLIANT;
    }
    if (isTopEngineMove || delta >= -0.04) {
      return BADGE_TYPES.BEST;
    }
    if (delta >= -0.12) {
      return BADGE_TYPES.EXCELLENT;
    }
    if (delta >= -0.22) {
      return BADGE_TYPES.GOOD;
    }
    if (delta >= -0.42) {
      return BADGE_TYPES.INACCURACY;
    }
    if (delta >= -0.65) {
      return BADGE_TYPES.MISTAKE;
    }
    return BADGE_TYPES.BLUNDER;
  }

  /**
   * Computes territory military control & threat projection across all 25 nodes.
   * +1.0 = Absolute British Dominance
   * -1.0 = Absolute Mysore Dominance
   *  0.0 = Contested / Neutral Buffer Zone
   */
  function computeTerritoryInfluence(gameState, adjacencies) {
    const numNodes = 25;
    const NODES_OFFSET = 12;
    const influence = new Float32Array(numNodes);

    if (!gameState || !gameState.vector) return influence;

    // Direct occupation weighting (+1.0 for British active, +0.75 for British tired, -1.0 for Mysore fort)
    for (let i = 0; i < numNodes; i++) {
      const idx = NODES_OFFSET + 3 * i;
      if (gameState.vector[idx]) {
        influence[i] += 1.0; // British fresh
      } else if (gameState.vector[idx + 1]) {
        influence[i] += 0.75; // British tired
      } else if (gameState.vector[idx + 2]) {
        influence[i] -= 1.0; // Mysore fort
      }
    }

    // Projected pressure onto adjacent nodes (0.25 per adjacent allied unit)
    if (adjacencies && Array.isArray(adjacencies)) {
      for (const [u, v] of adjacencies) {
        if (typeof u === 'number' && typeof v === 'number') {
          const uIdx = NODES_OFFSET + 3 * u;
          const vIdx = NODES_OFFSET + 3 * v;
          if (gameState.vector[uIdx] || gameState.vector[uIdx + 1]) influence[v] += 0.25;
          if (gameState.vector[uIdx + 2]) influence[v] -= 0.25;
          if (gameState.vector[vIdx] || gameState.vector[vIdx + 1]) influence[u] += 0.25;
          if (gameState.vector[vIdx + 2]) influence[u] -= 0.25;
        }
      }
    }

    // Clamp influence to [-1.0, 1.0]
    for (let i = 0; i < numNodes; i++) {
      influence[i] = Math.max(-1.0, Math.min(1.0, influence[i]));
    }

    return influence;
  }

  /**
   * Generates an SVG polyline evaluation trend chart.
   */
  function generateEvalSparklineSvg(evalHistory, width = 200, height = 40) {
    if (!evalHistory || evalHistory.length < 2) {
      return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><line x1="0" y1="${height / 2}" x2="${width}" y2="${height / 2}" stroke="#64748b" stroke-dasharray="2,2"/></svg>`;
    }

    const n = evalHistory.length;
    const points = [];
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * width;
      // map eval (-1.0 to 1.0) to y (height to 0)
      const ev = Math.max(-1.0, Math.min(1.0, evalHistory[i]));
      const y = height / 2 - (ev * (height / 2 - 4));
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }

    const midY = height / 2;
    return `
      <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" class="eval-sparkline">
        <line x1="0" y1="${midY}" x2="${width}" y2="${midY}" stroke="#334155" stroke-width="1" stroke-dasharray="3,3"/>
        <polyline fill="none" stroke="#eab308" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="${points.join(' ')}"/>
      </svg>
    `.trim();
  }

  const TDAnalytics = {
    BADGE_TYPES,
    classifyMove,
    computeTerritoryInfluence,
    generateEvalSparklineSvg
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDAnalytics;
  } else {
    global.TDAnalytics = TDAnalytics;
  }
})(typeof window !== 'undefined' ? window : this);
