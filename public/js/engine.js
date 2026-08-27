/**
 * Tiger's Day – Rule Engine & State Updater (JavaScript Port)
 * 1:1 match with Python game/engine.py, game/updater.py, and game/replay.py
 */

(function(global) {
  'use strict';

  const {
    EDGE_SOURCES, EDGE_DESTS, INDEX_MAP, CARD_VALUE,
    EDGES, NODES, CARDS, TURNS, NO_UNIT, ADJACENCY_MATRIX,
    KEYS, COASTAL, COASTAL_INDICES, MOVE_SPACE,
    WHO_TO_MOVE, MYSORE_CARDS, BRITISH_CARDS,
    MOVE_VECTOR_LENGTH, CARDS_ABBREV, NODES_ABBREV
  } = global.TDConstants || require('./state.js').TDConstants;

  const GameState = global.GameState || require('./state.js').GameState;

  // =========================================================================
  // 1. LEGAL MOVES CALCULATOR (Phase 0, 1, 2)
  // =========================================================================
  function getLegalMoves(state) {
    const fresh = state.fresh_armies;
    const tired = state.tired_armies;
    const forts = state.forts;
    const empty = state.empty;
    const mCards = state.mysore_cards;
    const bCards = state.british_cards;

    const legalDest = new Uint8Array(NODES);
    for (let i = 0; i < NODES; i++) {
      legalDest[i] = (empty[i] || forts[i]) ? 1 : 0;
    }
    if (state.is_battle && state.defender !== NO_UNIT) {
      legalDest[state.defender] = 0;
    }

    const mask = new Uint8Array(MOVE_VECTOR_LENGTH);
    let offset = 0;

    // -----------------------------------------------------------------------
    // Phase 0: British Move (to_move == 0)
    // -----------------------------------------------------------------------
    if (state.to_move === 0) {
      // 1. Move (EDGES)
      for (let i = 0; i < EDGES; i++) {
        const src = EDGE_SOURCES[i];
        const dest = EDGE_DESTS[i];
        if (fresh[src] && legalDest[dest]) {
          mask[offset + i] = 1;
        }
      }
      offset += EDGES;

      // 2. Tire (NODES)
      for (let i = 0; i < NODES; i++) {
        if (fresh[i]) {
          mask[offset + i] = 1;
        }
      }
      offset += NODES;
    } else {
      offset += EDGES + NODES; // Skip Phase 0
    }

    // -----------------------------------------------------------------------
    // Phase 1: Mysore Card (to_move == 1)
    // -----------------------------------------------------------------------
    if (state.to_move === 1) {
      // 1. Sepoy Mutiny: remove army not in Key City
      if (mCards[1]) {
        for (let i = 0; i < NODES; i++) {
          if ((fresh[i] || tired[i]) && !KEYS[i]) {
            mask[offset + i] = 1;
          }
        }
      }
      offset += NODES;

      // 2. French Alliance: deploy fort adjacent to another fort into empty
      if (mCards[2]) {
        for (let i = 0; i < NODES; i++) {
          if (empty[i]) {
            let adjToFort = false;
            for (let f = 0; f < NODES; f++) {
              if (forts[f] && (ADJACENCY_MATRIX[f][i] || ADJACENCY_MATRIX[i][f])) {
                adjToFort = true;
                break;
              }
            }
            if (adjToFort) mask[offset + i] = 1;
          }
        }
      }
      offset += NODES;

      // 3. Monsoon: flip fresh army to tired
      if (mCards[3]) {
        for (let i = 0; i < NODES; i++) {
          if (fresh[i]) mask[offset + i] = 1;
        }
      }
      offset += NODES;

      // 4. Cavalry Raid: force British discard
      if (mCards[4]) {
        mask[offset] = 1;
      }
      offset += 1;

      // 5. Sea Trade: move fort from coast to any empty node (NODES * COASTAL_INDICES.length)
      if (mCards[5]) {
        const numCoasts = COASTAL_INDICES.length;
        for (let n = 0; n < NODES; n++) {
          if (empty[n]) {
            for (let c = 0; c < numCoasts; c++) {
              const coastIdx = COASTAL_INDICES[c];
              if (forts[coastIdx]) {
                mask[offset + n * numCoasts + c] = 1;
              }
            }
          }
        }
      }
      offset += NODES * COASTAL_INDICES.length;

      // 6. Mysore Power: commit card in battle (CARDS)
      if (state.is_battle) {
        for (let i = 0; i < CARDS; i++) {
          if (mCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 7. Draw Iron Rockets (trade card 0 for cards 1..5)
      if (mCards[0]) {
        for (let i = 1; i < CARDS; i++) {
          if (!mCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 8. Draw Sepoy Mutiny (trade card 1 for cards 3..5)
      if (mCards[1]) {
        for (let i = 3; i < CARDS; i++) {
          if (!mCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 9. Draw French Alliance (trade card 2 for cards 3..5)
      if (mCards[2]) {
        for (let i = 3; i < CARDS; i++) {
          if (!mCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 10. Pass Mysore
      mask[offset] = 1;
      offset += 1;
    } else {
      offset += NODES * 3 + 1 + (NODES * COASTAL_INDICES.length) + CARDS * 4 + 1;
    }

    // -----------------------------------------------------------------------
    // Phase 2: British Card (to_move == 2)
    // -----------------------------------------------------------------------
    if (state.to_move === 2) {
      // 1. Highlanders: deploy fresh army on empty coast
      if (bCards[1]) {
        for (let i = 0; i < NODES; i++) {
          if (empty[i] && COASTAL[i]) mask[offset + i] = 1;
        }
      }
      offset += NODES;

      // 2. Royal Navy: move army to legal coast destination (NODES * COASTAL_INDICES.length)
      if (bCards[2]) {
        const numCoasts = COASTAL_INDICES.length;
        for (let n = 0; n < NODES; n++) {
          if (fresh[n] || tired[n]) {
            for (let c = 0; c < numCoasts; c++) {
              const coastIdx = COASTAL_INDICES[c];
              if (legalDest[coastIdx]) {
                mask[offset + n * numCoasts + c] = 1;
              }
            }
          }
        }
      }
      offset += NODES * COASTAL_INDICES.length;

      // 3. Divide and Rule: move fort not in Key City to empty along edge (EDGES)
      if (bCards[3]) {
        for (let i = 0; i < EDGES; i++) {
          const src = EDGE_SOURCES[i];
          const dest = EDGE_DESTS[i];
          if (forts[src] && !KEYS[src] && empty[dest]) {
            mask[offset + i] = 1;
          }
        }
      }
      offset += EDGES;

      // 4. Force March: move tired army along edge to legal dest (EDGES)
      if (bCards[4]) {
        for (let i = 0; i < EDGES; i++) {
          const src = EDGE_SOURCES[i];
          const dest = EDGE_DESTS[i];
          if (tired[src] && legalDest[dest]) {
            mask[offset + i] = 1;
          }
        }
      }
      offset += EDGES;

      // 5. Princely States: deploy tired army in empty Key City (NODES)
      if (bCards[5]) {
        for (let i = 0; i < NODES; i++) {
          if (empty[i] && KEYS[i]) mask[offset + i] = 1;
        }
      }
      offset += NODES;

      // 6. British Power: commit card in battle (CARDS)
      if (state.is_battle) {
        for (let i = 0; i < CARDS; i++) {
          if (bCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 7. Draw Wall Breach (trade card 0 for cards 1..5)
      if (bCards[0]) {
        for (let i = 1; i < CARDS; i++) {
          if (!bCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 8. Draw Highlanders (trade card 1 for cards 3..5)
      if (bCards[1]) {
        for (let i = 3; i < CARDS; i++) {
          if (!bCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 9. Draw Royal Navy (trade card 2 for cards 3..5)
      if (bCards[2]) {
        for (let i = 3; i < CARDS; i++) {
          if (!bCards[i]) mask[offset + i] = 1;
        }
      }
      offset += CARDS;

      // 10. Pass British
      mask[offset] = 1;
      offset += 1;
    }

    return mask;
  }

  // =========================================================================
  // 2. LEGAL MOVES DICTIONARY
  // =========================================================================
  function legalMovesDict(mask) {
    const list = [];
    let offset = 0;

    for (const [name, size, moveType] of MOVE_SPACE) {
      for (let idx = 0; idx < size; idx++) {
        if (mask[offset + idx]) {
          const moveIdx = offset + idx;
          let desc = "Undefined";

          if (moveType === "node") {
            desc = INDEX_MAP[idx];
          } else if (moveType === "edge") {
            const srcName = INDEX_MAP[EDGE_SOURCES[idx]];
            const destName = INDEX_MAP[EDGE_DESTS[idx]];
            desc = `${srcName} -> ${destName}`;
          } else if (moveType === "bcard") {
            desc = BRITISH_CARDS[idx];
          } else if (moveType === "mcard") {
            desc = MYSORE_CARDS[idx];
          } else if (moveType === "blank") {
            desc = "-";
          } else if (moveType === "coastal") {
            const numCoasts = COASTAL_INDICES.length;
            const node = INDEX_MAP[Math.floor(idx / numCoasts)];
            const coast = INDEX_MAP[COASTAL_INDICES[idx % numCoasts]];
            if (name === "Royal Navy") {
              desc = `${node} -> ${coast}`;
            } else if (name === "Sea Trade") {
              desc = `${coast} -> ${node}`;
            }
          }

          list.push({
            idx: moveIdx,
            type: name,
            desc: desc
          });
        }
      }
      offset += size;
    }

    return list;
  }

  // =========================================================================
  // 3. BATTLE & COMBAT RESOLUTION
  // =========================================================================
  function calculateBattleStrength(state) {
    if (!state.is_battle || state.defender === NO_UNIT) {
      return {
        attackerArmies: 0,
        defenderForts: 0,
        mysoreCardStrength: 0,
        netStrength: 0,
        attackerWinning: false
      };
    }

    const def = state.defender;
    let attackerArmies = 0;
    let defenderForts = 0;

    const fresh = state.fresh_armies;
    const tired = state.tired_armies;
    const forts = state.forts;

    for (let i = 0; i < NODES; i++) {
      if (ADJACENCY_MATRIX[def][i]) {
        if (fresh[i] || tired[i]) {
          attackerArmies++;
        }
        if (forts[i]) {
          defenderForts++;
        }
      }
    }

    const mysoreCardStrength = state.card_strength || 0; // Committed by Mysore in Phase 1 (0..3)
    const netStrength = attackerArmies - defenderForts - mysoreCardStrength;
    const attackerWinning = netStrength > 0;

    return {
      attackerArmies,
      defenderForts,
      mysoreCardStrength,
      netStrength,
      attackerWinning
    };
  }

  function isBattleWon(state, defender, netCardStrength) {
    let attackerStrength = 0;
    let defenderStrength = 0;

    const fresh = state.fresh_armies;
    const tired = state.tired_armies;
    const forts = state.forts;

    for (let i = 0; i < NODES; i++) {
      if (ADJACENCY_MATRIX[defender][i]) {
        if (fresh[i] || tired[i]) {
          attackerStrength++;
        }
        if (forts[i]) {
          defenderStrength++;
        }
      }
    }

    return (attackerStrength + netCardStrength) > defenderStrength;
  }

  function resolveBattles(state, attacker, defender, netCardStrength) {
    let battle1 = false;
    let battle2 = false;

    if (state.attacker !== NO_UNIT) {
      battle1 = isBattleWon(state, state.defender, netCardStrength);
      if (battle1) {
        state.mluck += 1;
      } else {
        state.bluck += 1;
      }
    }

    if (attacker !== NO_UNIT) {
      battle2 = isBattleWon(state, defender, 0);
      if (battle2) {
        state.mluck += 1;
      } else {
        state.bluck += 1;
      }
    }

    if (battle1) {
      state.set_node_tired_army(state.defender);
      state.set_node_empty(state.attacker);
    }

    if (battle2) {
      const isFresh = state.fresh_armies[attacker];
      state.set_node_empty(attacker);
      if (isFresh) {
        state.set_node_fresh_army(defender);
      } else {
        state.set_node_tired_army(defender);
      }
    }

    state.clear_battle();
    return state;
  }

  // =========================================================================
  // 4. GAME STATE ADVANCEMENT (get_next_state)
  // =========================================================================
  function getNextState(state, move) {
    const nextState = state.copy();
    let offset = 0;

    for (const [name, size, moveType] of MOVE_SPACE) {
      if (move >= offset && move < offset + size) {
        const idx = move - offset;

        if (moveType === "node") {
          if (name === "Tire") {
            nextState.set_node_tired_army(idx);
          } else if (name === "Sepoy Mutiny") {
            nextState.mysore_cards[1] = 0;
            if (state.attacker === idx) nextState.clear_battle();
            nextState.set_node_empty(idx);
          } else if (name === "French Alliance") {
            nextState.mysore_cards[2] = 0;
            nextState.set_node_fort(idx);
          } else if (name === "Monsoon") {
            nextState.mysore_cards[3] = 0;
            nextState.set_node_tired_army(idx);
          } else if (name === "Highlanders") {
            nextState.british_cards[1] = 0;
            nextState.set_node_fresh_army(idx);
          } else if (name === "Princely States") {
            nextState.british_cards[5] = 0;
            nextState.set_node_tired_army(idx);
          }
        } else if (moveType === "edge") {
          const src = EDGE_SOURCES[idx];
          const dest = EDGE_DESTS[idx];
          const isFortDefending = Boolean(state.forts[dest]);

          if (name === "Move") {
            if (isFortDefending) {
              nextState.set_node_tired_army(src);
              nextState.attacker = src;
              nextState.defender = dest;
              nextState.card_strength = 0;
            } else {
              nextState.set_node_empty(src);
              nextState.set_node_tired_army(dest);
            }
          } else if (name === "Divide and Rule") {
            nextState.british_cards[3] = 0;
            if (state.defender === src) {
              nextState.set_node_tired_army(src);
              nextState.set_node_empty(state.attacker);
              nextState.clear_battle();
            } else {
              nextState.set_node_empty(src);
            }
            nextState.set_node_fort(dest);
          } else if (name === "Force March") {
            nextState.british_cards[4] = 0;
            if (state.attacker === src) nextState.clear_battle();
            if (isFortDefending) {
              resolveBattles(nextState, src, dest, -state.card_strength);
            } else {
              nextState.set_node_empty(src);
              nextState.set_node_tired_army(dest);
            }
          }
        } else if (moveType === "bcard") {
          if (name === "British Power") {
            nextState.british_cards[idx] = 0;
            resolveBattles(nextState, NO_UNIT, NO_UNIT, CARD_VALUE[idx] - state.card_strength);
          } else if (name === "Draw Wall Breach") {
            nextState.british_cards[0] = 0;
            nextState.british_cards[idx] = 1;
          } else if (name === "Draw Highlanders") {
            nextState.british_cards[1] = 0;
            nextState.british_cards[idx] = 1;
          } else if (name === "Draw Royal Navy") {
            nextState.british_cards[2] = 0;
            nextState.british_cards[idx] = 1;
          }
        } else if (moveType === "mcard") {
          if (name === "Mysore Power") {
            nextState.mysore_cards[idx] = 0;
            nextState.card_strength = CARD_VALUE[idx];
          } else if (name === "Draw Iron Rockets") {
            nextState.mysore_cards[0] = 0;
            nextState.mysore_cards[idx] = 1;
          } else if (name === "Draw Sepoy Mutiny") {
            nextState.mysore_cards[1] = 0;
            nextState.mysore_cards[idx] = 1;
          } else if (name === "Draw French Alliance") {
            nextState.mysore_cards[2] = 0;
            nextState.mysore_cards[idx] = 1;
          }
        } else if (moveType === "blank") {
          if (name === "Cavalry Raid") {
            nextState.mysore_cards[4] = 0;
            nextState.bluck += 1;
          } else if (name === "Pass Mysore" || name === "Pass British") {
            // Pass
          }
        } else if (moveType === "coastal") {
          const numCoasts = COASTAL_INDICES.length;
          const node = Math.floor(idx / numCoasts);
          const coast = COASTAL_INDICES[idx % numCoasts];

          if (name === "Sea Trade") {
            nextState.mysore_cards[5] = 0;
            if (state.defender === coast) {
              nextState.set_node_tired_army(coast);
              nextState.set_node_empty(state.attacker);
              nextState.clear_battle();
            } else {
              nextState.set_node_empty(coast);
            }
            nextState.set_node_fort(node);
          } else if (name === "Royal Navy") {
            nextState.british_cards[2] = 0;
            if (state.attacker === node) nextState.clear_battle();
            const isFortDefending = Boolean(state.forts[coast]);
            if (isFortDefending) {
              resolveBattles(nextState, node, coast, -state.card_strength);
            } else {
              const isFresh = state.fresh_armies[node];
              nextState.set_node_empty(node);
              if (isFresh) {
                nextState.set_node_fresh_army(coast);
              } else {
                nextState.set_node_tired_army(coast);
              }
            }
          }
        }
        break;
      }
      offset += size;
    }

    // End of Phase 2 check
    if (state.to_move === 2) {
      resolveBattles(nextState, NO_UNIT, NO_UNIT, -state.card_strength);
      let hasFresh = false;
      const freshArr = nextState.fresh_armies;
      for (let i = 0; i < NODES; i++) {
        if (freshArr[i]) {
          hasFresh = true;
          break;
        }
      }
      if (state.turn !== 4 && !hasFresh) {
        nextState.turn_refresh();
      }
    }

    nextState.to_move = (state.to_move + 1) % 3;
    return nextState;
  }

  // =========================================================================
  // 5. LUCK OUTCOMES & RESOLUTION
  // =========================================================================
  function getLuckOutcomes(state) {
    const outcomes = [];

    if (state.bluck > 0) {
      for (let i = 0; i < CARDS; i++) {
        if (state.british_cards[i]) {
          const luckState = state.copy();
          luckState.british_cards[i] = 0;
          luckState.bluck -= 1;
          outcomes.push(luckState);
        }
      }
      if (outcomes.length === 0) {
        const luckState = state.copy();
        luckState.bluck -= 1;
        return [luckState];
      }
      return outcomes;
    }

    if (state.mluck > 0) {
      for (let i = 0; i < CARDS; i++) {
        if (state.mysore_cards[i]) {
          const luckState = state.copy();
          luckState.mysore_cards[i] = 0;
          luckState.mluck -= 1;
          outcomes.push(luckState);
        }
      }
      if (outcomes.length === 0) {
        const luckState = state.copy();
        luckState.mluck -= 1;
        return [luckState];
      }
      return outcomes;
    }

    return [state];
  }

  function resolveLuck(state, rng = Math.random) {
    let current = state;
    while (current.is_luck) {
      const outcomes = getLuckOutcomes(current);
      const idx = Math.floor(rng() * outcomes.length);
      current = outcomes[idx];
    }
    return current;
  }

  function resolveLuckWithTrajectory(state, rng = Math.random) {
    let current = state;
    const trajectory = [];
    while (current.is_luck) {
      const outcomes = getLuckOutcomes(current);
      const idx = Math.floor(rng() * outcomes.length);
      trajectory.push(idx);
      current = outcomes[idx];
    }
    return { finalState: current, trajectory };
  }

  function applyLuckTrajectory(state, trajectory = []) {
    let current = state;
    for (const idx of trajectory) {
      if (!current.is_luck) break;
      const outcomes = getLuckOutcomes(current);
      const chosenIdx = Math.min(Math.max(0, idx), outcomes.length - 1);
      current = outcomes[chosenIdx];
    }
    while (current.is_luck) {
      const outcomes = getLuckOutcomes(current);
      const idx = Math.floor(Math.random() * outcomes.length);
      current = outcomes[idx];
    }
    return current;
  }

  // =========================================================================
  // 6. GAME WINNER EVALUATION
  // =========================================================================
  function getStateWinner(state) {
    const fresh = state.fresh_armies;
    const tired = state.tired_armies;

    let keyArmiesCount = 0;
    for (let i = 0; i < 5; i++) { // First 5 nodes are Keys
      if (fresh[i] || tired[i]) keyArmiesCount++;
    }

    if (keyArmiesCount === 5) {
      return 1; // British Victory
    }

    let hasFresh = false;
    for (let i = 0; i < NODES; i++) {
      if (fresh[i]) {
        hasFresh = true;
        break;
      }
    }

    if (state.turn === 4 && !hasFresh && state.to_move === 0) {
      return -1; // Mysore Victory
    }

    return 0; // Game in progress
  }

  // =========================================================================
  // 7. ALGEBRAIC NOTATION & REPLAY PARSER
  // =========================================================================
  function notate(state, move) {
    let offset = 0;
    let gap = ">";
    let moveString = "";

    for (const [name, size, moveType] of MOVE_SPACE) {
      if (move >= offset && move < offset + size) {
        const idx = move - offset;
        if (CARDS_ABBREV[name]) {
          moveString += CARDS_ABBREV[name] + ":";
        }

        if (moveType === "node") {
          moveString += NODES_ABBREV[idx];
          return moveString;
        } else if (moveType === "edge") {
          const dest = EDGE_DESTS[idx];
          if (state.forts[dest]) {
            if (name === "Move" || name === "Force March") gap = "x";
          }
          const srcName = NODES_ABBREV[EDGE_SOURCES[idx]];
          const destName = NODES_ABBREV[dest];
          return moveString + srcName + gap + destName;
        } else if (moveType === "bcard") {
          let cardName = CARDS_ABBREV[BRITISH_CARDS[idx]];
          if (name === "British Power") {
            moveString += cardName + ":";
            cardName = "x";
          }
          return moveString + cardName;
        } else if (moveType === "mcard") {
          let cardName = CARDS_ABBREV[MYSORE_CARDS[idx]];
          if (name === "Mysore Power") {
            moveString += cardName + ":";
            cardName = "x";
          }
          return moveString + cardName;
        } else if (moveType === "blank") {
          return name === "Cavalry Raid" ? moveString : "pass";
        } else if (moveType === "coastal") {
          const numCoasts = COASTAL_INDICES.length;
          const node = NODES_ABBREV[Math.floor(idx / numCoasts)];
          const coastIdx = COASTAL_INDICES[idx % numCoasts];
          const coast = NODES_ABBREV[coastIdx];

          if (name === "Royal Navy") {
            if (state.forts[coastIdx]) gap = "x";
            return moveString + node + gap + coast;
          } else if (name === "Sea Trade") {
            return moveString + coast + gap + node;
          }
        }
      }
      offset += size;
    }
    return "none";
  }

  function generateGameData(state, matchMode = "human_vs_ai", humanSide = "british") {
    const nodes = [];
    const fresh = state.fresh_armies;
    const tired = state.tired_armies;
    const forts = state.forts;

    for (let i = 0; i < NODES; i++) {
      let aType = "empty";
      if (fresh[i]) aType = "fresh";
      else if (tired[i]) aType = "tired";
      else if (forts[i]) aType = "fort";

      nodes.push({
        name: INDEX_MAP[i],
        armyType: aType
      });
    }

    const mask = getLegalMoves(state);
    const moves = legalMovesDict(mask);
    const battleInfo = calculateBattleStrength(state);

    return {
      state_str: state.toString(),
      winner: getStateWinner(state),
      moves: moves,
      match_mode: matchMode,
      human_side: humanSide,
      ui_state: {
        british_cards: Array.from(state.british_cards).map(Boolean),
        mysore_cards: Array.from(state.mysore_cards).map(Boolean),
        turn: state.turn,
        who_to_move: WHO_TO_MOVE[state.to_move],
        attacker: state.attacker !== NO_UNIT ? INDEX_MAP[state.attacker] : "None",
        defender: state.defender !== NO_UNIT ? INDEX_MAP[state.defender] : "None",
        card_strength: state.card_strength,
        battle_info: battleInfo,
        net_strength: battleInfo.netStrength,
        attacker_armies: battleInfo.attackerArmies,
        defender_forts: battleInfo.defenderForts,
        nodes: nodes
      }
    };
  }

  const TDEngine = {
    getLegalMoves,
    legalMovesDict,
    calculateBattleStrength,
    isBattleWon,
    resolveBattles,
    getNextState,
    getLuckOutcomes,
    resolveLuck,
    resolveLuckWithTrajectory,
    applyLuckTrajectory,
    getStateWinner,
    notate,
    generateGameData
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDEngine;
  } else {
    global.TDEngine = TDEngine;
    Object.assign(global, TDEngine);
  }
})(typeof window !== 'undefined' ? window : this);

