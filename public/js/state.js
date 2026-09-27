/**
 * Tiger's Day – State & Constants Engine (JavaScript Port)
 * 1:1 match with Python game/state.py & game/constants.py
 */

(function(global) {
  'use strict';

  const EDGE_SOURCES = [
    // 0: Bombay | 1: Hyderabad | 2: Madras | 3: Seringapatam | 4: Coimbatore
    0, 0, 0,    1, 1, 1, 1,   2, 2, 2, 2,   3, 3, 3, 3,   4, 4, 4, 4,
    // 5: Satara | 6: Raichur | 7: Masulipatam | 8: Goa | 9: Darwar
    5, 5, 5, 5, 6, 6, 6, 6,   7, 7,         8, 8, 8,      9, 9, 9,
    // 10: Anantapur | 11: Chitaldoorg | 12: Mangalore | 13: Bangalore | 14: Vellore
    10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14,
    // 15: Mahé | 16: Pondicherry | 17: Erode | 18: Trichy | 19: Alwaye
    15, 15,     16, 16, 16,   17, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
    // 20: Dindigul | 21: Ramnad | 22: Travancore | 23: Ceylon | 24: Poona
    20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24
  ];

  const EDGE_DESTS = [
    // 0: Bombay | 1: Hyderabad | 2: Madras | 3: Seringapatam | 4: Coimbatore
    5, 8, 24,   6, 7, 10, 24, 7, 10, 14, 16, 12, 13, 15, 17, 15, 17, 19, 20,
    // 5: Satara | 6: Raichur | 7: Masulipatam | 8: Goa | 9: Darwar
    0, 6, 9, 24, 1, 5, 10, 11, 1, 2,        0, 9, 12,     5, 8, 11,
    // 10: Anantapur | 11: Chitaldoorg | 12: Mangalore | 13: Bangalore | 14: Vellore
    1, 2, 6, 14, 6, 9, 12, 13, 3, 8, 11,    3, 11, 14,    2, 10, 13, 17,
    // 15: Mahé | 16: Pondicherry | 17: Erode | 18: Trichy | 19: Alwaye
    3, 4,       2, 17, 18,    3, 4, 14, 16, 18, 16, 17, 20, 23, 4, 21, 22,
    // 20: Dindigul | 21: Ramnad | 22: Travancore | 23: Ceylon | 24: Poona
    4, 18, 21,  19, 20, 22, 23, 19, 21, 23, 18, 21, 22,   0, 1, 5
  ];

  const INDEX_MAP = {
    0: "Bombay", 1: "Hyderabad", 2: "Madras", 3: "Seringapatam", 4: "Coimbatore",
    5: "Satara", 6: "Raichur", 7: "Masulipatam", 8: "Goa", 9: "Darwar",
    10: "Anantapur", 11: "Chitaldoorg", 12: "Mangalore", 13: "Bangalore", 14: "Vellore",
    15: "Mahé", 16: "Pondicherry", 17: "Erode", 18: "Trichy", 19: "Alwaye",
    20: "Dindigul", 21: "Ramnad", 22: "Travancore", 23: "Ceylon", 24: "Poona"
  };

  const NODE_TO_IDX = {};
  for (const [k, v] of Object.entries(INDEX_MAP)) {
    NODE_TO_IDX[v] = parseInt(k, 10);
  }

  const CARD_VALUE = [3, 2, 2, 1, 1, 1];
  const EDGES = EDGE_SOURCES.length; // 84
  const NODES = Object.keys(INDEX_MAP).length; // 25
  const CARDS = CARD_VALUE.length; // 6
  const TURNS = 4;
  const NO_UNIT = -1;

  const ADJACENCY_MATRIX = Array.from({ length: NODES }, () => new Uint8Array(NODES));
  for (let i = 0; i < EDGES; i++) {
    ADJACENCY_MATRIX[EDGE_SOURCES[i]][EDGE_DESTS[i]] = 1;
  }

  // (Bombay, Hyderabad, Madras, Seringapatam, Coimbatore)
  const KEYS = new Uint8Array([
    1, 1, 1, 1, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0
  ]);

  // (Bombay, Madras, Masulipatam, Goa, Mangalore, Mahé, Pondicherry, Ramnad, Travancore, Ceylon)
  const COASTAL = new Uint8Array([
    1, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 1, 1, 0
  ]);

  const KEY_INDICES = [];
  for (let i = 0; i < NODES; i++) if (KEYS[i]) KEY_INDICES.push(i);

  const COASTAL_INDICES = [];
  for (let i = 0; i < NODES; i++) if (COASTAL[i]) COASTAL_INDICES.push(i);

  const MOVE_SPACE = [
    ["Move", EDGES, "edge"],
    ["Tire", NODES, "node"],
    ["Sepoy Mutiny", NODES, "node"],
    ["French Alliance", NODES, "node"],
    ["Monsoon", NODES, "node"],
    ["Cavalry Raid", 1, "blank"],
    ["Sea Trade", NODES * COASTAL_INDICES.length, "coastal"],
    ["Mysore Power", CARDS, "mcard"],
    ["Draw Iron Rockets", CARDS, "mcard"],
    ["Draw Sepoy Mutiny", CARDS, "mcard"],
    ["Draw French Alliance", CARDS, "mcard"],
    ["Pass Mysore", 1, "blank"],
    ["Highlanders", NODES, "node"],
    ["Royal Navy", NODES * COASTAL_INDICES.length, "coastal"],
    ["Divide and Rule", EDGES, "edge"],
    ["Force March", EDGES, "edge"],
    ["Princely States", NODES, "node"],
    ["British Power", CARDS, "bcard"],
    ["Draw Wall Breach", CARDS, "bcard"],
    ["Draw Highlanders", CARDS, "bcard"],
    ["Draw Royal Navy", CARDS, "bcard"],
    ["Pass British", 1, "blank"]
  ];

  const WHO_TO_MOVE = ["British Move", "Mysore Card", "British Card"];
  const MYSORE_CARDS = ["Iron Rockets", "Sepoy Mutiny", "French Alliance", "Monsoon", "Cavalry Raid", "Sea Trade"];
  const BRITISH_CARDS = ["Wall Breach", "Highlanders", "Royal Navy", "Divide and Rule", "Force March", "Princely States"];

  const GAME_VECTOR_LENGTH = NODES * 5 + CARDS * 2 + TURNS + 3 + 4; // 148

  let totalMoves = 0;
  for (const [, size] of MOVE_SPACE) totalMoves += size;
  const MOVE_VECTOR_LENGTH = totalMoves; // 953

  const CARDS_ABBREV = {
    "Iron Rockets": "IR", "Sepoy Mutiny": "SM", "French Alliance": "FA",
    "Monsoon": "MS", "Cavalry Raid": "CR", "Sea Trade": "ST",
    "Wall Breach": "WB", "Highlanders": "HL", "Royal Navy": "RN",
    "Divide and Rule": "DR", "Force March": "FM", "Princely States": "PS",
    "Draw Iron Rockets": "IR", "Draw Sepoy Mutiny": "SM", "Draw French Alliance": "FA",
    "Draw Wall Breach": "WB", "Draw Highlanders": "HL", "Draw Royal Navy": "RN"
  };

  const NODES_ABBREV = {
    0: "bom", 1: "hyd", 2: "mad", 3: "srp", 4: "cbt",
    5: "sat", 6: "rch", 7: "msp", 8: "goa", 9: "dwr",
    10: "ant", 11: "ctd", 12: "mlr", 13: "blr", 14: "vlr",
    15: "mhe", 16: "pdc", 17: "erd", 18: "tri", 19: "alw",
    20: "dng", 21: "rmd", 22: "trv", 23: "cyl", 24: "pna"
  };

  // =========================================================================
  // GameState Class
  // =========================================================================
  class GameState {
    static IDX_BRITISH_CARDS_OFFSET = 0;
    static IDX_MYSORE_CARDS_OFFSET = CARDS;
    static IDX_NODES_OFFSET = CARDS * 2; // 12
    static IDX_TURN_OFFSET = 12 + 3 * NODES; // 87
    static IDX_WHO_TO_MOVE_OFFSET = 87 + 4; // 91
    static IDX_COMBAT_STRENGTH_OFFSET = 91 + 3; // 94
    static IDX_ATTACKER_OFFSET = 94 + 4; // 98
    static IDX_DEFENDER_OFFSET = 98 + NODES; // 123

    constructor() {
      this.vector = new Uint8Array(GAME_VECTOR_LENGTH);
      this._attacker = NO_UNIT;
      this._defender = NO_UNIT;
      this._card_strength = 0;
      this._to_move = 0;
      this._turn = 1;
      this.bluck = 0;
      this.mluck = 0;

      // Cards start fully available (all 1s)
      for (let i = 0; i < CARDS; i++) {
        this.vector[GameState.IDX_BRITISH_CARDS_OFFSET + i] = 1;
        this.vector[GameState.IDX_MYSORE_CARDS_OFFSET + i] = 1;
      }
      this.turn = 1;
      this.to_move = 0;
      this.card_strength = 0;
    }

    default_setup() {
      this.set_node_fresh_army(NODE_TO_IDX["Bombay"]);
      this.set_node_fresh_army(NODE_TO_IDX["Hyderabad"]);
      this.set_node_fresh_army(NODE_TO_IDX["Madras"]);
      this.set_node_fresh_army(NODE_TO_IDX["Travancore"]);

      this.set_node_fort(NODE_TO_IDX["Darwar"]);
      this.set_node_fort(NODE_TO_IDX["Chitaldoorg"]);
      this.set_node_fort(NODE_TO_IDX["Mangalore"]);
      this.set_node_fort(NODE_TO_IDX["Bangalore"]);
      this.set_node_fort(NODE_TO_IDX["Seringapatam"]);
      this.set_node_fort(NODE_TO_IDX["Erode"]);
      this.set_node_fort(NODE_TO_IDX["Coimbatore"]);
      this.set_node_fort(NODE_TO_IDX["Mahé"]);
      this.set_node_fort(NODE_TO_IDX["Dindigul"]);

      this.turn = 1;
      this.to_move = 0;
    }

    copy() {
      const clone = new GameState();
      clone.vector.set(this.vector);
      clone._attacker = this._attacker;
      clone._defender = this._defender;
      clone._card_strength = this._card_strength;
      clone._to_move = this._to_move;
      clone._turn = this._turn;
      clone.bluck = this.bluck;
      clone.mluck = this.mluck;
      return clone;
    }

    set_node_fresh_army(node) {
      this.set_node_empty(node);
      this.vector[GameState.IDX_NODES_OFFSET + 3 * node] = 1;
    }

    set_node_tired_army(node) {
      this.set_node_empty(node);
      this.vector[GameState.IDX_NODES_OFFSET + 3 * node + 1] = 1;
    }

    set_node_fort(node) {
      this.set_node_empty(node);
      this.vector[GameState.IDX_NODES_OFFSET + 3 * node + 2] = 1;
    }

    set_node_empty(node) {
      const startIdx = GameState.IDX_NODES_OFFSET + 3 * node;
      this.vector[startIdx] = 0;
      this.vector[startIdx + 1] = 0;
      this.vector[startIdx + 2] = 0;
    }

    clear_battle() {
      this.attacker = NO_UNIT;
      this.defender = NO_UNIT;
      this.card_strength = 0;
    }

    // Dynamic arrays
    get fresh_armies() {
      const res = new Uint8Array(NODES);
      for (let i = 0; i < NODES; i++) {
        res[i] = this.vector[GameState.IDX_NODES_OFFSET + 3 * i];
      }
      return res;
    }

    get tired_armies() {
      const res = new Uint8Array(NODES);
      for (let i = 0; i < NODES; i++) {
        res[i] = this.vector[GameState.IDX_NODES_OFFSET + 3 * i + 1];
      }
      return res;
    }

    get forts() {
      const res = new Uint8Array(NODES);
      for (let i = 0; i < NODES; i++) {
        res[i] = this.vector[GameState.IDX_NODES_OFFSET + 3 * i + 2];
      }
      return res;
    }

    get empty() {
      const res = new Uint8Array(NODES);
      for (let i = 0; i < NODES; i++) {
        const has = this.vector[GameState.IDX_NODES_OFFSET + 3 * i] ||
                    this.vector[GameState.IDX_NODES_OFFSET + 3 * i + 1] ||
                    this.vector[GameState.IDX_NODES_OFFSET + 3 * i + 2];
        res[i] = has ? 0 : 1;
      }
      return res;
    }

    get mysore_cards() {
      return this.vector.subarray(GameState.IDX_MYSORE_CARDS_OFFSET, GameState.IDX_MYSORE_CARDS_OFFSET + CARDS);
    }

    get british_cards() {
      return this.vector.subarray(GameState.IDX_BRITISH_CARDS_OFFSET, GameState.IDX_BRITISH_CARDS_OFFSET + CARDS);
    }

    get to_move() {
      return this._to_move;
    }

    set to_move(val) {
      const idx = val % 3;
      this._to_move = idx;
      this.vector.fill(0, GameState.IDX_WHO_TO_MOVE_OFFSET, GameState.IDX_WHO_TO_MOVE_OFFSET + 3);
      this.vector[GameState.IDX_WHO_TO_MOVE_OFFSET + idx] = 1;
    }

    get turn() {
      return this._turn;
    }

    set turn(val) {
      this._turn = val;
      this.vector.fill(0, GameState.IDX_TURN_OFFSET, GameState.IDX_TURN_OFFSET + 4);
      if (val >= 1 && val <= 4) {
        this.vector[GameState.IDX_TURN_OFFSET + (val - 1)] = 1;
      }
    }

    get attacker() {
      return this._attacker;
    }

    set attacker(val) {
      this._attacker = val;
      this.vector.fill(0, GameState.IDX_ATTACKER_OFFSET, GameState.IDX_ATTACKER_OFFSET + NODES);
      if (val !== NO_UNIT && val >= 0 && val < NODES) {
        this.vector[GameState.IDX_ATTACKER_OFFSET + val] = 1;
      }
    }

    get defender() {
      return this._defender;
    }

    set defender(val) {
      this._defender = val;
      this.vector.fill(0, GameState.IDX_DEFENDER_OFFSET, GameState.IDX_DEFENDER_OFFSET + NODES);
      if (val !== NO_UNIT && val >= 0 && val < NODES) {
        this.vector[GameState.IDX_DEFENDER_OFFSET + val] = 1;
      }
    }

    get card_strength() {
      return this._card_strength;
    }

    set card_strength(val) {
      this._card_strength = val;
      this.vector.fill(0, GameState.IDX_COMBAT_STRENGTH_OFFSET, GameState.IDX_COMBAT_STRENGTH_OFFSET + 4);
      if (val >= 0 && val < 4) {
        this.vector[GameState.IDX_COMBAT_STRENGTH_OFFSET + val] = 1;
      }
    }

    get is_battle() {
      return this.attacker !== NO_UNIT;
    }

    get is_luck() {
      return Boolean(this.bluck || this.mluck);
    }

    turn_refresh() {
      this.turn += 1;
      // Fresh armies become whatever tired armies were
      for (let i = 0; i < NODES; i++) {
        const wasTired = this.vector[GameState.IDX_NODES_OFFSET + 3 * i + 1];
        this.vector[GameState.IDX_NODES_OFFSET + 3 * i] = wasTired;
        this.vector[GameState.IDX_NODES_OFFSET + 3 * i + 1] = 0;
      }
      for (let i = 0; i < CARDS; i++) {
        this.vector[GameState.IDX_MYSORE_CARDS_OFFSET + i] = 1;
        this.vector[GameState.IDX_BRITISH_CARDS_OFFSET + i] = 1;
      }
      this.bluck = 0;
      this.mluck = 0;
    }

    toString() {
      let str = "";
      for (let i = 0; i < GAME_VECTOR_LENGTH; i++) {
        str += this.vector[i] ? "1" : "0";
      }
      return str;
    }

    read_str(bitStr) {
      if (bitStr.length !== GAME_VECTOR_LENGTH) {
        throw new Error(`Invalid bit-string length! Expected ${GAME_VECTOR_LENGTH}, got ${bitStr.length}`);
      }
      const clone = this.copy();
      for (let i = 0; i < GAME_VECTOR_LENGTH; i++) {
        const ch = bitStr[i];
        if (ch !== '0' && ch !== '1') throw new Error("Bit-string must contain only 1s and 0s.");
        clone.vector[i] = ch === '1' ? 1 : 0;
      }

      // Attacker
      clone._attacker = NO_UNIT;
      for (let i = 0; i < NODES; i++) {
        if (clone.vector[GameState.IDX_ATTACKER_OFFSET + i]) {
          clone._attacker = i;
          break;
        }
      }

      // Defender
      clone._defender = NO_UNIT;
      for (let i = 0; i < NODES; i++) {
        if (clone.vector[GameState.IDX_DEFENDER_OFFSET + i]) {
          clone._defender = i;
          break;
        }
      }

      // Card strength
      clone._card_strength = 0;
      for (let i = 0; i < 4; i++) {
        if (clone.vector[GameState.IDX_COMBAT_STRENGTH_OFFSET + i]) {
          clone._card_strength = i;
          break;
        }
      }

      // To move
      clone._to_move = 0;
      for (let i = 0; i < 3; i++) {
        if (clone.vector[GameState.IDX_WHO_TO_MOVE_OFFSET + i]) {
          clone._to_move = i;
          break;
        }
      }

      // Turn
      clone._turn = 1;
      for (let i = 0; i < 4; i++) {
        if (clone.vector[GameState.IDX_TURN_OFFSET + i]) {
          clone._turn = i + 1;
          break;
        }
      }

      return clone;
    }
  }

  // Export to global scope / modules
  const TDConstants = {
    EDGE_SOURCES, EDGE_DESTS, INDEX_MAP, NODE_TO_IDX, CARD_VALUE,
    EDGES, NODES, CARDS, TURNS, NO_UNIT, ADJACENCY_MATRIX,
    KEYS, COASTAL, KEY_INDICES, COASTAL_INDICES, MOVE_SPACE,
    WHO_TO_MOVE, MYSORE_CARDS, BRITISH_CARDS,
    GAME_VECTOR_LENGTH, MOVE_VECTOR_LENGTH,
    CARDS_ABBREV, NODES_ABBREV
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GameState, TDConstants };
  } else {
    global.GameState = GameState;
    global.TDConstants = TDConstants;
    Object.assign(global, TDConstants);
  }
})(typeof window !== 'undefined' ? window : this);
