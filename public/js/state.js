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
  const MOVE_VECTOR_LENGTH = totalMoves; // 959

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

    get state() {
      return this.vector;
    }

    read_str(bitStr) {
      if (bitStr.length !== GAME_VECTOR_LENGTH) {
        throw new Error(`Invalid bit-string length! Expected ${GAME_VECTOR_LENGTH}, got ${bitStr.length}`);
      }
      for (let i = 0; i < GAME_VECTOR_LENGTH; i++) {
        const ch = bitStr[i];
        if (ch !== '0' && ch !== '1') throw new Error("Bit-string must contain only 1s and 0s.");
        this.vector[i] = ch === '1' ? 1 : 0;
      }

      // Validate territory occupation (each node has at most one unit: fresh, tired, or fort)
      for (let i = 0; i < NODES; i++) {
        const start = GameState.IDX_NODES_OFFSET + i * 3;
        const total = this.vector[start] + this.vector[start + 1] + this.vector[start + 2];
        if (total > 1) {
          const name = INDEX_MAP[i] || `Territory ${i}`;
          throw new Error(`Multiple units assigned to territory ${name}!`);
        }
      }

      // Attacker
      this._attacker = NO_UNIT;
      for (let i = 0; i < NODES; i++) {
        if (this.vector[GameState.IDX_ATTACKER_OFFSET + i]) {
          this._attacker = i;
          break;
        }
      }

      // Defender
      this._defender = NO_UNIT;
      for (let i = 0; i < NODES; i++) {
        if (this.vector[GameState.IDX_DEFENDER_OFFSET + i]) {
          this._defender = i;
          break;
        }
      }

      // Card strength
      this._card_strength = 0;
      for (let i = 0; i < 4; i++) {
        if (this.vector[GameState.IDX_COMBAT_STRENGTH_OFFSET + i]) {
          this._card_strength = i;
          break;
        }
      }

      // To move
      this._to_move = 0;
      for (let i = 0; i < 3; i++) {
        if (this.vector[GameState.IDX_WHO_TO_MOVE_OFFSET + i]) {
          this._to_move = i;
          break;
        }
      }

      // Turn
      this._turn = 1;
      for (let i = 0; i < 4; i++) {
        if (this.vector[GameState.IDX_TURN_OFFSET + i]) {
          this._turn = i + 1;
          break;
        }
      }

      return this;
    }

    zobristHash() {
      let hash = 0n;
      for (let i = 0; i < 148; i++) {
        if (this.vector[i] === 1) {
          hash ^= ZOBRIST_KEYS[i];
        }
      }
      return hash.toString(16);
    }
  }

  const ZOBRIST_HEX_KEYS = [
    "0xc6aa0c083f434d3d", "0x04a71bfb246e5adb", "0x7cc5f88557040745", "0x5a13541d232cc612", "0xa13698a483258126",
    "0xc6ff2a684ecbfb37", "0xad7cb0ad0936f3c3", "0x0c438955ec98f137", "0xbe78b6cf46dc69ed", "0x899181d0dac3a4f7",
    "0xe7170ac390e73f87", "0x2123ed073482b473", "0xac5440c449971460", "0x3d3a6e523ee6474f", "0xfa99aaee4fbb823a",
    "0xacb1881c71e94cf7", "0xdfb1cc7c215066b5", "0x28618e0a8b370b48", "0x34a31099fb80ea66", "0xf9ff886e8ec833ae",
    "0x070a5fbd85c8e9c5", "0xf273b707b9960cb2", "0x10cb14122fc66de5", "0x8573ec2fda7009f7", "0xaffaf29d9524adc2",
    "0xfe8b33ec1c1965da", "0xf0d8baa43939945d", "0xcfa04b778899960a", "0x43255f341d8e5a11", "0x63eeb5284a3d6036",
    "0xdb97acebc9984ad0", "0xddabdf8f6e0dc43e", "0x257b4a2d4e11d2e6", "0x956a0821007a74bd", "0x420c50842ae24213",
    "0xe57d9f26215aa0cf", "0x76cfabf877c0a308", "0x8b04571d0ee3e8ad", "0xc122701d026aa889", "0xafa3ac75c4a0815f",
    "0xd2410e796242e4b7", "0xe6c0e1382f56ac0b", "0x35053766e7c56b9a", "0x9a0393fff1131d71", "0xa50e1dbe251f57ff",
    "0x449cfe201c870a45", "0x4ff13148af0c57b5", "0xca5459109f70281c", "0xad3ff33a42f7f8b1", "0x9f645d9be179a0a0",
    "0x334c24844b4adc35", "0xeefa097c0ddfe8b9", "0xe9c9b4bd08fd8a3a", "0x3560199f60cde5b4", "0x0d0f1a78d156ef3f",
    "0xfd083bc0d2a3944f", "0xae4b8bf8312e1eb2", "0x0ac65f8b83b9cd61", "0x81f9356ed5b9a4fe", "0xe651933e5de0e621",
    "0x4538c6c79f35ff0a", "0x9156a255a56b52d5", "0x00901b066b5686d4", "0xf163efae0f1c3316", "0x3cf6bb567d04eaf4",
    "0xfb3ec881e24ed21f", "0x3f7263ce4beef35b", "0xd355d587bb8d8ecf", "0x4726dbeaac5259fc", "0x7318bddcdbfca578",
    "0x940c4b64ab1f81ea", "0xbcdc5e47a5540c2f", "0x16c8035fa85d73c5", "0x4fc34cfbd2dc865d", "0x981db1fa0f84e653",
    "0x6d297ce8a032f430", "0xf2a3e03baca5a5a2", "0xb8941e6a96d882db", "0xf6d338f3d5100f88", "0xee02043545208559",
    "0xd25247cae86b25f3", "0xad625e2e1d8aa0ab", "0xd3a59ad7ee364a0e", "0xba4908c49bfb1c72", "0x611b6d007da7214d",
    "0x78e3cd5ff6acb47d", "0xfe778e88c499aa5e", "0x7d0ddd982e57cb69", "0x0651eaaccfe51ba4", "0x9c565c1d7de9dac8",
    "0xde6b5d605d063b1d", "0xb274ef3efb57187f", "0x8aa48191b879cb24", "0xba31aa72fc6e59b3", "0x5a7386bf13d1c890",
    "0x81f5b5a87096d9ac", "0xe09882b39aaf0cc5", "0xb7160814e1ecb7af", "0x6a4da9a268cb3936", "0x4f95d44cf6f3030a",
    "0x98da546fcad61e37", "0xc47bd2473c51d3a1", "0x1a28cd5c6dacbb35", "0x3e2410617bb58e4f", "0xfbd683bffcb253d3",
    "0x1dc46f929aef91ab", "0x8dc10fe9a0550744", "0xf8f78bd2dbd2d9e8", "0xbbe7907768267cd1", "0x4d9405239899d6a1",
    "0xe8b1dc402ce44323", "0x204e6a2c1b9b4b3b", "0xacb0e4bdae2da119", "0xaa824863263adb19", "0xb5f7e36604f3c29b",
    "0x858fca3423800a80", "0xd09bf32d328f255b", "0x586da6e54bc44d94", "0xddf2e084f3536e3c", "0x3241068a8c04cbb0",
    "0xc98e26692ab7ff1d", "0xa300522468b473d3", "0x83ac6fc151cb759d", "0x5ff5f674f7820709", "0xeaa6d37958c8e3e5",
    "0x64a4661ac2343778", "0xaa48273ce054d847", "0xed737d70681d265f", "0xd7c983d0840cb245", "0x471981274bcc340f",
    "0x4c2bc7cac3a5fe20", "0x1caebb86b83c36d0", "0x23fe88c2f89d24a6", "0x7443aa86c77fd037", "0x7bacdee68b188d70",
    "0x9e6b1e2d1ea939d9", "0x8fd0bd52cb6be718", "0xa60e0e3a7df2549e", "0x681015d8b713b881", "0x39bfebd6a260460a",
    "0x0a3327b02b6d9f94", "0x21fe02eccee1506f", "0x29f1bf0e8e8ed16a", "0x7e67e1221dd4c533", "0x641588c2e9b4c4db",
    "0xb20fe3cfccec10c0", "0x12f1b25b09d5c9e8", "0x292c54af6d75b826"
  ];
  const ZOBRIST_KEYS = ZOBRIST_HEX_KEYS.map(h => BigInt(h));

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
    module.exports = Object.assign({ GameState, TDConstants }, TDConstants);
  } else {
    global.GameState = GameState;
    global.TDConstants = TDConstants;
    Object.assign(global, TDConstants);
  }
})(typeof window !== 'undefined' ? window : this);
