/**
 * Tiger's Day – Client-Side MCTS & ONNX WebAssembly Inference Engine
 * Runs 100% in the browser with zero server latency.
 */

(function(global) {
  'use strict';

  const {
    MOVE_VECTOR_LENGTH,
    GAME_VECTOR_LENGTH,
    INDEX_MAP,
    CARDS_ABBREV,
    NODES_ABBREV
  } = global.TDConstants || require('./state.js').TDConstants;

  const {
    getLegalMoves,
    getNextState,
    getLuckOutcomes,
    getStateWinner,
    resolveLuck,
    notate
  } = global.TDEngine || require('./engine.js');

  // =========================================================================
  // 1. MCTS SEARCH TREE NODE
  // =========================================================================
  class MCTSNode {
    constructor(state, parent = null, move = null, prior = 0.0) {
      this.state = state;
      this.parent = parent;
      this.move = move;
      this.prior = prior;
      this._moves = [];
      this._priors = [];
      this._nodes = [];
      this._children = null;
      this.visit_count = 0;
      this.value_sum = 0.0;
      this.offset = 0;
    }

    get eval() {
      if (this.parent && this.visit_count === 0) {
        return this.parent.eval;
      }
      return this.visit_count > 0 ? this.value_sum / this.visit_count : 0.0;
    }

    get is_expanded() {
      return this._nodes.length > 0;
    }

    // Expose the tree in action order; search creates only selected nodes.
    get children() {
      if (this._children === null) {
        this._children = new Map();
        for (let i = 0; i < this._nodes.length; i++) {
          this._nodes[i] ??= new MCTSNode(null, this, this._moves[i], this._priors[i]);
          this._children.set(this._moves[i], this._nodes[i]);
        }
      }
      return this._children;
    }

    // round robin from a random offset for unbiased luck sampling
    get luck_child() {
      return this._nodes[(this.offset + this.visit_count) % this._nodes.length];
    }

    // lazy child creation, start as null
    expand_decision(moves, priors) {
      this._moves = moves;
      this._priors = priors;
      this._nodes = new Array(moves.length).fill(null);
      this._children = null;
    }

    expand_luck() {
      const outcomes = getLuckOutcomes(this.state);
      const prior = 1.0 / outcomes.length;
      this._moves = outcomes.map((_, i) => i);
      this._priors = outcomes.map(() => prior);
      this._nodes = outcomes.map((outcome, i) => new MCTSNode(outcome, this, i, prior));
      this._children = null;
      this.offset = Math.floor(Math.random() * outcomes.length);
    }
  }

  // =========================================================================
  // 2. ONNX RUNTIME WEB INFERENCE WRAPPER
  // =========================================================================
  class ONNXModelWrapper {
    constructor(modelPath = './alphatiger.onnx') {
      this.modelPath = modelPath;
      this.session = null;
      this.loadPromise = null;
    }

    async init() {
      if (this.session) return this.session;
      if (this.loadPromise) return this.loadPromise;

      this.loadPromise = (async () => {
        if (typeof ort === 'undefined') {
          console.warn("⚠️ ort (onnxruntime-web) not detected. Using heuristic fallback.");
          return null;
        }

        try {
          if (ort.env && ort.env.wasm) {
            ort.env.wasm.numThreads = 1;
            ort.env.wasm.simd = true;
          }
          this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['wasm']
          });
          console.log(`✅ Loaded ONNX WebAssembly Model from ${this.modelPath}`);
          return this.session;
        } catch (err) {
          console.warn(`⚠️ Failed to load ONNX model (${this.modelPath}):`, err);
          return null;
        }
      })();

      return this.loadPromise;
    }

    async predict(state) {
      if (!this.session) {
        await this.init();
      }

      if (!this.session) {
        return {
          value: 0.0,
          rawLogits: new Float32Array(MOVE_VECTOR_LENGTH)
        };
      }

      const floatVec = new Float32Array(state.vector.length);
      for (let i = 0; i < state.vector.length; i++) {
        floatVec[i] = state.vector[i] ? 1.0 : 0.0;
      }

      const inputTensor = new ort.Tensor('float32', floatVec, [1, state.vector.length]);
      const inputName = this.session.inputNames[0] || 'board_state';

      const feeds = {};
      feeds[inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const valOut = results.value || results[this.session.outputNames[0]];
      const polOut = results.policy_logits || results[this.session.outputNames[1]];

      const value = valOut ? Number(valOut.data[0]) : 0.0;
      const rawLogits = polOut ? polOut.data : new Float32Array(MOVE_VECTOR_LENGTH);

      return { value, rawLogits };
    }
  }

  // =========================================================================
  // 3. MONTE CARLO TREE SEARCH (MCTS)
  // =========================================================================
  class MCTS {
    constructor(model, options = {}) {
      this.model = model;
      this.simulations = options.simulations || 400;
      this.ipuct = options.ipuct || 800;
      this.root = null;
    }

    selectChild(node) {
      // calculate once for every child
      const puct = 1.25 + Math.log((node.visit_count + this.ipuct) / this.ipuct);
      const visitsSqrt = Math.sqrt(node.visit_count);
      const mysore = node.state.to_move === 1;
      const parentEval = node.value_sum / node.visit_count;
      let bestScore = -Infinity;
      let bestIndex = -1;

      for (let i = 0; i < node._nodes.length; i++) {
        const child = node._nodes[i];
        const visits = child ? child.visit_count : 0;
        const value = visits ? child.value_sum / visits : parentEval;
        // flip evaluation for mysore
        const exploitation = mysore ? -value : value;
        const exploration = puct * node._priors[i] * (visitsSqrt / (1 + visits));
        const score = exploitation + exploration;

        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }

      node._nodes[bestIndex] ??= new MCTSNode(null, node, node._moves[bestIndex], node._priors[bestIndex]);
      return node._nodes[bestIndex];
    }

    backpropagate(node, value) {
      let curr = node;
      while (curr !== null) {
        curr.visit_count += 1;
        curr.value_sum += value;
        curr = curr.parent;
      }
    }

    // fresh tree every search so the bot never thinks beyond its sims
    async search(rootState, onProgress = null) {
      // local root so overlapping searches never share a tree
      const root = this.root = new MCTSNode(rootState.copy());

      for (let currentSim = 0; currentSim < this.simulations; currentSim++) {
        let node = root;

        // 1. Selection
        while (node.is_expanded) {
          node = node.state.is_luck ? node.luck_child : this.selectChild(node);
        }

        // 2. State Materialization (Lazy evaluation)
        if (node.state === null) {
          node.state = getNextState(node.parent.state, node.move);
        }

        // Luck resolution loop
        while (node.state.is_luck) {
          if (!node.is_expanded) {
            node.expand_luck();
          }
          node = node.luck_child;
        }

        // 3. Terminal evaluation check
        const reward = getStateWinner(node.state);
        if (reward !== 0) {
          this.backpropagate(node, reward);
          continue;
        }

        // 4. Neural Network Inference
        const { value, rawLogits } = await this.model.predict(node.state);
        const legalMask = getLegalMoves(node.state);

        let maxLogit = -Infinity;
        for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) {
          if (legalMask[i]) {
            if (rawLogits[i] > maxLogit) maxLogit = rawLogits[i];
          }
        }

        let sumExp = 0;
        const moves = [];
        const expLogits = [];
        for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) {
          if (legalMask[i]) {
            const e = Math.exp(rawLogits[i] - maxLogit);
            moves.push(i);
            expLogits.push(e);
            sumExp += e;
          }
        }

        // filter moves whose prior underflows to zero
        const policy = expLogits.map(e => e / sumExp);
        node.expand_decision(moves.filter((_, i) => policy[i] > 0), policy.filter(p => p > 0));
        this.backpropagate(node, value);

        if (onProgress && currentSim % 50 === 0) {
          onProgress(currentSim, this.simulations);
        }
      }

      return root;
    }

    async findMove(state, temperature = 0.0) {
      const root = await this.search(state);
      const counts = new Float32Array(MOVE_VECTOR_LENGTH);

      for (const [m, child] of root.children.entries()) {
        counts[m] = child.visit_count;
      }

      if (temperature === 0.0) {
        let bestMove = 0;
        let maxCount = -1;
        for (let m = 0; m < MOVE_VECTOR_LENGTH; m++) {
          if (counts[m] > maxCount) {
            maxCount = counts[m];
            bestMove = m;
          }
        }
        return { bestMove, counts };
      }

      let sum = 0;
      for (let m = 0; m < MOVE_VECTOR_LENGTH; m++) {
        counts[m] = Math.pow(counts[m], 1.0 / temperature);
        sum += counts[m];
      }

      if (sum === 0) {
        return { bestMove: 0, counts };
      }

      let r = Math.random() * sum;
      let chosenMove = 0;
      for (let m = 0; m < MOVE_VECTOR_LENGTH; m++) {
        r -= counts[m];
        if (r <= 0) {
          chosenMove = m;
          break;
        }
      }

      return { bestMove: chosenMove, counts };
    }

    getTopCandidateLines(limit = 3) {
      if (!this.root || this.root.children.size === 0) return [];

      const sortedChildren = Array.from(this.root.children.entries())
        .sort((a, b) => b[1].visit_count - a[1].visit_count);

      const topLines = [];
      const rootState = this.root.state;

      for (let i = 0; i < Math.min(limit, sortedChildren.length); i++) {
        const [move, node] = sortedChildren[i];
        const pvLine = [notate(rootState, move)];

        let currState = getNextState(rootState, move);
        let currNode = node;

        while (currState && !currState.is_luck && currNode.children.size > 0 && pvLine.length < 6) {
          let bestChildMove = null;
          let bestChildNode = null;
          let maxVisits = -1;

          for (const [m, c] of currNode.children.entries()) {
            if (c.visit_count > maxVisits) {
              maxVisits = c.visit_count;
              bestChildMove = m;
              bestChildNode = c;
            }
          }

          if (bestChildMove === null) break;
          pvLine.push(notate(currState, bestChildMove));
          currState = getNextState(currState, bestChildMove);
          currNode = bestChildNode;
        }

        topLines.push({
          moveIdx: move,
          moveName: pvLine.join(' '),
          eval: Number(node.eval.toFixed(2)),
          visits: node.visit_count
        });
      }

      return topLines;
    }
  }

  const TDMCTS = {
    MCTSNode,
    ONNXModelWrapper,
    MCTS
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDMCTS;
  } else {
    global.TDMCTS = TDMCTS;
    Object.assign(global, TDMCTS);
  }
})(typeof window !== 'undefined' ? window : this);
