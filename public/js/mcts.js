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
      this.children = new Map(); // moveIdx -> MCTSNode
      this.visit_count = 0;
      this.value_sum = 0.0;
    }

    get eval() {
      if (this.parent && this.visit_count === 0) {
        return this.parent.eval;
      }
      return this.visit_count > 0 ? this.value_sum / this.visit_count : 0.0;
    }

    get is_expanded() {
      return this.children.size > 0;
    }

    get is_luck() {
      return Boolean(this.state && this.state.is_luck);
    }

    expand_decision(actionPriors) {
      for (let move = 0; move < actionPriors.length; move++) {
        const prior = actionPriors[move];
        if (prior > 0.0 && !this.children.has(move)) {
          this.children.set(move, new MCTSNode(null, this, move, prior));
        }
      }
    }

    expand_luck() {
      const outcomes = getLuckOutcomes(this.state);
      const prior = 1.0 / outcomes.length;
      for (let i = 0; i < outcomes.length; i++) {
        if (!this.children.has(i)) {
          this.children.set(i, new MCTSNode(outcomes[i], this, i, prior));
        }
      }
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
      this.simulations = options.simulations || 200;
      this.ipuct = options.ipuct || 800;
      this.dalpha = options.dalpha || 0.5;
      this.depsilon = options.depsilon || 0.25;
      this.root = null;
    }

    sampleGamma(shape, scale = 1.0) {
      // Marsaglia and Tsang method for Gamma(alpha, 1) when alpha >= 1
      if (shape < 1) {
        return this.sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
      }
      const d = shape - 1 / 3;
      const c = 1 / Math.sqrt(9 * d);
      while (true) {
        let u = Math.random();
        let v = 0;
        let x = 0;
        do {
          const z = (Math.random() + Math.random() + Math.random() + Math.random() - 2) * 1.732; // Normal approx
          v = 1 + c * z;
        } while (v <= 0);
        v = v * v * v;
        x = (Math.random() + Math.random() - 1);
        if (u < 1 - 0.0331 * x * x * x * x) return d * v * scale;
        if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
      }
    }

    getDirichletNoise(length, alpha) {
      const samples = new Float32Array(length);
      let sum = 0;
      for (let i = 0; i < length; i++) {
        samples[i] = this.sampleGamma(alpha, 1.0);
        sum += samples[i];
      }
      if (sum > 0) {
        for (let i = 0; i < length; i++) samples[i] /= sum;
      }
      return samples;
    }

    selectChild(node, noiseDict = null) {
      let bestScore = -Infinity;
      let bestChild = null;

      for (const [move, child] of node.children.entries()) {
        const exploitation = node.state.to_move === 1 ? -child.eval : child.eval;

        let prior = child.prior;
        if (noiseDict && noiseDict.has(move)) {
          prior = (1 - this.depsilon) * prior + this.depsilon * noiseDict.get(move);
        }

        const puct = 1.25 + Math.log((node.visit_count + this.ipuct) / this.ipuct);

        const exploration = puct * prior * (Math.sqrt(node.visit_count) / (1 + child.visit_count));
        const score = exploitation + exploration;

        if (score > bestScore) {
          bestScore = score;
          bestChild = child;
        }
      }

      return bestChild;
    }

    backpropagate(node, value) {
      let curr = node;
      while (curr !== null) {
        curr.visit_count += 1;
        curr.value_sum += value;
        curr = curr.parent;
      }
    }

    async search(rootState, stop = true, onProgress = null) {
      if (!this.root || !this.root.state || this.root.state.toString() !== rootState.toString()) {
        this.root = new MCTSNode(rootState.copy());
      }

      let noiseDict = null;
      const warmup = stop ? Math.floor(this.simulations / 5) : this.simulations;
      const stopThreshold = 0.9;

      for (let currentSim = 0; currentSim < this.simulations; currentSim++) {
        let node = this.root;

        // Early stopping if single move dominates
        if (currentSim > warmup && this.root.children.size > 0) {
          let maxVisits = 0;
          for (const child of this.root.children.values()) {
            if (child.visit_count > maxVisits) maxVisits = child.visit_count;
          }
          if (this.root.visit_count > 0 && (maxVisits / this.root.visit_count) > stopThreshold) {
            return this.root;
          }
        }

        // 1. Selection
        while (node.is_expanded) {
          if (node.is_luck) {
            const childArr = Array.from(node.children.values());
            node = childArr[Math.floor(Math.random() * childArr.length)];
          } else {
            if (node === this.root && noiseDict === null && this.depsilon > 0) {
              const legalMoves = Array.from(this.root.children.keys());
              const noise = this.getDirichletNoise(legalMoves.length, this.dalpha);
              noiseDict = new Map();
              for (let i = 0; i < legalMoves.length; i++) {
                noiseDict.set(legalMoves[i], noise[i]);
              }
            }
            node = this.selectChild(node, node === this.root ? noiseDict : null);
            if (!node) break;
          }
        }

        if (!node) continue;

        // 2. State Materialization (Lazy evaluation)
        if (node.state === null) {
          if (node.parent && node.parent.state) {
            node.state = getNextState(node.parent.state, node.move);
          } else {
            continue;
          }
        }

        // Luck resolution loop
        while (node.is_luck) {
          if (!node.is_expanded) {
            node.expand_luck();
          }
          const childArr = Array.from(node.children.values());
          node = childArr[Math.floor(Math.random() * childArr.length)];
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
        const expLogits = new Float32Array(MOVE_VECTOR_LENGTH);
        for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) {
          if (legalMask[i]) {
            const e = Math.exp(rawLogits[i] - maxLogit);
            expLogits[i] = e;
            sumExp += e;
          }
        }

        const policy = new Float32Array(MOVE_VECTOR_LENGTH);
        if (sumExp > 0) {
          for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) {
            if (legalMask[i]) policy[i] = expLogits[i] / sumExp;
          }
        }

        node.expand_decision(policy);
        this.backpropagate(node, value);

        if (onProgress && currentSim % 50 === 0) {
          onProgress(currentSim, this.simulations);
        }
      }

      return this.root;
    }

    async findMove(state, temperature = 0.0) {
      const root = await this.search(state, true);
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
