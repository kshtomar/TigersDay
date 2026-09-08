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
    NODES_ABBREV,
    EDGE_SOURCES,
    EDGE_DESTS,
    COASTAL_INDICES,
    MOVE_SPACE,
    NO_UNIT
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
      this.activeProvider = 'wasm';
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

          // Check WebGPU hardware acceleration support
          if (typeof navigator !== 'undefined' && navigator.gpu) {
            try {
              this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['webgpu', 'wasm']
              });
              this.activeProvider = 'webgpu';
              console.log(`⚡ Loaded ONNX WebGPU Accelerated Model from ${this.modelPath}`);
              return this.session;
            } catch (gpuErr) {
              console.warn("⚠️ WebGPU session creation failed, falling back to WASM:", gpuErr);
            }
          }

          this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['wasm']
          });
          this.activeProvider = 'wasm';
          console.log(`✅ Loaded ONNX WebAssembly Model from ${this.modelPath}`);
          return this.session;
        } catch (err) {
          console.warn(`⚠️ Failed to load ONNX model (${this.modelPath}):`, err);
          return null;
        }
      })();

      return this.loadPromise;
    }

    getActiveProvider() {
      return this.activeProvider;
    }

    async predictBatch(states) {
      if (!this.session) await this.init();
      if (!this.session || !states || states.length === 0) return [];

      const B = states.length;
      const V_LEN = states[0].vector.length;
      const floatVec = new Float32Array(B * V_LEN);

      for (let b = 0; b < B; b++) {
        const s = states[b];
        for (let i = 0; i < V_LEN; i++) {
          floatVec[b * V_LEN + i] = s.vector[i] ? 1.0 : 0.0;
        }
      }

      const inputTensor = new ort.Tensor('float32', floatVec, [B, V_LEN]);
      const inputName = this.session.inputNames[0] || 'board_state';
      const feeds = {};
      feeds[inputName] = inputTensor;

      const results = await this.session.run(feeds);
      const valOut = results.value || results[this.session.outputNames[0]];
      const polOut = results.policy_logits || results[this.session.outputNames[1]];

      const outputs = [];
      for (let b = 0; b < B; b++) {
        const value = valOut ? Number(valOut.data[b]) : 0.0;
        const rawLogits = polOut ? polOut.data.subarray(b * MOVE_VECTOR_LENGTH, (b + 1) * MOVE_VECTOR_LENGTH) : new Float32Array(MOVE_VECTOR_LENGTH);
        outputs.push({ value, rawLogits });
      }

      return outputs;
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
      this.openingBook = options.openingBook || null;
      this.useOpeningBook = options.useOpeningBook !== false;
      this.transpositionTable = new Map();
    }

    async loadOpeningBook(url = './opening_book.json') {
      try {
        if (typeof fetch !== 'undefined') {
          const resp = await fetch(url);
          if (resp.ok) {
            this.openingBook = await resp.json();
            return this.openingBook;
          }
        }
      } catch (err) {
        console.warn("Could not load opening book:", err);
      }
      return null;
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

        // 4. Neural Network Inference or Transposition Table Lookup
        let value, policy;
        const zHash = node.state.zobristHash ? node.state.zobristHash() : node.state.toString();
        if (this.transpositionTable.has(zHash)) {
          const cached = this.transpositionTable.get(zHash);
          value = cached.value;
          policy = cached.policy;
        } else {
          const pred = await this.model.predict(node.state);
          value = pred.value;
          const rawLogits = pred.rawLogits;
          const legalMask = getLegalMoves(node.state);

          let maxLogit = -Infinity;
          for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) {
            if (legalMask[i] && rawLogits[i] > maxLogit) {
              maxLogit = rawLogits[i];
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

          policy = new Float32Array(MOVE_VECTOR_LENGTH);
          if (sumExp > 0) {
            for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) {
              if (legalMask[i]) policy[i] = expLogits[i] / sumExp;
            }
          }
          this.transpositionTable.set(zHash, { value, policy });
        }

        node.expand_decision(policy);
        this.backpropagate(node, value);

        if (onProgress && currentSim % 50 === 0) {
          onProgress(currentSim, this.simulations);
        }
      }

      return this.root;
    }

    async searchTimeBudget(rootState, timeBudgetMs = 1500, minSims = 50, maxSims = 2000, stop = true, onProgress = null) {
      const startTime = (typeof performance !== 'undefined' ? performance : Date).now();
      let effectiveBudget = timeBudgetMs;

      // Contested fortress bonus (+15% time on Turn 2/3)
      if (rootState.turn === 2 || rootState.turn === 3) {
        let contested = false;
        for (let i = 0; i < 25; i++) {
          if (rootState.forts[i]) {
            for (let j = 0; j < 25; j++) {
              if (TDConstants.ADJACENCY_MATRIX && TDConstants.ADJACENCY_MATRIX[i * 25 + j] && (rootState.fresh_armies[j] || rootState.tired_armies[j])) {
                contested = true;
                break;
              }
            }
          }
          if (contested) break;
        }
        if (contested) effectiveBudget = Math.floor(timeBudgetMs * 1.15);
      }

      // Early exit if <= 1 legal move
      const legalMask = getLegalMoves(rootState);
      let legalCount = 0;
      for (let i = 0; i < MOVE_VECTOR_LENGTH; i++) if (legalMask[i]) legalCount++;
      if (legalCount <= 1) {
        this.simulations = minSims;
        return this.search(rootState, false, onProgress);
      }

      this.simulations = maxSims;
      const root = await this.search(rootState, stop, (sim, total) => {
        if (onProgress) onProgress(sim, total);
      });
      return root;
    }

    async findMove(state, temperature = 0.0, timeBudgetMs = null) {
      if (this.useOpeningBook && this.openingBook) {
        const stateKey = state.toString();
        const entry = this.openingBook[stateKey];
        if (entry && entry.move !== undefined) {
          const legalMask = getLegalMoves(state);
          if (legalMask[entry.move]) {
            const counts = new Float32Array(MOVE_VECTOR_LENGTH);
            counts[entry.move] = 1;
            return { bestMove: entry.move, counts, isBook: true, notation: entry.notation };
          }
        }
      }

      const root = timeBudgetMs ? await this.searchTimeBudget(state, timeBudgetMs) : await this.search(state, true);
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

    decodeMoveGeometry(state, move) {
      if (!MOVE_SPACE) return { type: "Unknown", fromNode: null, toNode: null, isAttack: false };
      let offset = 0;
      for (const [name, size, moveType] of MOVE_SPACE) {
        if (move >= offset && move < offset + size) {
          const idx = move - offset;
          if (moveType === "edge") {
            const src = EDGE_SOURCES[idx];
            const dest = EDGE_DESTS[idx];
            return {
              type: name,
              fromNode: src,
              toNode: dest,
              isAttack: Boolean(state && state.forts && state.forts[dest])
            };
          } else if (moveType === "node") {
            return {
              type: name,
              fromNode: idx,
              toNode: idx,
              isAttack: false
            };
          } else if (moveType === "coastal") {
            const numCoasts = COASTAL_INDICES.length;
            const nodeIdx = Math.floor(idx / numCoasts);
            const coastIdx = COASTAL_INDICES[idx % numCoasts];
            if (name === "Royal Navy") {
              return {
                type: name,
                fromNode: nodeIdx,
                toNode: coastIdx,
                isAttack: Boolean(state && state.forts && state.forts[coastIdx])
              };
            } else {
              return {
                type: name,
                fromNode: coastIdx,
                toNode: nodeIdx,
                isAttack: false
              };
            }
          } else {
            const battleTerritory = (state && state.is_battle && state.defender !== undefined && state.defender !== NO_UNIT)
              ? state.defender
              : null;
            return {
              type: name,
              fromNode: battleTerritory,
              toNode: battleTerritory,
              isAttack: false
            };
          }
        }
        offset += size;
      }
      return { type: "Unknown", fromNode: null, toNode: null, isAttack: false };
    }

    getTopCandidateLines(limit = 3) {
      if (!this.root || this.root.children.size === 0) return [];

      const sortedChildren = Array.from(this.root.children.entries())
        .sort((a, b) => {
          if (b[1].visit_count !== a[1].visit_count) {
            return b[1].visit_count - a[1].visit_count;
          }
          return b[1].prior - a[1].prior;
        });

      const topLines = [];
      const rootState = this.root.state;
      const parsedLimit = Math.max(1, parseInt(limit, 10) || 3);
      const effectiveLimit = Math.min(parsedLimit, sortedChildren.length);

      for (let i = 0; i < effectiveLimit; i++) {
        const [move, node] = sortedChildren[i];
        const firstNotation = notate(rootState, move);
        const pvLine = [firstNotation];
        const geom = this.decodeMoveGeometry(rootState, move);

        let currState = (node && node.state) ? node.state : getNextState(rootState, move);
        let currNode = node;
        let reachedLuck = false;
        let luckType = null;

        // Check if move 1 immediately reached a luck state
        if (currState && currState.is_luck) {
          reachedLuck = true;
          if (currState.is_battle) {
            const targetName = (currState.defender !== undefined && INDEX_MAP[currState.defender]) ? INDEX_MAP[currState.defender] : "Fort";
            luckType = `Battle: ${targetName}`;
            pvLine.push(`[🎲 Battle: ${targetName}]`);
          } else {
            luckType = "Luck Roll";
            pvLine.push("[🎲 Luck Roll]");
          }
        } else {
          // Project forward step-by-step until luck state, terminal state, or max depth (10 plies)
          while (currState && !currState.is_luck && pvLine.length < 10) {
            const winner = getStateWinner(currState);
            if (winner !== 0) {
              pvLine.push(winner === 1 ? "[👑 British Victory]" : "[🐅 Mysore Victory]");
              break;
            }

            let bestChildMove = null;
            let bestChildNode = null;
            let maxVisits = -1;

            if (currNode && currNode.children && currNode.children.size > 0) {
              for (const [m, c] of currNode.children.entries()) {
                if (c.visit_count > maxVisits) {
                  maxVisits = c.visit_count;
                  bestChildMove = m;
                  bestChildNode = c;
                }
              }
            }

            // If frontier reached in MCTS tree, check legal moves for single-response or pass
            if (bestChildMove === null) {
              const legalMask = getLegalMoves(currState);
              let firstLegal = null;
              let legalCount = 0;
              for (let m = 0; m < MOVE_VECTOR_LENGTH; m++) {
                if (legalMask[m]) {
                  if (firstLegal === null) firstLegal = m;
                  legalCount++;
                }
              }
              if (legalCount === 1) {
                bestChildMove = firstLegal;
              } else {
                break;
              }
            }

            if (bestChildMove === null) break;

            pvLine.push(notate(currState, bestChildMove));
            currState = (bestChildNode && bestChildNode.state) ? bestChildNode.state : getNextState(currState, bestChildMove);
            currNode = bestChildNode;

            if (currState && currState.is_luck) {
              reachedLuck = true;
              if (currState.is_battle) {
                const targetName = (currState.defender !== undefined && INDEX_MAP[currState.defender]) ? INDEX_MAP[currState.defender] : "Fort";
                luckType = `Battle: ${targetName}`;
                pvLine.push(`[🎲 Battle: ${targetName}]`);
              } else {
                luckType = "Luck Roll";
                pvLine.push("[🎲 Luck Roll]");
              }
              break;
            }
          }
        }

        // Winrate calculation based on perspective
        const rawEval = Number(node.eval.toFixed(2));
        const winratePct = rootState.to_move === 1
          ? Math.round((1 - rawEval) / 2 * 100)
          : Math.round((rawEval + 1) / 2 * 100);

        topLines.push({
          rank: i + 1,
          moveIdx: move,
          firstMoveNotation: firstNotation,
          moveName: pvLine.join(' '),
          lineNotation: pvLine.join(' '),
          eval: rawEval,
          visits: node.visit_count,
          winrate: `${winratePct}%`,
          fromNode: geom.fromNode,
          toNode: geom.toNode,
          fromName: geom.fromNode !== null && INDEX_MAP ? INDEX_MAP[geom.fromNode] : null,
          toName: geom.toNode !== null && INDEX_MAP ? INDEX_MAP[geom.toNode] : null,
          actionType: geom.type,
          isAttack: geom.isAttack,
          reachedLuck: reachedLuck,
          luckType: luckType
        });
      }

      return topLines;
    }
  }

  function getCardInfoForMove(moveIndex) {
    if (!MOVE_SPACE || moveIndex === null || moveIndex === undefined || moveIndex < 0) return null;
    let offset = 0;
    for (const [name, size] of MOVE_SPACE) {
      if (moveIndex >= offset && moveIndex < offset + size) {
        const subIdx = moveIndex - offset;
        switch (name) {
          case "Sepoy Mutiny":
            return { faction: "mysore", cardName: "Sepoy Mutiny", cardIndex: 1, toNode: subIdx, isTrade: false };
          case "French Alliance":
            return { faction: "mysore", cardName: "French Alliance", cardIndex: 2, toNode: subIdx, isTrade: false };
          case "Monsoon":
            return { faction: "mysore", cardName: "Monsoon", cardIndex: 3, toNode: subIdx, isTrade: false };
          case "Cavalry Raid":
            return { faction: "mysore", cardName: "Cavalry Raid", cardIndex: 4, toNode: null, isTrade: false };
          case "Sea Trade": {
            const numCoasts = typeof COASTAL_INDICES !== 'undefined' ? COASTAL_INDICES.length : 5;
            const nodeIdx = Math.floor(subIdx / numCoasts);
            const coastIdx = (typeof COASTAL_INDICES !== 'undefined') ? COASTAL_INDICES[subIdx % numCoasts] : 0;
            return { faction: "mysore", cardName: "Sea Trade", cardIndex: 5, fromNode: coastIdx, toNode: nodeIdx, isTrade: false };
          }
          case "Mysore Power":
            return { faction: "mysore", cardName: "Iron Rockets", cardIndex: 0, toNode: null, isTrade: false, isPower: true };
          case "Draw Iron Rockets":
            return { faction: "mysore", cardName: "Iron Rockets", cardIndex: 0, tradeTargetCard: subIdx, isTrade: true };
          case "Draw Sepoy Mutiny":
            return { faction: "mysore", cardName: "Sepoy Mutiny", cardIndex: 1, tradeTargetCard: subIdx, isTrade: true };
          case "Draw French Alliance":
            return { faction: "mysore", cardName: "French Alliance", cardIndex: 2, tradeTargetCard: subIdx, isTrade: true };
          case "Highlanders":
            return { faction: "british", cardName: "Highlanders", cardIndex: 1, toNode: subIdx, isTrade: false };
          case "Royal Navy": {
            const numCoasts = typeof COASTAL_INDICES !== 'undefined' ? COASTAL_INDICES.length : 5;
            const nodeIdx = Math.floor(subIdx / numCoasts);
            const coastIdx = (typeof COASTAL_INDICES !== 'undefined') ? COASTAL_INDICES[subIdx % numCoasts] : 0;
            return { faction: "british", cardName: "Royal Navy", cardIndex: 2, fromNode: nodeIdx, toNode: coastIdx, isTrade: false };
          }
          case "Divide and Rule": {
            const src = typeof EDGE_SOURCES !== 'undefined' ? EDGE_SOURCES[subIdx] : null;
            const dest = typeof EDGE_DESTS !== 'undefined' ? EDGE_DESTS[subIdx] : null;
            return { faction: "british", cardName: "Divide and Rule", cardIndex: 3, fromNode: src, toNode: dest, isTrade: false };
          }
          case "Force March": {
            const src = typeof EDGE_SOURCES !== 'undefined' ? EDGE_SOURCES[subIdx] : null;
            const dest = typeof EDGE_DESTS !== 'undefined' ? EDGE_DESTS[subIdx] : null;
            return { faction: "british", cardName: "Force March", cardIndex: 4, fromNode: src, toNode: dest, isTrade: false };
          }
          case "Princely States":
            return { faction: "british", cardName: "Princely States", cardIndex: 5, toNode: subIdx, isTrade: false };
          case "British Power":
            return { faction: "british", cardName: "Wall Breach", cardIndex: 0, toNode: null, isTrade: false, isPower: true };
          case "Draw Wall Breach":
            return { faction: "british", cardName: "Wall Breach", cardIndex: 0, tradeTargetCard: subIdx, isTrade: true };
          case "Draw Highlanders":
            return { faction: "british", cardName: "Highlanders", cardIndex: 1, tradeTargetCard: subIdx, isTrade: true };
          case "Draw Royal Navy":
            return { faction: "british", cardName: "Royal Navy", cardIndex: 2, tradeTargetCard: subIdx, isTrade: true };
          default:
            return null;
        }
      }
      offset += size;
    }
    return null;
  }

  const TDMCTS = {
    MCTSNode,
    ONNXModelWrapper,
    MCTS,
    decodeMoveGeometry: (state, move) => (new MCTS(null)).decodeMoveGeometry(state, move),
    getCardInfoForMove
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = TDMCTS;
  } else {
    global.TDMCTS = TDMCTS;
    Object.assign(global, TDMCTS);
  }
})(typeof window !== 'undefined' ? window : this);
