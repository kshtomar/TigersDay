# 🐅 Tiger’s Day – Anglo-Mysore Wars (1767 – 1799)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![ONNX Runtime Web](https://img.shields.io/badge/ONNX_Runtime-WebAssembly-005CED.svg)](https://onnxruntime.ai/)
[![WebRTC](https://img.shields.io/badge/WebRTC-PeerJS_P2P-orange.svg)](https://peerjs.com/)
[![CI](https://github.com/kshtomar/TigersDay/actions/workflows/ci.yml/badge.svg)](https://github.com/kshtomar/TigersDay/actions)
[![Tests](https://img.shields.io/badge/Tests-134%20Passing-brightgreen.svg)](./TESTS.md)
[![PWA](https://img.shields.io/badge/PWA-Offline%20Ready-blueviolet.svg)](./public/manifest.json)

Welcome to **Tiger’s Day**, a strategic, asymmetric board wargame simulating the historical Anglo-Mysore Wars fought between **Tipu Sultan** (*The Tiger of Mysore*) and the British East India Company under commanders such as **Lord Cornwallis** and **General Harris**.

The project is an end-to-end full-stack artificial intelligence and game-engineering system. It pairs a **PyTorch reinforcement self-play research environment** with a **100% client-side WebAssembly game engine** running an **AlphaZero-inspired Deep Neural Network + Monte Carlo Tree Search (MCTS)** in the browser, along with **real-time Peer-to-Peer (P2P) WebRTC multiplayer**.

---

## 📑 Table of Contents

1. [Architectural Overview](#-architectural-overview)
2. [The Game: Rules & Mechanics](#-the-game-rules--mechanics)
   - [Map & Geography](#map--geography)
   - [Factions & Setup](#factions--setup)
   - [Turn Sequence & Impulse System](#turn-sequence--impulse-system)
   - [Combat & Siege Resolution](#combat--siege-resolution)
   - [Stochastic Luck Mechanics](#stochastic-luck-mechanics)
   - [Card Codex & Trading System](#card-codex--trading-system)
   - [Algebraic Replay Notation](#algebraic-replay-notation)
3. [AI & Neural Network Architecture](#-ai--neural-network-architecture)
   - [148-Bit Binary State Representation](#148-bit-binary-state-representation)
   - [959-Dimensional Action Space](#959-dimensional-action-space)
   - [Factorized Policy Head Architecture](#factorized-policy-head-architecture)
   - [Monte Carlo Tree Search (MCTS) Engine](#monte-carlo-tree-search-mcts-engine)
   - [Client-Side ONNX WebAssembly Inference](#client-side-onnx-webassembly-inference)
4. [Modernized Frontend & Visual Experience](#-modernized-frontend--visual-experience)
   - [Zero-Scroll Responsive Layout](#zero-scroll-responsive-layout)
   - [10 Bespoke Historical Themes](#10-bespoke-historical-themes)
   - [5 Modular Board Unit Styles](#5-modular-board-unit-styles)
   - [Interactive Battle Cards & Wax Seal Badges](#interactive-battle-cards--wax-seal-badges)
   - [Chess.com-Style Live Moves Notation & History Stepper](#chesscom-style-live-moves-notation--history-stepper)
   - [Zero-Scroll Command Settings Modal](#zero-scroll-command-settings-modal)
   - [Animated Siege Clash & Net Strength Tag](#animated-siege-clash--net-strength-tag)
5. [Real-Time P2P WebRTC Multiplayer](#-real-time-p2p-webrtc-multiplayer)
6. [Repository Directory Structure](#-repository-directory-structure)
7. [Quickstart & Execution Guide](#-quickstart--execution-guide)
   - [Option A: Pure Client-Side Static Hosting (Zero Server)](#option-a-pure-client-side-static-hosting-zero-server)
   - [Option B: Local FastAPI + AI Server](#option-b-local-fastapi--ai-server)
   - [Option C: Serverless Deployment (Vercel)](#option-c-serverless-deployment-vercel)
8. [AI Training, Arena & Research Workflows](#-ai-training-arena--research-workflows)
   - [Single-Thread Curriculum Training](#1-single-thread-curriculum-training)
   - [Multi-Process Parallel Training](#2-multi-process-parallel-training)
   - [Tournament Arena & Model Evaluation](#3-tournament-arena--model-evaluation)
   - [Exporting PyTorch Checkpoints to ONNX](#4-exporting-pytorch-checkpoints-to-onnx)
   - [Visualizing Model Weights (Heatmaps)](#5-visualizing-model-weights-heatmaps)
   - [Opening Book & Replay Analysis](#6-opening-book--replay-analysis)
9. [API Reference (FastAPI Backend)](#-api-reference-fastapi-backend)
10. [Quality Assurance, Automated Testing & CI/CD Pipeline](#-quality-assurance-automated-testing--cicd-pipeline)
11. [Roadmap & Future Directions](#-roadmap--future-directions)
12. [Historical Context](#-historical-context)

---

## 🏛️ Architectural Overview

Tiger's Day is engineered as a **dual-stack system**:
1. **Python Core (`game/`, `ai/`, `server.py`)**: High-performance NumPy vectorized rules engine, PyTorch AlphaZero neural training loops, curriculum learning, multiprocessing self-play, and tournament evaluation arena.
2. **JavaScript WebAssembly Client (`public/`, `public/js/`)**: Pure JavaScript 1:1 mirror of the state vector and rule engine, running `onnxruntime-web` (WebAssembly with SIMD) to execute neural inference and MCTS directly in the player's browser at 60 FPS without requiring a GPU server.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │               OFFLINE AI RESEARCH PIPELINE              │
                     │  ai/train.py  ──>  ai/models/alphatigerv13.pt           │
                     │  ai/multitrain.py (Curriculum Self-Play)                │
                     │  ai/arena.py (Tournament & Disagreement Analytics)      │
                     └────────────────────────────┬────────────────────────────┘
                                                  │ ai/onnx.py (Export)
                                                  ▼
                                       public/alphatiger.onnx
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
   ┌───────────────────────────┐                                     ┌───────────────────────────┐
   │    PYTHON BACKEND SERVER  │                                     │ 100% CLIENT-SIDE BROWSER  │
   │  server.py / api/index.py │                                     │  public/index.html & JS   │
   ├───────────────────────────┤                                     ├───────────────────────────┤
   │ • FastAPI / Uvicorn       │                                     │ • Pure JS GameState (148D)│
   │ • PyTorch or ONNX Runtime │                                     │ • In-Browser MCTS Engine  │
   │ • Server MCTS Eval Engine │                                     │ • ONNX WebAssembly (WASM) │
   │ • Replay Algebraic Parser │                                     │ • SVG Interactive Board   │
   │ • Vercel Serverless Ready │                                     │ • Stockfish-Style Eval Bar│
   └─────────────┬─────────────┘                                     └─────────────┬─────────────┘
                 │                                                                 │
                 │ HTTP API (/api/init, /api/play-move, /api/eval-step)            │ PeerJS WebRTC
                 └─────────────────────────────────────────────────────────────────┼───────────────┐
                                                                                   ▼               ▼
                                                                            [Browser 1] <=====> [Browser 2]
                                                                                (P2P Room: TIGER-ABCD)
```

---

## 📖 The Game: Rules & Mechanics

### Map & Geography
The board represents Southern India during the late 18th century, modeled as an undirected planar graph containing **25 territories (nodes)** and **84 directed edges (42 bidirectional connections)**:

* **5 Key Cities (`KEYS`):** *Bombay, Hyderabad, Madras, Seringapatam, Coimbatore*.
* **10 Coastal Territories (`COASTAL`):** *Bombay, Madras, Masulipatam, Goa, Mangalore, Mahé, Pondicherry, Ramnad, Travancore, Ceylon*.
* **Inland Territories:** *Satara, Poona, Raichur, Darwar, Anantapur, Chitaldoorg, Bangalore, Vellore, Erode, Trichy, Alwaye, Dindigul*.

### Factions & Setup
* **British East India Company (Attacker):**
  * Starts with **4 Fresh Armies** deployed at *Bombay, Hyderabad, Madras, and Travancore*.
  * Has access to 6 British tactical cards.
* **Sultanate of Mysore (Defender):**
  * Starts with **9 Forts** guarding strategic strongholds: *Darwar, Chitaldoorg, Mangalore, Bangalore, Seringapatam, Erode, Coimbatore, Mahé, and Dindigul*.
  * Has access to 6 Mysore tactical cards.

### Turn Sequence & Impulse System
The game lasts up to **4 Turns** (representing the 1st, 2nd, 3rd, and 4th Anglo-Mysore Wars). Within each turn, play alternates across three consecutive impulse phases (`to_move`):

1. **Phase 0 (`British Move`):**
   * The British player chooses **one Fresh Army** to move along an adjacent edge to an empty node or an enemy fort.
   * If the British moves to an empty territory, the army becomes **Tired**.
   * Alternatively, the British may choose to **Tire in place** without moving.
2. **Phase 1 (`Mysore Card`):**
   * The Mysore player plays a card from their hand for its tactical effect, commits it to an active battle, trades an active card to recover an exhausted card, or passes.
3. **Phase 2 (`British Card`):**
   * The British player plays a card for its tactical effect, commits it to an active battle, trades an active card to recover an exhausted card, or passes.
   * At the end of Phase 2, any pending combat is resolved.
   * If all British armies are **Tired** and Turn < 4, a **Turn Refresh** occurs:
     * `turn` advances by 1.
     * All Tired armies recover to become Fresh armies.
     * All exhausted cards are replenished back into active hands.
     * Stochastic luck states are cleared.

#### Victory Conditions
* **British Victory:** Occupy **all 5 Key Cities** (*Bombay, Hyderabad, Madras, Seringapatam, Coimbatore*) with British armies (fresh or tired) at any moment.
* **Mysore Victory:** Survive through Turn 4 without the British occupying all 5 Key Cities, or reduce the British forces such that they cannot win.

---

### Combat & Siege Resolution
When a British army moves into a territory defended by a Mysore Fort, a **Battle State** is triggered (`state.attacker = src`, `state.defender = dest`):

$$\text{Attacker Strength} = \sum (\text{Adjacent British Armies to Defender})$$
$$\text{Defender Strength} = \sum (\text{Adjacent Mysore Forts to Defender})$$
$$\text{Net Strength} = \text{Attacker Strength} - \text{Defender Strength} + (\text{British Card Strength} - \text{Mysore Card Strength})$$

* **British Victory ($\text{Net Strength} > 0$):**
  * The British capture the contested territory.
  * The defending Fort is destroyed and replaced by the Attacking Army (which becomes Tired).
  * The Sultanate of Mysore suffers casualties: **Mysore must discard 1 random active card** (`mluck += 1`).
* **Mysore Victory ($\text{Net Strength} \le 0$):**
  * The Mysore Fort successfully withstands the siege.
  * The attacking British army retreats or remains in place (Tired).
  * The British East India Company suffers casualties: **British must discard 1 random active card** (`bluck += 1`).

---

### Stochastic Luck Mechanics
Whenever a battle is won/lost or an ability such as *Cavalry Raid* is triggered, a faction accumulates pending random discards (`bluck` or `mluck`).
* The engine resolves luck by branching uniformly across all currently active cards held by the penalized player.
* In MCTS and self-play, luck nodes are handled via chance node rollouts, preserving stochastic expectations while keeping state transitions deterministic when tracking the specific luck trajectory.

---

### Card Codex & Trading System

Each faction holds **6 unique cards** with intrinsic battle values `[3, 2, 2, 1, 1, 1]`:

#### Sultanate of Mysore Hand
| Card Name | Value | Type | Tactical Effect |
| :--- | :---: | :--- | :--- |
| **Iron Rockets** | 3 | Battle / Trade | +3 Battle Strength. High-value card used to defend critical forts or trade for lower cards. |
| **Sepoy Mutiny** | 2 | Operation | Target and eliminate any British army (fresh or tired) **not** located in a Key City. |
| **French Alliance** | 2 | Operation | Deploy a new Fort on any empty territory that is adjacent to an existing Mysore Fort. |
| **Monsoon** | 1 | Operation | Target any Fresh British army and flip it to **Tired**, stalling British mobility. |
| **Cavalry Raid** | 1 | Operation | Launch a raid behind enemy lines: forces the British to discard **1 random card** (`bluck += 1`). |
| **Sea Trade** | 1 | Operation | Move any Fort located on a Coastal node to any empty territory on the board. |

#### British East India Company Hand
| Card Name | Value | Type | Tactical Effect |
| :--- | :---: | :--- | :--- |
| **Wall Breach** | 3 | Battle / Trade | +3 Battle Strength. Used to crack heavily defended Mysore forts or trade for lower cards. |
| **Highlanders** | 2 | Operation | Deploy a brand-new **Fresh Army** on any empty Coastal territory. |
| **Royal Navy** | 2 | Operation | Transport any Army (fresh or tired) to any legal Coastal destination. |
| **Divide and Rule** | 1 | Operation | Move an enemy Mysore Fort that is **not** in a Key City along an edge into an adjacent empty territory. |
| **Force March** | 1 | Operation | Move a **Tired Army** along an edge into an empty node or into battle against a fort. |
| **Princely States** | 1 | Operation | Deploy a new **Tired Army** directly into any empty **Key City**. |

#### Card-to-Card Trading Rules
A player can sacrifice an active higher-value card to reclaim an exhausted card:
* **Value 3 Card** (*Iron Rockets* or *Wall Breach*): May be traded to reclaim any exhausted card 1 through 5.
* **Value 2 Cards** (*Sepoy Mutiny / French Alliance* or *Highlanders / Royal Navy*): May be traded to reclaim any exhausted card 3 through 5.
* **Value 1 Cards**: Cannot be used to trade.

---

### Algebraic Replay Notation
The engine generates standardized algebraic move strings formatted in `game/replay.py`:

* `mad>pdc` : Movement from Madras (`mad`) to empty Pondicherry (`pdc`).
* `srp x blr` : Siege/attack from Seringapatam (`srp`) against Bangalore fort (`blr`).
* `SM:trv` : Mysore plays Sepoy Mutiny (`SM`) targeting British army in Travancore (`trv`).
* `FA:dwr` : Mysore plays French Alliance (`FA`) deploying a fort in Darwar (`dwr`).
* `RN:bom>goa` : British plays Royal Navy (`RN`) moving army from Bombay (`bom`) to coastal Goa (`goa`).
* `WB:x` : British commits Wall Breach (`WB`) for combat strength in battle.
* `IR:SM` : Mysore trades Iron Rockets (`IR`) to draw back Sepoy Mutiny (`SM`).
* `pass` : Player passes their card impulse.
* `+` : Turn refresh transition (Turn 1 $\to$ 2 $\to$ 3 $\to$ 4).
* `# 1-0` / `# 0-1` : Terminal game result (British Win: `1-0`, Mysore Win: `0-1`).

---

## 🧠 AI & Neural Network Architecture

The artificial intelligence powering Tiger's Day is an **AlphaZero-style architecture** designed specifically for asymmetric turn-based wargaming with imperfect card availability and stochastic chance events.

### 148-Bit Binary State Representation
The complete state of the game is compressed into a compact **148-bit boolean vector** (`GAME_VECTOR_LENGTH = 148`):

```
 0        6       12                                                        87   91   94   98       123      148
┌────────┬────────┬─────────────────────────────────────────────────────────┬────┬────┬────┬────────┬────────┐
│ Brit.  │ Mysore │                   Board Nodes (25 × 3)                  │Turn│Who │Comb│Attack. │Defend. │
│ Cards  │ Cards  │  [0..24]: Fresh Army (1 bit), Tired (1 bit), Fort (1 bit)│1..4│Move│Str.│ 1-hot  │ 1-hot  │
│ (6b)   │ (6b)   │                        (75 bits)                        │(4b)│(3b)│(4b)│ (25b)  │ (25b)  │
└────────┴────────┴─────────────────────────────────────────────────────────┴────┴────┴────┴────────┴────────┘
```

* **Bits `0..5`:** British active cards status (6 boolean bits).
* **Bits `6..11`:** Mysore active cards status (6 boolean bits).
* **Bits `12..86`:** 25 territories $\times$ 3 mutually-exclusive states (*Fresh Army, Tired Army, Fort, or all-zero for Empty*) = 75 bits.
* **Bits `87..90`:** One-hot encoding of current turn (Turns 1 to 4).
* **Bits `91..93`:** One-hot encoding of who moves (`British Move`, `Mysore Card`, `British Card`).
* **Bits `94..97`:** One-hot encoding of combat strength committed by Mysore (0, 1, 2, or 3).
* **Bits `98..122`:** One-hot encoding of attacking territory (25 bits, or all-zero if no battle).
* **Bits `123..147`:** One-hot encoding of defending territory (25 bits, or all-zero if no battle).

---

### 959-Dimensional Action Space
All possible actions across every phase map into a unified **959-element action vector** (`MOVE_VECTOR_LENGTH = 959`):

| Phase | Action Name | Dimension | Target Type |
| :--- | :--- | :---: | :--- |
| **Phase 0: British Move** (111) | Edge Movement | 86 | Edge (`src -> dest`) |
| | Tire in Place | 25 | Territory node (`0..24`) |
| **Phase 1: Mysore Card** (351) | Sepoy Mutiny | 25 | Territory node |
| | French Alliance | 25 | Territory node |
| | Monsoon | 25 | Territory node |
| | Cavalry Raid | 1 | Global blank action |
| | Sea Trade | 250 | Coastal origin $\times$ Destination ($25 \times 10$) |
| | Mysore Power | 6 | Card committed in combat |
| | Draw Iron Rockets | 6 | Card trade (trade card 0 for 1..5) |
| | Draw Sepoy Mutiny | 6 | Card trade (trade card 1 for 3..5) |
| | Draw French Alliance | 6 | Card trade (trade card 2 for 3..5) |
| | Pass Mysore | 1 | Blank pass action |
| **Phase 2: British Card** (497) | Highlanders | 25 | Coastal territory node |
| | Royal Navy | 250 | Origin node $\times$ Coastal destination ($25 \times 10$) |
| | Divide and Rule | 86 | Edge (`fort -> empty`) |
| | Force March | 86 | Edge (`tired -> dest`) |
| | Princely States | 25 | Key City territory node |
| | British Power | 6 | Card committed in combat |
| | Draw Wall Breach | 6 | Card trade (trade card 0 for 1..5) |
| | Draw Highlanders | 6 | Card trade (trade card 1 for 3..5) |
| | Draw Royal Navy | 6 | Card trade (trade card 2 for 3..5) |
| | Pass British | 1 | Blank pass action |

---

### Factorized Policy Head Architecture
Standard neural policy heads output an unconstrained dense vector over 959 logits. However, large coastal actions (*Royal Navy* and *Sea Trade*) account for over 500 logits ($250 + 250$), which can lead to overparameterization and slow convergence.

`AlphaTiger` implements **Action Space Factorization** in `ai/neural.py`:
* Instead of outputting 250 monolithic logits for *Royal Navy*, the policy network outputs 25 source logits and 25 destination logits.
* The joint probability logit is calculated via additive decomposition:
  $$\text{Logit}_{\text{RoyalNavy}}(src, dest) = \text{Logit}_{RN\_src}(src) + \text{Logit}_{RN\_dest}(dest)$$
* The same factorization is applied to *Sea Trade*, reducing policy output dimensionality by 400 parameters and improving sample efficiency during self-play training.

```
       Input (148D State Vector)
                  │
          Linear(148 -> 256) + ReLU
                  │
          Linear(256 -> 256) + ReLU
                  │
          Linear(256 -> 256) + ReLU
         ─────────┬─────────
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
Value Head                Policy Head
Linear(256 -> 64) + ReLU   Linear(256 -> 256) + ReLU
Linear(64 -> 1) + Tanh     Linear(256 -> Factorized Size)
     │                         │
Scalar Value [-1.0, +1.0]  Decompose & Broadcast Factorized RN/ST Logits
                           Full 959-Dimensional Policy Logits
```

---

### Monte Carlo Tree Search (MCTS) Engine
The MCTS algorithm (`ai/mcts.py` in Python and `public/js/mcts.js` in JavaScript) incorporates modern techniques:
* **PUCT Selection Formula:**
  $$\text{PUCT}(s, a) = Q(s, a) + c(s) \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
  where $c(s) = 1.25 + \ln\left(\frac{N(s) + c_{puct}}{c_{puct}}\right)$ with $c_{puct} = 800$.
* **Dirichlet Exploration Noise:** Blends Dirichlet noise ($\alpha = 0.5$, $\epsilon = 0.25$) at the root node during training to ensure diverse opening exploration.
* **Playout Cap Randomization:** During self-play (`ai/train.py`), 25% of moves are evaluated with full simulations (weight $p=1.0$) and 75% with 10% simulations (weight $p=0.0$), accelerating self-play generation without corrupting policy targets.
* **Subtree Reuse (`update_root`):** Preserves the search tree across both player moves and stochastic luck transitions.

---

### Client-Side ONNX WebAssembly Inference
* The trained PyTorch model is converted to an optimized ONNX model (`ai/onnx.py`).
* In the browser, `onnxruntime-web` runs WebAssembly with SIMD acceleration (`public/alphatiger.onnx`, ~1.5 MB).
* **INT8 Dynamic Quantization:** An ultra-compact quantized model [`public/alphatiger.quant.onnx`](./public/alphatiger.quant.onnx) (431 KB, ~72% reduction) is provided for instant loading and mobile performance.
* **Fast-Path Opening Book:** Integrates [`public/opening_book.json`](./public/opening_book.json) extracted from high-winrate replay lines to bypass tree search on early turns.
* The browser runs the full MCTS loop asynchronously, providing:
  * Zero server roundtrips and zero latency.
  * A real-time **Stockfish-style Evaluation Bar** showing who has the tactical advantage.
  * **Top Engine Lines (Principal Variations)** displayed dynamically in the UI.

---

## 🎨 Modernized Frontend & Visual Experience

The frontend web application (`public/`) delivers a modern, high-contrast, zero-scroll wargaming interface that combines rich historic aesthetics with fluid real-time interactivity:

### Zero-Scroll Responsive Layout
* **Unified Single-Screen Viewport:** The layout dynamically auto-fits standard laptops, desktop monitors, and tablets without requiring vertical scrolling, keeping the map, card decks, algebraic notation feed, and match status perpetually in view.
* **Centered Heraldic Standards:** Both Mysore (`🐅 MYSORE` in emerald green) and the British (`🦁 BRITISH` in imperial crimson) feature center-aligned faction banners with diagonal emblem accents in their respective card columns.

### 10 Bespoke Historical Themes
Players can select from 10 distinct, contrasting visual palettes directly from the Settings modal:
1. **Deccan Imperial (Default):** Warm historic parchment, antique gold (`#e5a93c`), and brass accents.
2. **Midnight Tiger:** Stealth dark mode with deep obsidian (`#070d18`) and glowing neon amber (`#f59e0b`).
3. **Royal Velvet:** Regal navy sapphire (`#0c1938`) with rich gold leaf (`#eab308`).
4. **Emerald Sultan:** Mysorean court aesthetic with deep jade (`#052620`), forest emerald (`#10b981`), and polished brass.
5. **Monsoon Mist:** Storm slate (`#182d38`), oceanic dark teal, and vibrant coastal paths (`#38bdf8`).
6. **Desert Rajput:** Sandstone (`#e4b878`), terracotta red (`#9a3412`), and azure seas (`#0284c7`).
7. **Cyber War-Room:** Tactical HUD aesthetic with void black (`#020617`), slate panels, and cyan neon edges (`#06b6d4`).
8. **Sepia Archive:** 18th-century copperplate engraving on warm aged cotton paper (`#f5eee0`) with umber ink (`#4a3820`).
9. **Crimson Crown:** British imperial regimental ruby (`#f43f5e`), dark charcoal velvet, and silver highlights.
10. **Ivory & Onyx:** Minimalist luxury monochrome with high-contrast pearl whites, deep blacks, and gold trim.

### 5 Modular Board Unit Styles
Players can customize how armies, forts, and strongholds render on the map:
1. **Tactical Tokens (Default):** 3D-embossed circular military medallions, stone citadels, and geometric star forts.
2. **Classic Squares:** Original wargame geometric red and green squares and diamonds with diagonal tire slashes.
3. **Regimental Crests:** Heraldic battle shields, crossed steel swords, and fortress tower battlements.
4. **Minimalist Counters:** Sleek circular counters with clean typographic star (`★`) and crescent (`☽`) faction glyphs.
5. **Antique Miniatures:** Sculpted 3D pewter and bronze relief figurines with metallic edge highlights.

### Interactive Battle Cards & Wax Seal Badges
* **Tactile Parchment Cards:** Styled with diagonal faction cuts matching the column headers and clean title labels.
* **Direct Wax Seal Action:** Tactile wax seals (+1, +2, +3) can be clicked directly during combat to commit battle strength without needing secondary menus.

### Chess.com-Style Live Moves Notation & History Stepper
* **Dedicated Moves Feed:** A dedicated right-hand panel recording every troop move, card operation, siege combat, and turn refresh in standard algebraic notation.
* **Historical Stepper Controls:** Inspect prior moves step-by-step (`⏮ First`, `◀ Prev`, `Next ▶`, `⏭ Live`) with an automatic read-only review banner, allowing players to review opening sequences and jump back to the live game seamlessly.

### Zero-Scroll Command Settings Modal
* **2-Column Master Dashboard:** Replacing narrow side drawers with a centered modal dialog featuring a wide, zero-scroll layout.
* **Complete Configuration at a Glance:** Directly adjust themes, board unit styles, match opponent mode (Pass & Play, WebAssembly AI, P2P Multiplayer, AI Spectator), Stockfish-style evaluation bar, MCTS simulation depth slider (250 – 1,000,000 rollouts), and binary state save/load strings.

### Bottom Map & Decks AI Analysis Dock (`#bottom-analysis-dock`)
* **Unified Analysis Dock:** The Stockfish-style winrate evaluation progress bar and multi-PV candidate move lines are integrated into a single glassmorphic dock situated directly beneath the Mysore deck, board map, and British deck.
* **Proportional Board & Cards Scaling:** Dynamic flexbox sizing smoothly scales the board SVG and player card decks proportionally to fit within the viewport without generating window scrollbars or encroaching into the notation sidebar.
* **Horizontal Scrolling Rail:** Candidate move cards are arranged in a horizontal scrolling rail, preserving a minimal vertical footprint and keeping the full map visible at all times.

### Top-K AI Candidate Moves & Variation Lines
* **Configurable Positive Integer $K$:** Configure any positive integer $k \ge 1$ in Settings; dynamic golden-ratio color assignment and runtime SVG marker generation support arbitrary $k > 5$, while positions with fewer legal moves than $k$ gracefully render all valid lines without placeholder errors.
* **Principal Variation (PV) Forward Simulation:** Traces candidate variations up to 10 plies deep using algebraic notation.
* **Stochastic Luck State Frontiers:** Forward search halts when encountering luck state boundaries (probabilistic battles or card draws), clearly marking variations with `[🎲 Battle: <Territory>]` or `[🎲 Luck Roll]`.
* **Visit Count Badges:** Candidate cards prominently display MCTS visit counts via dedicated `.candidate-visits-badge` pills (`210 visits`).

### Player Deck Card AI Number Badges & Visual Spatial Correspondence
* **Clean Number-Only Badges:** When tactical cards are recommended by the AI, the corresponding cards in the player's deck display a high-contrast rank number badge (`#1`, `#2`, etc.) positioned at `top: 4px; right: 36px` beside the wax seal without cluttering card art or descriptions. For card trade-in moves, `#rank ↺` designates the target card to reclaim.
* **Bi-Directional Spatial Correspondence:** Hovering a card in hand pulses the corresponding tactical arrow and destination territory ring on the board map; hovering candidate cards in the dock pulses both the card in hand and the map elements.
* **Clean Minimalist Command Iconography:** Purged distracting modern emoticons (`🧠`, `📜`, `💡`, `🎲`) and redundant `[TOP K]` boxes from headers, HUD pills, and notation sidebars for an authentic 18th-century military command aesthetic.

### Animated Siege Clash & Net Strength Tag
* **Dynamic Combat Marker:** When a battle is triggered, an animated crossed-swords shield appears along the combat vector between the attacking army and defending fort.
* **Live Net Strength Display:** Displays real-time net strength (`+N` in British red or `-N` in Mysore green) taking into account adjacent armies, adjacent forts, and committed card values.

### Zero-Dependency Procedural Web Audio Engine
* Built with the native Web Audio API in [`public/js/sound.js`](./public/js/sound.js).
* Synthesizes authentic period soundscapes without external audio asset downloads:
  * Resonant artillery cannon blasts with exponential frequency decay and noise bursts for siege clashes.
  * Percussive snare drum march cadences for army troop movements.
  * Crisp parchment card plays, wax seal cracks, and faction-specific victory fanfares (`playWin` / `playVictory`).
  * Mournful descending minor triad brass motifs with funeral drum thuds for defeat (`playDefeat`).
  * Metallic sword-yielding slides, two-tone white-flag surrender bugles, and hollow rim strikes for resignation (`playResign`).
  * Master volume slider and instant mute toggle in the Settings modal.

### 100% Offline Progressive Web App (PWA)
* Includes a standalone Web App Manifest ([`public/manifest.json`](./public/manifest.json)) and a cache-first Service Worker ([`public/sw.js`](./public/sw.js)).
* Allows complete offline installation and play across iOS, Android, macOS, Windows, and Linux with instant launch speeds and zero cloud dependencies.

### Accessibility (a11y) & Desktop Hotkeys
* Full WCAG compliance: board territories receive `tabindex="0"`, `role="button"`, dynamic ARIA labels, and respond to `Enter` and `Space`.
* Global hotkeys for rapid desktop gameplay:
  * `Escape`: Deselect unit / close modal drawers.
  * `ArrowLeft` / `ArrowRight`: Step backwards/forwards through algebraic move history.
  * `Z`: Undo move.
  * `R`: Rest active army in place.
  * `P`: Pass active card phase.
  * Automated input element detection prevents hotkeys from firing while typing in text inputs.

---

## 🌐 Real-Time P2P WebRTC Multiplayer

The multiplayer architecture (`public/js/multiplayer.js`) enables direct browser-to-browser connections with **zero backend server infrastructure**:

* **Signaling & P2P:** Uses **PeerJS** over public STUN servers to negotiate WebRTC data channels.
* **Room Codes:** Hosting a game generates an ephemeral 4-character room code (e.g., `TIGER-K7B2`). A second player enters this code to connect instantly.
* **State Synchronization Protocol:**
  * When a player makes a move, the client sends a `MOVE` packet containing:
    * `moveIdx`: The integer action (0–958).
    * `luckTrajectory`: Exact branch outcomes for any luck events triggered, ensuring deterministic lockstep.
    * `stateStr`: Full 148-bit verification string to detect and correct any state desynchronization.
  * Handshake and reset packets (`HANDSHAKE`, `SYNC_STATE`, `RESET_GAME`, `RESIGN`) handle session management.

---

## 📁 Repository Directory Structure

```text
TigersDay/
├── .github/
│   └── workflows/
│       └── ci.yml                   # Multi-stage CI pipeline (Lint, Unit, Integration, Matrix)
├── ai/                              # Reinforcement Learning & Neural Network Pipeline
│   ├── models/                      # Checkpoints (.pt) and exported ONNX model
│   │   ├── alphatiger.onnx          # WebAssembly ONNX model (active web model, 1.5MB)
│   │   ├── alphatigerv13.pt         # Latest trained PyTorch production checkpoint
│   │   └── alphatigerv7.pt..v12.pt  # Historical evolutionary training checkpoints
│   ├── arena.py                     # Head-to-head model evaluation tournament & analytics
│   ├── heatmap.py                   # Layer weight visualization using Seaborn & Matplotlib
│   ├── mcts.py                      # Vectorized Python Monte Carlo Tree Search
│   ├── multitrain.py                # Multi-worker parallel self-play training script
│   ├── neural.py                    # PyTorch AlphaTiger model & ONNXRuntime CPU wrapper
│   ├── onnx.py                      # Exporter script (PyTorch -> ONNX & INT8 quantization)
│   ├── opening_book.py              # Opening book extraction from high-tier game replays
│   └── train.py                     # Single-thread training loop with Curriculum Learning
│
├── api/                             # FastAPI Backend & Serverless API
│   ├── app.py                       # Unified FastAPI app factory, Pydantic schemas & LRU cache
│   └── index.py                     # Vercel serverless application handler
│
├── game/                            # Core Game Mechanics & Rules Engine (Python)
│   ├── constants.py                 # Graph topology, edges, cards, indices, 959 action space
│   ├── engine.py                    # Legal move masking & move description dictionary
│   ├── replay.py                    # Algebraic notation interpreter & opening book parser
│   ├── state.py                     # 148-bit GameState vector management & serialization
│   └── updater.py                   # O(1) action dispatch, combat math, & luck branching
│
├── public/                          # 100% Client-Side Web Application & PWA
│   ├── js/
│   │   ├── engine.js                # O(1) JavaScript rule engine, combat math, & repetition
│   │   ├── mcts.js                  # Browser MCTS & ONNX WebAssembly inference controller
│   │   ├── multiplayer.js           # PeerJS WebRTC P2P multiplayer with heartbeat & reconnect
│   │   ├── sound.js                 # Zero-dependency procedural Web Audio soundscape engine
│   │   ├── state.js                 # JavaScript GameState & 148D vector port
│   │   └── ui/
│   │       └── themes.js            # 10 bespoke themes & 5 modular unit token styles
│   ├── alphatiger.onnx              # Static WebAssembly neural network weights (1.5MB)
│   ├── alphatiger.quant.onnx        # INT8 dynamically quantized WebAssembly weights (431KB)
│   ├── index.html                   # Zero-scroll responsive UI, interactive SVG map, settings
│   ├── manifest.json                # PWA installation manifest specification
│   ├── opening_book.json            # Fast-path opening book lookup dictionary
│   ├── script.js                    # UI coordinator, move stepper, hotkeys, a11y handlers
│   ├── style.css                    # Responsive CSS design system across 5 device breakpoints
│   └── sw.js                        # Cache-first offline Service Worker
│
├── tests/                           # Complete Automated Test Suite (49 Tests)
│   ├── unit/                        # Python Unit Tests (State, Updater, MCTS)
│   │   ├── test_state.py            # Bit 94, invariants, serialization roundtrip, validation
│   │   ├── test_updater.py          # 959 action dispatch, battle2 strength, state transitions
│   │   └── test_mcts.py             # Opening book loading and querying
│   ├── integration/                 # Python Integration Tests (Parity, API, Gameplay)
│   │   ├── test_parity.py           # Cross-engine Python <-> JS byte parity
│   │   ├── test_api.py              # FastAPI endpoints (/init, /load-state, /play-move, /eval)
│   │   └── test_gameplay.py         # Multi-turn sequential play, luck resolution, undo rollback
│   └── js/                          # Node.js Test Battery (Engine & Frontend UI)
│       ├── engine.test.js           # Client engine invariants, 959 dispatch, repetition, sound
│       └── ui.test.js               # DOM structure, 6-device auto-sizer, CSS @media, a11y, PWA
│
├── logs/                            # Replay and tournament logs
├── package.json                     # Node test runner (`npm test`) and linter (`npm run lint`)
├── requirements.txt                 # Production & serverless dependencies (NumPy pinned)
├── requirements-dev.txt             # Full local development & PyTorch training dependencies
├── server.py                        # Local FastAPI development server using api.app
├── vercel.json                      # Vercel deployment and routing configuration
├── FUTURE_IMPROVEMENTS.md           # Completed work audit & next-gen roadmap
├── TESTS.md                         # Detailed catalog of all 49 automated test cases
└── README.md                        # Comprehensive documentation
```

---

## 🚀 Quickstart & Execution Guide

### Option A: Pure Client-Side Static Hosting (Zero Server)
The client-side web application can run without a Python backend or database.

#### Using Python:
```bash
python3 -m http.server 8000 --directory public
```

#### Using Node / npx:
```bash
npx serve public -p 8000
```

Navigate to **`http://localhost:8000`** in any modern web browser. The application will load the board, initialize `onnxruntime-web`, and allow you to play immediately against the local AI or host a P2P multiplayer match.

---

### Option B: Local FastAPI + AI Server
Running `server.py` starts a local FastAPI server with access to PyTorch GPU/CPU inference, server-side MCTS, and algebraic notation generators.

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   # If you plan to use PyTorch checkpoints (.pt):
   pip install -r requirements-dev.txt
   ```

3. **Run the server:**
   ```bash
   python server.py --mode human_vs_ai --human british --port 8000
   ```

#### `server.py` Command-Line Arguments
| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--mode` | string | `human_vs_ai` | Match mode: `human_vs_ai`, `human` (pass & play), or `ai_vs_ai` (spectator). |
| `--human` | string | `british` | Which faction the human plays in `human_vs_ai` (`british` or `mysore`). |
| `--ckpt` | string | `ai/models/alphatigerv13.pt` | Path to the default AlphaTiger PyTorch checkpoint. |
| `--ckpt_british`| string | `None` | Optional separate checkpoint specifically for the British AI. |
| `--ckpt_mysore` | string | `None` | Optional separate checkpoint specifically for the Mysore AI. |
| `--sims` | int | `400` | Number of MCTS rollout simulations per decision turn. |
| `--port` | int | `8000` | Web server listening port. |
| `--threshold` | float | `0.10` | Frequency threshold for logging top candidate moves to stdout. |

---

### Option C: Serverless Deployment (Vercel)
The repository includes a configured `vercel.json` and `api/index.py` for deployment on Vercel:
* `vercel.json` routes `/api/(.*)` to the FastAPI app at `api/index.py`, while serving all static assets from `public/`.
* `api/index.py` uses `onnxruntime` CPU execution and loads `public/alphatiger.onnx` to provide fast, sub-second responses within serverless memory constraints.

To deploy using the Vercel CLI:
```bash
npm install -g vercel
vercel
```

---

## 🏋️ AI Training, Arena & Research Workflows

### 1. Single-Thread Curriculum Training
`ai/train.py` trains `AlphaTiger` from scratch or continues training from an existing checkpoint using **Curriculum Learning**. Early curriculum stages initialize mid-game or late-game boards to learn terminal tactics before progressing to full 4-turn campaigns:

```bash
# Start standard curriculum training
python -m ai.train

# Resume training from a specific checkpoint with custom simulations
python -m ai.train --resume ai/models/alphatigerv13.pt --sims 800 --iters 2000

# Run a 10-cycle curriculum across all 5 stages
python -m ai.train --cycle --resume ai/models/alphatigerv13.pt
```

#### Curriculum Stages:
1. **End Game:** Near-terminal Turn 4 states with contested Key Cities.
2. **Late Game:** Turn 3 states with fortified strongholds.
3. **Mid Game:** Turn 2 states with mixed fresh and tired armies.
4. **Early Game:** Post-opening perturbed boards.
5. **Full Game / Deep Game:** Complete starting board with up to 4,000 MCTS simulations.

---

### 2. Multi-Process Parallel Training
`ai/multitrain.py` scales self-play data generation across multiple CPU cores using Python's `multiprocessing` spawn context:

```bash
python -m ai.multitrain --resume ai/models/alphatigerv13.pt --sims 400 --model_name "alphatigerv14.pt"
```

You can also pass a custom starting state file:
```bash
python -m ai.multitrain --state_file "scratch/opening_state.txt" --resume ai/models/alphatigerv13.pt
```

---

### 3. Tournament Arena & Model Evaluation
`ai/arena.py` runs an automated head-to-head tournament between two model checkpoints, alternating sides to eliminate faction bias:

```bash
python -m ai.arena \
  --ckpt1 ai/models/alphatigerv13.pt \
  --ckpt2 ai/models/alphatigerv12.pt \
  --sims1 400 \
  --sims2 400 \
  --games 20 \
  --log arena_log.txt
```

#### What the Arena Measures:
* Win rates for Player 1 and Player 2 playing both Mysore and British.
* **Move Disagreements:** Logs instances where Ckpt 1 and Ckpt 2 disagree on the optimal policy move.
* **Branching Complexity:** Computes game-tree complexity $\prod (\text{legal moves})$ and luck branching factors across games.
* Automatically appends complete game logs and algebraic notation to `replay_log.txt`.

---

### 4. Exporting PyTorch Checkpoints to ONNX
When a new `.pt` model has been trained, export it to ONNX for use in the browser:

```bash
python -m ai.onnx
```
This reads the default PyTorch model (`DEFAULT_MODEL` in `game/constants.py`), extracts its factorized weights, and exports an optimized ONNX graph with dynamic batch axes to `ai/models/alphatiger.onnx`.

Copy the exported file to `public/alphatiger.onnx` to update the web version:
```bash
cp ai/models/alphatiger.onnx public/alphatiger.onnx
```

---

### 5. Visualizing Model Weights (Heatmaps)
`ai/heatmap.py` produces weight-magnitude heatmaps using Seaborn and Matplotlib to inspect internal layer activations:

```bash
# Visualize first dense layer weights
python -m ai.heatmap --ckpt ai/models/alphatigerv13.pt --layer fc1.weight

# Visualize value head weights
python -m ai.heatmap --ckpt ai/models/alphatigerv13.pt --layer value_fc1.weight

# Visualize policy head output weights
python -m ai.heatmap --ckpt ai/models/alphatigerv13.pt --layer policy_fc2.weight
```

---

### 6. Opening Book & Replay Analysis
`game/replay.py` can parse `replay_log.txt` (generated during self-play or arena matches) to construct an opening book decision tree and compute win rates:

```bash
python -m game.replay
```

This outputs:
* **Opening Tree:** Recursive move sequence tree with empirical British vs. Mysore win rates.
* **Location Code Frequencies:** Heatmap of the most contested territories across games.
* **Popular Moves:** Frequency list of the most commonly played opening operations.

---

## 📡 API Reference (FastAPI Backend)

When running `server.py` or the serverless API at `api/index.py`, the following REST endpoints are available:

### `GET /api/init`
Initializes a new game board, executes default setup, resolves opening luck, and returns the full JSON state.

### `POST /api/load-state`
Restores a game from a 148-bit binary string.
* **Request Body:** `{"state_str": "010010111..."}`
* **Response:** Serialized game data, active legal moves, winner status, and board UI state.

### `POST /api/play-move`
Applies a move action and resolves any resulting stochastic luck outcomes.
* **Request Body:** `{"state_str": "...", "move_idx": 42}`
* **Response:** Updated game state object.

### `POST /api/play-ai`
Asks the server AI to evaluate the state using MCTS, select the best move, apply it, resolve luck, and return the new state.
* **Request Body:** `{"state_str": "..."}`
* **Response:** Updated game state after AI move execution.

### `POST /api/eval-step`
Performs batched MCTS rollout simulations for the live evaluation bar.
* **Request Body:** `{"state_str": "...", "batch_size": 200}`
* **Response:**
  ```json
  {
    "eval_score": 0.42,
    "total_sims": 600,
    "top_moves": [
      { "move_name": "mad>pdc SM:trv", "eval": 0.48 },
      { "move_name": "bom>sat", "eval": 0.35 }
    ]
  }
  ```

### `POST /api/get-notation`
Translates an array of historical move indices into algebraic notation.
* **Request Body:** `{"replay_log": [12, 114, 25, 412]}`
* **Response:** `{"notation": "mad>pdc SM:trv RN:bom>goa + 1-0"}`

---

## 🧪 Quality Assurance, Automated Testing & CI/CD Pipeline

The repository features a four-stage automated testing hierarchy guaranteeing absolute byte-for-byte parity, zero regressions, and full device responsiveness:

```bash
# 1. Execute full JavaScript engine, UI, Candidate Lines & Visual Regression battery (74 tests)
$ npm test

# 2. Execute Python unit test battery (38 tests)
$ python3 -m unittest discover -s tests/unit -v

# 3. Execute Python integration & API test battery (22 tests)
$ python3 -m unittest discover -s tests/integration -v

# 4. Run static syntax and linter checks
$ npm run lint
$ python3 -m compileall -q ai game api tests
```

* **Continuous Integration:** Automated on every commit and pull request via [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) across a matrix of Python 3.10, 3.11, 3.12 and Node.js 18, 20, 22.
* **Test Catalog:** For a detailed breakdown of all 134 individual test cases, engineering rationales, and pipeline locations, consult [`TESTS.md`](./TESTS.md).

---

## 🚀 Roadmap & Future Directions

All architectural priorities (P0 critical bug fixes, P1 engine optimization with $O(1)$ dispatch and INT8 quantization, P2 Web Audio and offline PWA immersion, P3 FastAPI unification and security, P4 CI/CD testing, and P5 Next-Gen Roadmap initiatives) are **100% completed, verified, and passing**.

Key Next-Generation capabilities delivered in P5 include:
1. **WebGPU Client-Side MCTS:** Browser neural inference with WebGPU hardware acceleration and graceful fallback to WebAssembly CPU.
2. **Tactical Blunder & Influence Analytics:** Chess.com-style blunder categorization (*Brilliant*, *Best*, *Inaccuracy*, *Mistake*, *Blunder*) and dynamic 25-node territory threat heatmaps.
3. **Global Matchmaking & WebSocket Relay:** Centralized lobby server (`api/lobby.py`) supporting matchmaking queues, ELO calculations, and live spectator broadcasts.
4. **Historical Campaign Scenarios Trilogy:** Multi-era campaign modes covering the First (1767–1769), Second (1780–1784), and Fourth (1799) Anglo-Mysore Wars with historical initial configurations.
5. **Distributed Multi-GPU Self-Play & Replay Buffer:** High-throughput experience replay memory (`ai/replay_buffer.py`) and distributed training harness (`ai/dist_train.py`).
6. **Self-Evolving Opening Book Pipeline:** Automated arena tournament engine (`ai/evolve_book.py`) refining and pruning opening variations autonomously.
7. **Headless Visual Regression Testing:** Automated WCAG 2.1 color contrast verification, SVG boundary checking, and multi-viewport scaling tests (`tests/js/visual.test.js`).

For complete technical specifications on roadmap milestones and deliverables, see [`FUTURE_IMPROVEMENTS.md`](./FUTURE_IMPROVEMENTS.md).

---

## 📜 Historical Context

The **Anglo-Mysore Wars** (1767–1799) were a series of four major military conflicts in South India:
1. **First War (1767–1769):** Hyder Ali decisively outmaneuvered British forces, dictating peace terms at the gates of Madras.
2. **Second War (1780–1784):** Tipu Sultan demonstrated technological dominance with **Mysorean iron-cased rockets**, inflicting defeats on the British at Pollilur.
3. **Third War (1790–1792):** Lord Cornwallis allied with the Marathas and the Nizam of Hyderabad, besieging Seringapatam and forcing the Treaty of Seringapatam.
4. **Fourth War (1799):** The British under General Harris breached the fortress walls of Seringapatam, where Tipu Sultan was killed in combat defending the water gate on May 4, 1799.

This simulation captures the asymmetric tension of the conflict: Mysore's fortified defensive posture and rocket capabilities against the British East India Company's mobile infantry and naval dominance.