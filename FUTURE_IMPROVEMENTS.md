# 🐅 Tiger’s Day — Comprehensive Codebase Audit & Future Improvements Roadmap

This document provides a systematic architectural record of the **Tiger’s Day (Anglo-Mysore Wars 1767–1799)** repository. It details all previously identified issues (P0–P4) that have been **fully resolved, verified, and integrated**, followed by the **Next-Generation Roadmap (P5)** outlining future engineering initiatives.

---

## 📑 Table of Contents

1. [Executive Summary & Current Health Status](#1-executive-summary--current-health-status)
2. [Completed Work Audit (P0 – P4 Status)](#2-completed-work-audit-p0--p4-status)
   - [P0 — Critical Bug Fixes & Interface Realignment](#p0--critical-bug-fixes--interface-realignment-completed)
   - [P1 — Game Engine & AI Pipeline Enhancements](#p1--game-engine--ai-pipeline-enhancements-completed)
   - [P2 — Frontend & UX Immersion](#p2--frontend--ux-immersion-completed)
   - [P3 — Server, API & Architecture Modernization](#p3--server-api--architecture-modernization-completed)
   - [P4 — Quality Assurance, Testing & CI/CD](#p4--quality-assurance-testing--cicd-completed)
3. [P5 — Next-Generation Roadmap & Future Initiatives](#3-p5--next-generation-roadmap--future-initiatives)
   - [5.1 Distributed Multi-GPU Self-Play Infrastructure](#51-distributed-multi-gpu-self-play-infrastructure)
   - [5.2 WebGPU Execution Provider for Client-Side MCTS](#52-webgpu-execution-provider-for-client-side-mcts)
   - [5.3 Centralized WebSocket Relay & Global Matchmaking Lobby](#53-centralized-websocket-relay--global-matchmaking-lobby)
   - [5.4 Real-Time Tactical Evaluation & Heatmap Analytics Dashboard](#54-real-time-tactical-evaluation--heatmap-analytics-dashboard)
   - [5.5 Historical Campaign Scenarios (1st, 2nd & 4th Anglo-Mysore Wars)](#55-historical-campaign-scenarios-1st-2nd--4th-anglo-mysore-wars)
   - [5.6 Self-Evolving Opening Book from Automated Tournaments](#56-self-evolving-opening-book-from-automated-tournaments)
   - [5.7 Automated Headless Visual Regression Testing](#57-automated-headless-visual-regression-testing)
4. [Updated Implementation Timeline](#4-updated-implementation-timeline)

---

## 1. Executive Summary & Current Health Status

Tiger's Day is an asymmetric strategic wargame combining historical simulation, deep reinforcement learning (AlphaZero + MCTS), 100% client-side WebAssembly ONNX inference, and WebRTC peer-to-peer multiplayer.

### Current System Health: **EXCELLENT (Production Ready)**
- **Automated Verification:** 49 automated tests passing with 0 failures across Python 3.10/3.11/3.12 and Node.js 18/20/22 (documented in [`TESTS.md`](./TESTS.md)).
- **Parity Guarantees:** 100% byte-for-byte mathematical parity between Python NumPy state transitions and JavaScript Uint8Array client transitions.
- **Continuous Integration:** Multi-stage GitHub Actions CI pipeline executing bytecode compilation, Ruff linting, Node syntax verification, unit matrix tests, and end-to-end integration tests on every commit and PR.
- **Model Efficiency:** Model compressed by ~72% via INT8 dynamic quantization ([`public/alphatiger.quant.onnx`](./public/alphatiger.quant.onnx), 431KB).
- **PWA & Offline Immersion:** 100% offline standalone PWA with cache-first Service Worker, Web Audio procedural soundscape, 10 bespoke themes, 5 unit styles, and WCAG keyboard navigation.

---

## 2. Completed Work Audit (P0 – P4 Status)

### P0 — Critical Bug Fixes & Interface Realignment `[COMPLETED]`

| Item | Description | Resolution Status | Verified In |
|---|---|---|---|
| **3.1 MCTS Signatures** | Divergent `__init__` and `find_move` arguments causing `TypeError` in server and arena | **Fixed:** Accepts `simulations`, `stop=True` parameter added, default simulations fallbacks implemented | [`ai/mcts.py`](./ai/mcts.py), [`ai/arena.py`](./ai/arena.py), [`tests/unit/test_mcts.py`](./tests/unit/test_mcts.py) |
| **3.2 Card Names Discrepancy** | Mysore and British card name arrays swapped in client history notation | **Fixed:** Realigned `MYSORE_CARD_NAMES` and `BRITISH_CARD_NAMES` with `game/constants.py` | [`public/script.js`](./public/script.js) |
| **3.3 Bit 94 Asymmetry** | Python uninitialized state left bit 94 at 0 while JS set bit 94 to 1 | **Fixed:** Explicitly set `card_strength = 0`, `turn = 1`, `to_move = 0` in Python constructor | [`game/state.py`](./game/state.py), [`tests/unit/test_state.py`](./tests/unit/test_state.py) |
| **3.4 `read_str` Validation Bug** | Loop counter caused out-of-bounds index exceptions during territory checks | **Fixed:** Iterates through all 25 `NODES` ensuring `sum <= 1` per node | [`game/state.py`](./game/state.py), [`public/js/state.js`](./public/js/state.js) |

---

### P1 — Game Engine & AI Pipeline Enhancements `[COMPLETED]`

| Item | Description | Resolution Status | Verified In |
|---|---|---|---|
| **4.1 Action Space Alignment** | Documentation and comments cited 953 actions instead of true 959 dimension | **Fixed:** Reconciled documentation and verified exact 959-action dimension across both engines | [`README.md`](./README.md), [`tests/js/engine.test.js`](./tests/js/engine.test.js) |
| **4.2 O(1) Action Dispatch** | Slow sequential range checks in `get_next_state` | **Fixed:** Precomputed `ACTION_DISPATCH` lookup tables in Python and JavaScript | [`game/updater.py`](./game/updater.py), [`public/js/engine.js`](./public/js/engine.js) |
| **4.3 Battle2 Net Card Strength** | Secondary combat branch omitted card combat strength modifiers | **Fixed:** Routed `net_card_strength` into combat resolution across all branches | [`game/updater.py`](./game/updater.py), [`tests/unit/test_updater.py`](./tests/unit/test_updater.py) |
| **4.4 Threefold Repetition** | Lack of repetition detection permitted infinite cycle moves | **Fixed:** Added `check_repetition` (Python) and `checkRepetition` (JS) | [`game/updater.py`](./game/updater.py), [`public/js/engine.js`](./public/js/engine.js) |
| **4.5 CUDA IPC Safeguards** | Multiprocessing worker spawn failed on CUDA contexts | **Fixed:** Cloned model to CPU with `share_memory()` before worker dispatch | [`ai/multitrain.py`](./ai/multitrain.py) |
| **4.6 ONNX INT8 Quantization** | 1.5MB base float32 model consumed excessive mobile bandwidth | **Fixed:** Created `quantize_model()` exporter, producing 431KB INT8 model | [`ai/onnx.py`](./ai/onnx.py), [`public/alphatiger.quant.onnx`](./public/alphatiger.quant.onnx) |
| **4.7 Opening Book Engine** | Full tree searches required on Turn 1 | **Fixed:** Generated `opening_book.json` with fast-path lookup in MCTS | [`ai/opening_book.py`](./ai/opening_book.py), [`public/opening_book.json`](./public/opening_book.json) |

---

### P2 — Frontend & UX Immersion `[COMPLETED]`

| Item | Description | Resolution Status | Verified In |
|---|---|---|---|
| **5.1 Procedural Web Audio** | Zero audio effects during tactical moves and combat | **Fixed:** Implemented zero-dependency procedural Web Audio engine (`TDSound`) | [`public/js/sound.js`](./public/js/sound.js), [`tests/js/ui.test.js`](./tests/js/ui.test.js) |
| **5.2 Theme Modularization** | 10 themes and 5 unit styles tangled in UI coordinator | **Fixed:** Extracted into standalone ES/CommonJS module (`THEMES`, `UNIT_STYLES`) | [`public/js/ui/themes.js`](./public/js/ui/themes.js) |
| **5.3 Dual-Engine Inference** | UI locked to client WASM without server fallback | **Fixed:** Configured dual-engine selector with server fallback to WASM | [`public/script.js`](./public/script.js), [`public/index.html`](./public/index.html) |
| **5.4 Offline PWA Support** | Game required active internet to load static assets | **Fixed:** Added `manifest.json` and cache-first Service Worker (`sw.js`) | [`public/manifest.json`](./public/manifest.json), [`public/sw.js`](./public/sw.js) |
| **5.5 Keyboard A11y & Hotkeys** | Mouse-only board interaction | **Fixed:** Added `tabindex="0"`, `role="button"`, ARIA labels, and hotkeys (`Esc`, `Z`, `R`, `P`, arrows) | [`public/script.js`](./public/script.js), [`tests/js/ui.test.js`](./tests/js/ui.test.js) |
| **5.6 WebRTC Resilience** | P2P disconnects on network fluctuation | **Fixed:** 5s bidirectional heartbeat (`PING`/`PONG`) and automatic reconnection state recovery | [`public/js/multiplayer.js`](./public/js/multiplayer.js) |

---

### P3 — Server, API & Architecture Modernization `[COMPLETED]`

| Item | Description | Resolution Status | Verified In |
|---|---|---|---|
| **6.1 Unified FastAPI App** | Duplicate code in `server.py` and `api/index.py` | **Fixed:** Created shared `create_app()` factory in `api/app.py` | [`api/app.py`](./api/app.py), [`server.py`](./server.py), [`api/index.py`](./api/index.py) |
| **6.2 Thread-Safe LRU Eval Cache** | Concurrent server evaluation requests lacked caching | **Fixed:** Built `EvalTreeLRUCache` with `asyncio.Lock` | [`api/app.py`](./api/app.py) |
| **6.3 Input Validation & Schemas** | Arbitrary payloads accepted by server endpoints | **Fixed:** Strict Pydantic models with `^[01]{148}$` regex validation | [`api/app.py`](./api/app.py), [`tests/integration/test_api.py`](./tests/integration/test_api.py) |

---

### P4 — Quality Assurance, Testing & CI/CD `[COMPLETED]`

| Item | Description | Resolution Status | Verified In |
|---|---|---|---|
| **7.1 Automated Test Suite** | 0 automated tests in repository | **Fixed:** Implemented 49 unit, integration, parity, and UI test cases | [`tests/`](./tests/), [`TESTS.md`](./TESTS.md) |
| **7.2 Dependency Pinning** | NumPy 2.x incompatibilities and unpinned wheels | **Fixed:** Pinned `numpy>=1.24.0,<=2.4.3` and updated `package.json` | [`requirements.txt`](./requirements.txt), [`package.json`](./package.json) |
| **7.3 GitHub Actions CI/CD** | No automated validation on git commits | **Fixed:** Multi-job CI pipeline running lint, unit tests, and integration tests | [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) |
| **7.4 Repository Cleanliness** | Loose session files and multi-megabyte logs in root | **Fixed:** Excluded `.ses` in `.gitignore` and organized logs under `logs/` | [`.gitignore`](./.gitignore), [`logs/`](./logs/) |

---

## 3. P5 — Next-Generation Roadmap & Future Initiatives

With core stabilization, performance optimization, and testing completed, the following roadmap outlines future development phases.

### 5.1 Distributed Multi-GPU Self-Play Infrastructure
- **Status:** Planning / Architecture Design
- **Objective:** Scale self-play game generation across multi-GPU nodes or cloud clusters (e.g., Slurm, Ray, Kubernetes).
- **Key Tasks:**
  1. Replace standard Python `multiprocessing` with a **Ray-based actor pool** distributing MCTS rollouts across multiple GPU workers.
  2. Implement asynchronous experience replay buffer streaming to centralized training workers via Redis or Apache Arrow IPC.
  3. Integrate mixed-precision training (`torch.cuda.amp`) and PyTorch 2.x `torch.compile(mode="max-autotune")` for 3x training acceleration.

### 5.2 WebGPU Execution Provider for Client-Side MCTS
- **Status:** Research / Prototyping
- **Objective:** Move browser neural evaluations from WebAssembly CPU to WebGPU hardware acceleration.
- **Key Tasks:**
  1. Add WebGPU EP (`webgpu`) initialization support in `public/js/mcts.js` with fallback to `wasm`.
  2. Implement batched parallel leaf evaluation in browser MCTS (evaluating 8–16 MCTS positions simultaneously in a single WebGPU tensor dispatch).
  3. Target: Reduce 500-simulation browser thinking time from ~600ms down to <80ms.

### 5.3 Centralized WebSocket Relay & Global Matchmaking Lobby
- **Status:** Architecture Design
- **Objective:** Enable global public matchmaking and overcome strict enterprise/carrier-grade symmetric NATs where WebRTC direct P2P fails.
- **Key Tasks:**
  1. Add a lightweight WebSocket lobby server (`lobby/server.py`) supporting:
     - Public matchmaking queue with ELO rating calculation.
     - Fallback WebSocket message relay when WebRTC ICE candidate negotiation fails.
     - Live spectator mode broadcasting move streams to observers.
  2. Add room search, private friend challenges, and player handle customization in the frontend.

### 5.4 Real-Time Tactical Evaluation & Heatmap Analytics Dashboard
- **Status:** Prototyping
- **Objective:** Provide Chess.com-style post-game review and in-game tactical overlays.
- **Key Tasks:**
  1. **Move Classification Engine:** Label historical moves with tactical badges: *Best Move*, *Excellent*, *Inaccuracy*, *Mistake*, *Blunder*, and *Brilliant*.
  2. **Territory Influence Overlay:** Render dynamic SVG heat gradients over the 25 territories indicating military control zones and threat projection.
  3. **Winrate Graph:** Visual interactive graph displaying evaluation score progression throughout the 4 turns.

### 5.5 Historical Campaign Scenarios (1st, 2nd & 4th Anglo-Mysore Wars)
- **Status:** Game Design / Expansion
- **Objective:** Expand beyond the Third Anglo-Mysore War (1790–1792) into a full historical campaign trilogy.
- **Key Tasks:**
  1. **Scenario 1: First Anglo-Mysore War (1767–1769):** Hyder Ali's rapid counter-offensive; Mysore begins with mobile cavalry armies; British defenses concentrated in Madras and Bombay.
  2. **Scenario 2: Second Anglo-Mysore War (1780–1784):** Battle of Pollilur; introduces French naval expeditionary cards (Admiral Suffren) and scorched-earth tactical options.
  3. **Scenario 3: Fourth Anglo-Mysore War (1799):** The Siege of Seringapatam; British coalition with Hyderabad Nizam vs Tipu Sultan's fortified Mysore capital with Rocket corps.
  4. Scenario selector integrated into Settings modal with custom starting state bitstrings.

### 5.6 Self-Evolving Opening Book from Automated Tournaments
- **Status:** Planned
- **Objective:** Automatically refine and expand `public/opening_book.json` directly from high-tier self-play tournaments.
- **Key Tasks:**
  1. Create a CI/CD cron action or training hook that runs weekly 100-game arena tournaments between model checkpoints.
  2. Automatically parse game notation logs, identify winning branches with winrates > 65%, and commit updated opening vectors to git automatically.

### 5.7 Automated Headless Visual Regression Testing
- **Status:** Planned
- **Objective:** Automatically catch UI layout shifts, SVG clipping, or color palette contrast regressions across browsers.
- **Key Tasks:**
  1. Add Playwright test suite capturing screenshots of all 10 themes and 5 unit token styles across standard viewport resolutions (375x812, 768x1024, 1366x768, 1920x1080).
  2. Pixel-diff screenshots against baseline golden images in CI, preventing CSS regressions.

---

## 4. Updated Implementation Timeline

```mermaid
gantt
    title Tiger's Day Roadmap & Milestone Evolution
    dateFormat  YYYY-MM-DD
    
    section Completed (v1.0 - v1.2)
    P0: Critical Bug Fixes & Interface Realignment :done, p0, 2026-09-01, 2026-09-03
    P1: Engine O(1) Dispatch & INT8 Quantization  :done, p1, 2026-09-03, 2026-09-05
    P2: Web Audio, Themes Modularization & PWA    :done, p2, 2026-09-05, 2026-09-06
    P3: FastAPI Unified App & Security Validation :done, p3, 2026-09-06, 2026-09-06
    P4: CI/CD Pipeline & 49 Automated Tests       :done, p4, 2026-09-06, 2026-09-07

    section Next-Gen Roadmap (v1.3 - v2.0)
    P5.1: WebGPU Client MCTS Acceleration         :active, p5_1, 2026-09-08, 10d
    P5.2: Tactical Blunder Analysis & Heatmaps    :p5_2, 2026-09-18, 12d
    P5.3: Global Matchmaking & WebSocket Relay    :p5_3, 2026-09-30, 14d
    P5.4: Multi-Era Historical Campaign Scenarios :p5_4, 2026-10-14, 14d
    P5.5: Distributed Ray Multi-GPU Training      :p5_5, 2026-10-28, 16d
    P5.6: Playwright Visual Regression Suite      :p5_6, 2026-11-13, 8d
```

---

*Last Updated: 2026-09-07 — All P0–P4 roadmap milestones completed and verified.*
