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

## 3. P5 — Next-Generation Roadmap & Future Initiatives (Completed)

All next-generation initiatives outlined below have been fully implemented, integrated into the live engine and UI, and validated with automated test suites.

### 5.1 Distributed Multi-GPU Self-Play Infrastructure
- **Status:** Completed
- **Deliverables:**
  - Thread-safe circular `ExperienceReplayBuffer` with compressed `.npz` chunk serialization in [`ai/replay_buffer.py`](./ai/replay_buffer.py).
  - Multi-worker training harness in [`ai/dist_train.py`](./ai/dist_train.py) supporting `torch.cuda.amp` mixed precision and `torch.compile(mode="max-autotune")`.
  - Comprehensive unit test suite in [`tests/unit/test_replay_buffer.py`](./tests/unit/test_replay_buffer.py).

### 5.2 WebGPU Execution Provider for Client-Side MCTS
- **Status:** Completed
- **Deliverables:**
  - Hardware-accelerated WebGPU EP (`webgpu`) initialization in [`public/js/mcts.js`](./public/js/mcts.js) with zero-downtime automatic fallback to `wasm`.
  - Batched parallel tensor evaluation via `predictBatch(states)` dispatching up to 16 positions simultaneously.
  - Active provider status exposed to UI evaluation bar and Settings modal dropdown.

### 5.3 Centralized WebSocket Relay & Global Matchmaking Lobby
- **Status:** Completed
- **Deliverables:**
  - Matchmaking queue, ELO rating delta calculation, and live spectator room manager in [`api/lobby.py`](./api/lobby.py).
  - FastAPI integration in [`api/app.py`](./api/app.py) mounting `GET /api/lobby/rooms`, `WebSocket /ws/lobby`, and `WebSocket /ws/room/{room_id}`.
  - Integration test in [`tests/integration/test_api.py`](./tests/integration/test_api.py).

### 5.4 Real-Time Tactical Evaluation & Heatmap Analytics Dashboard
- **Status:** Completed
- **Deliverables:**
  - Tactical move classification engine (*Brilliant*, *Best*, *Excellent*, *Good*, *Inaccuracy*, *Mistake*, *Blunder*) in [`public/js/analytics.js`](./public/js/analytics.js).
  - 25-node military control influence calculator with adjacent threat projection.
  - Interactive SVG evaluation sparkline graph generator.
  - Reactive SVG aura rings on territories and tactical badges integrated into [`public/script.js`](./public/script.js), [`public/style.css`](./public/style.css), and [`public/index.html`](./public/index.html).

### 5.5 Historical Campaign Scenarios (1st, 2nd & 4th Anglo-Mysore Wars)
- **Status:** Completed
- **Deliverables:**
  - Cross-platform campaign scenario state generators in [`game/scenarios.py`](./game/scenarios.py) and [`public/js/scenarios.js`](./public/js/scenarios.js):
    1. **1st Anglo-Mysore War (1767–1769):** Hyder Ali cavalry offensive and Maratha alliances.
    2. **2nd Anglo-Mysore War (1780–1784):** Battle of Pollilur, French alliance, and Carnatic incursions.
    3. **4th Anglo-Mysore War (1799):** Siege of Seringapatam, Nizam coalition, and Tipu's rocket bastion.
  - Settings modal campaign selector with instant state synchronization in [`public/index.html`](./public/index.html) and [`public/script.js`](./public/script.js).
  - Unit tests in [`tests/unit/test_scenarios.py`](./tests/unit/test_scenarios.py) and [`tests/js/engine.test.js`](./tests/js/engine.test.js).

### 5.6 Self-Evolving Opening Book from Automated Tournaments
- **Status:** Completed
- **Deliverables:**
  - Automated arena tournament simulation pipeline in [`ai/evolve_book.py`](./ai/evolve_book.py) discovering winning lines (>60% winrate) and pruning discredited branches.
  - Validation of move legality, ply-depth tracking, and lossless serialization to [`public/opening_book.json`](./public/opening_book.json).
  - Unit test suite in [`tests/unit/test_evolve_book.py`](./tests/unit/test_evolve_book.py).

### 5.7 Automated Headless Visual Regression Testing
- **Status:** Completed
- **Deliverables:**
  - Headless visual and structural regression test suite in [`tests/js/visual.test.js`](./tests/js/visual.test.js) asserting:
    1. WCAG 2.1 AA/AAA color contrast ratios across all 10 curated themes in [`public/js/ui/themes.js`](./public/js/ui/themes.js).
    2. Exact registration and properties of all 5 unit token aesthetics.
    3. SVG board geometry bounds, ensuring all 25 nodes and 5 key forts maintain $\ge 20$ px clearance.
    4. Multi-device responsive coordinate projection and minimum touch hitbox preservation across 4 standard viewports (Mobile, Tablet, Laptop, Desktop Ultrawide).

---

## 4. Updated Implementation Timeline

```mermaid
gantt
    title Tiger's Day Roadmap & Milestone Evolution
    dateFormat  YYYY-MM-DD
    
    section Completed Core (v1.0 - v1.2)
    P0: Critical Bug Fixes & Interface Realignment :done, p0, 2026-09-01, 2026-09-03
    P1: Engine O(1) Dispatch & INT8 Quantization  :done, p1, 2026-09-03, 2026-09-05
    P2: Web Audio, Themes Modularization & PWA    :done, p2, 2026-09-05, 2026-09-06
    P3: FastAPI Unified App & Security Validation :done, p3, 2026-09-06, 2026-09-06
    P4: CI/CD Pipeline & 49 Automated Tests       :done, p4, 2026-09-06, 2026-09-07

    section Next-Gen Roadmap (v1.3 - v2.0 Completed)
    P5.1: Distributed Multi-GPU Self-Play         :done, p5_1, 2026-09-07, 2026-09-07
    P5.2: WebGPU Client MCTS Acceleration         :done, p5_2, 2026-09-07, 2026-09-07
    P5.3: Global Matchmaking & WebSocket Relay    :done, p5_3, 2026-09-07, 2026-09-07
    P5.4: Tactical Blunder Analysis & Heatmaps    :done, p5_4, 2026-09-07, 2026-09-07
    P5.5: Historical Campaign Scenarios Trilogy   :done, p5_5, 2026-09-07, 2026-09-07
    P5.6: Self-Evolving Opening Book Pipeline     :done, p5_6, 2026-09-07, 2026-09-07
    P5.7: Headless Visual Regression Suite        :done, p5_7, 2026-09-07, 2026-09-07
```

---

*Last Updated: 2026-09-07 — All P0–P5 roadmap milestones completed, implemented, and verified across 64 automated tests.*

