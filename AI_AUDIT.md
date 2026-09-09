# 🐅 Tiger's Day — Deep Repository Audit

**Auditor:** Antigravity IDE  
**Date:** 2026-09-08  
**Scope:** Full-stack audit of game engine, AI pipeline, API server, client frontend, CI/CD, and deployment infrastructure  

---

## Table of Contents

1. [Bugs & Correctness Issues](#1-bugs--correctness-issues)
2. [Unintended Behavior & Subtle Defects](#2-unintended-behavior--subtle-defects)
3. [Performance & Efficiency Concerns](#3-performance--efficiency-concerns)
4. [Security & Robustness](#4-security--robustness)
5. [Architecture & Code Quality](#5-architecture--code-quality)
6. [Expansion & Iteration Opportunities](#6-expansion--iteration-opportunities)

---

## 1. Bugs & Correctness Issues

### ✅ ~~BUG-1: `dist_train.py` — `find_move` Return Value Mishandled~~ (FIXED)

**Commit `780d081`** — `find_move()` returns a `(move, policy)` tuple, but `play_single_game` assigned it to a single variable. Calling `.copy()` on a tuple and then sampling from it produced garbage training data. Fixed by unpacking: `move, policy = mcts.find_move(...)`.

---

### ✅ ~~BUG-2: `dist_train.py` — Forward Pass Output Order Swapped~~ (FIXED)

**Commit `780d081`** — `AlphaTiger.forward()` returns `(value, policy_logits)` but the destructuring was `pred_policies, pred_values = ...`, meaning value loss was computed against policy logits and vice versa. Fixed to `pred_values, pred_policies = ...`.

---

### 🟡 BUG-3: `get_state_winner` — Mysore Win Condition Subtlety

The Mysore attrition victory checks `state.turn == 4 and not state.fresh_armies.any() and state.to_move == 0`. The flow is correct but extremely subtle — the condition only fires once the full British cycle `0 → 1 → 2 → 0` completes on Turn 4, which is semantically correct. This is an invariant that should be documented with a comment explaining the reasoning.

---

### 🟡 BUG-4: `resolve_battles` Mutates State In-Place Despite Return Pattern

`resolve_battles(state, ...)` modifies `state` directly (e.g., `state.mluck += 1`, `state.set_node_tired_army(...)`) and also `return state`. All current callers pass `next_state`, so this is safe today, but the pattern is fragile — any future caller passing the original `state` would corrupt the game tree. Consider adding a defensive `state = state.copy()` at the top or documenting the mutation contract.

---

### ✅ ~~BUG-5: `replay.py` — `get_overall_british_wr` Division-by-Zero~~ (FIXED)

**Commit `780d081`** — `return british_wins / total_games` raised `ZeroDivisionError` when no completed games existed. Fixed to `return british_wins / total_games if total_games > 0 else 0.5`.

---

### ✅ ~~BUG-6: Service Worker Missing ONNX Model from Cache Manifest~~ (FIXED)

**Commit `780d081`** — `ASSETS_TO_CACHE` did not include `./alphatiger.quant.onnx` (432KB). Client-side WASM inference failed offline. Added to the cache manifest and bumped cache version to `v2.16`.

---

## 2. Unintended Behavior & Subtle Defects

### ⚠️ SUBTLE-1: MCTS Transposition Table — Zobrist Hash Collisions

The transposition table uses 64-bit Zobrist hashing to cache `(value, policy)` pairs for the 148-bit state space. No collision verification is performed — if two different states hash to the same key, the cached policy from the wrong state is silently used. The policy is a full 959-dimensional distribution, so using a wrong position's policy produces legal-but-strategically-wrong moves that are very hard to debug.

**Mitigation options:**
1. Store the state bit-string alongside the hash for verification on retrieval.
2. Use a bounded LRU cache with verification.
3. Accept the risk with a comment documenting the expected collision rate (~1 in 2^64).

---

### ⚠️ SUBTLE-2: `search_batch` Virtual Loss Arithmetic

Virtual loss adds 3.0 to `visit_count` and subtracts 3.0 from `value_sum` for in-flight paths. If a leaf node appears on multiple paths within the same batch (shallow tree), virtual loss accumulates. The `eval` property (`value_sum / visit_count`) doesn't guard against `visit_count == 0`, though `select_child` defensively uses `max(0.0, float(child.visit_count))`. Low probability but worth a guard.

---

### ⚠️ SUBTLE-3: `EvalTreeLRUCache` Stores Mutable MCTS Objects

The eval cache stores shared `MCTS` instances. While `asyncio.Lock` protects the cache lookup, if two concurrent requests share the same MCTS tree, `search()` mutations could interleave. Currently safe with single-threaded asyncio, but would break with workers.

---

### ⚠️ SUBTLE-4: `arena.py` Writes to Hardcoded `replay_log.txt` Path

`arena.py` opens `"replay_log.txt"` at project root with `"w"` mode (line 175) then `"a"` mode (line 131). Multiple concurrent arena runs would corrupt each other's output. The path should be parameterized.

---

### ⚠️ SUBTLE-5: `evolve_book.py` Heuristic Territory Count Uses Hardcoded Vector Indices

```python
british_count = np.sum(state.vector[12:87:3])   # sums fresh armies
mysore_count = np.sum(state.vector[14:87:3])     # sums forts
```

This conflates "British" with "fresh armies" and "Mysore" with "forts". Semantically incorrect — armies can belong to either side contextually. The heuristic may coincidentally work for the default scenario but is conceptually wrong.

---

### ⚠️ SUBTLE-6: `:memory:.ses` File in Repository Root

A stray `:memory:.ses` file exists at the project root. While `.gitignore` includes `*.ses`, this file may have been committed before the rule was added. Should be `git rm --cached`.

---

## 3. Performance & Efficiency Concerns

### ✅ ~~PERF-1: `GameState.__str__` — O(n²) String Concatenation~~ (FIXED)

**Commit `780d081`** — The MCTS hot path called `__str__` thousands of times per search. Replaced char-by-char concatenation loop with `''.join()` generator expression.

---

### ✅ ~~PERF-2: `GameState.copy()` — Redundant `__init__` Allocation~~ (FIXED)

**Commit `780d081`** — `copy()` called `GameState()` which ran `__init__` (allocating vector, setting all cards, turn, to_move) then immediately overwrote everything. Now uses `object.__new__(GameState)` to bypass `__init__`, eliminating ~30 redundant vector writes per copy on the hottest path in the codebase.

---

### 🐢 PERF-3: JS `GameState` Property Getters Allocate New `Uint8Array` Every Call

Every access to `fresh_armies`, `tired_armies`, `forts`, or `empty` allocates a new 25-byte typed array. During a 400-simulation client-side MCTS search, this creates thousands of short-lived allocations, pressuring the browser's GC.

**Mitigation:** Cache results per state mutation, or use index-based access patterns that read directly from `vector`.

---

### ✅ ~~PERF-4: `MetricsCollector.latencies` — `list.pop(0)` Is O(n)~~ (FIXED)

**Commit `780d081`** — Replaced `list` with `collections.deque(maxlen=1000)` for O(1) append and automatic eviction.

---

### 🐢 PERF-5: `LeaderboardManager` Opens a New SQLite Connection Per Method Call

`_get_connection()` is called in every method. `record_match` alone opens **three** separate connections (two `get_or_create_player` + one for updates). Use a persistent connection or connection pool instead.

---

### 🐢 PERF-6: `MCTS.search_time_budget` Contested Fortress Check Is O(NODES²)

Python-level nested iteration over nodes and adjacency. Can be vectorized:
```python
has_adjacent_armies = (ADJACENCY_MATRIX @ (root_state.fresh_armies | root_state.tired_armies)) > 0
contested = bool(np.any(root_state.forts & has_adjacent_armies))
```

---

## 4. Security & Robustness

### 🔒 SEC-1: CORS Allows All Origins (`allow_origins=["*"]`)

For a game server with a leaderboard and ELO ratings, wildcard CORS allows any website to make API calls. Restrict to the actual deployment domain and localhost for dev.

---

### 🔒 SEC-2: Rate Limiter Memory Leak — `buckets` Dict Grows Unbounded

Every unique IP gets a permanent entry in `TokenBucketRateLimiter.buckets`. No stale entry eviction exists. Under sustained traffic, this dict grows indefinitely.

**Fix:** Add periodic cleanup for entries older than `capacity / refill_rate` seconds, or use an LRU dict with a maximum size.

---

### 🔒 SEC-3: `record_match` Endpoint Has No Authentication

Anyone can POST to `/api/player/record-match` with arbitrary handles to manipulate ELO ratings. There's no session token, game proof, or server-side match verification.

**Recommendation:** Record matches server-side when games conclude via WebSocket, or add signed match tokens.

---

### ✅ ~~SEC-4: `room_id` Uses Predictable `random.randint`~~ (FIXED)

**Commit `780d081`** — Only 9,000 possible room IDs with `random.randint(1000, 9999)`. Replaced with `secrets.token_hex(4).upper()` (~4 billion possibilities, cryptographically secure).

---

## 5. Architecture & Code Quality

### 🏗️ ARCH-1: Dual Training Pipelines with Inconsistent Implementations

Three separate training modules exist:
1. `ai/train.py` — Single-thread trainer ✅ correct
2. `ai/multitrain.py` — Multi-process trainer ✅ correct
3. `ai/dist_train.py` — "Distributed" trainer ✅ **now fixed** (was broken, see BUG-1 & BUG-2)

`dist_train.py` still duplicates its own `play_single_game` rather than reusing `train.py`'s. Consider consolidating.

---

### 🏗️ ARCH-2: `play-ai` Endpoint Creates a New MCTS Instance Per Request

Every `/api/play-ai` call instantiates a fresh MCTS tree with no subtree reuse. The `eval-step` endpoint correctly caches MCTS instances, but the main play endpoint doesn't. Server-side play starts from scratch every move.

---

### 🏗️ ARCH-3: No Zombie Room Cleanup in Lobby

`MatchRoom` objects in `global_lobby.rooms` are never cleaned up after both players disconnect. `list_active_rooms()` returns **all rooms ever created** during the server's lifetime. Add a TTL or lazy cleanup.

---

### ✅ ~~ARCH-4: `heatmap.py` Has Hard Dependencies Not in `requirements-dev.txt`~~ (ALREADY OK)

`requirements-dev.txt` already includes `matplotlib>=3.7.0` and `seaborn>=0.12.0`. No action needed.

---

### 🏗️ ARCH-5: Wildcard `from ai.train import *` in `arena.py`

Imports all of `train.py`'s symbols into `arena.py`'s namespace, most unused. Use explicit imports for clarity.

---

## 6. Expansion & Iteration Opportunities

### 🚀 EXPAND-1: MCTS Tree Reuse Across Consecutive Server Moves

The MCTS `update_root()` method supports subtree retention, but server endpoints don't use it. Maintaining per-session MCTS trees would reduce latency by ~40-60% in mid-game positions and enable "pondering" during the opponent's turn.

---

### 🚀 EXPAND-2: Progressive Widening for High-Branching Card Phases

Card phases can have 50+ legal moves. Progressive widening would start with top-k moves by neural prior (e.g., k=10) and widen only as existing children are well-explored, significantly reducing per-simulation cost.

---

### 🚀 EXPAND-3: INT4 Quantization / Dynamic ONNX Batch Axes

The ONNX model uses INT8 quantization (432KB). INT4 could reduce to ~220KB. Adding dynamic batch axes would allow `predictBatch` to send N positions in a single ONNX session run instead of N individual runs.

---

### 🚀 EXPAND-4: Gzip/Brotli Compression for Opening Book

`opening_book.json` (18KB) could shrink to ~3-4KB with Brotli. Additionally, state keys are 148-character bit-strings — base64 encoding would reduce key length to ~25 characters.

---

### 🚀 EXPAND-5: Pondering (Background Search During Opponent's Turn)

During human vs. AI games, the AI idles while the human thinks. Pondering would continue MCTS search on the predicted opponent move, significantly improving perceived response time.

---

### 🚀 EXPAND-6: Prioritized Experience Replay

The `ExperienceReplayBuffer` uses uniform sampling. Prioritized Experience Replay (PER) would weight samples by TD-error, improving sample efficiency by 30-50% based on Atari RL literature.

---

### 🚀 EXPAND-7: PGN-Style Human-Readable Game Notation Export

The `.tdr` replay format uses raw move indices. A PGN-like format using the algebraic notation from `notate()` would enable sharing games on forums and analysis by external tools.

---

### 🚀 EXPAND-8: Spectator Mode Evaluation Overlay

The WebSocket relay supports spectators but only sends raw game state. Adding real-time eval data (eval bar, top candidate moves) would create a chess.com-like spectator experience.

---

### 🚀 EXPAND-9: Adaptive MCTS Simulation Budget

Fixed simulation counts (250 API, 400 default) could be replaced with an adaptive system that allocates more simulations to complex positions and fewer to forced sequences based on branching factor and eval uncertainty.

---

### 🚀 EXPAND-10: WebSocket Reconnection State Recovery

If a spectator disconnects and reconnects, the server sends `SYNC_STATE` but the client may not handle mid-game spectator joins gracefully (needs to reconstruct full UI state from bit-string).

---

## Summary Priority Matrix

| Status | ID | Category | Impact | Effort |
|--------|------|----------|--------|--------|
| ✅ Done | BUG-1 | Bug | `dist_train.py` garbage data | 5 min |
| ✅ Done | BUG-2 | Bug | `dist_train.py` swapped losses | 5 min |
| ✅ Done | BUG-5 | Bug | Division by zero in replay stats | 5 min |
| ✅ Done | BUG-6 | Bug | ONNX not cached for offline PWA | 5 min |
| ✅ Done | PERF-1 | Performance | `__str__` O(n²) on hot path | 10 min |
| ✅ Done | PERF-2 | Performance | `copy()` redundant init on hot path | 10 min |
| ✅ Done | PERF-4 | Performance | O(n) latency list pop | 10 min |
| ✅ Done | SEC-4 | Security | Predictable room IDs | 10 min |
| 🟡 Open | SEC-3 | Security | Unauthenticated leaderboard | 2 hrs |
| 🟡 Open | SEC-2 | Security | Rate limiter memory leak | 30 min |
| 🟡 Open | ARCH-3 | Quality | Zombie room accumulation | 30 min |
| 🟡 Open | SUBTLE-1 | Correctness | Zobrist hash collision risk | 1 hr |
| 🟢 Open | PERF-3 | Performance | JS typed array allocation churn | 2 hrs |
| 🟢 Open | PERF-5 | Performance | SQLite connection churn | 1 hr |
| 🟢 Open | ARCH-1 | Quality | Duplicate training module | 1 hr |
| 🔵 Open | EXPAND-1 | Feature | Server MCTS tree reuse | 3 hrs |
| 🔵 Open | EXPAND-5 | Feature | AI pondering | 4 hrs |
| 🔵 Open | EXPAND-2 | Feature | Progressive widening | 3 hrs |
| 🔵 Open | EXPAND-6 | Feature | Prioritized experience replay | 4 hrs |

---

*Audit complete. 6 bugs identified (4 now fixed), 6 subtle behavioral issues, 6 performance concerns (3 now fixed), 4 security items (1 now fixed), 5 architectural observations (1 already OK). 8 of 9 quick fixes applied in commit `780d081`.*
