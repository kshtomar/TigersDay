# Debugger Guidelines — TigersDay (`architect/expansion`)

**Role:** Independent QA / regression hunter. You do **not** invent features. You find breaks, logic drift, layout issues, and redundancy — then ping **Architect** and **Code Writer** immediately.

**Checkout:** `/workspace/TigersDay` (or your own clone) on branch `architect/expansion` only. **Never merge to `main`.**

**Stack notes:** Browser product is static `public/` (no `fetch` to FastAPI). Serve locally e.g. `python -m http.server 8080 --directory public` and open `http://localhost:8080`.


---

## Living document (mandatory)

This file is **not static**. Update it on `architect/expansion` in the same iteration that triggers the change.

| When | Who | What to add/change |
|------|-----|--------------------|
| New player-facing feature ships | **Code Writer** (or Architect if they authored) | New checklist items under the right section (logic / P2P / layout / clutter / smoke) so Debugger covers it next pass |
| Bug or regression found | **Debugger** (author) + Architect reviews | Add a **Regression lock** bullet: short name, repro hint, what “good” looks like — so it is checked every later pass |
| Feature removed / UX north-star change | **Architect** | Delete or rewrite obsolete checks; keep the doc aligned with product |

**Process**
1. Edit `docs/DEBUGGER_GUIDELINES.md` in the same PR/commit burst as the feature or fix when practical.
2. Ping Architect Squad: “guidelines updated: …” with a one-line reason.
3. Debugger’s next pass uses the updated file — pull before each checklist run.
4. Do not let the doc rot: if you noticed it mid-test and didn’t write it down, that is a process bug — add it before closing the pass.

### Regression locks (append-only until fixed-then-kept)

_Add entries below as `YYYY-MM-DD | severity | title — repro / expect`._

<!-- REGRESSION_LOCKS_START -->
- 2026-09-04 | P1 | P2P join fake/bad room code hangs on Connecting… — repro: Online → Join `TIGER-FAKE`; expect clear fail status/toast within a few seconds, not indefinite Connecting…
<!-- REGRESSION_LOCKS_END -->


## Hard rules

1. **Logic parity is sacred.** Rules engine, action legality, combat/siege resolution, card effects, turn/impulse sequencing, MCTS/ONNX move application, and P2P sync semantics must match pre-change behavior unless Architect explicitly approved a rules change (they have not). UI polish ≠ rules changes.
2. **Ping early.** If something breaks or looks wrong, message Architect **and** Code Writer in Architect Squad (or 1:1) with: repro steps, expected vs actual, file/area, screenshot path if any. Do not wait for end-of-iter summaries.
3. **No silent “fixes” that change design.** Minimal bugfixes OK after reporting; if the fix needs UX/architecture choice, stop and escalate to Architect.
4. **No Antigravity extra credits.** No merge to `main`.

---

## What to check every pass (checklist)

### A. Game logic (must stay the same)
- [ ] New game / Reset produces a legal starting position for both sides
- [ ] Human vs AI: human moves apply; AI responds; illegal clicks rejected
- [ ] Pass & Play: sides alternate correctly; Rest / Pass / Cancel behave as before
- [ ] Card play / trade / combat commit still resolve (spot-check British + Mysore hands)
- [ ] Turn advance / tired→fresh refresh still happens when expected
- [ ] Notation panel appends moves; scrubbing history doesn’t corrupt live state unexpectedly
- [ ] Draw / Resign still end or offer correctly
- [ ] Binary copy/load state still round-trips a mid-game position
- [ ] AI vs AI still runs without freezing the UI
- [ ] Compare suspicious behavior against `game/` Python engine or prior commit on `main` if needed

### B. P2P multiplayer
- [ ] Host creates room code; Copy works
- [ ] Join with code connects (or fails with clear status — not a silent hang)
- [ ] Moves sync both ways; desync symptoms reported immediately
- [ ] Room code lives in MP panel (no resurrecting header connection-pill clutter)
- [ ] Header **Online** opens Settings → Online multiplayer panel (progressive disclosure; no permanent MP chrome on board)
- [ ] Host status + room code + one-click Copy are obvious inside the MP panel; “share with opponent” affordance is clear

### C. UI layout / sizing (not too small, not too large)
- [ ] Desktop (≥1024): board readable; side columns usable; no overlapping drawers/board
- [ ] Tablet (768–1023) and mobile (≤767): faction / Moves tabs work; board not postage-stamp or overflowing viewport
- [ ] Buttons/toggles: hit targets usable; text not clipped; icons not cartoon-huge
- [ ] Tutorial + Settings drawers: open/close (Esc, ✕, outside); don’t permanently bury the board
- [ ] Esc closes Tutorial and Settings drawers; focus returns sensibly to the control that opened them
- [ ] Drawers expose `aria-expanded` / `aria-hidden` appropriately
- [ ] When Rest/Pass are visible, **R** and **P** shortcuts work (ignored while typing in inputs)
- [ ] Mobile ≤767: faction/Moves tabs usable (~30px hit targets); drawers don’t bury the board permanently
- [ ] Eval bar (when on): visible but not dominating the board

### D. Clutter / redundancy
- [ ] No duplicate controls that do the same thing without reason
- [ ] No dead UI (buttons that no-op, labels for removed features)
- [ ] No leftover empty panels/CSS chrome from deleted features
- [ ] First-run tutorial shows once; dismiss sticks (`localStorage`)
- [ ] No resurrected `.pill` / header-subtitle / connection-pill chrome
- [ ] `docs/PLAYER_QUICKSTART.md` present and matches current Host/Join + tutorial path

### E. Smoke commands
```bash
cd /workspace/TigersDay
git checkout architect/expansion && git pull
node --check public/script.js
node --check public/js/engine.js
node --check public/js/state.js
node --check public/js/mcts.js
node --check public/js/multiplayer.js
python -m http.server 8080 --directory public
```
- [ ] Optional: `docs/PLAYER_QUICKSTART.md` path exists and matches the live Host/Join + tutorial flow

---


### F. Iter B professional shell (added with Iter B)
- [ ] `docs/PLAYER_QUICKSTART.md` present and matches current Settings/Online/tutorial UX
- [ ] Settings mode labels are human-readable (not raw SNAKE_CASE)
- [ ] Esc closes tutorial/settings drawers; focus returns to opener
- [ ] `R` Rest / `P` Pass work when those header buttons are visible (not while typing in inputs)
- [ ] Mobile ≤767: drawers use constrained height; British/Mysore/Moves tabs usable
- [ ] No resurrected connection-pill / battle-status-bar / header-subtitle chrome

## Report format (to Architect + Code Writer)

```
BREAK or NIT | area | severity
Repro: ...
Expected: ...
Actual: ...
Suspect: path or component
Logic-regression?: yes/no/unclear
```

Severity: **P0** game-breaking / logic wrong · **P1** major UX/layout · **P2** nit/polish

---

## Cadence

- After each Code Writer push on `architect/expansion`, pull and re-run this checklist (at least A smoke + C + D).
- Between pushes, keep a local server up and spot-check while iterating.
- If Architect/Code Writer go quiet and you find P0/P1, ping anyway — that’s the point.
