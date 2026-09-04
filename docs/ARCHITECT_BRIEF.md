# Architect Brief — TigersDay Expansion

**Branch:** `architect/expansion` only (never merge to `main`; Kshitij merges)  
**Owner:** Architect (design) → Code Writer (implement) → Debugger (failures)  
**Runtime:** Antigravity `agy` + Gemini 3.8 Flash; no extra/paid credits

## Product snapshot

Tiger’s Day is a dual-stack Anglo-Mysore Wars wargame:
- Browser: client-side rules + ONNX/MCTS AI + PeerJS P2P (`public/`)
- Research: PyTorch training / arena (`ai/`, `game/`) + FastAPI (`server.py`)

Recent `main` work already added notation/UI and removed some outdated features. Expansion should make the **player product** feel professional without rewriting the neural stack.

## Problem

A new visitor can play, but the product still feels like a research demo: weak first-run guidance, multiplayer friction, repo weight/cruft, and no clear “what’s next” roadmap in-repo.

## Goals (expansion track)

1. Useful player-facing polish (onboarding, MP UX, clarity)
2. Remove confirmed excess (safe hygiene only)
3. Document growth avenues for a more professional personal project

## Non-goals

- Do not merge to `main`
- Do not redesign rules, action space, or retrain from scratch in early iters
- Do not delete `ai/models/alphatigerv*.pt` blindly without a canonical-model decision + references checked
- Do not enable Antigravity extra credits

## Confirmed excess / hygiene candidates

| Item | Evidence | Action |
|------|----------|--------|
| `:memory:.ses` at repo root | Opaque session-like file; not referenced by game code | Delete + add to `.gitignore` |
| Historical `ai/models/alphatigerv{7,8,10,11,12}*.pt` | Runtime browser uses `public/alphatiger.onnx`; training writes `checkpoints/` | Keep v13 (+ betas) for now; document canonical; defer bulk delete to a dedicated hygiene PR |
| Giant README | Excellent for research, heavy for players | Add short `docs/PLAYER_QUICKSTART.md`; don’t gut README yet |

## Iteration plan

### Iter A — First-run + hygiene (S–M) **← start here**
- Auto-open tutorial drawer on first visit (`localStorage` flag); keep manual `(i)` toggle
- One-click “Copy room code” + clearer Host/Join status copy in P2P panel
- Remove `:memory:.ses`; gitignore `*.ses` / `:memory:.ses`
- Add `docs/EXPANSION_NOTES.md` (what changed + next avenues)
- Smoke: static `public/index.html` loads; tutorial + MP panel still work

### Iter B — Professional shell (M)
- Player quickstart doc; tighten settings labels / empty states
- Keyboard focus path for primary board actions (a11y pass)
- Mobile tab polish if regressions found at ≤767px

### Iter C — Growth features (M–L) (pick after A/B)
- Local save/load game (binary state already copyable — persist to `localStorage`)
- Spectate / share notation export improvements
- Optional: model card for `alphatiger.onnx` + which `.pt` exported it

## Success criteria (Iter A)

- [ ] All commits on `architect/expansion` only
- [ ] First visit shows tutorial once; subsequent visits respect dismiss
- [ ] Host can copy room code in one click
- [ ] `:memory:.ses` gone from tree; ignored going forward
- [ ] `docs/EXPANSION_NOTES.md` lists 3–5 next professional avenues
- [ ] No merge to `main`; no Antigravity extra credits

## Handoff

Code Writer implements Iter A via `agy` (`--model gemini-3.8-flash --effort high`) in `/workspace/TigersDay`. Debugger owns red CI / conflicts only if they appear.

## Audit addendum (evidence pass)

Fold into Iter A if not already done:
- Deduplicate `switchMobileFactionTab` in `public/script.js` (Moves tab broken)
- Strip dead DOM refs from cleanup (`engine-lines-container`, `connection-pill`, debug console, etc.)
- Fix stale “Diplomatic Mission” string vs real card names
- P2P: `copyRoomCode` exists — improve discoverability/status, don’t reinvent
- Browser UI does not `fetch()` the FastAPI server; treat server as research/Vercel path only
