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

## Frontend north star (Kshitij)

Iterate the player UI toward:
- **Intuitive** — clear next action / turn phase; Host–Join and help easy to find
- **Professional** — consistent spacing, typography, control styling
- **Void of clutter** — progressive disclosure over permanent chrome; delete dead UI, don’t add noise

Applies to Iter A+ ongoing `public/` work.

## Iter B — Professional shell (autonomous)

Kshitij: keep going without asking. Still never merge to main.

1. `docs/PLAYER_QUICKSTART.md` — short play path (modes, tutorial, Online/P2P, AI eval)
2. Settings/empty-state copy pass — clearer labels, less jargon where players see it
3. Keyboard/a11y: sensible tab order + Esc closes drawers; primary actions reachable
4. Mobile ≤767px regression: faction/Moves tabs + drawers don’t fight the board
5. Clutter pass: remove leftover unused CSS from deleted chrome; tighten main board chrome only if it reduces noise

Success: commits on architect/expansion; EXPANSION_NOTES updated; node --check public/script.js; no merge; no .pt mass-delete; no training rewrite.

## Debugger

Strict QA guidelines: `docs/DEBUGGER_GUIDELINES.md`. Debugger keeps a local copy on `architect/expansion`, verifies logic parity + layout + anti-clutter, and pings Architect + Code Writer on breaks.

Debugger guidelines are a **living doc**: Code Writer adds checks when features ship; Debugger adds regression locks when breaks are found; Architect prunes obsolete items.

## Iter C — Persistence & share (autonomous)

1. Local save/load: persist binary game state to `localStorage` (named slot or timestamp list); load restores playable position; clear/overwrite safe
2. One-click export of full match algebraic notation from notation panel
3. Light clutter check after new controls — progressive disclosure, not new permanent chrome
4. Update EXPANSION_NOTES + DEBUGGER_GUIDELINES (living doc) with new checks
5. Model card stub only if cheap: note canonical ONNX source checkpoint in docs (no .pt deletes)

Success: architect/expansion only; logic parity; Debugger checklist updated; node --check OK.

## Iter D — Card & board visual themes (parked)

Kshitij approved parking after Iter C. CSS-variable skins on existing `.player-card` / seals / light board chrome. No new frameworks, no heavy assets, Settings theme picker, default = Campaign Parchment.

Themes (prototype 2–3 first, not all ten at once):
1. Campaign Parchment
2. Mysore Lacquer & Company Scarlet
3. Ink-Wash Campaign Map
4. Miniature Brass Instruments
5. Monsoon Glass (with solid fallback)
6. Sepoy Barracks Chalkboard
7. Rocket Festival Night
8. Treaty Ledger
9. Coastal Signal Flags
10. Twin Courts (split faction skins + phase token swap)

Living doc: add Debugger checks per theme (contrast, mobile, no layout break, logic unchanged).
