# Expansion Notes — Iter A

**Branch:** `architect/expansion`  
**North star:** intuitive · professional · void of clutter

## What changed (Iter A)

- **First-run tutorial:** auto-opens on first visit; dismiss via **Got it**, ✕, outside click, or Escape. Flag: `localStorage.tigersday_tutorial_dismissed`. Manual ⓘ toggle kept.
- **P2P Host/Join:** clearer status strings on `#p2p-status-pill`; room-code row click + **Copy room code** reuse `copyRoomCode()`; header **Online** opens settings → multiplayer panel (progressive disclosure, no permanent chrome).
- **Hygiene:** removed tracked `:memory:.ses`; `.gitignore` now ignores `*.ses` / `:memory:.ses`.
- **Audit fixes:** single `switchMobileFactionTab` (british / mysore / **moves**); stale card labels aligned to real hands (`Divide and Rule` / `Princely States`, not “Diplomatic Mission”); dead DOM refs stripped (`connection-pill`, engine-lines chrome, debug toggle/console stubs).

## Next professional avenues (from brief Iter B/C)

1. **Player quickstart** — short `docs/PLAYER_QUICKSTART.md`; tighten settings labels and empty states without gutting the research README.
2. **Keyboard / a11y path** — focus order for primary board actions (select unit → destination → Rest/Pass/Cancel).
3. **Mobile shell polish** — regression pass at ≤767px for faction/Moves tabs and drawer stacking.
4. **Local save/load** — persist binary game state to `localStorage` (clipboard copy already exists).
5. **Model card** — document which `alphatigerv*.pt` exported `public/alphatiger.onnx`; defer bulk `.pt` deletes until that canonical decision is recorded.

## Non-goals (still)

- No merge to `main`
- No training rewrite / mass-delete of `.pt` weights
- Browser UI does not `fetch()` FastAPI; treat `server.py` as research/Vercel path only
