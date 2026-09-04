# Expansion Notes — Iter B

**Branch:** `architect/expansion`  
**North star:** intuitive · professional · void of clutter

---

## What changed (Iter B)

- **Player Quickstart (`docs/PLAYER_QUICKSTART.md`):** Added a concise (~1–2 screens) player path explaining victory objectives, game modes, in-game tutorial, online multiplayer hosting/joining, AI evaluation bar, and keyboard shortcuts.
- **Settings & Copy Clarity:**
  - Replaced technical jargon with player-centric labels:
    - Mode selector: "PLAY MODE" with options "Human vs AI", "Pass & Play (Local)", "Online Multiplayer", and "AI vs AI (Spectate)".
    - Side selection: "Your side:".
    - Multiplayer: "Online multiplayer", "Join a room".
    - AI options: "Show AI evaluation bar", "AI strength (thinking time)".
    - State controls: "Game state", "Copy game state", "Load game state", with placeholder "Paste saved state…".
    - Removed player-facing mentions of "Stockfish-style", "WebAssembly", and "WebRTC".
  - Empty states: Updated move log message to "No moves yet — play to fill the log", and added defensive empty card deck guard.
  - Mode info label: Fixed `handleGameModeChange` to display human-readable labels (e.g. "Mode: Human vs AI") rather than uppercase SNAKE_CASE.
- **Keyboard & Accessibility (a11y):**
  - Added ARIA attributes across header controls and drawers (`aria-expanded`, `aria-controls`, `aria-hidden`, `role="region"`).
  - Tablist/tab ARIA roles on mobile faction tabs with dynamic `aria-selected` updating on tab change.
  - Sensible focus trapping and restoration: opening drawers places focus on the close button or first control; closing via Esc, close button, or backdrop returns focus to the button that opened the drawer.
  - Keyboard shortcuts for visible primary header actions: `R` for Rest Unit and `P` for Pass Phase when visible (guarded against active input/select fields to avoid key stealing); `Esc` closes open drawers or cancels current selection.
- **Mobile ≤767px Polish:**
  - Polished drawer container max-height and scrolling (`min(76vh, calc(100dvh - 46px - env(safe-area-inset-bottom)))`) across `.drawer-panel`, `#tutorial-drawer`, and `#settings-drawer` so drawers never awkwardly occlude the entire board.
  - Enhanced mobile faction tabs to 30px touch height with `touch-action: manipulation` without stealing board layout space.
- **Chrome Cleanup:**
  - Removed orphaned `.pill.disconnected` and `.pill.waiting` styles leftover from deleted connection-pill chrome.
  - Removed unused `.drawer-icon` and `.header-subtitle` styles.
  - Removed leftover battle sub-bar chrome (`.battle-status-bar`, `.battle-pill`, `.battle-details`, `.battle-vs`, `.battle-strength-pill`, `.battle-tip`) and unused `.review-icon` / `.mobile-tab-icon`.
  - Added `:focus-visible` rings for keyboard navigation and styled `.cards-empty-msg` empty-hand state.
  - Light board-card shadow declutter.

---

## What changed earlier (Iter A)

- **First-run tutorial:** auto-opens on first visit; dismiss via **Got it**, ✕, outside click, or Escape. Flag: `localStorage.tigersday_tutorial_dismissed`. Manual ⓘ toggle kept.
- **P2P Host/Join:** clearer status strings on `#p2p-status-pill`; room-code row click + **Copy room code** reuse `copyRoomCode()`; header **Online** opens settings → multiplayer panel (progressive disclosure, no permanent chrome).
- **Hygiene:** removed tracked `:memory:.ses`; `.gitignore` now ignores `*.ses` / `:memory:.ses`.
- **Audit fixes:** single `switchMobileFactionTab` (british / mysore / **moves**); stale card labels aligned to real hands (`Divide and Rule` / `Princely States`); dead DOM refs stripped (`connection-pill`, engine-lines chrome, debug toggle/console stubs).

---

## Remaining avenues (Iter C)

1. **Local save/load persistence** — persist saved binary states to `localStorage` with slot or timestamp management in addition to clipboard copy.
2. **Move notation export & sharing** — one-click copy of full match algebraic notation / PGN for post-game analysis or sharing.
3. **Model card & canonical weights** — record which `alphatigerv*.pt` checkpoint exported `public/alphatiger.onnx` before performing any weight cleanup.

---

## Non-goals (still)

- Never merge to `main`.
- No training rewrite / mass-delete of `.pt` weights.
- Browser UI does not `fetch()` FastAPI; treat `server.py` as research/Vercel path only.
