# Project Rules and Invariants for Tiger's Day

## Replay & Card Presentation Cleanliness (PERMANENT RULE)
1. **NO Status Pills on Cards**: NEVER add artificial `ACTIVE` or `USED` status pill badges to cards in replay/history mode (`.card-replay-status`). The natural card styling and existing `EXHAUSTED` stamp (`.card-exhausted-stamp`) already clearly show card state. Adding extra text pills creates unwanted clutter.
2. **NO Textual Card HUD Lists in Banners**: NEVER list out card names as text strings in banners (e.g. `#review-banner-cards-hud` or `#hist-mysore-cards-list`). The card decks on the left and right already display all cards.
3. **NO Hand Count Badges (e.g. X/6)**: NEVER add badges showing how many cards can be played (e.g. `mysore-hand-count`, `british-hand-count`, `mobile-mysore-count`, `mobile-british-count`) to faction column headers or mobile tabs.
4. **NO Floating / Extra Review Popup Banners**: NEVER add floating review popup banners above the board (e.g. `#historical-review-banner`). Historical mode state is already cleanly and seamlessly shown in the main `#turn-header` and step controls (`#btn-step-live`).
5. **Clean Button Labels**: Keep action buttons simple and clean (e.g. use `Export Replay`, NEVER append file extensions like `(.tdr)`).
6. **Permanent Invariant**: These markers and popups make the user interface worse and must NEVER be re-implemented under any circumstances.

## Git Protocol
- **NEVER push to remote repository**: The user pushes code manually (`never push yourself, I will do that`). Commit locally only.
