# Project Rules and Invariants for Tiger's Day

## Replay & Card Presentation Cleanliness (PERMANENT RULE)
1. **NO Status Pills on Cards**: NEVER add artificial `ACTIVE` or `USED` status pill badges to cards in replay/history mode (`.card-replay-status`). The natural card styling and existing `EXHAUSTED` stamp (`.card-exhausted-stamp`) already clearly show card state. Adding extra text pills creates unwanted clutter.
2. **NO Textual Card HUD Lists in Banners**: NEVER list out card names as text strings in banners (e.g. `#review-banner-cards-hud` or `#hist-mysore-cards-list`). The card decks on the left and right already display all cards.
3. **NO Hand Count Badges (e.g. X/6)**: NEVER add badges showing how many cards can be played (e.g. `mysore-hand-count`, `british-hand-count`, `mobile-mysore-count`, `mobile-british-count`) to faction column headers or mobile tabs.
4. **Permanent Invariant**: These markers make the user interface worse and must NEVER be re-implemented under any circumstances.

## Git Protocol
- **NEVER push to remote repository**: The user pushes code manually (`never push yourself, I will do that`). Commit locally only.
