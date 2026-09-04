# Tiger’s Day — Player Quickstart

A quick guide to playing **Tiger’s Day**, a strategic board wargame set during the Anglo-Mysore Wars (1767–1799) between Tipu Sultan (*The Tiger of Mysore*) and the British East India Company.

---

## 🎯 The Objective

- **British East India Co. (Attacker):** Win immediately by occupying all **5 Key Cities** (*Bombay, Hyderabad, Madras, Seringapatam, Coimbatore*) simultaneously with armies.
- **Sultanate of Mysore (Defender):** Win by defending through the end of **Turn 4** without the British holding all 5 Keys, or by decimating British forces.

---

## 🚀 How to Start a Game

1. **Open the Game:** Load `public/index.html` in any modern web browser (or serve the `public/` directory statically).
2. **Choose Play Mode:** Open **Settings** (⚙ icon in the header) and choose:
   - **Human vs AI** *(default)*: Play against the client-side neural AI.
   - **Pass & Play (Local)**: Two players take turns on the same device.
   - **Online Multiplayer**: Peer-to-peer match with a friend over the internet.
   - **AI vs AI (Spectate)**: Watch the neural network play both sides.
3. **Pick Your Side:** Select **British East India Company** or **Sultanate of Mysore** under **Your side:**.

---

## 📖 In-Game Tutorial & Rules

- On your first visit, the **Rules & Controls** drawer opens automatically.
- Reopen it anytime by clicking the **ⓘ** button in the top header.
- **Turn Flow:**
  1. **Phase 0 (British Move):** Move a Fresh Army to an adjacent territory, attack a Fort, or rest in place.
  2. **Phase 1 (Mysore Card):** Play a tactical card, commit strength to an ongoing battle, trade a card, or pass.
  3. **Phase 2 (British Card):** Play a tactical card, commit strength, trade, or pass. Combat resolves.
- **Turn Refresh:** When all British armies become tired, the turn counter advances (+1), tired armies become fresh, and all exhausted cards return to your hand.

---

## 🌐 Online Multiplayer (Host / Join)

Play directly with another person without installing anything:

1. Click the **Online** button in the header (or select *Online Multiplayer* in Settings).
2. **To Host:**
   - Click **Host Room**.
   - Click the room code box or **Copy room code** to copy your unique code (e.g. `TIGER-ABCD`).
   - Send the code to your friend.
3. **To Join:**
   - Enter your friend's room code under **Join a room** and click **Join**.
4. Once connected, the host plays British and the guest plays Mysore. The game updates move-by-move in real time.

---

## 🧠 AI Evaluation & Strength

- **Show AI evaluation bar:** Enable this toggle in Settings to view live win probability during AI play (green for Mysore, red for British).
- **AI strength (thinking time):** Adjust the slider in Settings to change the search depth (Monte Carlo Tree Search simulations). Lower values play faster; higher values think deeper.

---

## ⌨️ Controls & Keyboard Shortcuts

- **Mouse / Tap:** Click an active army to highlight valid destination territories. Click a card body to activate its power, or click the circular wax seal to commit battle strength.
- **Esc:** Close any open drawer (Tutorial or Settings), or cancel an active piece/card selection.
- **R:** Rest active unit (when *Rest Unit* is available in the header).
- **P:** Pass phase (when *Pass Phase* is available in the header).
- **History & Notation:** Switch to the **Moves** tab to review algebraic notation, and use the history stepper buttons (`⏮ ◀ ▶ ⏭`) to inspect earlier positions.
