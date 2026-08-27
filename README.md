# 🐅 Tiger’s Day – Anglo-Mysore Wars (1767 – 1799)

Welcome to **Tiger’s Day**, a strategic board game simulating the historical Anglo-Mysore Wars between Tipu Sultan (Sultanate of Mysore) and Lord Cornwallis (British East India Company). 

This project features a **pure client-side WebAssembly game engine**, an **AlphaZero-inspired Deep Neural Network + Monte Carlo Tree Search (MCTS) AI**, and **real-time Peer-to-Peer (P2P) WebRTC multiplayer**, capable of running 100% in the browser with zero server dependencies.

---

## 📖 The Game: Tiger's Day

### Overview
* **Asymmetric Factions:** Sultanate of Mysore (Defending with Forts & Rocket Artillery) vs. British East India Company (Attacking with Armies & Naval Superiority).
* **Duration:** 4 turns representing the four Anglo-Mysore Wars.
* **Victory Conditions:**
  * **British Victory:** Occupy all 5 Key Cities (*Bombay, Hyderabad, Madras, Seringapatam, Coimbatore*) with British armies.
  * **Mysore Victory:** Survive through Turn 4 while preventing British occupation of all 5 Key Cities, or eliminate all British armies.

### Core Rules
* **Impulse System:** Players alternate impulses until all British armies are exhausted (*Tired*).
* **Combat & Siege:** Moving into an enemy fort triggers a battle. The combat winner is determined by:
  $$\text{Net Strength} = \text{Attacking Armies} - \text{Defending Forts} + \text{Net Card Strength}$$
  If $\text{Net Strength} > 0$, the British capture the fort; otherwise, Mysore holds.
* **Card Play & Trading:** Cards can be played for unique tactical abilities (*e.g., Cavalry Raid, French Alliance, Royal Navy*), committed for combat strength, or traded to reclaim exhausted cards.

---

## 🧠 AI & WebAssembly Architecture

The AI is built on Deep Multi-Layer Perceptrons (MLP) paired with Monte Carlo Tree Search (MCTS), trained via reinforcement self-play.

### Game State Representation
The complete board state is serialized into a compact **148-bit binary vector**:
* **British & Mysore Cards (12 bits):** Active/exhausted status of each faction's 6 cards.
* **Board Nodes (75 bits):** 25 territories $\times$ 3 one-hot states (*Fresh Army, Tired Army, Fort, Empty*).
* **Turn Counter (4 bits):** One-hot encoding of Turns 1–4.
* **Impulse Indicator (3 bits):** *British Move*, *Mysore Card*, or *British Card*.
* **Combat State (33 bits):** Attacker location, Defender location, and Mysore committed battle strength (0–3).

### Client-Side ONNX WebAssembly Inference
* The trained PyTorch model (`AlphaTiger`) is exported to **ONNX** format (`alphatiger.onnx`).
* The browser runs **`onnxruntime-web` (WebAssembly)** to perform high-speed neural network evaluations on the client CPU without requiring a backend GPU/Python server.

### Browser MCTS Loop
* Performs real-time tree rollouts combining neural policy priors ($p$) and state value evaluations ($v$).
* Features PUCT exploration bonuses, Dirichlet root noise, and early stopping.
* Drives the **Stockfish-style live evaluation bar** and computes the top candidate engine lines in real time.

---

## 🌐 Real-Time P2P WebRTC Multiplayer

* Built on **PeerJS (WebRTC)** for zero-server, direct browser-to-browser online play.
* **Room Codes:** Host a match to generate a short room code (*e.g., `TIGER-ABCD`*), which a friend can enter to connect immediately.
* **Deterministic State Sync:** Synchronizes move actions, exact stochastic luck trajectories (*e.g., random battle discards, cavalry raids*), and full 148-bit state vectors to guarantee both clients stay in lockstep.

---

## 📁 Project Structure

```text
TigersDay/
├── ai/                      # Machine Learning & Training Pipeline (Python)
│   ├── models/              # Neural network checkpoints (.pt and exported .onnx)
│   ├── export_to_onnx.py    # Exporter from PyTorch to ONNX
│   ├── mcts.py              # Python Monte Carlo Tree Search
│   ├── neural.py            # Hybrid PyTorch & ONNX model definitions
│   ├── train.py             # Self-play training loop
│   └── arena.py             # Model evaluation tournament arena
│
├── game/                    # Core Python Game Logic & Rules
│   ├── constants.py         # 25-node map geometry, edges, card values, move spaces
│   ├── engine.py            # Legal move masking & move space dict (953 actions)
│   ├── state.py             # 148D GameState vector management
│   ├── updater.py           # Battle resolution & state transition logic
│   └── replay.py            # Algebraic notation & game replay parser
│
├── public/                  # 100% Client-Side Web Application
│   ├── js/
│   │   ├── state.js         # JavaScript GameState & 148D vector port
│   │   ├── engine.js        # JavaScript rule engine, combat, & luck resolution
│   │   ├── mcts.js          # Client-side MCTS & onnxruntime-web inference
│   │   └── multiplayer.js   # PeerJS WebRTC P2P multiplayer controller
│   ├── alphatiger.onnx      # Lightweight WebAssembly neural network weights
│   ├── index.html           # Interactive SVG board UI, drawers, & controls
│   ├── script.js            # UI event controller & game coordinator
│   └── style.css            # Historic Deccan parchment design system
│
├── api/                     # Optional Python Serverless API entrypoint
│   └── index.py             # FastAPI backend for serverless environments
│
├── vercel.json              # Vercel deployment configuration
├── requirements.txt         # Production dependencies
└── README.md                # Project documentation
```

---

## 🚀 Running Locally

Because the web application is 100% client-side, you can host it with any static file server:

### Using Python
```bash
python3 -m http.server 8000 --directory public
```

### Using Node / npx
```bash
npx serve public -p 8000
```

Open your browser at **`http://localhost:8000`**.

---

## 🏋️ Training the AI Locally

If you want to train or evaluate new AI models:

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
2. Run the self-play training loop:
   ```bash
   python -m ai.train
   ```
3. Export new weights to ONNX:
   ```bash
   python -m ai.export_to_onnx
   ```
   Copy the exported `alphatiger.onnx` into `public/alphatiger.onnx` for browser gameplay.