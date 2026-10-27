# 🐅 Tiger’s Day – Anglo-Mysore Wars (1767 – 1799)

Welcome to **Tiger’s Day**, a strategic board game simulating the historical Anglo-Mysore Wars between Tipu Sultan (Sultanate of Mysore) and Lord Cornwallis (British East India Company). 

This project features a **pure client-side WebAssembly game engine**, an **AlphaZero-inspired Deep Neural Network + Monte Carlo Tree Search (MCTS) AI**, and **real-time Peer-to-Peer (P2P) WebRTC multiplayer**, capable of running 100% in the browser with zero server dependencies.

---

## 📖 Game Rules

4 turn game - each turn is 1 of the 4 Anglo-Mysore Wars
Armies - move once per war, for the British
Forts - normally stationary, for Mysore
Only a single army or fort per space
Cannot deploy armies or forts on occupied space
*Move to adjacent unless specified
*Fort can never move into army

Tipu Sultan vs Cornwallis
Sultanate of Mysore vs East India Company

TURN SEQUENCE
British and Mysore draw all 6 cards
All British armies become Fresh
Play impulses until all armies are Tired

IMPULSE
British Tire a Fresh army and may move it
Mysore may play a card
British may play a card
Battles resolved if any

BATTLES when an army moves into a fort:
Army remains at starting point
Resolve at end of impulse simultaneously
Moving out cancels battle
Strength = 1 per adjacent fort or army + number if card
Adjacency counted on defending fort
Higher Strength wins, ties go to Mysore
If army loses, it does not move
If fort loses, remove it and army moves in
Losing player must discard
*Never move into existing defender

CARDS can be played for:
Card text
Draw a card of lower number
Add Strength = number to unresolved battle
*Discard means randomly throw a card away
*Draw means pick up a used card
*Keep all cards face up

3 Iron Rockets
2 Sepoy Mutiny - remove an army not in a key
2 French Alliance - deploy a fort adjacent to existing fort
1 Cavalry Raid - British discard
1 Sea Trade - move a fort from Coast to any
1 Monsoon - flip a Fresh army to Tired

3 Wall Breach
2 Highlanders - deploy a Fresh army on Coast
2 Royal Navy - move an army to any Coast
1 Divide and Rule - move a fort not in a key
1 Force March - move a Tired army
1 Princely States - deploy a Tired army in a key

Coast refers to spaces that touch the ocean
Key refers to square victory spaces
Subscripts refer to game setup

If an army in all 5 keys, British win
At the end of turn 4, Mysore win

### Game Properties

- Decisive: always a winner
- Acyclic: state cannot repeat
- Finite: will always end
- Stochastic: affected by luck
- Transparent: perfect information
- Asymmetric: players are different

---

## 🧠 AI & WebAssembly Architecture

The AI is built on Deep Multi-Layer Perceptrons (MLP) paired with Monte Carlo Tree Search (MCTS), trained via reinforcement self-play.

### Game State Representation
The complete board state is serialized into a compact **148-bit binary vector**:
* **British & Mysore Cards (12 bits):** Active/exhausted status of each faction's 6 cards.
* **Board Nodes (75 bits):** 25 territories $\times$ 3 one-hot states (*Fresh Army, Tired Army, Fort, Empty*).
* **Turn Counter (4 bits):** One-hot encoding of Turns 1–4.
* **Impulse Indicator (3 bits):** *British Move*, *Mysore Card*, or *British Card*.
* **Combat State (54 bits):** Attacker location, Defender location, and Mysore committed battle strength (0–3).

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
│   ├── engine.py            # Legal move masking & move space dict (959 actions)
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

CUDA is faster than CPU is faster than MPS
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
   After committing the browser weights, run `python3 scripts/model_info.py` and commit
   `public/model-info.js`. This matches the deployed weights to the model manifest and
   generates the tutorial's version, estimated Elo, and browser release date from Git.

### Model Versions

Models named `alphatigerv`
`v7` version number for later models
`beta` used for pretrain only models
ONNX for inference, PT for training

## Elo Ratings

Pairwise tournament to sample matchups
Joint Bradley-Terry maximum likelihood estimation
Anchor with alphatiger v8 at 1000 Elo
Round to nearest 100 due to error bars
