# Tiger's Day

Board game of the Anglo-Mysore Wars (1767–1799)
Play a friend online or AlphaTiger, an AlphaZero-style AI, in the browser

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

If an army in all 5 keys, British win
At the end of turn 4, Mysore win

### Game Properties

- Decisive: always a winner
- Acyclic: state cannot repeat
- Finite: will always end
- Stochastic: affected by luck
- Transparent: perfect information
- Asymmetric: players are different

## AI

- MLP with policy and value heads, guided by MCTS
- Trained by self-play, starting from end-game positions
- Policy covers 959 actions

### State

148-bit input vector, in order:

- 12: unused cards, 6 per side
- 75: fresh army, tired army, or fort for each of 25 spaces, all zero if empty
- 4: turn, one-hot
- 3: impulse phase, one of British move, Mysore card, British card
- 4: Mysore card strength in the current battle, 0–3
- 50: attacker and defender space, one-hot each

### Browser

- Network runs as ONNX in onnxruntime-web (WebAssembly)
- MCTS in JavaScript with PUCT, Dirichlet root noise, and early stopping
- Search drives the eval bar and top lines

## Multiplayer

- PeerJS (WebRTC), browser to browser
- PeerJS cloud server connects peers, STUN for NAT traversal
- Host shares a `TIGER-` room code
- Moves and luck outcomes are sent so both clients stay in sync

## Project Structure

```text
ai/        training, search, arena, models (Python)
game/      rules (Python)
public/    web app with JavaScript ports of rules and MCTS
api/       FastAPI endpoint for Vercel
server.py  local server with the Python AI
```

## Running Locally

```bash
python3 -m http.server 8000 --directory public
```

Open `http://localhost:8000`
`python server.py` serves the same UI with the Python AI instead

## Training

CUDA is faster than CPU is faster than MPS
Training picks CUDA, then MPS, then CPU, so pass `--device cpu` on Apple Silicon

1. Install dependencies
   ```bash
   pip install -r requirements-dev.txt
   ```
2. Train, saving `checkpoints/final.pt`
   ```bash
   python -m ai.train
   ```
3. Save the weights as `ai/models/pt/alphatigervN.pt` and export to ONNX
   ```bash
   python -m ai.onnx --input ai/models/pt/alphatigervN.pt --output ai/models/onnx/alphatigervN.onnx
   ```
4. Deploy by copying it to `public/alphatiger.onnx` and committing
5. Regenerate the tutorial's version, Elo, and release date, then commit `public/model-info.js`
   ```bash
   python3 ai/model_info.py
   ```

### Model Versions

Models named `alphatiger`
`v7` version number for later models
`beta` used for pretrain only models
ONNX for inference, PT for training

## Elo Ratings

Pairwise tournament to sample matchups
Joint Bradley-Terry maximum likelihood estimation
Anchor with v8 at 1000 Elo
Round to nearest 50 due to error bars
Sufficient games for +-25 Elo 95% CI
