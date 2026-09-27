import os
import sys
import random
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure root workspace is in Python path for Vercel's execution environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.constants import INDEX_MAP, WHO_TO_MOVE, MOVE_VECTOR_LENGTH, DEFAULT_SIMS
from game.state import GameState
from game.engine import get_legal_moves, legal_moves_dict
from game.updater import get_next_state, get_state_winner, get_luck_outcomes
from game.replay import notate, interpret
from ai.mcts import MCTS
from ai.neural import load_ai_model

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(title="Tiger's Day API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global AI Model Singletons (Cached across warm serverless invocations)
# ---------------------------------------------------------------------------
ai_model_british = load_ai_model()
ai_model_mysore = ai_model_british
DEFAULT_MCTS_SIMS = 250  # Tuned for fast sub-second serverless response


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class MoveRequest(BaseModel):
    state_str: str
    move_idx: int

class LoadRequest(BaseModel):
    state_str: str

class EvalStepRequest(BaseModel):
    state_str: str
    batch_size: int = 200

class HistoryRequest(BaseModel):
    replay_log: List[int]


# ---------------------------------------------------------------------------
# Game State Serialization & Helpers
# ---------------------------------------------------------------------------
def _resolve_luck(state: GameState) -> GameState:
    """Resolves stochastic luck states using uniform random outcome selection."""
    while state.is_luck:
        outcomes = get_luck_outcomes(state)
        idx = random.randrange(len(outcomes))
        state = outcomes[idx]
    return state

def _ai_move(state: GameState, sims: int = DEFAULT_MCTS_SIMS) -> GameState:
    """Executes MCTS on the current board state and applies the best move."""
    current_side = str(WHO_TO_MOVE[state.to_move]).lower()
    active_model = ai_model_british if "british" in current_side else ai_model_mysore
    
    mcts = MCTS(active_model, simulations=sims, depsilon=0)
    best_move, _ = mcts.find_move(state)
    
    next_state = get_next_state(state, best_move)
    return _resolve_luck(next_state)

def generate_game_data(state: GameState) -> Dict[str, Any]:
    """Translates Python GameState into structured JSON for frontend rendering."""
    nodes = []
    for i, name in enumerate(INDEX_MAP.values()):
        if state.fresh_armies[i]:
            a_type = "fresh"
        elif state.tired_armies[i]:
            a_type = "tired"
        elif state.forts[i]:
            a_type = "fort"
        else:
            a_type = "empty"
        nodes.append({"name": name, "armyType": a_type})

    mask = get_legal_moves(state)
    try:
        moves = legal_moves_dict(mask)
    except Exception:
        moves = {int(i): f"Move ID {i}" for i in np.where(mask)[0]}

    return {
        "state_str": "".join(["1" if bool(x) else "0" for x in state.vector]),
        "winner": get_state_winner(state),
        "moves": moves,
        "match_mode": "human_vs_ai",
        "human_side": "british",
        "ui_state": {
            "british_cards": [bool(x) for x in state.british_cards],
            "mysore_cards": [bool(x) for x in state.mysore_cards],
            "turn": int(state.turn),
            "who_to_move": WHO_TO_MOVE[state.to_move],
            "attacker": INDEX_MAP.get(state.attacker, "None") if state.attacker != -1 else "None",
            "defender": INDEX_MAP.get(state.defender, "None") if state.defender != -1 else "None",
            "card_strength": int(state.card_strength) if state.card_strength != -1 else 0,
            "nodes": nodes
        }
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/init")
async def init_game():
    """Initializes a new game board and resolves opening luck states."""
    state = GameState()
    state.default_setup()
    state = _resolve_luck(state)
    return generate_game_data(state)

@app.post("/api/load-state")
async def load_state(req: LoadRequest):
    """Loads a game state from a 143-bit binary string."""
    try:
        state = GameState().read_str(req.state_str)
        return generate_game_data(state)
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/play-move")
async def play_move(req: MoveRequest):
    """Executes a move submitted by the human player and resolves luck."""
    try:
        state = GameState().read_str(req.state_str)
        state = get_next_state(state, req.move_idx)
        state = _resolve_luck(state)
        return generate_game_data(state)
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/play-ai")
async def play_ai(req: LoadRequest):
    """Calculates and executes an AI move using MCTS and neural network."""
    try:
        state = GameState().read_str(req.state_str)
        if get_state_winner(state) != 0:
            return generate_game_data(state)

        state = _ai_move(state)
        return generate_game_data(state)
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/eval-step")
async def eval_step(req: EvalStepRequest):
    """Performs batched MCTS rollout simulations for the live evaluation bar."""
    try:
        state = GameState().read_str(req.state_str)
        current_side = str(WHO_TO_MOVE[state.to_move]).lower()
        active_model = ai_model_british if "british" in current_side else ai_model_mysore
        
        mcts = MCTS(active_model, simulations=min(req.batch_size, 300), depsilon=0)
        mcts.search(state, stop=False)
        
        score = float(mcts.root.eval) if mcts.root else 0.0
        best_children = sorted(
            mcts.root.children.items(),
            key=lambda item: item[1].visit_count,
            reverse=True
        ) if mcts.root else []

        top_moves = []
        for move, node in best_children[:3]:
            pv_line = [notate(state, move)]
            top_moves.append({"move_name": " ".join(pv_line), "eval": float(node.eval)})

        return {
            "eval_score": score,
            "total_sims": req.batch_size,
            "top_moves": top_moves
        }
    except Exception as e:
        return {"eval_score": 0.0, "total_sims": 0, "top_moves": [], "error": str(e)}

@app.post("/api/get-notation")
async def get_notation(req: HistoryRequest):
    """Translates move indices into historical algebraic notation."""
    try:
        algebraic_str, _ = interpret(req.replay_log)
        return {"notation": algebraic_str}
    except Exception as e:
        return {"error": str(e)}