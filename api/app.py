import os
import sys
import random
import asyncio
from collections import OrderedDict
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure root workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from game.constants import INDEX_MAP, WHO_TO_MOVE, MOVE_VECTOR_LENGTH, DEFAULT_SIMS
from game.state import GameState
from game.engine import get_legal_moves, legal_moves_dict
from game.updater import get_next_state, get_state_winner, get_luck_outcomes
from game.replay import notate, interpret
from ai.mcts import MCTS
from ai.neural import load_ai_model

# ---------------------------------------------------------------------------
# Strict Input Validation Schemas
# ---------------------------------------------------------------------------
STATE_REGEX = r"^[01]{148}$"

class MoveRequest(BaseModel):
    state_str: str = Field(..., pattern=STATE_REGEX, description="148-bit binary game state string")
    move_idx: int = Field(..., ge=0, lt=MOVE_VECTOR_LENGTH, description="Action index (0-958)")

class LoadRequest(BaseModel):
    state_str: str = Field(..., pattern=STATE_REGEX, description="148-bit binary game state string")

class AiMoveRequest(BaseModel):
    state_str: str = Field(..., pattern=STATE_REGEX, description="148-bit binary game state string")
    sims: Optional[int] = Field(default=250, ge=1, le=50000, description="MCTS rollout simulations count")

class EvalStepRequest(BaseModel):
    state_str: str = Field(..., pattern=STATE_REGEX, description="148-bit binary game state string")
    batch_size: int = Field(default=200, ge=1, le=5000, description="Number of MCTS simulations to step")

class HistoryRequest(BaseModel):
    replay_log: List[int] = Field(..., description="Array of executed move indices")


# ---------------------------------------------------------------------------
# Thread-Safe LRU Cache for MCTS Evaluation Trees
# ---------------------------------------------------------------------------
class EvalTreeLRUCache:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    async def set(self, key: str, value: Dict[str, Any]):
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    async def clear(self):
        async with self.lock:
            self.cache.clear()


# ---------------------------------------------------------------------------
# Core Helpers
# ---------------------------------------------------------------------------
def resolve_luck_stochastic(state: GameState) -> GameState:
    """Resolves stochastic luck states using uniform random outcome selection."""
    while state.is_luck:
        outcomes = get_luck_outcomes(state)
        if not outcomes:
            break
        idx = random.randrange(len(outcomes))
        state = outcomes[idx]
    return state

def generate_game_data(state: GameState, match_mode: str = "human_vs_ai", human_side: str = "british") -> Dict[str, Any]:
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
        "state_str": str(state),
        "winner": int(get_state_winner(state)),
        "moves": moves,
        "match_mode": match_mode,
        "human_side": human_side,
        "ui_state": {
            "british_cards": [bool(b) for b in state.british_cards],
            "mysore_cards": [bool(b) for b in state.mysore_cards],
            "turn": int(state.turn),
            "who_to_move": WHO_TO_MOVE[state.to_move],
            "attacker": INDEX_MAP[state.attacker] if state.attacker != -1 else "None",
            "defender": INDEX_MAP[state.defender] if state.defender != -1 else "None",
            "card_strength": int(state.card_strength),
            "nodes": nodes
        }
    }


# ---------------------------------------------------------------------------
# Application Factory
# ---------------------------------------------------------------------------
def create_app(mount_static: bool = False) -> FastAPI:
    app = FastAPI(title="Tiger's Day API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lazy load neural models on startup
    ai_models = {}
    eval_cache = EvalTreeLRUCache(capacity=32)

    def get_models():
        if not ai_models:
            model = load_ai_model()
            ai_models["british"] = model
            ai_models["mysore"] = model
        return ai_models

    @app.get("/api/init")
    async def init_game():
        state = GameState()
        state.default_setup()
        state = resolve_luck_stochastic(state)
        return generate_game_data(state)

    @app.post("/api/load-state")
    async def load_state(req: LoadRequest):
        try:
            state = GameState().read_str(req.state_str)
            state = resolve_luck_stochastic(state)
            return generate_game_data(state)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/play-move")
    async def play_move(req: MoveRequest):
        try:
            state = GameState().read_str(req.state_str)
            legal_mask = get_legal_moves(state)
            if not legal_mask[req.move_idx]:
                raise HTTPException(status_code=400, detail=f"Move {req.move_idx} is not legal in this state")

            next_state = get_next_state(state, req.move_idx)
            next_state = resolve_luck_stochastic(next_state)
            return generate_game_data(next_state)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/play-ai")
    async def play_ai(req: AiMoveRequest):
        try:
            state = GameState().read_str(req.state_str)
            models = get_models()
            current_side = str(WHO_TO_MOVE[state.to_move]).lower()
            active_model = models["british"] if "british" in current_side else models["mysore"]

            mcts = MCTS(active_model, simulations=req.sims, depsilon=0)
            best_move, policy = mcts.find_move(state)
            next_state = get_next_state(state, best_move)
            next_state = resolve_luck_stochastic(next_state)

            return {
                "best_move": int(best_move),
                "notation": notate(state, best_move),
                "eval": float(mcts.root.eval) if mcts.root else 0.0,
                "game_data": generate_game_data(next_state)
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/eval-step")
    async def eval_step(req: EvalStepRequest):
        try:
            state = GameState().read_str(req.state_str)
            models = get_models()
            current_side = str(WHO_TO_MOVE[state.to_move]).lower()
            active_model = models["british"] if "british" in current_side else models["mysore"]

            tree_data = await eval_cache.get(req.state_str)
            if tree_data is None:
                mcts_instance = MCTS(active_model, simulations=req.batch_size, depsilon=0)
                tree_data = {
                    "mcts": mcts_instance,
                    "total_sims": 0
                }
                await eval_cache.set(req.state_str, tree_data)

            mcts = tree_data["mcts"]
            mcts.simulations = req.batch_size
            mcts.search(state, stop=False)
            tree_data["total_sims"] += req.batch_size

            score = float(mcts.root.eval) if mcts.root else 0.0
            best_children = sorted(
                mcts.root.children.items(),
                key=lambda item: item[1].visit_count,
                reverse=True
            ) if mcts.root else []

            top_moves = []
            for move, node in best_children[:3]:
                top_moves.append({
                    "move": int(move),
                    "notation": notate(state, move),
                    "visits": node.visit_count,
                    "prior": float(node.prior),
                    "score": float(node.eval)
                })

            return {
                "score": score,
                "total_sims": tree_data["total_sims"],
                "top_moves": top_moves
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/get-notation")
    async def get_notation(req: HistoryRequest):
        try:
            algebraic, _ = interpret(req.replay_log)
            return {"notation": algebraic}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Optional static assets mounting (for local server execution)
    if mount_static:
        public_dir = os.path.join(BASE_DIR, "public")
        if os.path.exists(public_dir):
            app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

    return app
