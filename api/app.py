import os
import sys
import random
import asyncio
from collections import OrderedDict
from typing import List, Dict, Any, Optional

import json
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
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

class MatchRecordRequest(BaseModel):
    winner_handle: str = Field(..., min_length=1, max_length=50)
    loser_handle: str = Field(..., min_length=1, max_length=50)
    is_draw: bool = False
    winner_faction: str = "british"

from api.leaderboard import global_leaderboard
from api.metrics import global_metrics, global_rate_limiter


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

    @app.middleware("http")
    async def rate_limiting_and_metrics_middleware(request: Request, call_next):
        import time
        start_t = time.perf_counter()
        client_ip = request.client.host if request.client else "127.0.0.1"

        # Rate limit compute-intensive endpoints
        if request.url.path in ["/api/play-ai", "/api/eval-step"]:
            allowed, remaining, retry_after = await global_rate_limiter.is_allowed(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please wait before requesting AI moves."},
                    headers={"Retry-After": str(int(retry_after) + 1), "X-RateLimit-Remaining": "0"}
                )

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            global_metrics.record_request(elapsed_ms, is_error=response.status_code >= 400)
            return response
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            global_metrics.record_request(elapsed_ms, is_error=True)
            raise e

    # Lazy load neural models on startup
    ai_models: Dict[str, Any] = {}
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
            global_metrics.record_mcts_sims(req.sims or 250)
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
                global_metrics.record_cache_miss()
                mcts_instance = MCTS(active_model, simulations=req.batch_size, depsilon=0)
                tree_data = {
                    "mcts": mcts_instance,
                    "total_sims": 0
                }
                await eval_cache.set(req.state_str, tree_data)
            else:
                global_metrics.record_cache_hit()

            global_metrics.record_mcts_sims(req.batch_size)

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

    # -----------------------------------------------------------------------
    # WebSocket Matchmaking & Relay Routes (P5.3)
    # -----------------------------------------------------------------------
    from api.lobby import global_lobby, Player, MatchRoom

    # -----------------------------------------------------------------------
    # Production Observability & Persistent ELO Leaderboard (P6.15)
    # -----------------------------------------------------------------------
    @app.get("/api/metrics")
    async def get_metrics():
        active_rooms = len(global_lobby.rooms)
        return global_metrics.get_summary(active_rooms=active_rooms)

    @app.get("/api/leaderboard")
    async def get_leaderboard(limit: int = 20):
        return {"leaderboard": global_leaderboard.get_leaderboard(limit=min(100, max(1, limit)))}

    @app.post("/api/player/record-match")
    async def record_player_match(req: MatchRecordRequest):
        res = global_leaderboard.record_match(
            winner_handle=req.winner_handle,
            loser_handle=req.loser_handle,
            is_draw=req.is_draw,
            winner_faction=req.winner_faction
        )
        return {"status": "ok", "result": res}

    @app.get("/api/lobby/rooms")
    async def list_lobby_rooms():
        return {"rooms": global_lobby.list_active_rooms()}

    @app.websocket("/ws/lobby")
    async def websocket_lobby_matchmaking(websocket: WebSocket):
        await websocket.accept()
        player = None
        try:
            init_data = await websocket.receive_text()
            payload = json.loads(init_data)
            handle = payload.get("handle", f"Player_{random.randint(100, 999)}")
            elo = payload.get("elo", 1200)
            player = Player(handle=handle, websocket=websocket, elo=elo)

            room_id = await global_lobby.join_queue(player)
            if not room_id:
                await websocket.send_text(json.dumps({"type": "QUEUE_WAITING", "status": "Searching for opponent..."}))

            # Keep connection alive while queued
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "LEAVE_QUEUE":
                    await global_lobby.leave_queue(player)
                    break
        except WebSocketDisconnect:
            if player:
                await global_lobby.leave_queue(player)
        except Exception:
            if player:
                await global_lobby.leave_queue(player)

    @app.websocket("/ws/room/{room_id}")
    async def websocket_room_relay(websocket: WebSocket, room_id: str):
        await websocket.accept()
        role = "spectator"
        handle = f"User_{random.randint(100, 999)}"
        target_room = global_lobby.get_room(room_id)
        if target_room is None:
            room: MatchRoom = global_lobby.create_room(room_id, host_handle=handle)
        else:
            room = target_room

        try:
            init_data = await websocket.receive_text()
            payload = json.loads(init_data)
            handle = payload.get("handle", handle)
            role = payload.get("role", "spectator")

            if role in ["host", "british", "mysore", "guest"]:
                room.sockets[handle] = websocket
            else:
                room.spectators.add(websocket)

            # Send current room state if exists
            if room.state_str:
                await websocket.send_text(json.dumps({
                    "type": "SYNC_STATE",
                    "state_str": room.state_str,
                    "moves": room.moves_history
                }))

            await room.broadcast({
                "type": "USER_JOINED",
                "handle": handle,
                "role": role,
                "spectator_count": len(room.spectators)
            })

            # Relay loop
            while True:
                text_data = await websocket.receive_text()
                msg = json.loads(text_data)
                msg_type = msg.get("type")

                if msg_type == "MOVE":
                    if "state_str" in msg:
                        room.state_str = msg["state_str"]
                    if "move_idx" in msg:
                        room.moves_history.append(msg["move_idx"])

                # Relay packet to peer / spectators
                await room.broadcast(msg, sender=handle)

        except WebSocketDisconnect:
            room.sockets.pop(handle, None)
            if isinstance(websocket, WebSocket):
                room.spectators.discard(websocket)
            await room.broadcast({
                "type": "USER_LEFT",
                "handle": handle,
                "spectator_count": len(room.spectators)
            })
        except Exception:
            room.sockets.pop(handle, None)
            if isinstance(websocket, WebSocket):
                room.spectators.discard(websocket)

    # Optional static assets mounting (for local server execution)
    if mount_static:
        public_dir = os.path.join(BASE_DIR, "public")
        if os.path.exists(public_dir):
            app.mount("/", StaticFiles(directory=public_dir, html=True), name="static")

    return app
