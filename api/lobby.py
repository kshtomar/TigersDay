"""
Centralized WebSocket Matchmaking Lobby & Relay Server for Tiger's Day.
Provides:
- Public matchmaking queue with ELO rating matching.
- Real-time room discovery and spectator streaming.
- Fallback message relay when direct WebRTC P2P fails due to symmetric NATs.
"""

import asyncio
import json
import random
import secrets
import time
from typing import Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect


class Player:
    def __init__(self, handle: str, websocket: WebSocket, elo: int = 1200) -> None:
        self.handle = handle
        self.websocket = websocket
        self.elo = elo
        self.current_room: Optional[str] = None


class MatchRoom:
    def __init__(self, room_id: str, host_handle: str) -> None:
        self.room_id = room_id
        self.host_handle = host_handle
        self.guest_handle: Optional[str] = None
        self.sockets: Dict[str, WebSocket] = {}
        self.spectators: Set[WebSocket] = set()
        self.state_str: Optional[str] = None
        self.moves_history: List[int] = []
        self.created_at = time.time()

    async def broadcast(self, message: dict, sender: Optional[str] = None):
        """Broadcast message to players and spectators."""
        data = json.dumps(message)
        dead_sockets = []
        
        # Send to players
        for handle, ws in self.sockets.items():
            if sender and handle == sender:
                continue
            try:
                await ws.send_text(data)
            except Exception:
                dead_sockets.append(handle)
                
        for handle in dead_sockets:
            self.sockets.pop(handle, None)
            
        # Send to spectators
        dead_specs = []
        for ws in self.spectators:
            try:
                await ws.send_text(data)
            except Exception:
                dead_specs.append(ws)
                
        for ws in dead_specs:
            self.spectators.discard(ws)


class MatchmakingLobby:
    def __init__(self) -> None:
        self.queue: List[Player] = []
        self.rooms: Dict[str, MatchRoom] = {}
        self._lock = asyncio.Lock()

    async def join_queue(self, player: Player) -> Optional[str]:
        """Adds player to matchmaking queue; pairs if another player is waiting."""
        async with self._lock:
            # Check for best ELO match within queue
            if len(self.queue) > 0:
                # Find closest ELO
                self.queue.sort(key=lambda p: abs(p.elo - player.elo))
                opponent = self.queue.pop(0)
                room_id = f"TIGER-{secrets.token_hex(4).upper()}"
                room = MatchRoom(room_id, host_handle=opponent.handle)
                room.guest_handle = player.handle
                self.rooms[room_id] = room
                
                # Notify both players
                match_data_host = {"type": "MATCH_FOUND", "room_id": room_id, "role": "british", "opponent": player.handle}
                match_data_guest = {"type": "MATCH_FOUND", "room_id": room_id, "role": "mysore", "opponent": opponent.handle}
                
                try:
                    await opponent.websocket.send_text(json.dumps(match_data_host))
                except Exception:
                    pass
                try:
                    await player.websocket.send_text(json.dumps(match_data_guest))
                except Exception:
                    pass
                return room_id
            else:
                self.queue.append(player)
                return None

    async def leave_queue(self, player: Player):
        async with self._lock:
            self.queue = [p for p in self.queue if p.websocket != player.websocket]

    def create_room(self, room_id: str, host_handle: str) -> MatchRoom:
        room = MatchRoom(room_id, host_handle)
        self.rooms[room_id] = room
        return room

    def get_room(self, room_id: str) -> Optional[MatchRoom]:
        return self.rooms.get(room_id)

    def list_active_rooms(self) -> List[dict]:
        return [
            {
                "room_id": r.room_id,
                "host": r.host_handle,
                "guest": r.guest_handle,
                "moves_count": len(r.moves_history),
                "spectators": len(r.spectators),
                "created_at": r.created_at
            }
            for r in self.rooms.values()
        ]


# Singleton instance
global_lobby = MatchmakingLobby()
