import unittest
import asyncio
import json
from api.lobby import Player, MatchRoom, MatchmakingLobby, global_lobby
from api.app import create_app

class MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False

    async def send_text(self, data: str):
        if self.closed:
            raise RuntimeError("WebSocket closed")
        self.sent_messages.append(data)

    async def close(self):
        self.closed = True

class TestLobbyAndWebSocketRelay(unittest.TestCase):
    def setUp(self):
        self.lobby = MatchmakingLobby()

    def test_queue_pairing_and_role_assignment(self):
        """Test queue pairing assigns host as british and guest as mysore."""
        async def _run():
            ws1 = MockWebSocket()
            p1 = Player(handle="Player_Alpha", websocket=ws1, elo=1200)

            # Player 1 joins queue -> waiting
            room_id_1 = await self.lobby.join_queue(p1)
            self.assertIsNone(room_id_1)
            self.assertEqual(len(self.lobby.queue), 1)

            # Player 2 joins queue -> paired
            ws2 = MockWebSocket()
            p2 = Player(handle="Player_Beta", websocket=ws2, elo=1210)
            room_id_2 = await self.lobby.join_queue(p2)

            self.assertIsNotNone(room_id_2)
            self.assertTrue(room_id_2.startswith("TIGER-"))
            self.assertEqual(len(self.lobby.queue), 0)

            # Assert notifications received
            self.assertEqual(len(ws1.sent_messages), 1)
            self.assertEqual(len(ws2.sent_messages), 1)

            msg_p1 = json.loads(ws1.sent_messages[0])
            msg_p2 = json.loads(ws2.sent_messages[0])

            self.assertEqual(msg_p1["type"], "MATCH_FOUND")
            self.assertEqual(msg_p1["role"], "british")
            self.assertEqual(msg_p1["opponent"], "Player_Beta")

            self.assertEqual(msg_p2["type"], "MATCH_FOUND")
            self.assertEqual(msg_p2["role"], "mysore")
            self.assertEqual(msg_p2["opponent"], "Player_Alpha")

        asyncio.run(_run())

    def test_elo_sorting_in_queue(self):
        """Test matchmaking pairs closest ELO first."""
        async def _run():
            ws_base = MockWebSocket()
            p_base = Player(handle="Base", websocket=ws_base, elo=1500)
            ws_far = MockWebSocket()
            p_far = Player(handle="Far", websocket=ws_far, elo=2000)

            # Populate queue directly with 2 players to test closest ELO matching
            self.lobby.queue = [p_base, p_far]

            # Queue has [1500, 2000]
            # New player with 1520 joins -> should match with 1500 (Base), leaving 2000 (Far)
            ws_close = MockWebSocket()
            p_close = Player(handle="Close", websocket=ws_close, elo=1520)
            room_id = await self.lobby.join_queue(p_close)

            self.assertIsNotNone(room_id)
            self.assertEqual(len(self.lobby.queue), 1)
            self.assertEqual(self.lobby.queue[0].handle, "Far")

            msg_close = json.loads(ws_close.sent_messages[0])
            self.assertEqual(msg_close["opponent"], "Base")

        asyncio.run(_run())

    def test_room_spectator_broadcast_and_relay(self):
        """Test MatchRoom relays packets to peers and spectators, skipping sender."""
        async def _run():
            room = self.lobby.create_room("TIGER-9999", host_handle="Host")
            ws_host = MockWebSocket()
            ws_guest = MockWebSocket()
            ws_spec = MockWebSocket()

            room.sockets["Host"] = ws_host
            room.sockets["Guest"] = ws_guest
            room.spectators.add(ws_spec)

            # Host broadcasts a MOVE packet
            move_pkt = {"type": "MOVE", "moveIdx": 77, "stateStr": "1" + "0"*147}
            await room.broadcast(move_pkt, sender="Host")

            # Host should NOT receive own broadcast
            self.assertEqual(len(ws_host.sent_messages), 0)
            # Guest and spectator SHOULD receive broadcast
            self.assertEqual(len(ws_guest.sent_messages), 1)
            self.assertEqual(len(ws_spec.sent_messages), 1)

            payload = json.loads(ws_guest.sent_messages[0])
            self.assertEqual(payload["type"], "MOVE")
            self.assertEqual(payload["moveIdx"], 77)

        asyncio.run(_run())

    def test_dead_socket_eviction(self):
        """Test dead sockets that throw exceptions are cleanly purged."""
        async def _run():
            room = self.lobby.create_room("TIGER-1111", host_handle="Host")
            ws_dead = MockWebSocket()
            await ws_dead.close() # Closed socket throws on send_text

            room.sockets["DeadPlayer"] = ws_dead
            room.spectators.add(ws_dead)

            self.assertIn("DeadPlayer", room.sockets)
            self.assertIn(ws_dead, room.spectators)

            # Broadcast triggers eviction
            await room.broadcast({"type": "PING"})

            self.assertNotIn("DeadPlayer", room.sockets)
            self.assertNotIn(ws_dead, room.spectators)

        asyncio.run(_run())

    def test_asgi_websocket_lobby_lifecycle(self):
        """Test full ASGI WebSocket /ws/lobby connection and queueing."""
        app = create_app()

        async def _run():
            incoming_queue = asyncio.Queue()
            outgoing_messages = []

            scope = {
                "type": "websocket",
                "path": "/ws/lobby",
                "raw_path": b"/ws/lobby",
                "headers": [],
                "query_string": b"",
                "subprotocols": []
            }

            async def receive():
                return await incoming_queue.get()

            async def send(message):
                outgoing_messages.append(message)

            # 1. Start connection task
            app_task = asyncio.create_task(app(scope, receive, send))

            # Send connect
            await incoming_queue.put({"type": "websocket.connect"})
            await asyncio.sleep(0.01)

            # Send init payload
            init_json = json.dumps({"handle": "TestPlayer", "elo": 1300})
            await incoming_queue.put({"type": "websocket.receive", "text": init_json})
            await asyncio.sleep(0.01)

            # Verify accept and QUEUE_WAITING received
            accept_msg = [m for m in outgoing_messages if m["type"] == "websocket.accept"]
            self.assertEqual(len(accept_msg), 1)

            text_messages = [json.loads(m["text"]) for m in outgoing_messages if m.get("text")]
            self.assertTrue(any(m.get("type") == "QUEUE_WAITING" for m in text_messages))

            # Disconnect
            await incoming_queue.put({"type": "websocket.disconnect", "code": 1000})
            await asyncio.sleep(0.01)
            app_task.cancel()
            try:
                await app_task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

if __name__ == "__main__":
    unittest.main()
