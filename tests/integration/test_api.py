import unittest
import asyncio
import json
from api.app import create_app
from game.state import GameState

class TestApiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()

    def _run_request(self, method, path, body=None):
        async def _req():
            scope = {
                "type": "http",
                "method": method,
                "path": path,
                "raw_path": path.encode("ascii"),
                "query_string": b"",
                "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
                "server": ("testserver", 80),
                "client": ("127.0.0.1", 12345),
            }
            response_body = []
            status_code = 200

            async def receive():
                if body is not None:
                    encoded = json.dumps(body).encode("utf-8") if isinstance(body, (dict, list)) else body.encode("utf-8")
                    return {"type": "http.request", "body": encoded, "more_body": False}
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                elif message["type"] == "http.response.body":
                    response_body.append(message.get("body", b""))

            await self.app(scope, receive, send)
            raw = b"".join(response_body).decode("utf-8")
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
            return status_code, parsed

        return asyncio.run(_req())

    def test_api_init(self):
        status, data = self._run_request("GET", "/api/init")
        self.assertEqual(status, 200)
        self.assertIn("state_str", data)
        self.assertEqual(len(data["state_str"]), 148)
        self.assertIn("ui_state", data)
        self.assertIn("nodes", data["ui_state"])
        self.assertEqual(len(data["ui_state"]["nodes"]), 25)

    def test_api_load_state_valid(self):
        state = GameState()
        state.default_setup()
        status, data = self._run_request("POST", "/api/load-state", {"state_str": str(state)})
        self.assertEqual(status, 200)
        self.assertIn("state_str", data)

    def test_api_load_state_invalid_length(self):
        status, data = self._run_request("POST", "/api/load-state", {"state_str": "0" * 100})
        # Pydantic regex validation rejection (422) or Bad Request (400)
        self.assertIn(status, [400, 422])

    def test_api_play_move_valid(self):
        state = GameState()
        state.default_setup()
        # Move 77 is legal opening move
        status, data = self._run_request("POST", "/api/play-move", {
            "state_str": str(state),
            "move_idx": 77
        })
        self.assertEqual(status, 200)
        self.assertIn("state_str", data)

    def test_api_play_move_illegal(self):
        state = GameState()
        state.default_setup()
        # Move 958 (Pass British) is illegal when legal movement actions exist
        status, data = self._run_request("POST", "/api/play-move", {
            "state_str": str(state),
            "move_idx": 958
        })
        self.assertEqual(status, 400)

    def test_api_get_notation(self):
        status, data = self._run_request("POST", "/api/get-notation", {
            "replay_log": [77]
        })
        self.assertEqual(status, 200)
        self.assertIn("notation", data)
        self.assertTrue(len(data["notation"]) > 0)

    def test_api_play_ai(self):
        state = GameState()
        state.default_setup()
        status, data = self._run_request("POST", "/api/play-ai", {
            "state_str": str(state),
            "sims": 5
        })
        self.assertEqual(status, 200)
        self.assertIn("best_move", data)
        self.assertIsInstance(data["best_move"], int)
        self.assertIn("game_data", data)
        self.assertIn("notation", data)

    def test_api_eval_step(self):
        state = GameState()
        state.default_setup()
        status, data = self._run_request("POST", "/api/eval-step", {
            "state_str": str(state),
            "batch_size": 5
        })
        self.assertEqual(status, 200)
        self.assertIn("score", data)
        self.assertIn("total_sims", data)
        self.assertIn("top_moves", data)

if __name__ == '__main__':
    unittest.main()
