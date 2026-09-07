import unittest
import os
import json
import tempfile
from game.state import GameState
from ai.evolve_book import run_tournament_game, evolve_opening_book, choose_heuristic_move


class TestEvolveBook(unittest.TestCase):
    def test_choose_heuristic_move(self):
        state = GameState()
        state.default_setup()
        move = choose_heuristic_move(state)
        self.assertIsInstance(move, int)
        self.assertGreaterEqual(move, 0)
        self.assertLess(move, 960)

    def test_run_tournament_game(self):
        history, winner = run_tournament_game(max_plies=6)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        self.assertIn(winner, [0.0, 0.5, 1.0])

        s_key, move, notation, player = history[0]
        self.assertEqual(len(s_key), 148)
        self.assertIsInstance(move, int)
        self.assertIsInstance(notation, str)
        self.assertIn(player, [0, 1, 2])

    def test_evolve_opening_book_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_book = os.path.join(tmpdir, "test_book.json")

            # Seed with one existing entry
            initial_data = {
                "0" * 148: {
                    "move": 10,
                    "notation": "mad>pdc",
                    "total": 10,
                    "win_rate": 0.20  # Under prune threshold
                }
            }
            with open(temp_book, "w", encoding="utf-8") as f:
                json.dump(initial_data, f)

            summary = evolve_opening_book(
                num_games=6,
                max_book_ply=4,
                min_winrate=0.50,
                min_samples=1,
                prune_threshold=0.25,
                book_path=temp_book
            )

            self.assertIn("games_played", summary)
            self.assertEqual(summary["games_played"], 6)
            self.assertIn("added", summary)
            self.assertIn("pruned", summary)
            self.assertGreaterEqual(summary["pruned"], 1)  # The 0.20 entry should be pruned

            # Verify file exists and is valid JSON
            self.assertTrue(os.path.exists(temp_book))
            with open(temp_book, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertIsInstance(loaded, dict)
            self.assertNotIn("0" * 148, loaded)


if __name__ == "__main__":
    unittest.main()
