import unittest
import os
from game.state import GameState
from ai.opening_book import OpeningBook

class TestMCTSAndOpeningBook(unittest.TestCase):
    def test_opening_book_loading_and_lookup(self):
        book = OpeningBook()
        self.assertGreater(len(book.book), 0)
        
        # Test lookup with default setup state
        state = GameState()
        state.default_setup()
        
        # Check if default setup has an opening book move
        book_move = book.get_move(state)
        # Note: Depending on whether default state is in book, get_move returns int or None
        if book_move is not None:
            self.assertIsInstance(book_move, int)
            self.assertGreaterEqual(book_move, 0)
            self.assertLess(book_move, 959)

    def test_transposition_table_caching(self):
        """Assert that MCTS caches evaluations in transposition_table by Zobrist hash."""
        from ai.mcts import MCTS
        from ai.neural import DummyAlphaTiger

        model = DummyAlphaTiger()
        mcts = MCTS(model, simulations=10, depsilon=0)
        state = GameState()
        state.default_setup()

        self.assertEqual(len(mcts.transposition_table), 0)
        mcts.search(state, simulations=10)
        self.assertGreater(len(mcts.transposition_table), 0, "Transposition table must store evaluated states")

        # Root state's Zobrist hash must be in the table
        root_hash = state.zobrist_hash()
        self.assertIn(root_hash, mcts.transposition_table)

    def test_batched_search_with_virtual_loss(self):
        """Assert search_batch executes simulations in parallel chunks with virtual loss."""
        from ai.mcts import MCTS
        from ai.neural import DummyAlphaTiger

        model = DummyAlphaTiger()
        mcts = MCTS(model, simulations=24, depsilon=0)
        state = GameState()
        state.default_setup()

        root = mcts.search_batch(state, simulations=24, batch_size=8, virtual_loss=3.0)
        self.assertGreaterEqual(root.visit_count, 24)
        self.assertGreater(len(root.children), 0)

    def test_time_budgeted_mcts_search(self):
        """Assert search_time_budget respects duration constraints."""
        import time
        from ai.mcts import MCTS
        from ai.neural import DummyAlphaTiger

        model = DummyAlphaTiger()
        mcts = MCTS(model, simulations=50, depsilon=0)
        state = GameState()
        state.default_setup()

        t0 = time.perf_counter()
        move, policy = mcts.find_move(state, time_budget_ms=200, use_book=False)
        duration_ms = (time.perf_counter() - t0) * 1000.0

        self.assertIsInstance(move, int)
        self.assertGreaterEqual(move, 0)
        self.assertLess(move, 959)
        self.assertLess(duration_ms, 1500, "Time-budgeted search should terminate within a reasonable window")

if __name__ == '__main__':
    unittest.main()

