import unittest
import numpy as np
from game.state import GameState
from game.engine import get_legal_moves, legal_moves_dict
from game.updater import get_next_state, get_luck_outcomes, get_state_winner
from game.replay import notate

class TestGameplayIntegration(unittest.TestCase):
    def test_multistep_gameplay_flow(self):
        state = GameState()
        state.default_setup()

        # Step 1: Check initial legal moves
        legal_mask = get_legal_moves(state)
        self.assertGreater(np.sum(legal_mask), 0)

        legal_dict = legal_moves_dict(legal_mask)
        self.assertEqual(len(legal_dict), np.sum(legal_mask))

        # Step 2: Play the first legal move
        move_idx = int(np.where(legal_mask)[0][0])
        move_not = notate(state, move_idx)
        self.assertTrue(len(move_not) > 0)

        next_state = get_next_state(state, move_idx)
        self.assertNotEqual(str(state), str(next_state))

        # Step 3: Handle luck resolution if necessary
        steps = 0
        current = next_state
        while current.is_luck and steps < 10:
            outcomes = get_luck_outcomes(current)
            self.assertGreater(len(outcomes), 0)
            current = outcomes[0]
            steps += 1

        self.assertFalse(current.is_luck)
        self.assertIn(get_state_winner(current), [0, 1, -1])

    def test_multistep_consecutive_turns(self):
        """Simulates 5 consecutive turns to ensure state machine transitions smoothly."""
        current = GameState()
        current.default_setup()

        for step in range(5):
            self.assertEqual(len(str(current)), 148)
            winner = get_state_winner(current)
            if winner != 0:
                break

            legal_mask = get_legal_moves(current)
            self.assertGreater(np.sum(legal_mask), 0)
            move = int(np.where(legal_mask)[0][0])
            self.assertGreaterEqual(move, 0)
            self.assertLess(move, 959)

            next_state = get_next_state(current, move)
            while next_state.is_luck:
                outcomes = get_luck_outcomes(next_state)
                self.assertGreater(len(outcomes), 0)
                next_state = outcomes[0]

            current = next_state

        self.assertEqual(len(str(current)), 148)

    def test_undo_state_restoration(self):
        """Verifies state can be rolled back cleanly via bit-string."""
        initial = GameState()
        initial.default_setup()
        saved_str = str(initial)

        legal_mask = get_legal_moves(initial)
        move = int(np.where(legal_mask)[0][0])
        next_state = get_next_state(initial, move)

        self.assertNotEqual(str(next_state), saved_str)

        # Rollback
        restored = GameState()
        restored.read_str(saved_str)
        self.assertEqual(str(restored), saved_str)
        self.assertTrue(np.array_equal(restored.vector, initial.vector))

if __name__ == '__main__':
    unittest.main()
