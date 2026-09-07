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

if __name__ == '__main__':
    unittest.main()
