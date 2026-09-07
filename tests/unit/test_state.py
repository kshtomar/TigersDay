import unittest
import numpy as np
from game.state import GameState

class TestGameState(unittest.TestCase):
    def test_default_setup_invariants(self):
        state = GameState()
        state.default_setup()
        
        # Test bit 94 invariant (combat strength 0 is active, so bit 94 is 1)
        self.assertEqual(state.card_strength, 0)
        self.assertEqual(state.turn, 1)
        self.assertEqual(state.to_move, 0)
        self.assertEqual(state.vector[94], 1)
        
        # Length of string representation must be 148 bits
        state_str = state.to_str()
        self.assertEqual(len(state_str), 148)
        self.assertTrue(all(c in '01' for c in state_str))

    def test_serialization_roundtrip(self):
        state = GameState()
        state.default_setup()
        original_str = state.to_str()
        
        restored = GameState()
        restored.read_str(original_str)
        self.assertEqual(restored.to_str(), original_str)
        self.assertTrue(np.array_equal(state.vector, restored.vector))

    def test_read_str_validation_length(self):
        state = GameState()
        with self.assertRaises(ValueError):
            state.read_str("0" * 147)  # Too short
        with self.assertRaises(ValueError):
            state.read_str("0" * 149)  # Too long
        with self.assertRaises(ValueError):
            state.read_str("0" * 147 + "x")  # Invalid character

    def test_read_str_validation_territory_conflict(self):
        state = GameState()
        state.default_setup()
        vec = list(state.to_str())
        
        # Make node 0 both british_controlled (idx 0) and mysore_controlled (idx 25)
        vec[0] = '1'
        vec[25] = '1'
        conflict_str = "".join(vec)
        
        restored = GameState()
        with self.assertRaises(ValueError):
            restored.read_str(conflict_str)

    def test_repetition_detection(self):
        from game.updater import check_repetition
        state = GameState()
        state.default_setup()
        s_str = state.to_str()
        
        history = [s_str, "other_state"]
        self.assertFalse(check_repetition(history, state))
        
        history.append(s_str)
        self.assertTrue(check_repetition(history, state))

if __name__ == '__main__':
    unittest.main()
