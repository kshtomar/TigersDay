import unittest
import numpy as np
from game.state import GameState
from game.updater import ACTION_DISPATCH, MOVE_SPACE, get_next_state, resolve_battles

class TestGameUpdater(unittest.TestCase):
    def test_action_dispatch_table(self):
        self.assertEqual(len(ACTION_DISPATCH), 959)
        
        # Verify dispatch matches MOVE_SPACE structure
        offset = 0
        for name, size, move_type in MOVE_SPACE:
            for idx in range(size):
                item = ACTION_DISPATCH[offset + idx]
                self.assertEqual(item, (name, move_type, idx))
            offset += size
        self.assertEqual(offset, 959)

    def test_battle2_net_card_strength(self):
        state = GameState()
        state.default_setup()
        
        # Set up a battle condition
        state.attacker = 1
        state.defender = 2
        state.card_strength = 2  # Attacker has card advantage
        
        # In resolve_battles, battle2 calculation should pass net_card_strength
        # We test that resolve_battles completes without error and applies battle outcome
        from game.constants import NO_UNIT
        res = resolve_battles(state, attacker=1, defender=2, net_card_strength=2)
        self.assertEqual(res.attacker, NO_UNIT)
        self.assertEqual(res.defender, NO_UNIT)

    def test_get_next_state_dispatch(self):
        state = GameState()
        state.default_setup()
        
        # Move 77 is an opening move (M on edge 77)
        next_state = get_next_state(state, 77)
        self.assertIsNotNone(next_state)
        self.assertNotEqual(state.to_str(), next_state.to_str())

if __name__ == '__main__':
    unittest.main()
