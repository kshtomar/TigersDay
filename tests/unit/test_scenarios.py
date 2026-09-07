import unittest
from game.scenarios import get_scenario, list_scenarios, SCENARIOS_METADATA
from game.engine import get_legal_moves

class TestScenarios(unittest.TestCase):
    def test_list_scenarios(self):
        scenarios = list_scenarios()
        self.assertEqual(len(scenarios), 4)
        ids = [s["id"] for s in scenarios]
        self.assertIn("first_anglo_mysore_war", ids)
        self.assertIn("second_anglo_mysore_war", ids)
        self.assertIn("third_anglo_mysore_war", ids)
        self.assertIn("fourth_anglo_mysore_war", ids)

    def test_scenario_initializations(self):
        for s_meta in SCENARIOS_METADATA:
            state = get_scenario(s_meta["id"])
            self.assertEqual(len(state.to_str()), 148)
            self.assertEqual(state.turn, 1)
            self.assertEqual(state.to_move, 0)
            
            # Must have legal moves available
            legal_mask = get_legal_moves(state)
            self.assertGreater(legal_mask.sum(), 0, f"Scenario {s_meta['id']} should have legal moves")

if __name__ == '__main__':
    unittest.main()
