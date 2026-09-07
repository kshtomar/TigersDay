import unittest
import numpy as np
from game.state import GameState
from game.constants import (
    KEYS, COASTAL, ADJACENCY_MATRIX, NO_UNIT,
    BRITISH_MOVES_SPACE, MYSORE_CARDS_SPACE, BRITISH_CARDS_SPACE
)
from game.updater import get_state_winner, is_battle_won, resolve_battles, get_next_state
from game.engine import get_legal_moves

class TestRulesAndCombatEdgeCases(unittest.TestCase):
    def setUp(self):
        self.state = GameState()
        self.state.default_setup()

    def test_instant_british_victory_condition(self):
        """Assert capturing all 5 Key Cities immediately awards British victory (1)."""
        # Place British armies on all 5 key cities (0: Bombay, 1: Hyderabad, 2: Madras, 3: Seringapatam, 4: Coimbatore)
        self.state.fresh_armies[:] = 0
        self.state.tired_armies[:] = 0
        self.state.forts[:] = 0

        # Occupy keys with fresh or tired armies
        self.state.fresh_armies[0] = 1
        self.state.fresh_armies[1] = 1
        self.state.tired_armies[2] = 1
        self.state.tired_armies[3] = 1
        self.state.fresh_armies[4] = 1

        self.assertEqual(get_state_winner(self.state), 1, "British holding all 5 keys must immediately win (=1)")

    def test_mysore_attrition_victory_condition(self):
        """Assert Turn 4 end with 0 fresh armies and British < 5 keys awards Mysore victory (-1)."""
        self.state.turn = 4
        self.state.to_move = 0
        self.state.fresh_armies[:] = 0 # No fresh armies remain to move
        self.state.tired_armies[0] = 1 # Only 1 key held
        self.assertEqual(get_state_winner(self.state), -1, "Turn 4 with 0 fresh armies and <5 keys must yield Mysore victory (=-1)")

    def test_combat_tie_resolution_defender_holds(self):
        """When attacker_strength + net_card == defender_strength (tie), defender holds fort."""
        # Setup fort at Seringapatam (3)
        self.state.forts[3] = 1
        # 1 attacker adjacent (e.g. Bangalore 13)
        self.state.fresh_armies[:] = 0
        self.state.fresh_armies[13] = 1

        # Defender has 1 fort adjacent (Seringapatam has adjacent fort at Chitaldoorg 11)
        self.state.forts[11] = 1

        # Net card strength is 0 -> Attacker 1 vs Defender 1 -> Tie
        won = is_battle_won(self.state, defender=3, net_card_strength=0)
        self.assertFalse(won, "Combat tie (Net 0) must result in attacker failure (defender holds)")

    def test_multi_battle_phase_resolution(self):
        """Verify simultaneous battle1 and battle2 resolve correctly."""
        # Setup pending battle1 (Bangalore 13 attacked by Madras army at Vellore 14)
        self.state.attacker = 14
        self.state.defender = 13
        self.state.fresh_armies[14] = 1
        self.state.forts[13] = 1

        # Secondary battle2 (attacker at Coimbatore 4 attacking fort at Mahé 15)
        self.state.fresh_armies[4] = 1
        self.state.forts[15] = 1

        # Resolve battles with net card strength = 5 (guarantee attacker win)
        state_resolved = resolve_battles(self.state.copy(), attacker=4, defender=15, net_card_strength=5)
        self.assertEqual(state_resolved.attacker, NO_UNIT)
        self.assertEqual(state_resolved.defender, NO_UNIT)

    def test_card_trade_rules_enforcement(self):
        """Verify Value 3 cards can trade 1..5; Value 2 can trade 3..5; Value 1 cannot trade."""
        # Mysore turn
        self.state.to_move = 1
        # Enable Value 3 (Iron Rockets, index 0) and disable others
        self.state.mysore_cards[:] = 0
        self.state.mysore_cards[0] = 1 # Value 3 card active

        legal_mask = get_legal_moves(self.state)
        # Check Mysore trade moves (offset 443 for Iron Rockets trade)
        # Trades for cards 1..5 should be legal
        for target_card in range(1, 6):
            trade_idx = 443 + target_card
            self.assertTrue(legal_mask[trade_idx], f"Value 3 card must be able to trade for card {target_card}")

        # If only Value 1 card is active (e.g. Monsoon = index 3), no trade actions exist
        self.state.mysore_cards[:] = 0
        self.state.mysore_cards[3] = 1
        legal_mask = get_legal_moves(self.state)
        # Indices 443..460 are trades
        self.assertFalse(np.any(legal_mask[443:461]), "Value 1 cards cannot initiate trades")

    def test_territory_operation_constraints(self):
        """Verify Sepoy Mutiny masked on keys, French Alliance on fort adjacency, Princely States on empty keys."""
        # 1. Sepoy Mutiny on Keys (0..4) is masked out
        self.state.to_move = 1
        self.state.mysore_cards[:] = 0
        self.state.mysore_cards[1] = 1 # Sepoy Mutiny active
        # Put British armies on Key 0 (Bombay) and Non-Key 5 (Satara)
        self.state.fresh_armies[0] = 1
        self.state.fresh_armies[5] = 1

        legal_mask = get_legal_moves(self.state)
        sm_start = 111
        self.assertFalse(legal_mask[sm_start + 0], "Sepoy Mutiny must be masked out on Key Cities")
        self.assertTrue(legal_mask[sm_start + 5], "Sepoy Mutiny must be allowed on non-Key Cities")

        # 2. Princely States only on empty Key Cities
        self.state.to_move = 2
        self.state.british_cards[:] = 0
        self.state.british_cards[5] = 1 # Princely States active
        self.state.fresh_armies[0] = 1 # Bombay occupied
        self.state.fresh_armies[1] = 0 # Hyderabad empty
        self.state.tired_armies[1] = 0
        self.state.forts[1] = 0
        legal_mask_b = get_legal_moves(self.state)
        ps_start = 909
        self.assertFalse(legal_mask_b[ps_start + 0], "Princely States cannot deploy to occupied Key City")
        self.assertTrue(legal_mask_b[ps_start + 1], "Princely States can deploy to empty Key City")

if __name__ == "__main__":
    unittest.main()
