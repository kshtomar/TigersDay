import unittest
from game.state import GameState
from game.constants import (
    MOVE_SPACE, MOVE_VECTOR_LENGTH, NODES_ABBREV,
    COASTAL_INDICES, BRITISH_CARDS, MYSORE_CARDS
)
from game.replay import notate, interpret, build_move_tree

class TestReplayNotationAndInterpreter(unittest.TestCase):
    def setUp(self):
        self.state = GameState()
        self.state.default_setup()

    def test_move_notation_standard_troop_movement(self):
        """Test regular movement without fort siege (mad>pdc) and with fort siege (x)."""
        # Move from Madras (2) to Pondicherry (16)
        # Find index in MOVE_SPACE edge array
        from game.constants import EDGE_SOURCES, EDGE_DESTS
        edge_idx = None
        for i in range(len(EDGE_SOURCES)):
            if EDGE_SOURCES[i] == 2 and EDGE_DESTS[i] == 16:
                edge_idx = i
                break
        self.assertIsNotNone(edge_idx)

        # In default state, Pondicherry has no fort
        self.state.forts[16] = 0
        notation = notate(self.state, edge_idx)
        self.assertEqual(notation, "mad>pdc")

        # Now put a fort on Pondicherry -> should become madxpdc
        self.state.forts[16] = 1
        notation_siege = notate(self.state, edge_idx)
        self.assertEqual(notation_siege, "madxpdc")

    def test_move_notation_tactical_cards(self):
        """Test notation for Sepoy Mutiny, French Alliance, Monsoon, and Cavalry Raid."""
        # 1. Sepoy Mutiny on Travancore (22)
        sm_start = 111
        notation_sm = notate(self.state, sm_start + 22)
        self.assertEqual(notation_sm, "SM:trv")

        # 2. French Alliance on Darwar (9)
        fa_start = 136
        notation_fa = notate(self.state, fa_start + 9)
        self.assertEqual(notation_fa, "FA:dwr")

        # 3. Monsoon on Hyderabad (1)
        ms_start = 161
        notation_ms = notate(self.state, ms_start + 1)
        self.assertEqual(notation_ms, "MS:hyd")

        # 4. Cavalry Raid (186)
        cr_idx = 186
        notation_cr = notate(self.state, cr_idx)
        self.assertEqual(notation_cr, "CR")

    def test_move_notation_coastal_operations(self):
        """Test Royal Navy and Sea Trade coastal notation."""
        # Royal Navy: Bombay (0) to Goa (8)
        # Coastal indices: find index of Goa (8) in COASTAL_INDICES
        goa_coast_idx = list(COASTAL_INDICES).index(8)
        rn_start = 487
        rn_move = rn_start + 0 * len(COASTAL_INDICES) + goa_coast_idx
        # Default state Goa has no fort
        self.state.forts[8] = 0
        self.assertEqual(notate(self.state, rn_move), "RN:bom>goa")

        # Sea Trade: Mangalore (12) to Satara (5)
        mlr_coast_idx = list(COASTAL_INDICES).index(12)
        st_start = 187
        st_move = st_start + 5 * len(COASTAL_INDICES) + mlr_coast_idx
        self.assertEqual(notate(self.state, st_move), "ST:mlr>sat")

    def test_move_notation_combat_strength_and_trading(self):
        """Test Wall Breach / Iron Rockets power and card trading notation."""
        # 1. Mysore Power (Iron Rockets = 0) -> IR:x
        mp_start = 437
        self.assertEqual(notate(self.state, mp_start + 0), "IR:x")

        # 2. British Power (Wall Breach = 0) -> WB:x
        bp_start = 934
        self.assertEqual(notate(self.state, bp_start + 0), "WB:x")

        # 3. Card trading: Draw Iron Rockets (443) trading for Sepoy Mutiny (idx 1) -> IR:SM
        ir_trade = 443 + 1
        self.assertEqual(notate(self.state, ir_trade), "IR:SM")

        # 4. Card trading: Draw Wall Breach (940) trading for Highlanders (idx 1) -> WB:HL
        wb_trade = 940 + 1
        self.assertEqual(notate(self.state, wb_trade), "WB:HL")

        # 5. Passes
        self.assertEqual(notate(self.state, 461), "pass")
        self.assertEqual(notate(self.state, 958), "pass")

    def test_interpret_game_log(self):
        """Test interpret() with synthetic short game log."""
        # A sequence crossing from Turn 1 to Turn 2:
        replay_sample = [77, 162, 485, 10, 186, 4, 941, 80, 144, 485, 1, 134, 776, 56, 130, 938, 2, 65, 437, 483, 2, 74, 147, 941, 0, 87, 461, 955, 0]
        algebraic, state_history = interpret(replay_sample)

        self.assertIsInstance(algebraic, str)
        self.assertIn("+", algebraic, "Turn transition should insert '+' symbol")
        self.assertTrue(algebraic.endswith("1-0") or algebraic.endswith("0-1"), "Termination must end with score")
        self.assertEqual(len(state_history), len(replay_sample) + 1)

    def test_build_move_tree_aggregation(self):
        """Test build_move_tree tracks game counts, depths, and win outcomes."""
        synthetic_games = [
            ["mad>pdc", "SM:trv", "RN:bom>goa", "+", "#", "1-0"],
            ["mad>pdc", "SM:trv", "WB:HL", "+", "#", "1-0"],
            ["mad>pdc", "FA:dwr", "pass", "+", "#", "0-1"],
            ["bom>sat", "CR", "pass", "+", "#", "0-1"]
        ]

        tree = build_move_tree(synthetic_games, max_depth=3)

        self.assertIn("mad>pdc", tree)
        self.assertEqual(tree["mad>pdc"]["count"], 3)
        self.assertEqual(tree["mad>pdc"]["british_wins"], 2)

        # Child branch verification
        child_branches = tree["mad>pdc"]["next"]
        self.assertIn("SM:trv", child_branches)
        self.assertEqual(child_branches["SM:trv"]["count"], 2)
        self.assertEqual(child_branches["SM:trv"]["british_wins"], 2)

        self.assertIn("bom>sat", tree)
        self.assertEqual(tree["bom>sat"]["count"], 1)
        self.assertEqual(tree["bom>sat"]["british_wins"], 0)

    def test_tdr_export_and_import_roundtrip(self):
        """Verify .tdr serialization, format checking, and validation."""
        from game.replay import export_tdr, import_tdr

        moves = [77, 461, 958]
        tdr_str = export_tdr(
            moves=moves,
            player_british="Wellesley",
            player_mysore="Tipu",
            winner="0-1",
            metadata={"difficulty": "veteran"}
        )

        self.assertIsInstance(tdr_str, str)
        self.assertIn("TigerDayReplay", tdr_str)

        parsed = import_tdr(tdr_str)
        self.assertEqual(parsed["format"], "TigerDayReplay")
        self.assertEqual(parsed["moves"], moves)
        self.assertEqual(parsed["players"]["british"], "Wellesley")
        self.assertEqual(parsed["winner"], "0-1")

        # Invalid format throws ValueError
        with self.assertRaises(ValueError):
            import_tdr('{"format": "UnknownGame"}')

        # Invalid move index throws ValueError
        with self.assertRaises(ValueError):
            import_tdr('{"format": "TigerDayReplay", "moves": [-5]}')

if __name__ == "__main__":
    unittest.main()

