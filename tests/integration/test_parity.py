import unittest
import subprocess
import json
from game.state import GameState
from game.updater import get_next_state

class TestCrossEngineParity(unittest.TestCase):
    def test_default_state_string_parity(self):
        # 1. Python default state string
        py_state = GameState()
        py_state.default_setup()
        py_str = py_state.to_str()

        # 2. Node.js default state string
        js_code = """
        const { GameState } = require('./public/js/state.js');
        const s = new GameState();
        s.default_setup();
        process.stdout.write(s.toString());
        """
        js_str = subprocess.check_output(["node", "-e", js_code]).decode('utf-8').strip()

        self.assertEqual(py_str, js_str, "Default setup state strings must match byte-for-byte")

    def test_move_77_transition_parity(self):
        # 1. Python transition on move 77
        py_state = GameState()
        py_state.default_setup()
        py_next = get_next_state(py_state, 77)
        py_next_str = py_next.to_str()

        # 2. JS transition on move 77
        js_code = """
        const { GameState } = require('./public/js/state.js');
        const { TDEngine } = require('./public/js/engine.js');
        const s = new GameState();
        s.default_setup();
        const nextState = TDEngine.getNextState(s, 77);
        process.stdout.write(nextState.toString());
        """
        js_next_str = subprocess.check_output(["node", "-e", js_code]).decode('utf-8').strip()

        self.assertEqual(py_next_str, js_next_str, "Next state for move 77 must match byte-for-byte across engines")

    def test_50_ply_randomized_parity_battery(self):
        """Execute a continuous 50-ply legal simulation and assert byte-for-byte parity after every ply."""
        import random
        import numpy as np
        from game.engine import get_legal_moves
        from game.updater import get_luck_outcomes

        rng = random.Random(1799)
        py_state = GameState()
        py_state.default_setup()

        plies = []
        py_strings = []

        for _ in range(50):
            if py_state.is_luck:
                outcomes = get_luck_outcomes(py_state)
                if not outcomes:
                    break
                idx = rng.randrange(len(outcomes))
                plies.append(["luck", idx])
                py_state = outcomes[idx]
            else:
                legal = np.where(get_legal_moves(py_state))[0]
                if len(legal) == 0:
                    break
                move_idx = int(rng.choice(legal))
                plies.append(["move", move_idx])
                py_state = get_next_state(py_state, move_idx)
            py_strings.append(py_state.to_str())

        js_script = """
        const { GameState } = require('./public/js/state.js');
        const { TDEngine } = require('./public/js/engine.js');
        let s = new GameState();
        s.default_setup();
        const plies = JSON.parse(process.argv[1]);
        const results = [];
        for (const [kind, val] of plies) {
            if (kind === 'luck') {
                const outcomes = TDEngine.getLuckOutcomes(s);
                s = outcomes[val];
            } else {
                s = TDEngine.getNextState(s, val);
            }
            results.push(s.toString());
        }
        process.stdout.write(JSON.stringify(results));
        """

        output = subprocess.check_output(
            ["node", "-e", js_script, json.dumps(plies)]
        ).decode('utf-8').strip()
        js_strings = json.loads(output)

        self.assertEqual(len(py_strings), len(js_strings))
        self.assertGreaterEqual(len(py_strings), 50, "Expected at least 50 continuous simulated plies")
        for i, (py_str, js_str) in enumerate(zip(py_strings, js_strings)):
            self.assertEqual(
                py_str, js_str,
                f"Cross-engine state mismatch at ply {i+1} (Kind: {plies[i][0]}, Val: {plies[i][1]})"
            )

if __name__ == '__main__':
    unittest.main()

