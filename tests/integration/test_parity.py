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

if __name__ == '__main__':
    unittest.main()
