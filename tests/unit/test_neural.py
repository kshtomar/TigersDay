import unittest
import numpy as np
import torch

from game.constants import GAME_VECTOR_LENGTH, MOVE_VECTOR_LENGTH, NODES, COASTAL_INDICES, MOVE_SPACE
from game.state import GameState
from ai.neural import AlphaTiger, DummyAlphaTiger, load_ai_model, ONNX_AVAILABLE

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.device = torch.device("cpu")
        self.model = AlphaTiger(use_factorization=True).to(self.device)
        self.model.eval()

    def test_tensor_dimensions_and_value_bounds(self):
        """Assert AlphaTiger forward pass produces (B, 1) in [-1, 1] and (B, 959)."""
        x = torch.randn(self.batch_size, GAME_VECTOR_LENGTH, dtype=torch.float32)
        with torch.no_grad():
            value, policy_logits = self.model(x)

        # 1. Shape assertions
        self.assertEqual(value.shape, (self.batch_size, 1), "Value tensor must have shape (B, 1)")
        self.assertEqual(policy_logits.shape, (self.batch_size, MOVE_VECTOR_LENGTH), "Policy logits must have shape (B, 959)")

        # 2. Value range assertion [-1.0, 1.0]
        self.assertTrue(torch.all(value >= -1.0) and torch.all(value <= 1.0), "Value must be strictly bounded in [-1.0, 1.0]")

    def test_factorized_decomposition_logic(self):
        """Verify additive decomposition for Royal Navy and Sea Trade."""
        # Find RN and ST offsets in MOVE_SPACE
        rn_offset = 0
        st_offset = 0
        offset = 0
        for name, size, _ in MOVE_SPACE:
            if name == "Royal Navy":
                rn_offset = offset
            elif name == "Sea Trade":
                st_offset = offset
            offset += size

        self.assertEqual(rn_offset, self.model.rn_start)
        self.assertEqual(st_offset, self.model.st_start)

        # Mock known raw_logits to verify additive factorized slice
        raw_logits = torch.randn(2, self.model.factorized_size)
        
        # Extract manual factor slices
        idx = self.model.base_size
        rn_src = raw_logits[:, idx : idx + NODES]; idx += NODES
        rn_dest = raw_logits[:, idx : idx + NODES]; idx += NODES
        st_src = raw_logits[:, idx : idx + NODES]; idx += NODES
        st_dest = raw_logits[:, idx : idx + NODES]

        # Reconstruct policy slice using the model's buffers
        rn_manual = rn_src[:, self.model.rn_src_idx] + rn_dest[:, self.model.rn_dest_idx]
        st_manual = st_src[:, self.model.st_src_idx] + st_dest[:, self.model.st_dest_idx]

        # Check indexing for RN (first node=0 to first coast=COASTAL_INDICES[0])
        expected_first_rn = (rn_src[:, 0] + rn_dest[:, COASTAL_INDICES[0]]).numpy()
        np.testing.assert_allclose(rn_manual[:, 0].numpy(), expected_first_rn, rtol=1e-5)

    def test_model_predict_interface(self):
        """Verify model.predict(state) returns float and 959-dim numpy array."""
        state = GameState()
        state.default_setup()

        val, policy = self.model.predict(state)
        self.assertIsInstance(val, float)
        self.assertTrue(-1.0 <= val <= 1.0)
        self.assertIsInstance(policy, np.ndarray)
        self.assertEqual(policy.shape, (MOVE_VECTOR_LENGTH,))

    def test_dummy_model_fallback(self):
        """Verify DummyAlphaTiger returns 0.0 and zero logits without crashing."""
        dummy = DummyAlphaTiger()
        state = GameState()
        state.default_setup()

        val, policy = dummy.predict(state)
        self.assertEqual(val, 0.0)
        self.assertEqual(policy.shape, (MOVE_VECTOR_LENGTH,))
        self.assertTrue(np.all(policy == 0.0))

    def test_load_ai_model_graceful_fallback(self):
        """Verify load_ai_model returns DummyAlphaTiger if non-existent path is requested."""
        model = load_ai_model("non_existent_file_path_12345.onnx")
        self.assertIsNotNone(model)
        # Should have a predict method
        self.assertTrue(hasattr(model, "predict"))

if __name__ == "__main__":
    unittest.main()
