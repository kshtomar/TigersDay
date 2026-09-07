import unittest
import numpy as np
import os
from ai.replay_buffer import ExperienceReplayBuffer

class TestExperienceReplayBuffer(unittest.TestCase):
    def test_buffer_push_and_sample(self):
        buf = ExperienceReplayBuffer(capacity=10)
        self.assertEqual(len(buf), 0)

        # Push 5 samples
        for i in range(5):
            s = np.zeros(148, dtype=np.float32)
            s[i] = 1.0
            p = np.zeros(959, dtype=np.float32)
            p[i] = 1.0
            buf.push(s, p, float(i))

        self.assertEqual(len(buf), 5)

        # Sample batch of 3
        s_b, p_b, v_b = buf.sample(3)
        self.assertEqual(s_b.shape, (3, 148))
        self.assertEqual(p_b.shape, (3, 959))
        self.assertEqual(v_b.shape, (3, 1))

    def test_buffer_circular_overwrite(self):
        buf = ExperienceReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(np.zeros(148), np.zeros(959), float(i))
        self.assertEqual(len(buf), 3)

    def test_buffer_serialization(self):
        buf = ExperienceReplayBuffer(capacity=5)
        for i in range(3):
            buf.push(np.ones(148) * i, np.ones(959) * i, float(i))

        tmp_path = "tests/test_buf.npz"
        try:
            buf.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            loaded_buf = ExperienceReplayBuffer(capacity=5)
            loaded_buf.load(tmp_path)
            self.assertEqual(len(loaded_buf), 3)
            self.assertEqual(loaded_buf.values[0], 0.0)
            self.assertEqual(loaded_buf.values[2], 2.0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
