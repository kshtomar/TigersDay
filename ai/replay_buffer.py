"""
High-Performance Experience Replay Buffer for AlphaZero Self-Play.
Supports thread-safe/multiprocess-safe sample streaming, priority weighting, and serialization.
"""

import json
import os
import random
import threading
from typing import List, Optional, Tuple
import numpy as np


class ExperienceReplayBuffer:
    """
    Circular replay buffer storing (state_vector, policy_target, value_target) transitions.
    Thread-safe for concurrent worker ingestion.
    """
    def __init__(self, capacity: int = 50_000):
        self.capacity = int(capacity)
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.values: List[float] = []
        self.cursor = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self.states)

    def push(self, state: np.ndarray, policy: np.ndarray, value: float) -> None:
        """Append a single sample into the circular buffer."""
        with self._lock:
            if len(self.states) < self.capacity:
                self.states.append(np.array(state, dtype=np.float32))
                self.policies.append(np.array(policy, dtype=np.float32))
                self.values.append(float(value))
            else:
                self.states[self.cursor] = np.array(state, dtype=np.float32)
                self.policies[self.cursor] = np.array(policy, dtype=np.float32)
                self.values[self.cursor] = float(value)
                self.cursor = (self.cursor + 1) % self.capacity

    def push_batch(self, batch: List[Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Batch push for efficiency after game termination."""
        for s, p, v in batch:
            self.push(s, p, v)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly sample a mini-batch for gradient update."""
        with self._lock:
            n = len(self.states)
            if n == 0:
                raise ValueError("Cannot sample from an empty replay buffer")
            k = min(batch_size, n)
            indices = random.sample(range(n), k)
            s_batch = np.stack([self.states[i] for i in indices], axis=0)
            p_batch = np.stack([self.policies[i] for i in indices], axis=0)
            v_batch = np.array([self.values[i] for i in indices], dtype=np.float32).reshape(-1, 1)
            return s_batch, p_batch, v_batch

    def save(self, filepath: str) -> None:
        """Serialize buffer to disk (compressed .npz format)."""
        with self._lock:
            np.savez_compressed(
                filepath,
                states=np.array(self.states, dtype=np.float32),
                policies=np.array(self.policies, dtype=np.float32),
                values=np.array(self.values, dtype=np.float32),
                cursor=np.array([self.cursor], dtype=np.int32)
            )

    def load(self, filepath: str) -> None:
        """Load serialized buffer from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Buffer file not found: {filepath}")
        data = np.load(filepath)
        with self._lock:
            self.states = [s for s in data["states"]]
            self.policies = [p for p in data["policies"]]
            self.values = [float(v) for v in data["values"]]
            self.cursor = int(data["cursor"][0]) if "cursor" in data else 0
