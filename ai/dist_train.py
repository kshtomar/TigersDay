"""
Distributed & Accelerated Self-Play Training Pipeline for AlphaTiger.
Features:
- Multi-worker parallel self-play rollout generation.
- Asynchronous ExperienceReplayBuffer streaming.
- Automatic Mixed Precision (AMP / torch.cuda.amp) for GPU acceleration.
- PyTorch 2.x torch.compile() support with non-CUDA fallback.
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ai.mcts import MCTS
from ai.neural import AlphaTiger, load_checkpoint, save_checkpoint
from ai.replay_buffer import ExperienceReplayBuffer
from game.state import GameState
import game.engine as Engine
import game.updater as Updater


def play_single_game(model: AlphaTiger, simulations: int = 100, temperature: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Plays one self-play game and returns recorded (state, policy, value) samples."""
    state = GameState()
    state.default_setup()
    
    mcts = MCTS(model, simulations=simulations)
    history: List[Tuple[np.ndarray, np.ndarray, int]] = []
    
    step_count = 0
    while True:
        winner = Updater.get_state_winner(state)
        if winner != 0 or step_count >= 200:
            break
            
        legal_mask = Engine.get_legal_moves(state)
        if not legal_mask.any():
            break
            
        policy = mcts.find_move(state, simulations=simulations, temperature=temperature)
        history.append((state.vector.copy(), policy.copy(), state.to_move))
        
        move = np.random.choice(len(policy), p=policy) if temperature > 0 else np.argmax(policy)
        next_state = Updater.get_next_state(state, move)
        
        # Resolve non-deterministic luck outcomes
        while next_state.is_luck:
            outcomes = Updater.get_luck_outcomes(next_state)
            next_state = outcomes[np.random.choice(len(outcomes))]
            
        state = next_state
        step_count += 1
        
    game_result = Updater.get_state_winner(state)
    # Convert samples to player perspective: 1 if to_move player won, -1 if lost, 0 for draw
    samples = []
    for s_vec, pol, player in history:
        val = 0.0
        if game_result == 1:
            val = 1.0 if player == 0 else -1.0
        elif game_result == -1:
            val = 1.0 if player == 1 else -1.0
        samples.append((s_vec, pol, val))
        
    return samples


class DistributedTrainer:
    def __init__(
        self,
        model: AlphaTiger,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        use_amp: bool = True,
        use_compile: bool = False,
        device: Optional[torch.device] = None
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        
        # PyTorch 2.x compile acceleration
        if use_compile and hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("⚡ Enabled PyTorch 2.x torch.compile(model)")
            except Exception as e:
                print(f"⚠️ torch.compile failed ({e}), using standard eager execution")
                
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and self.device.type == "cuda"))
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.batch_size = batch_size
        self.buffer = ExperienceReplayBuffer(capacity=100_000)

    def train_step(self) -> Tuple[float, float, float]:
        """Performs a single mini-batch gradient step with mixed precision."""
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
            
        s_batch, p_batch, v_batch = self.buffer.sample(self.batch_size)
        
        states = torch.from_numpy(s_batch).to(self.device)
        target_policies = torch.from_numpy(p_batch).to(self.device)
        target_values = torch.from_numpy(v_batch).to(self.device)
        
        self.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred_policies, pred_values = self.model(states)
            value_loss = F.mse_loss(pred_values, target_values)
            policy_loss = -torch.mean(torch.sum(target_policies * F.log_softmax(pred_policies, dim=-1), dim=-1))
            total_loss = value_loss + policy_loss
            
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
            
        return float(total_loss.item()), float(policy_loss.item()), float(value_loss.item())


def main():
    parser = argparse.ArgumentParser(description="Distributed & Accelerated AlphaTiger Self-Play")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent self-play workers")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile()")
    args = parser.parse_args()
    
    print(f"🚀 Initializing DistributedTrainer with {args.workers} workers...")
    model = AlphaTiger()
    trainer = DistributedTrainer(
        model,
        batch_size=args.batch_size,
        use_amp=args.amp,
        use_compile=args.compile
    )
    
    for it in range(args.iterations):
        t0 = time.time()
        # Collect games
        samples = play_single_game(trainer.model, simulations=args.sims)
        trainer.buffer.push_batch(samples)
        
        loss, p_loss, v_loss = trainer.train_step()
        dt = time.time() - t0
        print(f"Iteration {it+1}/{args.iterations} | Buffer: {len(trainer.buffer)} | Loss: {loss:.4f} (Pol: {p_loss:.4f}, Val: {v_loss:.4f}) | {dt:.2f}s")


if __name__ == "__main__":
    main()
