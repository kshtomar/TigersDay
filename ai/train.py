import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import game.updater as Updater
import game.engine as Engine
from game.constants import *
from ai.mcts import MCTS
from ai.neural import AlphaTiger, load_checkpoint, save_checkpoint
from game.state import GameState


Sample = Tuple[np.ndarray, np.ndarray, float]

@dataclass
class CurriculumStage:
    """
   `state_factory` is a zero-argument callable that returns a fresh GameState.
    This is the hook for curriculum learning — early stages can return simplified
    or mid-game states so the model learns easier patterns first.

    """
    name: str
    state_factory: Callable[[], GameState]
    iterations: int                         # self-play + train cycles for this stage
    simulations: int = 100                  # MCTS simulations per move
    temperature: float = 1.0               # softmax temperature for move selection
    temperature_cutoff: int = 0           # after this many moves, temperature → 0


@dataclass
class TrainerConfig:
    # ── Replay buffer ──────────────────────────────────────────────────────────
    buffer_size: int = 50_000
    batch_size: int = 256
    # Don't start gradient updates until the buffer holds this many samples.
    # Prevents the network from over-fitting tiny early batches.
    min_buffer_size: int = 1_000

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # ── Training ──────────────────────────────────────────────────────────────
    # How many gradient steps to take after each self-play game.
    train_steps_per_iter: int = 1
    puct: float = 1.5

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every: int = 100                   # save a checkpoint every N global iters


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer: deque[Sample] = deque(maxlen=maxlen)

    def add(self, samples: List[Sample]) -> None:
        self.buffer.extend(samples)

    def sample(self, n: int) -> List[Sample]:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def _resolve_luck(state: GameState) -> tuple[GameState, list[int]]:
    """Randomly resolve any pending luck outcomes (bluck/mluck)."""
    luck_trajectory = []
    while state.is_luck:
        outcomes = Updater.get_luck_outcomes(state)

        idx = random.randrange(len(outcomes))

        state = outcomes[idx]
        luck_trajectory.append(idx)
    return state, luck_trajectory


def self_play_game(
    mcts: MCTS,
    state_factory: Callable[[], GameState],
    temperature: float,
    temperature_cutoff: int,
    simulations: int):
    """
    Play one game via MCTS self-play from `state_factory()`.

    Returns a list of training samples:  (state_vector, policy_target, outcome)
    where outcome is -1 for Mysore win and +1 for British win absolutely.

    Only decision states are recorded — luck resolutions have no
    learnable policy, so they are skipped.
    """
    state = state_factory()
    state, _ = _resolve_luck(state)

    history = []
    move_num = 0

    while True:
        winner = Updater.get_state_winner(state)
        if winner != 0:
            break

        temp = temperature if move_num < temperature_cutoff else 0.0

        # Playout cap randomization
        if random.random() < 0.25:
            move, policy = mcts.find_move(state, simulations, temp)
            pweight = 1.0
        else:
            move, policy = mcts.find_move(state, simulations // 10, temp)
            pweight = 0.0

        legal_mask = Engine.get_legal_moves(state)

        # Record this decision point
        history.append((state.vector.copy(), policy, legal_mask, np.float32(pweight)))

        # Sample a move and advance the state
        state = Updater.get_next_state(state, move)
        state, luck_history = _resolve_luck(state)
        mcts.update_root(move, luck_history)

        move_num += 1

    winner = Updater.get_state_winner(state)
    return [(sv, pt, lm, pw, np.float32(winner)) for sv, pt, lm, pw in history]


def train_step(
    model: AlphaTiger,
    optimizer: optim.Optimizer,
    batch: List[Sample],
    device: torch.device
) -> Tuple[float, float, float]:
    """One gradient update. Returns (total, value, policy) losses."""
    states, policies, masks, pweights, values = zip(*batch)

    state_t   = torch.tensor(np.array(states),   dtype=torch.float32, device=device)
    policy_t  = torch.tensor(np.array(policies), dtype=torch.float32, device=device)
    value_t   = torch.tensor(np.array(values),   dtype=torch.float32, device=device).unsqueeze(1)
    mask_t    = torch.tensor(np.array(masks),    dtype=torch.bool,    device=device)
    pweight_t = torch.tensor(np.array(pweights), dtype=torch.float32, device=device)

    pred_value, pred_logits = model(state_t)
    pred_logits = pred_logits.masked_fill(~mask_t, -1e9)

    # Mask out fast searches from policy loss
    raw_policy_loss = F.cross_entropy(pred_logits, policy_t, reduction = "none")
    policy_samples = pweight_t.sum().clamp(min=1.0)
    policy_loss = (raw_policy_loss * pweight_t).sum() / policy_samples

    value_loss  = F.mse_loss(pred_value, value_t)

    loss = value_loss + policy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), value_loss.item(), policy_loss.item()


def train(
    curriculum: List[CurriculumStage],
    config: TrainerConfig = TrainerConfig(),
    resume_path: Optional[str] = None,
) -> AlphaTiger:
    """
    Run the full curriculum.

    Args:
        curriculum:   Ordered list of CurriculumStage objects.
        config:       Hyper-parameters and I/O settings.
        resume_path:  Optional path to a checkpoint to resume from.

    Returns:
        The trained AlphaTiger model.
    """
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model     = AlphaTiger().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    buffer    = ReplayBuffer(config.buffer_size)

    global_iter = 0
    if resume_path and os.path.exists(resume_path):
        global_iter = load_checkpoint(model, optimizer, resume_path)
        print(f"Resumed ({model}) from iteration {global_iter}  ({resume_path}) on ({device})")

    for stage in curriculum:
        print(f"\n{'='*60}")
        print(f"  Stage: {stage.name}")
        print(f"  Iterations: {stage.iterations}  |  Simulations: {stage.simulations}")
        print(f"  Temperature: {stage.temperature}  (greedy after move {stage.temperature_cutoff})")
        print(f"{'='*60}")

        for i in range(stage.iterations):
            mcts = MCTS(model)
            # ── Self-play ────────────────────────────────────────────────────
            model.eval()
            samples = self_play_game(
                mcts,
                stage.state_factory,
                stage.temperature,
                stage.temperature_cutoff,
                stage.simulations
            )
            buffer.add(samples) # type: ignore 
            global_iter += 1

            # ── Training ─────────────────────────────────────────────────────
            total_loss = val_loss = pol_loss = 0.0
            steps = 0

            model.train()
            if len(buffer) >= config.min_buffer_size:
                for _ in range(config.train_steps_per_iter):
                    batch = buffer.sample(config.batch_size)
                    tl, vl, pl = train_step(model, optimizer, batch, device)
                    total_loss += tl
                    val_loss   += vl
                    pol_loss   += pl
                    steps      += 1

            # ── Logging ──────────────────────────────────────────────────────
            prefix = f"[{stage.name}] iter {i+1:>4}/{stage.iterations} | buf {len(buffer):>6}"
            if steps:
                if len(samples) == 0:
                    print("DEBUG: Game ended with zero samples! Check your GameState initialization.")
                else:
                    print(
                        f"{prefix} | loss {total_loss/steps:.4f} "
                        f"(val {val_loss/steps:.4f}  pol {pol_loss/steps:.4f})"
                        f" | game len {len(samples)} | winner {'british' if samples[0][-1] == 1 else 'mysore'}"
                    )
            else:
                print(f"{prefix} | warming up ({len(buffer)}/{config.min_buffer_size})")

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_iter % config.save_every == 0:
                path = os.path.join(config.checkpoint_dir, f"ckpt_{global_iter:06d}.pt")
                save_checkpoint(model, optimizer, global_iter, path)
                print(f"  ↳ saved {path}")

    final_path = os.path.join(config.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, global_iter, final_path)
    print(f"\nTraining complete — final model saved to {final_path}")
    return model


def stage_full_game() -> GameState:
    """Standard starting position."""
    state = GameState()
    state.default_setup()
    return state

def stage_early_game() -> GameState:
    """A reasonable position after the opening of the game."""
    state = GameState()
    state.default_setup()
    state.set_node_empty(random.randrange(NODES))
    state = perturb_state(state, 9)
    return state

def stage_mid_game() -> GameState:
    """A reasonable position midway through the game."""
    state = GameState()
    for _ in range(random.randrange(6,8)):
        state.set_node_fresh_army(random.randrange(NODES))
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    state.set_node_fort(NODE_TO_IDX["Erode"])
    state.set_node_fort(NODE_TO_IDX["Mahé"])
    for _ in range(random.randrange(4)):
        state.set_node_fort(random.randrange(NODES))
    state.turn = 2
    state = perturb_state(state, 9)
    return state

def stage_late_game() -> GameState:
    """A reasonable position late in the game."""
    state = GameState()
    for _ in range(random.randrange(6,8)):
        state.set_node_fresh_army(random.randrange(NODES))
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    for _ in range(random.randrange(6)):
        state.set_node_fort(random.randrange(NODES))
    state.turn = 3
    state = perturb_state(state, 6)
    return state

def stage_end_game() -> GameState:
    """A reasonable position before the ending of the game."""
    state = GameState()
    for _ in range(random.randrange(8)):
        state.set_node_fort(random.randrange(NODES))
    for _ in range(random.randrange(6,8)):
        state.set_node_fresh_army(random.randrange(NODES))
    state.turn = 4
    state = perturb_state(state, 6)
    return state

def perturb_state(state, depth):
    for _ in range(depth):
        if Updater.get_state_winner(state) != 0:
            break

        legal_mask = Engine.get_legal_moves(state)
        state = Updater.get_next_state(state, np.random.choice(np.nonzero(legal_mask)[0]))
        while state.is_luck:
            outcomes = Updater.get_luck_outcomes(state)
            state = random.choice(outcomes)
    return state

def setup_training_run(description: str):
    """Parses CLI args, builds the curriculum, and configures the trainer."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--resume", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--sims", type=int, default=None, help="Override MCTS simulations for all stages")
    parser.add_argument("--iters", type=int, default=None, help="Override games per stage for all stages")
    parser.add_argument("--cycle", action="store_true", help="Run 10 cycles of the 5 main stages instead")
    args = parser.parse_args()

    if args.cycle:
        curriculum = []
        for i in range(10):
            curriculum.extend([
                CurriculumStage(name="End Game",   state_factory=stage_end_game,   iterations=100, simulations=4000, temperature_cutoff = 9),
                CurriculumStage(name="Late Game",  state_factory=stage_late_game,  iterations=100, simulations=4000, temperature_cutoff = 12),
                CurriculumStage(name="Mid Game",   state_factory=stage_mid_game,   iterations=100, simulations=4000, temperature_cutoff = 15),
                CurriculumStage(name="Early Game", state_factory=stage_early_game, iterations=100, simulations=4000, temperature_cutoff = 18),
                CurriculumStage(name="Full Game",  state_factory=stage_full_game,  iterations=100, simulations=4000, temperature_cutoff = 21),
            ])
    else:
        curriculum = [
            CurriculumStage(name="End Game",   state_factory=stage_end_game,   iterations=5000, simulations=400, temperature_cutoff = 6),
            CurriculumStage(name="Late Game",  state_factory=stage_late_game,  iterations=5000, simulations=400, temperature_cutoff = 9),
            CurriculumStage(name="Mid Game",   state_factory=stage_mid_game,   iterations=5000, simulations=800, temperature_cutoff = 12),
            CurriculumStage(name="Early Game", state_factory=stage_early_game, iterations=5000, simulations=800, temperature_cutoff = 15),
            CurriculumStage(name="Full Game",  state_factory=stage_full_game,  iterations=5000, simulations=800, temperature_cutoff = 18),
            CurriculumStage(name="Deep Game",  state_factory=stage_full_game,  iterations=1000, simulations=4000, temperature_cutoff = 21),
        ]

    # Apply overrides
    if args.sims is not None:
        for stage in curriculum:
            stage.simulations = args.sims
    if args.iters is not None:
        for stage in curriculum:
            stage.iterations = args.iters

    config = TrainerConfig()

    return args, curriculum, config

if __name__ == "__main__":
    args, curriculum, config = setup_training_run("AlphaTiger Single Thread Trainer")
    train(curriculum, config=config, resume_path=args.resume)
