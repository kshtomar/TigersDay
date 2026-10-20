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


Sample = Tuple[np.ndarray, np.ndarray, np.ndarray, float]

@dataclass
class CurriculumStage:
    name: str
    state_factory: Callable[[], GameState]
    iterations: int
    simulations: int = 100
    temperature: float = 1.0
    # Set temperature to 0 after this many moves
    temperature_cutoff: int = 0


@dataclass
class TrainerConfig:
    buffer_size: int = 20000
    batch_size: int = 256
    # Avoid overfitting early batches
    min_buffer_size: int = 1000

    lr: float = 1e-3
    weight_decay: float = 1e-4

    # For each new position collected, how many replay samples to train on
    train_ratio: float = 8.0

    # Fraction of moves using full MCTS searches
    pcr: float = 0.25

    # None selects an available accelerator, falling back to CPU
    device: Optional[str] = None

    checkpoint_dir: str = "checkpoints"
    save_every: int = 100


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer: deque[Sample] = deque(maxlen=maxlen)

    def add(self, samples: List[Sample]) -> None:
        self.buffer.extend(samples)

    def sample(self, n: int) -> List[Sample]:
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class TrainingBudget:
    """Accumulate training credit from data, accumulating fractions across games."""
    def __init__(self, ratio):
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError("train_ratio must be finite and nonnegative")
        self.ratio = ratio
        self.credit = 0.0

    def steps_for(self, new_samples, batch_size, *, ready):
        if not ready:
            return 0
        self.credit += self.ratio * new_samples
        steps = int(self.credit // batch_size)
        self.credit -= steps * batch_size
        return steps


def _resolve_luck(state: GameState) -> tuple[GameState, list[int]]:
    luck_trajectory = []
    while state.is_luck:
        outcomes = Updater.get_luck_outcomes(state)

        idx = random.randrange(len(outcomes))

        state = outcomes[idx]
        # Return trajectory to allow game reconstruction
        luck_trajectory.append(idx)
    return state, luck_trajectory


def self_play_game(
    mcts: MCTS,
    state_factory: Callable[[], GameState],
    temperature: float,
    temperature_cutoff: int,
    simulations: int,
    pcr: float = 0.25):
    if not 0 <= pcr <= 1:
        raise ValueError("pcr must be between 0 and 1")
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
        if random.random() < pcr:
            move, policy = mcts.find_move(state, simulations, temp)
            # Train only on full searches
            legal_mask = Engine.get_legal_moves(state)
            history.append((state.vector.copy(), policy, legal_mask))
        else:
            # No dirichlet noise on fast searches
            move, _ = mcts.find_move(state, simulations // 10, temp, depsilon=0.0)

        state = Updater.get_next_state(state, move)
        state, luck_history = _resolve_luck(state)
        mcts.update_root(move, luck_history)

        move_num += 1

    winner = Updater.get_state_winner(state)
    samples = [(sv, pt, lm, np.float32(winner)) for sv, pt, lm in history]
    return samples, move_num, winner


def train_step(
    model: AlphaTiger,
    optimizer: optim.Optimizer,
    batch: List[Sample],
    device: torch.device
) -> Tuple[float, float, float]:
    states, policies, masks, values = zip(*batch)

    state_t   = torch.tensor(np.array(states),   dtype=torch.float32, device=device)
    policy_t  = torch.tensor(np.array(policies), dtype=torch.float32, device=device)
    value_t   = torch.tensor(np.array(values),   dtype=torch.float32, device=device).unsqueeze(1)
    mask_t    = torch.tensor(np.array(masks),    dtype=torch.bool,    device=device)

    pred_value, pred_logits = model(state_t)
    pred_logits = pred_logits.masked_fill(~mask_t, -1e9)

    policy_loss = F.cross_entropy(pred_logits, policy_t)

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
    *,
    device: Optional[str] = None,
) -> AlphaTiger:
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device(device or config.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    model = AlphaTiger().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    buffer    = ReplayBuffer(config.buffer_size)
    training_budget = TrainingBudget(config.train_ratio)

    global_iter = 0
    if resume_path:
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
            model.eval()
            samples, game_length, winner = self_play_game(
                mcts,
                stage.state_factory,
                stage.temperature,
                stage.temperature_cutoff,
                stage.simulations,
                pcr=config.pcr
            )
            buffer.add(samples) # type: ignore 
            global_iter += 1

            # ── Training ─────────────────────────────────────────────────────
            total_loss = val_loss = pol_loss = 0.0
            steps = 0

            model.train()
            training_batch_size = min(config.batch_size, len(buffer))
            train_steps = training_budget.steps_for(
                len(samples), training_batch_size,
                ready=len(buffer) >= config.min_buffer_size and training_batch_size > 0
            )
            if train_steps:
                for _ in range(train_steps):
                    batch = buffer.sample(config.batch_size)
                    tl, vl, pl = train_step(model, optimizer, batch, device)
                    total_loss += tl
                    val_loss   += vl
                    pol_loss   += pl
                    steps      += 1

            # ── Logging ──────────────────────────────────────────────────────
            prefix = (
                f"[{stage.name}] iter {i+1:>4}/{stage.iterations} | buf {len(buffer):>5}"
                f" | moves {game_length:>3} | samples {len(samples):>3}"
                f" | winner {('british' if winner == 1 else 'mysore'):<7}"
            )
            if steps:
                print(
                    f"{prefix} | loss {total_loss/steps:6.4f} "
                    f"(val {val_loss/steps:6.4f} pol {pol_loss/steps:6.4f})"
                )
            elif len(buffer) < config.min_buffer_size:
                print(f"{prefix} | warming up ({len(buffer)}/{config.min_buffer_size})")
            else:
                print(f"{prefix} | train credit {training_budget.credit:.0f}/{training_batch_size}")

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
    """Default setup position."""
    state = GameState()
    state.default_setup()
    return state

def stage_early_game() -> GameState:
    """Handmade position after the opening of the game."""
    state = GameState()
    state.default_setup()
    state.set_node_empty(random.randrange(NODES))
    state = perturb_state(state, 3)
    return state

def stage_mid_game() -> GameState:
    """Handmade position midway through the game."""
    state = GameState()
    for _ in range(random.randrange(5,7)):
        state.set_node_fresh_army(random.randrange(NODES))
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    state.set_node_fort(NODE_TO_IDX["Erode"])
    state.set_node_fort(NODE_TO_IDX["Mahé"])
    for _ in range(random.randrange(2,5)):
        state.set_node_fort(random.randrange(NODES))
    state.set_node_fresh_army(random.randrange(NODES))
    state.turn = 2
    state = perturb_state(state, 3)
    return state

def stage_late_game() -> GameState:
    """Handmade position late in the game."""
    state = GameState()
    for _ in range(random.randrange(4,7)):
        state.set_node_fresh_army(random.randrange(NODES))
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    for _ in range(random.randrange(2,5)):
        state.set_node_fort(random.randrange(NODES))
    state.set_node_fresh_army(random.randrange(NODES))
    state.turn = 3
    state = perturb_state(state, 3)
    return state

def stage_end_game() -> GameState:
    """Handmade position before the ending of the game."""
    state = GameState()
    for _ in range(random.randrange(6)):
        state.set_node_fort(random.randrange(NODES))
    for _ in range(random.randrange(6,8)):
        state.set_node_fresh_army(random.randrange(NODES))
    state.turn = 4
    state = perturb_state(state, 3)
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


class MixedStateFactory:
    """Randomly choose from provided state factories."""
    def __init__(self, factories):
        self.factories = tuple(factories)

    def __call__(self):
        return random.choice(self.factories)()

def setup_training_run(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--resume", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--sims", type=int, default=None, help="Override MCTS simulations for all stages")
    parser.add_argument("--iters", type=int, default=None, help="Override games per stage for all stages")
    parser.add_argument("--finetune", action="store_true",
                        help="Resume with a random curriculum phase for each game")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                        help="Training device (automatically selected when omitted)")
    parser.add_argument("--pcr",
                        type=float, default=0.25,
                        help="Fraction of full searches between 0 and 1 (default: 0.25; 1 disables PCR)")
    args = parser.parse_args()
    if not 0 <= args.pcr <= 1:
        parser.error("pcr must be between 0 and 1")

    if args.finetune:
        curriculum = [
            CurriculumStage(name="Fine Tune", state_factory=MixedStateFactory([
                stage_end_game, stage_late_game, stage_mid_game,
                stage_early_game, stage_full_game,
            ]), iterations=5000, simulations=4000, temperature_cutoff=21,
        )]
    else:
        curriculum = [
            CurriculumStage(name="End Game",   state_factory=stage_end_game,   iterations=5000, simulations=800, temperature_cutoff = 6),
            CurriculumStage(name="Late Game",  state_factory=stage_late_game,  iterations=5000, simulations=800, temperature_cutoff = 9),
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

    config = TrainerConfig(pcr=args.pcr, device=args.device)

    return args, curriculum, config

if __name__ == "__main__":
    args, curriculum, config = setup_training_run("AlphaTiger Single Thread Trainer")
    train(curriculum, config=config, resume_path=args.resume)
