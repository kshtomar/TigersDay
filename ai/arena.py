import math
import random
import numpy as np
import argparse
from statistics import NormalDist

import game.engine as Engine
import game.updater as Updater
from game.state import GameState
from game.replay import interpret
from game.constants import DEFAULT_MODEL, DEFAULT_SIMS
from ai.mcts import MCTS
from ai.neural import ONNXAlphaTiger
from ai.train import _resolve_luck

def play_match(model_mysore, model_british, sims_mysore: int, sims_british: int, log_file=None, seed: int = None, record: bool = False):
    """
    Plays a single match using pre-loaded models and logs the algebraic output.
    Returns: (winner, algebraic, data) where data is (positions, masks) if record=True else None.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    replay = []
    state = GameState()
    state.default_setup()
    
    # Spin up two separate MCTS brains
    mcts_mysore = MCTS(model_mysore, depsilon=0)
    mcts_british = MCTS(model_british, depsilon=0)

    move_num = 0
    positions, masks = [], []

    while Updater.get_state_winner(state) == 0:
        current_player = 'Mysore' if state.to_move == 1 else 'British'
        if record:
            positions.append(state.vector.copy())
            masks.append(Engine.get_legal_moves(state).astype(bool))

        temperature = 1.0 if move_num < 6 else 0.0
        active_mcts = mcts_mysore if current_player == 'Mysore' else mcts_british
        active_sims = sims_mysore if current_player == 'Mysore' else sims_british
        best_move, _ = active_mcts.find_move(state, active_sims, temperature)

        replay.append(best_move)
        state = Updater.get_next_state(state, best_move)
        state, luck_history = _resolve_luck(state)
        replay.extend(luck_history)

        # Update BOTH MCTS trees
        mcts_mysore.update_root(best_move, luck_history)
        mcts_british.update_root(best_move, luck_history)
        
        move_num += 1

    winner = Updater.get_state_winner(state)
    algebraic, _ = interpret(replay)
    if log_file:
        log_file.write(str(replay) + "\n")
        log_file.write(algebraic + "\n")
        log_file.flush()

    data = (np.asarray(positions, dtype=np.float32), np.asarray(masks, dtype=bool)) if record else None
    return winner, algebraic, replay, data


def elo_summary(wins: int, games: int, confidence: float = 0.95) -> str:
    """Return the observed Elo difference and Wilson confidence interval."""
    p = wins / games
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    denom = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denom
    margin = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / denom
    low, high = center - margin, center + margin

    def elo(rate: float) -> float:
        return 400 * math.log10(rate / (1 - rate))

    point = f"{elo(p):+.0f}" if 0 < p < 1 else "N/A"
    return f"{point} Elo ({elo(low):+.0f} to {elo(high):+.0f}; {confidence:.0%} CI)"


def run_arena(ckpt1_path: str, ckpt2_path: str, sims1: int, sims2: int, games_per_side: int, log_path: str, seed: int = 190100):
    model1 = ONNXAlphaTiger(ckpt1_path)
    model2 = ONNXAlphaTiger(ckpt2_path)

    matrix = {
        0: {'Mysore': 0, 'British': 0},
        1: {'Mysore': 0, 'British': 0}
    }

    print(f"⚔️ ARENA INITIATED: {ckpt1_path} vs {ckpt2_path} ⚔️")
    print(f"Logging match details to: {log_path}")
    print("-" * 50)

    with open(log_path, "w", encoding="utf-8") as log_file:
        # Phase 1: Ckpt1 as Mysore
        for i in range(games_per_side):
            winner, _, _, _ = play_match(model1, model2, sims1, sims2, log_file, seed=seed + i if seed is not None else None)
            
            if winner == -1:
                matrix[0]['Mysore'] += 1
                print(f"Game {i+1}/{games_per_side * 2} | mysore wins | Player 1")
            elif winner == 1:
                matrix[1]['British'] += 1
                print(f"Game {i+1}/{games_per_side * 2} | british win | Player 2")

        # Phase 2: Ckpt1 as British
        for i in range(games_per_side):
            winner, _, _, _ = play_match(model2, model1, sims2, sims1, log_file, seed=seed + i if seed is not None else None)
            
            if winner == 1:
                matrix[0]['British'] += 1
                print(f"Game {i+1+games_per_side}/{games_per_side * 2} | british win | Player 1")
            elif winner == -1:
                matrix[1]['Mysore'] += 1
                print(f"Game {i+1+games_per_side}/{games_per_side * 2} | mysore wins | Player 2")

    print("🎯 ARENA RESULTS 🎯")
    print("-" * 50)
    print(f"{'':<12} | {'Mysore Wins':<13} | {'British Wins':<14}")
    print("-" * 50)
    print(f"{'Player 1':<12} | {matrix[0]['Mysore']:<13} | {matrix[0]['British']:<14}")
    print(f"{'Player 2':<12} | {matrix[1]['Mysore']:<13} | {matrix[1]['British']:<14}")

    player1_wins = matrix[0]['Mysore'] + matrix[0]['British']
    print(f"Player 1 Advantage: {elo_summary(player1_wins, games_per_side * 2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AlphaTiger Arena and Visualizer.")
    
    parser.add_argument("--ckpt1", type=str, default=DEFAULT_MODEL, help="Path to Checkpoint 1")
    parser.add_argument("--ckpt2", type=str, default=DEFAULT_MODEL, help="Path to Checkpoint 2")
    parser.add_argument("--sims1", type=int, default=DEFAULT_SIMS, help="Simulations for Ckpt1")
    parser.add_argument("--sims2", type=int, default=DEFAULT_SIMS, help="Simulations for Ckpt2")
    parser.add_argument("--seed", type=int, default=190100, help="Base seed for paired games (default: 190100)")
    
    # Control the mode: 1 game per side = quick check. 20 games per side = real arena.
    parser.add_argument("--games", type=int, default=20, help="Games to play PER SIDE (Total matches = games * 2)")
    parser.add_argument("--log", type=str, default="arena_log.txt", help="File to write the detailed turn-by-turn data to")
    
    args = parser.parse_args()
    
    run_arena(
        ckpt1_path=args.ckpt1, 
        ckpt2_path=args.ckpt2, 
        sims1=args.sims1, 
        sims2=args.sims2, 
        games_per_side=args.games,
        log_path=args.log,
        seed=args.seed
    )