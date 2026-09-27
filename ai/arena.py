import math
import os
import random
import torch
import numpy as np
import argparse
import sys

import game.engine as Engine
import game.updater as Updater
from game.state import GameState
from game.replay import interpret
from game.constants import *
from ai.mcts import MCTS
from ai.neural import AlphaTiger, load_checkpoint, load_dynamic_model
from ai.train import *

def _resolve_luck_log(state: GameState, log_file):
    """Resolves luck and writes what happened to the log file."""
    luck_trajectory = []
    luck_branches = []
    while state.is_luck:    
        outcomes = Updater.get_luck_outcomes(state)
        luck_branches.append(len(outcomes))
        idx = random.randrange(len(outcomes))
        state = outcomes[idx]
        luck_trajectory.append(idx)
        log_file.write(f"🎲 Luck resolved! Outcome index: {idx} of {len(outcomes)}\n")
        log_file.write(str(state) + "\n")
    return state, luck_trajectory, luck_branches


def play_match(model_mysore, model_british, sims_mysore: int, sims_british: int, log_file) -> int:
    """
    Plays a single match using pre-loaded models and logs the output.
    Returns: -1 if Mysore wins, 1 if British wins.
    """
    def log(text):
        log_file.write(str(text) + "\n")
    
    replay = []

    state = GameState()
    state.default_setup()
    state, _ , _ = _resolve_luck_log(state, log_file)
    
    # Spin up two separate MCTS brains
    mcts_mysore = MCTS(model_mysore, simulations=sims_mysore, depsilon=0)
    mcts_british = MCTS(model_british, simulations=sims_british, depsilon=0)

    move_num = 0
    luck_branching_factors = []
    decision_branching_factors = []
    disagreement_count = 0
    
    log(f"\n=== STARTING MATCH: CLASH OF THE AIs ===")
    
    while Updater.get_state_winner(state) == 0:
        current_player = 'Mysore' if state.to_move == 1 else 'British'
        
        log("-" * 60)
        log(f"MOVE {move_num} | Turn: {state.turn} | To Move: {current_player}")
        log(state)

        decision_branching_factors.append(int(np.sum(Engine.get_legal_moves(state))))
        
        # Both models evaluate with variable openings
        temperature = 1.0 if move_num < 6 else 0.0
        move_m, _ = mcts_mysore.find_move(state, temperature)
        move_b, _ = mcts_british.find_move(state, temperature)
        
        assert mcts_mysore.root is not None
        assert mcts_british.root is not None

        # Check for disagreement
        if move_m != move_b:
            disagreement_count += 1
            log("\n🚨 AIS DISAGREE ON THE BEST MOVE! 🚨")
            log(f"--> Mysore AI prefers (Eval: {mcts_mysore.root.eval:+.3f}):")
            
            # Capture engine prints by temporarily redirecting stdout
            old_stdout = sys.stdout
            sys.stdout = log_file
            Engine.print_legal_moves(np.arange(MOVE_VECTOR_LENGTH) == move_m)
            sys.stdout = old_stdout
            
            log(f"--> British AI prefers (Eval: {mcts_british.root.eval:+.3f}):")
            old_stdout = sys.stdout
            sys.stdout = log_file
            Engine.print_legal_moves(np.arange(MOVE_VECTOR_LENGTH) == move_b)
            sys.stdout = old_stdout
            log("-" * 30)
        else:
            log("\n🤝 AIs agree on the best move.")

        # The actual player's AI makes the move
        if current_player == 'Mysore':
            best_move = move_m
            active_eval = mcts_mysore.root.eval
            active_mcts = mcts_mysore
        else:
            best_move = move_b
            active_eval = mcts_british.root.eval
            active_mcts = mcts_british
            
        log(f"\n{current_player} AI executes:")
        replay.append(best_move)
        old_stdout = sys.stdout
        sys.stdout = log_file
        Engine.print_legal_moves(np.arange(MOVE_VECTOR_LENGTH) == best_move)
        sys.stdout = old_stdout
        
        assert active_mcts.root is not None
        log(f"prior: {active_mcts.root.children[best_move].prior:.3f} | eval: {active_eval:+.3f}")
            
        # Execute move and resolve luck
        state = Updater.get_next_state(state, best_move)
        state, luck_history, luck_branches = _resolve_luck_log(state, log_file)
        replay.extend(luck_history)
        luck_branching_factors.extend(luck_branches)

        # Update BOTH MCTS trees
        mcts_mysore.update_root(best_move, luck_history)
        mcts_british.update_root(best_move, luck_history)
        
        move_num += 1

    winner = Updater.get_state_winner(state)
    log("=" * 60)
    log("MATCH OVER")
    with open("replay_log.txt", "a", encoding="utf-8") as replay_file:
        replay_file.write(str(replay) + "\n")
        replay_file.write(interpret(replay)[0] + "\n")
    if winner == -1: 
        log("Winner: Mysore!")
    else: 
        log("Winner: British!")
    log("=" * 60)
    log(f"Total Disagreements: {disagreement_count}/{move_num} turns")
    if decision_branching_factors:
        log(f"Game Tree Complexity: {math.prod(decision_branching_factors):.2e}")
        log(f"Average Choices per Turn: {np.mean(decision_branching_factors):.2f}")
    if luck_branching_factors:
        log(f"Luck happened {len(luck_branching_factors)} times with {math.prod(luck_branching_factors):.2e} Possibilities")

    return winner

def run_arena(ckpt1_path: str, ckpt2_path: str, sims1: int, sims2: int, games_per_side: int, log_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(ckpt1_path) or not os.path.exists(ckpt2_path):
        print(f"Missing checkpoint! Ckpt1: {ckpt1_path} | Ckpt2: {ckpt2_path}")
        return

    # Dynamically load models regardless of their internal architecture
    try:
        model1 = load_dynamic_model(ckpt1_path, device)
        model2 = load_dynamic_model(ckpt2_path, device)
    except Exception as e:
        print(f"Failed to dynamically load models: {e}")
        return

    model1.eval()
    model2.eval()

    matrix = {
        0: {'Mysore': 0, 'British': 0},
        1: {'Mysore': 0, 'British': 0}
    }

    print(f"⚔️ ARENA INITIATED: {ckpt1_path} vs {ckpt2_path} ⚔️")
    print(f"Logging match details to: {log_path}")
    print("-" * 50)

    open("replay_log.txt", "w", encoding="utf-8")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"ARENA LOG: {ckpt1_path} vs {ckpt2_path}\n\n")

        # Phase 1: Ckpt1 as Mysore
        for i in range(games_per_side):
            log_file.write(f"\n{'#'*40}\nGAME {i+1}: Ckpt1 (Mysore) vs Ckpt2 (British)\n{'#'*40}\n")
            winner = play_match(model1, model2, sims1, sims2, log_file)
            
            if winner == -1:
                matrix[0]['Mysore'] += 1
                print(f"Game {i+1}/{games_per_side * 2} | mysore wins | Player 1")
            elif winner == 1:
                matrix[1]['British'] += 1
                print(f"Game {i+1}/{games_per_side * 2} | british win | Player 2")

        # Phase 2: Ckpt1 as British
        for i in range(games_per_side):
            log_file.write(f"\n{'#'*40}\nGAME {i+1+games_per_side}: Ckpt2 (Mysore) vs Ckpt1 (British)\n{'#'*40}\n")
            winner = play_match(model2, model1, sims2, sims1, log_file)
            
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AlphaTiger Arena and Visualizer.")
    
    parser.add_argument("--ckpt1", type=str, default=DEFAULT_MODEL, help="Path to Checkpoint 1")
    parser.add_argument("--ckpt2", type=str, default=DEFAULT_MODEL, help="Path to Checkpoint 2")
    parser.add_argument("--sims1", type=int, default=DEFAULT_SIMS, help="Simulations for Ckpt1")
    parser.add_argument("--sims2", type=int, default=DEFAULT_SIMS, help="Simulations for Ckpt2")
    
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
        log_path=args.log
    )