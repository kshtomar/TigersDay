"""
Tiger's Day – Automated Tournament & Self-Evolving Opening Book Engine (P5.6)

Simulates arena tournament matches between agent policies or model checkpoints,
extracts high-winrate opening variations, prunes discredited lines, and
updates public/opening_book.json.
"""

import os
import json
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from game.state import GameState
from game.engine import get_legal_moves
from game.updater import get_next_state, get_luck_outcomes, get_state_winner
from game.replay import notate


def resolve_luck(state: GameState, max_depth: int = 15) -> GameState:
    """Deterministically or stochastically resolves intermediate luck states."""
    current = state
    depth = 0
    while current.is_luck and depth < max_depth:
        outcomes = get_luck_outcomes(current)
        if not outcomes:
            break
        current = random.choice(outcomes)
        depth += 1
    return current


def choose_heuristic_move(state: GameState) -> int:
    """Fast, tactically weighted heuristic move selection for tournament rollouts."""
    legal_mask = get_legal_moves(state)
    legal_indices = np.where(legal_mask)[0]
    if len(legal_indices) == 0:
        return -1

    # Prioritize cards (indices 100-300) and aggressive movement
    weights = np.ones(len(legal_indices), dtype=np.float32)
    for idx, move in enumerate(legal_indices):
        # Card plays
        if 100 <= move < 300:
            weights[idx] += 3.0
        # Movement moves
        elif move < 100:
            weights[idx] += 1.5
        # Tactical redeployment
        elif move >= 700:
            weights[idx] += 2.0

    probs = weights / np.sum(weights)
    return int(np.random.choice(legal_indices, p=probs))


def run_tournament_game(
    max_plies: int = 60,
    starting_state: Optional[GameState] = None
) -> Tuple[List[Tuple[str, int, str, int]], float]:
    """
    Simulates a single tournament game.
    Returns:
        recorded_plies: List of (state_key, move, notation, player_to_move)
        winner: 1.0 (British win), 0.0 (Mysore win), 0.5 (Draw)
    """
    if starting_state is None:
        state = GameState()
        state.default_setup()
    else:
        state = starting_state.copy()

    history: List[Tuple[str, int, str, int]] = []
    winner = 0.5

    for ply in range(max_plies):
        winner_code = get_state_winner(state)
        if winner_code != 0:
            winner = 1.0 if winner_code == 1 else 0.0
            break

        legal_mask = get_legal_moves(state)
        if not np.any(legal_mask):
            break

        current_player = state.to_move  # 0: British, 1: Mysore, 2: British Card
        move = choose_heuristic_move(state)
        if move == -1:
            break

        state_key = str(state)
        try:
            move_not = notate(state, move)
        except Exception:
            move_not = f"M{move}"

        history.append((state_key, move, move_not, current_player))

        try:
            next_state = get_next_state(state, move)
            if next_state.is_luck:
                next_state = resolve_luck(next_state)
            state = next_state
        except Exception:
            break

    # If game didn't hit terminal winner, evaluate territory dominance
    if winner == 0.5:
        british_count = np.sum(state.vector[12:87:3])
        mysore_count = np.sum(state.vector[14:87:3])
        if british_count > mysore_count:
            winner = 1.0
        elif mysore_count > british_count:
            winner = 0.0

    return history, winner


def evolve_opening_book(
    num_games: int = 50,
    max_book_ply: int = 6,
    min_winrate: float = 0.55,
    min_samples: int = 2,
    prune_threshold: float = 0.35,
    book_path: str = "public/opening_book.json",
    starting_state: Optional[GameState] = None
) -> Dict[str, Any]:
    """
    Runs arena tournament games, extracts winning variations, merges with
    the current opening book, prunes weak lines, and writes back the updated book.
    """
    # 1. Load existing book
    existing_book: Dict[str, Any] = {}
    if os.path.exists(book_path):
        try:
            with open(book_path, "r", encoding="utf-8") as f:
                existing_book = json.load(f)
        except Exception:
            existing_book = {}

    initial_count = len(existing_book)

    # 2. Run tournament games
    # state_str -> move -> {wins, total, notation}
    tournament_stats: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0.0, "total": 0, "notation": ""})
    )

    for _ in range(num_games):
        history, winner = run_tournament_game(max_plies=40, starting_state=starting_state)
        for ply_idx, (s_key, move, notation, player) in enumerate(history[:max_book_ply]):
            # Player perspective: 0/2 is British, 1 is Mysore
            side_won = winner if player in (0, 2) else (1.0 - winner)
            stats = tournament_stats[s_key][move]
            stats["total"] = int(stats["total"]) + 1
            stats["wins"] = float(stats["wins"]) + float(side_won)
            stats["notation"] = str(notation)

    added = 0
    updated = 0
    pruned = 0

    # 3. Integrate new discoveries
    for s_key, moves_dict in tournament_stats.items():
        best_candidate: Optional[int] = None
        best_winrate: float = -1.0
        best_stats: Optional[Dict[str, Any]] = None

        for move, stats in moves_dict.items():
            total_count = int(stats["total"])
            if total_count >= min_samples:
                wr = float(stats["wins"]) / total_count
                if wr >= min_winrate and wr > best_winrate:
                    best_winrate = wr
                    best_candidate = move
                    best_stats = stats

        if best_candidate is not None and best_stats is not None:
            if s_key in existing_book:
                # Merge totals
                curr = existing_book[s_key]
                if curr.get("move") == best_candidate:
                    total = int(curr.get("total", 0)) + int(best_stats["total"])
                    wins = (float(curr.get("win_rate", 0.5)) * int(curr.get("total", 0))) + float(best_stats["wins"])
                    new_wr = round(wins / max(1, total), 3)
                    existing_book[s_key]["total"] = total
                    existing_book[s_key]["win_rate"] = new_wr
                    updated += 1
                elif best_winrate > float(curr.get("win_rate", 0.0)) + 0.15:
                    # New superior variation discovered
                    existing_book[s_key] = {
                        "move": int(best_candidate),
                        "notation": str(best_stats["notation"]),
                        "total": int(best_stats["total"]),
                        "win_rate": round(best_winrate, 3)
                    }
                    updated += 1
            else:
                existing_book[s_key] = {
                    "move": int(best_candidate),
                    "notation": str(best_stats["notation"]),
                    "total": int(best_stats["total"]),
                    "win_rate": round(best_winrate, 3)
                }
                added += 1

    # 4. Prune underperforming variations with significant sample size
    keys_to_prune = []
    for s_key, entry in existing_book.items():
        if entry.get("total", 0) >= (min_samples * 2) and entry.get("win_rate", 0.5) < prune_threshold:
            keys_to_prune.append(s_key)

    for k in keys_to_prune:
        del existing_book[k]
        pruned += 1

    # 5. Persist updated opening book
    os.makedirs(os.path.dirname(book_path), exist_ok=True)
    with open(book_path, "w", encoding="utf-8") as f:
        json.dump(existing_book, f, indent=2)

    summary = {
        "initial_entries": initial_count,
        "final_entries": len(existing_book),
        "added": added,
        "updated": updated,
        "pruned": pruned,
        "games_played": num_games
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiger's Day Opening Book Evolution Pipeline")
    parser.add_argument("--games", type=int, default=20, help="Number of tournament games to simulate")
    parser.add_argument("--max-ply", type=int, default=6, help="Maximum opening ply depth to track")
    parser.add_argument("--winrate", type=float, default=0.55, help="Minimum winrate to qualify line")
    parser.add_argument("--min-samples", type=int, default=2, help="Minimum occurrences of variation")
    parser.add_argument("--output", type=str, default="public/opening_book.json", help="Path to output JSON")
    args = parser.parse_args()

    print(f"🏟️ Starting opening book evolution tournament ({args.games} games)...")
    res = evolve_opening_book(
        num_games=args.games,
        max_book_ply=args.max_ply,
        min_winrate=args.winrate,
        min_samples=args.min_samples,
        book_path=args.output
    )
    print(f"✅ Evolution complete: {res['added']} added, {res['updated']} updated, {res['pruned']} pruned. Total entries: {res['final_entries']}")
