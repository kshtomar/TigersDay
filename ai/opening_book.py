import json
import os
import ast
from collections import defaultdict
from game.state import GameState
from game.updater import get_next_state
from game.replay import notate

def build_opening_book(replay_log_path="replay_log.txt", output_path="public/opening_book.json", max_ply=4):
    """
    Parses replay_log.txt and creates an opening book dictionary keyed by state bit-string.
    Each entry provides the most successful opening move played from that state.
    """
    if not os.path.exists(replay_log_path):
        # Fallback to logs/replay_log.txt if moved
        replay_log_path = os.path.join("logs", "replay_log.txt")
        if not os.path.exists(replay_log_path):
            print(f"Warning: {replay_log_path} not found.")
            return {}

    # Map state_str -> move -> {"wins": float, "total": int, "notation": str}
    state_move_stats = defaultdict(lambda: defaultdict(lambda: {"wins": 0.0, "total": 0, "notation": ""}))

    with open(replay_log_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("[") and line.endswith("]"):
            try:
                moves = ast.literal_eval(line)
            except Exception:
                i += 1
                continue

            # Check outcome from notation line if available
            winner = 0.5
            if i + 1 < len(lines) and ("# 1-0" in lines[i+1] or "# 0-1" in lines[i+1]):
                winner_str = lines[i+1]
                if "# 1-0" in winner_str:
                    winner = 1.0 # British win
                elif "# 0-1" in winner_str:
                    winner = 0.0 # Mysore win

            state = GameState()
            state.default_setup()

            for ply, move in enumerate(moves[:max_ply]):
                state_key = str(state)
                # Faction playing
                current_player = state.to_move # 0: British Move, 1: Mysore Card, 2: British Card
                side_won = winner if current_player in (0, 2) else (1.0 - winner)

                move_notation = notate(state, move)
                stats = state_move_stats[state_key][move]
                stats["total"] += 1
                stats["wins"] += side_won
                stats["notation"] = move_notation

                try:
                    state = get_next_state(state, move)
                except Exception:
                    break

            i += 2
        else:
            i += 1

    # Convert to best move per state
    book = {}
    for state_key, moves_dict in state_move_stats.items():
        best_move = None
        best_score = -1.0
        best_stats = None

        for move, stats in moves_dict.items():
            # Score balances frequency and win rate
            score = (stats["wins"] / stats["total"]) * 0.4 + min(stats["total"], 20) * 0.03
            if score > best_score:
                best_score = score
                best_move = move
                best_stats = stats

        if best_move is not None:
            book[state_key] = {
                "move": int(best_move),
                "notation": best_stats["notation"],
                "total": best_stats["total"],
                "win_rate": round(best_stats["wins"] / max(1, best_stats["total"]), 3)
            }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(book, f, indent=2)

    print(f"✅ Opening book generated with {len(book)} state entries at {output_path}")
    return book

class OpeningBook:
    """Helper wrapper for loading and querying the opening book."""
    def __init__(self, book_path="public/opening_book.json"):
        self.book = {}
        paths = [
            book_path,
            os.path.join(os.path.dirname(__file__), "..", book_path),
            os.path.join("..", book_path)
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        self.book = json.load(f)
                    break
                except Exception:
                    pass

    def get_move(self, state):
        key = str(state)
        if key in self.book:
            return self.book[key].get("move")
        return None

if __name__ == "__main__":
    build_opening_book()
