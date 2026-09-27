import game.updater as Updater
from game.state import GameState
from game.constants import *

def interpret(replay_log):
    state = GameState()
    state.default_setup()

    algebraic = ""
    state_history = [state.vector.copy()]
    turn = state.turn

    for move in replay_log:
        if not state.is_luck:
            algebraic += notate(state, move) + " "
            state = Updater.get_next_state(state,move)
        else:
            state = Updater.get_luck_outcomes(state)[move]
            if len(np.where(state_history[-1] != state.vector)[0]) > 0:
                index = np.where(state_history[-1] ^ state.vector)[0][0]
                if index >= CARDS:
                    algebraic += CARDS_ABBREV[MYSORE_CARDS[index - CARDS]] + " "
                else:
                    algebraic += CARDS_ABBREV[BRITISH_CARDS[index]] + " "

        if state.turn != turn:
            algebraic += "+ "
            turn = state.turn

        state_history.append(state.vector.copy())

    algebraic += "# "
    if Updater.get_state_winner(state) == 1:
        algebraic += "1-0"
    else:
        algebraic += "0-1"

    return algebraic, state_history

def notate(state, move):
    offset = 0
    gap = ">"
    move_string = ""
    for name, size, move_type in MOVE_SPACE:
        if offset <= move < offset + size:
            idx = move - offset
            if name in CARDS_ABBREV:
                move_string += CARDS_ABBREV[name] + ":"
            if move_type == "node":
                move_string += NODES_ABBREV[idx]
                return move_string
            elif move_type == "edge":
                if state.forts[int(EDGE_DESTS[idx])]:
                    if name == "Move" or name == "Force March":
                        gap = "x"
                src_name = NODES_ABBREV[int(EDGE_SOURCES[idx])] 
                dest_name = NODES_ABBREV[int(EDGE_DESTS[idx])] 
                return move_string + src_name + gap + dest_name
            elif move_type == "bcard":
                if name == "British Power":
                    move_string += CARDS_ABBREV[BRITISH_CARDS[idx]] + ":"
                    card_name = "x"
                else:
                    card_name = CARDS_ABBREV[BRITISH_CARDS[idx]]
                return move_string + card_name
            elif move_type == "mcard":
                if name == "Mysore Power":
                    move_string += CARDS_ABBREV[MYSORE_CARDS[idx]] + ":"
                    card_name = "x"
                else:
                    card_name = CARDS_ABBREV[MYSORE_CARDS[idx]]
                return move_string + card_name
            elif move_type == "blank":
                if name == "Cavalry Raid":
                    return move_string
                else:
                    return "pass"
            elif move_type == "coastal":
                node = NODES_ABBREV[idx // len(COASTAL_INDICES)]
                coast = NODES_ABBREV[int(COASTAL_INDICES[idx % len(COASTAL_INDICES)])]
                if name == "Royal Navy":
                    if state.forts[int(COASTAL_INDICES[idx % len(COASTAL_INDICES)])]:
                        gap = "x"
                    return move_string + node + gap + coast
                elif name == "Sea Trade":
                    return move_string + coast + gap + node
        offset += size
    return "none"

def parse_replay_log(filepath):
    """Reads the log file and extracts just the move lists."""
    games = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('['):
                continue
            
            moves = line.split()
            games.append(moves)        
    return games

def build_move_tree(games, max_depth=6):
    """
    Builds a tree structure tracking how many times each sequence was played.
    Structure: { 'mad>pdc': {'count': 45, 'next': { 'SM:trv': ... } } }
    """
    tree = {}

    for game in games:
        moves = [
            move for move in game
            if move not in {"+", "#", "1-0", "0-1"}
        ]

        if "1-0" in game:
            british_win = True
        elif "0-1" in game:
            british_win = False
        else:
            continue

        current_node = tree

        for move in moves[:max_depth]:
            if move not in current_node:
                current_node[move] = {
                    "count": 0,
                    "british_wins": 0,
                    "next": {}
                }

            node = current_node[move]

            node["count"] += 1

            if british_win:
                node["british_wins"] += 1

            current_node = node["next"]

    return tree

def get_overall_british_wr(games):
    british_wins = 0
    total_games = 0

    for game in games:
        if "1-0" in game:
            british_wins += 1
            total_games += 1
        elif "0-1" in game:
            total_games += 1

    return british_wins / total_games

def beta_wr(british_wins, games, overall_wr, prior_strength=10):
    alpha = overall_wr * prior_strength
    beta = (1 - overall_wr) * prior_strength

    return (
        british_wins + alpha
    ) / (
        games + alpha + beta
    )

def print_tree(tree, current_depth=1, max_depth=6, min_games=3, indent="", overall_wr=0.5, prior_strength=10):
    """Recursively prints the tree sorted by the most popular moves."""
    sorted_moves = sorted(
        tree.items(),
        key=lambda item: item[1]["count"],
        reverse=True
    )

    for move, data in sorted_moves:
        count = data["count"]

        if count < min_games:
            continue

        british_wr = beta_wr(
            data["british_wins"],
            count,
            overall_wr,
            prior_strength
        )

        print(
            f"{indent}├── {move} "
            f"[{count} games - "
            f"WR {british_wr * 100:.1f}%]"
        )

        if current_depth < max_depth and data["next"]:
            print_tree(
                data["next"],
                current_depth=current_depth + 1,
                max_depth=max_depth,
                min_games=min_games,
                indent=indent + "│  ",
                overall_wr=overall_wr,
                prior_strength=prior_strength
            )

import re
from collections import Counter
from game.constants import NODES_ABBREV

def count_location_codes(filepath):
    counts = Counter()

    valid_locations = list(NODES_ABBREV.values())
    location_pattern = re.compile(r'|'.join(valid_locations))

    with open(filepath, "r") as f:
        for line in f:
            matches = location_pattern.findall(line.lower())
            counts.update(matches)

    return counts

def get_popular_moves(filepath, threshold=300):
    move_counts = Counter()

    with open(filepath, "r") as f:
        for line in f:
            if not line or line.startswith('['):
                continue
            tokens = line.split()
            valid_moves = [token for token in tokens if token not in {"+", "#", "1-0", "0-1"}]
            move_counts.update(valid_moves)

    print(f"MOVES PLAYED {threshold} TIMES")
    print("=" * 25)
    
    for move, count in move_counts.most_common():
        if count >= threshold:
            print(f"{move} {count}")
        else:
            break

if __name__ == "__main__":
    filepath = "replay_log.txt"

    games = parse_replay_log(filepath)

    print(f"Total Games Parsed: {len(games)}\n")
    print("OPENING BOOK")
    print("============")

    depth_to_analyze = 6
    min_games = 5

    move_tree = build_move_tree(
        games,
        max_depth=depth_to_analyze
    )

    print_tree(
        move_tree,
        max_depth=depth_to_analyze,
        min_games=min_games
    )

    counts = count_location_codes(filepath)

    print("LOCATION CODE FREQUENCIES")
    print("=========================")

    for code, count in counts.most_common():
        print(f"{code}: {count}")

    get_popular_moves(filepath)