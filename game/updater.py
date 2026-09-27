import numpy as np
from game.state import GameState
from game.constants import *

def main():
    default = GameState()
    default.default_setup()
    l = get_luck_outcomes(default)
    for state in l:
        print(state)

def get_next_state(state, move):
    #todo: optimize get_next_state function with direct indexing
    # state GSR vector, move int
    next_state = state.copy()
    offset = 0
    for name, size, move_type in MOVE_SPACE:
        if offset <= move < offset + size:
            idx = move - offset
            if move_type == "node":
                if name == "Tire":
                    next_state.set_node_tired_army(idx)
                elif name == "Sepoy Mutiny":
                    next_state.mysore_cards[1] = False
                    if state.attacker == idx:
                        next_state.clear_battle()
                    next_state.set_node_empty(idx)
                elif name == "French Alliance":
                    next_state.mysore_cards[2] = False
                    next_state.set_node_fort(idx)
                elif name == "Monsoon":
                    next_state.mysore_cards[3] = False
                    next_state.set_node_tired_army(idx)
                elif name == "Highlanders":
                    next_state.british_cards[1] = False
                    next_state.set_node_fresh_army(idx)
                elif name == "Princely States":
                    next_state.british_cards[5] = False
                    next_state.set_node_tired_army(idx)
            elif move_type == "edge":
                src = EDGE_SOURCES[idx]
                dest = EDGE_DESTS[idx]
                is_fort_defending = state.forts[dest]
                if name == "Move":
                    if is_fort_defending:
                        next_state.set_node_tired_army(src)
                        next_state.attacker = src
                        next_state.defender = dest
                        next_state.card_strength = 0
                    else:
                        next_state.set_node_empty(src)
                        next_state.set_node_tired_army(dest)
                elif name == "Divide and Rule":
                    next_state.british_cards[3] = False
                    if state.defender == src:
                        next_state.set_node_tired_army(src)
                        next_state.set_node_empty(state.attacker)
                        next_state.clear_battle()
                    else:
                        next_state.set_node_empty(src)
                    next_state.set_node_fort(dest)
                elif name == "Force March":
                    next_state.british_cards[4] = False
                    if state.attacker == src:
                        next_state.clear_battle()
                    if is_fort_defending:
                        next_state = resolve_battles(next_state, src, dest, -state.card_strength)
                    else:
                        next_state.set_node_empty(src)
                        next_state.set_node_tired_army(dest)
            elif move_type == "bcard":
                if name == "British Power":
                    next_state.british_cards[idx] = False
                    next_state = resolve_battles(next_state, NO_UNIT, NO_UNIT, CARD_VALUE[idx]-state.card_strength)
                elif name == "Draw Wall Breach":
                    next_state.british_cards[0] = False
                    next_state.british_cards[idx] = True
                elif name == "Draw Highlanders":
                    next_state.british_cards[1] = False
                    next_state.british_cards[idx] = True
                elif name == "Draw Royal Navy":
                    next_state.british_cards[2] = False
                    next_state.british_cards[idx] = True
            elif move_type == "mcard":
                if name == "Mysore Power":
                    next_state.mysore_cards[idx] = False
                    next_state.card_strength = CARD_VALUE[idx]
                elif name == "Draw Iron Rockets":
                    next_state.mysore_cards[0] = False
                    next_state.mysore_cards[idx] = True
                elif name == "Draw Sepoy Mutiny":
                    next_state.mysore_cards[1] = False
                    next_state.mysore_cards[idx] = True
                elif name == "Draw French Alliance":
                    next_state.mysore_cards[2] = False
                    next_state.mysore_cards[idx] = True
            elif move_type == "blank":
                if name == "Cavalry Raid":
                    next_state.mysore_cards[4] = False
                    next_state.bluck += 1
                elif name == "Pass Mysore":
                    break
                elif name == "Pass British":
                    break
            elif move_type == "coastal":
                node = idx // len(COASTAL_INDICES)
                coast = COASTAL_INDICES[idx % len(COASTAL_INDICES)]
                if name == "Sea Trade":
                    next_state.mysore_cards[5] = False
                    if state.defender == coast:
                        next_state.set_node_tired_army(coast)
                        next_state.set_node_empty(state.attacker)
                        next_state.clear_battle()
                    else:
                        next_state.set_node_empty(coast)
                    next_state.set_node_fort(node)
                if name == "Royal Navy":
                    next_state.british_cards[2] = False
                    if state.attacker == node:
                        next_state.clear_battle()
                    is_fort_defending = state.forts[coast]
                    if is_fort_defending:
                        next_state = resolve_battles(next_state, node, coast, -state.card_strength)
                    else:
                        is_fresh = state.fresh_armies[node]
                        next_state.set_node_empty(node)
                        if is_fresh:
                            next_state.set_node_fresh_army(coast)
                        else:
                            next_state.set_node_tired_army(coast)
            break

        offset += size

    if state.to_move == 2:
        next_state = resolve_battles(next_state, NO_UNIT, NO_UNIT, -state.card_strength)
        if state.turn != 4 and not next_state.fresh_armies.any():
            next_state.turn_refresh()
    next_state.to_move += 1
    return next_state

def get_state_winner(state):
    if np.sum((state.tired_armies | state.fresh_armies) & KEYS) == 5:
        return 1
    elif state.turn == 4 and not state.fresh_armies.any() and state.to_move == 0:
        return -1
    else:
        return 0

def is_battle_won(state, defender, net_card_strength):
    adjacent_mask = ADJACENCY_MATRIX[defender]
    attacker_strength = np.sum((state.fresh_armies | state.tired_armies) & adjacent_mask)
    defender_strength = np.sum(state.forts & adjacent_mask)
    return attacker_strength + net_card_strength > defender_strength

def resolve_battles(state, attacker, defender, net_card_strength):
    battle1 = False
    battle2 = False
    #first battle from state, second battle from parameters
    if state.attacker != NO_UNIT:
        battle1 = is_battle_won(state, state.defender, net_card_strength)
        if battle1:
            state.mluck += 1
        else:
            state.bluck += 1
    if attacker != NO_UNIT:
        battle2 = is_battle_won(state, defender, 0)
        if battle2:
            state.mluck += 1
        else:
            state.bluck += 1
    #preserve fresh/tired state of attacker
    if battle1:
        state.set_node_tired_army(state.defender)
        state.set_node_empty(state.attacker)
    if battle2:
        is_fresh = state.fresh_armies[attacker]
        state.set_node_empty(attacker)
        if is_fresh:
            state.set_node_fresh_army(defender)
        else:
            state.set_node_tired_army(defender)
    state.clear_battle()
    return state

def get_luck_outcomes(state):
    outcomes = []
    if state.bluck:
        for i in range(CARDS):
            if state.british_cards[i]:
                luck_state = state.copy()
                luck_state.british_cards[i] = False
                luck_state.bluck -= 1
                outcomes.append(luck_state)
        if not outcomes:
            luck_state = state.copy()
            luck_state.bluck -= 1
            return [luck_state]
        return outcomes
    
    if state.mluck:
        for i in range(CARDS):
            if state.mysore_cards[i]:
                luck_state = state.copy()
                luck_state.mysore_cards[i] = False
                luck_state.mluck -= 1
                outcomes.append(luck_state)
        if not outcomes:
            luck_state = state.copy()
            luck_state.mluck -= 1
            return [luck_state]
        return outcomes
    # no luck outcomes
    return [state]

"""
1) All the cards effects + Power play by either side
2) Random discards
3) Resolve battles  
"""
if __name__ == "__main__":
    main()