"""
Historical Campaign Scenarios for The Tiger's Day (1767–1799).
Provides distinct historical initializations across the four Anglo-Mysore Wars.
"""

from typing import Dict, List, Any
from game.constants import NODE_TO_IDX
from game.state import GameState


SCENARIOS_METADATA = [
    {
        "id": "first_anglo_mysore_war",
        "name": "First Anglo-Mysore War",
        "years": "1767–1769",
        "subtitle": "Hyder Ali's Counter-Offensive",
        "description": "Hyder Ali drives rapidly toward the gates of Madras. British forces must defend their coastal presidencies while Mysore forces mount rapid cavalry maneuvers."
    },
    {
        "id": "second_anglo_mysore_war",
        "name": "Second Anglo-Mysore War",
        "years": "1780–1784",
        "subtitle": "Battle of Pollilur & French Alliance",
        "description": "Tipu Sultan deploys iron-cased Mysorean rockets against Colonel Baillie at Pollilur. French naval forces threaten British maritime transit."
    },
    {
        "id": "third_anglo_mysore_war",
        "name": "Third Anglo-Mysore War",
        "years": "1790–1792",
        "subtitle": "Cornwallis's Triple Alliance (Grand Campaign)",
        "description": "The standard campaign. Lord Cornwallis leads British, Maratha, and Hyderabad forces in a coordinated grand offensive against the Kingdom of Mysore."
    },
    {
        "id": "fourth_anglo_mysore_war",
        "name": "Fourth Anglo-Mysore War",
        "years": "1799",
        "subtitle": "The Fall of Seringapatam",
        "description": "General Harris converges on Seringapatam. Tipu Sultan must make a desperate defensive stand at the fortress walls against overwhelming allied armies."
    }
]


def setup_first_war(state: GameState) -> GameState:
    """First Anglo-Mysore War (1767–1769) setup."""
    # British: Bombay, Madras, Vellore
    state.set_node_fresh_army(NODE_TO_IDX["Bombay"])
    state.set_node_fresh_army(NODE_TO_IDX["Madras"])
    state.set_node_fresh_army(NODE_TO_IDX["Vellore"])

    # Mysore: Darwar, Bangalore, Seringapatam, Coimbatore, Dindigul, Erode, Satara
    state.set_node_fort(NODE_TO_IDX["Darwar"])
    state.set_node_fort(NODE_TO_IDX["Bangalore"])
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    state.set_node_fort(NODE_TO_IDX["Dindigul"])
    state.set_node_fort(NODE_TO_IDX["Erode"])
    state.set_node_fort(NODE_TO_IDX["Satara"])

    state.turn = 1
    state.to_move = 0
    return state


def setup_second_war(state: GameState) -> GameState:
    """Second Anglo-Mysore War (1780–1784) setup."""
    # British: Bombay, Madras, Masulipatam, Travancore
    state.set_node_fresh_army(NODE_TO_IDX["Bombay"])
    state.set_node_fresh_army(NODE_TO_IDX["Madras"])
    state.set_node_fresh_army(NODE_TO_IDX["Masulipatam"])
    state.set_node_fresh_army(NODE_TO_IDX["Travancore"])

    # Mysore: Darwar, Chitaldoorg, Mangalore, Bangalore, Seringapatam, Erode, Coimbatore, Mahé, Pondicherry, Dindigul
    state.set_node_fort(NODE_TO_IDX["Darwar"])
    state.set_node_fort(NODE_TO_IDX["Chitaldoorg"])
    state.set_node_fort(NODE_TO_IDX["Mangalore"])
    state.set_node_fort(NODE_TO_IDX["Bangalore"])
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Erode"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    state.set_node_fort(NODE_TO_IDX["Mahé"])
    state.set_node_fort(NODE_TO_IDX["Pondicherry"])
    state.set_node_fort(NODE_TO_IDX["Dindigul"])

    state.turn = 1
    state.to_move = 0
    return state


def setup_third_war(state: GameState) -> GameState:
    """Third Anglo-Mysore War (1790–1792) default setup."""
    state.default_setup()
    return state


def setup_fourth_war(state: GameState) -> GameState:
    """Fourth Anglo-Mysore War (1799) setup."""
    # British & Allied: Bombay, Hyderabad, Madras, Bangalore, Vellore, Travancore
    state.set_node_fresh_army(NODE_TO_IDX["Bombay"])
    state.set_node_fresh_army(NODE_TO_IDX["Hyderabad"])
    state.set_node_fresh_army(NODE_TO_IDX["Madras"])
    state.set_node_fresh_army(NODE_TO_IDX["Bangalore"])
    state.set_node_fresh_army(NODE_TO_IDX["Vellore"])
    state.set_node_fresh_army(NODE_TO_IDX["Travancore"])

    # Mysore: Heavily fortified central redoubts (Seringapatam, Coimbatore, Mangalore, Dindigul)
    state.set_node_fort(NODE_TO_IDX["Seringapatam"])
    state.set_node_fort(NODE_TO_IDX["Coimbatore"])
    state.set_node_fort(NODE_TO_IDX["Mangalore"])
    state.set_node_fort(NODE_TO_IDX["Dindigul"])

    state.turn = 1
    state.to_move = 0
    return state


SCENARIO_BUILDERS = {
    "first_anglo_mysore_war": setup_first_war,
    "second_anglo_mysore_war": setup_second_war,
    "third_anglo_mysore_war": setup_third_war,
    "fourth_anglo_mysore_war": setup_fourth_war,
}


def get_scenario(scenario_id: str) -> GameState:
    """Instantiates a clean GameState with the specified scenario starting positions."""
    builder = SCENARIO_BUILDERS.get(scenario_id, setup_third_war)
    state = GameState()
    # Empty all nodes
    for i in range(25):
        state.set_node_empty(i)
    return builder(state)


def list_scenarios() -> List[Dict[str, Any]]:
    """Returns scenario metadata list."""
    return SCENARIOS_METADATA
