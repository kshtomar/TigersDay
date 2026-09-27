import numpy as np
from game.constants import *

class GameState:

    IDX_BRITISH_CARDS_OFFSET = 0   #index where this information begins
    IDX_MYSORE_CARDS_OFFSET = CARDS    # 6 cards for both mysore and british
    IDX_NODES_OFFSET = IDX_MYSORE_CARDS_OFFSET + CARDS   # 23 nodes * 3D vectors = 69
    IDX_TURN_OFFSET = IDX_NODES_OFFSET + 3*NODES     # 4 turns (One-hot)
    IDX_WHO_TO_MOVE_OFFSET = IDX_TURN_OFFSET + 4    # 3 options (One-hot)
    IDX_COMBAT_STRENGTH_OFFSET = IDX_WHO_TO_MOVE_OFFSET + 3 # 4 options (One-hot): Only ever stored for Mysore
    IDX_ATTACKER_OFFSET = IDX_COMBAT_STRENGTH_OFFSET + 4    # 23x2 Attacker/Defender (One-hot)
    IDX_DEFENDER_OFFSET = IDX_ATTACKER_OFFSET + NODES #Index from which Defender values start showing up

    IDX_BRITISH_CARDS = slice(IDX_BRITISH_CARDS_OFFSET, IDX_BRITISH_CARDS_OFFSET + CARDS)
    IDX_MYSORE_CARDS = slice(IDX_MYSORE_CARDS_OFFSET, IDX_MYSORE_CARDS_OFFSET + CARDS)
    IDX_TURN = slice(IDX_TURN_OFFSET, IDX_TURN_OFFSET + 4)
    IDX_WHO_TO_MOVE = slice(IDX_WHO_TO_MOVE_OFFSET, IDX_WHO_TO_MOVE_OFFSET + 3)
    IDX_COMBAT_STRENGTH = slice(IDX_COMBAT_STRENGTH_OFFSET, IDX_COMBAT_STRENGTH_OFFSET + 4)
    IDX_ATTACKER = slice(IDX_ATTACKER_OFFSET, IDX_ATTACKER_OFFSET + NODES)
    IDX_DEFENDER = slice(IDX_DEFENDER_OFFSET, IDX_DEFENDER_OFFSET + NODES)

    def __init__(self):
        self.vector = np.zeros(GAME_VECTOR_LENGTH, dtype=bool)
        # store as integer behind the scenes, one hot for the AI, -1 is blank
        self._attacker = NO_UNIT
        self._defender = NO_UNIT
        self._card_strength = 0
        self._to_move = 0
        self._turn = 1
        self.bluck = 0
        self.mluck = 0
        self.mysore_cards[:] = True
        self.british_cards[:] = True

    def default_setup(self):
        self.set_node_fresh_army(NODE_TO_IDX["Bombay"])
        self.set_node_fresh_army(NODE_TO_IDX["Hyderabad"])
        self.set_node_fresh_army(NODE_TO_IDX["Madras"])
        self.set_node_fresh_army(NODE_TO_IDX["Travancore"])

        self.set_node_fort(NODE_TO_IDX["Darwar"])
        self.set_node_fort(NODE_TO_IDX["Chitaldoorg"])
        self.set_node_fort(NODE_TO_IDX["Mangalore"])
        self.set_node_fort(NODE_TO_IDX["Bangalore"])
        self.set_node_fort(NODE_TO_IDX["Seringapatam"])
        self.set_node_fort(NODE_TO_IDX["Erode"])
        self.set_node_fort(NODE_TO_IDX["Coimbatore"])
        self.set_node_fort(NODE_TO_IDX["Mahé"])
        self.set_node_fort(NODE_TO_IDX["Dindigul"])

        self.turn = 1
        self.to_move = 0

    def copy(self):
        """Crucial for MCTS: Creates a fast, deep copy of the state."""
        new_state = GameState()
        new_state.vector = np.copy(self.vector)
        new_state._attacker = self._attacker
        new_state._defender = self._defender
        new_state._card_strength = self._card_strength
        new_state._to_move = self._to_move
        new_state._turn = self._turn
        new_state.bluck = self.bluck
        new_state.mluck = self.mluck
        return new_state

    def set_node_fresh_army(self, node):
        self.set_node_empty(node)
        self.vector[self.IDX_NODES_OFFSET + 3 * node] = True

    def set_node_tired_army(self, node):
        self.set_node_empty(node)
        self.vector[self.IDX_NODES_OFFSET + 3 * node + 1] = True

    def set_node_fort(self, node):
        self.set_node_empty(node)
        self.vector[self.IDX_NODES_OFFSET + 3 * node + 2] = True

    def set_node_empty(self, node):
        start_idx = self.IDX_NODES_OFFSET + 3 * node
        self.vector[start_idx : start_idx + 3] = False

    def clear_battle(self):
        self.attacker = NO_UNIT
        self.defender = NO_UNIT
        self.card_strength = 0

    @property
    def fresh_armies(self):
        return self.vector[self.IDX_NODES_OFFSET : self.IDX_TURN_OFFSET : 3]

    @property
    def tired_armies(self):
        return self.vector[self.IDX_NODES_OFFSET + 1 : self.IDX_TURN_OFFSET : 3]

    @property
    def forts(self):
        return self.vector[self.IDX_NODES_OFFSET + 2 : self.IDX_TURN_OFFSET : 3]

    @property
    def empty(self):
        return ~(self.fresh_armies | self.tired_armies | self.forts)
    
    @property
    def mysore_cards(self):
        return self.vector[self.IDX_MYSORE_CARDS]

    @property
    def british_cards(self):
        return self.vector[self.IDX_BRITISH_CARDS]

    @property
    def to_move(self):
        return self._to_move
    
    @to_move.setter
    def to_move(self, to_move_idx):
        self._to_move = to_move_idx % 3
        self.vector[self.IDX_WHO_TO_MOVE] = False
        self.vector[self.IDX_WHO_TO_MOVE_OFFSET + (to_move_idx % 3)] = True

    @property
    def turn(self):
        return self._turn
    
    @turn.setter
    def turn(self, turn_number):
        self._turn = turn_number
        self.vector[self.IDX_TURN] = False
        self.vector[self.IDX_TURN_OFFSET + (turn_number - 1)] = True

    @property
    def attacker(self):
        return self._attacker
    
    @attacker.setter
    def attacker(self, value: int):
        self._attacker = value
        self.vector[self.IDX_ATTACKER] = False
        if value != NO_UNIT:
            self.vector[self.IDX_ATTACKER_OFFSET + value] = True

    @property
    def defender(self):
        return self._defender
    
    @defender.setter
    def defender(self, value: int):
        self._defender = value
        self.vector[self.IDX_DEFENDER] = False
        if value != NO_UNIT:
            self.vector[self.IDX_DEFENDER_OFFSET + value] = True
    
    @property
    def card_strength(self):
        return self._card_strength
    
    @card_strength.setter
    def card_strength(self, value: int):
        self._card_strength = value
        self.vector[self.IDX_COMBAT_STRENGTH] = False
        self.vector[self.IDX_COMBAT_STRENGTH_OFFSET + value] = True

    @property
    def is_battle(self):
        return self.attacker != NO_UNIT
    
    @property
    def is_luck(self):
        return self.bluck or self.mluck
    
    def turn_refresh(self):
        self.turn += 1
        self.fresh_armies[:] = self.tired_armies
        self.tired_armies[:] = False
        self.mysore_cards[:] = True
        self.british_cards[:] = True
        self.bluck = 0
        self.mluck = 0
    
    def __str__(self):
        save = ""
        for i in range(GAME_VECTOR_LENGTH):
            if self.vector[i]:
                save += "1"
            else:
                save += "0"
        return save

    def read_str(self, bit_str):
        """ This function is utilized by the frontend and requires certain checks """

        if len(bit_str) != GAME_VECTOR_LENGTH:
            raise ValueError(f"Invalid input length! Expected {GAME_VECTOR_LENGTH} bits. ")
        new_state = self.copy() # creates a copy of itself just in case there is an error
        try:
            new_state.vector = np.array([bool(int(b)) for b in bit_str], dtype=bool)
        except ValueError:
            raise ValueError("Invalid input! Please provide a string consisting purely of 1s and 0s.")
        try:
            for i in range(12, 136):
                if all(new_state.vector[i : i+3]):
                    t_idx = (i - 12) // 3
                    name = INDEX_MAP[t_idx] if t_idx in INDEX_MAP else f"Bit {i}"
                    raise ValueError(f"Invalid Binary: Triple consecutive 1s detected starting at {name}")
        except ValueError:
            raise ValueError(f"Invalid Binary: Triple consecutive 1s detected starting at {name}")

        new_state._attacker = int(np.argmax(new_state.vector[new_state.IDX_ATTACKER])) if new_state.vector[new_state.IDX_ATTACKER].any() else NO_UNIT
        new_state._defender = int(np.argmax(new_state.vector[new_state.IDX_DEFENDER])) if new_state.vector[new_state.IDX_DEFENDER].any() else NO_UNIT
        new_state._card_strength = int(np.argmax(new_state.vector[new_state.IDX_COMBAT_STRENGTH]))
        new_state._to_move = int(np.argmax(new_state.vector[new_state.IDX_WHO_TO_MOVE]))
        new_state._turn = int(np.argmax(new_state.vector[new_state.IDX_TURN])) + 1

        return new_state

def main():
    default = GameState()
    default.default_setup()
    print(default)

    print(f"Coastal Length: {len(COASTAL_INDICES)}")
    print(f"Move Vector Length: {MOVE_VECTOR_LENGTH}")
    print(f"Game Vector Length: {GAME_VECTOR_LENGTH}")
    print(f"Edges Count: {EDGES//2}")

if __name__ == "__main__":
    main()
    