import math
import weakref
import numpy as np
import random
from game.constants import *
import game.updater as Updater
import game.engine as Engine

# declare constant for typing in C
NEG_INF = -np.inf

def mask_normalize(logits, mask):
    moves = np.flatnonzero(mask)
    policy = logits[moves]
    policy = np.exp(policy - policy.max())
    policy = policy / policy.sum()
    # filter nonzero moves in numpy for efficiency
    positive = policy > 0
    return moves[positive], policy[positive].astype(np.float64)


class Node:
    def __init__(self, state, parent = None, move = None, prior = 0.0):
        self.state = state
        self._parent = weakref.ref(parent) if parent is not None else None
        # move to get here
        self.move = move
        self._children = None
        self._priors = None
        self._moves = None
        self._nodes = []
        
        self.visit_count = 0
        self.value_sum = 0.0
        # stick to Python scalars for speed
        self.prior = float(prior)

    @property
    def eval(self):
        if self.parent and self.visit_count == 0:
            return self.parent.eval
            # assume unplayed nodes are as good as parent
        return self.value_sum / self.visit_count
    
    def expanded(self):
        return self._moves is not None and len(self._moves) > 0

    @property
    def children(self):
        if self._children is None:
            # Expose the tree in action order; search creates only selected nodes.
            self._children = {}
            for i in range(len(self._nodes)):
                move = int(self._moves[i])
                if self._nodes[i] is None:
                    self._nodes[i] = Node(None, self, move, self._priors[i])
                self._children[move] = self._nodes[i]
        return self._children
     
    @property
    def is_luck(self):
        return self.state.is_luck

    @property
    def parent(self):
        """Cleanly dereferences the weakref. Returns the Node or None."""
        return self._parent() if self._parent is not None else None

    @parent.setter
    def parent(self, value):
        """Allows assigning `node.parent = None` or `node.parent = new_node` directly."""
        if value is None:
            self._parent = None
        else:
            self._parent = weakref.ref(value)
    
    # mask and normalize before this
    def expand_decision(self, moves, priors):
        self._moves = moves
        self._priors = priors
        # lazy child creation, start as None
        self._nodes = [None] * len(self._moves)
        self._children = None
        # lazy evaluation, leave game state unexplored

    def expand_luck(self):
        luck_outcomes = Updater.get_luck_outcomes(self.state)
        prior = 1.0 / len(luck_outcomes)
        self._moves = np.arange(len(luck_outcomes), dtype=np.int_)
        self._priors = np.full(len(luck_outcomes), prior, dtype=np.float64)
        self._nodes = [Node(outcome, self, i, prior) for i, outcome in enumerate(luck_outcomes)]
        self._children = dict(enumerate(self._nodes))

class MCTS:
    def __init__(self, model, ipuct = 800, dalpha = 0.5, depsilon = 0.25):
        self.model = model
        self.ipuct = ipuct
        self.dalpha = dalpha
        self.depsilon = depsilon
        self.root = None
        self.noise = None
        self.legal_moves = Engine.get_legal_moves
        self.winner = Updater.get_state_winner
        self.rng = random
        self.np_rng = np.random

    def search(self, root_state, simulations, depsilon=None):

        # override depsilon for this search
        depsilon = self.depsilon if depsilon is None else depsilon
        if not 0 <= depsilon <= 1:
            raise ValueError("depsilon must be between 0 and 1")
        
        # only call search on decision nodes

        if self.root is None:
            self.root = Node(root_state.copy())

        # lazy generate dirichlet root noise
        self.noise = None

        for sim in range(simulations):
            node = self.root

            while node.expanded():
                # luck is slippery
                if node.state.is_luck:
                    node = self.rng.choice(list(node.children.values()))
                else:
                    # generate noise on simulation 1
                    if node is self.root and self.noise is None and depsilon > 0:
                        self.noise = self.np_rng.dirichlet([self.dalpha] * len(self.root._moves))
                    node = self.select_child(node, self.noise if node is self.root else None, depsilon=depsilon)

            # lazy evaluation, actually do it
            if node.state is None:
                node.state = Updater.get_next_state(node.parent.state, node.move)

            # handle more luck ply
            while node.state.is_luck:
                if not node.expanded():
                    node.expand_luck()  
                node = self.rng.choice(list(node.children.values()))
            
            reward = self.winner(node.state)
            if reward != 0:
                self.backpropagate(node, reward)
                # skip expansion if this is a win
                continue

            value, raw_logits = self.model.predict(node.state)
            legal_mask = self.legal_moves(node.state)
            moves, priors = mask_normalize(raw_logits, legal_mask)
            node.expand_decision(moves, priors)
            self.backpropagate(node, value)
        
        return self.root

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node._parent() if node._parent is not None else None

    def select_child(self, node, noise=None, depsilon=None):
        weight = self.depsilon if depsilon is None else depsilon
        best_score, best_child = NEG_INF, None
        # calculate once for every child
        puct = 1.25 + math.log((node.visit_count + self.ipuct) / self.ipuct)
        visits_sqrt = math.sqrt(node.visit_count)
        mysore = node.state.to_move == 1
        parent_eval = node.value_sum / node.visit_count
        children = node._nodes
        best_index = -1
        for i in range(len(children)):
            base_prior = float(node._priors[i])
            child = children[i]
            visits = child.visit_count if child is not None else 0
            value = child.value_sum / visits if visits else parent_eval
            # flip evaluation for mysore
            exploitation = -value if mysore else value
            prior = child.prior if child is not None else base_prior
            if noise is not None:
                prior = (1-weight) * prior + weight * float(noise[i])
            # blend dirichlet noise at select time
            exploration = puct * prior * (visits_sqrt / (1 + visits))
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
                best_index = i
        if best_child is None and best_index >= 0:
            best_child = children[best_index] = Node(
                None, node, int(node._moves[best_index]), node._priors[best_index],
            )
        return best_child

    # retain subtree if mcts already has it
    def update_root(self, action, luck_trajectory):
        if self.root is None or action not in self.root.children:
            self.root = None
            return
        
        current_node = self.root.children[action]

        if current_node.state is None:
            # drop subtree if unexpanded
            self.root = None
            return

        for idx in luck_trajectory:
            if idx in current_node.children:
                current_node = current_node.children[idx]
            else:
                self.root = None
                return
            
        self.root = current_node
        self.root.parent = None

    def find_move(self, state, simulations, temperature=0.0, *, depsilon=None):
        root = self.search(state, simulations, depsilon=depsilon)
        counts = np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)
        for move, child in root.children.items():
            counts[move] = child.visit_count

        # fallback if no visits
        if counts.sum() == 0:
            move = int(np.argmax(counts))
            policy = np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)
            policy[move] = 1.0
            return move, policy

        # always train on full policy distribution
        policy = counts / counts.sum()

        if temperature == 0.0:
            return int(np.argmax(counts)), policy

        # temperature affects move choice
        counts **= 1.0 / temperature
        tpolicy = counts / counts.sum()
        move = int(self.np_rng.choice(MOVE_VECTOR_LENGTH, p=tpolicy))
        return move, policy
