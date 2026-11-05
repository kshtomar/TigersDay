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
    def __init__(self, state, parent = None):
        self.state = state
        self._parent = weakref.ref(parent) if parent is not None else None
        # joint indexed lists
        self._priors = None
        self._moves = None
        self._children = []
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.offset = 0
        self.is_expanded = False

    @property
    def eval(self):
        if self.parent() and self.visit_count == 0:
            # assume unplayed nodes are as good as parent
            return self.parent().eval
        return self.value_sum / self.visit_count
    
    def child(self, i):
        if self._children[i] is None:
            self._children[i] = Node(Updater.get_next_state(self.state, int(self._moves[i])), self)
        return self._children[i]
    
    def parent(self):
        # handle weakref nonsense
        return self._parent() if self._parent is not None else None
    
    # mask and normalize before this
    def expand_decision(self, moves, priors):
        self._moves = moves
        self._priors = priors
        self.is_expanded = len(moves) > 0
        # lazy child creation, start as None
        self._children = [None] * len(self._moves)

    def expand_luck(self, rng):
        luck_outcomes = Updater.get_luck_outcomes(self.state)
        prior = 1.0 / len(luck_outcomes)
        self._moves = np.arange(len(luck_outcomes), dtype=np.int_)
        self._priors = np.full(len(luck_outcomes), prior, dtype=np.float64)
        self._children = [Node(outcome, self) for outcome in luck_outcomes]
        self.is_expanded = True
        # offset for unbiased choice in round robin luck sampling
        self.offset = int(rng.random() * len(self._children))

class MCTS:
    def __init__(self, model, ipuct = 800, dalpha = 0.5, depsilon = 0.25):
        self.model = model
        self.ipuct = ipuct
        self.dalpha = dalpha
        self.depsilon = depsilon
        self.root = None
        self.legal_moves = Engine.get_legal_moves
        self.winner = Updater.get_state_winner
        self.rng = random
        self.np_rng = np.random

    def search(self, root_state, simulations, depsilon=None):

        # override depsilon for this search
        depsilon = self.depsilon if depsilon is None else depsilon
        assert 0 <= depsilon <= 1, depsilon
        
        # only call search on decision nodes

        if self.root is None:
            self.root = Node(root_state.copy())

        # lazy generate dirichlet root noise
        noise = None

        for sim in range(simulations):
            node = self.root

            while node.is_expanded:
                # luck is slippery
                if node.state.is_luck:
                    node = node._children[(node.offset + node.visit_count) % len(node._children)]
                else:
                    # generate noise on simulation 1
                    if node is self.root and noise is None and depsilon > 0:
                        noise = self.np_rng.dirichlet([self.dalpha] * len(self.root._moves))
                    node = self.select_child(node, noise if node is self.root else None, depsilon=depsilon)

            # handle more luck ply
            while node.state.is_luck:
                if not node.is_expanded:
                    node.expand_luck(self.rng)
                node = node._children[(node.offset + node.visit_count) % len(node._children)]
            
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
            node = node.parent()

    def select_child(self, node, noise=None, depsilon=None):
        weight = self.depsilon if depsilon is None else depsilon
        best_score, best = NEG_INF, -1
        # calculate once for every child
        puct = 1.25 + math.log((node.visit_count + self.ipuct) / self.ipuct)
        visits_sqrt = math.sqrt(node.visit_count)
        mysore = node.state.to_move == 1
        parent_eval = node.value_sum / node.visit_count
        for i in range(len(node._children)):
            child = node._children[i]
            visits = child.visit_count if child is not None else 0
            value = child.value_sum / visits if visits else parent_eval
            # flip evaluation for mysore
            exploitation = -value if mysore else value
            prior = node._priors[i]
            if noise is not None:
                prior = (1-weight) * prior + weight * float(noise[i])
            # blend dirichlet noise at select time
            exploration = puct * prior * (visits_sqrt / (1 + visits))
            score = exploitation + exploration
            if score > best_score:
                best_score, best = score, i
        return node.child(best)

    # retain subtree if mcts already has it
    def update_root(self, action, luck_trajectory):
        if self.root is None:
            return

        slot = np.flatnonzero(np.asarray(self.root._moves) == action)
        new_root = self.root._children[slot[0]] if len(slot) else None
        # luck outcomes are numbered by slot, and luck children are created eagerly
        for idx in luck_trajectory:
            if new_root is None:
                break
            new_root = new_root._children[idx]

        self.root = new_root
        if new_root is not None:
            new_root._parent = None

    def find_move(self, state, simulations, temperature=0.0, *, depsilon=None):
        root = self.search(state, simulations, depsilon=depsilon)
        counts = np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)
        for i, child in enumerate(root._children):
            if child is not None:
                counts[root._moves[i]] = child.visit_count

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
