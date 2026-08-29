import numpy as np
import random
from game.constants import *
import game.updater as Updater
import game.engine as Engine

class Node:
    def __init__(self, state, parent = None, move = None, prior = 0.0):
        self.state = state
        self.parent = parent
        # move to get here
        self.move = move
        self.children = {}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def eval(self):
        if self.parent and self.visit_count == 0:
            return self.parent.eval
            # assume unplayed nodes are as good as parent
        return self.value_sum / self.visit_count
    
    @property
    def is_expanded(self):
        return len(self.children) > 0
     
    @property
    def is_luck(self):
        return self.state.is_luck
    
    # mask and normalize before this
    def expand_decision(self, action_priors):
        for move, prior in enumerate(action_priors):
            if prior > 0.0 and move not in self.children:
                self.children[move] = Node(None, self, move, prior)
                # lazy evaluation, leave game state unexplored

    def expand_luck(self):
        luck_outcomes = Updater.get_luck_outcomes(self.state)
        prior = 1.0 / len(luck_outcomes)

        for i, outcome in enumerate(luck_outcomes):
            if i not in self.children:
                self.children[i] = Node(outcome, self, i, prior)

class MCTS:
    def __init__(self, model, simulations = 100, ipuct = 800, dalpha = 0.5, depsilon = 0.25):
        self.model = model
        self.simulations = simulations
        self.ipuct = ipuct
        self.dalpha = dalpha
        self.depsilon = depsilon
        self.root = None

    def search(self, root_state):
        # only call search on decision nodes

        if self.root is None:
            self.root = Node(root_state.copy())

        # lazy generate dirichlet root noise
        noise_dict = None

        for sim in range(self.simulations):
            node = self.root

            while node.is_expanded:
                # luck is slippery
                if node.is_luck:
                    node = random.choice(list(node.children.values()))
                else:
                    # generate noise on simulation 1
                    if node is self.root and noise_dict is None:
                        legal_moves = list(self.root.children.keys())
                        noise = np.random.dirichlet([self.dalpha] * len(legal_moves))
                        noise_dict = {move: n for move, n in zip(legal_moves, noise)}
                    node = self.select_child(node, noise_dict if node is self.root else None)

            # lazy evaluation, actually do it
            if node.state is None:
                assert node.parent is not None
                node.state = Updater.get_next_state(node.parent.state, node.move)

            # handle more luck ply
            while node.is_luck:
                if not node.is_expanded:
                    node.expand_luck()  
                node = random.choice(list(node.children.values()))
            
            reward = Updater.get_state_winner(node.state)
            if reward != 0:
                self.backpropagate(node, reward)
                # skip expansion if this is a win
                continue

            value, raw_logits = self.model.predict(node.state)
            legal_mask = Engine.get_legal_moves(node.state)
            masked_logits = np.where(legal_mask == 1, raw_logits, -np.inf)
            max_logit = np.max(masked_logits)
            exp_logits = np.exp(masked_logits - max_logit)
            policy = exp_logits / np.sum(exp_logits)
            node.expand_decision(policy)
            self.backpropagate(node, value)
        
        return self.root

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def select_child(self, node, noise_dict=None):
        best_score, best_child = -np.inf, None
        for move, child in node.children.items():
            exploitation = -child.eval if node.state.to_move == 1 else child.eval
            # flip evaluation for mysore turn

            prior = child.prior
            if noise_dict is not None and move in noise_dict:
                prior = (1-self.depsilon) * prior + self.depsilon * noise_dict[move]
            # blend dirichlet noise at select time

            puct = 1.25 + np.log((node.visit_count + self.ipuct) / self.ipuct)
            # dynamic puct with base 1.25 and ipuct specified
            exploration = puct * prior * (np.sqrt(node.visit_count) / (1 + child.visit_count))
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
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

    def find_move(self, state, temperature = 0.0):
        root = self.search(state)
        counts = np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)
        for m, child in root.children.items():
            counts[m] = child.visit_count

        if temperature == 0.0 or counts.sum() == 0:
            move = int(np.argmax(counts))
            policy = np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)
            policy[move] = 1.0
            return move, policy

        counts **= 1.0 / temperature
        policy = counts / counts.sum()
        move = int(np.random.choice(MOVE_VECTOR_LENGTH, p=policy))
        return move, policy