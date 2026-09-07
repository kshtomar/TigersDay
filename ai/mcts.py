import os
import weakref
import numpy as np
import random
from game.constants import *
import game.updater as Updater
import game.engine as Engine

class Node:
    def __init__(self, state, parent = None, move = None, prior = 0.0):
        self.state = state
        self._parent = weakref.ref(parent) if parent is not None else None
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
    def __init__(self, model, simulations = DEFAULT_SIMS, ipuct = 800, dalpha = 0.5, depsilon = 0.25, opening_book = None):
        self.model = model
        self.simulations = simulations
        self.ipuct = ipuct
        self.dalpha = dalpha
        self.depsilon = depsilon
        self.root = None
        self.transposition_table = {}
        
        if opening_book is not None:
            self.opening_book = opening_book
        else:
            self.opening_book = None
            for book_path in ["public/opening_book.json", "../public/opening_book.json"]:
                if os.path.exists(book_path):
                    try:
                        import json
                        with open(book_path, "r", encoding="utf-8") as f:
                            self.opening_book = json.load(f)
                        break
                    except Exception:
                        pass

    def search(self, root_state, simulations = None, stop = False):
        # only call search on decision nodes
        sims = self.simulations if simulations is None else simulations

        if self.root is None:
            self.root = Node(root_state.copy())

        # lazy generate dirichlet root noise
        noise_dict = None
        warmup = sims // 5 if stop else sims
        stop_threshold = 0.9

        for sim in range(sims):
            # Early stopping if a single move dominates
            if stop and sim > warmup and len(self.root.children) > 0:
                max_visits = max(child.visit_count for child in self.root.children.values())
                if self.root.visit_count > 0 and (max_visits / self.root.visit_count) > stop_threshold:
                    return self.root

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

            # Check Transposition Table (Zobrist Hash)
            z_hash = node.state.zobrist_hash()
            if z_hash in self.transposition_table:
                value, policy = self.transposition_table[z_hash]
            else:
                value, raw_logits = self.model.predict(node.state)
                legal_mask = Engine.get_legal_moves(node.state)
                masked_logits = np.where(legal_mask == 1, raw_logits, -np.inf)
                max_logit = np.max(masked_logits)
                exp_logits = np.exp(masked_logits - max_logit)
                policy = exp_logits / np.sum(exp_logits)
                self.transposition_table[z_hash] = (value, policy)

            node.expand_decision(policy)
            self.backpropagate(node, value)
        
        return self.root

    def search_batch(self, root_state, simulations = None, batch_size = 8, virtual_loss = 3.0, stop = False):
        """
        Batched leaf evaluation with Virtual Loss.
        Simultaneously selects up to batch_size trajectories, evaluates leaves together
        via model.predict_batch(), and backpropagates while reverting virtual loss.
        """
        sims = self.simulations if simulations is None else simulations
        if self.root is None:
            self.root = Node(root_state.copy())

        if not self.root.is_expanded:
            # Expand root first
            z_hash = self.root.state.zobrist_hash()
            if z_hash in self.transposition_table:
                val, pol = self.transposition_table[z_hash]
            else:
                val, raw_p = self.model.predict(self.root.state)
                mask = Engine.get_legal_moves(self.root.state)
                masked = np.where(mask == 1, raw_p, -np.inf)
                max_l = np.max(masked)
                exp_l = np.exp(masked - max_l)
                pol = exp_l / np.sum(exp_l)
                self.transposition_table[z_hash] = (val, pol)
            self.root.expand_decision(pol)
            self.root.visit_count += 1
            self.root.value_sum += val

        sim_count = 0
        warmup = sims // 5 if stop else sims
        stop_threshold = 0.9

        while sim_count < sims:
            current_batch = min(batch_size, sims - sim_count)
            paths = []
            leaves_to_eval = []
            leaves_nodes = []

            for _ in range(current_batch):
                node = self.root
                path = [node]

                while node.is_expanded:
                    if node.is_luck:
                        node = random.choice(list(node.children.values()))
                    else:
                        node = self.select_child(node)
                    path.append(node)

                # Apply virtual loss to path
                for p_node in path:
                    p_node.visit_count += virtual_loss
                    p_node.value_sum -= virtual_loss

                if node.state is None:
                    assert node.parent is not None
                    node.state = Updater.get_next_state(node.parent.state, node.move)

                while node.is_luck:
                    if not node.is_expanded:
                        node.expand_luck()
                    node = random.choice(list(node.children.values()))
                    path.append(node)

                paths.append((path, node))

            # Evaluate leaves
            for path, leaf in paths:
                reward = Updater.get_state_winner(leaf.state)
                if reward != 0:
                    self._backprop_revert_vl(path, reward, virtual_loss)
                    continue

                z_hash = leaf.state.zobrist_hash()
                if z_hash in self.transposition_table:
                    val, pol = self.transposition_table[z_hash]
                    leaf.expand_decision(pol)
                    self._backprop_revert_vl(path, val, virtual_loss)
                else:
                    leaves_to_eval.append(leaf.state)
                    leaves_nodes.append((path, leaf, z_hash))

            if leaves_to_eval:
                if hasattr(self.model, "predict_batch"):
                    vals, policies = self.model.predict_batch(leaves_to_eval)
                else:
                    vals = []
                    policies = []
                    for st in leaves_to_eval:
                        v, p = self.model.predict(st)
                        vals.append(v)
                        policies.append(p)
                    vals = np.array(vals)
                    policies = np.array(policies)

                for i, (path, leaf, z_hash) in enumerate(leaves_nodes):
                    v = float(vals[i])
                    raw_p = policies[i]
                    mask = Engine.get_legal_moves(leaf.state)
                    masked = np.where(mask == 1, raw_p, -np.inf)
                    max_l = np.max(masked)
                    exp_l = np.exp(masked - max_l)
                    pol = exp_l / np.sum(exp_l)

                    self.transposition_table[z_hash] = (v, pol)
                    leaf.expand_decision(pol)
                    self._backprop_revert_vl(path, v, virtual_loss)

            sim_count += current_batch

            if stop and sim_count > warmup and len(self.root.children) > 0:
                max_visits = max(child.visit_count for child in self.root.children.values())
                if self.root.visit_count > 0 and (max_visits / self.root.visit_count) > stop_threshold:
                    break

        return self.root

    def _backprop_revert_vl(self, path, true_value, virtual_loss):
        """Reverts virtual loss and accumulates true evaluation."""
        for node in path:
            node.visit_count = (node.visit_count - virtual_loss) + 1
            node.value_sum = (node.value_sum + virtual_loss) + true_value

    def search_time_budget(self, root_state, time_budget_ms = 1500, min_sims = 50, max_sims = 3000, batch_size = 8, stop = True):
        """
        Time-budgeted MCTS search with dynamic time allocation.
        Allocates up to +15% time during contested Turn 2/3 fortress battles,
        and terminates in <50ms if only 1 move is legal or move dominates.
        """
        import time
        start_time = time.perf_counter()

        # Dynamic allocation: Check contested fortress battles on Turn 2 or 3
        effective_budget = time_budget_ms
        if root_state.turn in [2, 3]:
            # Contested fortress check
            contested = any(
                root_state.forts[i] and any(root_state.fresh_armies[j] or root_state.tired_armies[j] for j in np.where(ADJACENCY_MATRIX[i])[0])
                for i in range(NODES)
            )
            if contested:
                effective_budget = int(time_budget_ms * 1.15)

        # Pre-check legal moves: if only 1 legal move, exit immediately
        legal_count = int(np.sum(Engine.get_legal_moves(root_state)))
        if legal_count <= 1:
            return self.search(root_state, simulations=min_sims, stop=False)

        sims_done = 0
        while sims_done < max_sims:
            chunk = min(batch_size, max_sims - sims_done)
            self.search_batch(root_state, simulations=chunk, batch_size=chunk, stop=stop)
            sims_done += chunk

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if sims_done >= min_sims:
                # Check dominant move
                if len(self.root.children) > 0:
                    max_v = max(c.visit_count for c in self.root.children.values())
                    if self.root.visit_count > 0 and (max_v / self.root.visit_count) > 0.92:
                        break

            if elapsed_ms >= effective_budget:
                break

        return self.root

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def select_child(self, node, noise_dict=None):
        best_score, best_child = -np.inf, None
        visit_cnt = max(1.0, float(node.visit_count))
        for move, child in node.children.items():
            exploitation = -child.eval if node.state.to_move == 1 else child.eval
            # flip evaluation for mysore turn

            prior = child.prior
            if noise_dict is not None and move in noise_dict:
                prior = (1-self.depsilon) * prior + self.depsilon * noise_dict[move]
            # blend dirichlet noise at select time

            puct = 1.25 + np.log((visit_cnt + self.ipuct) / self.ipuct)
            child_visits = max(0.0, float(child.visit_count))
            exploration = puct * prior * (np.sqrt(visit_cnt) / (1.0 + child_visits))
            score = exploitation + exploration
            if best_child is None or score > best_score:
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

    def find_move(self, state, simulations = None, temperature = 0.0, stop = True, use_book = True, time_budget_ms = None, batch_size = 8):
        if use_book and self.opening_book:
            key = str(state)
            if key in self.opening_book:
                book_move = self.opening_book[key].get("move")
                legal_mask = Engine.get_legal_moves(state)
                if book_move is not None and legal_mask[book_move]:
                    policy = np.zeros(MOVE_VECTOR_LENGTH, dtype=np.float32)
                    policy[book_move] = 1.0
                    return int(book_move), policy

        if time_budget_ms is not None:
            root = self.search_time_budget(state, time_budget_ms=time_budget_ms, batch_size=batch_size, stop=stop)
        elif batch_size and batch_size > 1:
            root = self.search_batch(state, simulations=simulations, batch_size=batch_size, stop=stop)
        else:
            root = self.search(state, simulations, stop = stop)
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