"""
Monte Carlo Tree Search implementation for AlphaZero-like Xiangqi AI

This module implements the MCTS algorithm used by AlphaZero to select the best moves
during both training (self-play) and evaluation.
"""

import math
import numpy as np
from copy import deepcopy

from .config import *
from .utils import get_legal_actions_mask, action_to_move


class Node:
    """Node in the MCTS tree"""

    def __init__(self, prior=0.0, parent=None):
        self.parent = parent
        self.children = {}  # Maps action -> Node

        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

        # Cache for legal actions
        self.is_expanded = False
        self.legal_actions = None

    def expanded(self):
        """Check if the node has been expanded"""
        return self.is_expanded

    def value(self):
        """Get the mean value of the node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        """Select a child node according to the UCB formula"""
        # UCB score = Q + U where
        # Q = node_value
        # U = prior * sqrt(parent_visits) / (1 + child_visits)

        best_score = -float("inf")
        best_action = -1
        best_child = None

        # Explore-exploit trade-off with UCB
        for action, child in self.children.items():
            # Q value (exploitation)
            q_value = (
                -child.value()
            )  # Negative because we view from opponent's perspective

            # U value (exploration)
            u_value = (
                c_puct
                * child.prior
                * math.sqrt(self.visit_count)
                / (1 + child.visit_count)
            )

            # UCB score
            ucb_score = q_value + u_value

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, actions, priors):
        """Expand the node with given actions and priors"""
        self.is_expanded = True

        # Normalize priors over legal actions
        priors_sum = sum(priors[a] for a in actions)

        # Create children for each legal action
        for action in actions:
            # Get normalized prior for this action
            prior = (
                priors[action] / priors_sum if priors_sum > 0 else 1.0 / len(actions)
            )
            self.children[action] = Node(prior=prior, parent=self)

        self.legal_actions = actions

    def update(self, value):
        """Update node statistics with the observed value"""
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """Monte Carlo Tree Search for AlphaZero"""

    def __init__(
        self, model, game_adapter, num_simulations=NUM_SIMULATIONS, c_puct=C_PUCT
    ):
        """
        Initialize MCTS

        Args:
            model: Neural network model that provides policy and value predictions
            game_adapter: Adapter that interfaces with the Xiangqi game
            num_simulations: Number of simulations to run per search
            c_puct: Exploration constant in UCB formula
        """
        self.model = model
        self.game_adapter = game_adapter
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None

    def search(self, board, current_player, dirichlet_noise=False):
        """
        Perform MCTS search from the current board state

        Args:
            board: Current game board
            current_player: Current player ('red' or 'black')
            dirichlet_noise: Whether to add Dirichlet noise to the root node

        Returns:
            Action probabilities based on visit counts
        """
        # Set up root node
        self.root = Node()

        # Get state representation for neural network
        state = self.game_adapter.board_to_planes(board, current_player)

        # Get policy and value prediction from neural network
        policy, _ = self.model.predict(state)

        # Add Dirichlet noise to root policy for exploration (only during training)
        if dirichlet_noise:
            legal_actions = self.game_adapter.get_legal_actions(board, current_player)
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal_actions))

            for i, action in enumerate(legal_actions):
                policy[action] = (1 - DIRICHLET_NOISE_FACTOR) * policy[
                    action
                ] + DIRICHLET_NOISE_FACTOR * noise[i]

        # Get legal actions
        legal_actions = self.game_adapter.get_legal_actions(board, current_player)
        self.root.expand(legal_actions, policy)

        # Run simulations
        for i in range(self.num_simulations):
            self._simulate(board, current_player)

        # Calculate action probabilities based on visit counts
        visit_counts = np.zeros(ACTION_SIZE)
        for action, child in self.root.children.items():
            visit_counts[action] = child.visit_count

        # Normalize visit counts to get probabilities
        if sum(visit_counts) > 0:
            action_probs = visit_counts / sum(visit_counts)
        else:
            # Fallback to uniform distribution if no visits
            legal_actions_mask = get_legal_actions_mask(board, current_player)
            action_probs = legal_actions_mask / sum(legal_actions_mask)

        return action_probs

    def _simulate(self, board, current_player):
        """Run a single MCTS simulation"""
        node = self.root
        search_path = [node]

        # Make a copy of the game state
        sim_board = deepcopy(board)
        sim_player = current_player

        # Select phase - traverse tree until we reach a leaf node
        action = -1
        while node.expanded() and node.children:
            action, node = node.select_child(self.c_puct)
            search_path.append(node)

            # Apply action to the simulation board
            from_pos, to_pos = action_to_move(action)
            sim_board, sim_player = self.game_adapter.make_move(
                sim_board, sim_player, from_pos, to_pos
            )

            # Check if game is over
            game_over, winner = self.game_adapter.is_game_over(sim_board)
            if game_over:
                # Propagate game result
                value = 0
                if winner is not None:
                    value = 1 if winner == current_player else -1

                # Backpropagate
                for node in reversed(search_path):
                    node.update(-value)  # Flip value when going up the tree
                return

        # Expand phase - expand the leaf node if the game is not over
        state = self.game_adapter.board_to_planes(sim_board, sim_player)
        policy, value = self.model.predict(state)

        # Check if game is over
        game_over, winner = self.game_adapter.is_game_over(sim_board)
        if not game_over:
            # Get legal actions for expansion
            legal_actions = self.game_adapter.get_legal_actions(sim_board, sim_player)
            node.expand(legal_actions, policy)
        else:
            # Game is over, calculate value
            value = 0
            if winner is not None:
                value = 1 if winner == current_player else -1

        # Backpropagate - update statistics for all nodes in the search path
        for node in reversed(search_path):
            # Value is from the perspective of the player who just moved
            # When we go up the tree, we flip the perspective
            node.update(-value)
            value = -value

    def select_action(self, board, current_player, temperature=TEMPERATURE):
        """
        Select an action based on MCTS search results

        Args:
            board: Current game board
            current_player: Current player ('red' or 'black')
            temperature: Temperature for action selection

        Returns:
            Selected action
        """
        # Run MCTS search
        action_probs = self.search(board, current_player)

        # Apply temperature to the action probabilities
        if temperature == 0:
            # Deterministic selection - choose the action with highest probability
            action = np.argmax(action_probs)
        else:
            # Sample action based on the visit count distribution
            action_probs = action_probs ** (1 / temperature)
            action_probs = action_probs / sum(action_probs)
            action = np.random.choice(ACTION_SIZE, p=action_probs)

        return action

    def update_with_move(self, action):
        """
        Update the tree with a new move, reusing the subtree if possible

        Args:
            action: Action taken
        """
        # If action is in the children of the root, reuse that subtree
        if self.root is not None and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # Otherwise, reset the tree
            self.root = None
