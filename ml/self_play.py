"""
Self-play module for AlphaZero Xiangqi implementation

This module implements self-play functionality to generate training data by having
the AlphaZero model play against itself.
"""

import numpy as np
import time
import os
import logging
from tqdm import tqdm

from .config import *
from .utils import ensure_dir, save_training_data, get_timestamp, get_legal_actions_mask

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("xiangqi.self_play")


class SelfPlay:
    """
    Self-play implementation for Xiangqi AlphaZero
    """

    def __init__(self, game_adapter, mcts, model):
        """
        Initialize self-play

        Args:
            game_adapter: Adapter for the Xiangqi game
            mcts: Monte Carlo Tree Search instance
            model: Neural network model
        """
        self.game_adapter = game_adapter
        self.mcts = mcts
        self.model = model

    def execute_episode(self, temperature_drop_step=TEMPERATURE_DROP_STEP):
        """
        Play a single game episode

        Args:
            temperature_drop_step: After how many steps to drop the temperature

        Returns:
            training_examples: List of (state, policy, value) tuples
        """
        # List to store training examples: (state, policy, value)
        training_examples = []

        # Initialize game
        board, current_player = self.game_adapter.reset()
        step = 0

        # History of states to detect repetitions
        state_history = {}

        # Play until game is over
        while True:
            step += 1

            # Convert board to neural network input
            state = self.game_adapter.board_to_planes(board, current_player)

            # Set temperature based on move number
            if step <= temperature_drop_step:
                temperature = TEMPERATURE
            else:
                temperature = 0.1  # Near-deterministic play after temperature_drop_step

            # Use MCTS to get action probabilities
            # Add Dirichlet noise to root for exploration
            add_dirichlet = True
            action_probs = self.mcts.search(
                board, current_player, dirichlet_noise=add_dirichlet
            )

            # Store current state, probabilities for training
            training_examples.append((state, action_probs))

            # Choose action based on probabilities
            action = self.mcts.select_action(board, current_player, temperature)

            # Apply action to get new state
            from_pos, to_pos = self.game_adapter.action_to_move(action)
            board, current_player = self.game_adapter.make_move(
                board, current_player, from_pos, to_pos
            )

            # Update MCTS tree for efficiency
            self.mcts.update_with_move(action)

            # Convert board state to string for repetition detection
            board_str = self.game_adapter.to_string(board)
            if board_str in state_history:
                state_history[board_str] += 1
                # Detect threefold repetition (draw)
                if state_history[board_str] >= 3:
                    game_result = 0  # Draw
                    break
            else:
                state_history[board_str] = 1

            # Check if game is over
            game_over, winner = self.game_adapter.is_game_over(board)
            if game_over:
                # Determine game result from perspective of each state's player
                if winner is None:  # Draw
                    game_result = 0
                else:
                    # Map the winner to a result value
                    game_result = 1 if winner == current_player else -1

                break

            # Limit game length
            if step >= 200:  # Avoid excessively long games
                game_result = 0  # Draw
                break

        # Assign rewards to all stored states
        # Format: (state, policy, value)
        return [
            (state, policy, game_result if i % 2 == 0 else -game_result)
            for i, (state, policy) in enumerate(training_examples)
        ]

    def generate_games(self, num_games=NUM_SELF_PLAY_GAMES):
        """
        Generate multiple games of self-play data

        Args:
            num_games: Number of games to play

        Returns:
            training_data: List of training examples from all games
        """
        training_data = []
        start_time = time.time()

        logger.info(f"Generating {num_games} self-play games...")

        for i in tqdm(range(num_games)):
            # Execute one game episode
            episode_data = self.execute_episode()
            training_data.extend(episode_data)

            # Save intermediate results every 10 games
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_games} games")

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Self-play completed in {duration:.2f} seconds")
        logger.info(f"Collected {len(training_data)} training examples")

        return training_data

    def generate_and_save_games(self, num_games=NUM_SELF_PLAY_GAMES):
        """
        Generate self-play games and save the training data

        Args:
            num_games: Number of games to play

        Returns:
            filename: Path to the saved training data
        """
        # Generate self-play data
        training_data = self.generate_games(num_games)

        # Create directory if it doesn't exist
        ensure_dir(DATA_DIR)

        # Save data with timestamp
        timestamp = get_timestamp()
        filename = os.path.join(DATA_DIR, f"self_play_data_{timestamp}.npz")
        save_training_data(training_data, filename)

        logger.info(f"Training data saved to {filename}")

        return filename
