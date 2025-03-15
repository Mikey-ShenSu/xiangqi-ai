"""
Evaluator module for AlphaZero Xiangqi

This module provides functionality to evaluate and compare different versions
of the AlphaZero model by having them play against each other.
"""

import logging
import time
import os
import numpy as np
from tqdm import tqdm

from .config import *
from .mcts import MCTS
from .utils import ensure_dir, get_timestamp

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("xiangqi.evaluator")


class Evaluator:
    """Evaluator for AlphaZero Xiangqi models"""

    def __init__(self, game_adapter):
        """
        Initialize the evaluator

        Args:
            game_adapter: Adapter for the Xiangqi game
        """
        self.game_adapter = game_adapter

    def play_game(self, model1, model2, mcts1=None, mcts2=None, render=False):
        """
        Play a single game between two models

        Args:
            model1: First model (plays red)
            model2: Second model (plays black)
            mcts1: MCTS instance for model1 (optional)
            mcts2: MCTS instance for model2 (optional)
            render: Whether to render the game board at each step

        Returns:
            result: 1 if model1 wins, -1 if model2 wins, 0 for draw
        """
        # Initialize MCTS if not provided
        if mcts1 is None:
            mcts1 = MCTS(model1, self.game_adapter, num_simulations=NUM_SIMULATIONS)
        if mcts2 is None:
            mcts2 = MCTS(model2, self.game_adapter, num_simulations=NUM_SIMULATIONS)

        # Initialize game
        board, current_player = self.game_adapter.reset()

        # History of states to detect repetitions
        state_history = {}

        # Play until game is over
        step = 0
        while True:
            step += 1

            # Select MCTS based on current player
            if current_player == "red":
                mcts = mcts1
                model = model1
            else:
                mcts = mcts2
                model = model2

            # Use lower temperature for more deterministic play during evaluation
            temperature = EVAL_TEMPERATURE

            # Render the board if requested
            if render:
                print(f"Step {step}, {current_player} to move")
                print(self.game_adapter.to_string(board))
                print()

            # Select action using MCTS
            action = mcts.select_action(board, current_player, temperature)

            # Apply action to get new state
            from_pos, to_pos = self.game_adapter.action_to_move(action)
            board, current_player = self.game_adapter.make_move(
                board, current_player, from_pos, to_pos
            )

            # Update MCTS trees
            mcts1.update_with_move(action)
            mcts2.update_with_move(action)

            # Convert board state to string for repetition detection
            board_str = self.game_adapter.to_string(board)
            if board_str in state_history:
                state_history[board_str] += 1
                # Detect threefold repetition (draw)
                if state_history[board_str] >= 3:
                    if render:
                        print("Game drawn by threefold repetition")
                    return 0  # Draw
            else:
                state_history[board_str] = 1

            # Check if game is over
            game_over, winner = self.game_adapter.is_game_over(board)
            if game_over:
                if render:
                    print(f"Game over. Winner: {winner if winner else 'Draw'}")
                    print(self.game_adapter.to_string(board))

                # Determine result from model1's perspective
                if winner is None:  # Draw
                    return 0
                elif winner == "red":  # Model1 (red) wins
                    return 1
                else:  # Model2 (black) wins
                    return -1

            # Limit game length
            if step >= 200:  # Avoid excessively long games
                if render:
                    print("Game drawn by move limit")
                return 0  # Draw

    def evaluate(self, model1, model2, num_games=EVAL_EPISODES, render=False):
        """
        Evaluate two models by playing them against each other

        Args:
            model1: First model
            model2: Second model
            num_games: Number of games to play
            render: Whether to render the games

        Returns:
            win_rate: Win rate of model1 against model2
            results: List of game results
        """
        model1_stats = {"wins": 0, "losses": 0, "draws": 0}
        results = []

        logger.info(f"Evaluating models over {num_games} games")
        start_time = time.time()

        # Play half of the games with model1 as red and model2 as black
        half_games = num_games // 2

        # Progress bar for first half
        logger.info("Playing first half with model1 as red")
        for i in tqdm(range(half_games)):
            # Model1 plays as red, model2 as black
            result = self.play_game(model1, model2, render=render)
            results.append(result)

            # Update stats
            if result == 1:
                model1_stats["wins"] += 1
            elif result == -1:
                model1_stats["losses"] += 1
            else:
                model1_stats["draws"] += 1

        # Progress bar for second half
        logger.info("Playing second half with model1 as black")
        for i in tqdm(range(half_games, num_games)):
            # Model2 plays as red, model1 as black
            # Need to negate the result since we're swapping the roles
            result = self.play_game(model2, model1, render=render)
            results.append(-result)  # Negate to get model1's perspective

            # Update stats
            if result == -1:  # Model1 (black) wins
                model1_stats["wins"] += 1
            elif result == 1:  # Model2 (red) wins
                model1_stats["losses"] += 1
            else:
                model1_stats["draws"] += 1

        end_time = time.time()
        duration = end_time - start_time

        # Calculate win rate
        win_rate = (model1_stats["wins"] + 0.5 * model1_stats["draws"]) / num_games

        logger.info(f"Evaluation completed in {duration:.2f} seconds")
        logger.info(f"Model1 stats: {model1_stats}")
        logger.info(f"Win rate of model1 vs model2: {win_rate:.4f}")

        return win_rate, results

    def evaluate_against_random(self, model, num_games=EVAL_EPISODES, render=False):
        """
        Evaluate a model against random play (baseline)

        Args:
            model: Model to evaluate
            num_games: Number of games to play
            render: Whether to render the games

        Returns:
            win_rate: Win rate of model against random play
        """

        # For random play, we'll use a custom model that returns uniform probabilities
        class RandomModel:
            def predict(self, state):
                # Return uniform policy and random value
                policy = np.ones(ACTION_SIZE) / ACTION_SIZE
                value = np.random.uniform(-0.1, 0.1)  # Small random value
                return policy, value

        random_model = RandomModel()

        # Evaluate model against random play
        win_rate, _ = self.evaluate(model, random_model, num_games, render)

        return win_rate

    def is_new_model_better(self, new_model, best_model, num_games=EVAL_EPISODES):
        """
        Determine if the new model is better than the best model so far

        Args:
            new_model: New model to evaluate
            best_model: Current best model
            num_games: Number of games to play

        Returns:
            is_better: Whether the new model is better
            win_rate: Win rate of new model against best model
        """
        logger.info("Evaluating new model against best model")
        win_rate, _ = self.evaluate(new_model, best_model, num_games)

        # Check if win rate exceeds threshold
        is_better = win_rate >= WIN_RATE_THRESHOLD

        if is_better:
            logger.info(
                f"New model is better with win rate {win_rate:.4f} ≥ {WIN_RATE_THRESHOLD}"
            )
        else:
            logger.info(
                f"New model is not better. Win rate {win_rate:.4f} < {WIN_RATE_THRESHOLD}"
            )

        return is_better, win_rate

    def log_evaluation(self, model1_name, model2_name, win_rate, results):
        """
        Log evaluation results to file

        Args:
            model1_name: Name of the first model
            model2_name: Name of the second model
            win_rate: Win rate of model1 vs model2
            results: List of game results
        """
        ensure_dir(os.path.join(MODEL_DIR, "evaluation"))
        timestamp = get_timestamp()
        log_file = os.path.join(MODEL_DIR, "evaluation", f"eval_{timestamp}.txt")

        with open(log_file, "w") as f:
            f.write(f"Evaluation: {model1_name} vs {model2_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Games played: {len(results)}\n")
            f.write(f"Win rate of {model1_name}: {win_rate:.4f}\n\n")

            wins = results.count(1)
            losses = results.count(-1)
            draws = results.count(0)

            f.write(f"{model1_name} wins: {wins}\n")
            f.write(f"{model2_name} wins: {losses}\n")
            f.write(f"Draws: {draws}\n\n")

            f.write("Individual game results:\n")
            for i, result in enumerate(results):
                result_str = (
                    "Win" if result == 1 else "Loss" if result == -1 else "Draw"
                )
                f.write(f"Game {i+1}: {result_str}\n")

        logger.info(f"Evaluation log saved to {log_file}")
