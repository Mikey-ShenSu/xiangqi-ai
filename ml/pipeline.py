"""
AlphaZero Pipeline for Xiangqi

This module integrates all components of the AlphaZero implementation for Xiangqi,
providing a unified interface for the complete training and evaluation pipeline.
"""

import os
import logging
import glob
from datetime import datetime
import torch

from .config import *
from .model import XiangqiNetwork
from .mcts import MCTS
from .game_adapter import XiangqiGameAdapter
from .self_play import SelfPlay
from .training import TrainingPipeline
from .evaluator import Evaluator
from .utils import ensure_dir, get_timestamp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "xiangqi_alphazero.log"
            )
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("xiangqi.pipeline")


class AlphaZeroPipeline:
    """
    Complete AlphaZero training and evaluation pipeline for Xiangqi
    """

    def __init__(self):
        """Initialize the AlphaZero pipeline"""
        # Create directories
        ensure_dir(MODEL_DIR)
        ensure_dir(DATA_DIR)
        
        # Print CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PyTorch device available: {device}")
        print(f"PyTorch using device: {device}")
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            print(f"GPU: {gpu_name}")

        # Initialize components
        self.game_adapter = XiangqiGameAdapter()
        self.model = XiangqiNetwork()
        self.best_model = XiangqiNetwork()
        self.mcts = None  # Will be initialized later
        self.self_play = None  # Will be initialized later
        self.training = TrainingPipeline(self.model)
        self.evaluator = Evaluator(self.game_adapter)

        logger.info("AlphaZero pipeline initialized")

    def initialize_components(self):
        """Initialize or reinitialize components that depend on the model"""
        self.mcts = MCTS(self.model, self.game_adapter)
        self.self_play = SelfPlay(self.game_adapter, self.mcts, self.model)
        logger.info("Pipeline components initialized")

    def run_self_play(self, num_games=NUM_SELF_PLAY_GAMES):
        """
        Run self-play to generate training data

        Args:
            num_games: Number of games to play

        Returns:
            data_file: Path to the generated data file
        """
        logger.info(f"Starting self-play for {num_games} games")
        if self.self_play is None:
            self.initialize_components()

        # Generate and save self-play data
        data_file = self.self_play.generate_and_save_games(num_games)
        logger.info(f"Self-play completed, data saved to {data_file}")

        return data_file

    def run_training(self, data_files, num_epochs=NUM_EPOCHS):
        """
        Train the model on the provided data

        Args:
            data_files: List of data file paths to train on
            num_epochs: Number of epochs to train for

        Returns:
            final_loss: Final training loss
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        # Train the model
        final_loss = self.training.train_from_files(data_files, num_epochs)
        logger.info(f"Training completed with final loss: {final_loss:.4f}")

        # Reinitialize components with the updated model
        self.initialize_components()

        return final_loss

    def run_evaluation(self, num_games=EVAL_EPISODES):
        """
        Evaluate the current model against the best model

        Args:
            num_games: Number of games to play for evaluation

        Returns:
            is_better: Whether the current model is better than the best model
            win_rate: Win rate of the current model against the best model
        """
        logger.info("Starting evaluation against best model")

        # Load best model if it exists
        best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
        if os.path.exists(best_model_path):
            trainer = TrainingPipeline(self.best_model)
            trainer.load_best_model()
            logger.info("Loaded best model for comparison")
        else:
            logger.info("No best model found, current model will become the best model")
            self.save_best_model()
            return True, 1.0

        # Evaluate current model against best model
        is_better, win_rate = self.evaluator.is_new_model_better(
            self.model, self.best_model, num_games
        )

        # If current model is better, update best model
        if is_better:
            logger.info(
                "Current model is better than the best model, updating best model"
            )
            self.save_best_model()

        return is_better, win_rate

    def save_best_model(self):
        """Save the current model as the best model"""
        # Copy current model parameters to best model
        self.best_model.load_state_dict(self.model.state_dict())

        # Save to disk
        self.training.save_model(is_best=True)
        logger.info("Saved current model as the best model")

    def run_complete_iteration(
        self,
        iteration,
        num_self_play_games=NUM_SELF_PLAY_GAMES,
        num_epochs=NUM_EPOCHS,
        num_eval_games=EVAL_EPISODES,
    ):
        """
        Run one complete iteration of the AlphaZero training pipeline

        Args:
            iteration: Current iteration number
            num_self_play_games: Number of self-play games to generate
            num_epochs: Number of training epochs
            num_eval_games: Number of evaluation games

        Returns:
            is_better: Whether the new model is better than the previous best
            win_rate: Win rate of the new model against the previous best
        """
        timestamp = get_timestamp()
        logger.info(f"Starting iteration {iteration} at {timestamp}")

        # 1. Self-play
        data_file = self.run_self_play(num_self_play_games)

        # 2. Training
        # Get all recent data files (last 5 iterations)
        recent_files = self.get_recent_data_files(5)
        if data_file not in recent_files:
            recent_files.append(data_file)

        self.run_training(recent_files, num_epochs)

        # 3. Evaluation
        is_better, win_rate = self.run_evaluation(num_eval_games)

        logger.info(f"Completed iteration {iteration}")
        return is_better, win_rate

    def get_recent_data_files(self, num_iterations=5):
        """
        Get the most recent self-play data files

        Args:
            num_iterations: Number of most recent iterations to include

        Returns:
            List of file paths to the most recent data files
        """
        data_files = glob.glob(os.path.join(DATA_DIR, "self_play_data_*.npz"))
        data_files.sort(key=os.path.getmtime, reverse=True)
        return data_files[: min(len(data_files), num_iterations)]

    def train(self, num_iterations=10):
        """
        Run the complete AlphaZero training pipeline for multiple iterations

        Args:
            num_iterations: Number of iterations to run
        """
        logger.info(f"Starting AlphaZero training for {num_iterations} iterations")

        # Load latest model if it exists
        latest_model_path = os.path.join(MODEL_DIR, LATEST_MODEL_NAME)
        if os.path.exists(latest_model_path):
            self.training.load_latest_model()
            logger.info("Loaded latest model to continue training")
            self.initialize_components()
        else:
            logger.info("No existing model found, starting with a new model")

        # Run iterations
        for i in range(1, num_iterations + 1):
            is_better, win_rate = self.run_complete_iteration(i)

            # Log results
            logger.info(
                f"Iteration {i} - Win rate: {win_rate:.4f}, "
                + f"Model improved: {is_better}"
            )

        logger.info("Training completed")

    def play_human(self, temperature=0.1):
        """
        Allow a human to play against the trained model

        Args:
            temperature: Temperature parameter for move selection
        """
        # Load best model if available
        best_model_path = os.path.join(MODEL_DIR, BEST_MODEL_NAME)
        if os.path.exists(best_model_path):
            self.training.load_best_model()
            logger.info("Loaded best model for play")
        else:
            logger.info("No best model found, using current model")

        self.initialize_components()

        # Initialize game
        board, current_player = self.game_adapter.reset()

        # Game loop
        game_over = False
        while not game_over:
            print("\nCurrent board:")
            print(self.game_adapter.to_string(board))
            print(f"Current player: {current_player}")

            if current_player == "red":  # Human plays as red
                # Get move from human
                try:
                    from_row = int(input("Enter from row (0-9): "))
                    from_col = int(input("Enter from column (0-8): "))
                    to_row = int(input("Enter to row (0-9): "))
                    to_col = int(input("Enter to column (0-8): "))

                    from_pos = (from_row, from_col)
                    to_pos = (to_row, to_col)

                    # Validate move
                    if not self.game_adapter.game.is_valid_move(
                        board, from_pos, to_pos, current_player
                    ):
                        print("Invalid move! Try again.")
                        continue

                    # Make the move
                    board, current_player = self.game_adapter.make_move(
                        board, current_player, from_pos, to_pos
                    )
                except ValueError:
                    print("Invalid input! Please enter numbers only.")
                    continue
            else:  # AI plays as black
                print("AI is thinking...")

                # Use MCTS to select action
                action = self.mcts.select_action(board, current_player, temperature)

                # Convert action to move
                from_pos, to_pos = self.game_adapter.action_to_move(action)
                print(f"AI move: {from_pos} -> {to_pos}")

                # Make the move
                board, current_player = self.game_adapter.make_move(
                    board, current_player, from_pos, to_pos
                )

            # Check if game is over
            game_over, winner = self.game_adapter.is_game_over(board)
            if game_over:
                print("\nGame over!")
                print(self.game_adapter.to_string(board))
                if winner:
                    print(f"Winner: {winner}")
                else:
                    print("Game drawn")

        print("Thank you for playing!")
