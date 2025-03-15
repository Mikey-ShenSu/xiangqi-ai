#!/usr/bin/env python
"""
Xiangqi AlphaZero main script

This script provides a command-line interface to run the AlphaZero implementation
for Xiangqi (Chinese Chess).
"""

import os
import sys
import argparse
import logging

from ml.pipeline import AlphaZeroPipeline
from ml.config import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "xiangqi_alphazero.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("xiangqi.main")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Xiangqi AlphaZero")

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the AlphaZero model")
    train_parser.add_argument(
        "--iterations", type=int, default=10, help="Number of training iterations"
    )
    train_parser.add_argument(
        "--self-play-games",
        type=int,
        default=NUM_SELF_PLAY_GAMES,
        help="Number of self-play games per iteration",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs per iteration",
    )
    train_parser.add_argument(
        "--eval-games",
        type=int,
        default=EVAL_EPISODES,
        help="Number of evaluation games per iteration",
    )

    # Self-play command
    self_play_parser = subparsers.add_parser(
        "self-play", help="Generate self-play data"
    )
    self_play_parser.add_argument(
        "--games",
        type=int,
        default=NUM_SELF_PLAY_GAMES,
        help="Number of self-play games to generate",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument(
        "--games", type=int, default=EVAL_EPISODES, help="Number of evaluation games"
    )
    eval_parser.add_argument(
        "--render", action="store_true", help="Render games during evaluation"
    )

    # Play command
    play_parser = subparsers.add_parser("play", help="Play against the trained model")
    play_parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for move selection"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create AlphaZero pipeline
    pipeline = AlphaZeroPipeline()

    if args.command == "train":
        # Run training
        logger.info(f"Starting training for {args.iterations} iterations")
        logger.info(f"Self-play games per iteration: {args.self_play_games}")
        logger.info(f"Training epochs per iteration: {args.epochs}")
        logger.info(f"Evaluation games per iteration: {args.eval_games}")

        # Start training
        pipeline.train(args.iterations)

    elif args.command == "self-play":
        # Generate self-play data
        logger.info(f"Generating {args.games} self-play games")

        # Initialize components
        pipeline.initialize_components()

        # Run self-play
        data_file = pipeline.run_self_play(args.games)
        logger.info(f"Self-play data saved to {data_file}")

    elif args.command == "evaluate":
        # Evaluate the model
        logger.info(f"Evaluating model over {args.games} games")

        # Initialize components
        pipeline.initialize_components()

        # Load best model if available
        best_model_exists = os.path.exists(os.path.join(MODEL_DIR, BEST_MODEL_NAME))
        if best_model_exists:
            # Evaluate against best model
            logger.info("Evaluating against best model")
            is_better, win_rate = pipeline.run_evaluation(args.games)
            logger.info(f"Win rate against best model: {win_rate:.4f}")
        else:
            # Evaluate against random play
            logger.info("No best model found, evaluating against random play")
            win_rate = pipeline.evaluator.evaluate_against_random(
                pipeline.model, args.games, render=args.render
            )
            logger.info(f"Win rate against random play: {win_rate:.4f}")

    elif args.command == "play":
        # Play against the trained model
        logger.info("Starting human vs AI game")
        logger.info(f"Temperature: {args.temperature}")

        # Start game
        pipeline.play_human(args.temperature)

    else:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
