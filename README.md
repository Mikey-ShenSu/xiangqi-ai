# Xiangqi (Chinese Chess) / 中國象棋

A Python implementation of Xiangqi (Chinese Chess) featuring both console-based and graphical user interfaces with Chinese character display.

## Game Overview

Xiangqi is a traditional Chinese board game similar to Western chess but with different pieces and rules. The game is played on a 9×10 board with pieces placed at the intersections of the grid lines.

## Features

- Complete implementation of Xiangqi rules
- Console-based text interface
- GUI interface with mouse-based interaction
- Chinese character display for piece names and game status
- Visual indicators for valid moves, checks, and game state

## Piece Rules and Chinese Names

### General (King) / 帥(紅) 將(黑)

- Moves one point horizontally or vertically
- Cannot leave the palace (3x3 grid)
- Cannot face the opponent's General directly with no pieces in between

### Advisor / 仕(紅) 士(黑)

- Moves one point diagonally
- Cannot leave the palace (3x3 grid)

### Elephant / 相(紅) 象(黑)

- Moves exactly two points diagonally
- Cannot cross the river (stays on its own side)
- Cannot jump over pieces

### Horse / 馬

- Moves one point orthogonally followed by one point diagonally outward
- Cannot jump over pieces (blocked if there's a piece at the orthogonal step)

### Chariot (Rook) / 車

- Moves any number of points horizontally or vertically
- Cannot jump over pieces

### Cannon / 炮(紅) 砲(黑)

- Moves like the Chariot for non-capturing moves
- Must jump over exactly one piece (of either color) to capture
- Cannot jump over pieces when not capturing

### Soldier (Pawn) / 兵(紅) 卒(黑)

- Before crossing the river: moves one point forward only
- After crossing the river: can move one point forward or horizontally
- Never moves backward

## How to Play

### Text-Based Interface

Run the console version with:

```
python main.py
```

Enter moves in the format `x1 y1 x2 y2` to move a piece from position (x1,y1) to (x2,y2).
Type `quit` to exit the game.

### Graphical Interface

Run the GUI version with:

```
python gui_game.py
```

- **Click on a piece** to select it. Valid moves will be highlighted in green.
- **Click on a valid move square** to move the selected piece.
- The current player's turn is displayed at the top of the screen in Chinese.
- Check status is displayed at the bottom when applicable.
- All pieces are displayed with their traditional Chinese characters.

## Requirements

- Python 3.6+
- Pygame (for GUI version)

Install the required packages with:

```
pip install pygame
```

## License

MIT License

# Xiangqi AlphaZero

This project implements an AlphaZero-like reinforcement learning system for Xiangqi (Chinese Chess). It combines deep learning with Monte Carlo Tree Search to create a powerful Xiangqi AI that learns to play through self-play, without human knowledge or guidance.

## Features

- Complete AlphaZero implementation for Xiangqi
- Neural network architecture with policy and value heads
- Monte Carlo Tree Search (MCTS) for action selection
- Self-play reinforcement learning framework
- Training and evaluation pipeline
- Human vs AI play interface

## Project Structure

```
xiangqi-ai/
├── gui_game.py            # Xiangqi GUI game implementation
├── main.py                # Main GUI entry point
├── ml_main.py             # AlphaZero main entry point
├── ml/                    # Machine learning modules
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and configuration
│   ├── model.py           # Neural network architecture
│   ├── mcts.py            # Monte Carlo Tree Search implementation
│   ├── game_adapter.py    # Interface between AlphaZero and Xiangqi game
│   ├── self_play.py       # Self-play game generation
│   ├── training.py        # Neural network training
│   ├── evaluator.py       # Model evaluation
│   ├── pipeline.py        # Complete training pipeline
│   └── utils.py           # Utility functions
├── saved_models/          # For storing trained models
└── training_data/         # For storing self-play game data
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm

Install dependencies with:

```bash
pip install torch numpy matplotlib tqdm
```

## Usage

### Training the Model

To train the AlphaZero model from scratch:

```bash
python ml_main.py train --iterations 10
```

Options:

- `--iterations`: Number of training iterations (default: 10)
- `--self-play-games`: Number of self-play games per iteration (default: configured in ml/config.py)
- `--epochs`: Number of training epochs per iteration (default: configured in ml/config.py)
- `--eval-games`: Number of evaluation games per iteration (default: configured in ml/config.py)

### Generating Self-Play Data

To generate self-play data for training:

```bash
python ml_main.py self-play --games 100
```

Options:

- `--games`: Number of self-play games to generate (default: configured in ml/config.py)

### Evaluating the Model

To evaluate the trained model:

```bash
python ml_main.py evaluate --games 20 --render
```

Options:

- `--games`: Number of evaluation games (default: configured in ml/config.py)
- `--render`: Render games during evaluation (optional)

### Playing Against the AI

To play against the trained model:

```bash
python ml_main.py play
```

Options:

- `--temperature`: Temperature parameter for move selection (default: 0.1)

## How It Works

### AlphaZero Algorithm

The AlphaZero algorithm consists of the following key components:

1. **Neural Network**: A deep neural network with a policy head and a value head. The policy head predicts the probability of taking each action, while the value head estimates the expected outcome of the game.

2. **Monte Carlo Tree Search (MCTS)**: A search algorithm that uses the neural network to guide the search process. It balances exploration and exploitation to find the best moves.

3. **Self-Play**: The model plays against itself to generate training data. The MCTS algorithm is used to select moves during self-play.

4. **Training**: The neural network is trained on the self-play data to improve its policy and value predictions.

5. **Evaluation**: The model is evaluated against previous versions to track its progress.

### Training Process

The training process follows an iterative approach:

1. **Self-Play**: Generate games by having the current model play against itself using MCTS.
2. **Training**: Train the neural network on the generated data.
3. **Evaluation**: Evaluate the new model against the best previous model.
4. **Repeat**: If the new model is better, it becomes the best model. Repeat the process.

## Customization

You can customize the AlphaZero implementation by modifying the hyperparameters in `ml/config.py`. Key parameters include:

- Neural network architecture (channels, layers)
- MCTS parameters (simulations, exploration constant)
- Training parameters (batch size, learning rate)
- Self-play parameters (games, temperature)

## Acknowledgments

This implementation is inspired by DeepMind's AlphaZero algorithm and various open-source implementations in the community. The Xiangqi game logic is based on the existing GUI implementation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
