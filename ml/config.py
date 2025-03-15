"""
Configuration for AlphaZero implementation for Xiangqi (Chinese Chess)
"""

# Game parameters
BOARD_WIDTH = 9
BOARD_HEIGHT = 10
ACTION_SIZE = 9 * 10 * 9 * 10  # All possible from-to combinations

# Neural Network parameters
NUM_CHANNELS = 256  # Number of channels in residual blocks
NUM_RESIDUAL_BLOCKS = 19  # Number of residual blocks in the neural network
VALUE_HEAD_HIDDEN_SIZE = 256  # Size of the hidden layer in value head
L2_REGULARIZATION = 1e-4  # L2 regularization factor
LEARNING_RATE = 0.001  # Initial learning rate
LEARNING_RATE_DECAY_STEPS = 100000  # Number of steps for learning rate decay
LEARNING_RATE_DECAY_RATE = 0.1  # Learning rate decay factor

# Monte Carlo Tree Search parameters
NUM_SIMULATIONS = 800  # Number of simulations per move
C_PUCT = 1.0  # Exploration constant
DIRICHLET_ALPHA = 0.3  # Alpha parameter for Dirichlet noise
DIRICHLET_NOISE_FACTOR = 0.25  # Weight of Dirichlet noise
TEMPERATURE = 1.0  # Initial temperature for action selection
TEMPERATURE_DROP_STEP = 15  # Move number to drop temperature to 0

# Training parameters
BATCH_SIZE = 2048  # Batch size for training
BUFFER_SIZE = 500000  # Size of replay buffer
EPOCHS_PER_SAVE = 1  # Save model every n epochs
NUM_SELF_PLAY_GAMES = 5000  # Number of self-play games to generate
TRAINING_STEPS = 50  # Number of training steps per epoch
NUM_EPOCHS = 100  # Number of epochs

# Evaluation parameters
EVAL_EPISODES = 100  # Number of games to play for evaluation
EVAL_TEMPERATURE = 0.1  # Temperature for evaluation
WIN_RATE_THRESHOLD = 0.55  # Win rate needed to replace the champion model

# Paths
MODEL_DIR = "./saved_models"
DATA_DIR = "./training_data"
LATEST_MODEL_NAME = "xiangqi_latest.pt"
BEST_MODEL_NAME = "xiangqi_best.pt"
