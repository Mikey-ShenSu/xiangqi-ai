"""
Utility functions for Xiangqi AlphaZero implementation
"""

import numpy as np
import torch
import os
import pickle
from datetime import datetime


def ensure_dir(directory):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def board_to_planes(board, current_player):
    """
    Convert a board state to a set of feature planes for the neural network.

    Each piece type for each player gets its own binary feature plane.
    Additional planes for game state info (current player, etc.).

    Args:
        board: The Xiangqi board state (2D array)
        current_player: The current player ('red' or 'black')

    Returns:
        np.array: Feature planes representing the board (19, 10, 9)
    """
    # 7 piece types * 2 colors + 5 auxiliary features = 19 planes
    # Board size is 10x9
    planes = np.zeros((19, 10, 9), dtype=np.float32)

    # Map piece characters to plane indices
    piece_to_plane = {
        'G': 0,  # Red General
        'A': 1,  # Red Advisor
        'E': 2,  # Red Elephant
        'H': 3,  # Red Horse
        'R': 4,  # Red Chariot
        'C': 5,  # Red Cannon
        'S': 6,  # Red Soldier
        'g': 7,  # Black General
        'a': 8,  # Black Advisor
        'e': 9,  # Black Elephant
        'h': 10, # Black Horse
        'r': 11, # Black Chariot
        'c': 12, # Black Cannon
        's': 13, # Black Soldier
    }

    # Fill piece planes from the 2D board array
    for i in range(10):  # rows
        for j in range(9):  # columns
            piece = board[i][j]
            if piece != ' ':
                plane_idx = piece_to_plane.get(piece)
                if plane_idx is not None:
                    planes[plane_idx, i, j] = 1

    # Current player plane
    if current_player == "red":
        planes[14].fill(1)  # All 1s if red to move
    
    # TODO: Add more auxiliary planes like:
    # - Move count
    # - Repetition count
    # - Check status
    # - Legal moves mask

    return planes


def action_to_move(action_idx, board_size=(10, 9)):
    """
    Convert a flattened action index to a move (from_pos, to_pos)

    Args:
        action_idx: Integer index of the action
        board_size: Tuple (height, width) of the board

    Returns:
        tuple: ((from_x, from_y), (to_x, to_y)) representing the move
    """
    height, width = board_size
    total_positions = height * width

    # Decode the action index
    from_idx = action_idx // (total_positions)
    to_idx = action_idx % (total_positions)

    # Convert indices to 2D positions
    from_y, from_x = divmod(from_idx, width)
    to_y, to_x = divmod(to_idx, width)

    return ((from_x, from_y), (to_x, to_y))


def move_to_action(from_pos, to_pos, board_size=(10, 9)):
    """
    Convert a move (from_pos, to_pos) to a flattened action index

    Args:
        from_pos: (x, y) position of the piece to move
        to_pos: (x, y) position to move to
        board_size: Tuple (height, width) of the board

    Returns:
        int: Flattened action index
    """
    height, width = board_size
    total_positions = height * width

    from_x, from_y = from_pos
    to_x, to_y = to_pos

    from_idx = from_y * width + from_x
    to_idx = to_y * width + to_x

    return from_idx * total_positions + to_idx


def get_legal_actions_mask(board, current_player):
    """
    Create a binary mask of legal actions for the current player.

    Args:
        board: The Xiangqi board
        current_player: The current player ('red' or 'black')

    Returns:
        np.array: Binary mask where 1 = legal action, 0 = illegal action
    """
    action_space_size = 9 * 10 * 9 * 10  # All possible from-to combinations
    mask = np.zeros(action_space_size, dtype=np.float32)

    # Find all pieces for the current player
    for piece in board.pieces:
        if not piece.captured and piece.color == current_player:
            valid_moves = piece.get_valid_moves(board)

            # Set mask to 1 for all valid moves from this piece
            for to_pos in valid_moves:
                action_idx = move_to_action(piece.position, to_pos)
                if action_idx < action_space_size:  # Safety check
                    mask[action_idx] = 1

    return mask


def save_training_data(data, filename):
    """Save training data (state, policy, value) to a file"""
    ensure_dir(os.path.dirname(filename))
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_training_data(filename):
    """Load training data from a file"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def get_timestamp():
    """Get current timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
