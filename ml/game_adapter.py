"""
Game adapter for Xiangqi AlphaZero implementation

This module serves as an interface between the AlphaZero implementation and the
existing Xiangqi game, providing methods to convert between different representations
and handle game logic.
"""

import numpy as np
import sys
import os
from copy import deepcopy

# Add the parent directory to the path to import the game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui_game import Game
from main import General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier

from .config import *
from .utils import action_to_move as utils_action_to_move
from .utils import move_to_action as utils_move_to_action
from .utils import get_legal_actions_mask


class XiangqiGameAdapter:
    """
    Adapter class that interfaces between AlphaZero and the Xiangqi game
    """

    def __init__(self):
        """Initialize the game adapter"""
        # Create a game instance for rule checking
        self.game = Game()

    def reset(self):
        """
        Reset the game to its initial state

        Returns:
            board: The initial board state
            current_player: The starting player ('red')
        """
        # Instead of calling reset_game() which doesn't exist,
        # create a new game instance which will be initialized with the starting position
        self.game = Game()

        # Return the initial board and player
        return self.game.board, "red"

    def board_to_planes(self, board, current_player):
        """
        Convert the board to a set of feature planes for the neural network

        Args:
            board: The current board state
            current_player: The current player ('red' or 'black')

        Returns:
            planes: A numpy array of shape (19, 10, 9) representing the board state
        """
        # Create planes
        planes = np.zeros((19, 10, 9), dtype=np.float32)

        # Go through each piece in the board
        for piece in board.pieces:
            if piece.captured:
                continue

            x, y = piece.position

            if isinstance(piece, General):
                plane_idx = 0 if piece.color == "red" else 7
            elif isinstance(piece, Advisor):
                plane_idx = 1 if piece.color == "red" else 8
            elif isinstance(piece, Elephant):
                plane_idx = 2 if piece.color == "red" else 9
            elif isinstance(piece, Horse):
                plane_idx = 3 if piece.color == "red" else 10
            elif isinstance(piece, Chariot):
                plane_idx = 4 if piece.color == "red" else 11
            elif isinstance(piece, Cannon):
                plane_idx = 5 if piece.color == "red" else 12
            elif isinstance(piece, Soldier):
                plane_idx = 6 if piece.color == "red" else 13
            else:
                continue

            planes[plane_idx, y, x] = 1

        # Add current player plane
        if current_player == "red":
            planes[14].fill(1)  # All 1s if red to move

        return planes

    def get_legal_actions(self, board, current_player):
        """
        Get all legal actions for the current player

        Args:
            board: The current board state
            current_player: The current player ('red' or 'black')

        Returns:
            legal_actions: A list of legal action indices
        """
        legal_actions = []

        # Go through all pieces of the current player
        for piece in board.pieces:
            if piece.captured or piece.color != current_player:
                continue

            # Get valid moves for this piece
            valid_moves = piece.get_valid_moves(board)
            from_pos = piece.position

            # Convert each move to an action index
            for to_pos in valid_moves:
                action = self.move_to_action(from_pos, to_pos)
                legal_actions.append(action)

        return legal_actions

    def make_move(self, board, current_player, from_pos, to_pos):
        """
        Make a move on the board

        Args:
            board: The current board state
            current_player: The current player ('red' or 'black')
            from_pos: The position to move from (i, j)
            to_pos: The position to move to (i, j)

        Returns:
            new_board: The new board state after the move
            next_player: The next player ('red' or 'black')
        """
        # Create a deep copy of the board
        new_board = deepcopy(board)

        # Make the move
        piece = new_board.get_piece_at(from_pos[0], from_pos[1])
        target = new_board.get_piece_at(to_pos[0], to_pos[1])

        if target:
            target.captured = True
            new_board.pieces.remove(target)

        # Update piece position
        new_board.grid[from_pos[0]][from_pos[1]] = None
        new_board.grid[to_pos[0]][to_pos[1]] = piece
        piece.position = to_pos

        # Switch player
        next_player = "black" if current_player == "red" else "red"

        return new_board, next_player

    def is_game_over(self, board):
        """
        Check if the game is over

        Args:
            board: The current board state

        Returns:
            is_over: Boolean indicating if the game is over
            winner: The winner ('red', 'black', or None for draw)
        """
        # Check if any general is captured
        red_general_found = False
        black_general_found = False

        for piece in board.pieces:
            if isinstance(piece, General) and not piece.captured:
                if piece.color == "red":
                    red_general_found = True
                else:
                    black_general_found = True

        # If a general is missing, the game is over
        if not red_general_found:
            return True, "black"
        if not black_general_found:
            return True, "red"

        # Check for checkmate or stalemate
        red_has_moves = len(self.get_legal_actions(board, "red")) > 0
        black_has_moves = len(self.get_legal_actions(board, "black")) > 0

        # If a player has no legal moves, it's either checkmate or stalemate
        if not red_has_moves:
            # Check if red is in check
            red_in_check = board.is_check("red")
            if red_in_check:
                return True, "black"  # Checkmate
            else:
                return True, None  # Stalemate (draw)

        if not black_has_moves:
            # Check if black is in check
            black_in_check = board.is_check("black")
            if black_in_check:
                return True, "red"  # Checkmate
            else:
                return True, None  # Stalemate (draw)

        # Game is not over
        return False, None

    def get_game_ended(self, board, current_player):
        """
        Get the game result from the perspective of the current player

        Args:
            board: The current board state
            current_player: The current player ('red' or 'black')

        Returns:
            result: 1 for win, -1 for loss, 0 for ongoing game or draw
        """
        game_over, winner = self.is_game_over(board)

        if not game_over:
            return 0  # Game is not over

        if winner is None:
            return 0  # Draw

        return 1 if winner == current_player else -1

    def get_symmetries(self, board, pi):
        """
        Get symmetrical equivalents of the board state and policy

        In Xiangqi, unlike chess, there are no valid board symmetries due to
        the asymmetrical nature of the palace and river, so we just return the original

        Args:
            board: The board state
            pi: The policy vector

        Returns:
            List of (board, pi) tuples representing symmetrical states
        """
        # Xiangqi doesn't have valid symmetries, so just return the original
        return [(board, pi)]

    def to_string(self, board):
        """
        Convert the board to a string representation

        Args:
            board: The board state

        Returns:
            str: String representation of the board
        """
        result = []
        for y in range(10):
            row = []
            for x in range(9):
                piece = board.get_piece_at(x, y)
                if piece:
                    if piece.color == "red":
                        if isinstance(piece, General):
                            char = "G"
                        elif isinstance(piece, Advisor):
                            char = "A"
                        elif isinstance(piece, Elephant):
                            char = "E"
                        elif isinstance(piece, Horse):
                            char = "H"
                        elif isinstance(piece, Chariot):
                            char = "R"
                        elif isinstance(piece, Cannon):
                            char = "C"
                        elif isinstance(piece, Soldier):
                            char = "S"
                        else:
                            char = "."
                    else:  # Black
                        if isinstance(piece, General):
                            char = "g"
                        elif isinstance(piece, Advisor):
                            char = "a"
                        elif isinstance(piece, Elephant):
                            char = "e"
                        elif isinstance(piece, Horse):
                            char = "h"
                        elif isinstance(piece, Chariot):
                            char = "r"
                        elif isinstance(piece, Cannon):
                            char = "c"
                        elif isinstance(piece, Soldier):
                            char = "s"
                        else:
                            char = "."
                else:
                    char = "."
                row.append(char)
            result.append(" ".join(row))
        return "\n".join(result)

    def action_to_move(self, action_idx, board_size=(10, 9)):
        """
        Convert a flattened action index to a move (from_pos, to_pos)

        Args:
            action_idx: Integer index of the action
            board_size: Tuple (height, width) of the board

        Returns:
            tuple: ((from_x, from_y), (to_x, to_y)) representing the move
        """
        return utils_action_to_move(action_idx, board_size)

    def move_to_action(self, from_pos, to_pos, board_size=(10, 9)):
        """
        Convert a move (from_pos, to_pos) to a flattened action index

        Args:
            from_pos: (x, y) position of the piece to move
            to_pos: (x, y) position to move to
            board_size: Tuple (height, width) of the board

        Returns:
            int: Flattened action index
        """
        return utils_move_to_action(from_pos, to_pos, board_size)
