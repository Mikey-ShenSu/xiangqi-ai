#!/usr/bin/env python3
# Xiangqi (Chinese Chess) Game Implementation


class Piece:
    """Base class for all Xiangqi pieces."""

    def __init__(self, color, position):
        """
        Initialize a piece.

        Args:
            color: 'red' or 'black' representing the piece's color
            position: tuple (x, y) representing the piece's position on the board
        """
        self.color = color
        self.position = position
        self.captured = False

    def get_valid_moves(self, board):
        """
        Get all valid moves for this piece on the given board.

        Args:
            board: the game board

        Returns:
            A list of valid positions (x, y) this piece can move to
        """
        raise NotImplementedError("Subclasses must implement this method")

    def __str__(self):
        return f"{self.color} {self.__class__.__name__}"


class General(Piece):
    """
    The General (also known as King) in Xiangqi.

    Rules:
    - Moves one point horizontally or vertically
    - Cannot leave the palace (3x3 grid)
    - Cannot face the opponent's General directly with no pieces in between
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Define possible moves (one step in each direction)
        possible_moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for move in possible_moves:
            nx, ny = move

            # Check if the move is within the palace
            if self.is_in_palace(nx, ny) and self.is_valid_position(board, nx, ny):
                valid_moves.append(move)

        return valid_moves

    def is_in_palace(self, x, y):
        """Check if the given position is within the palace."""
        if self.color == "red":
            return 3 <= x <= 5 and 0 <= y <= 2
        else:  # black
            return 3 <= x <= 5 and 7 <= y <= 9

    def is_valid_position(self, board, x, y):
        """Check if the piece can move to the given position."""
        # Check if position is on the board
        if not (0 <= x <= 8 and 0 <= y <= 9):
            return False

        # Check if position is occupied by a friendly piece
        piece_at_position = board.get_piece_at(x, y)
        if piece_at_position and piece_at_position.color == self.color:
            return False

        return True


class Advisor(Piece):
    """
    The Advisor in Xiangqi.

    Rules:
    - Moves one point diagonally
    - Cannot leave the palace (3x3 grid)
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Define possible diagonal moves
        possible_moves = [
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        ]

        for move in possible_moves:
            nx, ny = move

            # Check if the move is within the palace
            if self.is_in_palace(nx, ny) and self.is_valid_position(board, nx, ny):
                valid_moves.append(move)

        return valid_moves

    def is_in_palace(self, x, y):
        """Check if the given position is within the palace."""
        if self.color == "red":
            return 3 <= x <= 5 and 0 <= y <= 2
        else:  # black
            return 3 <= x <= 5 and 7 <= y <= 9

    def is_valid_position(self, board, x, y):
        """Check if the piece can move to the given position."""
        # Check if position is on the board
        if not (0 <= x <= 8 and 0 <= y <= 9):
            return False

        # Check if position is occupied by a friendly piece
        piece_at_position = board.get_piece_at(x, y)
        if piece_at_position and piece_at_position.color == self.color:
            return False

        return True


class Elephant(Piece):
    """
    The Elephant in Xiangqi.

    Rules:
    - Moves exactly two points diagonally
    - Cannot cross the river (stay on its own side)
    - Cannot jump over pieces
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Define possible diagonal moves (2 steps away)
        possible_moves = [
            (x + 2, y + 2, x + 1, y + 1),  # Move SE, check NE blocking point
            (x + 2, y - 2, x + 1, y - 1),  # Move NE, check SE blocking point
            (x - 2, y + 2, x - 1, y + 1),  # Move SW, check NW blocking point
            (x - 2, y - 2, x - 1, y - 1),  # Move NW, check SW blocking point
        ]

        for move in possible_moves:
            nx, ny, bx, by = move  # nx,ny = destination, bx,by = blocking point

            # Check if the move is valid
            if (
                self.is_on_own_side(nx, ny)
                and self.is_valid_position(board, nx, ny)
                and board.get_piece_at(bx, by) is None
            ):  # No piece blocking the path
                valid_moves.append((nx, ny))

        return valid_moves

    def is_on_own_side(self, x, y):
        """Check if the given position is on the piece's own side of the river."""
        if not (0 <= x <= 8 and 0 <= y <= 9):
            return False

        if self.color == "red":
            return y <= 4  # Red side is bottom (0-4)
        else:  # black
            return y >= 5  # Black side is top (5-9)

    def is_valid_position(self, board, x, y):
        """Check if the piece can move to the given position."""
        # Check if position is on the board
        if not (0 <= x <= 8 and 0 <= y <= 9):
            return False

        # Check if position is occupied by a friendly piece
        piece_at_position = board.get_piece_at(x, y)
        if piece_at_position and piece_at_position.color == self.color:
            return False

        return True


class Horse(Piece):
    """
    The Horse in Xiangqi.

    Rules:
    - Moves one point orthogonally followed by one point diagonally outward
    - Cannot jump over pieces (moves are blocked if a piece is at the orthogonal step)
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Define possible moves with their blocking points
        # (destination_x, destination_y, blocking_x, blocking_y)
        possible_moves = [
            (x + 1, y + 2, x, y + 1),  # Move up-right
            (x + 2, y + 1, x + 1, y),  # Move right-up
            (x + 2, y - 1, x + 1, y),  # Move right-down
            (x + 1, y - 2, x, y - 1),  # Move down-right
            (x - 1, y - 2, x, y - 1),  # Move down-left
            (x - 2, y - 1, x - 1, y),  # Move left-down
            (x - 2, y + 1, x - 1, y),  # Move left-up
            (x - 1, y + 2, x, y + 1),  # Move up-left
        ]

        for move in possible_moves:
            nx, ny, bx, by = move  # nx,ny = destination, bx,by = blocking point

            # Check if blocking point is clear and destination is valid
            if (
                self.is_valid_position(board, nx, ny)
                and board.get_piece_at(bx, by) is None
            ):  # No piece blocking the path
                valid_moves.append((nx, ny))

        return valid_moves

    def is_valid_position(self, board, x, y):
        """Check if the piece can move to the given position."""
        # Check if position is on the board
        if not (0 <= x <= 8 and 0 <= y <= 9):
            return False

        # Check if position is occupied by a friendly piece
        piece_at_position = board.get_piece_at(x, y)
        if piece_at_position and piece_at_position.color == self.color:
            return False

        return True


class Chariot(Piece):
    """
    The Chariot (Rook) in Xiangqi.

    Rules:
    - Moves any number of points horizontally or vertically
    - Cannot jump over pieces
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Check moves in all four directions (up, right, down, left)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            while 0 <= nx <= 8 and 0 <= ny <= 9:
                piece_at_position = board.get_piece_at(nx, ny)

                if piece_at_position is None:
                    # Empty square, can move here
                    valid_moves.append((nx, ny))
                    nx, ny = nx + dx, ny + dy
                elif piece_at_position.color != self.color:
                    # Enemy piece, can capture and stop
                    valid_moves.append((nx, ny))
                    break
                else:
                    # Friendly piece, cannot move here and stop
                    break

        return valid_moves


class Cannon(Piece):
    """
    The Cannon in Xiangqi.

    Rules:
    - Moves like the Chariot (any number of points horizontally or vertically)
    - Must jump over exactly one piece (of either color) to capture
    - Cannot jump over pieces when not capturing
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Check moves in all four directions (up, right, down, left)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for dx, dy in directions:
            # First find all normal moves (without jumping)
            nx, ny = x + dx, y + dy
            while 0 <= nx <= 8 and 0 <= ny <= 9:
                piece_at_position = board.get_piece_at(nx, ny)

                if piece_at_position is None:
                    # Empty square, can move here
                    valid_moves.append((nx, ny))
                    nx, ny = nx + dx, ny + dy
                else:
                    # Found a piece, can't move here but might be able to jump over to capture
                    # Check for potential capture after jumping
                    nx, ny = nx + dx, ny + dy
                    while 0 <= nx <= 8 and 0 <= ny <= 9:
                        jump_target = board.get_piece_at(nx, ny)

                        if jump_target is None:
                            # Empty square after jumping, continue looking
                            nx, ny = nx + dx, ny + dy
                        elif jump_target.color != self.color:
                            # Enemy piece after jumping, can capture
                            valid_moves.append((nx, ny))
                            break
                        else:
                            # Friendly piece after jumping, cannot capture
                            break
                    break

        return valid_moves


class Soldier(Piece):
    """
    The Soldier (Pawn) in Xiangqi.

    Rules:
    - Before crossing the river: moves one point forward only
    - After crossing the river: can move one point forward or horizontally
    - Never moves backward
    """

    def get_valid_moves(self, board):
        valid_moves = []
        x, y = self.position

        # Define forward direction based on color
        forward = 1 if self.color == "red" else -1

        # Forward move
        nx, ny = x, y + forward
        if self.is_valid_position(board, nx, ny):
            valid_moves.append((nx, ny))

        # Check if the soldier has crossed the river
        has_crossed_river = (self.color == "red" and y > 4) or (
            self.color == "black" and y < 5
        )

        if has_crossed_river:
            # Horizontal moves (left and right)
            for nx in [x - 1, x + 1]:
                if self.is_valid_position(board, nx, y):
                    valid_moves.append((nx, y))

        return valid_moves

    def is_valid_position(self, board, x, y):
        """Check if the piece can move to the given position."""
        # Check if position is on the board
        if not (0 <= x <= 8 and 0 <= y <= 9):
            return False

        # Check if position is occupied by a friendly piece
        piece_at_position = board.get_piece_at(x, y)
        if piece_at_position and piece_at_position.color == self.color:
            return False

        return True


class Board:
    """Represents the Xiangqi board."""

    def __init__(self):
        """Initialize an empty 9x10 board."""
        self.grid = [[None for _ in range(10)] for _ in range(9)]
        self.pieces = []
        self.setup_board()

    def setup_board(self):
        """Set up the initial board position with all pieces."""
        # Red pieces (bottom side)
        # Chariot (Rook)
        self.add_piece(Chariot("red", (0, 0)))
        self.add_piece(Chariot("red", (8, 0)))

        # Horse (Knight)
        self.add_piece(Horse("red", (1, 0)))
        self.add_piece(Horse("red", (7, 0)))

        # Elephant
        self.add_piece(Elephant("red", (2, 0)))
        self.add_piece(Elephant("red", (6, 0)))

        # Advisor
        self.add_piece(Advisor("red", (3, 0)))
        self.add_piece(Advisor("red", (5, 0)))

        # General (King)
        self.add_piece(General("red", (4, 0)))

        # Cannon
        self.add_piece(Cannon("red", (1, 2)))
        self.add_piece(Cannon("red", (7, 2)))

        # Soldier (Pawn)
        for i in range(5):
            self.add_piece(Soldier("red", (i * 2, 3)))

        # Black pieces (top side)
        # Chariot (Rook)
        self.add_piece(Chariot("black", (0, 9)))
        self.add_piece(Chariot("black", (8, 9)))

        # Horse (Knight)
        self.add_piece(Horse("black", (1, 9)))
        self.add_piece(Horse("black", (7, 9)))

        # Elephant
        self.add_piece(Elephant("black", (2, 9)))
        self.add_piece(Elephant("black", (6, 9)))

        # Advisor
        self.add_piece(Advisor("black", (3, 9)))
        self.add_piece(Advisor("black", (5, 9)))

        # General (King)
        self.add_piece(General("black", (4, 9)))

        # Cannon
        self.add_piece(Cannon("black", (1, 7)))
        self.add_piece(Cannon("black", (7, 7)))

        # Soldier (Pawn)
        for i in range(5):
            self.add_piece(Soldier("black", (i * 2, 6)))

    def add_piece(self, piece):
        """Add a piece to the board."""
        x, y = piece.position
        self.grid[x][y] = piece
        self.pieces.append(piece)

    def move_piece(self, from_pos, to_pos):
        """
        Move a piece from one position to another.

        Args:
            from_pos: tuple (x, y) representing the current position
            to_pos: tuple (x, y) representing the target position

        Returns:
            True if the move was valid and executed, False otherwise
        """
        fx, fy = from_pos
        tx, ty = to_pos

        piece = self.get_piece_at(fx, fy)
        if piece is None:
            return False

        # Check if the move is valid for this piece
        valid_moves = piece.get_valid_moves(self)
        if (tx, ty) not in valid_moves:
            return False

        # Check if there's a piece at the target position (capture)
        target_piece = self.get_piece_at(tx, ty)
        if target_piece:
            target_piece.captured = True
            self.pieces.remove(target_piece)

        # Move the piece
        self.grid[fx][fy] = None
        self.grid[tx][ty] = piece
        piece.position = (tx, ty)

        return True

    def get_piece_at(self, x, y):
        """Get the piece at the given position, or None if empty."""
        if 0 <= x <= 8 and 0 <= y <= 9:
            return self.grid[x][y]
        return None

    def is_check(self, color):
        """
        Determine if the general of the given color is in check.

        Args:
            color: 'red' or 'black'

        Returns:
            True if the general is in check, False otherwise
        """
        # Find the general
        general = None
        for piece in self.pieces:
            if isinstance(piece, General) and piece.color == color:
                general = piece
                break

        if not general:
            return False  # General not found (should not happen in a valid game)

        # Check if any opponent's piece can capture the general
        for piece in self.pieces:
            if piece.color != color:  # Opponent's piece
                valid_moves = piece.get_valid_moves(self)
                if general.position in valid_moves:
                    return True

        return False

    def is_checkmate(self, color):
        """
        Determine if the general of the given color is in checkmate.

        Args:
            color: 'red' or 'black'

        Returns:
            True if the general is in checkmate, False otherwise
        """
        if not self.is_check(color):
            return False

        # Check if any move can get out of check
        for piece in self.pieces:
            if piece.color == color:
                valid_moves = piece.get_valid_moves(self)
                original_pos = piece.position

                for move in valid_moves:
                    # Try the move
                    tx, ty = move
                    fx, fy = original_pos
                    target_piece = self.get_piece_at(tx, ty)
                    was_capture = False

                    if target_piece:
                        was_capture = True
                        target_piece.captured = True
                        self.pieces.remove(target_piece)

                    # Move the piece temporarily
                    self.grid[fx][fy] = None
                    self.grid[tx][ty] = piece
                    piece.position = (tx, ty)

                    # Check if still in check
                    still_in_check = self.is_check(color)

                    # Undo the move
                    self.grid[tx][ty] = target_piece
                    self.grid[fx][fy] = piece
                    piece.position = original_pos

                    if was_capture:
                        target_piece.captured = False
                        self.pieces.append(target_piece)

                    if not still_in_check:
                        return False  # Found a move that gets out of check

        return True  # No move can get out of check

    def display(self):
        """Print the current board state to the console."""
        # Unicode representations for pieces
        piece_symbols = {
            ("red", "General"): "帥",
            ("red", "Advisor"): "仕",
            ("red", "Elephant"): "相",
            ("red", "Horse"): "傌",
            ("red", "Chariot"): "俥",
            ("red", "Cannon"): "炮",
            ("red", "Soldier"): "兵",
            ("black", "General"): "將",
            ("black", "Advisor"): "士",
            ("black", "Elephant"): "象",
            ("black", "Horse"): "馬",
            ("black", "Chariot"): "車",
            ("black", "Cannon"): "砲",
            ("black", "Soldier"): "卒",
        }

        print("  0 1 2 3 4 5 6 7 8")
        print(" ┌─┬─┬─┬─┬─┬─┬─┬─┐")

        for y in range(9, -1, -1):
            print(f"{y}│", end="")
            for x in range(9):
                piece = self.get_piece_at(x, y)
                if piece:
                    symbol = piece_symbols.get(
                        (piece.color, piece.__class__.__name__), "?"
                    )
                    print(symbol, end="")
                else:
                    if y == 4 or y == 5:  # River
                        if x % 2 == 0:
                            print("~", end="")
                        else:
                            print("~", end="")
                    else:
                        print("·", end="")

                if x < 8:
                    print("─", end="")

            print("│")

            if y > 0:
                if y == 5:  # River
                    print(" ├─┼─┼─┼─┼─┼─┼─┼─┤")
                else:
                    print(" ├─┼─┼─┼─┼─┼─┼─┼─┤")

        print(" └─┴─┴─┴─┴─┴─┴─┴─┘")


class Game:
    """Controls the game flow and state."""

    def __init__(self):
        self.board = Board()
        self.current_player = "red"  # Red moves first
        self.game_over = False
        self.winner = None

    def switch_player(self):
        """Switch the current player."""
        self.current_player = "black" if self.current_player == "red" else "red"

    def make_move(self, from_pos, to_pos):
        """
        Attempt to make a move.

        Args:
            from_pos: tuple (x, y) representing the current position
            to_pos: tuple (x, y) representing the target position

        Returns:
            True if the move was valid and executed, False otherwise
        """
        fx, fy = from_pos
        piece = self.board.get_piece_at(fx, fy)

        # Check if there's a piece at the source position
        if piece is None:
            print("No piece at that position.")
            return False

        # Check if it's the current player's piece
        if piece.color != self.current_player:
            print(
                f"It's {self.current_player}'s turn, but you selected a {piece.color} piece."
            )
            return False

        # Try to move the piece
        if self.board.move_piece(from_pos, to_pos):
            # Check for checkmate
            opponent = "black" if self.current_player == "red" else "red"
            if self.board.is_checkmate(opponent):
                self.game_over = True
                self.winner = self.current_player
                print(f"{self.current_player.capitalize()} wins by checkmate!")
            elif self.board.is_check(opponent):
                print(f"{opponent.capitalize()} is in check!")

            # Switch players if the game is not over
            if not self.game_over:
                self.switch_player()

            return True
        else:
            print("Invalid move.")
            return False

    def play(self):
        """Start the game and play until game over."""
        print("Welcome to Xiangqi (Chinese Chess)!")
        print("Enter moves as 'x1 y1 x2 y2' to move from (x1,y1) to (x2,y2)")
        print("Enter 'quit' to exit the game.")

        while not self.game_over:
            self.board.display()
            print(f"\n{self.current_player.capitalize()}'s turn.")

            user_input = input("Enter your move: ")

            if user_input.lower() == "quit":
                print("Game ended by player.")
                break

            try:
                x1, y1, x2, y2 = map(int, user_input.split())
                self.make_move((x1, y1), (x2, y2))
            except ValueError:
                print("Invalid input. Please enter move as 'x1 y1 x2 y2'.")

        if self.winner:
            print(f"\nGame over! {self.winner.capitalize()} wins!")

        print("Thank you for playing!")


if __name__ == "__main__":
    game = Game()
    game.play()
