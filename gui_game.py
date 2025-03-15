#!/usr/bin/env python3
# Xiangqi (Chinese Chess) GUI Implementation
import pygame
import sys
import logging
from main import (
    Game,
    Board,
    General,
    Advisor,
    Elephant,
    Horse,
    Chariot,
    Cannon,
    Soldier,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("xiangqi.log"), logging.StreamHandler()],
)
logger = logging.getLogger("xiangqi")

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 900
CELL_SIZE = 70
PIECE_SIZE = 60
# Calculate board dimensions
BOARD_WIDTH = 8 * CELL_SIZE
BOARD_HEIGHT = 9 * CELL_SIZE
# Calculate offsets to center the board
BOARD_OFFSET_X = (SCREEN_WIDTH - BOARD_WIDTH) // 2
# Leave space at the top for UI text
UI_MARGIN_TOP = 8
STATUS_HEIGHT = 30
CHECK_HEIGHT = 30
BOARD_OFFSET_Y = UI_MARGIN_TOP + STATUS_HEIGHT + CHECK_HEIGHT + 20

# Colors
BACKGROUND_COLOR = (235, 209, 166)  # Light wooden color
BOARD_COLOR = (220, 179, 92)  # Darker wooden color
LINE_COLOR = (0, 0, 0)  # Black
HIGHLIGHT_COLOR = (255, 255, 0, 128)  # Transparent yellow
VALID_MOVE_COLOR = (0, 255, 0, 128)  # Transparent green
RED_PLAYER_COLOR = (255, 0, 0)
BLACK_PLAYER_COLOR = (0, 0, 0)
TEXT_COLOR = (50, 50, 50)


# Create a function to get a font that supports Chinese characters
def get_chinese_font(size, bold=False):
    """Get a font that can render Chinese characters properly"""
    # Try multiple font options in order of preference
    font_options = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "STHeiti",
        "Heiti SC",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Hiragino Sans GB",
        "Songti SC",
        "SimSun",
        "WenQuanYi Micro Hei",
    ]

    # Try system fonts that support Chinese
    for font_name in font_options:
        try:
            if pygame.font.match_font(font_name):
                return pygame.font.SysFont(font_name, size, bold=bold)
        except:
            pass

    # Fallback to default font
    return pygame.font.SysFont("Arial", size, bold=bold)


class XiangqiGUI:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("象棋 (中国象棋)")

        self.game = Game()
        self.board = self.game.board
        logger.info("Xiangqi game initialized")

        # Log initial board state
        logger.debug("Initial board state:")
        for piece in self.board.pieces:
            if not piece.captured:
                logger.debug(
                    f"  {piece.color} {piece.__class__.__name__} at {piece.position}"
                )

        # Patch the Game.make_move method to add additional logging
        original_make_move = self.game.make_move

        def make_move_with_logging(from_pos, to_pos):
            logger.debug(f"Attempting move from {from_pos} to {to_pos}")

            # Check for general facing
            red_general = None
            black_general = None
            for piece in self.board.pieces:
                if isinstance(piece, General) and not piece.captured:
                    if piece.color == "red":
                        red_general = piece
                    else:
                        black_general = piece

            if red_general and black_general:
                if red_general.position[0] == black_general.position[0]:
                    # Check if there are pieces between them
                    min_y = min(red_general.position[1], black_general.position[1])
                    max_y = max(red_general.position[1], black_general.position[1])
                    has_pieces_between = False
                    for piece in self.board.pieces:
                        if (
                            not piece.captured
                            and piece.position[0] == red_general.position[0]
                            and min_y < piece.position[1] < max_y
                        ):
                            has_pieces_between = True
                            break

                    if not has_pieces_between:
                        logger.warning(
                            "Generals are facing each other with no pieces between!"
                        )

            # Try the move and see if it's successful
            success = original_make_move(from_pos, to_pos)

            if not success:
                logger.warning(f"Move failed from {from_pos} to {to_pos}")

            return success

        # Replace the method with our logging version
        self.game.make_move = make_move_with_logging

        # Chinese piece names and labels
        self.piece_names = {
            ("red", "General"): "帅",
            ("red", "Advisor"): "仕",
            ("red", "Elephant"): "相",
            ("red", "Horse"): "马",
            ("red", "Chariot"): "车",
            ("red", "Cannon"): "炮",
            ("red", "Soldier"): "兵",
            ("black", "General"): "将",
            ("black", "Advisor"): "士",
            ("black", "Elephant"): "象",
            ("black", "Horse"): "马",
            ("black", "Chariot"): "车",
            ("black", "Cannon"): "炮",
            ("black", "Soldier"): "卒",
        }

        # Chinese UI text translations in simplified Chinese
        self.ui_text = {
            "red_turn": "红方回合",
            "black_turn": "黑方回合",
            "red_wins": "红方胜利！",
            "black_wins": "黑方胜利！",
            "check": "将军！",
            "game_over": "游戏结束！",
        }

        # Font for text
        self.font = get_chinese_font(24)
        self.piece_font = get_chinese_font(30, bold=True)
        self.river_font = get_chinese_font(40)

        # Load assets
        self.load_images()

        # Game state
        self.selected_piece = None
        self.selected_position = None
        self.valid_moves = []
        self.clock = pygame.time.Clock()

    def load_images(self):
        """Load piece images and scale them"""
        self.images = {}

        # Dictionary mapping piece class names to image file prefixes
        piece_images = {
            "General": "general",
            "Advisor": "advisor",
            "Elephant": "elephant",
            "Horse": "horse",
            "Chariot": "chariot",
            "Cannon": "cannon",
            "Soldier": "soldier",
        }

        # Create placeholder pieces with simple circles
        for piece_class, image_prefix in piece_images.items():
            for color in ["red", "black"]:
                # Create a surface for the piece
                piece_surface = pygame.Surface(
                    (PIECE_SIZE, PIECE_SIZE), pygame.SRCALPHA
                )

                # Draw circle background
                bg_color = (240, 210, 150) if color == "red" else (50, 50, 50)
                pygame.draw.circle(
                    piece_surface,
                    bg_color,
                    (PIECE_SIZE // 2, PIECE_SIZE // 2),
                    PIECE_SIZE // 2,
                )

                # Draw border
                pygame.draw.circle(
                    piece_surface,
                    (0, 0, 0),
                    (PIECE_SIZE // 2, PIECE_SIZE // 2),
                    PIECE_SIZE // 2,
                    2,
                )

                # Add Chinese character label
                piece_label = self.piece_names[(color, piece_class)]
                if color == "red":
                    text_color = (200, 0, 0)
                else:
                    text_color = (255, 255, 255)

                # Use the Chinese-compatible font
                text = self.piece_font.render(piece_label, True, text_color)
                text_rect = text.get_rect(center=(PIECE_SIZE // 2, PIECE_SIZE // 2))
                piece_surface.blit(text, text_rect)

                # Store the image
                self.images[(color, piece_class)] = piece_surface

    def draw_board(self):
        """Draw the Xiangqi board"""
        # Fill background
        self.screen.fill(BACKGROUND_COLOR)

        # Draw turn indicator and game status in simplified Chinese
        # Position at the top of the board with margin
        status_text = (
            self.ui_text["red_turn"]
            if self.game.current_player == "red"
            else self.ui_text["black_turn"]
        )
        if self.game.game_over:
            winner_text = (
                self.ui_text["red_wins"]
                if self.game.winner == "red"
                else self.ui_text["black_wins"]
            )
            status_text = f"{self.ui_text['game_over']} {winner_text}"

        status_surface = self.font.render(status_text, True, TEXT_COLOR)
        self.screen.blit(
            status_surface,
            (
                SCREEN_WIDTH // 2 - status_surface.get_width() // 2,
                UI_MARGIN_TOP,
            ),
        )

        # Check indicator in Chinese
        check_y_pos = UI_MARGIN_TOP + STATUS_HEIGHT
        if not self.game.game_over:
            opponent = "black" if self.game.current_player == "red" else "red"
            if self.board.is_check(opponent):
                check_text = self.ui_text["check"]
                check_surface = self.font.render(check_text, True, RED_PLAYER_COLOR)
                self.screen.blit(
                    check_surface,
                    (
                        SCREEN_WIDTH // 2 - check_surface.get_width() // 2,
                        check_y_pos,
                    ),
                )

        # Draw the board grid (9x10)
        board_width = 8 * CELL_SIZE
        board_height = 9 * CELL_SIZE

        # Draw the board background
        board_rect = pygame.Rect(
            BOARD_OFFSET_X, BOARD_OFFSET_Y, board_width, board_height
        )
        pygame.draw.rect(self.screen, BOARD_COLOR, board_rect)

        # Draw horizontal lines
        for y in range(10):
            start_pos = (BOARD_OFFSET_X, BOARD_OFFSET_Y + y * CELL_SIZE)
            end_pos = (BOARD_OFFSET_X + board_width, BOARD_OFFSET_Y + y * CELL_SIZE)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos, end_pos, 2)

        # Draw vertical lines
        for x in range(9):
            start_pos = (BOARD_OFFSET_X + x * CELL_SIZE, BOARD_OFFSET_Y)
            end_pos = (BOARD_OFFSET_X + x * CELL_SIZE, BOARD_OFFSET_Y + board_height)
            pygame.draw.line(self.screen, LINE_COLOR, start_pos, end_pos, 2)

        # Draw the palace diagonals
        # Red palace (bottom)
        pygame.draw.line(
            self.screen,
            LINE_COLOR,
            (BOARD_OFFSET_X + 3 * CELL_SIZE, BOARD_OFFSET_Y + 7 * CELL_SIZE),
            (BOARD_OFFSET_X + 5 * CELL_SIZE, BOARD_OFFSET_Y + 9 * CELL_SIZE),
            2,
        )
        pygame.draw.line(
            self.screen,
            LINE_COLOR,
            (BOARD_OFFSET_X + 5 * CELL_SIZE, BOARD_OFFSET_Y + 7 * CELL_SIZE),
            (BOARD_OFFSET_X + 3 * CELL_SIZE, BOARD_OFFSET_Y + 9 * CELL_SIZE),
            2,
        )

        # Black palace (top)
        pygame.draw.line(
            self.screen,
            LINE_COLOR,
            (BOARD_OFFSET_X + 3 * CELL_SIZE, BOARD_OFFSET_Y),
            (BOARD_OFFSET_X + 5 * CELL_SIZE, BOARD_OFFSET_Y + 2 * CELL_SIZE),
            2,
        )
        pygame.draw.line(
            self.screen,
            LINE_COLOR,
            (BOARD_OFFSET_X + 5 * CELL_SIZE, BOARD_OFFSET_Y),
            (BOARD_OFFSET_X + 3 * CELL_SIZE, BOARD_OFFSET_Y + 2 * CELL_SIZE),
            2,
        )

        # Draw the river
        for x in range(0, 8, 2):
            river_text = self.river_font.render("楚", True, LINE_COLOR)
            self.screen.blit(
                river_text,
                (
                    BOARD_OFFSET_X
                    + (x + 0.5) * CELL_SIZE
                    - river_text.get_width() // 2,
                    BOARD_OFFSET_Y + 4.5 * CELL_SIZE - river_text.get_height() // 2,
                ),
            )
            river_text = self.river_font.render("河", True, LINE_COLOR)
            self.screen.blit(
                river_text,
                (
                    BOARD_OFFSET_X
                    + (x + 1.5) * CELL_SIZE
                    - river_text.get_width() // 2,
                    BOARD_OFFSET_Y + 4.5 * CELL_SIZE - river_text.get_height() // 2,
                ),
            )

    def draw_pieces(self):
        """Draw all pieces on the board"""
        for piece in self.board.pieces:
            if not piece.captured:
                x, y = piece.position
                # Convert board coordinates to screen coordinates (invert y)
                # Position pieces at intersections instead of cell centers
                screen_x = BOARD_OFFSET_X + x * CELL_SIZE - PIECE_SIZE // 2
                screen_y = BOARD_OFFSET_Y + (9 - y) * CELL_SIZE - PIECE_SIZE // 2

                piece_image = self.images[(piece.color, piece.__class__.__name__)]
                self.screen.blit(piece_image, (screen_x, screen_y))

    def draw_selected_highlight(self):
        """Highlight the selected piece and valid moves"""
        if self.selected_position:
            x, y = self.selected_position
            # Convert board coordinates to screen coordinates (invert y)
            screen_x = BOARD_OFFSET_X + x * CELL_SIZE
            screen_y = BOARD_OFFSET_Y + (9 - y) * CELL_SIZE

            # Draw selection highlight
            highlight_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(
                highlight_surface, HIGHLIGHT_COLOR, (0, 0, CELL_SIZE, CELL_SIZE)
            )
            self.screen.blit(
                highlight_surface,
                (screen_x - CELL_SIZE // 2, screen_y - CELL_SIZE // 2),
            )

        # Draw valid moves
        for move_x, move_y in self.valid_moves:
            # Convert board coordinates to screen coordinates (invert y)
            screen_x = BOARD_OFFSET_X + move_x * CELL_SIZE
            screen_y = BOARD_OFFSET_Y + (9 - move_y) * CELL_SIZE

            # Draw move indicator
            move_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(
                move_surface, VALID_MOVE_COLOR, (0, 0, CELL_SIZE, CELL_SIZE)
            )
            self.screen.blit(
                move_surface, (screen_x - CELL_SIZE // 2, screen_y - CELL_SIZE // 2)
            )

    def board_to_screen(self, pos):
        """Convert board position (x, y) to screen coordinates"""
        x, y = pos
        # Convert to intersection coordinates
        screen_x = BOARD_OFFSET_X + x * CELL_SIZE
        screen_y = BOARD_OFFSET_Y + (9 - y) * CELL_SIZE
        return (screen_x, screen_y)

    def screen_to_board(self, pos):
        """Convert screen coordinates to board position (x, y)"""
        screen_x, screen_y = pos

        # Check if click is within the board area with some margin for intersections
        if (
            BOARD_OFFSET_X - CELL_SIZE // 2
            <= screen_x
            <= BOARD_OFFSET_X + 8 * CELL_SIZE + CELL_SIZE // 2
            and BOARD_OFFSET_Y - CELL_SIZE // 2
            <= screen_y
            <= BOARD_OFFSET_Y + 9 * CELL_SIZE + CELL_SIZE // 2
        ):
            # Find the closest intersection
            board_x = round((screen_x - BOARD_OFFSET_X) / CELL_SIZE)
            # Invert y coordinate since board's origin is bottom-left
            board_y = 9 - round((screen_y - BOARD_OFFSET_Y) / CELL_SIZE)

            if 0 <= board_x <= 8 and 0 <= board_y <= 9:
                return (board_x, board_y)

        return None

    def handle_click(self, pos):
        """Handle mouse click on the board"""
        board_pos = self.screen_to_board(pos)

        if not board_pos:
            logger.debug("Click outside valid board area")
            return

        if self.game.game_over:
            logger.debug("Game is already over, ignoring click")
            return

        logger.debug(f"Click at board position: {board_pos}")

        if self.selected_piece:
            logger.debug(
                f"Selected piece: {self.selected_piece.__class__.__name__} at {self.selected_position}"
            )
            # If a piece is already selected, try to move it
            if board_pos in self.valid_moves:
                logger.info(
                    f"Moving {self.selected_piece.color} {self.selected_piece.__class__.__name__} from {self.selected_position} to {board_pos}"
                )

                # Check if we're about to capture a General
                target_piece = self.board.get_piece_at(board_pos[0], board_pos[1])
                direct_general_capture = target_piece and isinstance(
                    target_piece, General
                )
                if direct_general_capture:
                    logger.info(
                        f"Direct General capture attempt: {self.selected_piece.color} {self.selected_piece.__class__.__name__} capturing {target_piece.color} General"
                    )

                # Log the state before move
                logger.debug("Board state before move:")
                for piece in self.board.pieces:
                    if not piece.captured:
                        logger.debug(
                            f"  {piece.color} {piece.__class__.__name__} at {piece.position}"
                        )

                # Make the move
                success = self.game.make_move(self.selected_position, board_pos)

                if success:
                    logger.info(
                        f"Move successful, now {self.game.current_player}'s turn"
                    )

                    # Log the state after move
                    logger.debug("Board state after move:")
                    has_black_general = False
                    has_red_general = False
                    for piece in self.board.pieces:
                        if not piece.captured:
                            logger.debug(
                                f"  {piece.color} {piece.__class__.__name__} at {piece.position}"
                            )
                            if isinstance(piece, General):
                                if piece.color == "red":
                                    has_red_general = True
                                else:
                                    has_black_general = True
                        elif isinstance(piece, General):
                            logger.info(f"{piece.color} General has been captured!")

                    # Check for general capture - first priority
                    # Direct check for missing generals is more reliable than checking captured pieces
                    if (
                        direct_general_capture
                        or not has_black_general
                        or not has_red_general
                    ):
                        logger.info("A General has been captured! Game is over.")
                        self.game.game_over = True
                        if not has_black_general:
                            self.game.winner = "red"
                            logger.info("Red wins by capturing the black General")
                        elif not has_red_general:
                            self.game.winner = "black"
                            logger.info("Black wins by capturing the red General")

                        # Clear selection and skip the rest of the checks
                        self.selected_piece = None
                        self.selected_position = None
                        self.valid_moves = []
                        return

                    # Only check for checkmate if no general was captured
                    # Important: We need to check if the opponent (who just moved) put the current player in check
                    # After the move, self.game.current_player is already the other player
                    opponent = "red" if self.game.current_player == "black" else "black"
                    logger.debug(
                        f"Checking if {self.game.current_player} is in check..."
                    )
                    if self.board.is_check(self.game.current_player):
                        logger.info(f"{self.game.current_player} is in check!")
                        print(f"{self.game.current_player.capitalize()} is in check!")

                        # Check if current player has any valid moves left
                        logger.debug(
                            f"Checking if {self.game.current_player} has any valid moves..."
                        )
                        has_valid_moves = False
                        for piece in self.board.pieces:
                            if (
                                piece.color == self.game.current_player
                                and not piece.captured
                            ):
                                valid_moves = piece.get_valid_moves(self.board)
                                if valid_moves:
                                    logger.debug(
                                        f"{self.game.current_player} {piece.__class__.__name__} at {piece.position} has {len(valid_moves)} valid moves"
                                    )
                                    has_valid_moves = True
                                    break

                        if not has_valid_moves:
                            # It's checkmate
                            logger.info(
                                f"CHECKMATE! {self.game.current_player} has no valid moves"
                            )
                            self.game.game_over = True
                            self.game.winner = opponent
                            logger.info(
                                f"Game over! {self.game.winner} wins by checkmate"
                            )
                else:
                    logger.warning(
                        f"Move from {self.selected_position} to {board_pos} failed"
                    )
            else:
                logger.debug(
                    f"Invalid move to {board_pos}, not in valid moves: {self.valid_moves}"
                )

            # Clear selection
            logger.debug("Clearing selection")
            self.selected_piece = None
            self.selected_position = None
            self.valid_moves = []
        else:
            # Select a piece
            x, y = board_pos
            piece = self.board.get_piece_at(x, y)

            if piece:
                logger.debug(
                    f"Found {piece.color} {piece.__class__.__name__} at {board_pos}"
                )
                if piece.color == self.game.current_player:
                    self.selected_piece = piece
                    self.selected_position = board_pos
                    self.valid_moves = piece.get_valid_moves(self.board)
                    logger.debug(
                        f"Selected piece with {len(self.valid_moves)} valid moves: {self.valid_moves}"
                    )
                else:
                    logger.debug(
                        f"Cannot select {piece.color} piece during {self.game.current_player}'s turn"
                    )
            else:
                logger.debug(f"No piece at {board_pos}")

    def run(self):
        """Main game loop"""
        logger.info("Starting Xiangqi game")
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    logger.info("Game window closed")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        self.handle_click(event.pos)

            # Draw everything
            self.draw_board()
            self.draw_selected_highlight()
            self.draw_pieces()

            # Update the display
            pygame.display.flip()

            # Cap the frame rate
            self.clock.tick(60)

        logger.info("Game ended")
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    gui = XiangqiGUI()
    gui.run()
