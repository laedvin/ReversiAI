import numpy as np
from reversi_ai.reversi.game_board import GameBoard

WHITE = 1
BLACK = 2
EMPTY = 0


class ReversiEnvironment:
    def __init__(self):
        super(ReversiEnvironment, self).__init__()
        self.player_turn = WHITE
        self.game_board = GameBoard()

    def get_board(self):
        return self.game_board.get_board()

    def step(self, coord, matrix_coord=False):
        """
        A player makes a move, updating the game state. Returns states. Also
        returns whether or not a player wins.

        Args:
            coord: The location to place the piece.
            matrix_coord: If True, assumes coord is in matrix coordinates.

        Returns:
            stepped: False if no change in board state, else True
            board: Board state
            player_turn: the player's turn
            game_result: game result (-1, 0, 1, 2 if not decided, draw, white
                         won and black won, respectively)
        """
        stepped = False
        if not matrix_coord:
            if self.game_board.place_piece(coord, self.player_turn):
                stepped = True
                if self.player_turn == WHITE:
                    self.player_turn = BLACK
                else:
                    self.player_turn = WHITE
        else:
            if self.game_board.place_piece(
                self.game_board.matrix_to_coordinates(coord), self.player_turn
            ):
                stepped = True
                if self.player_turn == WHITE and self.game_board.has_moves(BLACK):
                    self.player_turn = BLACK
                elif self.player_turn == BLACK and self.game_board.has_moves(WHITE):
                    self.player_turn = WHITE

        board = self.game_board.get_board()
        game_result = self.check_win()

        return stepped, board, self.player_turn, game_result

    def reset(self):
        self.game_board.reset()
        self.player_turn = WHITE
        # TODO: Reset any self variables in ReversiEnvironment

    def check_win(self):
        """
        Game ends if neither player can place a piece. Number of WHITE vs BLACK
        pieces get counted.
        :return: -1 if game is unfinished, 0 if draw, 1 if WHITE wins and 2 if
                 BLACK wins
        """
        if not self.game_board.has_moves(WHITE) and not self.game_board.has_moves(BLACK):
            board = self.game_board.get_board()
            score = 0
            for i in range(8):
                for j in range(8):
                    if board[i][j] == WHITE:
                        score += 1
                    elif board[i][j] == BLACK:
                        score -= 1
            if score > 0:
                return 1
            elif score < 0:
                return 2
            else:
                return 0
        else:
            return -1

    def calculate_score(self, board=None):
        """Calculates the score of a board

        Args:
            board: The board, optional

        Returns: score_white, score_black
        """
        score_white = 0
        score_black = 0
        if board is None:
            board = self.get_board()
        for i in range(8):
            for j in range(8):
                if board[i][j] == WHITE:
                    score_white += 1
                elif board[i][j] == BLACK:
                    score_black += 1
        return score_white, score_black
