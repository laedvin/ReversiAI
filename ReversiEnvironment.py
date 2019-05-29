import numpy as np
from GameBoard import GameBoard

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
        A player makes a move, updating the game state. Returns states. Also returns whether or not a player wins.
        :param coord: The location to place the piece.
        :param matrix_coord: If on, assumes coord is in matrix coordinates.
        :return: Board state, the player's turn and
        game result (-1, 0, 1, 2 if not decided, draw, white won and black won, respectively)
        """
        if not matrix_coord:
            if self.game_board.place_piece(coord, self.player_turn):
                if self.player_turn == WHITE:
                    self.player_turn = BLACK
                else:
                    self.player_turn = WHITE
        else:
            if self.game_board.place_piece(self.game_board.matrix_to_coordinates(coord), self.player_turn):
                if self.player_turn == WHITE and self.game_board.has_moves(BLACK):
                    self.player_turn = BLACK
                elif self.player_turn == BLACK and self.game_board.has_moves(WHITE):
                    self.player_turn = WHITE

        board = self.game_board.get_board()
        game_result = self.check_win()

        return board, self.player_turn, game_result

    def reset(self):
        self.game_board.reset()
        # TODO: Reset any self variables in ReversiEnvironment

    def check_win(self):
        """
        Game ends if neither player can place a piece. Number of WHITE vs BLACK pieces gets counted.
        :return: -1 if game is unfinished, 0 if draw , 1 if WHITE wins and 2 if BLACK wins
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


