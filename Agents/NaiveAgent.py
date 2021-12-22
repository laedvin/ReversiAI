from GameBoard import GameBoard
import numpy as np


class NaiveAgent:
    """
    This agent plays the move that maximizes the immediate score gain.
    """
    def __init__(self, player):
        super(NaiveAgent, self).__init__()
        self.board = GameBoard()
        self.player = player

    def predict(self, state):
        moves = self.board.find_moves(self.player, state)
        if len(moves) > 0:
            best_move = moves[int(np.random.uniform(0, len(moves), 1)[0])]
            best_score = 0
            for move in moves:
                score = 0
                branch_board = GameBoard()
                branch_board.board = np.copy(state)
                branch_board.place_piece(
                    branch_board.matrix_to_coordinates(move),
                    self.player
                )
                for i in range(8):
                    for j in range(8):
                        if branch_board.board[i][j] == self.player:
                            score += 1
                if score > best_score:
                    best_score = score
                    best_move = move

            return best_move
