from GameBoard import GameBoard
import numpy as np


class RandomAgent:
    """
    This agent plays a random move.
    """
    def __init__(self, player):
        super(RandomAgent, self).__init__()
        self.board = GameBoard()
        self.player = player

    def predict(self, state):
        moves = self.board.find_moves(self.player, state)
        if len(moves) > 0:
            return moves[int(np.random.uniform(0, len(moves), 1)[0])]
