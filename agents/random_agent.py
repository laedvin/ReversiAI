from agents.basic_agent import BasicAgent
import numpy as np


class RandomAgent(BasicAgent):
    """
    This agent plays a random move.
    """

    def __init__(self, player=1):
        super(RandomAgent, self).__init__(player)

    def predict(self, state):
        moves = self.game_board.find_moves(self.own_player, state)
        if len(moves) > 0:
            return moves[int(np.random.uniform(0, len(moves), 1)[0])]
