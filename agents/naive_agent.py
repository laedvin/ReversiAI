from agents.basic_agent import BasicAgent
from reversi.game_board import GameBoard
import numpy as np


class NaiveAgent(BasicAgent):
    """
    This agent plays the move that maximizes the immediate score gain.
    """

    def __init__(self, player=1):
        super(NaiveAgent, self).__init__(player)

    def predict(self, state):
        moves = self.game_board.find_moves(self.own_player, state)
        best_move_candidates = moves[
            int(np.random.uniform(0, len(moves), 1)[0])
        ]
        best_score = 0
        for move in moves:
            score = 0
            branch_board = GameBoard()
            branch_board.board = np.copy(state)
            branch_board.place_piece(
                branch_board.matrix_to_coordinates(move), self.own_player
            )
            for i in range(8):
                for j in range(8):
                    if branch_board.board[i][j] == self.own_player:
                        score += 1
            if score > best_score:
                best_score = score
                best_move_candidates = [move]
            elif score == best_score:
                best_move_candidates += [move]
        return best_move_candidates[
            int(np.random.uniform(0, len(best_move_candidates), 1)[0])
        ]
