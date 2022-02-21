from reversi.game_board import GameBoard
from agents.basic_agent import BasicAgent
import numpy as np


class NDepthSearchAgent(BasicAgent):
    """This agent searches n layers deep

    n is the number of its own turns into the future the agent searches.
    n = 1 means that it will only look at the current turn + its opponent's
    following turn.
    """

    def __init__(self, player, depth=1):
        super(NDepthSearchAgent, self).__init__(player)
        self.depth = int(depth)

    def play(self, state):
        """Place a piece given the input state.

        For all of the possible moves, it performs a recursive search to some
        depth and calculates the best worst-case score.

        Args: state of the board
        Returns: The best possible move if it exists, or None
        """
        self.game_board.board = state
        best_move = None
        moves = self.game_board.find_moves(self.own_player, state)
        if len(moves) == 1:
            best_move = moves[0]
        elif len(moves) > 1:
            best_move = moves[0]
            best_score = -np.inf
            for move in moves:
                score = self.recursive_search(
                    self.game_board, move, self.depth
                )
                if score > best_score:
                    best_score = score
                    best_move = move
        return best_move

    def recursive_search(self, input_board, input_move, depth):
        """Given an input board, input move and depth, search for all
        possible plays and find the best score.
        """
        # Cast input_board to GameBoard object?
        if depth == 1:
            input_board.place_piece(
                input_board.matrix_to_coordinates(input_move), self.own_player
            )
            moves = input_board.find_moves(self.opponent)
            if len(moves) == 0:
                best_score = input_board.calculate_score()
            elif len(moves) == 1:
                input_board.place_piece(
                    input_board.matrix_to_coordinates(moves[0]), self.opponent
                )
                best_score = input_board.calculate_score()
            elif len(moves) > 1:
                best_move = moves[int(np.random.uniform(0, len(moves), 1)[0])]
                best_score = 0
                for move in moves:
                    # Let the opponent play until it's this agent's turn
                    opponent_turn = True
                    score = 0
                    branch_board = GameBoard()
                    branch_board.board = np.copy(input_board.get_board())
                    branch_board.place_piece(
                        branch_board.matrix_to_coordinates(move),
                        self.own_player,
                    )
                    # Calc score...
                    for i in range(8):
                        for j in range(8):
                            if branch_board.board[i][j] == self.player:
                                score += 1
                    if score > best_score:
                        best_score = score
                        best_move = move
        else:
            pass
        pass

    def let_opponent_play(
        self,
    ):
        """Recursively let opponent play until they cannot"""
