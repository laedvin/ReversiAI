from agents.basic_agent import BasicAgent
from reversi.game_board import GameBoard
import numpy as np


class BranchAggregateAgent(BasicAgent):
    """
    This agent plays the move that maximizes the score difference in its next
    turn, assuming an opponent that plays random moves.
    """

    def __init__(self, player=1):
        super(BranchAggregateAgent, self).__init__(player)

    def predict(self, state):
        moves = self.board.find_moves(self.own_player, state)
        if len(moves) > 0:
            best_move = moves[int(np.random.uniform(0, len(moves), 1)[0])]
            best_score = -np.inf
            # Try every possible move
            for move in moves:
                branch_score = -np.inf
                branch_board = GameBoard()
                branch_board.board = np.copy(state)
                branch_board.place_piece(
                    branch_board.matrix_to_coordinates(move), self.own_player
                )
                branch_state = branch_board.get_board()

                opponent_moves = branch_board.find_moves(self.opponent)
                # If opponent can't play, find the best move after that
                if len(opponent_moves) == 0:
                    next_moves = branch_board.find_moves(self.own_player)
                    # TODO: What if neither player can play?
                    if len(next_moves) == 0:
                        branch_score = 0
                        for i in range(8):
                            for j in range(8):
                                if branch_board.board[i][j] == self.own_player:
                                    branch_score += 1
                                elif branch_board.board[i][j] == self.opponent:
                                    branch_score -= 1
                    else:
                        for next_move in next_moves:
                            score = 0
                            branch_board = GameBoard()
                            branch_board.board = np.copy(branch_state)
                            branch_board.place_piece(
                                branch_board.matrix_to_coordinates(next_move),
                                self.own_player,
                            )
                            for i in range(8):
                                for j in range(8):
                                    if (
                                        branch_board.board[i][j]
                                        == self.own_player
                                    ):
                                        score += 1
                                    elif (
                                        branch_board.board[i][j]
                                        == self.opponent
                                    ):
                                        score -= 1
                            if score > branch_score:
                                branch_score = score

                # If opponent can play, find the average outcome of the branch
                else:
                    branch_score = 0
                    for opponent_move in opponent_moves:
                        score = -np.inf
                        # This is the likely future score outcome of the next
                        # move the opponent plays

                        branch_board = GameBoard()
                        branch_board.board = np.copy(branch_state)
                        branch_board.place_piece(
                            branch_board.matrix_to_coordinates(opponent_move),
                            self.opponent,
                        )
                        branch_branch_state = branch_board.get_board()

                        next_moves = branch_board.find_moves(self.own_player)
                        if len(next_moves) == 0:
                            next_opp_moves = branch_board.find_moves(
                                self.opponent
                            )
                            if len(next_opp_moves) == 0:
                                # The scenario where no-one can play
                                score = 0
                                for i in range(8):
                                    for j in range(8):
                                        if (
                                            branch_board.board[i][j]
                                            == self.own_player
                                        ):
                                            score += 1
                                        elif (
                                            branch_board.board[i][j]
                                            == self.opponent
                                        ):
                                            score -= 1
                            else:
                                # The scenario where only this agent can't play
                                score = 0
                                for next_opp_move in next_opp_moves:
                                    branch_board = GameBoard()
                                    branch_board.board = np.copy(
                                        branch_branch_state
                                    )
                                    branch_board.place_piece(
                                        branch_board.matrix_to_coordinates(
                                            next_opp_move
                                        ),
                                        self.opponent,
                                    )
                                    for i in range(8):
                                        for j in range(8):
                                            if (
                                                branch_board.board[i][j]
                                                == self.own_player
                                            ):
                                                score += 1
                                            elif (
                                                branch_board.board[i][j]
                                                == self.opponent
                                            ):
                                                score -= 1
                                score /= len(next_opp_moves)
                        else:
                            for next_move in next_moves:
                                next_score = 0
                                branch_board = GameBoard()
                                branch_board.board = np.copy(
                                    branch_branch_state
                                )
                                branch_board.place_piece(
                                    branch_board.matrix_to_coordinates(
                                        next_move
                                    ),
                                    self.own_player,
                                )
                                for i in range(8):
                                    for j in range(8):
                                        if (
                                            branch_board.board[i][j]
                                            == self.own_player
                                        ):
                                            next_score += 1
                                        elif (
                                            branch_board.board[i][j]
                                            == self.opponent
                                        ):
                                            next_score -= 1
                                if next_score > score:
                                    score = next_score
                        branch_score += score
                    branch_score /= len(opponent_moves)

                if branch_score > best_score:
                    best_score = branch_score
                    best_move = move

            return best_move
