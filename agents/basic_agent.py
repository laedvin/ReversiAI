from abc import ABC, abstractmethod
from reversi.game_board import GameBoard


class BasicAgent(ABC):
    def __init__(self, player=1):
        super(BasicAgent, self).__init__()
        self.game_board = GameBoard()
        self.own_player = None
        self.opponent = None
        self.set_player(player)

    def set_player(self, player):
        if player != 1 and player != 2:
            raise ValueError(
                f"Wrong player setting; it should be either {1} or {2}, but "
                f"encountered {player}"
            )
        self.own_player = player
        self.opponent = 2 if self.own_player == 1 else 1

    @abstractmethod
    def predict(self, state):
        pass
