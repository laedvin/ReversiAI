import tkinter as tk
import numpy as np
from os.path import abspath, join, dirname
from agents.residual_tower_policy_agent import ResidualTowerPolicyAgent
from reversi.reversi_environment import ReversiEnvironment
from agents.random_agent import RandomAgent
from agents.naive_agent import NaiveAgent
from genetic_algorithm.lineage import Lineage


WHITE = 1
BLACK = 2

MAP_NAME = {"1": "WHITE", "2": "BLACK"}


class GameWindow:
    def __init__(self, agent=None):
        self.agent = agent
        if self.agent and self.agent.own_player == 1:
            self.agent_player = 1
            print("Starting a game vs an AI playing white")
        elif self.agent and self.agent.own_player == 2:
            self.agent_player = 2
            print("Starting a game vs an AI playing black")
        else:
            self.agent_player = None
            print("Starting a player vs player game")

        # Initialize the game board
        self.cell_size = 64
        self.text_height = self.cell_size
        self.window = tk.Tk(className="Reversi")
        self.canvas = tk.Canvas(
            self.window,
            bg="green",
            width=self.cell_size * 8,
            height=self.cell_size * 8 + self.text_height,
        )
        self.reversi = ReversiEnvironment()
        for i in range(8):
            for j in range(8):
                cell_box = np.array([i, j, (i + 1), (j + 1)]) * self.cell_size + 2
                x_min, y_min, x_max, y_max = cell_box
                self.canvas.create_line(x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max)

        self.white_score_text = self.canvas.create_text(
            str(self.cell_size * 0.5),
            self.cell_size * 8 + self.text_height / 2,
            text="",
            fill="white",
            font=("courier", int(self.text_height * 0.8), "bold"),
        )
        self.black_score_text = self.canvas.create_text(
            int(self.cell_size * 7.5),
            self.cell_size * 8 + self.text_height / 2,
            text="",
            fill="black",
            font=("courier", int(self.text_height * 0.8), "bold"),
        )

        state = self.reversi.get_board()
        self.update_game_state(state)

        if self.agent_player == 1:
            self.agent_loop_logic()
        else:
            print(f"{MAP_NAME['1']} to play")

        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.callback)

    def update_game_state(self, state):
        white_score, black_score = self.reversi.calculate_score()
        self.canvas.itemconfigure(self.white_score_text, text=str(white_score))
        self.canvas.itemconfigure(self.black_score_text, text=str(black_score))
        for i in range(8):
            for j in range(8):
                if state[i][j] == WHITE:
                    cell_box = np.array([i, j, (i + 1), (j + 1)]) * self.cell_size + 2
                    self.canvas.create_oval(
                        cell_box[0],
                        cell_box[1],
                        cell_box[2],
                        cell_box[3],
                        fill="white",
                    )
                elif state[i][j] == BLACK:
                    cell_box = np.array([i, j, (i + 1), (j + 1)]) * self.cell_size + 2
                    self.canvas.create_oval(
                        cell_box[0],
                        cell_box[1],
                        cell_box[2],
                        cell_box[3],
                        fill="black",
                    )

    def callback(self, event):
        """Assumes that it's a player's turn. If the resulting board state is an agent's turn,
        let the agent place its piece before exiting the callback.
        """
        stepped, next_player, game_result = self.player_place_piece(event)
        if game_result == 0:
            print("It's a draw!")
        elif game_result == 1:
            print("WHITE wins!")
        elif game_result == 2:
            print("BLACK wins!")
        elif stepped and next_player != self.agent_player:
            print(f"{MAP_NAME[str(next_player)]} to play")
        elif stepped and next_player == self.agent_player:
            # If it's the agent's turn
            next_player, game_result = self.agent_loop_logic()
            if game_result == 0:
                print("It's a draw!")
            elif game_result == 1:
                print("WHITE wins!")
            elif game_result == 2:
                print("BLACK wins!")
            elif game_result == -1:
                print(f"{MAP_NAME[str(next_player)]} to play")

    def player_place_piece(self, event):
        """Places a piece depending on the player's mouse click.

        Converts the mouse coordinate to board coordinate.

        Args:
            event: the callback event

        Returns: (stepped, next_player, game_result)

        """
        stepped = False
        next_player = None
        game_result = -1
        coord = np.array([event.x - 2, event.y - 2])
        if (coord[0] > 8 * self.cell_size - 1 or coord[0] < 0) or (
            coord[1] > 8 * self.cell_size - 1 or coord[1] < 0
        ):
            print("Outside the board")
        else:
            # Find the board coordinate and the bounding box of the cell (in
            # canvas coordinates)
            board_coord = np.floor(coord / self.cell_size).astype(int)
            stepped, state, next_player, game_result = self.reversi.step(
                board_coord, matrix_coord=True
            )
            self.update_game_state(state)
        return stepped, next_player, game_result

    def agent_loop_logic(self):
        """Agent plays until it's not its turn anymore"""
        next_player = self.reversi.player_turn
        game_result = -1
        while next_player == self.agent_player and game_result == -1:
            next_player, game_result = self.agent_place_piece()
        return next_player, game_result

    def agent_place_piece(self):
        """Lets the agent place a piece"""
        next_player = None
        game_result = -1
        state = self.reversi.get_board()
        if self.agent_player == self.reversi.player_turn:
            location = self.agent.predict(state)
            _, state, next_player, game_result = self.reversi.step(location, matrix_coord=True)
            self.update_game_state(state)
            print(f"Agent played on {location}")
        return next_player, game_result


def main():
    path_to_lineage = abspath(
        join(dirname(__file__), "genetic_algorithm/lineages/no_baseline_agents/")
    )
    lineage = Lineage(path_to_lineage)
    pop = lineage.get_pop_from_gen(lineage.current_gen)
    elos = [individual["elo"] for individual in pop.pop]
    best_individual = pop.pop[np.argmax(elos)]
    agent = ResidualTowerPolicyAgent(2)
    agent.set_genome(best_individual["genome"])
    game_window = GameWindow(agent=agent)
    tk.mainloop()


if __name__ == "__main__":
    main()
