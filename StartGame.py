import tkinter as tk
import sys
import numpy as np
from ReversiEnvironment import ReversiEnvironment
from Agents.RandomAgent import RandomAgent


WHITE = 1
BLACK = 2


class GameWindow:
    def __init__(self, agent_white="human", agent_black="human"):
        self.cell_size = 64
        self.text_height = self.cell_size
        self.window = tk.Tk(className="bla")
        self.canvas = tk.Canvas(
            self.window,
            bg="green",
            width=self.cell_size * 8,
            height=self.cell_size * 8 + self.text_height,
        )
        self.reversi = ReversiEnvironment()
        for i in range(8):
            for j in range(8):
                cell_box = (
                    np.array([i, j, (i + 1), (j + 1)]) * self.cell_size + 2
                )
                x_min, y_min, x_max, y_max = cell_box
                self.canvas.create_line(
                    x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
                )

        state = self.reversi.get_board()
        self.update_game_state(state)

        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.callback)

    def update_game_state(self, state):
        for i in range(8):
            for j in range(8):
                if state[i][j] == WHITE:
                    cell_box = (
                        np.array([i, j, (i + 1), (j + 1)]) * self.cell_size + 2
                    )
                    self.canvas.create_oval(
                        cell_box[0],
                        cell_box[1],
                        cell_box[2],
                        cell_box[3],
                        fill="white",
                    )
                elif state[i][j] == BLACK:
                    cell_box = (
                        np.array([i, j, (i + 1), (j + 1)]) * self.cell_size + 2
                    )
                    self.canvas.create_oval(
                        cell_box[0],
                        cell_box[1],
                        cell_box[2],
                        cell_box[3],
                        fill="black",
                    )

    def callback(self, event):
        # Transform to coordinates on the board
        # The canvas is offset by 2, so x between 0 and 1 is outside of the
        # green background.

        player = self.reversi.player_turn
        coord = np.array([event.x - 2, event.y - 2])
        if (coord[0] > 8 * self.cell_size - 1 or coord[0] < 0) or (
            coord[1] > 8 * self.cell_size - 1 or coord[1] < 0
        ):
            print("Outside the board")
        else:
            # Find the board coordinate and the bounding box of the cell (in
            # canvas coordinates)
            board_coord = np.floor(coord / self.cell_size).astype(int)
            state, player_turn, game_result = self.reversi.step(
                board_coord, matrix_coord=True
            )
            self.update_game_state(state)
            if game_result == 0:
                print("It's a draw!")
            elif game_result == 1:
                print("WHITE wins!")
            elif game_result == 2:
                print("BLACK wins!")


def main():
    white_agent = RandomAgent(1)
    game_window = GameWindow(agent_white=white_agent)
    # TODO: Figure out a way to play with an AI. Maybe with tk.after()?
    tk.mainloop()


if __name__ == "__main__":
    main()
