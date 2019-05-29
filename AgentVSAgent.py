from ReversiEnvironment import ReversiEnvironment
from RandomAgent import RandomAgent

WHITE = 1
BLACK = 2


def main():
    reversi = ReversiEnvironment()
    playing = True
    white_agent = RandomAgent(1)
    black_agent = RandomAgent(2)

    while playing:
        player = reversi.player_turn
        state = reversi.get_board()
        # print(state)
        if player == WHITE:
            location = white_agent.predict(state)
        else:
            location = black_agent.predict(state)
        board, player_turn, game_result = reversi.step(location, matrix_coord=True)
        x = location[1]+1
        y = 8-location[0]
        board_loc = (x, y)
        # print("Player "+str(player)+" placing on "+str(board_loc))
        if game_result == 0:
            print("It's a draw!")
            playing = False
        elif game_result == 1:
            print("WHITE wins!")
            playing = False
        elif game_result == 2:
            print("BLACK wins!")
            playing = False


if __name__ == '__main__':
    main()