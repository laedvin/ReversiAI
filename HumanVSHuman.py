from ReversiEnvironment import ReversiEnvironment

WHITE = 1
BLACK = 2


def main():
    reversi = ReversiEnvironment()
    playing = True

    while playing:
        player = reversi.player_turn
        print(reversi.get_board())
        if player == WHITE:
            input_string = input("WHITE's turn: ")
            split_list = input_string.split()
            x = int(split_list[0])
            y = int(split_list[1])
            location = (x, y)
            print(location)
        else:
            input_string = input("BLACK's turn: ")
            split_list = input_string.split()
            x = int(split_list[0])
            y = int(split_list[1])
            location = (x, y)
            print(location)
        board, player_turn, game_result = reversi.step(location)
        if game_result == 0:
            print("It's a draw!")
            playing = False
        elif game_result == 1:
            print("WHITE wins!")
            playing = False
        elif game_result == 2:
            print("BLACK wins!")
            playing = False


if __name__ == "__main__":
    main()
