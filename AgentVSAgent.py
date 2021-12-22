from ReversiEnvironment import ReversiEnvironment
from Agents.RandomAgent import RandomAgent
from Agents.NaiveAgent import NaiveAgent
from Agents.BranchAggregateAgent import BranchAggregateAgent

WHITE = 1
BLACK = 2


def play_game():
    reversi = ReversiEnvironment()
    playing = True
    white_agent = NaiveAgent(1)
    black_agent = BranchAggregateAgent(2)

    while playing:
        player = reversi.player_turn
        state = reversi.get_board()
        # print(state)
        if player == WHITE:
            location = white_agent.predict(state)
        else:
            location = black_agent.predict(state)
        board, player_turn, game_result = reversi.step(
            location, matrix_coord=True
        )
        x = location[1] + 1
        y = 8 - location[0]
        board_loc = (x, y)
        # print("Player "+str(player)+" placing on "+str(board_loc))
        if game_result == 0:
            # print("It's a draw!")
            playing = False
        elif game_result == 1:
            # print("WHITE wins!")
            playing = False
        elif game_result == 2:
            # print("BLACK wins!")
            playing = False
    return game_result


def main():
    white_wins = 0
    black_wins = 0
    games = 1
    for i in range(games):
        game_result = play_game()
        if game_result == 1:
            white_wins += 1
        elif game_result == 2:
            black_wins += 1
    print(f"{white_wins}, {black_wins}")


if __name__ == "__main__":
    main()
