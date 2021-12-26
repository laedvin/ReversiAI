import numpy as np


from reversi.reversi_environment import ReversiEnvironment
from agents.random_agent import RandomAgent
from agents.naive_agent import NaiveAgent
from agents.branch_aggregate_agent import BranchAggregateAgent
from agents.simple_nn_agent import SimpleNNAgent
from timeit import default_timer as timer

WHITE = 1
BLACK = 2


def play_game(white, black):
    white.set_player(1)
    black.set_player(2)
    reversi = ReversiEnvironment()
    playing = True

    # genome = black_agent.get_genome()
    # matrices = black_agent.genome_to_parameters(genome)
    # state_dict = black_agent.net.state_dict()
    # for param_name, matrix in zip(state_dict, matrices):
    #    with torch.no_grad():
    #        state_dict[param_name] = matrix
    # genome_2 = black_agent.get_genome()

    while playing:
        player = reversi.player_turn
        state = reversi.get_board()
        # print(state)
        if player == WHITE:
            location = white.predict(state)
        else:
            location = black.predict(state)
        _, _, _, game_result = reversi.step(location, matrix_coord=True)
        if game_result == 0:
            playing = False
        elif game_result == 1:
            playing = False
        elif game_result == 2:
            playing = False
    return game_result


def main():
    player_1 = SimpleNNAgent()
    player_2 = SimpleNNAgent()
    p1_wins = 0
    p2_wins = 0
    games = 500
    start = timer()
    for i in range(games):
        if i % 2:
            game_result = play_game(player_1, player_2)
            if game_result == 1:
                p1_wins += 1
            elif game_result == 2:
                p2_wins += 1
        else:
            game_result = play_game(player_2, player_1)
            if game_result == 1:
                p2_wins += 1
            elif game_result == 2:
                p1_wins += 1
    end = timer()
    print(f"Player 1 | {p1_wins} -- {p2_wins} | Player 2")
    print(f"Elapsed time: {end-start:.3f}s")


if __name__ == "__main__":
    main()
