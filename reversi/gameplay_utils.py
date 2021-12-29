import torch
from reversi.reversi_environment import ReversiEnvironment


def play_game(white, black):
    """White agent plays against Black agent.
    Args:
        white: The white agent
        black: The black agent
    Returns: 1 if white wins, 0.5 if draw, 0 if black wins
    """
    white.set_player(1)
    black.set_player(2)
    reversi = ReversiEnvironment()
    playing = True
    while playing:
        player = reversi.player_turn
        state = reversi.get_board()
        if player == 1:
            location = white.predict(state)
        else:
            location = black.predict(state)
        _, _, _, game_result = reversi.step(location, matrix_coord=True)
        if game_result == 0:
            playing = False
            score = 0.5  # Draw
        elif game_result == 1:
            playing = False
            score = 1  # White wins
        elif game_result == 2:
            playing = False
            score = 0  # Black wins
    return score


def play_match(agent_a, agent_b):
    score_a = play_game(agent_a, agent_b)
    score_b = play_game(agent_b, agent_a)
    return score_a, score_b


def play_game_nn_train(white, black, detach=False):
    """White agent plays against Black agent

    This also returns each board state that were in the game and the player's
    chosen moves

    Args:
        white: The white agent
        black: The black agent

    Returns: (score, states, player_turns, picked_moves, network_outputs)
    """
    white.set_player(1)
    black.set_player(2)
    reversi = ReversiEnvironment()
    playing = True
    states = []
    player_turns = []
    picked_moves = []
    network_outputs = torch.zeros(0, 64)
    while playing:
        player = reversi.player_turn
        state = reversi.get_board()
        if player == 1:
            location, out = white.predict_with_raw_output(state, greedy=False)
        else:
            location, out = black.predict_with_raw_output(state, greedy=False)

        player_turns.append(player)
        states.append(state)
        picked_moves.append(location)
        network_outputs = torch.cat(
            (network_outputs, out.reshape(-1, 64)), dim=0
        )

        if detach:
            network_outputs = network_outputs.detach()

        _, _, _, game_result = reversi.step(location, matrix_coord=True)
        if game_result == 0:
            playing = False
            score = 0.5  # Draw
        elif game_result == 1:
            playing = False
            score = 1  # White wins
        elif game_result == 2:
            playing = False
            score = 0  # Black wins
    return score, states, player_turns, picked_moves, network_outputs
