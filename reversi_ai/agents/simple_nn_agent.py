import numpy as np
import torch
from torch import nn


from reversi_ai.agents.basic_agent import Agent


class SimpleNNAgent(Agent):
    """A simple neural network agent"""

    def __init__(self, player=1):
        super(SimpleNNAgent, self).__init__(player)
        self.net = SimpleNN()
        self.softmax = nn.Softmax(dim=0)

    def predict(self, board, greedy=True):
        """The agent predict the piece to play

        Picks a move to play, either greedily or with softmax weights

        Args:
            board: the board state gotten from game_board.get_board()
            greedy: whether or not to pick moves greedily

        Returns: the move to play
        """
        # Prepare input to network
        board = np.copy(board)
        self.game_board.board = np.copy(board)
        board[board.astype(bool)] = board[board.astype(bool)] * 2 - 3
        board_flattened = torch.flatten(torch.from_numpy(board))
        x = torch.cat((board_flattened, torch.tensor([self.own_player * 2 - 3]))).float()

        with torch.no_grad():
            out = self.net.forward(x)

        # Process output from network
        filter = np.full((8, 8), -np.inf)
        moves = self.game_board.find_moves(self.own_player)
        for move in moves:
            filter[move] = 0
        out = out + filter.flatten()
        move_distribution = self.softmax(out).detach().numpy()
        possible_indices = np.nonzero(move_distribution)[0]
        move_distribution = move_distribution[possible_indices]

        # Reconsider moves in case the model predicts 0 for a possible move
        xs, ys = np.unravel_index(possible_indices, (8, 8))
        moves = list(zip(xs, ys))

        # Pick a move
        if greedy:
            move = moves[np.argmax(move_distribution)]
        else:
            rng = np.random.default_rng()
            move = rng.choice(moves, p=move_distribution)
        return tuple(move)

    def predict_with_raw_output(self, board, greedy=True):
        """The agent predict the piece to play

        Picks a move to play, either greedily or with softmax weights.
        This version also returns the linear network outputs with gradients.

        Args:
            board: the board state gotten from game_board.get_board()
            greedy: whether or not to pick moves greedily

        Returns: the move to play and the raw neural network output
        """
        # Prepare input to network
        board = np.copy(board)
        self.game_board.board = np.copy(board)
        board[board.astype(bool)] = board[board.astype(bool)] * 2 - 3
        board_flattened = torch.flatten(torch.from_numpy(board))
        x = torch.cat((board_flattened, torch.tensor([self.own_player * 2 - 3]))).float()

        out = self.net.forward(x)

        # Process output from network
        filter = np.full((8, 8), -np.inf)
        moves = self.game_board.find_moves(self.own_player)
        for move in moves:
            filter[move] = 0
        out_filtered = out + torch.from_numpy(filter.flatten())
        move_distribution = self.softmax(out_filtered).detach().numpy()
        possible_indices = np.nonzero(move_distribution)[0]
        move_distribution = move_distribution[possible_indices]

        # Reconsider moves in case the model predicts 0 for a possible move
        xs, ys = np.unravel_index(possible_indices, (8, 8))
        moves = list(zip(xs, ys))

        # Pick a move
        if greedy:
            move = moves[np.argmax(move_distribution)]
        else:
            rng = np.random.default_rng()
            move = rng.choice(moves, p=move_distribution)
        return tuple(move), out

    def train_on_data(self, states, players, target_moves, raw_outputs):
        """

        Args:
            states: a list of all states
            players: a list of the players whose turn it was to play
            target_moves: a list of all the target moves

        """
        states = torch.flatten(states, start_dim=1)
        states = torch.cat(
            (
                states,
                (players * 2 - 3).reshape((states.shape[0], 1)),
            ),
            dim=1,
        )

        # Ravel the moves/indices to get onehot vectors
        target_moves = target_moves.detach().numpy()
        target_moves = np.ravel_multi_index((target_moves[:, 0], target_moves[:, 1]), (8, 8))
        column_indices = np.arange(target_moves.shape[0])
        target_onehot = torch.zeros((states.shape[0], 64))
        target_onehot[column_indices, target_moves] = 1

    @staticmethod
    def parameters_to_genome(net):
        """Transforms a network's parameters to a genome.

        The genome is just a flattening and a concatenation of the parameters.

        Args:
            net: a sequential pytorch nn

        Return: 1D numpy vector
        """
        genome = np.array([])
        for m in net.modules():
            if isinstance(m, nn.Linear):
                genome = np.append(genome, m.weight.flatten().detach())
                genome = np.append(genome, m.bias.flatten().detach())
        return genome

    @staticmethod
    def genome_to_parameters(genome):
        # Hardcoded for a specific network architecture
        shapes = [(32, 65), (32), (32, 32), (32), (64, 32), (64)]
        matrices = []
        for shape in shapes:
            matrix = torch.from_numpy(genome[0 : np.prod(shape)]).reshape(shape)
            matrices.append(matrix)
            genome = genome[np.prod(shape) :]
        return matrices

    def get_genome(self):
        return self.parameters_to_genome(self.net)

    def set_genome(self, genome):
        # Hardcoded for a specific network architecture
        matrices = self.genome_to_parameters(genome)
        state_dict = self.net.state_dict()
        for param_name, matrix in zip(state_dict, matrices):
            with torch.no_grad():
                state_dict[param_name] = matrix


class SimpleNN(nn.Module):
    """Neural network with 2 hidden layers"""

    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 32),  # 64 squares and 1 player indicator
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )

    def forward(self, board):
        """Forwards a flattened board tensor"""
        return self.net(board)
