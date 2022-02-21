import numpy as np
import torch
from torch import nn


from agents.basic_agent import BasicAgent


class ResidualTowerPolicyAgent(BasicAgent):
    """A neural network agent that"""

    def __init__(self, player=1, use_cuda=False):
        super(ResidualTowerPolicyAgent, self).__init__(player)
        self.softmax = nn.Softmax(dim=0)
        if use_cuda:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device("cpu")
        self.net = ResidualTowerPolicy(32).to(self.device)

    def predict(self, input_board):
        """The agent predict the piece to play

        Returns: the move to play
        """
        # Prepare input to network
        input_board = np.copy(input_board)
        self.game_board.board = np.copy(input_board)

        input_board[input_board.astype(bool)] = (
            input_board[input_board.astype(bool)] * 2 - 3
        )

        # Create the input layers
        white_pieces = np.zeros((8, 8))
        white_pieces[input_board == 1] = 1
        black_pieces = np.zeros((8, 8))
        black_pieces[input_board == 2] = 1

        white_pieces = torch.from_numpy(white_pieces).to(device=self.device)
        black_pieces = torch.from_numpy(black_pieces).to(device=self.device)
        player = torch.ones(8, 8, device=self.device) * self.own_player

        x = torch.stack((white_pieces, black_pieces, player)).float()
        x = x.unsqueeze(0)

        # Get output
        with torch.no_grad():
            out = self.net.forward(x)
        out = out.squeeze()
        out = out.cpu()

        # Filter out illegal moves
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
        rng = np.random.default_rng()
        move = rng.choice(moves, p=move_distribution)
        return tuple(move)

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
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                genome = np.append(genome, m.weight.flatten().detach())
                genome = np.append(genome, m.bias.flatten().detach())
        return genome

    @staticmethod
    def genome_to_parameters(genome):
        """Maps genome to a set of parameters"""
        # Hardcoded for 5 residual blocks and 32 filters per conv
        input_param_shapes = [(32, 3, 3, 3), (32)]
        residual_block_param_shapes = [(32, 32, 3, 3), (32)] * 2 * 5
        policy_head_param_shapes = [(2, 32, 1, 1), (2), (64, 128), (64)]
        shapes = (
            input_param_shapes
            + residual_block_param_shapes
            + policy_head_param_shapes
        )
        matrices = []
        for shape in shapes:
            matrix = torch.from_numpy(genome[0 : np.prod(shape)]).reshape(
                shape
            )
            matrices.append(matrix)
            genome = genome[np.prod(shape) :]
        return matrices

    def get_genome(self):
        return self.parameters_to_genome(self.net)

    def set_genome(self, genome):
        matrices = self.genome_to_parameters(genome)
        state_dict = self.net.state_dict()
        for param_name, matrix in zip(state_dict, matrices):
            with torch.no_grad():
                state_dict[param_name] = matrix


class ResidualTowerPolicy(nn.Module):
    """A Residual Tower network with a Policy head, based on AlphaGo Zero's architecture"""

    def __init__(self, num_filters):
        super(ResidualTowerPolicy, self).__init__()

        self.input_layer = nn.Conv2d(3, 32, 3, padding=1)

        self.res_blocks = nn.Sequential(
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
            ResidualBlock(num_filters),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, 1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        """Forwards a flattened board tensor"""
        out = self.input_layer(x)
        out = self.res_blocks(out)
        out = self.policy_head(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, 3, padding=1),
        )

    def forward(self, x):
        out = self.double_conv(x) + x
        relu = nn.ReLU()
        return relu(out)
