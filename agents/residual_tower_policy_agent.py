import numpy as np
import torch
from torch import nn


from agents.genetic_agent import GeneticAgent

NUM_RES_BLOCKS = 2


class ResidualTowerPolicyAgent(GeneticAgent):
    """A neural network agent that..."""

    def __init__(self, player=1, use_cuda=False):
        super(ResidualTowerPolicyAgent, self).__init__(player)
        self.softmax = nn.Softmax(dim=0)
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.net = ResidualTowerPolicy(32).to(self.device)
        self.net.apply(self._initialize_weights)
        self.net.apply(self._initialize_biases)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def _initialize_biases(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
                else:
                    print("Warning, fan_in was calculated to be 0")
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / np.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
                else:
                    print("Warning, fan_in was calculated to be 0")

    def predict(self, input_board):
        """The agent predict the piece to play

        Returns: the move to play
        """
        # Prepare input to network
        input_board = np.copy(input_board)
        self.game_board.board = np.copy(input_board)

        input_board[input_board.astype(bool)] = input_board[input_board.astype(bool)] * 2 - 3

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

    def get_genome(self):
        """Converts the network's parameters to a genome.

        The genome consist of several chromosomes, each of which is all the weights and biases of
        a convolutional or dense layer.

        Return: List of chromosome dicts with the keys "weight" and "bias". The list of chromosomes
            are ordered in the same way as looping through the modules in a network.
        """
        genome = []
        for m in self.net.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                chromosome = {
                    "weight": m.weight.detach().clone(),
                    "bias": m.bias.detach().clone(),
                }
                genome.append(chromosome)

        return genome

    def set_genome(self, genome):
        """Sets the network parameters to the values given by the genome

        Args:
            genome: The genome -- a list of dictionaries, each of which has the keys "weight" and
                "bias".
        """
        learnable_layers = [
            m for m in self.net.modules() if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)
        ]
        if len(learnable_layers) != len(genome):
            raise ValueError(
                f"The number of learnable layers in the model ({len(learnable_layers)}) is not "
                f"equal to the number of chromosomes in the genome ({len(genome)})"
            )
        for layer, chromosome in zip(learnable_layers, genome):
            layer.weight = nn.Parameter(chromosome["weight"])
            layer.bias = nn.Parameter(chromosome["bias"])


class ResidualTowerPolicy(nn.Module):
    """A Residual Tower network with a Policy head, based on AlphaGo Zero's architecture"""

    def __init__(self, num_filters):
        super(ResidualTowerPolicy, self).__init__()

        self.input_layer = nn.Conv2d(3, 32, 3, padding=1)

        self.res_blocks = nn.Sequential(
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
