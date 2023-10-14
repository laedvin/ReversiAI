from abc import abstractmethod
from reversi_ai.agents.basic_agent import Agent


class GeneticAgent(Agent):
    def __init__(self, player=1):
        super(GeneticAgent, self).__init__(player)

    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def get_genome(self):
        """Converts the network's parameters to a genome.

        The genome consist of several chromosomes, each of which is all the weights and biases of
        a convolutional or dense layer.

        Return: List of chromosome dicts with the keys "weight" and "bias". The list of chromosomes
            are ordered in the same way as looping through the modules in a network.
        """
        pass

    @abstractmethod
    def set_genome(self, genome):
        """Sets the network parameters to the values given by the genome

        Args:
            genome: The genome -- a list of dictionaries, each of which has the keys "weight" and
                "bias".
        """
        pass
