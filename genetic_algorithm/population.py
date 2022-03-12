import itertools
import random
import numpy as np
from multiprocessing import Pool, cpu_count

from reversi.gameplay_utils import play_game
from agents.residual_tower_policy_agent import ResidualTowerPolicyAgent
from agents.random_agent import RandomAgent
from agents.naive_agent import NaiveAgent


NUM_CPUS = min(6, cpu_count())


class Population:
    def __init__(self, config, initial_elo=1000, existing_population=None):
        super(Population, self).__init__()
        self.config = config
        if existing_population is not None:
            if len(existing_population) == self.config["pop_size"]:
                self.pop = existing_population
            else:
                raise ValueError(
                    f"Pop size of existing population does not match the pop "
                    f"size in the config file; {len(existing_population)} vs {self.config['pop_size']}"
                )
        else:
            self.pop = self.initialize_population(self.config["pop_size"], initial_elo)

    def placement_matches(self, n, baseline_individual=None):
        """Every individual plays n matches against a baseline agent.

        Each match includes one game as white and one game as black.

        Updates only Elo.

        Args:
            n: the number of rounds to play
            baseline_individual: A specific individual to measure Elo against.
                If None, a naive agent will be used.
        """
        genomes = [individual["genome"] for individual in self.pop]
        individual_ids = [individual["id"] for individual in self.pop] * n
        with Pool(NUM_CPUS) as pool:
            if baseline_individual:
                baseline_elo = baseline_individual["elo"]
                baseline_genome = baseline_individual["genome"]
                result = pool.starmap(
                    self.play_match,
                    [(id, -1, genomes[id], baseline_genome) for id in individual_ids],
                )
                # Format the resulting list of tuples so it aligns with the other case
                result = [
                    (id, score_as_white, score_as_black)
                    for id, _, score_as_white, score_as_black in result
                ]
            else:
                result = pool.starmap(
                    self.play_match_vs_naive,
                    [(id, genomes[id]) for id in individual_ids],
                )
                baseline_elo = self.config["initial_elo"]
        for id, score_as_white, score_as_black in result:
            score = (score_as_white + 1 - score_as_black) / 2
            elo_delta, _ = self.calculate_elo(
                self.pop[id]["elo"],
                baseline_elo,
                score,
                k_factor=self.config["k_factor_p"],
            )
            self.pop[id]["elo"] = max(self.pop[id]["elo"] + elo_delta, self.config["elo_floor"])

    def round_robin(self, n):
        """Plays n rounds of round robin

        In each round of round robin, each individual plays one match
        (two games) against each of the other individuals in the population.
        The players in a match take turn in playing as white and black.

        Their stats and Elo will be updated.

        Args:
            n: the number of rounds to play

        """
        genomes = [individual["genome"] for individual in self.pop]
        individual_ids = [individual["id"] for individual in self.pop]
        pairings = []
        for i in range(n):
            partial_pairings = list(itertools.permutations(individual_ids, r=2))
            random.shuffle(partial_pairings)
            pairings += partial_pairings

        # Play a round robin round
        with Pool(NUM_CPUS) as pool:
            result = pool.starmap(
                self.play_match,
                [(a, b, genomes[a], genomes[b]) for a, b in pairings],
            )
        for id_a, id_b, score_a, score_b in result:
            self.update_stats(id_a, id_b, score_a, a_is_white=True)
            self.update_stats(id_a, id_b, 1 - score_b, a_is_white=False)

            # Update Elo
            final_score = (score_a + (1 - score_b)) / 2
            elo_a_delta, elo_b_delta = self.calculate_elo(
                self.pop[id_a]["elo"],
                self.pop[id_b]["elo"],
                final_score,
                k_factor=self.config["k_factor_rr"],
            )
            self.pop[id_a]["elo"] = max(
                self.pop[id_a]["elo"] + elo_a_delta,
                self.config["elo_floor"],
            )
            self.pop[id_b]["elo"] = max(
                self.pop[id_b]["elo"] + elo_b_delta,
                self.config["elo_floor"],
            )

    def update_stats(self, id_a, id_b, score_a, a_is_white=True):
        """Updates stats for player A and B for a single game within a match

        Args:
            id_a: id of player A
            id_b: id of player B
            score_a: 1 if A won, 0 if B won, 0.5 if draw
            a_is_white: Whether or not A played as white
        """
        if a_is_white:
            color_a = "white"
            color_b = "black"
        else:
            color_a = "black"
            color_b = "white"

        self.pop[id_a]["games"] += 1
        self.pop[id_a][color_a]["games"] += 1
        self.pop[id_b]["games"] += 1
        self.pop[id_b][color_b]["games"] += 1

        if score_a == 0.5:
            self.pop[id_a]["draws"] += 1
            self.pop[id_a][color_a]["draws"] += 1
            self.pop[id_b]["draws"] += 1
            self.pop[id_b][color_b]["draws"] += 1
        else:
            self.pop[id_a]["wins"] += score_a
            self.pop[id_a]["losses"] += 1 - score_a
            self.pop[id_a][color_a]["wins"] += score_a
            self.pop[id_a][color_a]["losses"] += 1 - score_a
            self.pop[id_b]["wins"] += 1 - score_a
            self.pop[id_b]["losses"] += score_a
            self.pop[id_b][color_b]["wins"] += 1 - score_a
            self.pop[id_b][color_b]["losses"] += score_a

    def update_genome(self, genomes):
        """Updates the genome of every individual

        Args:
            genomes: A list of genomes matching the pop size
        """
        if not (len(genomes) == self.config["pop_size"]):
            raise ValueError(
                f"Number of genomes and individuals don't match; {len(genomes)} genomes vs "
                f"{self.config['pop_size']} individuals"
            )
        for idx, genome in enumerate(genomes):
            self.pop[idx]["genome"] = genome

    @staticmethod
    def calculate_elo(elo_x, elo_y, s_x, k_factor=20):
        """Calculates the Elo update for player X vs Y given score for X

        The Elo should be updated after a whole match (2 games where X and Y)
        take turn playing white and black.

        Args:
            elo_x: The Elo of player X
            elo_y: The Elo of player Y
            s_x: The winrate for X in a single match
            k_factor: How much the Elo adjusts

        Returns: The Elo changes for player X and player Y
        """
        # Score
        s_y = 1 - s_x

        # Calculated expected score
        q_x = 10 ** (elo_x / 400)
        q_y = 10 ** (elo_y / 400)
        e_x = q_x / (q_x + q_y)
        e_y = q_y / (q_x + q_y)

        elo_x_delta = (s_x - e_x) * k_factor
        elo_y_delta = (s_y - e_y) * k_factor

        return elo_x_delta, elo_y_delta

    @staticmethod
    def play_match(id_a, id_b, genome_a, genome_b):
        agent_a = ResidualTowerPolicyAgent()
        agent_b = ResidualTowerPolicyAgent()
        agent_a.set_genome(genome_a)
        agent_b.set_genome(genome_b)
        score_a = play_game(agent_a, agent_b)
        score_b = play_game(agent_b, agent_a)
        return id_a, id_b, score_a, score_b

    @staticmethod
    def play_match_vs_random(id, genome):
        agent_a = ResidualTowerPolicyAgent()
        agent_b = RandomAgent()
        agent_a.set_genome(genome)
        score_a = play_game(agent_a, agent_b)
        score_b = play_game(agent_b, agent_a)
        return id, score_a, score_b

    @staticmethod
    def play_match_vs_naive(id, genome):
        agent_a = ResidualTowerPolicyAgent()
        agent_b = NaiveAgent()
        agent_a.set_genome(genome)
        score_a = play_game(agent_a, agent_b)
        score_b = play_game(agent_b, agent_a)
        return id, score_a, score_b

    @staticmethod
    def initialize_population(pop_size, initial_elo=1000):
        """Initialize population for the residual tower policy agent

        Sets their genome and Elo (which is to be used as fitness)

        """
        pop = []
        for id in range(pop_size):
            sample_agent = ResidualTowerPolicyAgent()
            genome = sample_agent.get_genome()
            individual = {
                "id": id,
                "genome": genome,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "white": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
                "black": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
                "elo": initial_elo,
            }
            pop.append(individual)
        return np.array(pop)

    @staticmethod
    def initialize_agent():
        """Initializes a random (residual tower policy) agent"""
        return ResidualTowerPolicyAgent()
