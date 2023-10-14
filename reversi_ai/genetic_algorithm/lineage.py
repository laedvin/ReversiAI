import os
from os.path import abspath, join
import json
import torch
from copy import deepcopy
from pathlib import Path
import glob
from timeit import default_timer as timer
import numpy as np
from reversi_ai.genetic_algorithm.population import Population


class Lineage:
    """Lineage of a species

    This class tracks the populations of several generations of a species.
    The resulting descendant populations from an initial population is a
    lineage.

    Stores a lineage in a directory. The path to the directory is the unique
    identifier of a lineage.
    """

    def __init__(self, path, config=None):
        """Initializes or loads a lineage

        If config isn't given, it will attempt to find a config.json in path.
        If there isn't a config.json, a new file will be created with some
        default values.

        Specification of config:
            pop_size: population size
            mutation_rate: mutation rate of genome
            crossover: ?
            round_robin_rounds: number of round robin rounds to play in each
                generation
            fitness_factor: WIP, some kind of factor that determines the rate
                of reproduction given fitness (i.e. given Elo).

            Should also include some neural network configuration

        Args:
            path: absolute path to a lineage
            config: optional dictionary with some hyperparameter configuration
        """
        super(Lineage, self).__init__()
        self.path = path
        Path(self.path).mkdir(parents=True, exist_ok=True)

        if os.path.isfile(join(path, "config.json")):
            with open(join(path, "config.json"), "r") as f:
                print("Loading config from existing config file.")
                self.config = json.load(f)
        elif config:
            self.config = config
            print("Creating new config file.")
            with open(join(path, "config.json"), "w") as f:
                json.dump(self.config, f, indent=2)
        else:
            self.config = {
                "pop_size": 25,
                "num_childs": 5,
                "mutation_rate": 0.02,
                "round_robin_rounds": 8,
                "placement_matches": 5,
                "adjustment_matches": 5,
                "elo_attractiveness": 20,
                "k_factor_rr": 20,  # Round robin
                "k_factor_p": 40,  # Placement
                "initial_elo": 1000,
                "elo_floor": 100,
            }
            print("Creating new config file using default values.")
            with open(join(path, "config.json"), "w") as f:
                json.dump(self.config, f)

        # Determine if there exist saved generations
        generations = glob.glob(join(self.path, "generation_*"))
        if generations:
            generations = [Path(p).stem for p in generations]
            generations = [int(name.split("_")[1]) for name in generations]
            self.current_gen = max(generations)
            self.current_pop = self.get_pop_from_gen(self.current_gen)
        else:
            print("Creating generation 0")
            self.current_gen = 0
            initial_elo = self.config["initial_elo"]
            self.current_pop = Population(self.config, initial_elo=initial_elo)
            self.determine_population_elo()
            self.save_current_generation()

    def save_current_generation(self):
        """Save the population of the current generation"""
        generation_path = abspath(join(self.path, f"generation_{self.current_gen}.npz"))
        with open(generation_path, "wb") as f:
            np.savez_compressed(f, self.current_pop.pop)

    def get_pop_from_gen(self, generation_id):
        """Get the population from a given generation"""
        generation_path = abspath(join(self.path, f"generation_{generation_id}.npz"))
        with open(generation_path, "rb"):
            individuals = np.load(generation_path, allow_pickle=True)["arr_0"]
        return Population(self.config, existing_population=individuals)

    def determine_population_elo(self):
        """Determine the Elo ratings of the individuals in the population"""
        start = timer()
        # Find best previous agent
        baseline_individual = None
        if self.current_gen > 0:
            old_pop = self.get_pop_from_gen(self.current_gen - 1)
            old_elos = [individual["elo"] for individual in old_pop.pop]
            baseline_individual = old_pop.pop[np.argmax(old_elos)]

        print("Performing placement matches")
        self.current_pop.placement_matches(
            self.config["placement_matches"],
            baseline_individual=baseline_individual,
        )

        print("Performing round robin matches")
        self.current_pop.round_robin(self.config["round_robin_rounds"])

        print("Performing adjustment matches")
        self.current_pop.placement_matches(
            self.config["adjustment_matches"],
            baseline_individual=baseline_individual,
        )
        end = timer()
        print(f"Elo determination took {end-start} seconds")
        elos = [individual["elo"] for individual in self.current_pop.pop]
        average_elo = np.mean(elos)
        print(f"Average Elo for generation {self.current_gen} is {average_elo}")
        best = self.current_pop.pop[np.argmax(elos)]
        worst = self.current_pop.pop[np.argmin(elos)]
        print(
            "The best individual was:\n"
            f"elo: {best['elo']}, wins: {best['wins']}, losses: {best['losses']}, "
            f"wins as white: {best['white']['wins']}, losses as white: {best['white']['losses']}, "
            f"wins as black: {best['black']['wins']}, losses as black: {best['black']['losses']} "
        )
        print(
            "The worst individual was:\n"
            f"elo: {worst['elo']}, wins: {worst['wins']}, losses: {worst['losses']}, "
            f"wins as white: {worst['white']['wins']}, losses as white: {worst['white']['losses']}, "
            f"wins as black: {worst['black']['wins']}, losses as black: {worst['black']['losses']} "
        )

    def advance_generation(self):
        # Find the top (pop_size - offspring) individuals by Elo
        ids, elos = zip(*[(x["id"], x["elo"]) for x in self.current_pop.pop])
        ids = np.array(list(ids))
        elos = np.array(list(elos))
        elo_sorted_idx = (-elos).argsort()
        elo_sorted_ids = ids[elo_sorted_idx]
        num_survivors = self.config["pop_size"] - self.config["num_childs"]

        print(f"Population Elo, descending order: {elos[elo_sorted_idx]}")

        top_genomes = [self.current_pop.pop[i]["genome"] for i in elo_sorted_ids[0:num_survivors]]

        # Create children
        mating_pairs = self.select_mating_pairs(self.config["num_childs"])
        child_genomes = [self.reproduce(*pair) for pair in mating_pairs]

        self.current_gen += 1

        print(f"Creating generation {self.current_gen}")
        self.current_pop = Population(self.config, initial_elo=self.config["initial_elo"])
        self.current_pop.update_genome(top_genomes + child_genomes)

        self.determine_population_elo()
        self.save_current_generation()

    def reproduce(self, id_a, id_b):
        """Generates the genome of a child from two parents.

        This process includes chromosome selection and mutation

        Args:
            id_a: id of the first parent
            id_b: id of the second parent

        Returns: The genome of the child
        """

        # Chromosome selection
        genome_a = self.current_pop.pop[id_a]["genome"]
        genome_b = self.current_pop.pop[id_b]["genome"]
        parent_genomes = (genome_a, genome_b)
        num_chromosomes = len(genome_a)
        chromosome_choice = np.random.choice(a=[0, 1], size=(num_chromosomes,))
        child_genome = [
            deepcopy(parent_genomes[parent][chromosome])
            for (chromosome, parent) in enumerate(chromosome_choice)
        ]

        # Mutation. Change the values of some parameter to that of a newly initialized agent
        # This will ensure that the mutated parameters are chosen from the correct distribution
        random_genome = self.current_pop.initialize_agent().get_genome()
        for chromosome, random_chromosome in zip(child_genome, random_genome):
            mutated_weights = torch.rand(chromosome["weight"].shape) < self.config["mutation_rate"]
            mutated_biases = torch.rand(chromosome["bias"].shape) < self.config["mutation_rate"]
            chromosome["weight"][mutated_weights] = random_chromosome["weight"][mutated_weights]
            chromosome["bias"][mutated_biases] = random_chromosome["bias"][mutated_biases]

        return child_genome

    def select_mating_pairs(self, num_pairs):
        """Select which individuals should mate

        Picks the first half of a pair from a Boltzmann distribution using Elo
        as energy and with a temperature depending on elo_attractiveness.

        Picks the second half of a pair in the same way, but it can't repick
        the first half.

        Args:
            num_pairs: The number of pairs to pick

        Returns: A list of tuples containing the ids of the mating pairs
        """
        elo_attractiveness = self.config["elo_attractiveness"]
        ids, elos = zip(*[(x["id"], x["elo"]) for x in self.current_pop.pop])
        ids = np.array(list(ids))
        elos = np.array(list(elos))

        # Calculate weights using Boltzmann distribution
        beta = np.log(10) / 400 * elo_attractiveness
        z = np.sum(np.exp(beta * elos))
        p = np.exp(beta * elos) / z

        first_halfs = np.random.choice(ids, num_pairs, p=p)
        pairs = []
        for first_half in first_halfs:
            mask = np.ones(ids.size, dtype=bool)
            mask[first_half] = False
            partner_ids = ids[mask]
            partner_elos = elos[mask]
            partner_z = np.sum(np.exp(beta * partner_elos))
            partner_p = np.exp(beta * partner_elos) / partner_z
            second_half = np.random.choice(partner_ids, 1, p=partner_p)[0]
            pairs.append((first_half, second_half))
        return pairs
