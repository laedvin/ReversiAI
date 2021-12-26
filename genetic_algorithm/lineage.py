import os
from os.path import abspath, join
import json
from pathlib import Path
import glob
import numpy as np
import deepdish as dd
from genetic_algorithm.population import Population


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

        self.config = config
        if self.config is not None:
            with open(join(path, "config.json"), "w") as f:
                json.dump(self.config, f)
        elif os.path.isfile(join(path, "config.json")):
            with open(join(path, "config.json"), "r") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "pop_size": 50,
                "mutation_rate": 1,
                "crossover": 1,
                "round_robin_rounds": 2,
                "placement_matches": 10,
                "elo_attractiveness": 2,
                "k_factor_rr": 20,  # Round robin
                "k_factor_p": 40,  # Placement
                "random_agent_elo": 500,
                "elo_floor": 100,
            }
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
            initial_elo = self.config["random_agent_elo"]
            self.current_pop = Population(self.config, initial_elo=initial_elo)
            self.determine_population_elo()
            self.save_current_generation()

    def save_current_generation(self):
        """Save the population of the current generation"""
        generation_path = abspath(
            join(self.path, f"generation_{self.current_gen}.h5")
        )
        dd.io.save(generation_path, self.current_pop.pop)

    def get_pop_from_gen(self, generation_id):
        """Get the population from a given generation"""
        generation_path = abspath(
            join(self.path, f"generation_{generation_id}.h5")
        )
        individuals = dd.io.load(generation_path)
        return Population(self.config, existing_population=individuals)

    def determine_population_elo(self):
        print("Performing placement matches")
        self.current_pop.placement_matches(self.config["placement_matches"])
        print("Performing round robin matches")
        self.current_pop.round_robin(self.config["round_robin_rounds"])

    def advance_generation(self):
        average_elo = np.mean(
            [individual["elo"] for individual in self.current_pop.pop]
        )
        initial_elo = (average_elo + self.config["random_agent_elo"]) / 2
        self.current_gen += 1
        mating_pairs = self.select_mating_pairs()
        new_genomes = [self.reproduce(*pair) for pair in mating_pairs]

        print(f"Simulating generation {self.current_gen}")
        self.current_pop = Population(self.config, initial_elo=initial_elo)
        1 / 0
        self.determine_population_elo()
        self.save_current_generation()

    def reproduce(self, id_a, id_b):
        """Generates the genome of a child from two parents

        Args:
            id_a: id of the first parent
            id_b: id of the second parent

        Retunrs: The genome of the child
        """
        return self.current_pop[id_a]["genome"]

    def select_mating_pairs(self):
        """Select which individuals should mate

        Picks the first half of a pair from a Boltzmann distribution using Elo
        as energy and with a temperature depending on elo_attractiveness.

        Picks the second half of a pair in the same way, but it can't repick
        the first half.

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

        first_halfs = np.random.choice(ids, ids.size, p=p)
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
