from os.path import abspath, join, dirname
import numpy as np
from genetic_algorithm.lineage import Lineage
from genetic_algorithm.population import Population


def main():
    """Evaluates all the generations in a lineage"""
    path_to_lineage = abspath(join(dirname(__file__), "genetic_algorithm/lineages/res_tower/"))
    lineage = Lineage(path_to_lineage)
    config = {
        "pop_size": lineage.current_gen,
        "mutation_rate": 0.02,
        "mutation_var": 0.1,
        "crossover_rate": 0.9,
        "round_robin_rounds": 5,
        "placement_matches": 5,
        "adjustment_matches": 5,
        "elo_attractiveness": 20,
        "k_factor_rr": 20,  # Round robin
        "k_factor_p": 40,  # Placement
        "initial_elo": 500,
        "elo_floor": 100,
    }
    # Find the best individual from each generation
    best_individuals = []
    for generation in range(lineage.current_gen):
        pop = lineage.get_pop_from_gen(generation)
        elos = [individual["elo"] for individual in pop.pop]
        best_individual = pop.pop[np.argmax(elos)]
        # Reset non-genome stats
        best_individual["id"] = generation
        best_individual["elo"] = config["initial_elo"]
        best_individual["games"] = 0
        best_individual["wins"] = 0
        best_individual["losses"] = 0
        best_individual["draws"] = 0
        best_individual["white"] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        best_individual["black"] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        best_individuals.append(best_individual)
    evaluation_population = Population(
        config, config["initial_elo"], existing_population=best_individuals
    )
    print("Playing round robin matches")
    evaluation_population.round_robin(config["round_robin_rounds"])
    print(evaluation_population.pop)


if __name__ == "__main__":
    main()
