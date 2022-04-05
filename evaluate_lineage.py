from os.path import abspath, join, dirname
from multiprocessing import Pool
import numpy as np
from genetic_algorithm.lineage import Lineage
from genetic_algorithm.population import Population

CONFIG = {
    "pop_size": 100,
    "mutation_rate": 0.02,
    "mutation_var": 0.1,
    "round_robin_rounds": 5,
    "control_matches": 20,
    "adjustment_matches": 5,
    "elo_attractiveness": 20,
    "k_factor_rr": 20,  # Round robin
    "k_factor_p": 40,  # Placement
    "initial_elo": 500,
    "elo_floor": 100,
}

MAX_EVALUATION_POP_SIZE = 20


def main():
    """Evaluates all the generations in a lineage"""
    path_to_lineage = abspath(
        join(dirname(__file__), "genetic_algorithm/lineages/no_baseline_agents/")
    )
    lineage = Lineage(path_to_lineage)

    # Decide which generations to evaluate
    CONFIG["pop_size"] = min(lineage.current_gen, MAX_EVALUATION_POP_SIZE)

    if CONFIG["pop_size"] == MAX_EVALUATION_POP_SIZE:
        generations = np.round(np.linspace(0, lineage.current_gen, MAX_EVALUATION_POP_SIZE)).astype(
            int
        )
    else:
        generations = range(CONFIG["pop_size"])
    # Find the best individual from each generation
    best_individuals = []
    for idx, generation in enumerate(generations):
        pop = lineage.get_pop_from_gen(generation)
        elos = [individual["elo"] for individual in pop.pop]
        best_individual = pop.pop[np.argmax(elos)]
        # Reset non-genome stats
        best_individual["id"] = idx
        best_individual["elo"] = CONFIG["initial_elo"]
        best_individual["games"] = 0
        best_individual["wins"] = 0
        best_individual["losses"] = 0
        best_individual["draws"] = 0
        best_individual["white"] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        best_individual["black"] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        best_individuals.append(best_individual)
    evaluation_population = Population(
        CONFIG, CONFIG["initial_elo"], existing_population=best_individuals
    )
    # Do some evaluations
    print("Playing matches vs Random Agent")
    evaluate_vs_control(
        evaluation_population, CONFIG["control_matches"], Population.play_match_vs_random
    )
    print_stats(evaluation_population)

    print("Resetting stats\n")
    reset_stats(evaluation_population)

    print("Playing matches vs Naive Agent")
    evaluate_vs_control(
        evaluation_population, CONFIG["control_matches"], Population.play_match_vs_naive
    )
    print_stats(evaluation_population)

    print("Resetting stats\n")
    reset_stats(evaluation_population)

    print("Playing round robin matches")
    evaluation_population.round_robin(CONFIG["round_robin_rounds"])
    print_stats(evaluation_population)


def reset_stats(population):
    for individual in population.pop:
        individual["elo"] = CONFIG["initial_elo"]
        individual["games"] = 0
        individual["wins"] = 0
        individual["losses"] = 0
        individual["draws"] = 0
        individual["white"] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        individual["black"] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}


def print_stats(population):
    elo = [i["elo"] for i in population.pop]
    winrate_white = [
        i["white"]["wins"] / (i["white"]["wins"] + i["white"]["losses"]) for i in population.pop
    ]
    winrate_black = [
        i["black"]["wins"] / (i["black"]["wins"] + i["black"]["losses"]) for i in population.pop
    ]
    winrate_overall = [i["wins"] / (i["wins"] + i["losses"]) for i in population.pop]

    print("Resulting Elo ordered by generation")
    print([f"{x:0.2f}" for x in elo])
    print()
    print("Resulting white winrate ordered by generation")
    print([f"{x:0.2f}" for x in winrate_white])
    print()
    print("Resulting black winrate ordered by generation")
    print([f"{x:0.2f}" for x in winrate_black])
    print()
    print("Resulting overall winrate ordered by generation")
    print([f"{x:0.2f}" for x in winrate_overall])
    print("\n")


def evaluate_vs_control(population, n_matches, play_match_func):
    genomes = [individual["genome"] for individual in population.pop]
    individual_ids = [individual["id"] for individual in population.pop] * n_matches
    with Pool(6) as pool:
        result = pool.starmap(
            play_match_func,
            [(id, genomes[id]) for id in individual_ids],
        )
        baseline_elo = CONFIG["initial_elo"]
    for id, score_as_white, score_as_black in result:
        update_stats_vs_control(population, id, score_as_white, is_white=True)
        update_stats_vs_control(population, id, 1 - score_as_black, is_white=False)

        # For Elo calculation
        score = (score_as_white + 1 - score_as_black) / 2
        elo_delta, _ = population.calculate_elo(
            population.pop[id]["elo"],
            baseline_elo,
            score,
            k_factor=CONFIG["k_factor_p"],
        )
        population.pop[id]["elo"] = max(population.pop[id]["elo"] + elo_delta, CONFIG["elo_floor"])


def update_stats_vs_control(population, id, score, is_white=True):
    """Updates stats for player A and B for a single game within a match

    Args:
        id: id of player
        score: 1 if player won, 0 if player lost, 0.5 if draw
        is_white: Whether or not the player played as white
    """
    if is_white:
        color_a = "white"
    else:
        color_a = "black"

    population.pop[id]["games"] += 1
    population.pop[id][color_a]["games"] += 1

    if score == 0.5:
        population.pop[id]["draws"] += 1
        population.pop[id][color_a]["draws"] += 1
    else:
        population.pop[id]["wins"] += score
        population.pop[id]["losses"] += 1 - score
        population.pop[id][color_a]["wins"] += score
        population.pop[id][color_a]["losses"] += 1 - score


if __name__ == "__main__":
    main()
