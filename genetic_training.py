from os.path import abspath, join, dirname

from genetic_algorithm.lineage import Lineage


def main():
    config = {
        "pop_size": 25,
        "num_childs": 20,
        "mutation_rate": 0.02,
        "round_robin_rounds": 8,
        "placement_matches": 0,
        "adjustment_matches": 0,
        "elo_attractiveness": 20,
        "k_factor_rr": 20,  # Round robin
        "k_factor_p": 40,  # Placement
        "initial_elo": 500,
        "elo_floor": 100,
    }
    lineage = Lineage(
        abspath(join(dirname(__file__), "genetic_algorithm/lineages/test/")),
        config=config,
    )
    while lineage.current_gen < 300:
        lineage.advance_generation()
        print(f"Saved generation {lineage.current_gen}")


if __name__ == "__main__":
    main()
