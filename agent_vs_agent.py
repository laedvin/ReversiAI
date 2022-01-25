import numpy as np
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

from reversi.reversi_environment import ReversiEnvironment
from reversi.gameplay_utils import play_match
from agents.random_agent import RandomAgent
from agents.naive_agent import NaiveAgent
from agents.branch_aggregate_agent import BranchAggregateAgent
from agents.simple_nn_agent import SimpleNNAgent
from genetic_algorithm.population import Population


WHITE = 1
BLACK = 2
NUM_CPUS = cpu_count()


def main():
    player_a = NaiveAgent()
    player_b = NaiveAgent()
    pa_elo = 500
    pb_elo = 500
    pa_wins = 0
    pb_wins = 0
    matches = 1000

    start = timer()
    with Pool(NUM_CPUS) as pool:
        result = pool.starmap(
            play_match,
            [(player_a, player_b) for i in range(matches)],
        )
    for score_a, score_b in result:
        final_score = (score_a + (1 - score_b)) / 2
        da_elo, db_elo = Population.calculate_elo(
            pa_elo, pb_elo, final_score, k_factor=10
        )
        pa_elo += da_elo
        pb_elo += db_elo

        pa_wins += score_a if score_a != 0.5 else 0
        pb_wins += 1 - score_a if score_a != 0.5 else 0

        pa_wins += 1 - score_b if score_b != 0.5 else 0
        pb_wins += score_b if score_b != 0.5 else 0

    end = timer()
    print(
        f"Agent A (Elo: {pa_elo:.3f}) | "
        f"{pa_wins} -- {2*matches - pa_wins - pb_wins} -- {pb_wins} | "
        f"Agent B (Elo: {pb_elo:.3f})"
    )
    print(f"Elapsed time: {end-start:.3f}s")


if __name__ == "__main__":
    main()
