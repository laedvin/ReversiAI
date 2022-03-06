from os.path import abspath, join, dirname

from genetic_algorithm.lineage import Lineage


def main():
    lineage = Lineage(abspath(join(dirname(__file__), "genetic_algorithm/lineages/res_tower/")))
    while lineage.current_gen < 200:
        lineage.advance_generation()
        print(f"Saved generation {lineage.current_gen}")


if __name__ == "__main__":
    main()
