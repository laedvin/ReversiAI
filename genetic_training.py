from os.path import abspath, join, dirname
import deepdish as dd

from genetic_algorithm.lineage import Lineage


def main():
    lineage = Lineage(
        abspath(
            join(dirname(__file__), "genetic_algorithm/lineages/test_lineage/")
        )
    )
    lineage.advance_generation()
    print(f"Saved generation {lineage.current_gen}")


if __name__ == "__main__":
    main()
