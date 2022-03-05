from os.path import abspath, join, dirname
import deepdish as dd

from imitation_learning.imitation_learning import ImitationSession
from agents.simple_nn_agent import SimpleNNAgent
from agents.naive_agent import NaiveAgent


def main():
    student = SimpleNNAgent()
    teacher = NaiveAgent()
    path = abspath(
        join(dirname(__file__), "imitation_learning/models/test_imitation/")
    )
    imitation_session = ImitationSession(student, teacher, path)
    imitation_session.train(500, 20, 200, lr=0.01)


if __name__ == "__main__":
    main()
