import copy
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from os.path import join, abspath

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from reversi.gameplay_utils import play_game_nn_train


class ImitationSession:
    def __init__(self, student, teacher, path):
        """Initializes an imitation learning session

        Args:
            student: the student agent that will learn
            teacher: the teacher agent that will teach
            path: path to save models

        """
        super(ImitationSession, self).__init__()
        self.student = student
        self.teacher = teacher
        self.path = path
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def train(self, games, validation_games, epochs, lr=0.001):
        """Trains the student agent

        In each epoch of training, the student agent will play multiple games
        against itself. The teacher agent will evaluate each board state and
        suggest which move to make. These suggested moves are onehot vectors
        that are treated as labels for their respective input state. At the end
        of each epoch, the student performs backpropagation given the suggested
        moves.

        Args:
            games: games to play per epoch
            epochs: epochs to train for

        Returns: ?
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.student.net.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"Training on epoch {epoch}")
            optimizer.zero_grad()
            player_a = copy.deepcopy(self.student)
            player_b = copy.deepcopy(self.student)
            player_a.net = self.student.net
            player_b.net = self.student.net

            train_states = []
            train_player_turns = []
            train_predicted_moves = []
            train_target_moves = []
            train_raw_outputs = torch.zeros(0, 64)

            val_states = []
            val_player_turns = []
            val_predicted_moves = []
            val_target_moves = []
            val_raw_outputs = torch.zeros(0, 64)

            for game in range(games):
                (
                    _,
                    game_states,
                    game_player_turns,
                    game_predicted_moves,
                    game_raw_outputs,
                ) = play_game_nn_train(player_a, player_b)

                train_states += game_states
                train_player_turns += game_player_turns
                train_predicted_moves += game_predicted_moves
                train_raw_outputs = torch.cat(
                    (train_raw_outputs, game_raw_outputs)
                )

            for state, player in zip(train_states, train_player_turns):
                self.teacher.set_player(player)
                train_target_moves.append(self.teacher.predict(state))

            train_states = torch.tensor(np.array(train_states))
            train_player_turns = torch.tensor(train_player_turns)
            train_predicted_moves = torch.tensor(train_predicted_moves)
            train_target_moves = np.array(train_target_moves)

            # Do some data formatting
            train_states = torch.flatten(train_states, start_dim=1)
            train_states = torch.cat(
                (
                    train_states,
                    (train_player_turns * 2 - 3).reshape(
                        (train_states.shape[0], 1)
                    ),
                ),
                dim=1,
            )

            train_target_moves = np.ravel_multi_index(
                (train_target_moves[:, 0], train_target_moves[:, 1]), (8, 8)
            )
            train_target_moves = torch.from_numpy(train_target_moves)

            # Train on training data
            loss = criterion(train_raw_outputs, train_target_moves)
            loss.backward()
            optimizer.step()

            # Play validation games
            play_validation_game = partial(play_game_nn_train, detach=True)
            with Pool() as pool:
                with torch.no_grad():
                    result = pool.starmap(
                        play_validation_game,
                        [(player_a, player_b)] * validation_games,
                    )
            for (
                _,
                game_states,
                game_player_turns,
                game_predicted_moves,
                game_raw_outputs,
            ) in result:
                val_states += game_states
                val_player_turns += game_player_turns
                val_predicted_moves += game_predicted_moves
                val_raw_outputs = torch.cat(
                    (val_raw_outputs, game_raw_outputs)
                )

            for state, player in zip(val_states, val_player_turns):
                self.teacher.set_player(player)
                val_target_moves.append(self.teacher.predict(state))

            val_states = torch.tensor(np.array(val_states))
            val_player_turns = torch.tensor(val_player_turns)
            val_predicted_moves = torch.tensor(val_predicted_moves)
            val_target_moves = np.array(val_target_moves)

            # Do some data formatting
            val_states = torch.flatten(val_states, start_dim=1)
            val_states = torch.cat(
                (
                    val_states,
                    (val_player_turns * 2 - 3).reshape(
                        (val_states.shape[0], 1)
                    ),
                ),
                dim=1,
            )

            val_target_moves = np.ravel_multi_index(
                (val_target_moves[:, 0], val_target_moves[:, 1]), (8, 8)
            )
            val_target_moves = torch.from_numpy(val_target_moves)

            # Validate on validation data
            loss = criterion(val_raw_outputs, val_target_moves)
            print(f"Validation loss: {loss}")
            print(f"Genome mean: {np.mean(self.student.get_genome())}")
            torch.save(
                self.student.net.state_dict(),
                abspath(join(self.path, f"model_epoch_{epoch}.pt")),
            )
