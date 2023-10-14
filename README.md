# ReversiAI

Reversi and some attempts to make an AI

## Installation

Install virtual environment using `poetry`:

```{code:bash}
poetry install
```

## Ways to run the code

There are two kinds of main scripts: ones that start a training session and ones that let you or an AI agent play.

### Play the game

To play, use

```{code:bash}
reversi-ai play
```

### Train AI

The two scripts `genetic_training.py` and `imitation_training.py` can be used to train agents. The agents are saved to the directory specified in the respective script.

### Use trained AI

...