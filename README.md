# ReversiAI

Reversi and some attempts to make an AI

## Requirements

Install the required requirements in the requirements.txt required to run the scripts with `pip` with the command

`pip install -r requirements.txt`

## Ways to run the code

There are two kinds of main scripts: ones that start a training session and ones that let you or an AI agent play.

### Play the game


To play with two human players in the terminal (Inputs are coordinates separated by a space; I am not responsible for figuring out how the input coordinates translates to the shown game board)
`python human_vs_human.py`

To see the result of agents playing against each other (for specific configurations, see the script itself):
`python agent_vs_agent.py`

To play with two human players in a graphical application:
`python start_game.py`


### Train AI

The two scripts `genetic_training.py` and `imitation_training.py` can be used to train agents. The agents are saved to the directory specified in the respective script.

### Use trained AI

...