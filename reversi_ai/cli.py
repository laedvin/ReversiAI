import click
from reversi_ai.play_game import init_agent, play_game


@click.group()
def cli():
    pass


@cli.group()
def play():
    pass


@play.command()
def ai():
    agent = init_agent(True, "")
    play_game(agent=agent)


@play.command()
def pvp():
    play_game()


@cli.command()
def train():
    print("train an ai!")
