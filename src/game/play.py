from src.game.env_init import game_init
from random import choice
import time

from src.models.random_action import RandomActionModel


def play(config_file, tickrate=35):
    game = game_init(config_file, 'test')
    sleep_rate = 1 / tickrate

    episodes = 1

    actions = game.get_available_buttons()
    available_variables = game.get_available_game_variables()

    model = RandomActionModel(actions, available_variables, 'test')

    game.new_episode()
    while not game.is_episode_finished():
        variables = game.get_available_game_variables()
        action = model.action(variables)
        game.make_action(action)
        time.sleep(sleep_rate)

    game.close()
