from src.game.env_init import game_init
from random import choice
import time

from src.models.random_action import RandomActionModel


def train(config_file):
    game = game_init(config_file, 'train')

    actions = game.get_available_buttons()
    available_variables = game.get_available_game_variables()

    model = RandomActionModel(actions, available_variables, 'train')

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            # Screenshot
            screen = game.get_state().screen_buffer

            variables = game.get_available_game_variables()
            action = model.action(variables, True)
            game.make_action(action)

    game.close()
