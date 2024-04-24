from src.game.env_init import game_init
from random import choice
import time

from src.models.random_action import RandomActionModel
import os
import vizdoom as vzd


def train(config_file):
    game = game_init(config_file, 'train')

    actions = game.get_available_buttons()
    available_variables = game.get_available_game_variables()

    model = RandomActionModel(actions, available_variables, 'train')

    episodes = 1
    for i in range(episodes):
        game.new_episode()
        iter = 0
        while not game.is_episode_finished():
            # Screenshot
            screen = game.get_state().screen_buffer

            variables = game.get_available_game_variables()
            action = model.action(variables, True)
            game.make_action(action)
            iter += 1
        print("Episode finished. Iterations: ", iter)
    game.close()


if __name__ == '__main__':
    CONFIG_PATH = os.path.join(vzd.scenarios_path, "deadly_corridor.cfg")
    train(CONFIG_PATH)
