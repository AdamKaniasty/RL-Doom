from src.game.env_init import game_init
from random import choice
import time
import gymnasium as gym

from src.game.gymnasium_wrapper import VizDOOM
from src.models.random_action import RandomActionModel


def play(config_file, tickrate=35):
    sleep_rate = 1 / tickrate
    episodes = 1

    env = gym.make('Vizdoom-v0', level=config_file, mode='test')

    actions = env.unwrapped.get_available_buttons()
    available_variables = env.unwrapped.get_available_game_variables()

    model = RandomActionModel(actions, available_variables, 'test')

    state = env.reset()
    terminated = False
    while not terminated:
        action = model.action(state)
        state = env.step(action)
        terminated = state[2]
        variables = state[0]['gamevariables']
        time.sleep(sleep_rate)
