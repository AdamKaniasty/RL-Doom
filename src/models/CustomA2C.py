import os

import gymnasium as gym
from stable_baselines3 import A2C

from src.metrics.metric_episode_ammo import SB3_Episode_Ammo
from src.metrics.metric_episode_distance import SB3_Episode_Distance
from src.metrics.metric_episode_health import SB3_Episode_Health
from src.metrics.metric_episode_killcount import SB3_Episode_Killcount
from src.metrics.metric_episode_return import SB3_Episode_Return
from src.metrics.metric_episode_steps import SB3_Episode_Steps
from src.metrics.metric_timestep_reward import SB3_Timestep_Reward
from src.models.nn_module import CustomPolicy
from src.utils.screen_preprocess import PreprocessFrameAndGameVariables


class CustomA2C_Model:
    def __init__(self, config_file, mode='train', pretrained=None):
        self.env = gym.make('Vizdoom-v0', level=config_file, mode=mode)
        self.env = PreprocessFrameAndGameVariables(self.env)
        if pretrained:
            print("Loading pretrained model")
            self.model = A2C.load(pretrained, self.env, tensorboard_log="../models/logs/a2c")
        else:
            print("Creating new model")
            self.model = A2C(CustomPolicy, self.env, n_steps=2048, verbose=1, tensorboard_log="../models/logs/a2c")

    def train(self, steps=1000):
        instance = len(os.listdir(f"../models/logs/a2c/custom_metrics")) + 1
        callbacks = [SB3_Episode_Distance(model='a2c', instance=instance),
                     SB3_Episode_Steps(model='a2c', instance=instance),
                     SB3_Episode_Killcount(model='a2c', instance=instance),
                     SB3_Episode_Ammo(model='a2c', instance=instance),
                     SB3_Episode_Health(model='a2c', instance=instance),
                     SB3_Episode_Return(model='a2c', instance=instance),
                     SB3_Timestep_Reward(model='a2c', instance=instance)]
        self.model.learn(total_timesteps=steps, progress_bar=True, callback=callbacks)

    def save(self, path):
        self.model.save("../models/weights/" + path)

    def test(self):
        stable_env = self.model.get_env()
        # Now instead of only one episode, we can test multiple episodes
        for _ in range(5):
            state = stable_env.reset()
            terminated = False
            while not terminated:
                action, _ = self.model.predict(state)
                state, _, terminated, _ = stable_env.step(action)
