import os
import time

import gymnasium as gym
from stable_baselines3 import PPO

from src.metrics.metric_episode_ammo import SB3_Episode_Ammo
from src.metrics.metric_episode_distance import SB3_Episode_Distance
from src.metrics.metric_episode_health import SB3_Episode_Health
from src.metrics.metric_episode_killcount import SB3_Episode_Killcount
from src.metrics.metric_episode_return import SB3_Episode_Return
from src.metrics.metric_episode_steps import SB3_Episode_Steps
from src.metrics.metric_timestep_reward import SB3_Timestep_Reward
from src.models.nn_module import CustomPolicy
from src.utils.screen_preprocess import PreprocessFrameAndGameVariables


class CustomPPO_Model:
    def __init__(self, config_file, mode='train', pretrained=None):
        self.env = gym.make('Vizdoom-v0', level=config_file, mode=mode)
        self.env = PreprocessFrameAndGameVariables(self.env)
        if pretrained:
            print("Loading pretrained model")
            self.model = PPO.load(pretrained, self.env, tensorboard_log="./src/models/logs/ppo", learning_rate=0.00001,
                                  n_steps=8192, clip_range=0.1, gamma=0.95, gae_lambda=0.9, ent_coef=0.05)
        else:
            print("Creating new model")
            self.model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log="./src/models/logs/ppo",
                             learning_rate=0.00001, n_steps=8192, clip_range=0.1, gamma=0.95, gae_lambda=0.9,
                             ent_coef=0.05)
            # self.model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log="./src/models/logs/ppo",
            #                  learning_rate=0.00001, n_steps=8192, clip_range=0.1, gamma=0.95, gae_lambda=0.9)

    def train(self, steps=1000):
        instance = len(os.listdir(f"./src/models/logs/ppo/custom_metrics")) + 1
        callbacks = [SB3_Episode_Distance(model='ppo', instance=instance),
                     SB3_Episode_Steps(model='ppo', instance=instance),
                     SB3_Episode_Killcount(model='ppo', instance=instance),
                     SB3_Episode_Ammo(model='ppo', instance=instance),
                     SB3_Episode_Health(model='ppo', instance=instance),
                     SB3_Episode_Return(model='ppo', instance=instance),
                     SB3_Timestep_Reward(model='ppo', instance=instance)]
        self.model.learn(total_timesteps=steps, progress_bar=True, callback=callbacks)

    def save(self, path):
        self.model.save("./src/models/weights/" + path)

    def test(self):
        stable_env = self.model.get_env()
        # Now instead of only one episode, we can test multiple episodes
        for _ in range(5):
            state = stable_env.reset()
            terminated = False
            while not terminated:
                action, _ = self.model.predict(state, deterministic=True)
                state, _, terminated, _ = stable_env.step(action)
                time.sleep(0.05)
