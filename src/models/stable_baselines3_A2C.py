import gymnasium as gym
from stable_baselines3 import A2C

from src.metrics.metric_episode_distance import SB3_Episode_Distance


class A2C_Model:
    def __init__(self, config_file, mode='train', pretrained=None):
        self.env = gym.make('Vizdoom-v0', level=config_file, mode=mode)
        if pretrained:
            print("Loading pretrained model")
            self.model = A2C.load(pretrained, self.env, tensorboard_log="./src/models/logs/a2c")
        else:
            print("Creating new model")
            self.model = A2C("MultiInputPolicy", self.env, verbose=1, tensorboard_log="./src/models/logs/a2c")

    def train(self, steps=1000):
        callback = SB3_Episode_Distance()
        self.model.learn(total_timesteps=steps, progress_bar=True, callback=callback)

    def save(self, path='a2c_vizdoom'):
        self.model.save("./src/models/weights/" + path)

    def test(self):
        stable_env = self.model.get_env()
        state = stable_env.reset()
        terminated = False
        while not terminated:
            action, _ = self.model.predict(state)
            state, _, terminated, _ = stable_env.step(action)
