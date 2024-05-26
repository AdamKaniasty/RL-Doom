import gymnasium as gym
from stable_baselines3 import A2C

from src.metrics.metric_episode_distance import SB3_Episode_Distance
from src.metrics.metric_episode_steps import SB3_Episode_Steps


class A2C_Model:
    """
    Our own class to handle the A2C model from Stable Baselines 3. It allows to train, save and test the model.
    It uses the environment registered as 'Vizdoom-v0' (registered in src/game/__init__.py) with the supplied config_path.
    Most important parameters when initializing the class:
    - config_file: path to the configuration file of the Vizdoom environment
    - mode: 'train' or 'test'
    - pretrained: path to the pretrained model if we want to test it
    """

    def __init__(self, config_file, mode='train', pretrained=None):
        self.env = gym.make('Vizdoom-v0', level=config_file, mode=mode)
        if pretrained:
            print("Loading pretrained model")
            self.model = A2C.load(pretrained, self.env, tensorboard_log="./src/models/logs/a2c")
        else:
            print("Creating new model")
            self.model = A2C("MultiInputPolicy", self.env, verbose=1, tensorboard_log="./src/models/logs/a2c")

    def train(self, steps=1000):
        callbacks = [SB3_Episode_Distance(), SB3_Episode_Steps()]
        self.model.learn(total_timesteps=steps, progress_bar=True, callback=callbacks)

    def save(self, path='a2c_vizdoom'):
        self.model.save("./src/models/weights/" + path)

    def test(self):
        stable_env = self.model.get_env()
        # state = stable_env.reset()
        # terminated = False
        # while not terminated:
        #     action, _ = self.model.predict(state)
        #     state, _, terminated, _ = stable_env.step(action)
        #
        # Now instead of only one episode, we can test multiple episodes
        for _ in range(5):
            state = stable_env.reset()
            terminated = False
            while not terminated:
                action, _ = self.model.predict(state)
                state, _, terminated, _ = stable_env.step(action)
