import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import torch as th

from src.metrics.metric_episode_distance import SB3_Episode_Distance
from src.metrics.metric_episode_steps import SB3_Episode_Steps
from src.utils.screen_preprocess import PreprocessFrameAndGameVariables


class CNN_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN_Block, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, 32, 8, 4, 0)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv_2 = nn.Conv2d(32, 32, 4, 2, 0)
        self.conv_3 = nn.Conv2d(32, out_channels, 4, 2, 0)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv_1(x))
        x = self.pool_1(x)
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        return self.flatten(x)


class Linear_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear_Block, self).__init__()
        self.linear_1 = nn.Linear(in_features, 64)
        self.linear_2 = nn.Linear(64, out_features)
        self.relu = nn.ReLU()
        self.linear_1.register_forward_hook(self.forward_hook)
        self.iterations = 0

    def forward(self, x):
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

    def forward_hook(self, module, inp, out):
        # print(self.iterations)
        self.iterations += 1


class CustomNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(CustomNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space['screen'].shape[0]

        self.cnn = CNN_Block(n_input_channels, 5)

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space['screen'].sample()[None]).float()).shape[1]

        self.linear = Linear_Block(n_flatten + observation_space['gamevariables'].shape[0], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        image_obs = observations['screen']
        scalar_obs = observations['gamevariables']
        cnn_output = self.cnn(image_obs)
        concatenated = th.cat((cnn_output, scalar_obs), dim=1)
        return self.linear(concatenated)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           features_extractor_class=CustomNN,
                                           features_extractor_kwargs=dict(features_dim=256))


class CustomPPO_Model:
    def __init__(self, config_file, mode='train', pretrained=None):
        self.env = gym.make('Vizdoom-v0', level=config_file, mode=mode)
        self.env = PreprocessFrameAndGameVariables(self.env)
        if pretrained:
            print("Loading pretrained model")
            self.model = PPO.load(pretrained, self.env, tensorboard_log="./src/models/logs/ppo")
        else:
            print("Creating new model")
            self.model = PPO(CustomPolicy, self.env, verbose=1, tensorboard_log="./src/models/logs/ppo")

    def train(self, steps=1000):
        callbacks = [SB3_Episode_Distance(), SB3_Episode_Steps()]
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
                action, _ = self.model.predict(state)
                state, _, terminated, _ = stable_env.step(action)
