import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict


class PreprocessFrameAndGameVariables(gym.ObservationWrapper):
    def __init__(self, env, screen_shape=(150, 300)):
        super(PreprocessFrameAndGameVariables, self).__init__(env)
        self.screen_shape = screen_shape
        self.observation_space = Dict({
            'screen': Box(low=0, high=255, shape=(self.screen_shape[1], self.screen_shape[0], 1), dtype=np.uint8),
            'gamevariables': env.observation_space['gamevariables']
        })

    def observation(self, obs):
        return {
            'screen': self._process_screen(obs['screen']),
            'gamevariables': obs['gamevariables']
        }

    def _process_screen(self, screen):
        if len(screen.shape) == 3 and screen.shape[2] == 3:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, self.screen_shape, interpolation=cv2.INTER_AREA)

        screen = np.expand_dims(screen, axis=-1)
        return screen
