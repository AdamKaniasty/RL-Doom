import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from src.game.gymnasium_wrapper import VizDOOM
import matplotlib.pyplot as plt


class PreprocessFrameAndGameVariables(gym.ObservationWrapper):
    def __init__(self, env, screen_shape=(160, 90)):
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
        if len(screen.shape) == 4:
            screen = screen.squeeze()
        screen = np.transpose(screen, (1, 2, 0))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = self._crop(screen, 30, 30)
        screen = cv2.resize(screen, self.screen_shape, interpolation=cv2.INTER_AREA)
        screen = cv2.GaussianBlur(screen, (5, 5), 0)
        screen = cv2.equalizeHist(screen)
        screen = cv2.resize(screen, self.screen_shape, interpolation=cv2.INTER_AREA)
        screen = self._add_crosshair(screen, 8)
        screen = np.expand_dims(screen, axis=-1)

        return screen

    def _add_crosshair(self, image, size=10):
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        cv2.line(image, (center_x - size, center_y), (center_x + size, center_y), (255, 255, 255), 1)
        cv2.line(image, (center_x, center_y - size), (center_x, center_y + size), (255, 255, 255), 1)

        return image

    def _crop(self, image, top, down):
        return image[top:-down, :]


if __name__ == '__main__':
    env = VizDOOM(level='deadly_corridor')
    env = PreprocessFrameAndGameVariables(env)
    env.reset()

    frames = []
    for _ in range(3):
        state, reward, done, info, _ = env.step(env.action_space.sample())
        print(f'Screen shape: {state["screen"].shape}')
        print(f'Game variables: {state["gamevariables"]}')
        print(f'Reward: {reward}')
        print(f'Done: {done}')
        print(f'Info: {info}')
        print()
        frames.append(state['screen'])

    plt.figure(figsize=(15, 10))
    for i, frame in enumerate(frames):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame, cmap='gray')
        plt.axis('off')
    plt.show()
