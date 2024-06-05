import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from src.game.gymnasium_wrapper import VizDOOM
import matplotlib.pyplot as plt


class PreprocessFrameAndGameVariables(gym.ObservationWrapper):
    def __init__(self, env, screen_shape=(256, 100)):
        super(PreprocessFrameAndGameVariables, self).__init__(env)
        self.screen_shape = screen_shape
        self.observation_space = Box(low=0, high=255, shape=(self.screen_shape[1], self.screen_shape[0], 2),
                                     dtype=np.uint8)

    def observation(self, obs):
        screen = obs['screen']
        processed_screen = self._process_screen(screen)
        edges = self._detect_edges(screen)
        combined_screen = np.concatenate((processed_screen, edges), axis=-1)
        return combined_screen

    def _process_screen(self, screen):
        if len(screen.shape) == 4:
            screen = screen.squeeze()
        screen = np.transpose(screen, (1, 2, 0))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = self._crop(screen, 60, 70)
        screen = cv2.equalizeHist(screen)
        screen = cv2.resize(screen, self.screen_shape, interpolation=cv2.INTER_AREA)
        # screen = self._add_crosshair(screen, 6)
        screen = np.expand_dims(screen, axis=-1)
        return screen

    def _detect_edges(self, screen):
        if len(screen.shape) == 4:
            screen = screen.squeeze()
        screen = np.transpose(screen, (1, 2, 0))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(screen, 80, 180)
        edges = self._crop(edges, 60, 70)
        edges = cv2.resize(edges, self.screen_shape, interpolation=cv2.INTER_AREA)
        edges = np.expand_dims(edges, axis=-1)
        return edges

    def _add_crosshair(self, image, size=10):
        center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        cv2.line(image, (center_x - size, center_y), (center_x + size, center_y), (255, 255, 255), 1)
        cv2.line(image, (center_x, center_y - size), (center_x, center_y + size), (255, 255, 255), 1)
        return image

    def _crop(self, image, top, down):
        return image[top:-down, :]


if __name__ == '__main__':
    env = gym.make('Vizdoom-v0', level='deadly_corridor')
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
        plt.subplot(2, len(frames), i + 1)
        plt.imshow(frame[:, :, 0], cmap='gray')
        plt.title('Processed Screen')
        plt.axis('off')

        plt.subplot(2, len(frames), i + len(frames) + 1)
        plt.imshow(frame[:, :, 1], cmap='gray')
        plt.title('Edges')
        plt.axis('off')
    plt.show()
