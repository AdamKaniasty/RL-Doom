from math import sqrt

from src.metrics.sb3_metric_abstract import SB3_Metric_Callback
import tensorflow as tf


class SB3_Episode_Steps(SB3_Metric_Callback):
    def __init__(self, verbose=0):
        super(SB3_Episode_Steps, self).__init__(verbose, name="Episode Steps")
        self.step_counter = 0
        self.episode_counter = 0

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            self._on_episode_end()

        self.step_counter += 1

        return True

    def _on_episode_end(self) -> bool:
        game_state = self._get_game_state()
        if game_state is None:
            return False

        with self._logger.as_default():
            tf.summary.scalar(self.name, self.step_counter, step=self.episode_counter)
            self._logger.flush()
        self.step_counter = 0
        self.episode_counter += 1

        return True
