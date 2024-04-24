from math import sqrt

from src.metrics.sb3_metric_abstract import SB3_Metric_Callback
import tensorflow as tf


class SB3_Episode_Distance(SB3_Metric_Callback):
    def __init__(self, verbose=0):
        super(SB3_Episode_Distance, self).__init__(verbose, name="Episode Length")
        self.step_counter = 0
        self.episode_counter = 0

    def _on_step(self) -> bool:
        env = self.training_env.envs[0]

        # print("Step: ", self.step_counter)
        #
        # print(env.unwrapped.game.is_episode_finished())
        # game_state = self._get_game_state()
        # distance = self._compute_custom_metric(game_state)
        # print(f"Distance: {distance}")

        if 'done' in self.locals and self.locals['done']:
            self._on_episode_end()

        self.step_counter += 1

        return True

    def _on_episode_end(self) -> bool:
        print(f"Episode {self.episode_counter} ended")
        game_state = self._get_game_state()

        print("Step counter: ", self.step_counter)

        if game_state is None:
            return False

        distance = self._compute_custom_metric(game_state)
        print(f"Distance: {distance}")

        with self._logger.as_default():
            tf.summary.scalar(self.name, distance, step=self.episode_counter)
            self._logger.flush()
        self.step_counter = 0
        self.episode_counter += 1

        return True

    def _compute_custom_metric(self, game_state=None):
        game_state = game_state.game_variables
        return sqrt(game_state[3]**2 + game_state[4]**2 + game_state[5]**2)
