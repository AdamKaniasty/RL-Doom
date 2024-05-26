from math import sqrt

from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Distance(SB3_Metric_Callback):
    def __init__(self, verbose=0):
        super(SB3_Episode_Distance, self).__init__(verbose, name="Episode Distance")
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

        distance = self._compute_custom_metric(game_state)

        self._logger.add_scalar(self.name, distance, self.episode_counter)
        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1

        return True

    def _compute_custom_metric(self, game_state=None):
        game_state = game_state.game_variables
        return sqrt(game_state[3]**2 + game_state[4]**2 + game_state[5]**2)
