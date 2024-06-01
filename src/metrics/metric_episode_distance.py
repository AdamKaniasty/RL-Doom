from math import sqrt

from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Distance(SB3_Metric_Callback):
    """
    This custom metric measures the distance from the starting point (0,0,0) in the 3D space.
    It measures it for each episode.
    """

    def __init__(self, verbose=0):
        # Agent starts at (0,0,0) in the 3D space
        super(SB3_Episode_Distance, self).__init__(verbose, name="Distance from (0,0,0) in each episode")
        self.step_counter = 0
        self.episode_counter = 1
        self.last_distance = 0

        # Prev_distance is necessary becacuse when ('done' in self.locals and self.locals['done']) is True,
        # the distance is already reset to 0. So we need to remember the distance from the previous step.
        self.prev_distance = 0

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            # print("Metryka episode_distance odnotowała koniec epizodu. prev_distance: ", self.prev_distance)
            self._on_episode_end()

        # Update distance travelled by the agent:
        self.prev_distance = self.last_distance
        distance = self._get_game_variable(3)
        self.last_distance = distance

        self.step_counter += 1

        return True

    def _on_episode_end(self) -> bool:
        # Uncomment to plot the distance per episode, with the episode number on the x-axis.
        # self._logger.add_scalar(self.name, self.last_distance, self.episode_counter)

        self._logger.add_scalar(self.name, self.prev_distance, self.num_timesteps)
        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1
        self.last_distance = 0  # Reset the distance for the next episode
        self.prev_distance = 0

        return True

    # def _compute_custom_metric(self, game_state=None):
    #     game_state = game_state.game_variables
    #
    #     # Euclidean distance
    #     return sqrt(game_state[3]**2 + game_state[4]**2 + game_state[5]**2)
