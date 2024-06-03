from math import sqrt

from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Steps(SB3_Metric_Callback):
    """
    This custom metric measures the number of steps in each episode.
    """

    def __init__(self, verbose=0, model='ppo', instance=0):
        super(SB3_Episode_Steps, self).__init__(verbose, name="Episode Length in Steps", model=model, instance=instance)
        self.step_counter = 0
        self.episode_counter = 1

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            # print("Metryka episode_steps odnotowała koniec epizodu. Liczba kroków: ", self.step_counter)
            self._on_episode_end()

        self.step_counter += 1
        return True

    def _on_episode_end(self) -> bool:
        game_state = self._get_game_state()
        if game_state is None:
            return False

        # Uncomment to plot the number of steps per episode, with the episode number on the x-axis
        # self._logger.add_scalar(self.name, self.step_counter, self.episode_counter)

        # Plot the number of steps per episode, with the number of timesteps on the x-axis
        self._logger.add_scalar(self.name, self.step_counter, self.num_timesteps)
        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1

        return True
