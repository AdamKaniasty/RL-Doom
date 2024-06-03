from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Killcount(SB3_Metric_Callback):
    """
    This custom metric measures the number of opponents killed in each episode.
    At the beggining of each episode the killcount is reset to 0.
    At first stages of training, the killcount may be 0 for many episodes.
    The goal is to reach 6 kills per episode - the number of opponents in the corridor.
    """

    def __init__(self, verbose=0, model='ppo', instance=0):
        # Agent starts with 0 kills
        super(SB3_Episode_Killcount, self).__init__(verbose, name="Episode Killcount", model=model, instance=instance)
        self.step_counter = 0
        self.episode_counter = 1
        self.last_killcount = 0
        self.prev_killcount = 0
        self.total_kills = 0

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            # print("Metryka episode_killcount odnotowała koniec epizodu. Killcount: ", self.prev_killcount)
            self._on_episode_end()

        # Update killcount:
        self.prev_killcount = self.last_killcount
        killcount = self._get_game_variable(2)
        self.last_killcount = killcount

        self.step_counter += 1

        return True

    def _on_episode_end(self) -> bool:
        # Uncomment to plot the killcount per episode, with the episode number on the x-axis.
        # self._logger.add_scalar(self.name, self.last_killcount, self.episode_counter)
        self.total_kills += self.prev_killcount

        self._logger.add_scalar(self.name, self.prev_killcount, self.num_timesteps)
        # Add the total number of kills to the tensorboard as text - this can also be done as a separate metric
        self._logger.add_text("Total kills", str(self.total_kills), self.num_timesteps)

        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1
        self.last_killcount = 0  # Reset the killcount for the next episode
        self.prev_killcount = 0

        return True
