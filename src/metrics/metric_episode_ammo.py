from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Ammo(SB3_Metric_Callback):
    """
    This custome metric measures the amount of ammo used by the agent in each episode.
    """

    def __init__(self, verbose=0, model='ppo', instance=0):
        super(SB3_Episode_Ammo, self).__init__(verbose, name="Ammo Usage in each Episode", model=model,
                                               instance=instance)
        self.step_counter = 0
        self.episode_counter = 1
        self.last_ammo = 52
        self.prev_ammo = 52
        self.start_ammo = 52  # Starting ammo value in the game

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            self._on_episode_end()

        # Update ammo usage:
        self.prev_ammo = self.last_ammo
        ammo = self._get_game_variable(0)
        self.last_ammo = ammo

        self.step_counter += 1

        return True

    def _on_episode_end(self) -> bool:
        self._logger.add_scalar(self.name, self.start_ammo - self.prev_ammo, self.episode_counter)
        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1
        self.last_ammo = self.start_ammo
        self.prev_ammo = self.start_ammo

        return True
