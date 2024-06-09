from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Health(SB3_Metric_Callback):
    """
    This custom metric measures the amount of health that agent has at the end of each episode.
    Note that information about the agent's death is not available (bug in vizdoom?).
    Instead, health is approximated in the following manner:
     - If the agent was close to the end of corridor at the end of the episode, it is assumed that the agent is NOT dead.
     - If the time-limit of the episode is reached, it is assumed that the agent is NOT dead.
     - Otherwise, when episode ends, the agent is assumed to be DEAD. Its health is set to 0.
    """

    def __init__(self, verbose=0, model='ppo', instance=0):
        super(SB3_Episode_Health, self).__init__(verbose, name="Health at Episode End", model=model, instance=instance)
        self.step_counter = 0
        self.episode_counter = 1

        self.last_health = 100
        self.prev_health = 100

        self.last_position = 0
        self.prev_position = 0

        self.number_of_deaths = 0
        self.number_of_wins = 0

        self.dead = False
        self.timeout_steps = 1000

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            self._on_episode_end()

        # Update health:
        self.prev_health = self.last_health
        health = self._get_game_variable(1)
        self.last_health = health

        # Update position:
        self.prev_position = self.last_position
        position = self._get_game_variable(3)
        self.last_position = position

        self.step_counter += 1

        return True

    def _on_episode_end(self) -> bool:
        # Check if the agent died or not:
        if self.prev_position >= 1260:
            print("Agent is not dead. Position: ", self.prev_position)
            self.dead = False
            self.number_of_wins += 1
        elif self.step_counter >= self.timeout_steps - 1:
            print("Agent is not dead. Timeout reached. Steps: ", self.step_counter)
            self.dead = False
        else:
            self.prev_health = 0
            self.dead = True
            self.number_of_deaths += 1

        # Add the health to the tensorboard
        self._logger.add_scalar(self.name, self.prev_health, self.episode_counter)
        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1
        self.last_health = 100
        self.prev_health = 100

        self.last_position = 0
        self.prev_position = 0

        self.dead = False

        return True

    def _on_training_end(self) -> None:
        # Add the number of deaths to the tensorboard as text
        self._logger.add_text("Number of episodes in training", str(self.episode_counter - 1), self.num_timesteps)
        self._logger.add_text("Number of Deaths during training", str(self.number_of_deaths), self.num_timesteps)
        self._logger.add_text("Number of Wins during training", str(self.number_of_wins), self.num_timesteps)

        self._logger.flush()

        return None
