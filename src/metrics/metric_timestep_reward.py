from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Timestep_Reward(SB3_Metric_Callback):
    """
    This custom metric measures the reward in each timestep.
    The rewards are added to total return and discounted by the factor gamma.
    Total return for the current timestep is logged in each timestep.
    The total return (episode_return) is reset to 0 at the beginning of each episode.
    """

    def __init__(self, model='ppo', instance=0):
        super(SB3_Timestep_Reward, self).__init__(name="Timestep Reward from the beggining of Episode", model=model,
                                                  instance=instance)
        # self.model.get_parameters().get('gamma')
        self.step_counter = 0
        self.episode_counter = 1
        self.episode_return = 0
        self.gamma = 0.99
        self.first_run = True
        self.env = None

    def _on_step(self) -> bool:
        if self.first_run:
            self.first_run = False
            # print("Episode_return callback checks gamma factor directly: gamma = ", self.model.gamma)
            self.gamma = self.model.gamma
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped'):
                self.env = env.unwrapped

        if 'done' in self.locals and self.locals['done']:
            # print("Episode_return callback detected end of episode. Episode return: ", self.episode_return)
            self._on_episode_end()

        # Update episode_return by adding the reward received in the current step
        reward = self.env.current_reward
        # print("Reward after step: ", self.step_counter, "current_reward = ", reward)
        self.episode_return = self.episode_return * self.gamma + reward

        # Log the total return for the current timestep
        self._logger.add_scalar(self.name, self.episode_return, self.num_timesteps)
        self._logger.flush()

        self.step_counter += 1
        return True

    def _on_episode_end(self) -> bool:

        self.step_counter = 0
        self.episode_counter += 1
        self.episode_return = 0

        return True
