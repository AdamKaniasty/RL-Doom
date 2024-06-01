from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_Episode_Return(SB3_Metric_Callback):
    """
    This custom metric measures the return received by the agent in each episode.
    The return is the discounted (by the factor gamma) sum of reward received by the agent.
    """

    def __init__(self):
        super(SB3_Episode_Return, self).__init__(name="Episode Return")
        # self.model.get_parameters().get('gamma')
        self.step_counter = 0
        self.episode_counter = 1
        self.episode_return = 0
        self.episode_return_no_discount = 0
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
        self.episode_return_no_discount += reward

        self.step_counter += 1
        return True

    def _on_episode_end(self) -> bool:
        self._logger.add_scalars(self.name, {'discounted': self.episode_return,
                                            'undiscounted': self.episode_return_no_discount}, self.num_timesteps)
        self._logger.flush()

        self.step_counter = 0
        self.episode_counter += 1
        self.episode_return = 0
        self.episode_return_no_discount = 0

        return True
