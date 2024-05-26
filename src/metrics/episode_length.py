from src.metrics.sb3_metric_abstract import SB3_Metric_Callback


class SB3_EpisodeLength_Callback(SB3_Metric_Callback):
    def __init__(self, verbose=0):
        super(SB3_EpisodeLength_Callback, self).__init__(verbose, name="Episode Length")
        self.step_counter = 0
        self.episode_counter = 0

    def _on_step(self) -> bool:
        self.step_counter += 1
        if 'done' in self.locals and self.locals['done']:
            self.on_episode_end()

        return True

    def on_episode_end(self):
        print("Episode Length: ", self.step_counter)


# def _on_rollout_end(self) -> bool:
#     with self._logger.as_default():
#         tf.summary.scalar(self.name, self.step_counter, step=self.episode_counter)
#         self._logger.flush()
#     self.step_counter = 0
#     self.episode_counter += 1
#
#     return True

def _compute_custom_metric(self):
    return self.num_timesteps
