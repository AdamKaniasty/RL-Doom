from stable_baselines3.common.callbacks import BaseCallback
import tensorflow as tf


class SB3_Metric_Callback(BaseCallback):
    def __init__(self, verbose=0, name="SB3_Metric_Callback"):
        super(SB3_Metric_Callback, self).__init__(verbose)
        self._logger = tf.summary.create_file_writer(logdir="./src/models/logs/custom_metrics")
        self.name = name

    def _on_step(self) -> bool:
        game_state = self._get_game_state()
        if game_state is None:
            return False
        print(game_state.game_variables)  # GLHF
        custom_metric = self._compute_custom_metric()

        with self._logger.as_default():
            tf.summary.scalar(self.name, custom_metric, step=self.num_timesteps)
            self._logger.flush()

        return True

    def _compute_custom_metric(self):
        return self.num_timesteps

    def _get_game_state(self):
        env = self.training_env.envs[0]
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'game'):
            return env.unwrapped.game.get_state()
        return None
