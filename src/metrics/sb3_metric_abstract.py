import os

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class SB3_Metric_Callback(BaseCallback):
    def __init__(self, verbose=0, name="SB3_Metric_Callback", model='ppo', instance=0):
        super(SB3_Metric_Callback, self).__init__(verbose)
        # self._logger = tf.summary.create_file_writer(logdir="./src/models/logs/a2c")
        self.name = name
        self._logger = SummaryWriter(log_dir=f"./src/models/logs/{model}/custom_metrics/{instance}")

    def _on_step(self) -> bool:
        game_state = self._get_game_state()
        if game_state is None:
            return False
        custom_metric = self._compute_custom_metric()

        self._logger.add_scalar(self.name, custom_metric, self.num_timesteps)
        self._logger.flush()

        return True

    def _compute_custom_metric(self):
        return self.num_timesteps

    def _get_game_state(self):
        env = self.training_env.envs[0]
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'game'):
            return env.unwrapped.game.get_state()
        return None

    def _get_doom_game(self):
        env = self.training_env.envs[0]
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'game'):
            return env.unwrapped.game

        return None

    def _get_game_variable(self, index):
        game_state = self._get_game_state()
        if game_state is None:
            return None
        return game_state.game_variables[index]
