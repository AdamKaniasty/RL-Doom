from src.rewards.reward_abstract import AbstractReward


class KillingReward(AbstractReward):

    def __init__(self, health_reward=0.2, damage_penalty=-0.2, enemy_reward=2.0, goal_reward=10.0, ammo_penalty=-0.01,
                 death_penalty=0, moving_x_reward=0.001, moving_y_reward=0.001, moving_z_reward=0.001):
        super().__init__(health_reward, damage_penalty, enemy_reward, goal_reward, ammo_penalty, death_penalty,
                         moving_x_reward, moving_y_reward, moving_z_reward)