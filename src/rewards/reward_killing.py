from src.rewards.reward_abstract import AbstractReward


class KillingReward(AbstractReward):

    def __init__(self, health_reward=0.1, damage_penalty=-0.1, enemy_reward=3.0, goal_reward=10.0, ammo_penalty=-0.01,
                 death_penalty=0):
        super().__init__(health_reward, damage_penalty, enemy_reward, goal_reward, ammo_penalty, death_penalty)