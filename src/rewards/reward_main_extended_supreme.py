from src.rewards.reward_main_extended import ExtendedReward


class ExtendedSupremeReward(ExtendedReward):
    def __init__(self, health_reward=2, damage_penalty=1, enemy_reward=150.0, ammo_penalty=0.8, x_penalty=0.5):
        super().__init__(health_reward, damage_penalty, enemy_reward, ammo_penalty, x_penalty)
