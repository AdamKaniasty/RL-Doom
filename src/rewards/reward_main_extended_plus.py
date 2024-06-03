from src.rewards.reward_main_extended import ExtendedReward


class ExtendedPlusReward(ExtendedReward):
    def __init__(self, health_reward=2.5, damage_penalty=1, enemy_reward=180.0, ammo_penalty=0.9, x_penalty=0.6):
        super().__init__(health_reward, damage_penalty, enemy_reward, ammo_penalty, x_penalty)
