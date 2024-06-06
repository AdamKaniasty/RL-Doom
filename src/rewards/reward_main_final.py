from src.rewards.reward_main_extended_plus import ExtendedPlusReward


class FinalReward(ExtendedPlusReward):
    def __init__(self, health_reward=0.1, damage_penalty=1, enemy_reward=4, ammo_penalty=0.5, x_penalty=0):
        super().__init__(health_reward, damage_penalty, enemy_reward, ammo_penalty, x_penalty)
