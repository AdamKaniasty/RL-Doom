from src.rewards.reward_main_extended_plus import ExtendedPlusReward


class FinalReward(ExtendedPlusReward):
    def __init__(self, health_reward=1, damage_penalty=1, enemy_reward=200.0, ammo_penalty=5, x_penalty=0):
        super().__init__(health_reward, damage_penalty, enemy_reward, ammo_penalty, x_penalty)
