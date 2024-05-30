from src.rewards.reward_abstract import AbstractReward

class Reward(AbstractReward):
    """
    Reward class created as main reward class used in the project. It inherits from the AbstractReward class.
    It is used to calculate reward based on the change in game state. It has the following attributes:
    """
    def __init__(self, health_reward=1, damage_penalty=1, enemy_reward=150.0, ammo_penalty=1):
        """
        Penalties and rewards are defined as absolute values. Their sign will be changed in the evaluation
        process.
        """
        self.health_reward = health_reward
        self.damage_penalty = damage_penalty
        self.enemy_reward = enemy_reward
        self.ammo_penalty = ammo_penalty
        self.first_run = True
        self.prev_health = 0
        self.prev_ammo = 0
        self.prev_kills = 0

    def evaluate(self, game_state):
        if game_state is None:
            return 0.0
        game_state = game_state.game_variables
        if self.first_run:
            self.prev_health = game_state[1]
            self.prev_ammo = game_state[0]
            self.prev_kills = game_state[2]
            self.first_run = False
            return 0.0

        reward = 0.0
        # Calculate reward based on the change in game state
        # I am pretty sure that this 6 lines of code here works good
        health_change = game_state[1] - self.prev_health
        ammo_change = game_state[0] - self.prev_ammo
        kills_change = game_state[2] - self.prev_kills
        reward += health_change * self.health_reward
        reward += ammo_change * self.ammo_penalty
        reward += kills_change * self.enemy_reward

        # Update previous game state
        self.prev_health = game_state[1]
        self.prev_ammo = game_state[0]
        self.prev_kills = game_state[2]
        return reward

    def reset(self):
        self.first_run = True
        self.prev_health = 0
        self.prev_ammo = 0
        self.prev_kills = 0

