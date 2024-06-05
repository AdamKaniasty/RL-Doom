from src.rewards.reward_main import Reward


class ExtendedReward(Reward):
    """
    Extended reward class created as main reward class used in the project. It inherits from the Reward class.
    It is used to calculate reward based on the change in game state. It has the following attributes:
    - health_reward: reward for change in health
    - damage_penalty: penalty for change in damage
    - enemy_reward: reward for killing an enemy
    - ammo_penalty: penalty for change in ammo
    - x_penalty: penalty for change in x position, it was created to balance default reward function which is added in
    gymnasium_wrapper.py to this reward function
    """

    def __init__(self, health_reward=1, damage_penalty=1, enemy_reward=150.0, ammo_penalty=1, x_penalty=0.3):
        super().__init__(health_reward, damage_penalty, enemy_reward, ammo_penalty)
        self.x_penalty = x_penalty

    def evaluate(self, game_state):
        if game_state is None:
            return 0.0
        game_state = game_state.game_variables
        if self.first_run:
            self.prev_health = game_state[1]
            self.prev_ammo = game_state[0]
            self.prev_kills = game_state[2]
            self.prev_x = game_state[3]
            self.first_run = False
            return 0.0
        reward = 0.0
        # Calculate reward based on the change in game state
        health_change = game_state[1] - self.prev_health
        ammo_change = game_state[0] - self.prev_ammo
        kills_change = game_state[2] - self.prev_kills
        x_change = game_state[3] - self.prev_x
        reward += health_change * self.health_reward
        reward += ammo_change * self.ammo_penalty
        reward += kills_change * self.enemy_reward
        reward += x_change * self.x_penalty * (-1)

        # Update previous game state
        self.prev_health = game_state[1]
        self.prev_ammo = game_state[0]
        self.prev_kills = game_state[2]
        self.prev_x = game_state[3]
        return reward
