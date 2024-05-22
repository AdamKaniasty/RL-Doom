class AbstractReward:
    def __init__(self, health_reward=0.1, damage_penalty=-0.1, enemy_reward=1.0, goal_reward=10.0, ammo_penalty=-0.01,
                 death_penalty=-10.0):
        # Define reward values
        self.health_reward = health_reward
        self.damage_penalty = damage_penalty
        self.enemy_reward = enemy_reward
        self.goal_reward = goal_reward
        self.ammo_penalty = ammo_penalty
        self.death_penalty = death_penalty
        self.first_run = True
        self.prev_health = 0
        self.prev_ammo = 0
        self.prev_kills = 0
        self.cumulated_reward = 0.0

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
        health_change = game_state[1] - self.prev_health
        ammo_change = game_state[0] - self.prev_ammo
        kills_change = game_state[2] - self.prev_kills
        reward += health_change * self.health_reward
        reward += ammo_change * self.ammo_penalty
        reward += kills_change * self.enemy_reward
        if game_state[-1] != 0:
            # Player died
            reward += self.death_penalty

        # Update previous game state
        self.prev_health = game_state[1]
        self.prev_ammo = game_state[0]
        self.prev_kills = game_state[2]
        self.cumulated_reward += reward
        return reward

    def reset(self):
        self.first_run = True
        self.prev_health = 0
        self.prev_ammo = 0
        self.prev_kills = 0
        self.cumulated_reward = 0.0

    def get_cumulated_reward(self):
        return self.cumulated_reward
