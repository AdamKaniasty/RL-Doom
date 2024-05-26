
class Reward_corridor:
    def __init__(self):

        # self.goal_reward = ???
        # self.dead_penalty = -100.0
        # self.closer_reward = 1
        # self.further_penalty = -1

        # TODO: change this values relative to the changing value of death_penalty and reaching_the_goal reward
        self.ammo_penalty = 5  # -5 for each ammo use
        self.hitcount_reward = 200  # +200 for each hit
        self.damage_taken_penalty = -10  # -10 for each damage taken

        self.first_run = True
        self.health = 100
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52
        # self.x = 0

    def evaluate(self, game_state):
        if game_state is None:
            return 0.0

        variables = game_state.game_variables

        if self.first_run:
            self.health = variables[1]
            self.ammo = variables[0]
            self.hitcount = variables[12]
            self.damage_taken = variables[13]
            # self.x = variables[3]

            self.first_run = False
            return 0.0

        reward = 0.0
        # Calculate reward based on the change in game state
        damage_taken_change = variables[13] - self.damage_taken
        ammo_change = variables[0] - self.ammo
        hitcount_change = variables[12] - self.hitcount
        # x_change = variables[3] - self.x

        # damage_taken: 10 -> 20 nagroda 10 * (-10)
        # ammo: 50 -> 40 nagroda (-10)*5
        # hitcount: 0 -> 1 nagroda 1 * 200
        reward = damage_taken_change * self.damage_taken_penalty + ammo_change * self.ammo_penalty + hitcount_change * self.hitcount_reward


        # Update previous game state
        self.health = variables[1]
        self.ammo = variables[0]
        self.hitcount = variables[12]
        self.damage_taken = variables[13]
        # self.x = variables[3]

        # Return calculated reward
        return reward