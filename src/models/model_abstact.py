class DoomModel:
    def __init__(self, actions, variables, mode):
        self.mode = mode
        self.actions = actions
        self.actions_encoded = [[1 if i == j else 0 for i in range(len(actions))] for j in range(len(actions))]
        self.available_variables = variables

    def action(self, state):
        pass
