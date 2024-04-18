import numpy as np
from random import choice

from src.models.model_abstact import DoomModel


class RandomActionModel(DoomModel):

    def __init__(self, actions, variables, mode):
        super().__init__(actions, variables, mode)

    def action(self, state, encoded=False):
        if encoded:
            return choice(self.actions_encoded)
        return choice(self.actions)
