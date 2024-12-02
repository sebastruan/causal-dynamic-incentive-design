import random

from strategies.istrategy import IStrategy

class RandomStrategy(IStrategy):
    def __init__(self):
        self.accumulated_utility = 0

    def choose_action(self, **kwargs):
        return random.randint(0, 1)
