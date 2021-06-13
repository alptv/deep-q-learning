import random
import numpy as np
import torch

from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, Q_function, action_count):
        self.Q_function = Q_function
        self.action_count = action_count

    @abstractmethod
    def select_action(self, state):
        pass


class EpsGreedyAgent(Agent, ABC):
    def __init__(self, Q_function, action_count, eps_max, eps_min, eps_decay):
        super(EpsGreedyAgent, self).__init__(Q_function, action_count)
        self.step_number = 0
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def select_action(self, state):
        self.step_number += 1
        if random.random() < self._compute_epsilon():
            return random.randrange(self.action_count)
        else:
            return self.Q_function(state.data).argmax().item()

    def _compute_epsilon(self):
        return max(self.eps_max - self.eps_decay * (self.step_number - 1),
                   self.eps_min)
