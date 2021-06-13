import random


class Experience:
    def __init__(self, state, action, next_state, reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
    def is_terminal(self):
        return self.next_state.is_terminal()


class ExperienceMemory:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("Memory capacity cannot be less or equal to zero")
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def add(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.index] = experience
            self.index = (self.index + 1) % self.capacity

    def provide_batch(self, batch_size):
        if self.can_provide_batch(batch_size):
            return random.sample(self.memory, batch_size)
        raise ValueError(
            "Can't provide batch due to lack of experience/capacity")

    def can_provide_batch(self, batch_size):
        return len(self.memory) >= batch_size
