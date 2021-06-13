import gym
import torch
import torchvision.transforms as T
from gym.wrappers import Monitor

from abc import ABC, abstractmethod


class State:
    def __init__(self, data, terminal):
        self.data = data
        self.terminal = terminal

    def is_terminal(self):
        return self.terminal

    def get_data(self):
        return self.data


class EnvManager(ABC):
    def __init__(self):
        self.terminal = False

    def is_done(self) -> bool:
        return self.terminal

    @abstractmethod
    def get_state(self) -> State:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def action_count(self):
        pass

    @abstractmethod
    def take_action(self, action):
        pass


class GymEnvManager(EnvManager, ABC):
    def __init__(self, env, video_path=None, device=torch.device('cpu')):
        super(GymEnvManager, self).__init__()
        self.env = env
        if video_path is not None:
            self.env = Monitor(env, video_path, force=True)
        self.state = None
        self.device = device
        self.reset()

    def reset(self):
        gym_state_data = self.env.reset()
        self.state = State(self.convert_gym_state_data(gym_state_data), False)
        self.terminal = False

    def close(self):
        self.env.close()

    def action_count(self):
        return self.env.action_space.n

    def take_action(self, action):
        gym_next_state_data, gym_rewards, self.terminal, _ = self.env.step(
            action)
        next_state_data = self.convert_gym_state_data(gym_next_state_data)
        rewards = self._change_rewards(gym_rewards, self.state.get_data(),
                                       next_state_data)
        self.state = State(next_state_data, self.terminal)
        return rewards

    def get_state(self):
        return self.state

    def convert_gym_state_data(self, gym_state_data):
        return self._change_state(
            torch.from_numpy(gym_state_data).float().to(self.device))

    @abstractmethod
    def _change_rewards(self, rewards, state_data, next_state_data):
        pass

    @abstractmethod
    def _change_state(self, state_data):
        pass


class MountainCarEnvManager(GymEnvManager):
    def __init__(self, video_path=None, device=torch.device('cpu')):
        super(MountainCarEnvManager, self).__init__(gym.make('MountainCar-v0'),
                                                    video_path, device)

    def feature_count(self):
        return 2

    def _change_rewards(self, rewards, state_data, next_state_data):
        rewards += next_state_data[0] * next_state_data[0]
        rewards += next_state_data[1] * next_state_data[1]
        rewards -= state_data[0] * state_data[0]
        rewards -= state_data[1] * state_data[1]
        return rewards.item()

    def _change_state(self, next_state):
        next_state[0] += 0.5
        next_state[0] *= 10.0
        next_state[1] *= 100.0
        return next_state
