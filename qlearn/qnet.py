import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, action_count, feature_count):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=feature_count, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=action_count)

    def forward(self, t):
        t = self.fc1(t)
        t = torch.tanh(t)

        t = self.fc2(t)
        t = torch.tanh(t)

        t = self.out(t)
        return t


class QFunction:
    def __init__(self, network):
        self.network = network

    def __call__(self, state):
        with torch.no_grad():
            return self.network(state.unsqueeze(0)).squeeze(0)
