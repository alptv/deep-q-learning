import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim

from qlearn.agents import EpsGreedyAgent
from qlearn.env import MountainCarEnvManager
from qlearn.qnet import QNetwork, QFunction
from qlearn.fit import fit
from qlearn.memory import ExperienceMemory



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
gamma = 0.99
eps_max = 1
eps_min = 0.1
eps_decay = (eps_max - eps_min) / 100_000
                                    
memory_capacity = 300_000
lr = 0.0002
target_update = 3000
episode_count = 2000


memory = ExperienceMemory(memory_capacity)
env_manager = env_manager = MountainCarEnvManager(device=device)



Q_network = QNetwork(env_manager.action_count(), env_manager.feature_count()).to(device)


agent = EpsGreedyAgent(Q_function=QFunction(Q_network),
                       action_count=env_manager.action_count(),
                       eps_max=eps_max,
                       eps_min=eps_min,
                       eps_decay=eps_decay)

optimizer = optim.Adam(params=Q_network.parameters(), lr=lr)

fit(episode_count, gamma, target_update, batch_size, env_manager, agent, memory, Q_network, optimizer, device)

torch.save(Q_network.state_dict(), './net_weights')