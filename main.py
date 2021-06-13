import torch
from qlearn.qnet import QNetwork, QFunction
from qlearn.env import MountainCarEnvManager
from qlearn.agents import EpsGreedyAgent

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_manager = env_manager = MountainCarEnvManager('./demonstration', device)
Q_network = QNetwork(env_manager.action_count(),
                     env_manager.feature_count()).to(device)
Q_network.load_state_dict(torch.load('./net_weights', map_location=device))

agent = EpsGreedyAgent(Q_function=QFunction(Q_network),
                       action_count=env_manager.action_count(),
                       eps_max=0,
                       eps_min=0,
                       eps_decay=0)

while not env_manager.is_done():
    state = env_manager.get_state()
    action = agent.select_action(state)
    env_manager.take_action(action)

env_manager.close()