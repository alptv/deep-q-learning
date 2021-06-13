import torch
import copy

from qlearn.memory import Experience


def experiences_to_tensors(experiences, device):
    states = []
    next_states = []
    actions = []
    rewards = []
    terminal = []
    for experience in experiences:
        states.append(experience.state.data)
        next_states.append(experience.next_state.data)
        actions.append(
            torch.tensor(experience.action, dtype=torch.int64).to(device))
        rewards.append(
            torch.tensor(experience.reward, dtype=torch.float32).to(device))
        terminal.append(torch.tensor(experience.is_terminal()).to(device))
    return torch.stack(states), torch.stack(actions), torch.stack(next_states), \
           torch.stack(rewards), torch.stack(terminal)


def gather_actions_Q_values(Q_values, actions, device):
    item_count = Q_values.shape[0]
    actions_Q_values = torch.zeros(item_count).to(device)
    for i in range(item_count):
        actions_Q_values[i] = Q_values[i][actions[i]]
    return actions_Q_values


def max_future_rewards(Q_values, terminal, device):
    item_count = terminal.shape[0]
    max_rewards = torch.zeros(item_count).to(device)
    for i in range(item_count):
        if not terminal[i]:
            max_rewards[i] = Q_values[i].max()
        else:
            max_rewards[i] = 0
    return max_rewards


def fit(episode_count, gamma, target_update, batch_size, env_manager, agent,
          memory, Q_network, optimizer, device):

    target_Q_network = copy.deepcopy(Q_network)
    target_Q_network.eval()
    target_Q_network.to(device)

    episode_tracker = EpisodeTracker()
    step = 0
    for _ in range(episode_count):
        while not env_manager.is_done():
            step += 1
            if step % target_update == 0:
                target_Q_network.load_state_dict(Q_network.state_dict())

            state = env_manager.get_state()
            action = agent.select_action(state)
            reward = env_manager.take_action(action)
            next_state = env_manager.get_state()
            memory.add(Experience(state, action, next_state, reward))
            episode_tracker.add_reward(reward)

            if memory.can_provide_batch(batch_size):
                experiences = memory.provide_batch(batch_size)
                states, actions, next_states, rewards, terminal = experiences_to_tensors(
                    experiences, device)

                Q_values = gather_actions_Q_values(Q_network(states), actions,
                                                   device)

                next_Q_values = max_future_rewards(
                    target_Q_network(next_states), terminal, device)

                target_Q_values = gamma * next_Q_values + rewards

                loss = torch.nn.SmoothL1Loss()(Q_values, target_Q_values)
                episode_tracker.add_loss(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        episode_tracker.track()
        episode_tracker.reset()
        env_manager.reset()
    env_manager.close()


class EpisodeTracker:
    def __init__(self):
        self.episode_number = 1
        self.reward = 0
        self.loss = 0

    def add_reward(self, delta):
        self.reward += delta

    def add_loss(self, delta):
        self.loss += delta

    def reset(self):
        self.episode_number += 1
        self.reward = 0
        self.loss = 0

    def track(self):
        print("Episode number: {}, Total reward: {}, Total loss: {}".format(
            self.episode_number, self.reward, self.loss), flush=True)
