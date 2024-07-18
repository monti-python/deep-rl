import numpy as np
from collections import deque
# Gym
import gymnasium as gym
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from typing import Sequence


class Policy(nn.Module):
    def __init__(self, s_size: int, a_size: int, h_sizes: Sequence[int], **kwargs):
        super(Policy, self).__init__()        
        # Initialize the layers
        layers = []
        input_size = s_size
        # Add hidden layers
        for h_size in h_sizes:
            layers.append(nn.Linear(input_size, h_size))
            input_size = h_size
        # Add the output layer
        layers.append(nn.Linear(input_size, a_size))
        # Store the layers in a ModuleList
        self.layers = nn.ModuleList(layers)
        self.device = kwargs.get('device') or "cpu"

    # Define the forward pass
    def forward(self, x):
        # Pass the input through each layer with ReLU activation
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        # Apply softmax to the output layer
        return F.softmax(self.layers[-1](x), dim=1)

    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

def reinforce(env, policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=print_every)
    scores = []
    # Iterate over n_training_episodes
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state, info = env.reset()
        # Gather all log_probabilities and rewards for episode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Calculate the returns
        n_steps = len(rewards)
        returns = deque(maxlen=n_steps)
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item() ## eps is the smallest representable float
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Compute loss-function:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagate and compute new weights
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print("Episode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_deque)))

    return scores


def train(env, **hyperparameters):
    # Create policy and place it to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = hyperparameters.get("model", None) or Policy(
        hyperparameters["state_space"],
        hyperparameters["action_space"],
        hyperparameters["h_sizes"],
        device=device
    ).to(device)
    # Initialize optimizer
    cartpole_optimizer = optim.Adam(policy.parameters(), lr=hyperparameters["lr"])
    # Train model
    scores = reinforce(
        env,
        policy,
        cartpole_optimizer,
        hyperparameters["n_training_episodes"],
        hyperparameters["max_t"],
        hyperparameters["gamma"],
        100,
    )
    return policy, scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy, eval_seed=None, **kwargs):
    episode_rewards = []
    eval_env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        name_prefix=f"{kwargs.get('env_id')}_pg",
        episode_trigger=lambda i: i==0,
        #disable_logger=True,
    )
    for episode in tqdm(range(n_eval_episodes)):
        if eval_seed:
            state, info = eval_env.reset(seed=eval_seed[episode])
        else:
            state, info = eval_env.reset()
        step = 0
        terminated = False
        truncated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action, _ = policy.act(state)
            new_state, reward, terminated, truncated, _ = eval_env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

