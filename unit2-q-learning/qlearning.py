import numpy as np
import gymnasium as gym
import tqdm
import pickle5 as pickle
from tqdm.notebook import tqdm
import random

def greedy_policy(qtable: np.ndarray, state: int) -> int:
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(qtable[state])
    return action

def epsilon_greedy_policy(qtable: np.ndarray, state: int, eps: float) -> int:
    # Randomly generate a number between 0 and 1
    random_num = random.random()
    # if random_num > greater than epsilon --> exploitation
    if random_num > eps:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = greedy_policy(qtable, state)
    # else --> exploration
    else:
        action = random.randrange(qtable.shape[1])

    return action


def train(
    n_training_episodes: int,
    min_epsilon: float,
    max_epsilon: float,
    decay_rate: float,
    env: gym.Env,
    max_steps: int,
    learning_rate: float,
    gamma: float,
    qtable: np.ndarray,
    **kwargs,
):
    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # Reset the environment
        state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(qtable, state, epsilon)
            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            td_target = reward + gamma * np.max(qtable[new_state])
            td_error = td_target - qtable[state][action]
            qtable[state][action] += learning_rate * td_error
            # If terminated or truncated finish the episode
            if terminated or truncated:
                break
            # Our next state is the new state
            state = new_state

    return qtable

def evaluate_agent(env, max_steps, n_eval_episodes, qtable, eval_seed, **kwargs):
    episode_rewards = []
    eval_env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="./video",
        name_prefix=f"{kwargs.get('env_id')}_ql",
        episode_trigger=lambda i: i==0,
        disable_logger=True,
    )
    for episode in tqdm(range(n_eval_episodes)):
        if eval_seed:
            state, info = eval_env.reset(seed=eval_seed[episode])
        else:
            state, info = eval_env.reset()
        step = 0
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(qtable, state)
            new_state, reward, terminated, truncated, info = eval_env.step(action)
            total_rewards_ep += reward
            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward