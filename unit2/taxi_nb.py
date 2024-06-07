# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: deep-rl
#     language: python
#     name: python3
# ---

# # Explore env

import gymnasium as gym
env = gym.make("Taxi-v3", render_mode="rgb_array")

state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")

# # Train Q-Learning

# +
import gymnasium as gym
import numpy as np
import qlearning as ql

env = gym.make("Taxi-v3", render_mode="rgb_array")
qtable = np.zeros((env.observation_space.n, env.action_space.n))

model = {
    "env_id": "Taxi-v3",           # Name of the environment
    "qtable": qtable,              # Q-Table
    # Training parameters
    "max_steps": 99,               # Max steps per episode
    "n_training_episodes": 1000000,  # Total training episodes
    "learning_rate": 0.7,          # Learning rate
    "gamma": 0.95,                 # Discounting rate
    # Evaluation parameters
    "n_eval_episodes": 100,        # Total number of test episodes
    "eval_seed": [                 # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
        16,54,165,177,191,191,120,80,149,178,48,38,6,125,174,
        73,50,172,100,148,146,6,25,40,68,148,49,167,9,97,164,
        176,61,7,54,55, 161,131,184,51,170,12,120,113,95,126,
        51,98,36,135,54,82,45,95,89,59,95,124,9,113,58,85,51,
        134,121,169,105,21,30,11,50,65,12,43,82,145,152,97,106,
        55,31,85,38,112,102,168,123,97,21,83,158,26,80,63,5,81,
        32,11,28,148
    ],
    # Exploration parameters
    "max_epsilon": 1.0,            # Exploration probability at start
    "min_epsilon": 0.05,           # Minimum exploration probability
    "decay_rate": 0.005,           # Exponential decay rate for exploration prob
}

ql.train(env=env, **model)
# -

mean_reward, std_reward = ql.evaluate_agent(env=env, **model)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# # Push to Hub

# +
from hf_utils import push_to_hub

username = "monti-python"
repo_name = "q-Taxi-v3"

push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=model,
    env=env
)
# -

# # Load and evaluate other models

# +
import gymnasium as gym
from hf_utils import load_from_hub
from qlearning import evaluate_agent

model = load_from_hub(repo_id="ThomasSimonini/q-Taxi-v3", filename="q-learning.pkl") # Try to use another model
print(model)
env = gym.make(model["env_id"])
evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])
