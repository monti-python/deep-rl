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

# # Explore environment

# +
import gymnasium as gym

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")
# -

# We create our environment with gym.make("<name_of_the_environment>")- `is_slippery=False`: The agent always moves in the intended direction due to the non-slippery nature of the frozen lake (deterministic).
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action

# # Train Q-Learning

# +
# Train our agent
import qlearning as ql
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array")

Qtable_frozenlake = np.zeros((env.observation_space.n, env.action_space.n))

model = {
    "env_id": "FrozenLake-v1",     # Name of the environment
    "qtable": Qtable_frozenlake,   # Qtable object
    # Training parameters
    "max_steps": 99,               # Max steps per episode
    "learning_rate": 0.7,          # Learning rate
    "gamma": 0.95,                 # Discounting rate
    "n_training_episodes": 10000,  # Total training episodes
    "n_eval_episodes": 100,        # Total number of test episodes
    "eval_seed": [],               # The evaluation seed of the environment
    # Exploration parameters
    "max_epsilon": 1.0,            # Exploration probability at start
    "min_epsilon": 0.05,           # Minimum exploration probability
    "decay_rate": 0.0005,          # Exponential decay rate for exploration prob
}

Qtable_frozenlake = ql.train(env=env, **model)
# -

# Evaluate our Agent
mean_reward, std_reward = ql.evaluate_agent(env=env, **model)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

# # Publish to the Hub

# +
from ..hf_utils import push_to_hub

username = "monti-python"
repo_name = "q-FrozenLake-v1-4x4-noSlippery"

push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=model,
    env=env
)
