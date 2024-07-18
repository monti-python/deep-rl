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

# # Train CartPole Policy

# +
import gym
from policy_gradient import train, evaluate_agent

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id)
# Create the evaluation env
eval_env = gym.make(env_id, render_mode="rgb_array")

cartpole_hyperparameters = {
    "env_id": "CartPole-v1",
    "state_space": env.observation_space.shape[0],
    "action_space": env.action_space.n,
    "h_sizes": [16],
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 0.99,
    "lr": 5e-3,
    "env_id": env_id,
}

policy, scores = train(env, **cartpole_hyperparameters)

evaluate_agent(
    eval_env,
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["n_evaluation_episodes"],
    policy,
    env_id=eval_env,
)
# -

# # Push CartPole Model to Hub

# +
from hf_utils import push_to_hub

username = "monti-python"
repo_name = "CartPole-v1"
push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=policy,
    hyperparameters=cartpole_hyperparameters,
    eval_env=eval_env,
)

# -

# # Train PixelCopter Policy

# +
import gymnasium

env_id = "Pixelcopter-PLE-v0"
env = gymnasium.make("GymV21Environment-v0", env_id=env_id)

pixelcopter_hyperparameters = {
    "state_space": int(env.observation_space.shape[0]),
    "action_space": int(env.action_space.n),
    "h_sizes": [64, 128],
    "n_training_episodes": 50000,
    "n_evaluation_episodes": 10,
    "max_t": 10000,
    "gamma": 0.99,
    "lr": 1e-4,
    "env_id": env_id,
    "model": policy,
}

policy, scores = train(env, **pixelcopter_hyperparameters)
# Evaluate agent
eval_env = gymnasium.make("GymV21Environment-v0", env_id=env_id, render_mode="rgb_array")
evaluate_agent(
    eval_env,
    pixelcopter_hyperparameters["max_t"],
    pixelcopter_hyperparameters["n_evaluation_episodes"],
    policy,
    env_id=env_id,
)
# -

# # Push Pixelcopter Model to Hub

# +
from hf_utils import push_to_hub

username = "monti-python"
repo_name = "Pixelcopter-PLE-v0"
push_to_hub(
    repo_id=f"{username}/{repo_name}",
    model=policy,
    hyperparameters=pixelcopter_hyperparameters,
    eval_env=eval_env,
)

# -
