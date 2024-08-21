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

# # Explore Robotic Arm Environment

# +
import gymnasium as gym

env_id = "PandaReachDense-v3"
# Create the env
env = gym.make(env_id)
# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space
# Print size of the observation space
print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation
# Print size of the action space
print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action
# -

# # Train Agent

# +
import gymnasium as gym
import panda_gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# Initialize env
env = make_vec_env(env_id, n_envs=4)
# Adding this wrapper to normalize the observation and the reward
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
# Train model
model = A2C(policy = "MultiInputPolicy", env = env, verbose=1)
model.learn(1_000_000)
# Save the model and  VecNormalize statistics when saving the agent
model.save("a2c-PandaReachDense-v3")
env.save("vec_normalize.pkl")
# -

# # Evaluate Agent

# +
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

# Load the saved statistics
eval_env = DummyVecEnv([lambda: gym.make("PandaReachDense-v3")])
eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
# We need to override the render_mode
eval_env.render_mode = "rgb_array"
# # Record video
eval_env = VecVideoRecorder(
    venv=eval_env,
    video_folder="./video",
    name_prefix=f"PandaReach",
    record_video_trigger=lambda i: i==0,
)

# do not update them at test time
eval_env.training = False
# reward normalization is not needed at test time
eval_env.norm_reward = False

# Load the agent
model = A2C.load("a2c-PandaReachDense-v3")
mean_reward, std_reward = evaluate_policy(model, eval_env)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
# -

# # Push Model to Hub

# +
from huggingface_sb3 import package_to_hub

package_to_hub(
    model=model,
    model_name=f"a2c-{env_id}",
    model_architecture="A2C",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=f"monti-python/a2c-{env_id}", # Change the username
    commit_message="Initial commit",
)
