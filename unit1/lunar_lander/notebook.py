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

# # Explore Observation and Action Spaces

# +
import gymnasium as gym
env = gym.make("LunarLander-v2")
env.reset()
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action
# -

# # Take random actions

# +
import gymnasium as gym
import moviepy.editor as mpy

# First, we create our environment called LunarLander-v2
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Then we reset this environment
observation, info = env.reset()

# List to store frames
frames = []

for _ in range(200):
  # Take a random action
  action = env.action_space.sample()
  print("Action taken:", action)

  # Do this action in the environment and get
  # next_state, reward, terminated, truncated and info
  observation, reward, terminated, truncated, info = env.step(action)

  # Render the environment and get the frame as an RGB array
  frame = env.render()
  frames.append(frame)

  # If the game is terminated (in our case we land, crashed) or truncated (timeout)
  if terminated or truncated:
      # Reset the environment
      print("Environment is reset")
      observation, info = env.reset()

env.close()

clip = mpy.ImageSequenceClip(frames, fps=30)
clip.write_videofile("lunar_lander_random.mp4", codec="libx264")
# -

# # Use PPO

from ppo import train, evaluate
train()
evaluate()

# # Publish to HF Hub

from ppo import send_to_hf
send_to_hf()
