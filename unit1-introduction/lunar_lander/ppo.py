import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub
import logging


model_name = "ppo-LunarLander"

def train():
    # Create environment
    env = gym.make("LunarLander-v2")
    # Instantiate the agent
    try:
        logging.info(f"Loading model: '{model_name}'")
        model = PPO.load(model_name, env=env)
    except FileNotFoundError as e:
        logging.info(f"Model '{model_name}' not found. Creating new...")
        model = PPO(
            policy = 'MlpPolicy',
            env = env,
            n_steps = 1024,
            batch_size = 64,
            n_epochs = 4,
            gamma = 0.999,
            gae_lambda = 0.98,
            ent_coef = 0.01,
            verbose=1,
        )

    # Train it
    logging.info("Traning agent...")
    model.learn(total_timesteps=1e6)
    # Save the model
    logging.info("Saving the model to disk...")
    model.save(model_name)
    env.close()


def evaluate():
    # Evaluate the model
    model = PPO.load(model_name)
    eval_env = Monitor(gym.make("LunarLander-v2", render_mode="rgb_array"))
    eval_env = gym.wrappers.RecordVideo(
        env=eval_env,
        video_folder="./video",
        name_prefix="lunar_lander_ppo",
        episode_trigger=lambda i: i==0,
        disable_logger=True,
    )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True, render=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def send_to_hf():
    ## Define a repo_id
    repo_id = "monti-python/deep-rl"
    # Define the name of the environment
    env_id = "LunarLander-v2"
    # Create the evaluation env and set the render_mode="rgb_array"
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])
    model_name = "ppo-LunarLander"
    model = PPO.load(model_name, env=eval_env)
    # Define the model architecture we used
    model_architecture = "PPO"
    # This method saves, evaluates, generates a model card, and records
    # a replay video of your agent before pushing the repo to the hub
    package_to_hub(
        model=model,
        model_name=model_name,
        model_architecture=model_architecture,
        env_id=env_id,
        eval_env=eval_env,
        repo_id=repo_id,
        commit_message="Publish PPO model for LunarLander-v2"
    )