import gymnasium as gym
from stable_baselines3 import PPO
import torch
import os
import time

# Import your multitask policy definition
from cartpole.leg_3 import MultiTaskPolicy  # <-- from your training file

MODEL_PATH = "ppo_multitask_lunar_walker.zip"
EPISODES = 3


def run_agent(model_path, task_name="lunar", episodes=3, render=True):
    """
    Run evaluation episodes for a trained multitask PPO model
    on either LunarLander or BipedalWalker.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Choose environment based on task
    if task_name == "lunar":
        env_id = "LunarLander-v2"
    elif task_name == "walker":
        env_id = "BipedalWalker-v3"
    else:
        raise ValueError(f"Unknown task: {task_name}")

    env = gym.make(env_id, render_mode="human" if render else None)

    # Load the multitask PPO model
    model = PPO.load(model_path)
    model.policy.set_task(task_name)  # important: choose correct head

    print(f"\n🎮 Evaluating {task_name.upper()} ({env_id}) for {episodes} episodes...\n")

    total_reward = 0
    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        step = 0

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)  # SB3 inference
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if render:
                time.sleep(0.01)  # slow down for visualization

        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, Steps = {step}")
        total_reward += ep_reward
        total_steps += step

    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes

    print("\n---------------------------------------------------------")
    print(f"Task: {task_name.upper()}")
    print(f"Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.1f}")
    print(f"Total Episodes = {episodes}")
    print("---------------------------------------------------------\n")

    env.close()


if __name__ == "__main__":
    # Run both environments back-to-back
    run_agent(MODEL_PATH, task_name="lunar", episodes=EPISODES, render=True)
    run_agent(MODEL_PATH, task_name="walker", episodes=EPISODES, render=True)