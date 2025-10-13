# Copyright (c) 2025, Gary Lvov
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gymnasium as gym
import torch

# from safetensors.torch import load_file  # <-- safetensors
from main import ActorCritic  # adjust if your structure is different


def run_agent(model_path, episodes=5, render=True):
    """Load a trained Actor-Critic model and run evaluation episodes."""

    # --- Environment ---
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # --- Load model ---
    model = ActorCritic(state_dim, n_actions)
    # state_dict = load_file(model_path)  # safetensors loads a dict of tensors
    # model.load_state_dict(state_dict)
    model.eval()

    # --- Run episodes ---
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        ep_steps = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()  # pick best action

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

        print(f"Episode {ep+1}: Reward = {ep_reward}, Steps = {ep_steps}")

    env.close()


if __name__ == "__main__":
    # Example usage: adjust path if needed
    run_agent("ppo_cartpole.safetensors", episodes=5, render=True)
