#!/usr/bin/env python
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
import os

os.environ["TORCH_DISABLE_SOURCE_INSPECTION"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

# import safetensors
from torch.distributions import Categorical

# add tensorboard later
# Actor-Critic MLP
# Maybe seperate into the actor and the critic classes?
# Maybe add dropout and fully connected layers
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.up(self.down(x))

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LeakyReLU(0.01), nn.Linear(64, 64), nn.LeakyReLU(0.01)
        )
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# Maybe make into a seperate method?
def create_cartpole_model(model_path):
    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(state_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # Hyperparameters
    gamma = 0.99
    eps_clip = 0.2
    ppo_epochs = 4
    batch_size = 64
    steps_per_update = 2048
    epochs = 90

    # TensorBoard 
    writer = SummaryWriter("runs/cartpole_ppo")

    # Training loop
    for epoch in range(epochs):
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        state = env.reset()[0]
        ep_rewards = []
        episode_lengths = []
        ep_reward = 0
        episode_steps = 0

        # For each step in an epoch
        for step in range(steps_per_update):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            episode_steps += 1
            done = terminated or truncated

            # Store trajectory
            states.append(state)
            actions.append(action.item())
            log_probs.append(dist.log_prob(action).item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            ep_reward += reward
            state = next_state

            if done:
                ep_rewards.append(ep_reward)
                avg_reward = np.mean(ep_rewards)
                ep_reward = 0
                episode_lengths.append(episode_steps)
                episode_steps = 0
                state = env.reset()[0]

        # Compute discounted returns & advantages
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            G = r + gamma * G * (1 - d)
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)
        advantages = returns - values

        # Convert lists to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)

        # PPO update
        for _ in range(ppo_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                logits, value = model(batch_states)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)

                # Actor loss
                ratios = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()

                # Critic loss
                critic_loss = ((batch_returns - value.squeeze()) ** 2).mean()

                loss = actor_loss + 0.5 * critic_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Logging per epoch
        avg_reward = np.mean(ep_rewards)
        max_reward = np.max(ep_rewards)
        writer.add_scalar("Reward/Avg", avg_reward, epoch)
        writer.add_scalar("Reward/Max", max_reward, epoch)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Avg Reward: {np.mean(ep_rewards):.2f}, Max Reward: {np.max(ep_rewards):.2f}, Avg Steps: {np.mean(episode_lengths):.2f}"
            )

    # Save model
    save_file(model.state_dict(), model_path)

    print("Training complete. Model saved as ", model_path)
    writer.close()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a trained CartPole agent and save to a safetensor model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the safetensors model file (or just filename to search in workspace)"
    )
    
    args = parser.parse_args()
    
    # Run the agent
    create_cartpole_model(model_path=args.model_path)