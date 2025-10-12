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
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# --- Environment ---
env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()
state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# PPO | Continiuous Actor Critic | stable baselines


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        return self.model(X)


class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64, 32),
            activation(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.model(X)


# --- Hyperparameters ---
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
critic_opt = optim.Adam(critic.parameters(), lr=3e-3)
gamma = 0.99
eps_clip = 0.2
epochs = 1000
steps_per_update = 2048

# --- PPO Training Loop ---
for ep in range(epochs):
    state, _ = env.reset()
    done = False
    log_probs = []
    values = []
    rewards = []
    states = []
    actions = []

    while len(states) < steps_per_update:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        probs = actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        value = critic(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action.item())

        log_prob = dist.log_prob(action)

        # store trajectory
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        states.append(state_tensor)
        actions.append(action)
        # a
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

    # --- Compute advantages ---
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    values = torch.stack(values).squeeze()
    advantages = returns - values.detach()

    # --- Update Actor ---
    for _ in range(4):  # PPO epochs
        for log_prob, advantage, state, action in zip(
            log_probs, advantages, states, actions
        ):
            probs = actor(state)
            dist = Categorical(probs)
            new_log_prob = dist.log_prob(action)
            ratio = (new_log_prob - log_prob).exp()
            clipped = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(ratio * advantage, clipped)
            actor_opt.zero_grad()
            loss.backward()
            actor_opt.step()

    # --- Update Critic ---
    for _ in range(4):
        for state, ret in zip(states, returns):
            value = critic(state)
            loss = (ret - value) ** 2
            critic_opt.zero_grad()
            loss.backward()
            critic_opt.step()

env.close()
