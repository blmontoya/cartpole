import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from safetensors.torch import save_file

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/multitask_ppo")

def make_env(env_id):
    return lambda: gym.make(env_id)

# MultiTask Network
class MultiTaskActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
        )
        # Input adapters
        self.input_adapters = nn.ModuleDict({
            "lunar": nn.Linear(8, 128),
            "walker": nn.Linear(24, 128)
        })
        # Actor heads
        self.actor_heads = nn.ModuleDict({
            "lunar": nn.Linear(128, 4),   # discrete actions
            "walker": nn.Linear(128, 6)   # BipedalWalker has 6 continuous actions, not 4
        })
        # Critic heads
        self.critic_heads = nn.ModuleDict({
            "lunar": nn.Linear(128, 1),
            "walker": nn.Linear(128, 1)
        })
        # Log std for continuous actions (6 dims for BipedalWalker)
        self.log_std = nn.Parameter(torch.zeros(6))
        # Current task
        self.current_task = "lunar"
    
    def set_task(self, task_name):
        self.current_task = task_name

    def forward_actor(self, obs):
        x = torch.relu(self.input_adapters[self.current_task](obs))
        features = self.shared_backbone(x)
        if self.current_task == "lunar":
            logits = self.actor_heads["lunar"](features)
            dist = Categorical(logits=logits)
        else:
            # Clamp log_std to prevent explosion
            log_std = torch.clamp(self.log_std, -20, 2)
            mu = self.actor_heads["walker"](features)
            dist = Normal(mu, self.log_std.exp())
        return dist

    def forward_critic(self, obs):
        x = torch.relu(self.input_adapters[self.current_task](obs))
        features = self.shared_backbone(x)
        value = self.critic_heads[self.current_task](features)
        return value.squeeze(-1)


# PPO Helper Functions
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation
    rewards, values, dones: lists of tensors [num_steps]
    """
    advantages = []
    gae = 0
    
    # Get next value (0 for terminal states)
    next_value = torch.zeros_like(values[-1])
    
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[step + 1]
        
        # TD error
        delta = rewards[step] + gamma * next_val * (1 - dones[step]) - values[step]
        # GAE
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    
    advantages = torch.stack(advantages)
    returns = advantages + torch.stack(values)
    return returns, advantages


# Training Loop
def train_multitask():
    # Create vectorized environments
    envs = {
        "lunar": DummyVecEnv([make_env("LunarLander-v3") for _ in range(4)]),
        "walker": DummyVecEnv([make_env("BipedalWalker-v3") for _ in range(4)])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    total_cycles = 300      # More cycles for better training
    steps_per_task = 2048   # More steps per rollout
    ppo_epochs = 10
    minibatch_size = 256    # Larger minibatch for more data
    clip_coef = 0.2
    
    for cycle in range(total_cycles):
        # 70% walker, 30% lunar during warmup
        task = "walker" if np.random.random() < 0.7 else "lunar"
        #task = "lunar" if cycle % 2 == 0 else "walker"
        print(f"\nTraining task: {task} (cycle {cycle})")
        model.set_task(task)
        env = envs[task]

        # For logging per cycle
        cycle_actor_loss = 0.0
        cycle_critic_loss = 0.0
        cycle_entropy = 0.0
        num_minibatches = 0


        obs = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        obs = obs / (obs.abs().max() + 1e-8)
        
        
        # Collect rollout       
        obs_list, actions_list, log_probs_list = [], [], []
        rewards_list, dones_list, values_list = [], [], []

        # Initialize cumulative reward tracking
        episode_reward = torch.zeros(env.num_envs, device=device)  # running reward per environment
        completed_rewards = []  # store total rewards for finished episodes

        for step in range(steps_per_task):
            with torch.no_grad():
                dist = model.forward_actor(obs)
                value = model.forward_critic(obs)
                action = dist.sample()
                if task == "lunar":
                    log_prob = dist.log_prob(action)
                else:
                    log_prob = dist.log_prob(action).sum(-1)

            # Step environment
            if task == "walker":
                action_np = torch.tanh(action).cpu().numpy()
            else:
                action_np = action.cpu().numpy()

            obs_next, reward, done, info = env.step(action_np)

            #if task == "walker":
            #    reward = reward / 100.0

            # Update episode reward
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            episode_reward += reward_tensor

            # When an environment is done, save its episode reward
            for i, d in enumerate(done):
                if bool(d):
                    completed_rewards.append(episode_reward[i].item())
                    episode_reward[i] = 0.0  # reset for next episode

            # Store rollout
            obs_list.append(obs)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)
            rewards_list.append(reward_tensor)
            dones_list.append(torch.tensor(done, dtype=torch.float32, device=device))

            obs = torch.tensor(obs_next, dtype=torch.float32, device=device)
            # Example running mean/std per task
            #obs = obs / (obs.abs().max() + 1e-8)

        # Compute returns and normalized advantages
        returns, advantages = compute_gae(rewards_list, values_list, dones_list)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch
        obs_batch = torch.stack(obs_list).view(-1, obs_list[0].shape[-1])
        actions_batch = torch.stack(actions_list).view(-1, *actions_list[0].shape[1:])
        log_probs_batch = torch.stack(log_probs_list).view(-1)
        returns_batch = returns.view(-1)
        advantages_batch = advantages.view(-1)
        # PPO Update - multiple epochs over minibatches
        batch_size = obs_batch.shape[0]
        global_step = 0  # count across all minibatches
        for epoch in range(ppo_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs_batch[mb_indices]
                mb_actions = actions_batch[mb_indices]
                mb_old_log_probs = log_probs_batch[mb_indices]
                mb_returns = returns_batch[mb_indices]
                mb_advantages = advantages_batch[mb_indices]

                dist = model.forward_actor(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)
                if task == "walker":
                    new_log_probs = new_log_probs.sum(-1)

                new_values = model.forward_critic(mb_obs)

                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (mb_returns - new_values).pow(2).mean()
                entropy = dist.entropy().mean()

                critic_coeff = 0.05 
                entropy_coeff = 0.01 if task == "lunar" else 0.001
                loss = actor_loss + critic_coeff * critic_loss - entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                # Accumulate metrics
                cycle_actor_loss += actor_loss.item()
                cycle_critic_loss += critic_loss.item()
                cycle_entropy += entropy.item()
                num_minibatches += 1

        # Average over minibatches in the cycle
        avg_actor_loss = cycle_actor_loss / num_minibatches
        avg_critic_loss = cycle_critic_loss / num_minibatches
        avg_entropy = cycle_entropy / num_minibatches

        # Logging to TensorBoard
        writer.add_scalar(f"Loss/Actor_{task}", avg_actor_loss, cycle)
        writer.add_scalar(f"Loss/Critic_{task}", avg_critic_loss, cycle)
        writer.add_scalar(f"Loss/Entropy_{task}", avg_entropy, cycle)

        # Reward logging (already per cycle)
        avg_reward = sum(completed_rewards) / len(completed_rewards) if completed_rewards else 0.0
        max_reward = max(completed_rewards) if completed_rewards else 0.0
        writer.add_scalar(f"Reward/{task}", avg_reward, cycle)
        writer.add_scalar(f"RewardMax/{task}", max_reward, cycle)

        print(f"  Avg Reward: {avg_reward:.2f}, Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}, Entropy: {avg_entropy:.4f}")


    print("Training completed!")
    
    # Close environments
    for env in envs.values():
        env.close()

    return model

if __name__ == "__main__":
    model = train_multitask()
    save_file(model.state_dict(), "multitask_model.safetensors")