#!/usr/bin/env python3
import gymnasium as gym
import torch
import argparse
import os
from safetensors.torch import load_file
from lunar_walker_mlp import MultiTaskActorCritic

def run_agent(model_path, task="lunar", episodes=5, render=True, deterministic=True):
    """Load a trained multitask model and run evaluation episodes."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Validate task
    if task not in ["lunar", "walker"]:
        raise ValueError(f"Task must be 'lunar' or 'walker', got: {task}")
    
    # Create environment
    env_ids = {
        "lunar": "LunarLander-v3",
        "walker": "BipedalWalker-v3"
    }
    env = gym.make(env_ids[task], render_mode="human" if render else None)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskActorCritic().to(device)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.set_task(task)
    
    print(f"\n{'='*60}")
    print(f"Running {task.upper()} environment")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes}")
    print(f"Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"{'='*60}\n")
    
    total_reward = 0
    total_steps = 0
    
    # Run episodes
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        ep_reward = 0
        ep_steps = 0
        done = False
        
        while not done:
            with torch.no_grad():
                dist = model.forward_actor(obs)
                
                if deterministic:
                    # Use best action
                    if task == "lunar":
                        action = torch.argmax(dist.probs).item()
                    else:
                        action = dist.mean.cpu().numpy()[0]
                        action = torch.tanh(torch.tensor(action)).numpy()
                else:
                    # Sample from distribution
                    action_tensor = dist.sample()
                    if task == "lunar":
                        action = action_tensor.item()
                    else:
                        action = torch.tanh(action_tensor).cpu().numpy()[0]
            
            obs_next, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
            
            obs = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
            
            print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, Step = {ep_steps}", end="\r")
        
        total_reward += ep_reward
        total_steps += ep_steps
        print(f"Episode {ep+1}: Reward = {ep_reward:.2f}, Steps = {ep_steps}     ")
    
    avg_reward = total_reward / episodes
    avg_steps = total_steps / episodes
    
    print("\n" + "="*60)
    print(f"RESULTS")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Total Episodes: {episodes}")
    print("="*60 + "\n")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a trained multitask agent from a safetensors model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        %(prog)s model.safetensors --task lunar
        %(prog)s model.safetensors --task walker --episodes 10
        %(prog)s model.safetensors --task lunar --stochastic --no-render
        """
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the safetensors model file"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["lunar", "walker"],
        default="lunar",
        help="Which environment to run: 'lunar' for LunarLander or 'walker' for BipedalWalker (default: lunar)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (run headless)"
    )
    
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic action sampling instead of deterministic (best action)"
    )
    
    args = parser.parse_args()
    
    # Run the agent
    run_agent(
        model_path=args.model_path,
        task=args.task,
        episodes=args.episodes,
        render=not args.no_render,
        deterministic=not args.stochastic
    )
