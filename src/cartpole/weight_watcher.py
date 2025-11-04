#!/usr/bin/env python
import weightwatcher as ww
import weightwatcher.weightwatcher as core

# Patch for the PEFT bug in 0.7.x
if not hasattr(core, "PEFT"):
    core.PEFT = "peft"

import torch
import numpy as np
from safetensors.torch import load_file
from cartpole.main import ActorCritic 
import pandas as pd
import gymnasium as gym
import argparse

model_add = "min_ppo_cartpole.safetensors"
stats_add = "min_ppo_stats.csv"

def run_weight_watcher(model_path, render=False):
    # --- Environment ---
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # --- Load model ---
    model = ActorCritic(state_dim, n_actions)
    state_dict = load_file(model_path)  # safetensors loads a dict of tensors
    model.load_state_dict(state_dict)

    print("✅ Model loaded successfully.")
    # --- Analyze Linear layers ---
    stats = []
    for name, param in model.named_parameters():
        if len(param.shape) > 1:  # skip biases
            W = param.data
            mean = W.mean().item()
            std = W.std().item()
            # Spectral norm (largest singular value)
            try:
                u, s, v = torch.linalg.svd(W)
                spectral_norm = s.max().item()
            except RuntimeError:
                spectral_norm = float("nan")

            stats.append({
                "layer": name,
                "shape": list(W.shape),
                "mean": mean,
                "std": std,
                "spectral_norm": spectral_norm,
                "num_params": W.numel()
            })

    # Save to CSV
    df = pd.DataFrame(stats)
    df.to_csv(stats_add, index=False)
    print(df)

    # --- Mini summary ---
    total_params = df["num_params"].sum()
    avg_spectral_norm = df["spectral_norm"].mean()
    max_spectral_norm = df["spectral_norm"].max()
    layer_max_norm = df.loc[df["spectral_norm"].idxmax(), "layer"]

    print("\n=== Model Summary ===")
    print(f"Total Linear parameters: {total_params}")
    print(f"Average spectral norm: {avg_spectral_norm:.4f}")
    print(f"Maximum spectral norm: {max_spectral_norm:.4f} (Layer: {layer_max_norm})")
    print("=====================\n")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run weight watcher on a safetensors model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the safetensors model file"
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=4,
        help="State dimension (default: 4 for CartPole)"
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=2,
        help="Number of actions (default: 2 for CartPole)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_stats.csv",
        help="Output CSV file path (default: model_stats.csv)"
    )
    
    args = parser.parse_args()

    # Run weight watcher analysis
    run_weight_watcher(
        args.model_path, 
        state_dim=args.state_dim,
        n_actions=args.n_actions,
        render=args.render,
        output_csv=args.output
    )