#!/usr/bin/env python

import torch
import numpy as np
from safetensors.torch import load_file
from cartpole.main import ActorCritic 
import argparse

def analyze_svd(state_dict, energy_threshold=0.95):
    """Analyze SVD for all 2D tensors (weight matrices) in state dict"""
    svd_info = {}
    
    for name, tensor in state_dict.items():
        # Only analyze 2D tensors (weight matrices), skip biases and 1D tensors
        if len(tensor.shape) == 2:
            W = tensor.cpu().numpy()
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            energy = np.cumsum(S ** 2) / np.sum(S ** 2)
            k = np.searchsorted(energy, energy_threshold) + 1

            svd_info[name] = {
                "shape": W.shape,
                "rank_full": len(S),
                "k_95": k,
                "energy_95": energy_threshold,
                "compression_ratio": k / len(S),
            }

            print(f"Layer {name}: shape {W.shape}, rank {len(S)}, k={k} (~{k/len(S)*100:.1f}% kept)")
    
    return svd_info

def summarize_svd(svd_info):
    if not svd_info:
        print("No 2D weight matrices found in model!")
        return 0.0
    
    ks = [v["k_95"] for v in svd_info.values()]
    full_ranks = [v["rank_full"] for v in svd_info.values()]
    overall_ratio = sum(ks) / sum(full_ranks)
    print(f"\nOverall rank ratio: {overall_ratio:.2f} "
          f"(≈{overall_ratio*100:.1f}% of weights needed for {svd_info[list(svd_info.keys())[0]]['energy_95']*100:.0f}% energy)")
    return overall_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SVD analysis on any safetensors model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the safetensors model file"
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.95,
        help="Energy threshold for SVD analysis (default: 0.95)"
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Key to extract from state dict if model is nested (e.g., 'model', 'state_dict')"
    )
    
    args = parser.parse_args()

    # Load model weights
    print(f"Loading model from: {args.model_path}")
    state_dict = load_file(args.model_path)
    
    # Handle nested state dicts
    if args.key and args.key in state_dict:
        state_dict = state_dict[args.key]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        print("Detected nested 'model' key, extracting...")
        state_dict = state_dict["model"]
    
    print(f"Found {len(state_dict)} tensors in state dict\n")

    # Run SVD analysis
    info = analyze_svd(state_dict, energy_threshold=args.energy_threshold)
    k_value = summarize_svd(info)

    print(f"\nEstimated model compression factor (k): {k_value:.2f}")