#!/usr/bin/env python
import torch
import numpy as np
from safetensors.torch import load_file
import pandas as pd
import argparse
from pathlib import Path


def analyze_weights(state_dict, output_csv="model_stats.csv"):
    """Analyze weight statistics for all tensors in state dict"""
    stats = []
    
    for name, param in state_dict.items():
        W = param
        
        # Basic stats for all tensors
        stat_entry = {
            "layer": name,
            "shape": list(W.shape),
            "mean": W.mean().item(),
            "std": W.std().item(),
            "min": W.min().item(),
            "max": W.max().item(),
            "num_params": W.numel()
        }
        
        # Spectral norm only for 2D tensors (weight matrices)
        if len(W.shape) == 2:
            try:
                u, s, v = torch.linalg.svd(W)
                stat_entry["spectral_norm"] = s.max().item()
                stat_entry["condition_number"] = (s.max() / s.min()).item() if s.min() > 0 else float("inf")
            except RuntimeError:
                stat_entry["spectral_norm"] = float("nan")
                stat_entry["condition_number"] = float("nan")
        else:
            stat_entry["spectral_norm"] = None
            stat_entry["condition_number"] = None
        
        stats.append(stat_entry)

    # Create csv_files directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df = pd.DataFrame(stats)
    df.to_csv(output_csv, index=False)
    print(df.to_string())

    # --- Summary ---
    total_params = df["num_params"].sum()
    weight_matrices = df[df["spectral_norm"].notna()]
    
    print("\n=== Model Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Total layers: {len(df)}")
    print(f"Weight matrices (2D): {len(weight_matrices)}")
    
    if len(weight_matrices) > 0:
        avg_spectral_norm = weight_matrices["spectral_norm"].mean()
        max_spectral_norm = weight_matrices["spectral_norm"].max()
        layer_max_norm = weight_matrices.loc[weight_matrices["spectral_norm"].idxmax(), "layer"]
        
        print(f"Average spectral norm: {avg_spectral_norm:.4f}")
        print(f"Maximum spectral norm: {max_spectral_norm:.4f} (Layer: {layer_max_norm})")
        
        avg_condition = weight_matrices["condition_number"].replace([float('inf')], float('nan')).mean()
        if not np.isnan(avg_condition):
            print(f"Average condition number: {avg_condition:.2f}")
    
    print(f"\nResults saved to: {output_csv}")
    print("=====================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run weight analysis on any safetensors model")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the safetensors model file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated from model name in workspace/csv_files/)"
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Key to extract from state dict if model is nested (e.g., 'model', 'state_dict')"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="watchers",
        help="Prefix for the CSV filename (default: watchers)"
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

    # Determine output CSV path
    if args.output is None:
        # Get the safetensors filename without extension
        model_name = Path(args.model_path).stem  # e.g., "min_ppo"
        
        # Find workspace root (go up until we find workspace directory)
        current_dir = Path.cwd()
        workspace_root = None
        
        for parent in [current_dir] + list(current_dir.parents):
            if parent.name == "workspace":
                workspace_root = parent
                break
        
        # If we can't find workspace, use current directory's parent
        if workspace_root is None:
            workspace_root = current_dir
            print(f"Warning: Could not find 'workspace' directory, using {workspace_root}")
        
        # Create path with prefix: workspace/csv_files/watchers_model_name.csv
        output_csv = workspace_root / "csv_files" / f"{args.prefix}_{model_name}.csv"
    else:
        output_csv = args.output

    # Run analysis
    analyze_weights(state_dict, output_csv=str(output_csv))