
import torch
import numpy as np
from safetensors.torch import load_file
from cartpole.main import ActorCritic 

model_add = "min_ppo_cartpole.safetensors"

def analyze_svd(model, energy_threshold=0.95):
    svd_info = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data.cpu().numpy()
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

            print(f"Layer {name}: rank {len(S)}, k={k} (~{k/len(S)*100:.1f}% kept)")
    return svd_info


def summarize_svd(svd_info):
    ks = [v["k_95"] for v in svd_info.values()]
    full_ranks = [v["rank_full"] for v in svd_info.values()]
    overall_ratio = sum(ks) / sum(full_ranks)
    print(f"\nOverall rank ratio: {overall_ratio:.2f} "
          f"(≈{overall_ratio*100:.1f}% of weights needed for 95% energy)")
    return overall_ratio


if __name__ == "__main__":
    # Load model weights
    state_dim = 4      # CartPole obs dim
    n_actions = 2      # CartPole action dim
    model = ActorCritic(state_dim, n_actions)

    ckpt = load_file(model_add)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    # Run SVD analysis
    info = analyze_svd(model)
    k_value = summarize_svd(info)

    print(f"\nEstimated model compression factor (k): {k_value:.2f}")