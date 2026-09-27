import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from game.constants import *
from ai.neural import AlphaTiger, load_dynamic_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a heatmap for AlphaTiger model weights.")
    parser.add_argument(
        '--ckpt', 
        type=str, 
        default=DEFAULT_MODEL, 
        help='Path to the model checkpoint file (default: uses DEFAULT_MODEL from constants)'
    )
    parser.add_argument(
        '--layer', 
        type=str, 
        default='fc1.weight', 
        help='Name of the model layer to visualize (e.g., fc1.weight, value_fc1.weight, policy_fc2.weight)'
    )
    args = parser.parse_args()

    device = torch.device('cpu')
    model = load_dynamic_model(args.ckpt, device)
    state_dict = model.state_dict()

    if args.layer not in state_dict:
        available_layers = list(state_dict.keys())
        raise ValueError(
            f"Layer '{args.layer}' not found. Available layers are:\n- " + 
            "\n- ".join(available_layers)
        )

    weights = state_dict[args.layer].cpu().numpy()

    if weights.ndim > 2:
        weights = weights.reshape(weights.shape[0], -1)

    plt.figure(figsize=(16, 8))
    sns.heatmap(weights, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Weight Magnitude'})

    plt.title(f'AlphaTiger Layer Weights ({args.layer})', fontsize=16)
    plt.xlabel('Input Dimension', fontsize=12)
    plt.ylabel('Output Dimension', fontsize=12)
    plt.tight_layout()

    plt.show()