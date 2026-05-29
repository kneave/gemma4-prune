#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""
Quick expert ranking using the model's own per_expert_scale weights.

No forward passes needed — extracts the learned importance weights directly
from the router's per_expert_scale parameter. Experts with low scale values
are less important according to the model's own training.

Gemma 4 router structure per layer:
  - router.proj.weight [128, 2816]     (routing projection)
  - router.scale [2816]               (input scaling)
  - router.per_expert_scale [128]      (expert importance weights)

Usage:
    python 01a_quick_rank.py --model-path /home/kitt/gemma4-prune/bf16 \
                             --output-dir /home/kitt/gemma4-prune/calibration
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np


def quick_rank(model_path, output_dir):
    """Extract per_expert_scale from the model and rank experts."""
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    from collections import defaultdict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try loading from safetensors directly (fast, no GPU needed)
    model_dir = Path(model_path)
    safetensor_files = sorted(model_dir.glob("model-*.safetensors"))
    if not safetensor_files:
        safetensor_files = [model_dir / "model.safetensors"]

    print(f"Scanning {len(safetensor_files)} safetensors files for router weights...")

    # Extract per_expert_scale and proj.weight norms per layer
    layer_scales = {}  # layer_idx -> per_expert_scale tensor
    layer_proj_norms = {}  # layer_idx -> per-expert norm of proj.weight rows

    for st_file in safetensor_files:
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                if "router.per_expert_scale" in key:
                    # Extract layer index
                    # Key format: model.language_model.layers.{L}.router.per_expert_scale
                    parts = key.split(".")
                    for i, p in enumerate(parts):
                        if p == "layers":
                            layer_idx = int(parts[i + 1])
                            break
                    tensor = f.get_tensor(key).float().numpy()
                    layer_scales[layer_idx] = tensor
                    print(f"  Layer {layer_idx:2d}: per_expert_scale shape {tensor.shape}, "
                          f"min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

                if "router.proj.weight" in key and "experts" not in key:
                    parts = key.split(".")
                    for i, p in enumerate(parts):
                        if p == "layers":
                            layer_idx = int(parts[i + 1])
                            break
                    tensor = f.get_tensor(key).float()  # [128, 2816]
                    # Per-expert norm of each row
                    norms = tensor.norm(dim=1).numpy()
                    layer_proj_norms[layer_idx] = norms

    num_layers = len(layer_scales)
    num_experts = len(next(iter(layer_scales.values())))
    print(f"\nFound {num_layers} layers, {num_experts} experts per layer")

    # Compute ranking using per_expert_scale * proj_norm as importance score
    # (same principle as esokullu: combine learned importance with routing strength)
    stats = {
        "model_path": str(model_path),
        "method": "per_expert_scale",
        "num_samples": 0,  # no calibration data used
        "num_experts": num_experts,
        "num_layers": num_layers,
        "top_k_experts": 8,
        "per_layer": {},
    }

    print("\n=== Expert Ranking by per_expert_scale ===")
    for layer_idx in range(num_layers):
        scales = layer_scales.get(layer_idx, np.ones(num_experts))
        proj_norms = layer_proj_norms.get(layer_idx, np.ones(num_experts))

        # Combined score: scale * norm (both contribute to routing importance)
        scores = scales * proj_norms

        # Normalize
        if scores.sum() > 0:
            norm_scores = scores / scores.sum()
        else:
            norm_scores = scores

        ranked = np.argsort(scores)[::-1].tolist()

        # Print top and bottom for this layer
        top5 = ranked[:5]
        bottom5 = ranked[-5:]
        print(f"  Layer {layer_idx:2d}: top5={top5}, bottom5={bottom5}, "
              f"scale_range=[{scales.min():.4f}, {scales.max():.4f}]")

        stats["per_layer"][str(layer_idx)] = {
            "per_expert_scale": scales.tolist(),
            "proj_norms": proj_norms.tolist(),
            "scores": scores.tolist(),
            "normalized_scores": norm_scores.tolist(),
            "ranked_experts": ranked,
            # Fill in compatibility fields expected by 02_prune_experts.py
            "counts": [0] * num_experts,
            "weight_sums": scores.tolist(),
            "total_tokens": 0,
        }

    # Save
    stats_path = output_dir / "activation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved ranking to {stats_path}")

    # Print summary of which experts survive at different keep counts
    print("\n=== Summary: Experts to Keep ===")
    for keep_count in [16, 24, 32, 48, 64]:
        all_kept = set(range(num_experts))  # start with all
        layer_agreement = np.zeros(num_experts)

        for layer_idx in range(num_layers):
            ranked = stats["per_layer"][str(layer_idx)]["ranked_experts"]
            top_k = set(ranked[:keep_count])
            for e in top_k:
                layer_agreement[e] += 1

        # Experts kept by ALL layers
        unanimous = np.sum(layer_agreement == num_layers)
        # Experts kept by MOST layers (>80%)
        majority = np.sum(layer_agreement >= num_layers * 0.8)

        print(f"  keep={keep_count:3d}: {unanimous:3d} experts kept by ALL layers, "
              f"{majority:3d} by >80% layers, "
              f"avg layer overlap={layer_agreement.mean()/num_layers*100:.1f}%")

    # Print cross-layer expert stability
    print("\n=== Most Consistently Important Experts (top 20 across all layers) ===")
    layer_agreement = np.zeros(num_experts)
    for layer_idx in range(num_layers):
        ranked = stats["per_layer"][str(layer_idx)]["ranked_experts"]
        for rank, expert in enumerate(ranked[:32]):  # top 32
            layer_agreement[expert] += (32 - rank) / 32  # weighted by rank

    top_overall = np.argsort(layer_agreement)[::-1][:20]
    for i, expert in enumerate(top_overall):
        print(f"  Expert {expert:3d}: importance score = {layer_agreement[expert]:.2f} "
              f"(kept in {int(np.sum([expert in stats['per_layer'][str(l)]['ranked_experts'][:32] for l in range(num_layers)]))}/{num_layers} layers' top-32)")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick expert ranking using model's own per_expert_scale")
    parser.add_argument("--model-path", type=str, default="/home/kitt/gemma4-prune/bf16",
                        help="Path to the BF16 model")
    parser.add_argument("--output-dir", type=str, default="/home/kitt/gemma4-prune/calibration",
                        help="Directory to save ranking results")
    args = parser.parse_args()

    quick_rank(args.model_path, args.output_dir)