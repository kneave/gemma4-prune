#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""
Expert pruning script for Gemma 4 26B-A4B MoE.

Reads calibration stats and slices out unused experts from the model.
Works directly with safetensors files for memory efficiency.

Gemma 4 MoE structure per layer:
  - router: Gemma4TextRouter with proj.weight [128, 2816], scale [2816], per_expert_scale [128]
  - experts: Gemma4TextExperts with gate_up_proj [128, 1408, 2816] and down_proj [128, 2816, 704]
  - mlp: shared expert (not pruned)

Usage:
    python 02_prune_experts.py --model-path /home/kitt/gemma4-prune/bf16 \
                               --calibration /home/kitt/gemma4-prune/calibration/activation_stats.json \
                               --output-dir /home/kitt/gemma4-prune/pruned \
                               --keep-experts 32
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch


def load_calibration_stats(calibration_path):
    with open(calibration_path, "r") as f:
        return json.load(f)


def compute_experts_to_keep(stats, keep_count, strategy="top_k"):
    """Determine which experts to keep for each layer."""
    keep_map = {}
    for layer_idx_str, layer_data in stats["per_layer"].items():
        layer_idx = int(layer_idx_str)
        ranked = layer_data["ranked_experts"]

        if strategy == "top_k":
            keep_indices = sorted(ranked[:keep_count])
        elif strategy == "uniform":
            all_indices = list(range(len(ranked)))
            step = len(all_indices) / keep_count
            keep_indices = sorted([all_indices[int(i * step)] for i in range(keep_count)])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        keep_map[layer_idx] = keep_indices
    return keep_map


def prune_model_safetensors(model_path, output_dir, keep_map, num_experts_original=128):
    """
    Prune experts at the safetensors level.
    
    Gemma 4 MoE weight naming pattern:
      model.language_model.layers.{L}.router.proj.weight    -> [128, 2816]
      model.language_model.layers.{L}.router.scale          -> [2816]
      model.language_model.layers.{L}.router.per_expert_scale -> [128]
      model.language_model.layers.{L}.experts.gate_up_proj  -> [128, 1408, 2816]
      model.language_model.layers.{L}.experts.down_proj      -> [128, 2816, 704]
      model.language_model.layers.{L}.mlp.*                  -> shared (not pruned)
    """
    from safetensors import safe_open
    from safetensors.torch import save_file
    import re

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_path)
    safetensor_files = sorted(model_dir.glob("model-*.safetensors"))
    if not safetensor_files:
        safetensor_files = [model_dir / "model.safetensors"]

    print(f"Found {len(safetensor_files)} safetensors files")

    # Load and update config
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    old_num_experts = text_config.get("num_experts", num_experts_original)
    new_num_experts = len(keep_map[0])

    text_config["num_experts"] = new_num_experts
    if "num_local_experts" in text_config:
        text_config["num_local_experts"] = new_num_experts

    top_k = text_config.get("num_experts_per_tok", text_config.get("top_k_experts", 8))
    if top_k > new_num_experts:
        print(f"  Reducing top_k from {top_k} to {new_num_experts}")
        text_config["num_experts_per_tok"] = new_num_experts
        if "top_k_experts" in text_config:
            text_config["top_k_experts"] = new_num_experts

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config: num_experts {old_num_experts} -> {new_num_experts}")

    # Also check for model.safetensors.index.json (shard index)
    index_file = model_dir / "model.safetensors.index.json"
    has_index = index_file.exists()

    # Process each safetensors file
    total_params_before = 0
    total_params_after = 0
    all_new_tensor_sizes = {}  # tensor_name -> (numel * element_size)

    for st_file in safetensor_files:
        print(f"  Processing {st_file.name}...")

        new_tensors = {}
        with safe_open(str(st_file), framework="pt") as f:
            keys = f.keys()
            for key in keys:
                tensor = f.get_tensor(key)
                total_params_before += tensor.numel()

                # Check for MoE weights that need pruning
                pruned = False

                # Pattern: model.language_model.layers.{L}.router.proj.weight
                # Shape: [128, 2816] -> keep rows for kept experts
                router_proj_match = re.match(
                    r'model\.language_model\.layers\.(\d+)\.router\.proj\.weight',
                    key
                )
                if router_proj_match:
                    layer_idx = int(router_proj_match.group(1))
                    keep = keep_map.get(layer_idx, keep_map[0])
                    new_tensors[key] = tensor[keep, :]
                    pruned = True

                # Pattern: model.language_model.layers.{L}.router.per_expert_scale
                # Shape: [128] -> keep elements for kept experts
                expert_scale_match = re.match(
                    r'model\.language_model\.layers\.(\d+)\.router\.per_expert_scale',
                    key
                )
                if expert_scale_match:
                    layer_idx = int(expert_scale_match.group(1))
                    keep = keep_map.get(layer_idx, keep_map[0])
                    new_tensors[key] = tensor[keep]
                    pruned = True

                # Pattern: model.language_model.layers.{L}.experts.gate_up_proj
                # Shape: [128, 1408, 2816] -> slice dim 0 for kept experts
                experts_gate_up_match = re.match(
                    r'model\.language_model\.layers\.(\d+)\.experts\.gate_up_proj',
                    key
                )
                if experts_gate_up_match:
                    layer_idx = int(experts_gate_up_match.group(1))
                    keep = keep_map.get(layer_idx, keep_map[0])
                    new_tensors[key] = tensor[keep, :, :]
                    pruned = True

                # Pattern: model.language_model.layers.{L}.experts.down_proj
                # Shape: [128, 2816, 704] -> slice dim 0 for kept experts
                experts_down_match = re.match(
                    r'model\.language_model\.layers\.(\d+)\.experts\.down_proj',
                    key
                )
                if experts_down_match:
                    layer_idx = int(experts_down_match.group(1))
                    keep = keep_map.get(layer_idx, keep_map[0])
                    new_tensors[key] = tensor[keep, :, :]
                    pruned = True

                if not pruned:
                    new_tensors[key] = tensor

                total_params_after += new_tensors[key].numel()
                all_new_tensor_sizes[key] = new_tensors[key].numel() * new_tensors[key].element_size()

        # Save pruned safetensors
        output_file = output_dir / st_file.name
        save_file(new_tensors, str(output_file))
        print(f"    Saved {len(new_tensors)} tensors to {output_file.name}")

    # Update shard index if it existed
    if has_index:
        with open(index_file) as f:
            index = json.load(f)

        new_index = {"metadata": index.get("metadata", {})}
        new_weight_map = {}

        # Update weight map — file assignments stay the same, but we need to
        # note which tensors were pruned for shape verification
        for weight_name, shard_file in index.get("weight_map", {}).items():
            new_weight_map[weight_name] = shard_file

        new_index["weight_map"] = new_weight_map

        # Update metadata with new total size (sum across ALL saved shards)
        if "metadata" in index:
            total_size = sum(all_new_tensor_sizes.values())
            new_index["metadata"]["total_size"] = total_size

        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)

    # Copy non-weight files
    for f in model_dir.iterdir():
        if f.name.endswith((".json", ".model", ".txt", ".jinja")) and f.name != "config.json":
            shutil.copy2(f, output_dir / f.name)

    # Save keep map
    keep_info = {
        "original_num_experts": old_num_experts,
        "new_num_experts": new_num_experts,
        "keep_indices_per_layer": {str(k): v for k, v in keep_map.items()},
        "top_k_experts": text_config.get("num_experts_per_tok", text_config.get("top_k_experts", 8)),
        "total_params_before": total_params_before,
        "total_params_after": total_params_after,
        "reduction_pct": round((1 - total_params_after / total_params_before) * 100, 1),
    }
    with open(output_dir / "pruning_info.json", "w") as f:
        json.dump(keep_info, f, indent=2)

    print(f"\nPruning complete!")
    print(f"  Experts: {old_num_experts} -> {new_num_experts} per layer")
    print(f"  Params: {total_params_before:,} -> {total_params_after:,}")
    print(f"  Reduction: {keep_info['reduction_pct']}%")
    print(f"  Output: {output_dir}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune experts from Gemma 4 MoE model")
    parser.add_argument("--model-path", type=str, default="/home/kitt/gemma4-prune/bf16",
                        help="Path to the BF16 model")
    parser.add_argument("--calibration", type=str,
                        default="/home/kitt/gemma4-prune/calibration/activation_stats.json",
                        help="Path to calibration stats JSON")
    parser.add_argument("--output-dir", type=str, default="/home/kitt/gemma4-prune/pruned",
                        help="Output directory for pruned model")
    parser.add_argument("--keep-experts", type=int, default=32,
                        help="Number of experts to keep per layer (default: 32)")
    parser.add_argument("--strategy", type=str, default="top_k",
                        choices=["top_k", "uniform"],
                        help="Expert selection strategy")
    args = parser.parse_args()

    stats = load_calibration_stats(args.calibration)
    keep_map = compute_experts_to_keep(stats, args.keep_experts, strategy=args.strategy)

    prune_model_safetensors(
        args.model_path, args.output_dir, keep_map,
        num_experts_original=stats.get("num_experts", 128)
    )