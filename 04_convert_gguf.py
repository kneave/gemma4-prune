#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""
Convert pruned Gemma 4 model to GGUF format.

Two-step process:
1. Convert safetensors → GGUF F16
2. Quantize F16 → Q4_K_M using llama-quantize

Usage:
    python 04_convert_gguf.py --model-path /home/kitt/gemma4-prune/pruned-48 \
                               --output-dir /home/kitt/gemma4-prune \
                               --quantize Q4_K_M

If disk space is tight, use --delete-intermediate to remove the F16 GGUF
after quantization completes.
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

# GGUF constants
GGUF_MAGIC = 0x46475547  # "GGUF"
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 5
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGML tensor types
GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ2_XXS": 16,
    "IQ2_XS": 17,
    "IQ2_S": 18,
    "IQ3_XXS": 19,
    "IQ1_S": 20,
    "IQ4_NL": 21,
    "IQ3_S": 22,
    "IQ2_M": 23,
    "IQ4_XS": 24,
    "IQ1_M": 25,
    "BF16": 26,
}


def write_gguf_string(f, s):
    """Write a GGUF-format string."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)


def write_gguf_value(f, value, vtype):
    """Write a GGUF key-value pair value."""
    f.write(struct.pack("<I", vtype))
    if vtype == GGUF_TYPE_STRING:
        write_gguf_string(f, value)
    elif vtype == GGUF_TYPE_UINT32:
        f.write(struct.pack("<I", value))
    elif vtype == GGUF_TYPE_FLOAT32:
        f.write(struct.pack("<f", value))
    elif vtype == GGUF_TYPE_ARRAY:
        arr_type = value[0]
        arr = value[1]
        f.write(struct.pack("<I", arr_type))
        f.write(struct.pack("<Q", len(arr)))
        for item in arr:
            if arr_type == GGUF_TYPE_STRING:
                write_gguf_string(f, item)
            elif arr_type == GGUF_TYPE_UINT32:
                f.write(struct.pack("<I", item))
            elif arr_type == GGUF_TYPE_FLOAT32:
                f.write(struct.pack("<f", item))
    else:
        raise ValueError(f"Unsupported GGUF value type: {vtype}")


def convert_to_gguf(model_path, output_path):
    """Convert HuggingFace safetensors model to GGUF F16 format."""
    model_path = Path(model_path)
    output_path = Path(output_path)

    print(f"Converting {model_path} → {output_path}")

    # Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer_path = model_path / "tokenizer.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"

    # Collect all safetensor files
    safetensor_files = sorted(model_path.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files in {model_path}")

    print(f"Found {len(safetensor_files)} safetensor files")

    # Read all tensor metadata first
    tensor_infos = []  # (name, shape, dtype)
    total_params = 0

    for st_file in safetensor_files:
        with safe_open(str(st_file), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Skip vision tower tensors - GGUF doesn't support them well
                if "vision" in key or "image" in key.lower():
                    print(f"  Skipping vision tensor: {key}")
                    continue
                tensor_infos.append((key, tensor.shape, tensor.dtype))
                total_params += tensor.numel()

    print(f"Total tensors to convert: {len(tensor_infos)}")
    print(f"Total parameters: {total_params:,}")

    # Build GGUF metadata
    # Map HuggingFace config to GGUF architecture params
    arch = "gemma4"  # Gemma 4 uses gemma4 architecture in llama.cpp

    metadata = {}

    # General metadata
    metadata["general.architecture"] = (arch, GGUF_TYPE_STRING)
    metadata["general.name"] = (f"gemma-4-26b-a4b-pruned-e{config.get('num_experts', 48)}", GGUF_TYPE_STRING)

    # Model dimensions from config
    lang_config = config
    # Navigate past model.language_model if needed
    if "model_type" in config:
        lang_config = config

    hidden_size = config.get("hidden_size", 2816)
    intermediate_size = config.get("intermediate_size", 2816)
    num_attention_heads = config.get("num_attention_heads", 16)
    num_hidden_layers = config.get("num_hidden_layers", 30)
    num_key_value_heads = config.get("num_key_value_heads", 8)
    num_experts = config.get("num_experts", 48)
    num_experts_per_tok = config.get("num_experts_per_tok", min(8, num_experts))
    vocab_size = config.get("vocab_size", 262144)
    head_dim = hidden_size // num_attention_heads
    rms_norm_eps = config.get("rms_norm_eps", 1e-6)

    # Gemma-specific
    metadata[f"{arch}.block_count"] = (num_hidden_layers, GGUF_TYPE_UINT32)
    metadata[f"{arch}.context_length"] = (config.get("max_position_embeddings", 131072), GGUF_TYPE_UINT32)
    metadata[f"{arch}.embedding_length"] = (hidden_size, GGUF_TYPE_UINT32)
    metadata[f"{arch}.feed_forward_length"] = (intermediate_size, GGUF_TYPE_UINT32)
    metadata[f"{arch}.attention.head_count"] = (num_attention_heads, GGUF_TYPE_UINT32)
    metadata[f"{arch}.attention.head_count_kv"] = (num_key_value_heads, GGUF_TYPE_UINT32)
    metadata[f"{arch}.attention.layer_norm_rms_epsilon"] = (rms_norm_eps, GGUF_TYPE_FLOAT32)
    metadata[f"{arch}.attention.key_length"] = (head_dim, GGUF_TYPE_UINT32)
    metadata[f"{arch}.attention.value_length"] = (head_dim, GGUF_TYPE_UINT32)
    metadata[f"{arch}.expert_count"] = (num_experts, GGUF_TYPE_UINT32)
    metadata[f"{arch}.expert_used_count"] = (num_experts_per_tok, GGUF_TYPE_UINT32)
    metadata[f"{arch}.vocab_size"] = (vocab_size, GGUF_TYPE_UINT32)

    # Layer scalars
    if "layer_scalar_type" in config:
        metadata[f"{arch}.layer_scalar_type"] = (GGML_TYPES.get(config["layer_scalar_type"], 0), GGUF_TYPE_UINT32)

    # Write GGUF file
    print(f"Writing F16 GGUF to {output_path}...")

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_infos)))  # n_tensors
        f.write(struct.pack("<Q", len(metadata)))  # n_kv

        # Write metadata
        for key, (value, vtype) in metadata.items():
            write_gguf_string(f, key)
            write_gguf_value(f, value, vtype)

        # Write tensor info (name, n_dims, dims, type, offset)
        # First pass: calculate offsets
        data_offset = f.tell() + len(tensor_infos) * (8 + 4 + 255 * 4 + 4 + 8)  # Approximate
        # Align to 32 bytes
        padding = (32 - (data_offset % 32)) % 32

        # More precise: write tensor infos first, then data
        # Calculate exact sizes
        tensor_data_offsets = []
        current_offset = 0

        # We need to write tensor infos first, then pad, then write tensor data
        # Save position after header + metadata
        tensor_info_start = f.tell()

        # Placeholder - we'll write actual tensor info after calculating sizes
        # Use a two-pass approach

    # Two-pass: first calculate all sizes and positions, then write
    with open(output_path, "wb") as f:
        # --- Header ---
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_infos)))  # n_tensors
        f.write(struct.pack("<Q", len(metadata)))  # n_kv

        # --- Metadata ---
        for key, (value, vtype) in metadata.items():
            write_gguf_string(f, key)
            write_gguf_value(f, value, vtype)

        # --- Tensor Info ---
        # For each tensor: name(string) + n_dims(uint32) + dims(n_dims * uint64) + dtype(uint32) + offset(uint64)
        offsets = []
        for name, shape, dtype in tensor_infos:
            write_gguf_string(f, name)
            n_dims = len(shape)
            f.write(struct.pack("<I", n_dims))
            for dim in reversed(shape):  # GGUF uses reversed dims
                f.write(struct.pack("<Q", dim))

            # Type
            if dtype == torch.bfloat16:
                ggml_type = GGML_TYPES["BF16"]
            elif dtype == torch.float16:
                ggml_type = GGML_TYPES["F16"]
            elif dtype == torch.float32:
                ggml_type = GGML_TYPES["F32"]
            else:
                ggml_type = GGML_TYPES["F16"]  # Default
            f.write(struct.pack("<I", ggml_type))
            # Offset placeholder - will be filled later
            offsets.append(f.tell())
            f.write(struct.pack("<Q", 0))  # placeholder offset

        # Align data start to 32 bytes
        pos = f.tell()
        padding = (32 - (pos % 32)) % 32
        f.write(b"\x00" * padding)

        # --- Tensor Data ---
        data_start = f.tell()

        # Update offsets
        for i, (name, shape, dtype) in enumerate(tensor_infos):
            offset_pos = offsets[i]
            current_pos = f.tell()
            # Calculate actual offset relative to data_start
            actual_offset = current_pos - data_start

            # Write tensor data
            # Read from safetensors
            st_file_idx = 0
            tensor = None
            for st_file in safetensor_files:
                with safe_open(str(st_file), framework="pt") as sf:
                    if name in sf.keys():
                        tensor = sf.get_tensor(name)
                        break

            if tensor is None:
                raise ValueError(f"Tensor {name} not found in any safetensor file")

            # Convert to appropriate format
            if dtype == torch.bfloat16:
                # BF16 -> BF16 (GGUF supports it)
                data = tensor.numpy().view(np.uint16)
            elif dtype == torch.float16:
                data = tensor.numpy().view(np.uint16)
            elif dtype == torch.float32:
                data = tensor.numpy().astype(np.float32)
            else:
                data = tensor.numpy()

            # Write data
            f.write(data.tobytes())

            # Update offset in tensor info
            saved_pos = f.tell()
            f.seek(offset_pos)
            f.write(struct.pack("<Q", actual_offset))
            f.seek(saved_pos)

            if (i + 1) % 50 == 0:
                print(f"  Written {i+1}/{len(tensor_infos)} tensors...")

        print(f"Done! Written {len(tensor_infos)} tensors")
        print(f"Output size: {output_path.stat().st_size / 1e9:.2f} GB")


def quantize_gguf(gguf_path, quant_type, output_path=None):
    """Quantize GGUF F16 to specified quantization using llama-quantize."""
    gguf_path = Path(gguf_path)

    if output_path is None:
        output_path = gguf_path.parent / gguf_path.name.replace("-f16", f"-{quant_type}")

    # Find llama-quantize
    quantize_bin = None
    search_paths = [
        Path.home() / "llama.cpp-turboquant" / "bin" / "llama-quantize",
        Path.home() / "llama.cpp" / "llama-quantize",
        Path.home() / "llama.cpp" / "build" / "bin" / "llama-quantize",
    ]

    for path in search_paths:
        if path.exists():
            quantize_bin = path
            break

    if quantize_bin is None:
        quantize_bin = shutil.which("llama-quantize")

    if quantize_bin is None:
        print(f"llama-quantize not found.")
        print(f"Please quantize manually: llama-quantize {gguf_path} {output_path} {quant_type}")
        return None

    cmd = [str(quantize_bin), str(gguf_path), str(output_path), quant_type]
    print(f"Quantizing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"Quantization failed: {result.stderr}")
        return None

    print(f"Quantized to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e9:.2f} GB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert pruned model to GGUF")
    parser.add_argument("--model-path", type=str, default="/home/kitt/gemma4-prune/pruned-48")
    parser.add_argument("--output-dir", type=str, default="/home/kitt/gemma4-prune")
    parser.add_argument("--quantize", type=str, default="Q4_K_M",
                        choices=["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0"])
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--delete-intermediate", action="store_true",
                        help="Delete F16 GGUF after quantization")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)

    if not args.skip_convert:
        f16_path = output_dir / f"pruned-e48-f16.gguf"
        convert_to_gguf(model_path, f16_path)
    else:
        f16_path = output_dir / f"pruned-e48-f16.gguf"

    if not args.skip_quantize:
        quant_path = quantize_gguf(f16_path, args.quantize)
        if quant_path and args.delete_intermediate:
            print(f"Deleting intermediate F16 GGUF: {f16_path}")
            f16_path.unlink()
    else:
        quant_path = f16_path

    if quant_path:
        print(f"\nFinal output: {quant_path}")
        print(f"  Size: {quant_path.stat().st_size / 1e9:.2f} GB")