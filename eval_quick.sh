#!/bin/bash
# Quick quality evaluation script for pruned model
# Tests multiple prompts at different quant levels

set -e

MODEL_DIR="/home/kitt/models/gguf"
BUILD_BIN="/home/kitt/llama.cpp/build/bin"
export LD_LIBRARY_PATH="$BUILD_BIN:$LD_LIBRARY_PATH"

PROMPTS=(
    "What is 2+2? Answer briefly."
    "Explain what gravity is in one sentence."
    "Write a Python function that reverses a string."
    "What is the capital of France?"
    "Count from 1 to 5."
)

# Test Q4_K_M
echo "=== Testing Q4_K_M ==="
for prompt in "${PROMPTS[@]}"; do
    echo "PROMPT: $prompt"
    "$BUILD_BIN/llama-cli" \
        -m "$MODEL_DIR/gemma4-26b-a4b-e48-Q4_K_M.gguf" \
        -ngl 99 -c 2048 --temp 0.3 --reasoning off \
        -p "$prompt" --n-predict 80 2>&1 | \
        grep -v "ggml_\|Device\|Loading\|▄\|██\|build\|model:\|modali\|avail\|/exit\|/regen\|/clear\|/read\|/glob\|memory_\|CUDA\|Host\|Exit\|t/s\|^$" | \
        head -5
    echo "---"
done