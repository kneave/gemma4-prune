#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""
Calibration script for Gemma 4 26B-A4B MoE expert pruning.

Features:
  - Checkpoints every 25 samples (auto-resume on restart)
  - Saves progress on Ctrl-C (SIGINT/SIGTERM)
  - Timestamped progress logging with ETA

Usage:
    python 01_calibrate.py --model-path /home/kitt/gemma4-prune/bf16 \
                           --output-dir /home/kitt/gemma4-prune/calibration \
                           --num-samples 6000 \
                           --device auto

    # Auto-resumes from checkpoint:
    python 01_calibrate.py --model-path /home/kitt/gemma4-prune/bf16 \
                           --output-dir /home/kitt/gemma4-prune/calibration

Output:
    calibration/checkpoint.npz   - incremental checkpoint (binary, fast)
    calibration/activation_stats.json - final stats (on completion)
"""

import argparse
import os
# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Use all CPU cores for inference
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import json
import signal
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import time

import torch
torch.set_num_threads(4)  # Match physical cores (4 cores, 8 threads)
import numpy as np

CHECKPOINT_EVERY = 25  # save progress every N samples


def load_calibration_data(num_samples=6000, seed=42):
    """Prepare calibration dataset: 50% English prose, 25% code, 25% math."""
    np.random.seed(seed)
    samples = []

    english_texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used in typography.",
        "In recent years, artificial intelligence has made significant strides in natural language processing.",
        "The committee decided to postpone the meeting until next Thursday due to scheduling conflicts.",
        "Climate change remains one of the most pressing challenges facing the global community today.",
        "She walked through the park, enjoying the autumn leaves and the crisp morning air.",
        "The economic impact of the pandemic has been felt across all sectors of industry.",
        "Education systems around the world are adapting to new technologies and teaching methodologies.",
        "The restaurant on the corner serves excellent pasta and has a beautiful outdoor seating area.",
        "Research suggests that regular exercise can significantly improve mental health outcomes.",
        "The library's collection includes over fifty thousand volumes covering a wide range of subjects.",
        "He couldn't believe his eyes when he saw the sunset from the mountain peak.",
        "The new highway connects the two cities and reduces travel time by almost half.",
        "Democracy depends on an informed citizenry and transparent governmental processes.",
        "The orchestra performed a stunning rendition of Beethoven's ninth symphony.",
        "Sustainable agriculture practices are becoming increasingly important for food security.",
        "The architect designed a building that harmonizes perfectly with its natural surroundings.",
        "Public transportation systems play a vital role in reducing urban carbon emissions.",
        "The documentary explored the history of civil rights movements across different nations.",
        "Advances in medical imaging have revolutionized diagnostic capabilities in hospitals.",
        "The novel explores themes of identity, belonging, and the immigrant experience.",
    ]

    code_texts = [
        'def fibonacci(n):\n    """Calculate the nth Fibonacci number."""\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)',
        'class LinkedList:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n    \n    def reverse(self):\n        prev = None\n        curr = self\n        while curr:\n            nxt = curr.next\n            curr.next = prev\n            prev = curr\n            curr = nxt\n        return prev',
        'SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at >= "2024-01-01"\nGROUP BY u.name\nHAVING order_count > 5\nORDER BY order_count DESC;',
        'import React, { useState, useEffect } from "react";\n\nexport function DataGrid({ endpoint }) {\n  const [data, setData] = useState([]);\n  const [loading, setLoading] = useState(true);\n  \n  useEffect(() => {\n    fetch(endpoint)\n      .then(res => res.json())\n      .then(json => { setData(json); setLoading(false); });\n  }, [endpoint]);\n  \n  if (loading) return <Spinner />;\n  return <table>{data.map(row => <Row key={row.id} {...row} />)}</table>;\n}',
        'func QuickSort(arr []int) []int {\n    if len(arr) <= 1 {\n        return arr\n    }\n    pivot := arr[0]\n    var less, greater []int\n    for _, v := range arr[1:] {\n        if v <= pivot {\n            less = append(less, v)\n        } else {\n            greater = append(greater, v)\n        }\n    }\n    result := append(QuickSort(less), pivot)\n    return append(result, QuickSort(greater)...)\n}',
        '# Docker Compose for microservices\nversion: "3.8"\nservices:\n  api:\n    build: ./api\n    ports: ["8080:8080"]\n    environment:\n      DATABASE_URL: postgres://db:5432/app\n    depends_on: [db, redis]\n  db:\n    image: postgres:15\n    volumes: [pgdata:/var/lib/postgresql/data]\n  redis:\n    image: redis:7-alpine',
        'async function fetchUserData(userId: string): Promise<User> {\n  const cacheKey = `user:${userId}`;\n  const cached = await redis.get(cacheKey);\n  if (cached) return JSON.parse(cached);\n  \n  const response = await fetch(`/api/users/${userId}`);\n  if (!response.ok) throw new Error(`HTTP ${response.status}`);\n  const user = await response.json();\n  await redis.setex(cacheKey, 3600, JSON.stringify(user));\n  return user;\n}',
        'FROM python:3.11-slim AS builder\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nRUN python -m pytest tests/ && python -m build\n\nFROM python:3.11-slim\nCOPY --from=builder /app/dist/*.whl /tmp/\nRUN pip install /tmp/*.whl && rm -rf /tmp/*.whl\nCMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]',
        'impl<T: Ord> BinaryTree<T> {\n    fn insert(&mut self, value: T) {\n        match self {\n            BinaryTree::Leaf => *self = BinaryTree::Node(Box::new(Node {\n                value, left: BinaryTree::Leaf, right: BinaryTree::Leaf,\n            })),\n            BinaryTree::Node(node) => {\n                if value < node.value {\n                    node.left.insert(value);\n                } else {\n                    node.right.insert(value);\n                }\n            }\n        }\n    }\n}',
        'resource "aws_ecs_service" "app" {\n  name            = "${var.project}-${var.env}-service"\n  cluster         = aws_ecs_cluster.main.id\n  task_definition = aws_ecs_task_definition.app.arn\n  desired_count   = var.desired_count\n  launch_type     = "FARGATE"\n  \n  load_balancer {\n    target_group_arn = aws_lb_target_group.app.arn\n    container_name   = "app"\n    container_port   = 8080\n  }\n}',
    ]

    math_texts = [
        "Given f(x) = 3x^2 + 2x - 5, find f'(x) and determine where f is increasing. Solution: f'(x) = 6x + 2, which is positive when x > -1/3.",
        "Prove that the sum of the first n natural numbers equals n(n+1)/2. Base case: n=1, 1 = 1(2)/2 = 1. Inductive step: assume true for k, then sum to k+1 = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.",
        "Calculate the determinant of the matrix [[2,3,1],[4,1,2],[3,2,5]]. Expanding along the first row: 2(5-4) - 3(20-6) + 1(8-3) = 2 - 42 + 5 = -35.",
        "If P(A) = 0.6, P(B) = 0.4, and P(AUB) = 0.8, find P(A∩B). By inclusion-exclusion: P(A∩B) = P(A) + P(B) - P(A∪B) = 0.6 + 0.4 - 0.8 = 0.2.",
        "Solve the differential equation dy/dx = 2xy. This is separable: dy/y = 2x dx, so ln|y| = x^2 + C, giving y = Ce^(x^2).",
        "Find the eigenvalues of [[4,1],[2,3]]. The characteristic polynomial is (4-λ)(3-λ) - 2 = λ^2 - 7λ + 10 = (λ-5)(λ-2). Eigenvalues: 5 and 2.",
        "Compute the line integral ∫_C (x+y)ds where C is the unit circle. Parameterize: x=cos(t), y=sin(t), ds=dt. The integral becomes ∫₀²π (cos(t)+sin(t))dt = 0.",
        "Using the binomial theorem, expand (x+1)^6. The coefficients are C(6,0), C(6,1),...,C(6,6) = 1,6,15,20,15,6,1. So (x+1)^6 = x^6 + 6x^5 + 15x^4 + 20x^3 + 15x^2 + 6x + 1.",
        "Find the maximum of f(x) = -x^3 + 6x^2 + 15x + 1 on [-2,4]. f'(x) = -3x^2 + 12x + 15 = -3(x^2-4x-5) = -3(x-5)(x+1). Critical points: x=-1, x=5. f(-1) = 7, f(5) = -125+150+75+1 = 101. Check endpoints: f(-2) = 3, f(4) = -64+96+60+1 = 93. Maximum is 101 at x=5 if in domain, else 93 at x=4.",
        "Show that √2 is irrational. Assume √2 = p/q in lowest terms. Then 2q² = p², so p² is even, so p is even. Let p = 2k. Then 2q² = 4k², so q² = 2k², so q² is even, so q is even. But then p/q isn't in lowest terms. Contradiction.",
    ]

    n_english = int(num_samples * 0.50)
    n_code = int(num_samples * 0.25)
    n_math = num_samples - n_english - n_code

    for i in range(n_english):
        samples.append({"text": english_texts[i % len(english_texts)], "category": "english"})
    for i in range(n_code):
        samples.append({"text": code_texts[i % len(code_texts)], "category": "code"})
    for i in range(n_math):
        samples.append({"text": math_texts[i % len(math_texts)], "category": "math"})

    np.random.shuffle(samples)
    return samples


def save_checkpoint(activation_stats, num_experts, num_layers, processed,
                    output_dir, samples_total):
    """Save checkpoint to .npz (fast binary format)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "processed": processed,
        "samples_total": samples_total,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "timestamp": datetime.now().isoformat(),
    }
    for layer_idx in range(num_layers):
        data[f"count_{layer_idx}"] = activation_stats[layer_idx]["count"]
        data[f"weight_{layer_idx}"] = activation_stats[layer_idx]["weight_sum"]
        data[f"tokens_{layer_idx}"] = np.array([activation_stats[layer_idx]["total_tokens"]])

    np.savez_compressed(output_dir / "checkpoint.npz", **data)
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Checkpoint saved at {processed}/{samples_total}")


def load_checkpoint(output_dir, num_experts, num_layers):
    """Load checkpoint if it exists. Returns (activation_stats, processed_count) or None."""
    checkpoint_path = Path(output_dir) / "checkpoint.npz"
    if not checkpoint_path.exists():
        return None

    data = np.load(checkpoint_path, allow_pickle=True)
    processed = int(data["processed"])
    ckpt_layers = int(data["num_layers"])
    ckpt_experts = int(data["num_experts"])

    if ckpt_layers != num_layers or ckpt_experts != num_experts:
        print(f"  Checkpoint incompatible ({ckpt_layers}L/{ckpt_experts}E vs {num_layers}L/{num_experts}E), starting fresh")
        return None

    activation_stats = {}
    for layer_idx in range(num_layers):
        activation_stats[layer_idx] = {
            "count": data[f"count_{layer_idx}"],
            "weight_sum": data[f"weight_{layer_idx}"],
            "total_tokens": int(data[f"tokens_{layer_idx}"][0]),
        }

    print(f"  Resumed from checkpoint: {processed} samples already processed")
    return activation_stats, processed


def calibrate(model_path, output_dir, num_samples=6000, device="auto",
              max_seq_length=512, load_in_4bit=False):
    """Run calibration: pass data through model and log router activations.

    Note: 4-bit loading is not supported for Gemma 4 multimodal models
    due to accelerate/transformers device_map conflicts. Use BF16 with
    device_map="auto" (CPU offload) instead — this is the default.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")

    if load_in_4bit:
        print("  WARNING: 4-bit loading not supported for Gemma 4 multimodal model.")
        print("  Falling back to BF16 with device_map='auto' (CPU offload).")

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map="cpu", low_cpu_mem_usage=True,
        )
    else:
        # Use max_memory to ensure accelerate stacks GPU layers contiguously,
        # minimising GPU↔CPU crossings. 22GiB leaves room for KV cache.
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "22GiB", "cpu": "120GiB"},
            low_cpu_mem_usage=True,
        )

    # Navigate to the text model inside the multimodal wrapper
    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        text_model = model.model.language_model
        print("Found text model at model.model.language_model")
    elif hasattr(model, 'text_model'):
        text_model = model.text_model
        print("Found text model at model.text_model")
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        text_model = model.model
        print("Found text model at model.model")
    else:
        text_model = model
        print("Using model directly as text model")

    # Config
    if hasattr(model.config, 'text_config'):
        config = model.config.text_config
    else:
        config = model.config

    num_experts = getattr(config, 'num_experts', 128)
    num_layers = getattr(config, 'num_hidden_layers', 30)
    print(f"Model has {num_layers} layers, {num_experts} experts per layer")

    # Load tokenizer
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to resume from checkpoint
    ckpt = load_checkpoint(output_dir, num_experts, num_layers)

    if ckpt:
        activation_stats, processed_count = ckpt
        print(f"  Resuming from sample {processed_count}")
    else:
        # Initialize fresh activation tracking
        activation_stats = {}
        for layer_idx in range(num_layers):
            activation_stats[layer_idx] = {
                "count": np.zeros(num_experts, dtype=np.int64),
                "weight_sum": np.zeros(num_experts, dtype=np.float64),
                "total_tokens": 0,
            }
        processed_count = 0

    # Hook into routers
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            with torch.no_grad():
                logits = output[0] if isinstance(output, tuple) else output
                logits = logits.float()

                top_k = min(8, logits.shape[-1])
                top_k_indices = torch.topk(logits, top_k, dim=-1).indices

                for expert_idx in top_k_indices.cpu().numpy().flatten():
                    activation_stats[layer_idx]["count"][expert_idx] += 1

                probs = torch.softmax(logits, dim=-1)
                for expert_idx in range(logits.shape[-1]):
                    activation_stats[layer_idx]["weight_sum"][expert_idx] += probs[:, expert_idx].sum().item()

                activation_stats[layer_idx]["total_tokens"] += logits.shape[0]

        return hook_fn

    print("Registering router hooks...")
    for layer_idx, layer in enumerate(text_model.layers):
        if hasattr(layer, 'router'):
            hook = layer.router.register_forward_hook(make_hook(layer_idx))
            hooks.append(hook)
        else:
            print(f"  WARNING: Layer {layer_idx} has no 'router' attribute, skipping")

    print(f"Registered {len(hooks)} hooks")

    # Load calibration data
    print(f"Preparing {num_samples} calibration samples...")
    samples = load_calibration_data(num_samples)

    # Skip already-processed samples
    if processed_count > 0:
        samples = samples[processed_count:]
        print(f"  Skipping first {processed_count} samples, {len(samples)} remaining")

    # Signal handler — save checkpoint on Ctrl-C
    def handle_signal(signum, frame):
        print(f"\n  Signal {signum} received, saving checkpoint...")
        save_checkpoint(activation_stats, num_experts, num_layers,
                        processed_count, output_dir, num_samples)
        print("  Checkpoint saved. Exiting gracefully.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # ============ SENTINEL: quick sanity check then exit ============
    # Use --dry-run to verify model loads + hooks work without running
    if os.environ.get("CALIBRATE_DRY_RUN") == "1":
        print("\n=== DRY RUN: Model loaded, hooks registered, exiting ===")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Text model type: {type(text_model).__name__}")
        print(f"  Layers: {num_layers}, Experts: {num_experts}")
        print(f"  Hooks registered: {len(hooks)}")
        print(f"  Device: {next(model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, "
                  f"{torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
        print(f"  Checkpoint exists: {(output_dir / 'checkpoint.npz').exists()}")
        print("  Set CALIBRATE_DRY_RUN=0 or unset to run calibration.")
        return None

    # ============ END SENTINEL ============

    # Run calibration
    model.eval()
    print(f"Running calibration: {processed_count} already done, {len(samples)} remaining...")
    start_time = time.time()
    total_processed = processed_count

    with torch.no_grad():
        for i, sample in enumerate(samples):
            text = sample["text"]

            messages = [{"role": "user", "content": text}]
            try:
                input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                input_text = f"<user>\n{text}\n<model>\n"

            inputs = tokenizer(input_text, return_tensors="pt", max_length=max_seq_length, truncation=True)

            if device != "cpu":
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            try:
                outputs = model(**inputs)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on sample {processed_count + i}, skipping...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM on sample {processed_count + i}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            total_processed += 1
            processed_count += 1

            # Progress every 10 samples, checkpoint every 25
            if total_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = (total_processed - processed_count + (total_processed - (processed_count - 1) + 1) ) / elapsed if elapsed > 0 else 0
                # Simpler: use total_processed and elapsed directly
                rate = total_processed / elapsed if elapsed > 0 else 0
                remaining_samples = num_samples - total_processed
                eta_secs = remaining_samples / rate if rate > 0 else 0
                eta_str = f"{int(eta_secs // 3600)}h{int((eta_secs % 3600) // 60)}m" if eta_secs > 60 else f"{int(eta_secs)}s"
                now = datetime.now().strftime('%H:%M:%S')
                print(f"  [{now}] {total_processed}/{num_samples} ({total_processed/num_samples*100:.1f}%) — "
                      f"{rate:.1f} smp/s — ETA {eta_str}")

            if total_processed % CHECKPOINT_EVERY == 0:
                save_checkpoint(activation_stats, num_experts, num_layers,
                                total_processed, output_dir, num_samples)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"Calibration complete. Processed {total_processed} total samples.")

    # Final save
    save_checkpoint(activation_stats, num_experts, num_layers,
                    total_processed, output_dir, num_samples)

    # Compute final statistics -> JSON
    stats = {
        "model_path": str(model_path),
        "num_samples": total_processed,
        "num_experts": num_experts,
        "num_layers": num_layers,
        "top_k_experts": 8,
        "per_layer": {},
    }

    for layer_idx in range(num_layers):
        layer_stats = activation_stats[layer_idx]
        counts = layer_stats["count"]
        weight_sums = layer_stats["weight_sum"]
        total_tokens = layer_stats["total_tokens"]

        scores = weight_sums * counts
        if scores.sum() > 0:
            norm_scores = scores / scores.sum()
        else:
            norm_scores = scores

        ranked = np.argsort(scores)[::-1]

        stats["per_layer"][str(layer_idx)] = {
            "counts": counts.tolist(),
            "weight_sums": weight_sums.tolist(),
            "total_tokens": total_tokens,
            "scores": scores.tolist(),
            "normalized_scores": norm_scores.tolist(),
            "ranked_experts": ranked.tolist(),
        }

    stats_path = output_dir / "activation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved activation stats to {stats_path}")

    # Print summary
    print("\n=== Expert Activation Summary ===")
    for layer_idx in range(num_layers):
        counts = activation_stats[layer_idx]["count"]
        active = np.count_nonzero(counts)
        top5 = np.argsort(counts)[::-1][:5]
        zero = np.sum(counts == 0)
        print(f"  Layer {layer_idx:2d}: {active:3d} experts activated, "
              f"{zero:3d} never used, top 5: {top5.tolist()}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Gemma 4 MoE for expert pruning")
    parser.add_argument("--model-path", type=str, default="/home/kitt/gemma4-prune/bf16",
                        help="Path to the BF16 model")
    parser.add_argument("--output-dir", type=str, default="/home/kitt/gemma4-prune/calibration",
                        help="Directory to save calibration results")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of calibration samples (50% English, 25% code, 25% math)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cpu or auto (GPU with CPU offload)")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length for calibration samples")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (NF4) for faster calibration")
    args = parser.parse_args()

    calibrate(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )