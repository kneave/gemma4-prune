#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""
Before/after evaluation script for pruned Gemma 4 MoE model.

Supports evaluation via llama-server (GGUF) or HuggingFace (BF16).
Tests English, code, math, and handoff behavior.

Usage:
    # Using GGUF models via llama-server:
    python 05_evaluate.py --original ~/models/gguf/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
                          --pruned ~/models/gguf/gemma4-english-E32-Q4_K_M.gguf \
                          --output-dir /home/kitt/gemma4-prune/eval

    # Using HuggingFace models:
    python 05_evaluate.py --original-hf /home/kitt/gemma4-prune/bf16 \
                          --pruned-hf /home/kitt/gemma4-prune/pruned \
                          --output-dir /home/kitt/gemma4-prune/eval
"""

import argparse
import json
import time
from pathlib import Path


EVAL_PROMPTS = {
    "english": [
        {"prompt": "Explain the concept of recursion in programming.", "min_words": 30},
        {"prompt": "What are the key principles of object-oriented programming?", "min_words": 40},
        {"prompt": "Summarize the difference between TCP and UDP in networking.", "min_words": 20},
        {"prompt": "I'm feeling overwhelmed with my todo list. Any advice?", "min_words": 30},
        {"prompt": "Explain why the sky is blue in simple terms.", "min_words": 20},
        {"prompt": "What factors should I consider when choosing between renting and buying a home?", "min_words": 40},
    ],
    "code": [
        {"prompt": "Write a Python function that finds the two numbers in a list that add up to a target.", "min_words": 15},
        {"prompt": "Write a Python class implementing a stack with push, pop, and peek operations.", "min_words": 10},
        {"prompt": "Write a bash script that finds all .log files older than 7 days and deletes them.", "min_words": 10},
        {"prompt": "Implement a function to check if a string is a palindrome, ignoring case and non-alphanumeric characters.", "min_words": 10},
        {"prompt": "Write a SQL query to find the top 5 customers by total order value.", "min_words": 10},
    ],
    "math": [
        {"prompt": "Solve for x: 3x + 7 = 22", "min_words": 3},
        {"prompt": "What is the derivative of f(x) = x^3 + 2x^2 - 5x + 1?", "min_words": 5},
        {"prompt": "Calculate the area of a circle with radius 7 cm.", "min_words": 3},
        {"prompt": "If P(A) = 0.6, P(B) = 0.4, and P(A∪B) = 0.8, find P(A∩B).", "min_words": 5},
        {"prompt": "Prove that √2 is irrational by contradiction.", "min_words": 20},
    ],
    "handoff": [
        {"prompt": "Implement a full distributed consensus algorithm from scratch.", "criteria": "Should suggest handoff to cloud model"},
        {"prompt": "Design a novel cryptographic hash function resistant to quantum attacks.", "criteria": "Should acknowledge limitation and suggest escalation"},
        {"prompt": "What's 2 + 2?", "criteria": "Should answer directly, no handoff"},
    ],
}


def evaluate_with_llama_server(gguf_path, prompts, model_name="model", port=8081,
                                llama_cpp_bin=None):
    """Evaluate using llama-server (local inference)."""
    import subprocess
    import requests

    # Find llama-server
    if llama_cpp_bin:
        server_bin = Path(llama_cpp_bin) / "llama-server"
    else:
        server_bin = shutil.which("llama-server")
        if server_bin is None:
            # Check known locations
            for loc in [Path.home() / "llama.cpp-turboquant" / "bin" / "llama-server",
                        Path.home() / "llama.cpp" / "llama-server"]:
                if loc.exists():
                    server_bin = str(loc)
                    break

    if server_bin is None:
        print("ERROR: llama-server not found")
        return None

    print(f"\nStarting llama-server for {model_name}...")
    print(f"  GGUF: {gguf_path}")
    print(f"  Port: {port}")

    server_proc = subprocess.Popen(
        [str(server_bin), "-m", str(gguf_path), "--port", str(port),
         "-ngl", "99", "-c", "2048", "--host", "127.0.0.1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    print("  Waiting for server to start...")
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(1)
    else:
        server_proc.terminate()
        raise RuntimeError("Server failed to start within 60 seconds")

    print("  Server ready!")

    results = []
    try:
        for category, cat_prompts in prompts.items():
            print(f"\n  Evaluating {category} ({len(cat_prompts)} prompts)...")
            for i, item in enumerate(cat_prompts):
                prompt = item["prompt"]
                messages = [{"role": "user", "content": prompt}]

                start = time.time()
                try:
                    r = requests.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions",
                        json={"messages": messages, "max_tokens": 512, "temperature": 0.3},
                        timeout=120,
                    )
                    elapsed = time.time() - start

                    if r.status_code == 200:
                        resp = r.json()
                        response = resp["choices"][0]["message"]["content"]
                        tokens = resp["usage"]["completion_tokens"]
                        tps = tokens / elapsed if elapsed > 0 else 0
                    else:
                        response = f"ERROR: {r.status_code}"
                        tokens = tps = 0
                        elapsed = time.time() - start
                except Exception as e:
                    response = f"ERROR: {e}"
                    tokens = tps = 0
                    elapsed = time.time() - start

                results.append({
                    "category": category, "prompt": prompt, "response": response,
                    "tokens": tokens, "time": round(elapsed, 2), "tps": round(tps, 2),
                    "min_words": item.get("min_words", 0),
                })
                print(f"    [{category}/{i+1}] {tokens} tok, {elapsed:.1f}s, {tps:.1f} tok/s")
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=10)

    return results


def evaluate_with_hf(model_path, prompts, model_name="model"):
    """Evaluate using HuggingFace transformers (BF16 models)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

    print(f"\nLoading {model_name} from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
    )

    try:
        processor = AutoProcessor.from_pretrained(model_path)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    results = []

    with torch.no_grad():
        for category, cat_prompts in prompts.items():
            print(f"\n  Evaluating {category} ({len(cat_prompts)} prompts)...")
            for i, item in enumerate(cat_prompts):
                prompt = item["prompt"]
                messages = [{"role": "user", "content": prompt}]

                try:
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    input_text = f"<user>\n{prompt}\n<model>\n"

                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                start = time.time()
                outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.3,
                                          do_sample=True, pad_token_id=tokenizer.eos_token_id)
                elapsed = time.time() - start

                new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                token_count = len(new_tokens)
                tps = token_count / elapsed if elapsed > 0 else 0

                results.append({
                    "category": category, "prompt": prompt, "response": response,
                    "tokens": token_count, "time": round(elapsed, 2), "tps": round(tps, 2),
                    "min_words": item.get("min_words", 0),
                })
                print(f"    [{category}/{i+1}] {token_count} tok, {elapsed:.1f}s, {tps:.1f} tok/s")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


def compute_scores(results):
    scores = {"total_prompts": 0, "total_tokens": 0, "avg_tps": 0, "by_category": {}}
    tps_values = []

    for category in ["english", "code", "math", "handoff"]:
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue
        cat_tokens = sum(r["tokens"] for r in cat_results)
        cat_tps = [r["tps"] for r in cat_results if r["tps"] > 0]
        avg_tps = sum(cat_tps) / len(cat_tps) if cat_tps else 0
        met_min = sum(1 for r in cat_results if len(r["response"].split()) >= r.get("min_words", 0))

        scores["by_category"][category] = {
            "prompts": len(cat_results), "total_tokens": cat_tokens,
            "avg_tps": round(avg_tps, 2), "min_words_met": met_min,
            "min_words_total": len(cat_results),
            "min_words_pct": round(met_min / len(cat_results) * 100) if cat_results else 0,
        }
        scores["total_prompts"] += len(cat_results)
        scores["total_tokens"] += cat_tokens
        tps_values.extend(cat_tps)

    scores["avg_tps"] = round(sum(tps_values) / len(tps_values), 2) if tps_values else 0
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned vs original model")
    parser.add_argument("--original", type=str, help="Path to original GGUF model")
    parser.add_argument("--pruned", type=str, help="Path to pruned GGUF model")
    parser.add_argument("--original-hf", type=str, help="Path to original HF model (BF16)")
    parser.add_argument("--pruned-hf", type=str, help="Path to pruned HF model (BF16)")
    parser.add_argument("--output-dir", type=str, default="/home/kitt/gemma4-prune/eval")
    parser.add_argument("--port", type=int, default=8081, help="Port for llama-server")
    parser.add_argument("--categories", type=str, nargs="+",
                        default=["english", "code", "math", "handoff"],
                        choices=["english", "code", "math", "handoff"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = {k: EVAL_PROMPTS[k] for k in args.categories}

    orig_results = None
    pruned_results = None

    if args.original:
        print("=== Evaluating ORIGINAL model ===")
        orig_results = evaluate_with_llama_server(args.original, prompts, "original", args.port)
    elif args.original_hf:
        print("=== Evaluating ORIGINAL model (HuggingFace) ===")
        orig_results = evaluate_with_hf(args.original_hf, prompts, "original")

    if args.pruned:
        print("\n=== Evaluating PRUNED model ===")
        pruned_results = evaluate_with_llama_server(args.pruned, prompts, "pruned", args.port + 1)
    elif args.pruned_hf:
        print("\n=== Evaluating PRUNED model (HuggingFace) ===")
        pruned_results = evaluate_with_hf(args.pruned_hf, prompts, "pruned")

    # Save and compare
    if orig_results:
        with open(output_dir / "original_results.json", "w") as f:
            json.dump(orig_results, f, indent=2)
    if pruned_results:
        with open(output_dir / "pruned_results.json", "w") as f:
            json.dump(pruned_results, f, indent=2)

    if orig_results and pruned_results:
        orig_scores = compute_scores(orig_results)
        pruned_scores = compute_scores(pruned_results)
        comparison = {"original": orig_scores, "pruned": pruned_scores,
                      "diff": {
                          "avg_tps": round(pruned_scores["avg_tps"] - orig_scores["avg_tps"], 2),
                          "by_category": {
                              cat: {
                                  "tps_diff": round(pruned_scores["by_category"][cat]["avg_tps"] - orig_scores["by_category"][cat]["avg_tps"], 2),
                                  "quality_diff": round(pruned_scores["by_category"][cat]["min_words_pct"] - orig_scores["by_category"][cat]["min_words_pct"], 1),
                              }
                              for cat in args.categories if cat in orig_scores["by_category"] and cat in pruned_scores["by_category"]
                          }
                      }}
        with open(output_dir / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        print("\n=== Comparison ===")
        print(f"Original avg tok/s: {orig_scores['avg_tps']}")
        print(f"Pruned avg tok/s:   {pruned_scores['avg_tps']}")
        print(f"Speed diff:         {comparison['diff']['avg_tps']:+.2f} tok/s")
    elif pruned_results:
        scores = compute_scores(pruned_results)
        print(f"\nPruned model: avg {scores['avg_tps']} tok/s")
        for cat, data in scores["by_category"].items():
            print(f"  {cat}: {data['avg_tps']} tok/s, quality {data['min_words_pct']}%")