#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""Analyze lang_test results from the saved JSON."""
import json
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

with open("/home/kitt/gemma4-prune/lang_test_results.json") as f:
    data = json.load(f)

all_langs = list(data.keys())
num_layers = len(data["english"]["top1"])
num_experts = len(data["english"]["top1"][0])

print(f"Languages: {all_langs}")
print(f"Layers: {num_layers}, Experts per layer: {num_experts}")
print(f"Tokens per language:")
for lang in all_langs:
    print(f"  {lang:>10}: {data[lang]['tokens']:>6}")

# 1. Cosine similarity per layer
print("\n" + "="*80)
print("COSINE SIMILARITY: English vs other languages (top-1 activation counts)")
print("="*80)
print(f"{'Layer':>6}", end="")
for lang in all_langs[1:]:
    print(f"  {lang[:6]:>6}", end="")
print()

avg_sims = defaultdict(list)
for layer_idx in range(num_layers):
    en_vec = np.array(data["english"]["top1"][layer_idx], dtype=np.float64)
    print(f"{layer_idx:>6}", end="")
    for lang in all_langs[1:]:
        lang_vec = np.array(data[lang]["top1"][layer_idx], dtype=np.float64)
        if norm(en_vec) > 0 and norm(lang_vec) > 0:
            sim = np.dot(en_vec, lang_vec) / (norm(en_vec) * norm(lang_vec))
        else:
            sim = 0.0
        avg_sims[lang].append(sim)
        print(f"  {sim:>6.4f}", end="")
    print()

# 2. Average similarity
print("\n--- Average similarity to English ---")
for lang in all_langs[1:]:
    print(f"  {lang:>10}: {np.mean(avg_sims[lang]):.4f}")
overall = np.mean([np.mean(v) for v in avg_sims.values()])
print(f"  {'OVERALL':>10}: {overall:.4f}")

# 3. Language-specific experts
print("\n" + "="*80)
print("LANGUAGE-SPECIFIC EXPERT ANALYSIS")
print("="*80)

en_top5_all = set()
non_en_top5_all = set()
non_en_by_expert = defaultdict(set)

for layer_idx in range(num_layers):
    en_vec = np.array(data["english"]["top1"][layer_idx])
    en_top5 = set(np.argsort(en_vec)[-5:].tolist())
    en_top5_all |= en_top5
    for lang in all_langs[1:]:
        lang_vec = np.array(data[lang]["top1"][layer_idx])
        lang_top5 = set(np.argsort(lang_vec)[-5:].tolist())
        non_en_top5_all |= lang_top5
        for e in lang_top5:
            non_en_by_expert[e].add(lang)

language_specific = non_en_top5_all - en_top5_all
shared = en_top5_all & non_en_top5_all
english_only = en_top5_all - non_en_top5_all

print(f"\n  Experts ever in English top-5: {len(en_top5_all)}")
print(f"  Experts ever in non-English top-5: {len(non_en_top5_all)}")
print(f"  Shared (both EN and non-EN): {len(shared)}")
print(f"  Language-specific (non-EN only): {len(language_specific)}")
print(f"  English-specific (EN only): {len(english_only)}")

if language_specific:
    print(f"\n  Top language-specific experts (non-EN only):")
    sorted_specific = sorted(
        [(e, non_en_by_expert[e]) for e in language_specific],
        key=lambda x: len(x[1]), reverse=True
    )[:20]
    for expert_id, langs in sorted_specific:
        print(f"    Expert {expert_id:>3}: {', '.join(sorted(langs))}")

# 4. Per-layer detail for interesting layers
print("\n--- Per-layer top-5 comparison (selected layers) ---")
for layer_idx in [0, 5, 10, 15, 20, 25, 29]:
    if layer_idx >= num_layers:
        continue
    print(f"\n  Layer {layer_idx}:")
    en_vec = np.array(data["english"]["top1"][layer_idx])
    en_top5 = np.argsort(en_vec)[-5:][::-1]
    en_counts = en_vec[en_top5]
    print(f"    English:  top5={en_top5.tolist()}, counts={en_counts.tolist()}")
    for lang in ["chinese", "japanese", "arabic", "russian"]:
        if lang in data:
            lang_vec = np.array(data[lang]["top1"][layer_idx])
            lang_top5 = np.argsort(lang_vec)[-5:][::-1]
            lang_counts = lang_vec[lang_top5]
            overlap = len(set(en_top5) & set(lang_top5))
            print(f"    {lang:>10}: top5={lang_top5.tolist()}, counts={lang_counts.tolist()}, overlap={overlap}/5")

# 5. Probability distribution analysis
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*80)
# How much probability mass goes to top-5 vs rest, per language
for lang in all_langs:
    total_mass = 0
    top5_mass = 0
    for layer_idx in range(num_layers):
        prob = np.array(data[lang]["prob"][layer_idx])
        total_mass += prob.sum()
        top5_idx = np.argsort(prob)[-5:]
        top5_mass += prob[top5_idx].sum()
    pct = (top5_mass / total_mass * 100) if total_mass > 0 else 0
    print(f"  {lang:>10}: top-5 experts get {pct:.1f}% of probability mass")