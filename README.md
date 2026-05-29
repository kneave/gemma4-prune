# Gemma 4 Expert Pruning Pipeline

Compress Gemma 4 26B-A4B (128 experts) to an English-focused model with fewer experts,
suitable for single P40 GPU (24 GB VRAM) inference after retraining.

Based on: [esokullu/gemma4-turkish-26b-a4b-pruned](https://huggingface.co/esokullu/gemma4-turkish-26b-a4b-pruned)

## Keegan's Note

The idea behind this was to throw a load of sample data at an instrumented model to see which experts activated, after which we could see which were active or not and prune them and see how useful the model was afterwards. 

It turns out ripping out all but 48 experts wasn't actually terrible, it was vaguely coherant but ask it the capital of France or to tell a joke and it quickly went of the rails. Ask it about potatoes, seriously.

It was looking promising though running the heal script on CPU would've taken over a year which isn't great really. If anyone manages to get this working please let me know!

## Setup

```bash
source /home/kitt/ai/gemma4-prune-venv/bin/activate
```

Venv is at `/home/kitt/ai/gemma4-prune-venv/` with PyTorch 2.7.1+cu126 (P40 compatible).

## Model Structure (Gemma 4 26B-A4B)

```
Gemma4ForConditionalGeneration
└── model (Gemma4Model)
    ├── vision_tower
    ├── language_model (Gemma4TextModel)
    │   ├── embed_tokens
    │   ├── layers[0..29]
    │   │   ├── self_attn
    │   │   ├── mlp (shared expert, NOT pruned)
    │   │   │   ├── gate_proj [2112, 2816]
    │   │   │   ├── up_proj [2112, 2816]
    │   │   │   └── down_proj [2816, 2112]
    │   │   ├── router (Gemma4TextRouter, PRUNED)
    │   │   │   ├── proj.weight [128, 2816]
    │   │   │   ├── scale [2816]
    │   │   │   └── per_expert_scale [128]
    │   │   └── experts (Gemma4TextExperts, PRUNED)
    │   │       ├── gate_up_proj [128, 1408, 2816]
    │   │       └── down_proj [128, 2816, 704]
    │   └── norm
    └── lm_head
```

Key: Experts are **fused 3D tensors** (not ModuleList). Pruning = slicing dim 0.
Router `per_expert_scale` is a 1D tensor per layer that also needs slicing.

## Pipeline

Download the BF16 weights from [google/gemma-4-26b-a4b-it](https://huggingface.co/google/gemma-4-26b-a4b-it).

```
1. Download BF16 weights          (~1-2 hours)
2. Calibrate (log expert usage)   (~4-6 hours on P40+CPU)
3. Prune (slice experts)          (~10 minutes)
4. [Optional] LoRA heal           (~2-4 hours on P40)
5. Convert to GGUF + quantize     (~1 hour)
6. Evaluate (before/after)        (~30 minutes)
```

## Commands

```bash
tmux new -s gemma4-prune
source /home/kitt/ai/gemma4-prune-venv/bin/activate

# Step 1: Download (already done in ~/gemma4-prune/bf16/)

# Step 2: Calibrate
python /home/kitt/gemma4-prune/01_calibrate.py \
  --model-path /home/kitt/gemma4-prune/bf16 \
  --output-dir /home/kitt/gemma4-prune/calibration \
  --num-samples 6000 --device auto

# Step 3: Prune to 32 experts
python /home/kitt/gemma4-prune/02_prune_experts.py \
  --model-path /home/kitt/gemma4-prune/bf16 \
  --calibration /home/kitt/gemma4-prune/calibration/activation_stats.json \
  --output-dir /home/kitt/gemma4-prune/pruned \
  --keep-experts 32

# Step 4 (skip for initial test): LoRA heal
python /home/kitt/gemma4-prune/03_lora_heal.py \
  --model-path /home/kitt/gemma4-prune/pruned \
  --output-dir /home/kitt/gemma4-prune/healed \
  --create-dataset --num-samples 25000

# Step 5: Convert to GGUF
python /home/kitt/gemma4-prune/04_convert_gguf.py \
  --model-path /home/kitt/gemma4-prune/pruned \
  --quantize Q4_K_M \
  --copy-to-models --model-name gemma4-english-E32

# Step 6: Evaluate
python /home/kitt/gemma4-prune/05_evaluate.py \
  --original ~/models/gguf/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --pruned ~/models/gguf/gemma4-english-E32-Q4_K_M.gguf \
  --output-dir /home/kitt/gemma4-prune/eval
```

## Expected Results

| Config | File Size | VRAM | Speed (P40) | Quality |
|--------|-----------|------|-------------|---------|
| 128 experts (original) | ~16 GB | Tight | ~42 tok/s | Full |
| 32 experts (target) | ~5 GB | Comfortable | ~60+ tok/s | Good English |
| 48 experts (moderate) | ~7 GB | Good | ~55 tok/s | Better code/math |

## Disk Space

| Step | Peak Usage | Can Delete After |
|------|------------|-----------------|
| Download BF16 | ~52 GB | After pruning |
| Calibrate | ~55 GB | After pruning |
| Prune | ~70 GB | After conversion |
| Convert | ~20 GB | After copying to models/ |

**Strategy**: You have ~95 GB free. Delete intermediate files between steps.

## Monitoring

Cron job checks tmux session every 30 minutes and reports progress in Discord.