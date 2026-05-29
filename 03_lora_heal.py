#!/usr/bin/env python3
"""
LoRA healing for pruned Gemma 4 MoE model.

After expert pruning (128->64), the router's gating network needs recalibration.
The softmax was trained to distribute probability across 128 experts; removing
64 leaves gaps that flatten confidence, causing repetition loops.

Based on proven precedent: esokullu/gemma4-turkish-26b-a4b-pruned
  - r=32, alpha=64, 2 epochs, ~25k samples
  - LoRA-healed on A100 in ~2 hours

Adapted for CPU-only training:
  - QLoRA (4-bit base) to reduce memory footprint
  - fp32 compute dtype for CPU compatibility (no BF16 on CPU)
  - English/code/math focus instead of Turkish
  - Real datasets (FineWeb-Edu, CodeAlpaca, OpenMathInstruct-2)

Training data mix: 60% English, 25% code, 15% math
"""

import os
import sys
import json
import torch
import argparse
import time
from pathlib import Path
from transformers import TrainerCallback


def load_english_data(num_samples, seed=42):
    """Load English text from HuggingFace FineWeb-Edu dataset."""
    from datasets import load_dataset
    print(f"Loading English data (FineWeb-Edu), targeting {num_samples} samples...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    texts = []
    for example in dataset:
        if len(texts) >= num_samples:
            break
        text = example.get("text", "").strip()
        if len(text) < 200 or len(text) > 8000:
            continue
        ascii_ratio = sum(1 for c in text[:500] if ord(c) < 128) / min(len(text), 500)
        if ascii_ratio < 0.9:
            continue
        texts.append(text)
        if len(texts) % 2000 == 0 and len(texts) > 0:
            print(f"  English: {len(texts)}/{num_samples} samples loaded")

    print(f"Loaded {len(texts)} English samples")
    return texts


def load_code_data(num_samples, seed=42):
    """Load code from CodeAlpaca-20k (ungated alternative to StarCoder)."""
    from datasets import load_dataset
    print(f"Loading code data (CodeAlpaca-20k), targeting {num_samples} samples...")

    dataset = load_dataset(
        "sahil2801/CodeAlpaca-20k",
        split="train",
        streaming=True,
    )

    texts = []
    for example in dataset:
        instruction = example.get("instruction", "").strip()
        output = example.get("output", "").strip()
        inp = example.get("input", "").strip()

        if not output:
            continue

        # Format as instruction-response pair
        if inp:
            text = f"Instruction: {instruction}\nInput: {inp}\nOutput:\n{output}"
        else:
            text = f"Instruction: {instruction}\nOutput:\n{output}"

        if len(text) < 30 or len(text) > 4000:
            continue

        texts.append(text)
        if len(texts) >= num_samples:
            break

    print(f"Loaded {len(texts)} code samples")
    return texts


def load_math_data(num_samples, seed=42):
    """Load math reasoning from OpenMathInstruct-2."""
    from datasets import load_dataset
    print(f"Loading math data (OpenMathInstruct-2), targeting {num_samples} samples...")

    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train",
        streaming=True,
    )

    texts = []
    for example in dataset:
        if len(texts) >= num_samples:
            break
        problem = example.get("problem", "").strip()
        solution = example.get("generated_solution", "").strip()

        if not problem or not solution:
            continue

        text = f"Problem: {problem}\n\nSolution:\n{solution}"
        if len(text) < 50 or len(text) > 4000:
            continue

        texts.append(text)
        if len(texts) % 1000 == 0 and len(texts) > 0:
            print(f"  Math: {len(texts)}/{num_samples} samples loaded")

    print(f"Loaded {len(texts)} math samples")
    return texts


class TelemetryCallback(TrainerCallback):
    """Logs training progress with time estimates and step-level telemetry."""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        self.step_times = []
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print(f"\n{'='*60}")
        print(f"  TRAINING STARTED — {self.total_steps} total steps")
        print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 0:
            return
        now = time.time()

        # Track step timing
        if self.last_log_time:
            step_duration = now - self.last_log_time
            self.step_times.append(step_duration)
        self.last_log_time = now

        # Use rolling average of last 10 steps for speed estimation
        recent = self.step_times[-10:]
        avg_step_time = sum(recent) / len(recent)

        completed = state.global_step
        remaining = self.total_steps - completed
        eta_seconds = remaining * avg_step_time

        loss = logs.get("loss", None) if logs else None
        lr = logs.get("learning_rate", None) if logs else None

        elapsed = now - self.start_time
        progress_pct = (completed / self.total_steps) * 100

        parts = [
            f"[Step {completed}/{self.total_steps}] ({progress_pct:.1f}%)",
        ]
        if loss is not None:
            parts.append(f"loss={loss:.4f}")
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        parts.append(f"speed={1/avg_step_time:.2f} steps/s")
        parts.append(f"elapsed={self._fmt_duration(elapsed)}")
        parts.append(f"ETA={self._fmt_duration(eta_seconds)}")

        print(" ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Total steps: {state.global_step}")
        print(f"  Total time: {self._fmt_duration(total_time)}")
        if self.step_times:
            avg = sum(self.step_times) / len(self.step_times)
            print(f"  Avg speed: {1/avg:.2f} steps/s ({avg:.2f}s/step)")
        print(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    @staticmethod
    def _fmt_duration(seconds):
        seconds = int(seconds)
        h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        elif m > 0:
            return f"{m}m{s:02d}s"
        else:
            return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description="LoRA heal pruned Gemma 4 MoE")
    parser.add_argument("--model-path", default="/home/kitt/gemma4-prune/pruned-48",
                        help="Path to pruned BF16 model")
    parser.add_argument("--output-dir", default="/home/kitt/gemma4-prune/lora-healed",
                        help="Output directory for merged model")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (precedent: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (precedent: 64)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (precedent: 2)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--num-samples", type=int, default=25000, help="Target number of training samples")
    parser.add_argument("--save-steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # ---- Setup ----
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"=== LoRA Healing (Precedent Parameters) ===")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Target samples: {args.num_samples} (60% English, 25% code, 15% math)")
    print(f"Batch size: {args.batch_size} x {args.gradient_accumulation} accum = {args.batch_size * args.gradient_accumulation} effective")
    print(f"Device: CPU-only (no GPU)")
    print()

    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset
    import random

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load training data ----
    #- 60% FineWeb-Edu English, 25% CodeAlpaca code, 15% OpenMathInstruct-2 math
    n_english = int(args.num_samples * 0.60)
    n_code = int(args.num_samples * 0.25)
    n_math = int(args.num_samples * 0.15)

    english_texts = load_english_data(n_english, seed=args.seed)
    code_texts = load_code_data(n_code, seed=args.seed)
    math_texts = load_math_data(n_math, seed=args.seed)

    # If code data is smaller than target, scale proportionally
    if len(code_texts) < n_code:
        print(f"  Note: Only {len(code_texts)} code samples available (requested {n_code}). Using all.")
        print(f"  Total samples will be lower than target — this is OK for healing.")

    # Combine and shuffle
    all_texts = english_texts + code_texts + math_texts
    random.shuffle(all_texts)
    print(f"\nTotal training samples: {len(all_texts)}")
    print(f"  English: {len(english_texts)} ({len(english_texts)/len(all_texts)*100:.0f}%)")
    print(f"  Code: {len(code_texts)} ({len(code_texts)/len(all_texts)*100:.0f}%)")
    print(f"  Math: {len(math_texts)} ({len(math_texts)/len(all_texts)*100:.0f}%)")

    if len(all_texts) < 5000:
        print(f"\nWARNING: Only {len(all_texts)} samples loaded. Consider checking dataset availability.")
        print("Proceeding with available data, but quality may be reduced.")

    # ---- Load tokenizer ----
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab size: {len(tokenizer)}")

    # ---- Load model in 4-bit ----
    # Gemma4 config says Gemma4ForConditionalGeneration (includes vision encoder),
    # which OOMs the P40. We must load as text-only (Gemma4ForCausalLM) and
    # use device_map={'': 0} to avoid accelerate offload bugs with Params4bit.
    # E56 at 4-bit text-only is ~22.6GB — too tight for training on 24GB.
    # E48 at 4-bit text-only is ~19.8GB — leaves ~4.2GB for LoRA + optimizer.
    # With gradient_checkpointing + paged_adamw_8bit, E48 JUST fits.
    # If you have more VRAM, change --model-path to pruned-56 or pruned-64.
    print("Loading text-only model in 4-bit (QLoRA)...")
    print(f"Model path: {args.model_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )

    # CPU-only: no CUDA checks needed

    # CPU-only: load entirely on CPU, no GPU offload
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model = prepare_model_for_kbit_training(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable before LoRA: {trainable_before:,}")

    # ---- Configure LoRA (precedent parameters) ----
    # Collect LoRA target modules — only language model modules, NOT vision tower.
    # Vision tower uses Gemma4ClippableLinear which PEFT can't LoRA.
    # String patterns like "gate_proj" match both vision and language modules,
    # so we must enumerate explicitly.
    target_module_names = []
    for name, module in model.named_modules():
        if not name.startswith("model.language_model."):
            continue
        # Attention projections (Linear4bit — LoRA-compatible)
        if any(name.endswith(f".self_attn.{p}") for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
            target_module_names.append(name)
        # MLP projections (Linear4bit — LoRA-compatible)
        elif any(name.endswith(f".mlp.{p}") for p in ("gate_proj", "up_proj", "down_proj")):
            target_module_names.append(name)
        # Router projection (Linear4bit — critical for recalibrating expert gating)
        elif name.endswith(".router.proj"):
            target_module_names.append(name)

    print(f"LoRA target modules: {len(target_module_names)} (language model only)")
    print(f"  Attention: {sum(1 for n in target_module_names if 'self_attn' in n)}")
    print(f"  MLP: {sum(1 for n in target_module_names if 'mlp' in n)}")
    print(f"  Router: {sum(1 for n in target_module_names if 'router' in n)}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_module_names,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Tokenize dataset ----
    print(f"\nTokenizing {len(all_texts)} training samples...")

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    dataset_dict = {"text": all_texts}
    raw_dataset = Dataset.from_dict(dataset_dict)
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Filter out sequences that are too short
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 50,
        desc="Filtering short sequences",
    )

    print(f"Dataset size after filtering: {len(tokenized_dataset)} samples")

    # ---- Data collator ----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # ---- Training arguments ----
    effective_batch = args.batch_size * args.gradient_accumulation
    total_steps = (len(tokenized_dataset) // effective_batch) * args.epochs

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(total_steps * args.warmup_ratio),
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_cpu=True,
        fp16=False,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
    )

    print(f"\n=== Training Configuration ===")
    print(f"Total steps: ~{total_steps}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Learning rate: {args.lr}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"Scheduler: cosine")
    print(f"Precision: fp32 (CPU)")
    print(f"Gradient checkpointing: enabled")
    print(f"Optimizer: adamw_torch (CPU-compatible)")
    print(f"Epochs: {args.epochs}")
    print()

    # ---- Train ----
    telemetry = TelemetryCallback(total_steps)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[telemetry],
    )

    print("Starting LoRA healing training...")
    train_result = trainer.train()

    # ---- Save LoRA adapters ----
    print("\nSaving LoRA adapters...")
    lora_output = os.path.join(args.output_dir, "lora-adapter")
    model.save_pretrained(lora_output)
    tokenizer.save_pretrained(lora_output)
    print(f"LoRA adapter saved to: {lora_output}")

    # ---- Merge LoRA into base model ----
    print("\nMerging LoRA into base model...")
    from peft import AutoPeftModelForCausalLM

    # Merge on CPU (no GPU needed)
    print("Loading base model for merging (fp32, CPU)...")
    merged_model = AutoPeftModelForCausalLM.from_pretrained(
        lora_output,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    print("Merging adapter weights...")
    merged_model = merged_model.merge_and_unload()

    merged_output = os.path.join(args.output_dir, "merged-bf16")
    print(f"Saving merged model to: {merged_output}")
    merged_model.save_pretrained(merged_output, safe_serialization=True)
    tokenizer.save_pretrained(merged_output)

    # Save config
    merge_info = {
        "base_model": args.model_path,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "num_samples": len(all_texts),
        "effective_batch_size": effective_batch,
        "total_steps": total_steps,
        "seed": args.seed,
        "data_mix": {
            "english": len(english_texts),
            "code": len(code_texts),
            "math": len(math_texts),
        },
        "train_loss": train_result.training_loss,
    }
    with open(os.path.join(merged_output, "healing_info.json"), "w") as f:
        json.dump(merge_info, f, indent=2)

    print(f"\nDone! Merged model saved to: {merged_output}")
    print(f"Next steps:")
    print(f"  1. Convert to GGUF:")
    print(f"     python /home/kitt/ai/llama.cpp/convert_hf_to_gguf.py {merged_output} --outfile /home/kitt/gemma4-prune/lora-healed-e64-f16.gguf --outtype f16")
    print(f"  2. Quantize:")
    print(f"     /home/kitt/ai/llama.cpp/build/bin/llama-quantize /home/kitt/gemma4-prune/lora-healed-e64-f16.gguf /home/kitt/models/gguf/gemma4-26b-a4b-e64-healed-Q4_K_M.gguf Q4_K_M")
    print(f"  3. Test:")
    print(f"     LD_LIBRARY_PATH=/home/kitt/ai/llama.cpp/build/bin:$LD_LIBRARY_PATH \\")
    print(f"       /home/kitt/ai/llama.cpp/build/bin/llama-cli -m /home/kitt/models/gguf/gemma4-26b-a4b-e64-healed-Q4_K_M.gguf -ngl 99 -c 4096 --temp 0.7 --reasoning off")


if __name__ == "__main__":
    main()