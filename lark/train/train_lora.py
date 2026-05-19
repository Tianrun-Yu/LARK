#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA SFT training on Q2 train.jsonl data.

Usage:
  python train_lora.py \
    --train_data  ./data/selections/qwen2.5-7b/random/topb_B1/train.jsonl \
    --model_path  Qwen/qwen2.5-7b \
    --output_dir  ./checkpoints/qwen2.5-7b/random/topb_B1
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_jsonl(path: str):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


class SFTDataset(TorchDataset):
    """
    Tokenizes each example with the chat template.
    Labels are set to -100 for the system+user (prompt) tokens so that loss
    is computed only on the assistant (completion) tokens.
    """

    def __init__(self, examples, tokenizer, max_length):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records    = []

        for ex in examples:
            messages = ex["messages"]

            # Full text: system + user + assistant
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Prompt only: system + user (up to but not including assistant turn)
            prompt_messages = [m for m in messages if m["role"] != "assistant"]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,  # adds the assistant-turn opener
            )

            full_ids   = tokenizer(full_text,   truncation=True,
                                   max_length=max_length)["input_ids"]
            prompt_ids = tokenizer(prompt_text, truncation=True,
                                   max_length=max_length)["input_ids"]

            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            # Truncate labels to same length as input_ids
            labels = labels[: len(full_ids)]

            # χ²-optimal per-sample weight (1.0 for methods without weighting)
            train_weight = float(ex.get("meta", {}).get("train_weight", 1.0))

            self.records.append({
                "input_ids":      full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels":         labels,
                "train_weight":   train_weight,
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # Return numpy arrays; FastCollator uses torch.from_numpy (zero-copy)
        r = self.records[idx]
        return {
            "input_ids":      np.array(r["input_ids"],      dtype=np.int64),
            "attention_mask": np.array(r["attention_mask"],  dtype=np.int64),
            "labels":         np.array(r["labels"],          dtype=np.int64),
            "length":         len(r["input_ids"]),   # used by group_by_length
            "train_weight":   np.float32(r["train_weight"]),
        }


class FastCollator:
    """
    Pads a batch of variable-length sequences with numpy pre-allocation,
    then converts to tensor via torch.from_numpy (zero-copy).
    Avoids the slow 'Creating tensor from list of numpy.ndarrays' path.
    """
    def __init__(self, pad_token_id: int = 0,
                 label_pad_token_id: int = -100,
                 pad_to_multiple_of: int = 8):
        self.pad_token_id       = pad_token_id
        self.label_pad_id       = label_pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        B = len(features)
        inp  = np.full((B, max_len), self.pad_token_id,  dtype=np.int64)
        mask = np.zeros((B, max_len),                     dtype=np.int64)
        lbl  = np.full((B, max_len), self.label_pad_id,  dtype=np.int64)
        wts  = np.ones(B, dtype=np.float32)

        for i, f in enumerate(features):
            L = f["input_ids"].shape[0]
            inp[i,  :L] = f["input_ids"]
            mask[i, :L] = f["attention_mask"]
            lbl[i,  :L] = f["labels"]
            if "train_weight" in f:
                wts[i] = float(f["train_weight"])

        return {
            "input_ids":      torch.from_numpy(inp),
            "attention_mask": torch.from_numpy(mask),
            "labels":         torch.from_numpy(lbl),
            "train_weight":   torch.from_numpy(wts),
        }


# ---------------------------------------------------------------------------
# AIME eval dataset (cross-entropy loss on answer tokens only)
# ---------------------------------------------------------------------------

class AIMEEvalDataset(TorchDataset):
    """
    Builds a tiny eval dataset from AIME.json.
    Each example: system + problem → \boxed{answer}
    Loss is computed only on the answer tokens.
    """
    def __init__(self, aime_path: str, tokenizer, max_length: int = 512,
                 system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}"):
        self.records = []
        with open(aime_path, "r") as f:
            problems = json.load(f)

        for item in problems:
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": item["prompt"]},
                {"role": "assistant", "content": f"\\boxed{{{item['answer']}}}"},
            ]
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_messages = messages[:2]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)

            full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]

            self.records.append({
                "input_ids":      np.array(full_ids, dtype=np.int64),
                "attention_mask": np.array([1] * len(full_ids), dtype=np.int64),
                "labels":         np.array(labels, dtype=np.int64),
                "length":         len(full_ids),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ---------------------------------------------------------------------------
# Callback: plot train loss vs AIME eval loss
# ---------------------------------------------------------------------------

class LossPlotCallback(TrainerCallback):
    """
    After each logging step, reads train/eval losses from Trainer state
    and saves a loss curve PNG to output_dir.
    """
    def __init__(self, output_dir: str, train_dataset, data_collator, n_init_samples: int = 8):
        self.output_dir       = output_dir
        self.train_dataset    = train_dataset
        self.data_collator    = data_collator
        self.n_init_samples   = n_init_samples
        os.makedirs(output_dir, exist_ok=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return
        # Compute step-0 train loss on a small random sample of training data
        try:
            import random
            indices = random.sample(range(len(self.train_dataset)),
                                    min(self.n_init_samples, len(self.train_dataset)))
            samples = [self.train_dataset[i] for i in indices]
            batch   = self.data_collator(samples)
            batch   = {k: v.to(model.device) for k, v in batch.items()}
            model.eval()
            with torch.no_grad():
                loss = model(**batch).loss.item()
            model.train()
            state.log_history.insert(0, {"step": 0, "loss": loss})
            print(f"[INFO] step 0 train loss = {loss:.4f}")
        except Exception as e:
            print(f"[WARN] step-0 train loss failed: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return
        # eval_loss at step 0 is logged here when Trainer runs initial eval
        self._save_plot(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return
        self._save_plot(state)

    def _save_plot(self, state):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            train_steps, train_losses = [], []
            eval_steps,  eval_losses  = [], []

            for entry in state.log_history:
                if "loss" in entry:
                    train_steps.append(entry["step"])
                    train_losses.append(entry["loss"])
                if "eval_loss" in entry:
                    eval_steps.append(entry["step"])
                    eval_losses.append(entry["eval_loss"])

            if not train_steps:
                return

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_steps, train_losses, label="Train Loss", alpha=0.8)
            if eval_steps:
                ax.plot(eval_steps, eval_losses, "r-o", label="AIME Eval Loss",
                        markersize=5, linewidth=2)
            ax.set_xlabel("Steps")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss vs AIME Eval Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(self.output_dir, "loss_curve.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

            # also save CSV
            import csv
            csv_path = os.path.join(self.output_dir, "loss_log.csv")
            t_dict = dict(zip(train_steps, train_losses))
            e_dict = dict(zip(eval_steps,  eval_losses))
            all_steps = sorted(set(train_steps) | set(eval_steps))
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "train_loss", "eval_loss"])
                for s in all_steps:
                    w.writerow([s, t_dict.get(s, ""), e_dict.get(s, "")])
        except Exception as e:
            print(f"[WARN] plot/csv save failed: {e}")


# ---------------------------------------------------------------------------
# Weighted trainer: applies per-sample χ²-optimal weights to the CE loss
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Overrides compute_loss to support per-sample train_weight from the batch.
    If all weights are 1.0 (or train_weight is absent), behaviour is identical
    to the standard Trainer.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("train_weight", None)   # (B,) float32 or None

        # If weights are all 1 (uniform), skip manual computation for speed
        if weights is None or (weights == 1.0).all():
            if weights is not None:
                inputs["labels"] = inputs.get("labels")  # already in inputs
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # Manual weighted cross-entropy
        labels = inputs.pop("labels")                # remove so model doesn't compute loss
        outputs = model(**inputs)
        logits = outputs.logits                      # (B, T, V)

        # Shift for causal LM: predict token t+1 from token t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        B, T, V = shift_logits.shape
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fct(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        ).view(B, T)                                 # (B, T)

        # Mean over non-masked tokens per sample
        non_pad = (shift_labels != -100).float()
        denom = non_pad.sum(dim=-1).clamp(min=1.0)
        per_sample_loss = (per_token_loss * non_pad).sum(dim=-1) / denom  # (B,)

        # Apply χ²-optimal weights; normalise so Σw = B (preserves scale)
        weights = weights.to(per_sample_loss.device).float()
        w_sum = weights.sum().clamp(min=1e-8)
        loss = (per_sample_loss * weights).sum() / w_sum * B / B  # = weighted mean

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LoRA SFT training on Q2 train.jsonl."
    )
    ap.add_argument("--train_data",   required=True,
                    help="Path to train.jsonl produced by Q2 selection.")
    ap.add_argument("--model_path",   required=True,
                    help="Path to the student base model directory.")
    ap.add_argument("--output_dir",   required=True,
                    help="Directory to save checkpoints and final adapter.")
    ap.add_argument("--lora_config",  default=None,
                    help="Path to lora_default.yaml (auto-detected if omitted).")
    ap.add_argument("--sft_config",   default=None,
                    help="Path to sft_default.yaml (auto-detected if omitted).")
    # Per-run overrides
    ap.add_argument("--lr",           type=float, default=None)
    ap.add_argument("--epochs",       type=int,   default=None)
    ap.add_argument("--batch_size",   type=int,   default=None,
                    help="Per-device train batch size.")
    ap.add_argument("--grad_accum",   type=int,   default=None,
                    help="Gradient accumulation steps.")
    ap.add_argument("--max_length",   type=int,   default=None)
    ap.add_argument("--lora_rank",    type=int,   default=None)
    ap.add_argument("--lora_alpha",   type=int,   default=None)
    ap.add_argument("--lora_dropout", type=float, default=None)
    ap.add_argument("--aime_data",   type=str,   default=None,
                    help="Path to AIME.json for eval loss monitoring.")
    ap.add_argument("--eval_steps",  type=int,   default=10,
                    help="Compute AIME eval loss every N steps (should match logging_steps).")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Load configs
    # -----------------------------------------------------------------------
    configs_dir = Path(__file__).parent / "configs"
    lora_cfg = load_yaml(args.lora_config or str(configs_dir / "lora_default.yaml"))
    sft_cfg  = load_yaml(args.sft_config  or str(configs_dir / "sft_default.yaml"))

    # Apply CLI overrides
    if args.lr           is not None: sft_cfg["learning_rate"]              = args.lr
    if args.epochs       is not None: sft_cfg["num_train_epochs"]            = args.epochs
    if args.batch_size   is not None: sft_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum   is not None: sft_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.max_length   is not None: sft_cfg["max_seq_length"]              = args.max_length
    if args.lora_rank    is not None: lora_cfg["r"]                          = args.lora_rank
    if args.lora_alpha   is not None: lora_cfg["lora_alpha"]                 = args.lora_alpha
    if args.lora_dropout is not None: lora_cfg["lora_dropout"]               = args.lora_dropout

    max_seq_length = sft_cfg.get("max_seq_length", 32768)

    print(f"[INFO] train_data   : {args.train_data}")
    print(f"[INFO] model_path   : {args.model_path}")
    print(f"[INFO] output_dir   : {args.output_dir}")
    print(f"[INFO] max_seq_len  : {max_seq_length}")
    print(f"[INFO] lora_cfg     : {lora_cfg}")
    print(f"[INFO] sft_cfg      : {sft_cfg}")

    # -----------------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # required for causal LM training

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    raw     = load_train_jsonl(args.train_data)
    dataset = SFTDataset(raw, tokenizer, max_seq_length)
    print(f"[INFO] dataset size : {len(dataset)}")

    # AIME eval dataset (optional)
    aime_path = args.aime_data or "./data/benchmarks/AIME.json"
    eval_dataset = None
    if os.path.exists(aime_path):
        eval_dataset = AIMEEvalDataset(aime_path, tokenizer)
        print(f"[INFO] AIME eval set : {len(eval_dataset)} problems")

    # -----------------------------------------------------------------------
    # Model (load in bf16, device_map="auto" for multi-GPU)
    # -----------------------------------------------------------------------
    # Use PyTorch 2.x built-in SDPA (no extra install needed, similar speed to flash_attn)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print("[INFO] using attn_implementation=sdpa")
    model.enable_input_require_grads()   # needed for gradient checkpointing + PEFT

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Training arguments
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=sft_cfg.get("num_train_epochs", 10),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 64),
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.03),
        bf16=True,
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=sft_cfg.get("save_strategy", "epoch"),
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,       # batch similar-length seqs → less padding waste
        length_column_name="length",
        report_to="none",
        # AIME eval loss
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_on_start=eval_dataset is not None,   # eval at step 0 before training
        per_device_eval_batch_size=1,
    )

    # -----------------------------------------------------------------------
    # Fast collator: numpy pre-allocation + zero-copy torch.from_numpy
    # -----------------------------------------------------------------------
    data_collator = FastCollator(
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[LossPlotCallback(args.output_dir, dataset, data_collator)],
    )

    # Resume from checkpoint if one exists in output_dir
    import glob
    ckpt_dirs = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    resume = ckpt_dirs[-1] if ckpt_dirs else None
    if resume:
        print(f"[INFO] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # -----------------------------------------------------------------------
    # Save final adapter
    # -----------------------------------------------------------------------
    adapter_dir = os.path.join(args.output_dir, "adapter_final")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"[OK] adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
