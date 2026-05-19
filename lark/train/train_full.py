#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full parameter SFT with:
  - Flash Attention 2
  - ZeRO-3 (DeepSpeed)
  - bf16
  - Sequence Packing (greedy bin-packing + position_ids reset)
  - Liger Kernels (fused RMSNorm, SwiGLU, CrossEntropy)
  - Gradient Checkpointing
  - Per-sample train_weight support (from meta.train_weight)

Usage (4 GPUs):
  torchrun --nproc_per_node=4 train_full.py \
    --train_data  ./data/selections/qwen2.5-math-7b/rsr/chi2_B5/train.jsonl \
    --model_path  Qwen/qwen2.5-math-7b \
    --output_dir  ./checkpoints/qwen2.5-math-7b/rsr/chi2_B5_full

Usage (8 GPUs):
  torchrun --nproc_per_node=8 train_full.py ...
"""

import os
import json
import csv
import random
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml
import torch
# Force fp32 accumulation in bf16 matmuls on A100 (no memory overhead).
# Without this, A100 uses bf16 reduced-precision accumulators in tensor cores,
# which can produce NaN in lm_head backward matmuls during fine-tuning.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)


# ---------------------------------------------------------------------------
# Optional: Liger kernels (fused ops — must apply BEFORE model load)
# ---------------------------------------------------------------------------

def apply_liger(model_type: str = "qwen2"):
    try:
        if model_type == "qwen2":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True,
                rms_norm=True,
                swiglu=True,
                fused_linear_cross_entropy=False,  # Disabled: incompatible with sequence packing
            )
        print("[INFO] Liger kernels applied (rope/rms_norm/swiglu only)")
    except ImportError:
        print("[WARN] liger_kernel not installed, skipping")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_jsonl(path: str) -> List[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Tokenize helpers
# ---------------------------------------------------------------------------

def tokenize_example(ex: dict, tokenizer, max_length: int):
    """Returns (input_ids, labels, train_weight) as lists/float."""
    messages = ex["messages"]
    train_weight = float(ex.get("meta", {}).get("train_weight", 1.0))

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    prompt_messages = [m for m in messages if m["role"] != "assistant"]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[:len(full_ids)]

    return full_ids, labels, train_weight


# ---------------------------------------------------------------------------
# Greedy sequence packing
# ---------------------------------------------------------------------------

def pack_sequences(records: List[dict], max_length: int) -> List[dict]:
    """
    Greedy bin-packing: fills bins of max_length tokens.
    Each bin becomes one training example with:
      - input_ids:    concatenated token ids
      - labels:       concatenated labels (-100 for prompt tokens)
      - position_ids: reset to 0 at each sequence boundary
      - token_weights: per-token train_weight (replicated from source example)
    """
    # Sort by length descending for better bin utilisation
    records = sorted(records, key=lambda r: len(r["input_ids"]), reverse=True)

    bins: List[List[dict]] = []
    bin_lengths: List[int] = []

    for rec in records:
        L = len(rec["input_ids"])
        if L > max_length:
            continue  # single over-length example: skip

        placed = False
        for i, bl in enumerate(bin_lengths):
            if bl + L <= max_length:
                bins[i].append(rec)
                bin_lengths[i] += L
                placed = True
                break
        if not placed:
            bins.append([rec])
            bin_lengths.append(L)

    packed = []
    for bin_recs in bins:
        input_ids    = []
        labels       = []
        position_ids = []
        token_weights= []

        for rec in bin_recs:
            seq_len = len(rec["input_ids"])
            input_ids    += rec["input_ids"]
            labels       += rec["labels"]
            position_ids += list(range(seq_len))            # reset per sequence
            token_weights+= [rec["train_weight"]] * seq_len # broadcast weight to tokens

        packed.append({
            "input_ids":     input_ids,
            "labels":        labels,
            "position_ids":  position_ids,
            "token_weights": token_weights,
            "length":        len(input_ids),
        })

    print(f"[INFO] Packing: {len(records)} examples → {len(packed)} packed bins "
          f"(ratio {len(records)/max(1,len(packed)):.1f}x)")
    return packed


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PackedSFTDataset(TorchDataset):
    def __init__(self, examples: List[dict], tokenizer, max_length: int):
        # Tokenize — no packing, one example per record
        self.records = []
        for ex in examples:
            ids, lbl, w = tokenize_example(ex, tokenizer, max_length)
            L = len(ids)
            self.records.append({
                "input_ids":     np.array(ids,              dtype=np.int64),
                "labels":        np.array(lbl,              dtype=np.int64),
                "position_ids":  np.arange(L,               dtype=np.int64),
                "token_weights": np.full(L, w,              dtype=np.float32),
                "length":        L,
            })
        print(f"[INFO] Dataset: {len(self.records)} examples (packing disabled)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]   # already numpy — no copy needed


# ---------------------------------------------------------------------------
# AIME eval dataset (unchanged — no packing needed for eval)
# ---------------------------------------------------------------------------

class AIMEEvalDataset(TorchDataset):
    def __init__(self, aime_path: str, tokenizer, max_length: int = 1024,
                 system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}"):
        self.records = []
        with open(aime_path) as f:
            problems = json.load(f)

        for item in problems:
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": item["prompt"]},
                {"role": "assistant", "content": f"\\boxed{{{item['answer']}}}"},
            ]
            full_text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_msgs = messages[:2]
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

            full_ids   = tokenizer(full_text,   truncation=True, max_length=max_length)["input_ids"]
            prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]

            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + full_ids[prompt_len:]
            labels = labels[:len(full_ids)]

            self.records.append({
                "input_ids":     np.array(full_ids, dtype=np.int64),
                "labels":        np.array(labels,   dtype=np.int64),
                "position_ids":  np.array(list(range(len(full_ids))), dtype=np.int64),
                "token_weights": np.ones(len(full_ids), dtype=np.float32),
                "length":        len(full_ids),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


# ---------------------------------------------------------------------------
# Collator: handles variable-length packed sequences
# ---------------------------------------------------------------------------

class PackedCollator:
    def __init__(self, pad_token_id: int = 0, pad_to_multiple_of: int = 64):
        self.pad_token_id    = pad_token_id
        self.pad_to_multiple = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)
        if self.pad_to_multiple:
            max_len = ((max_len + self.pad_to_multiple - 1)
                       // self.pad_to_multiple * self.pad_to_multiple)
        B = len(features)

        inp  = np.full((B, max_len), self.pad_token_id, dtype=np.int64)
        lbl  = np.full((B, max_len), -100,              dtype=np.int64)
        pos  = np.zeros((B, max_len),                   dtype=np.int64)
        mask = np.zeros((B, max_len),                   dtype=np.int64)
        wts  = np.zeros((B, max_len),                   dtype=np.float32)

        for i, f in enumerate(features):
            L = f["input_ids"].shape[0]
            inp[i,  :L] = f["input_ids"]
            lbl[i,  :L] = f["labels"]
            pos[i,  :L] = f["position_ids"]
            mask[i, :L] = 1
            wts[i,  :L] = f["token_weights"]

        return {
            "input_ids":      torch.from_numpy(inp),
            "attention_mask": torch.from_numpy(mask),
            "labels":         torch.from_numpy(lbl),
            "position_ids":   torch.from_numpy(pos),
            "token_weights":  torch.from_numpy(wts),
        }


# ---------------------------------------------------------------------------
# Weighted trainer: per-token weighted cross-entropy
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Applies per-token weights to the CE loss.
    For uniform weights (all 1.0) behaviour is identical to standard Trainer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Our compute_loss does NOT use num_items_in_batch for normalization.
        # The base Trainer.__init__ sets self.model_accepts_loss_kwargs=True because
        # Qwen2's forward has **kwargs. We override it to False so HF Trainer will:
        #   1. Not compute num_items_in_batch (saves memory/compute)
        #   2. Divide loss by gradient_accumulation_steps before backward,
        #      so gradients are properly averaged (not summed) across micro-batches.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        token_weights = inputs.pop("token_weights", None)   # (B, T) float32, 0=pad
        labels        = inputs.pop("labels")

        # Always use chunked CE to avoid materialising full logits (B, T, V)
        # ~10 GB at seq_len=32768, vocab=152k. Liger fused_linear_cross_entropy
        # is disabled because it is incompatible with sequence packing.
        import torch.nn.functional as F
        CHUNK = 512  # peak extra mem ≈ CHUNK × vocab × 2B ≈ 148 MB per chunk

        hf_model     = model.module if hasattr(model, "module") else model
        _rank0 = (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)

        # Check and REPAIR lm_head weight NaN/Inf BEFORE forward pass (all ranks)
        lm_w = hf_model.lm_head.weight
        lm_nan = lm_w.isnan().any().item() or lm_w.isinf().any().item()
        if lm_nan:
            with torch.no_grad():
                lm_w.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            if _rank0:
                print(f"[REPAIR] micro={getattr(self,'_debug_count',0)} "
                      f"Repaired NaN/Inf in lm_head.weight", flush=True)
        if _rank0 and (lm_nan or (not hasattr(self, '_debug_count') or self._debug_count < 20)):
            print(f"[WEIGHT CHECK] micro={getattr(self,'_debug_count',0)} "
                  f"lm_head.weight: nan={lm_nan} "
                  f"min={lm_w.float().min().item():.3e} "
                  f"max={lm_w.float().max().item():.3e} "
                  f"dtype={lm_w.dtype}", flush=True)

        # ── Per-layer NaN hooks (register once, fire on first NaN) ─────────────
        if not hasattr(self, '_layer_hooks_registered'):
            self._layer_hooks_registered = True
            self._nan_detected_layer = -1
            self._layer_hooks = []
            layers = hf_model.model.layers

            def make_hook(layer_idx, n_layers):
                def hook(module, inp, out):
                    if self._nan_detected_layer >= 0:
                        return  # already found, skip
                    tensor_out = out[0] if isinstance(out, tuple) else out
                    if not torch.is_tensor(tensor_out):
                        return
                    if tensor_out.isnan().any().item() or tensor_out.isinf().any().item():
                        self._nan_detected_layer = layer_idx
                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                            of = tensor_out.float()
                            inp0 = inp[0] if isinstance(inp, tuple) and torch.is_tensor(inp[0]) else None
                            inp_nan = (inp0.isnan().any().item() or inp0.isinf().any().item()) if inp0 is not None else '?'
                            print(f"[NaN LAYER] First NaN/Inf at layer {layer_idx}/{n_layers-1} | "
                                  f"out: min={of.min():.3e} max={of.max():.3e} mean={of.mean():.3e} | "
                                  f"input_nan={inp_nan} | shape={tuple(tensor_out.shape)}", flush=True)
                return hook

            for i, layer in enumerate(layers):
                h = layer.register_forward_hook(make_hook(i, len(layers)))
                self._layer_hooks.append(h)
            if _rank0:
                print(f"[INFO] Registered NaN detection hooks on {len(layers)} transformer layers", flush=True)

        # Log input_ids stats before forward to detect extreme inputs
        if _rank0 and getattr(self, '_debug_count', 0) < 5:
            ids = inputs["input_ids"]
            print(f"[INPUT] micro={getattr(self, '_debug_count', 0)} input_ids: shape={tuple(ids.shape)} "
                  f"min={ids.min().item()} max={ids.max().item()} "
                  f"attn_mask_sum={inputs['attention_mask'].sum().item()}", flush=True)

        position_ids = inputs.get("position_ids")
        transformer_out = hf_model.model(
            input_ids      = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            position_ids   = position_ids,
        )
        hidden  = transformer_out.last_hidden_state          # (B, T, H)
        lm_head = hf_model.lm_head

        shift_hidden  = hidden[..., :-1, :].contiguous()    # (B, T-1, H)
        shift_labels  = labels[..., 1:].contiguous()        # (B, T-1)

        # Uniform weight = 1 for all non-padding positions when train_weight absent
        if token_weights is not None:
            shift_weights = token_weights[..., 1:].contiguous().to(hidden.device)
        else:
            shift_weights = (shift_labels != -100).float()

        B, T, H = shift_hidden.shape
        # Use float32 accumulators to avoid bfloat16 precision loss when summing
        # thousands of per-token losses (bf16 has only ~3 decimal digits precision)
        total_loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
        total_w    = torch.zeros((), device=hidden.device, dtype=torch.float32)

        # FP32 lm_head matmul: hidden states can reach |h|~274 in BF16 models,
        # causing h @ W > 65504 (BF16 overflow → inf logits → NaN CE loss).
        # Running the matmul in FP32 avoids this overflow entirely.
        # Gradient: FP32 grad flows back via ToCopyBackward to BF16 lm_head.weight.
        lm_head_w = lm_head.weight.float()
        lm_head_b = lm_head.bias.float() if lm_head.bias is not None else None

        for start in range(0, T, CHUNK):
            end          = min(start + CHUNK, T)
            chunk_h      = shift_hidden[:, start:end, :]    # (B, chunk, H)  bf16
            chunk_labels = shift_labels[:, start:end]       # (B, chunk)
            chunk_w      = shift_weights[:, start:end]      # (B, chunk)

            chunk_logits = F.linear(chunk_h.float(), lm_head_w, lm_head_b)  # FP32
            V = chunk_logits.shape[-1]

            # Diagnose: log if logits themselves are NaN/Inf despite finite inputs
            if _rank0 and (chunk_logits.isnan().any().item() or chunk_logits.isinf().any().item()):
                print(f"[NaN LOGITS] micro={getattr(self,'_debug_count',0)} "
                      f"chunk={start}:{end} "
                      f"chunk_h: nan={chunk_h.isnan().any().item()} "
                      f"min={chunk_h.float().min().item():.3e} max={chunk_h.float().max().item():.3e} "
                      f"lm_w: nan={lm_head_w.isnan().any().item()} "
                      f"min={lm_head_w.min().item():.3e} max={lm_head_w.max().item():.3e}",
                      flush=True)

            chunk_ce = F.cross_entropy(
                chunk_logits.view(-1, V),
                chunk_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(B, -1)                                   # (B, chunk) float32

            valid_mask  = (chunk_labels != -100)            # (B, chunk) bool
            # Use torch.where instead of multiplication to avoid Inf*0=NaN (IEEE 754):
            # chunk_ce could be Inf at ignored positions; multiplying by 0 would give NaN.
            chunk_contribution = torch.where(valid_mask, chunk_ce * chunk_w,
                                             torch.zeros_like(chunk_ce))
            total_loss += chunk_contribution.sum()
            total_w    += chunk_w[valid_mask].sum()
            del chunk_logits, chunk_ce, chunk_contribution

        # ── NaN / debug tracking ────────────────────────────────────────────
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        _rank0 = (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)

        # Always check for NaN/Inf in key tensors and report immediately
        _nan_hidden   = hidden.isnan().any().item() or hidden.isinf().any().item()
        _nan_loss     = total_loss.isnan().item() or total_loss.isinf().item()
        _nan_w        = total_w.isnan().item()  or total_w.isinf().item()
        _nan_weights  = (shift_weights.isnan().any().item() or shift_weights.isinf().any().item()
                         if token_weights is not None else False)

        if _rank0 and (_nan_hidden or _nan_loss or _nan_w or _nan_weights):
            print(f"[NaN ALERT] step={self._debug_count} "
                  f"hidden_nan={_nan_hidden} total_loss={total_loss.item()} "
                  f"total_w={total_w.item()} weights_nan={_nan_weights} "
                  f"B={B} T={T} hidden.dtype={hidden.dtype}", flush=True)
            # Extra: check hidden stats (min/max to catch Inf)
            h_f = hidden.float()
            print(f"[NaN ALERT] hidden: min={h_f.min().item():.3e} max={h_f.max().item():.3e} "
                  f"mean={h_f.mean().item():.3e}", flush=True)

        # Verbose for first 20 micro-batches (enough to cover first global step + a few)
        if _rank0 and self._debug_count < 20:
            tw = total_w.item()
            tl = total_loss.item()
            ratio = tl / tw if tw > 0 else float('inf')
            print(f"[DEBUG] micro={self._debug_count} total_loss={tl:.4f} total_w={tw:.4f} "
                  f"loss={ratio:.4f} B={B} T={T} "
                  f"valid_frac={(shift_labels != -100).float().mean().item():.3f} "
                  f"w_mean={shift_weights.float().mean().item():.4f} "
                  f"hidden_dtype={hidden.dtype}", flush=True)
        self._debug_count += 1
        # ── end debug ────────────────────────────────────────────────────────

        # Safe division: avoid gradient explosion (1/1e-8 = 1e8) when total_w≈0
        if total_w.item() > 0:
            loss = total_loss / total_w
        else:
            loss = total_loss * 0.0  # zero loss, zero gradient for all-padding batches

        # NaN guard: replace NaN/Inf loss with a ZERO loss connected to the model graph.
        #
        # Why not a disconnected leaf tensor:
        #   torch.zeros(..., requires_grad=True) breaks DeepSpeed ZeRO-2 bucket tracking
        #   → IndexError: list index out of range in reduce_ipg_grads.
        #
        # Why (shift_hidden + lm_head_w) * 0.0 works:
        #   • shift_hidden is connected to ALL transformer params (embed, attn, MLP, norm)
        #   • lm_head_w is connected to lm_head.weight
        #   → backward fires gradient hooks for EVERY parameter (all ZeRO-2 buckets filled)
        #   • Gradient of (anything * 0.0) = 0 for all params → zero weight update
        #   • nan_to_num handles the case where shift_hidden or lm_head_w are themselves NaN
        if not loss.isfinite():
            if _rank0:
                print(f"[NaN SKIP] micro={self._debug_count - 1} "
                      f"total_loss={total_loss.item()} total_w={total_w.item()} "
                      f"nan_layer={getattr(self,'_nan_detected_layer',-1)} "
                      f"— replacing with graph-connected zero loss", flush=True)
            # nan_to_num ensures no NaN/Inf propagates into sum() if activations are NaN
            h_safe  = shift_hidden.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            lm_safe = lm_head_w.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            loss = (h_safe.float().sum() + lm_safe.sum()) * 0.0

        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Fully override to avoid HF's `logits = outputs[1:]` which crashes when
        # compute_loss returns (loss, None) from the chunked weighted CE path.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
            loss = loss.mean().detach()
        return (loss, None, None)

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Completely custom save: model weights only, rank 0 only.
        Never calls super() to avoid DeepSpeed saving optimizer states
        (which would require ~45 GB of CPU RAM and cause SIGKILL OOM).
        """
        import gc, os
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        gc.collect()
        torch.cuda.empty_cache()

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Sync all ranks before any I/O
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Only rank 0 writes — other ranks just wait at the barrier above
        if self.args.process_index == 0:
            os.makedirs(output_dir, exist_ok=True)

            # Unwrap DeepSpeedEngine → underlying HF model
            hf_model = self.accelerator.unwrap_model(self.model)

            # safe_serialization=True: safetensors streams GPU tensors to disk
            # one at a time (~1 GB peak CPU RAM, not 15 GB full state_dict copy)
            hf_model.save_pretrained(output_dir, safe_serialization=True)

            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)

            # trainer_state.json is required by resume_from_checkpoint
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # All ranks wait for rank 0 to finish before continuing training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Rotate old checkpoints (rank 0 only)
        if self.args.process_index == 0:
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)


# ---------------------------------------------------------------------------
# Step-1 checkpoint callback: save once at step 1 to verify saving works,
# then let the normal save_steps schedule take over.
# ---------------------------------------------------------------------------

class SaveAtStep1Callback(TrainerCallback):
    """Saves a checkpoint after the very first optimizer step."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_save = True
        return control


# ---------------------------------------------------------------------------
# Loss plot callback (same as train_lora.py)
# ---------------------------------------------------------------------------

class LossPlotCallback(TrainerCallback):
    def __init__(self, output_dir: str, train_dataset, data_collator, n_init_samples: int = 8):
        self.output_dir    = output_dir
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.n_init_samples= n_init_samples
        os.makedirs(output_dir, exist_ok=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Disabled: ZeRO-3 requires all ranks to participate in forward pass simultaneously.
        # Single-rank forward in on_train_begin causes NCCL deadlock.
        pass

    def on_evaluate(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self._save_plot(state)

    def on_log(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self._save_plot(state)

    def _save_plot(self, state):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            train_steps, train_losses = [], []
            eval_steps,  eval_losses  = [], []
            for entry in state.log_history:
                if "loss"      in entry: train_steps.append(entry["step"]); train_losses.append(entry["loss"])
                if "eval_loss" in entry: eval_steps.append(entry["step"]);  eval_losses.append(entry["eval_loss"])

            if not train_steps:
                return

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(train_steps, train_losses, label="Train Loss", alpha=0.8)
            if eval_steps:
                ax.plot(eval_steps, eval_losses, "r-o", label="AIME Eval Loss", markersize=5, linewidth=2)
            ax.set_xlabel("Steps"); ax.set_ylabel("Loss")
            ax.set_title("Training Loss vs AIME Eval Loss (Full Fine-tune)")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "loss_curve.png"), dpi=150)
            plt.close()

            csv_path = os.path.join(self.output_dir, "loss_log.csv")
            t_dict = dict(zip(train_steps, train_losses))
            e_dict = dict(zip(eval_steps,  eval_losses))
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "train_loss", "eval_loss"])
                for s in sorted(set(train_steps) | set(eval_steps)):
                    w.writerow([s, t_dict.get(s, ""), e_dict.get(s, "")])
        except Exception as e:
            print(f"[WARN] plot/csv save failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Full-parameter SFT with packing, ZeRO-3, Flash Attn, Liger")
    ap.add_argument("--train_data",  required=True)
    ap.add_argument("--model_path",  required=True)
    ap.add_argument("--output_dir",  required=True)
    ap.add_argument("--sft_config",  default=None)
    ap.add_argument("--deepspeed",   default=None, help="Path to ZeRO-3 json. Auto-detected if omitted.")
    ap.add_argument("--lr",          type=float, default=None)
    ap.add_argument("--epochs",      type=int,   default=None)
    ap.add_argument("--batch_size",  type=int,   default=None)
    ap.add_argument("--grad_accum",  type=int,   default=None)
    ap.add_argument("--max_length",  type=int,   default=None)
    ap.add_argument("--aime_data",   type=str,   default=None)
    ap.add_argument("--eval_steps",  type=int,   default=10)
    ap.add_argument("--no_liger",    action="store_true", help="Disable Liger kernels")
    ap.add_argument("--local_rank",  type=int, default=-1, help="Local rank (injected by DeepSpeed/torchrun)")
    ap.add_argument("--fp32",        action="store_true", help="Train in fp32 (avoids bf16 NaN for non-math models)")
    args = ap.parse_args()

    configs_dir = Path(__file__).parent / "configs"

    # -----------------------------------------------------------------------
    # 1. Liger kernels (must be applied BEFORE model load)
    # -----------------------------------------------------------------------
    if not args.no_liger:
        apply_liger("qwen2")

    # -----------------------------------------------------------------------
    # 2. Config
    # -----------------------------------------------------------------------
    sft_cfg = load_yaml(args.sft_config or str(configs_dir / "sft_full.yaml"))
    if args.lr:         sft_cfg["learning_rate"]              = args.lr
    if args.epochs:     sft_cfg["num_train_epochs"]            = args.epochs
    if args.batch_size: sft_cfg["per_device_train_batch_size"] = args.batch_size
    if args.grad_accum: sft_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.max_length: sft_cfg["max_seq_length"]              = args.max_length

    max_seq_length = sft_cfg.get("max_seq_length", 16384)
    ds_path = args.deepspeed or str(configs_dir / "zero2_bf16.json")

    print(f"[INFO] train_data  : {args.train_data}")
    print(f"[INFO] model_path  : {args.model_path}")
    print(f"[INFO] output_dir  : {args.output_dir}")
    print(f"[INFO] max_seq_len : {max_seq_length}")
    print(f"[INFO] deepspeed   : {ds_path}")
    print(f"[INFO] sft_cfg     : {sft_cfg}")

    # -----------------------------------------------------------------------
    # 3. Tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -----------------------------------------------------------------------
    # 4. Dataset
    # -----------------------------------------------------------------------
    raw      = load_train_jsonl(args.train_data)
    dataset  = PackedSFTDataset(raw, tokenizer, max_seq_length)
    print(f"[INFO] packed bins  : {len(dataset)}")

    aime_path    = args.aime_data or "./data/benchmarks/AIME.json"
    eval_dataset = None
    if os.path.exists(aime_path):
        eval_dataset = AIMEEvalDataset(aime_path, tokenizer)
        print(f"[INFO] AIME eval set: {len(eval_dataset)} problems")

    # -----------------------------------------------------------------------
    # 5. Model — full parameters, Flash Attention 2, bf16
    #    ZeRO-3: do NOT use device_map, let DeepSpeed shard params across GPUs
    # -----------------------------------------------------------------------
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("[WARN] flash_attn not installed, falling back to sdpa")
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        # No device_map — ZeRO-3 handles placement
    )
    print(f"[INFO] attn_implementation = {attn_impl}")
    model.enable_input_require_grads()

    # -----------------------------------------------------------------------
    # 5b. NaN gradient safety hooks (registered BEFORE DeepSpeed wrapping)
    #
    # BF16 attention-score overflow (Q·K^T > 65504) produces NaN hidden states
    # → NaN loss → NaN gradients. These hooks fire on each parameter gradient
    # BEFORE DeepSpeed's allreduce hook, converting any NaN/Inf gradient to zero.
    # Net effect: the offending micro-batch contributes zero gradient (skipped).
    # Must be registered before Trainer.__init__ so our hooks fire first.
    # -----------------------------------------------------------------------
    _nan_grad_skips = [0]   # mutable counter visible inside closure

    def _make_nan_guard_hook(param_name):
        def _hook(grad):
            if grad is None:
                return grad
            if not grad.isfinite().all():
                _nan_grad_skips[0] += 1
                return torch.zeros_like(grad)
            return grad
        return _hook

    n_hooks = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.register_hook(_make_nan_guard_hook(name))
            n_hooks += 1
    print(f"[INFO] Registered NaN→0 gradient hooks on {n_hooks} parameters "
          f"(before DeepSpeed wrapping)", flush=True)

    # -----------------------------------------------------------------------
    # 6. Training arguments
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=sft_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=sft_cfg.get("learning_rate", 1e-5),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.05),
        bf16=not args.fp32,
        fp16=False,
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=sft_cfg.get("save_strategy", "steps"),
        save_steps=sft_cfg.get("save_steps", 100),
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        save_only_model=sft_cfg.get("save_only_model", True),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
        report_to="none",
        logging_nan_inf_filter=False,  # Don't silently replace NaN/Inf with old loss — show it clearly
        deepspeed=ds_path,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_on_start=False,  # Disabled: causes NCCL deadlock with ZeRO-3
        per_device_eval_batch_size=1,
    )

    # -----------------------------------------------------------------------
    # 7. Collator + Trainer
    # -----------------------------------------------------------------------
    data_collator = PackedCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=64,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[SaveAtStep1Callback(), LossPlotCallback(args.output_dir, dataset, data_collator)],
    )

    # Resume from the latest *complete* checkpoint (must have trainer_state.json)
    import glob
    ckpt_dirs = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
    resume = None
    for d in reversed(ckpt_dirs):
        if os.path.exists(os.path.join(d, "trainer_state.json")):
            resume = d
            break
        else:
            print(f"[WARN] skipping incomplete checkpoint (no trainer_state.json): {d}")
    if resume:
        print(f"[INFO] resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # -----------------------------------------------------------------------
    # 8. Save final model
    # -----------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[OK] model saved → {final_dir}")


if __name__ == "__main__":
    main()
