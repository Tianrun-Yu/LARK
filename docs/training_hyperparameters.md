# Full-Parameter SFT on a 7B Student — Hyperparameters & Software Stack

This document records the exact training recipe used in the LARK paper for
the **full-parameter** supervised fine-tuning of a 7B student model
(e.g. Qwen-2.5-7B) on a single 8 × NVIDIA A100 (80 GB) node, together with
the acceleration libraries the training script depends on.  Config files
are shipped under `lark/train/configs/`.

## 1. Hyperparameters

### 1.1 Headline setting (5,000-trajectory experiments)

Used for the main results in the paper.  Defined in
[`lark/train/configs/sft_full.yaml`](../lark/train/configs/sft_full.yaml).

| Hyperparameter                  | Value           | Notes |
|---------------------------------|-----------------|-------|
| Learning rate                   | `1.0e-5`        | Cosine decay |
| Number of epochs                | `5`             | |
| Per-device train batch size     | `1`             | One packed sequence per micro-step |
| Gradient accumulation steps     | `16`            | Override at launch time as needed |
| **Effective global batch size** | **`64`**        | `nproc_per_node × per_device × grad_accum` |
| Maximum sequence length         | `32768`         | Long-CoT trajectories |
| LR scheduler                    | `cosine`        | |
| Warm-up ratio                   | `0.05`          | Linear warm-up over the first 5 % of steps |
| Optimizer                       | `AdamW (torch)` | HF Trainer default; `(β₁,β₂,ε)=(0.9, 0.999, 1e-8)` |
| Weight decay                    | `0.0`           | HF Trainer default |
| Gradient clipping (`max_grad_norm`) | `1.0`       | Delegated to DeepSpeed via `"gradient_clipping": "auto"` |
| Mixed precision                 | `bf16`          | `fp16` is never used |
| Logging steps                   | `10`            | |
| Save strategy                   | `steps`         | every 50 steps, keep last 1 |
| Save only model                 | `true`          | optimizer states are not persisted |

> **Launching with 8 GPUs:**  if you run with `--nproc_per_node=8`, override
> `--grad_accum 8` so that the effective global batch stays at 64
> (8 × 1 × 8 = 64).  The default value `16` in the YAML targets a 4-GPU
> setup (4 × 1 × 16 = 64).

### 1.2 Ablation setting (500-trajectory runs)

Used in the κ-verification experiments (Appendix).  Defined in
[`lark/train/configs/sft_full_500.yaml`](../lark/train/configs/sft_full_500.yaml).
Identical to §1.1 except:

| Hyperparameter              | Value     | Notes |
|-----------------------------|-----------|-------|
| Learning rate               | `5.0e-6`  | Linearly scaled down for the smaller dataset |
| Gradient accumulation steps | `8`       | Designed for `--nproc_per_node=8` → effective batch 64 |
| Logging steps               | `5`       | More frequent logs on the shorter run |

### 1.3 Numerical-stability tweak

`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`
is set in `train_full.py` before model load.  This forces fp32 accumulation
inside bf16 tensor-core matmuls on Ampere/Hopper GPUs and suppresses
occasional NaN gradients in the LM-head backward pass during long-CoT
training (no measurable speed cost, no extra memory).

## 2. Acceleration packages

Concrete versions are pinned in [`requirements.txt`](../requirements.txt).

| Layer                  | Library                       | What we use it for |
|------------------------|-------------------------------|--------------------|
| Compiler / runtime     | **PyTorch 2.9** (CUDA 12.8)   | autograd, bf16 tensor cores |
| Distributed training   | **DeepSpeed** (ZeRO Stage 2)  | optimizer-state + gradient sharding across 8 GPUs, fp32 communication for allreduce |
| Model hub & Trainer    | **Transformers 4.57** (HuggingFace) | `Trainer` loop, tokenizer, `AutoModelForCausalLM` |
| Attention              | **Flash-Attention 2.8**       | `attn_implementation="flash_attention_2"`; auto-falls back to `sdpa` if not installed |
| Fused element-wise ops | **Liger-Kernel**              | fused RoPE, RMSNorm, SwiGLU (fused linear-CE is **disabled** because it is incompatible with sequence packing) |
| Memory                 | **Gradient checkpointing**    | `use_reentrant=False`, non-reentrant variant |
| Throughput             | **Sequence packing**          | custom `PackedCollator` (greedy bin-packing + per-sequence `position_ids` reset) — packs many short trajectories into one packed sequence per micro-batch |
| Distributed launcher   | **`torch.distributed.run` / `torchrun`** | `--nproc_per_node=8` |

### Single-line install (matches `requirements.txt`)

```bash
pip install "torch>=2.4" "transformers>=4.45" "accelerate>=0.30" \
            "deepspeed>=0.14" "peft>=0.12" "liger-kernel>=0.5" \
            "flash-attn>=2.5"
```

## 3. Launch command (full-parameter SFT, 8 × A100)

```bash
torchrun --nproc_per_node=8 -m lark.train.train_full \
    --train_data  data/selections/chi2_B3/train.jsonl \
    --model_path  Qwen/Qwen2.5-7B \
    --output_dir  checkpoints/qwen2.5-7b/chi2_B3 \
    --sft_config  lark/train/configs/sft_full.yaml \
    --deepspeed   lark/train/configs/zero2_bf16.json \
    --grad_accum  8
```

With this configuration:

* Peak GPU memory per device:  ≈ 65–75 GB (well within 80 GB).
* Wall-clock on 5,000 trajectories, 5 epochs:  ≈ 3–5 hours on 8 × A100 80 GB.
* Approx. 391 optimizer steps (`5000 / 64 × 5`).

## 4. DeepSpeed config (ZeRO-2, bf16)

Shipped as
[`lark/train/configs/zero2_bf16.json`](../lark/train/configs/zero2_bf16.json):

```json
{
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "reduce_scatter": true,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8
  },
  "bf16":  { "enabled": true },
  "fp16":  { "enabled": false },
  "communication_data_type": "fp32",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

ZeRO Stage 2 is sufficient for a 7B model on 8 × A100 80 GB
(optimizer + gradients sharded, parameters replicated).  ZeRO Stage 3 is
also shipped (`zero3.json`) for use with smaller VRAM budgets or larger
backbones.
