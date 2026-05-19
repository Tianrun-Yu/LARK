#!/usr/bin/env bash
# Minimal end-to-end LARK demo.
# Assumes:
#   - candidate pool at data/candidates/pool.jsonl  (1+ problem, K>=2 candidates)
#   - benchmark file at data/benchmarks/MATH-L5.json
#   - 8 x A100 (80 GB) visible to torchrun
# Run from the repo root.

set -euo pipefail

STUDENT=${STUDENT:-Qwen/Qwen2.5-7B}
POOL=${POOL:-data/candidates/pool.jsonl}
BENCH=${BENCH:-data/benchmarks/MATH-L5.json}
B=${B:-3}
RUN=${RUN:-demo}

SCORES=data/scores/${RUN}_lark.json
SEL=data/selections/${RUN}_chi2_B${B}/train.jsonl
CKPT=checkpoints/${RUN}_chi2_B${B}
EVAL_DIR=results/eval/${RUN}_chi2_B${B}/$(basename "${BENCH}" .json)

# 1. Score every candidate with LARK
python -m lark.score_lark \
    --input         "${POOL}" \
    --output        "${SCORES}" \
    --student_model "${STUDENT}" \
    --top_k         50 \
    --max_length    32768

# 2. Select training trajectories with chi^2-tempered LARK at budget B
python -m lark.selection.select_chi2 \
    --scores  "${SCORES}" \
    --B       "${B}" \
    --output  "${SEL}"

# 3. Full-parameter SFT on 8 GPUs
torchrun --nproc_per_node=8 -m lark.train.train_full \
    --train_data  "${SEL}" \
    --model_path  "${STUDENT}" \
    --output_dir  "${CKPT}" \
    --sft_config  lark/train/configs/sft_full.yaml \
    --deepspeed   lark/train/configs/zero2_bf16.json \
    --grad_accum  8

# 4. ACC@5 evaluation
python -m lark.eval.generate \
    --model_path           "${CKPT}/final" \
    --benchmark            "$(basename "${BENCH}" .json)" \
    --benchmark_path       "${BENCH}" \
    --output_dir           "${EVAL_DIR}" \
    --tensor_parallel_size 1 \
    --max_new_tokens       12288

python -m lark.eval.extract_answer --eval_dir "${EVAL_DIR}"
python -m lark.eval.score          --eval_dir "${EVAL_DIR}"

echo "[done] eval results: ${EVAL_DIR}/summary.json"
