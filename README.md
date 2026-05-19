# LARK: Learnability-grounded Anchor-time Ranking

Official implementation of the LARK selection rule for weighted supervised
fine-tuning over multi-teacher candidate pools.

LARK estimates a per-trajectory *learnability score* $\hat g_k$ from a single
forward pass through the reference student model and then assigns a weighted
SFT loss over the top-$B$ trajectories via a closed-form $\chi^2$-tempered
rule.

## Repository layout

```
LARK/
├── lark/
│   ├── score_lark.py            # forward-pass g_hat computation (Q1+Q2 in one pass)
│   ├── selection/
│   │   └── select_chi2.py       # chi^2-tempered LARK closed-form
│   ├── train/
│   │   ├── train_full.py        # full-parameter SFT (DeepSpeed ZeRO-2)
│   │   ├── train_lora.py        # LoRA SFT (single GPU)
│   │   └── configs/             # SFT + DeepSpeed YAML/JSON configs
│   └── eval/
│       ├── generate.py          # vLLM / SGLang sampler (ACC@5)
│       ├── extract_answer.py    # boxed-answer extraction
│       ├── score.py             # exact / math-verify scoring
│       └── aggregate.py         # cross-(method, benchmark) CSV table
├── scripts/
│   └── run_example.sh           # end-to-end demo on a tiny pool
├── requirements.txt
├── LICENSE                      # MIT
└── README.md
```

## Installation

```bash
git clone https://github.com/Tianrun-Yu/LARK.git
cd LARK
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with PyTorch 2.9 / CUDA 12.8 on NVIDIA A100 (80 GB).

## Data

This repository ships **code only**. You will need:

1. **Candidate trajectory pool**: a JSONL file of $(\text{problem}, \text{candidate})$
   records (one trajectory per line) in the format consumed by
   `lark/score_lark.py` (see its docstring). The paper's candidate pool is
   built from NuminaMath problems with rollouts from 11 teacher models
   (DeepSeek-R1, GPT-OSS-120B/20B, Magistral, Nemotron-Super, Phi-4-Reason+,
   Qwen3-235B/30B/8B/4B, QwQ-32B); the precise list and re-sampling protocol
   are described in the paper's Appendix.

2. **Evaluation benchmarks**: place the standard JSON files for the
   benchmarks you want to evaluate on (`AIME.json`, `AMC.json`, `CPQA.json`,
   `MATH-L1.json` ... `MATH-L5.json`) under `data/benchmarks/`. The expected
   schema is one list of `{"prompt": str, "answer": str, ...}` per file.

## End-to-end pipeline

### 1. Score every candidate with LARK ($\hat g_k$)

```bash
python -m lark.score_lark \
    --input         data/candidates/pool.jsonl \
    --output        data/scores/lark_ghat.json \
    --student_model Qwen/Qwen2.5-7B \
    --top_k         50 \
    --max_length    32768
```

This produces one $\hat g_k$ per candidate (grouped by problem) using a single
forward pass per trajectory.

### 2. Select training trajectories ($\chi^2$-tempered LARK)

```bash
python -m lark.selection.select_chi2 \
    --scores  data/scores/lark_ghat.json \
    --B       3 \
    --output  data/selections/chi2_B3/train.jsonl
```

The output JSONL has the standard chat-style record with a per-sample
`meta.train_weight` field that `train_full.py` honours.

### 3. Train

```bash
torchrun --nproc_per_node=8 -m lark.train.train_full \
    --train_data  data/selections/chi2_B3/train.jsonl \
    --model_path  Qwen/Qwen2.5-7B \
    --output_dir  checkpoints/qwen2.5-7b/chi2_B3 \
    --sft_config  lark/train/configs/sft_full.yaml \
    --deepspeed   lark/train/configs/zero2_bf16.json \
    --grad_accum  8
```

### 4. Evaluate (vLLM, ACC@5)

```bash
python -m lark.eval.generate \
    --model_path           checkpoints/qwen2.5-7b/chi2_B3/final \
    --benchmark            MATH-L5 \
    --benchmark_path       data/benchmarks/MATH-L5.json \
    --output_dir           results/eval/chi2_B3/MATH-L5 \
    --tensor_parallel_size 1 \
    --max_new_tokens       12288

python -m lark.eval.extract_answer --eval_dir results/eval/chi2_B3/MATH-L5
python -m lark.eval.score          --eval_dir results/eval/chi2_B3/MATH-L5
python -m lark.eval.aggregate      --eval_root results/eval --output_csv results/table.csv
```

See [`scripts/run_example.sh`](scripts/run_example.sh) for a self-contained
demo on a small candidate pool.

## License

MIT — see [`LICENSE`](LICENSE).
