#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LARK score (g_hat) — combined Q1+Q2 forward-pass implementation.

For every (problem, candidate trajectory) pair, this script does a single
forward pass through the reference student model and records two per-
trajectory statistics:

    L_k       = mean per-token cross-entropy on the assistant tokens
    Brier_k   = mean per-token Brier score on the assistant tokens
              = (1/T) sum_t ||one_hot(y_t) - pi_t||^2          (top-K approx.)

It then derives the forward-pass learnability estimator

    rho_k = Brier_k / L_k
    g_hat_k = ( L_k / sum_i L_i ) * ( 2*rho_k - sum_i rho_i*L_i / sum_i L_i )

per problem.  These are exactly the quantities used in the LARK closed-form
selection rules in `lark/selection/select_topb.py`, `select_chi2.py`, and
`select_kl.py`.

Input format
------------
A JSONL file where each line is one (problem, candidate trajectory) record:

    {
      "problem_id":   int,             # index of the problem within the corpus
      "candidate_id": int,             # 0..K-1 within the same problem_id
      "teacher":      "deepseek-r1",   # free-form label (used in output only)
      "messages":     [                # chat-style trajectory
          {"role": "system", "content": ...},
          {"role": "user",   "content": ...},
          {"role": "assistant", "content": ...}
      ]
    }

Output format
-------------
A JSON file whose `scores` field is a per-problem dict:

    {
      "method": "lark_ghat",
      "student_model": "...",
      "top_k": 50,
      "scores": {
          "0": [{"candidate_id": 0, "teacher": "...",
                 "L": 1.23, "Brier": 0.45, "rho": 0.37, "g_hat": 0.012}, ...],
          "1": [ ... ],
          ...
      }
    }

Quick start
-----------
    python -m lark.score_lark \\
        --input         data/candidates/pool.jsonl \\
        --output        out/lark_scores.json \\
        --student_model Qwen/Qwen2.5-7B \\
        --top_k         50 \\
        --batch_size    1 \\
        --max_length    32768
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------------------------------------------------------------- #
# Tokenisation                                                                #
# --------------------------------------------------------------------------- #

def tokenise_record(record: Dict, tokenizer, max_length: int):
    """Returns (input_ids, asst_start, asst_end) where asst_[start, end) is the
    assistant-token range we score on. None if the assistant turn is empty
    after truncation."""
    messages = record["messages"]

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        [m for m in messages if m["role"] != "assistant"],
        tokenize=False, add_generation_prompt=True,
    )

    full_ids   = tokenizer(full_text,   truncation=True,
                           max_length=max_length).input_ids
    prompt_ids = tokenizer(prompt_text, truncation=True,
                           max_length=max_length).input_ids

    if len(prompt_ids) >= len(full_ids) - 1:
        return None
    return full_ids, len(prompt_ids), len(full_ids)


# --------------------------------------------------------------------------- #
# Brier + cross-entropy on the assistant tokens                               #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def score_one(model, input_ids, asst_start, asst_end, top_k, device):
    """Single forward pass; returns (L_mean, Brier_mean) over the assistant
    tokens.  Brier is computed with a top-K renormalised softmax to avoid an
    [T, V] tensor on long contexts."""
    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    out = model(input_ids=ids)
    logits = out.logits[0, :-1, :]               # (T-1, V)
    labels = ids[0, 1:]                          # (T-1,)

    # The next-token prediction at position t-1 corresponds to token t.
    # Assistant span [asst_start, asst_end) ⇒ logits index [asst_start-1, asst_end-1).
    sl = asst_start - 1
    el = asst_end   - 1
    if el <= sl:
        return None

    logits_a = logits[sl:el].float()             # (T_a, V)
    labels_a = labels[sl:el]                     # (T_a,)

    # Cross-entropy (log_softmax over full vocab is exact, no approximation).
    L = F.cross_entropy(logits_a, labels_a, reduction="mean").item()

    # Brier: keep top-K logits, renormalise, compute ||e_y - pi||^2.
    K = min(top_k, logits_a.shape[-1])
    topk_logits, topk_idx = logits_a.topk(K, dim=-1)            # (T_a, K)
    topk_p = F.softmax(topk_logits, dim=-1)                     # (T_a, K)

    # p_true: probability the renormalised top-K assigns to y_t (0 if y_t ∉ top-K)
    is_true = (topk_idx == labels_a.unsqueeze(-1))              # (T_a, K) bool
    p_true  = (topk_p * is_true).sum(dim=-1)                    # (T_a,)
    p_sq    = (topk_p * topk_p).sum(dim=-1)                     # (T_a,)
    brier_t = 1.0 - 2.0 * p_true + p_sq                         # (T_a,)
    Brier   = brier_t.mean().item()

    return L, Brier


# --------------------------------------------------------------------------- #
# g_hat aggregation                                                           #
# --------------------------------------------------------------------------- #

def compute_g_hat(per_problem_rows: List[Dict]) -> List[Dict]:
    """Given the L/Brier pair for every candidate of one problem, fill in
    rho and g_hat in-place and return the (now complete) row list."""
    Ls   = [r["L"]     for r in per_problem_rows]
    Bs   = [r["Brier"] for r in per_problem_rows]
    rhos = [b / l if l > 1e-30 else 0.0 for b, l in zip(Bs, Ls)]

    sum_L      = sum(Ls)
    sum_rho_L  = sum(r * l for r, l in zip(rhos, Ls))

    for r, L, rho in zip(per_problem_rows, Ls, rhos):
        r["rho"]   = rho
        if sum_L > 1e-30:
            r["g_hat"] = (L / sum_L) * (2.0 * rho - sum_rho_L / sum_L)
        else:
            r["g_hat"] = float("nan")
    return per_problem_rows


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",         required=True,
                   help="JSONL file of {problem_id, candidate_id, teacher, messages}")
    p.add_argument("--output",        required=True,
                   help="Where to write the lark_scores JSON.")
    p.add_argument("--student_model", required=True,
                   help="HuggingFace model id or local path of the student.")
    p.add_argument("--top_k",         type=int, default=50)
    p.add_argument("--max_length",    type=int, default=32768)
    p.add_argument("--batch_size",    type=int, default=1,
                   help="Reserved; current implementation runs one record at a time.")
    p.add_argument("--device",        default="cuda")
    p.add_argument("--dtype",         default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16,
             "float16":  torch.float16,
             "float32":  torch.float32}[args.dtype]
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[lark] loading {args.student_model} on {device} ({args.dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, dtype=dtype, device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Stream the input, scoring one record at a time and grouping by problem.
    rows_by_problem: Dict[int, List[Dict]] = defaultdict(list)
    n_lines = n_scored = n_skipped = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            rec = json.loads(line)
            t   = tokenise_record(rec, tokenizer, args.max_length)
            if t is None:
                n_skipped += 1
                continue
            input_ids, asst_start, asst_end = t
            res = score_one(model, input_ids, asst_start, asst_end,
                            args.top_k, device)
            if res is None:
                n_skipped += 1
                continue
            L, B = res
            rows_by_problem[rec["problem_id"]].append({
                "candidate_id": rec.get("candidate_id", len(rows_by_problem[rec["problem_id"]])),
                "teacher":      rec.get("teacher", ""),
                "L":            L,
                "Brier":        B,
            })
            n_scored += 1
            if n_scored % 25 == 0:
                print(f"  scored {n_scored} / {n_lines}", flush=True)

    print(f"[lark] {n_scored} scored, {n_skipped} skipped, "
          f"{len(rows_by_problem)} problems", flush=True)

    # Compute g_hat per problem
    for pid, rows in rows_by_problem.items():
        compute_g_hat(rows)

    out = {
        "method":        "lark_ghat",
        "student_model": args.student_model,
        "top_k":         args.top_k,
        "n_problems":    len(rows_by_problem),
        "n_scored":      n_scored,
        "n_skipped":     n_skipped,
        "scores":        {str(pid): rows
                          for pid, rows in sorted(rows_by_problem.items())},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".",
                exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[lark] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
