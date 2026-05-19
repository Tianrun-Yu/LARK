#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q2 – KL-regularized reweighting.

Reads a Q1 scores.json and for each problem computes the optimal reweighting
distribution under KL divergence from a uniform prior:

  q_n^{KL} = arg max_{q_n ∈ Δ^{K_n}}  { <q_n, e_n> - τ · KL(q_n || p_n) }

With uniform prior p_{n,k} = 1/K_n the closed-form solution simplifies to

  q_{n,k}^{KL} = softmax(e_n / τ)_k
               = exp(e_{n,k}/τ) / Σ_j exp(e_{n,j}/τ)

Temperature interpretation:
  τ → 0  :  q converges to the argmax one-hot  (hard top-1)
  τ → ∞  :  q converges to the uniform prior   (no preference)

Selection from q_n:
  --selection argmax  → always pick argmax_k q_{n,k} = argmax_k e_{n,k}
                        (deterministic; τ has no effect on selection order)
  --selection sample  → draw one k ~ q_n^{KL}  (stochastic; requires --seed)

Output:
  q2_root/<student>/<q1_baseline>/kl_tau<τ>/train.jsonl
  q2_root/<student>/<q1_baseline>/kl_tau<τ>/train.manifest.json

Usage:
  python select_kl.py \\
    --q1_root    ./data/scores \\
    --q2_root    ./data/selections \\
    --rsr_root   ./data/candidates \\
    --students   qwen2.5-7b \\
    --q1_baseline global_naturalness \\
    --tau        1.0 \\
    --selection  sample \\
    --seed       0
"""

import os
import json
import argparse
import hashlib
import random
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_sua(messages: List[Dict]) -> Tuple[str, str, str]:
    sys_c = user_c = asst_c = ""
    for m in messages:
        r = m.get("role", "")
        c = m.get("content", "")
        if r == "system":      sys_c = c
        elif r == "user":      user_c = c
        elif r == "assistant": asst_c = c
    return sys_c, user_c, asst_c


def load_json_list(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ---------------------------------------------------------------------------
# KL-reweighted distribution  q_n^{KL}
# ---------------------------------------------------------------------------

def kl_weights(energies: List[float], tau: float) -> List[float]:
    """
    Compute q_{n,k} = softmax(e_n / τ)_k with numerical stability.

    NaN entries are treated as -inf (zero weight).
    Returns a probability vector summing to 1.
    """
    clean = [e if e == e else float("-inf") for e in energies]

    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")

    # Numerically stable softmax
    scaled = [e / tau for e in clean]
    max_s  = max(scaled)
    exps   = [math.exp(s - max_s) if s != float("-inf") else 0.0 for s in scaled]
    total  = sum(exps)

    if total == 0.0:
        # All -inf (all NaN): fall back to uniform
        K = len(energies)
        return [1.0 / K] * K

    return [x / total for x in exps]


def weighted_sample(weights: List[float], rng: random.Random) -> int:
    """Draw one index according to the given probability weights."""
    r = rng.random()
    cumsum = 0.0
    for i, w in enumerate(weights):
        cumsum += w
        if r <= cumsum:
            return i
    return len(weights) - 1   # numerical fallback


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def select_kl(
    q1_root: str,
    q2_root: str,
    rsr_root: str,
    student: str,
    q1_baseline: str,
    tau: float,
    selection: str,
    seed: int,
    n_samples: int = 1,
    n_problems: int = 5000,
):
    # Format tau for directory name (avoid trailing zeros)
    tau_str     = f"{tau:g}"
    method_name = f"kl_tau{tau_str}" if n_samples == 1 else f"kl_tau{tau_str}_B{n_samples}"

    # ------------------------------------------------------------------
    # Load Q1 scores
    # ------------------------------------------------------------------
    q1_scores_path = os.path.join(q1_root, student, q1_baseline, "scores.json")
    if not os.path.isfile(q1_scores_path):
        raise FileNotFoundError(f"Q1 scores not found: {q1_scores_path}")

    with open(q1_scores_path, "r", encoding="utf-8") as f:
        q1_data = json.load(f)

    scores        = q1_data["scores"]           # [C][N]
    cand_files_q1 = q1_data["candidate_files"]  # [{fidx, teacher, path}, ...]
    C = len(scores)
    N = q1_data["n_problems"]
    n_problems = min(n_problems, N)

    print(f"[INFO] loaded Q1 scores: {C} candidates × {N} problems")
    print(f"[INFO] q1_baseline={q1_baseline}  τ={tau}  selection={selection}  n_samples={n_samples}  seed={seed}")

    # ------------------------------------------------------------------
    # Q2: compute q_n^{KL} and select B fidx per problem
    # ------------------------------------------------------------------
    rng = random.Random(seed)

    # selected[pid] = list of (fidx, score, weight) tuples, length = n_samples
    selected: List[List[tuple]] = []

    for pid in range(n_problems):
        row     = [scores[fi][pid] for fi in range(C)]
        weights = kl_weights(row, tau)
        clean_row = [s if s == s else float("-inf") for s in row]

        draws = []
        if selection == "argmax":
            fidx = max(range(C), key=lambda i: weights[i])
            draws = [(fidx, clean_row[fidx], weights[fidx])] * n_samples
        else:   # sample (with replacement)
            for _ in range(n_samples):
                fidx = weighted_sample(weights, rng)
                draws.append((fidx, clean_row[fidx], weights[fidx]))

        selected.append(draws)

    # ------------------------------------------------------------------
    # Load selected trajectories and write train.jsonl
    # ------------------------------------------------------------------
    # Build mapping: fidx -> [(pid, sample_idx, score, weight), ...]
    file_to_entries: Dict[int, List[tuple]] = defaultdict(list)
    for pid, draws in enumerate(selected):
        for sidx, (fidx, score, weight) in enumerate(draws):
            file_to_entries[fidx].append((pid, sidx, score, weight))

    # Pre-allocate output: n_problems * n_samples lines
    total_lines = n_problems * n_samples
    output_lines: List[Optional[str]] = [None] * total_lines

    for fidx, entries in file_to_entries.items():
        entry   = cand_files_q1[fidx]
        teacher = entry["teacher"]
        fp      = entry["path"]
        print(f"[LOAD] fidx={fidx:02d} {teacher} :: {os.path.basename(fp)}  ({len(entries)} draws)")

        data = load_json_list(fp)
        for (pid, sidx, score, weight) in entries:
            ex = data[pid]
            sys_c, user_c, asst_c = get_sua(ex["messages"])
            out_obj = {
                "messages": [
                    {"role": "system",    "content": sys_c},
                    {"role": "user",      "content": user_c},
                    {"role": "assistant", "content": asst_c},
                ],
                "meta": {
                    "problem_id":    pid,
                    "sample_idx":    sidx,
                    "teacher":       teacher,
                    "run_file":      os.path.basename(fp),
                    "q1_baseline":   q1_baseline,
                    "q2_method":     method_name,
                    "tau":           tau,
                    "selection":     selection,
                    "n_samples":     n_samples,
                    "seed":          seed,
                    "score":         float(score),
                    "q_weight":      float(weight),
                },
            }
            output_lines[pid * n_samples + sidx] = json.dumps(out_obj, ensure_ascii=False)
        del data

    missing = sum(1 for x in output_lines if x is None)
    if missing:
        raise RuntimeError(f"Selection incomplete: {missing} lines missing!")

    out_dir  = os.path.join(q2_root, student, q1_baseline, method_name)
    out_path = os.path.join(out_dir, "train.jsonl")
    ensure_dir(out_dir)

    with open(out_path, "w", encoding="utf-8") as w:
        for line in output_lines:
            w.write(line + "\n")

    manifest_path = os.path.join(out_dir, "train.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "q1_baseline":   q1_baseline,
            "q2_method":     method_name,
            "tau":           tau,
            "selection":     selection,
            "n_samples":     n_samples,
            "seed":          seed,
            "n_problems":    n_problems,
            "n_candidates":  C,
            "total_lines":   total_lines,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {total_lines} lines → {out_path}")
    print(f"[OK] {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Q2 KL-regularized reweighting: softmax(e/τ) distribution over candidates."
    )
    ap.add_argument("--q1_root",     type=str, required=True,
                    help="Q1 output root, e.g. ./data/scores")
    ap.add_argument("--q2_root",     type=str, required=True,
                    help="Q2 output root, e.g. ./data/selections")
    ap.add_argument("--rsr_root",    type=str, required=True)
    ap.add_argument("--students",    type=str, required=True)
    ap.add_argument("--q1_baseline", type=str, required=True,
                    help="Q1 baseline name, e.g. global_naturalness")
    ap.add_argument("--tau",         type=float, default=1.0,
                    help="KL regularization temperature τ > 0. "
                         "Small τ → hard argmax; large τ → uniform.")
    ap.add_argument("--selection",   type=str, default="sample",
                    choices=["argmax", "sample"],
                    help="How to pick one trajectory: "
                         "'argmax' returns the highest-weight candidate (τ-independent order); "
                         "'sample' draws k ~ q_n^{KL} (requires --seed).")
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--n_samples",   type=int, default=1,
                    help="Number of trajectories to sample per problem (B). "
                         "Default 1 keeps original behaviour.")
    ap.add_argument("--n_problems",  type=int, default=5000)
    args = ap.parse_args()

    students = [s.strip() for s in args.students.split(",") if s.strip()]
    for s in students:
        print(f"\n=== student: {s} ===")
        select_kl(
            q1_root=args.q1_root,
            q2_root=args.q2_root,
            rsr_root=args.rsr_root,
            student=s,
            q1_baseline=args.q1_baseline,
            tau=args.tau,
            selection=args.selection,
            seed=args.seed,
            n_samples=args.n_samples,
            n_problems=args.n_problems,
        )


if __name__ == "__main__":
    main()
