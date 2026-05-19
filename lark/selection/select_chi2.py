#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q2 – χ²-optimal deterministic top-B selection with theoretical weights.

For each problem n with K candidates ranked by energy e_{n,1} ≥ ... ≥ e_{n,K}:

  Active set  A = {1, ..., B}   (top-B by energy)
  Threshold   e_{B+1}           (energy of first excluded candidate)
  Mean        ē_A = (1/B) Σ_{k∈A} e_k

  Optimal weight:
    q*_k = (1/B) · (e_k − e_{B+1}) / (ē_A − e_{B+1})   for k ∈ A
    q*_k = 0                                              for k ∉ A

  This is the closed-form solution of:
    max_{q ∈ Δ^K}  <q, e_n>  s.t.  D_{χ²}(q || p) ≤ B/K²
  with τ_B = B(ē_A − e_{B+1}) / K determined by B (no free hyperparameter).

Output:
  q2_root/<student>/<q1_baseline>/chi2_B<B>/train.jsonl
  q2_root/<student>/<q1_baseline>/chi2_B<B>/train.manifest.json

Usage:
  python select_chi2.py \\
    --q1_root    ./data/scores \\
    --q2_root    ./data/selections \\
    --rsr_root   ./data/candidates \\
    --students   qwen2.5-7b \\
    --q1_baseline global_naturalness \\
    --B          5
"""

import os
import json
import argparse
import hashlib
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
# χ²-optimal deterministic top-B weights
# ---------------------------------------------------------------------------

def chi2_topb_weights(energies: List[float], B: int) -> List[float]:
    """
    Deterministic top-B selection with χ²-optimal weights.

    Active set A = top-B candidates by energy.
    Threshold e_{B+1} = energy of first excluded candidate.
    Mean ē_A = mean energy of the active set.

    Weight for k ∈ A:
        q*_k = (1/B) · (e_k − e_{B+1}) / (ē_A − e_{B+1})
    Weight for k ∉ A:
        q*_k = 0

    NaN energies are treated as −∞ (never selected).
    Falls back to uniform over valid candidates when degenerate.
    """
    K = len(energies)
    clean = [e if e == e else float("-inf") for e in energies]

    # Sort indices by energy descending; exclude -inf (NaN) entries
    sorted_idx = sorted(range(K), key=lambda i: clean[i], reverse=True)
    valid_idx = [i for i in sorted_idx if clean[i] != float("-inf")]

    if not valid_idx:
        return [1.0 / K] * K   # all NaN: uniform fallback

    B_eff = min(B, len(valid_idx))
    top_idx = valid_idx[:B_eff]

    # e_{B+1}: energy of the first excluded candidate
    if B_eff < len(valid_idx):
        e_threshold = clean[valid_idx[B_eff]]
    else:
        # All valid candidates selected; push threshold just below minimum
        e_threshold = clean[top_idx[-1]] - 1e-8

    e_top = [clean[i] for i in top_idx]
    e_bar_A = sum(e_top) / B_eff
    denom = e_bar_A - e_threshold

    weights = [0.0] * K
    if denom <= 1e-12:
        # Degenerate: all top-B have the same energy → uniform over active set
        for i in top_idx:
            weights[i] = 1.0 / B_eff
    else:
        for i in top_idx:
            weights[i] = (clean[i] - e_threshold) / (denom * B_eff)

    return weights


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def select_chi2(
    q1_root: str,
    q2_root: str,
    rsr_root: str,
    student: str,
    q1_baseline: str,
    B: int,
    n_problems: int = 5000,
):
    method_name = f"chi2_B{B}"

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
    print(f"[INFO] q1_baseline={q1_baseline}  B={B}  method={method_name}")

    # ------------------------------------------------------------------
    # Q2: compute χ²-optimal top-B weights for each problem
    # ------------------------------------------------------------------

    # selected[pid] = list of (fidx, score, weight) tuples, length = B
    selected: List[List[tuple]] = []

    for pid in range(n_problems):
        row     = [scores[fi][pid] for fi in range(C)]
        weights = chi2_topb_weights(row, B)
        clean_row = [s if s == s else float("-inf") for s in row]

        # Take top-B by energy directly (avoids missing entries when weight==0 due to ties)
        valid_sorted = sorted(
            [(fi, clean_row[fi]) for fi in range(C) if clean_row[fi] != float("-inf")],
            key=lambda t: t[1], reverse=True
        )[:B]
        draws = [(fi, score, weights[fi]) for fi, score in valid_sorted]

        selected.append(draws)

    # ------------------------------------------------------------------
    # Load selected trajectories and write train.jsonl
    # ------------------------------------------------------------------
    # Build mapping: fidx -> [(pid, sample_idx, score, weight), ...]
    file_to_entries: Dict[int, List[tuple]] = defaultdict(list)
    for pid, draws in enumerate(selected):
        for sidx, (fidx, score, weight) in enumerate(draws):
            file_to_entries[fidx].append((pid, sidx, score, weight))

    # Pre-allocate output: n_problems * B lines
    total_lines = n_problems * B
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
                    "B":             B,
                    "score":         float(score),
                    "train_weight":  float(weight),   # χ²-optimal per-sample weight
                },
            }
            output_lines[pid * B + sidx] = json.dumps(out_obj, ensure_ascii=False)
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
            "B":             B,
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
        description="Q2 χ²-optimal deterministic top-B selection with theoretical weights."
    )
    ap.add_argument("--q1_root",     type=str, required=True,
                    help="Q1 output root, e.g. ./data/scores")
    ap.add_argument("--q2_root",     type=str, required=True,
                    help="Q2 output root, e.g. ./data/selections")
    ap.add_argument("--rsr_root",    type=str, required=True)
    ap.add_argument("--students",    type=str, required=True)
    ap.add_argument("--q1_baseline", type=str, required=True,
                    help="Q1 baseline name, e.g. global_naturalness")
    ap.add_argument("--B",           type=int, default=5,
                    help="Number of top-B trajectories to select per problem. "
                         "τ is derived automatically from B and the data.")
    ap.add_argument("--n_problems",  type=int, default=5000)
    args = ap.parse_args()

    students = [s.strip() for s in args.students.split(",") if s.strip()]
    for s in students:
        print(f"\n=== student: {s} ===")
        select_chi2(
            q1_root=args.q1_root,
            q2_root=args.q2_root,
            rsr_root=args.rsr_root,
            student=s,
            q1_baseline=args.q1_baseline,
            B=args.B,
            n_problems=args.n_problems,
        )


if __name__ == "__main__":
    main()
