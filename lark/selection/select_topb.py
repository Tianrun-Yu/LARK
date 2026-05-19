#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Q2 – Top-B hard selection.

Reads a Q1 scores.json (full C×N score matrix) and selects, for each problem,
the top-B candidates under the energy scores. With B=1 this is the standard
argmax selection used in all previous baselines.

Formal definition (see paper §Q2):

  q_n^{top-B} = arg max_{q_n ∈ Δ^{K_n}}  <q_n, e_n>
                s.t.  ||q_n||_∞ ≤ 1/B

When energies are strictly ordered the unique solution places weight 1/B on
each of the top-B candidates and 0 on the rest.

For B=1 this collapses to the argmax one-hot vector.

Selection from q_n:
  --selection argmax  → always pick the single highest-energy candidate
                        (same as B=1; deterministic)
  --selection sample  → draw one trajectory ~ Uniform(top-B) (needs --seed)

Output:
  q2_root/<student>/<q1_baseline>/topb_B<B>/train.jsonl
  q2_root/<student>/<q1_baseline>/topb_B<B>/train.manifest.json

Usage:
  python select_topb.py \\
    --q1_root    ./data/scores \\
    --q2_root    ./data/selections \\
    --rsr_root   ./data/candidates \\
    --students   qwen2.5-7b \\
    --q1_baseline global_naturalness \\
    --B          1 \\
    --seed       0
"""

import os
import json
import glob
import argparse
import hashlib
import random
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
# Core Q2 selection
# ---------------------------------------------------------------------------

def topb_select(
    scores_row: List[float],   # e_n: scores for all C candidates for problem n
    B: int,
    selection: str,            # "argmax" or "sample"
    rng: random.Random,
) -> int:
    """
    Return the selected candidate index (fidx) for one problem.

    Top-B: keep the B highest-scoring candidates, then:
      - argmax: always return the single best (= top-1 regardless of B)
      - sample:  draw uniformly from the top-B
    NaN scores are treated as -inf (never selected unless all are NaN).
    """
    C = len(scores_row)
    # Replace NaN with -inf
    clean = [s if s == s else float("-inf") for s in scores_row]
    # Sort indices by score descending
    ranked = sorted(range(C), key=lambda i: clean[i], reverse=True)
    top_b  = ranked[:B]

    if selection == "argmax":
        return top_b[0]
    else:  # sample
        return rng.choice(top_b)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def select_topb(
    q1_root: str,
    q2_root: str,
    rsr_root: str,
    student: str,
    q1_baseline: str,
    B: int,
    selection: str,
    seed: int,
    n_problems: int = 5000,
):
    method_name = f"topb_B{B}"

    # ------------------------------------------------------------------
    # Load Q1 scores
    # ------------------------------------------------------------------
    q1_scores_path = os.path.join(q1_root, student, q1_baseline, "scores.json")
    if not os.path.isfile(q1_scores_path):
        raise FileNotFoundError(f"Q1 scores not found: {q1_scores_path}")

    with open(q1_scores_path, "r", encoding="utf-8") as f:
        q1_data = json.load(f)

    scores       = q1_data["scores"]          # [C][N]
    cand_files_q1 = q1_data["candidate_files"] # [{fidx, teacher, path}, ...]
    C = len(scores)
    N = q1_data["n_problems"]
    n_problems = min(n_problems, N)

    print(f"[INFO] loaded Q1 scores: {C} candidates × {N} problems")
    print(f"[INFO] q1_baseline={q1_baseline}  B={B}  selection={selection}  seed={seed}")

    # ------------------------------------------------------------------
    # Q2: select top-B fidx list per problem (outputs B lines per problem)
    # ------------------------------------------------------------------
    rng = random.Random(seed)

    # selected[pid] = list of (fidx, score) tuples, length = B
    selected: List[List[tuple]] = []

    for pid in range(n_problems):
        row = [scores[fi][pid] for fi in range(C)]
        clean_row = [s if s == s else float("-inf") for s in row]
        ranked = sorted(range(C), key=lambda i: clean_row[i], reverse=True)
        top_b  = ranked[:B]

        if selection == "argmax":
            # Output all top-B in score-descending order
            draws = [(fidx, clean_row[fidx]) for fidx in top_b]
        else:  # sample B times with replacement from top-B (uniform)
            draws = [(rng.choice(top_b), None) for _ in range(B)]
            draws = [(fidx, clean_row[fidx]) for fidx, _ in draws]

        selected.append(draws)

    # ------------------------------------------------------------------
    # Load selected trajectories and write train.jsonl
    # ------------------------------------------------------------------
    from collections import defaultdict
    # Build mapping: fidx -> [(pid, sample_idx, score), ...]
    file_to_entries: Dict[int, List[tuple]] = defaultdict(list)
    for pid, draws in enumerate(selected):
        for sidx, (fidx, score) in enumerate(draws):
            file_to_entries[fidx].append((pid, sidx, score))

    total_lines = n_problems * B
    output_lines: List[Optional[str]] = [None] * total_lines

    for fidx, entries in file_to_entries.items():
        entry   = cand_files_q1[fidx]
        teacher = entry["teacher"]
        fp      = entry["path"]
        print(f"[LOAD] fidx={fidx:02d} {teacher} :: {os.path.basename(fp)}  ({len(entries)} draws)")

        data = load_json_list(fp)
        for (pid, sidx, score) in entries:
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
                    "selection":     selection,
                    "seed":          seed,
                    "score":         float(score) if score is not None else float("nan"),
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
            "selection":     selection,
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
        description="Q2 Top-B hard selection: select top-B candidates from Q1 score matrix."
    )
    ap.add_argument("--q1_root",     type=str, required=True,
                    help="Q1 output root, e.g. ./data/scores")
    ap.add_argument("--q2_root",     type=str, required=True,
                    help="Q2 output root, e.g. ./data/selections")
    ap.add_argument("--rsr_root",    type=str, required=True,
                    help="RSR_data root (trajectories are loaded from here via paths in scores.json)")
    ap.add_argument("--students",    type=str, required=True,
                    help="Comma-separated student folder names")
    ap.add_argument("--q1_baseline", type=str, required=True,
                    help="Q1 baseline name, e.g. global_naturalness, rsr, random")
    ap.add_argument("--B",           type=int, default=1,
                    help="Number of top candidates to keep (B=1 → argmax selection).")
    ap.add_argument("--selection",   type=str, default="argmax",
                    choices=["argmax", "sample"],
                    help="How to pick one trajectory from top-B: "
                         "'argmax' always returns the best; "
                         "'sample' draws uniformly from the top-B pool.")
    ap.add_argument("--seed",        type=int, default=0,
                    help="RNG seed for 'sample' selection and tie-breaking.")
    ap.add_argument("--n_problems",  type=int, default=5000)
    args = ap.parse_args()

    students = [s.strip() for s in args.students.split(",") if s.strip()]
    for s in students:
        print(f"\n=== student: {s} ===")
        select_topb(
            q1_root=args.q1_root,
            q2_root=args.q2_root,
            rsr_root=args.rsr_root,
            student=s,
            q1_baseline=args.q1_baseline,
            B=args.B,
            selection=args.selection,
            seed=args.seed,
            n_problems=args.n_problems,
        )


if __name__ == "__main__":
    main()
