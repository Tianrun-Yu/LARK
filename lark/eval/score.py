#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ACC@5 from extracted answers.

ACC@5 = fraction of problems where at least one of the 5 extracted answers
        is correct.  An empty extracted answer is always incorrect.

Reads:  <eval_dir>/answers.jsonl
Writes: <eval_dir>/scores.jsonl    – per-problem correctness
        <eval_dir>/summary.json    – aggregate ACC@5 (and per-level for math500)

Usage:
  python score.py --eval_dir <path>
"""

import os
import re
import json
import argparse
from collections import defaultdict


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize(ans: str) -> str:
    """
    Light normalization for answer comparison:
      - strip whitespace
      - lowercase
      - attempt numeric simplification (3.0 → 3, 0.500 → 0.5)
    """
    ans = ans.strip().lower()
    # Remove surrounding $ signs (LaTeX)
    ans = ans.strip("$").strip()
    try:
        f = float(ans)
        # Integer check
        if f == int(f):
            return str(int(f))
        # Remove trailing zeros
        return f"{f:.10g}"
    except (ValueError, OverflowError):
        pass
    return ans


def is_correct(extracted: str, gold: str) -> bool:
    """Return True iff the extracted answer matches the gold answer."""
    if not extracted:
        return False
    return normalize(extracted) == normalize(gold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Score ACC@5.")
    ap.add_argument("--eval_dir", required=True,
                    help="Directory with answers.jsonl; "
                         "writes scores.jsonl + summary.json here.")
    args = ap.parse_args()

    ans_path     = os.path.join(args.eval_dir, "answers.jsonl")
    scores_path  = os.path.join(args.eval_dir, "scores.jsonl")
    summary_path = os.path.join(args.eval_dir, "summary.json")

    if not os.path.isfile(ans_path):
        raise FileNotFoundError(f"answers.jsonl not found: {ans_path}")

    if os.path.isfile(summary_path):
        print(f"[SKIP] already exists: {summary_path}")
        return

    records = []
    with open(ans_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    per_problem = []
    with open(scores_path, "w", encoding="utf-8") as w:
        for rec in records:
            gold           = rec["gold_answer"]
            trial_correct  = [is_correct(e, gold) for e in rec["extracted"]]
            acc5           = 1 if any(trial_correct) else 0

            out = {
                "problem_id":    rec["problem_id"],
                "level":         rec.get("level"),
                "gold_answer":   gold,
                "extracted":     rec["extracted"],
                "trial_correct": trial_correct,
                "acc5":          acc5,
            }
            per_problem.append(out)
            w.write(json.dumps(out, ensure_ascii=False) + "\n")

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    total      = len(per_problem)
    total_acc5 = sum(p["acc5"] for p in per_problem) / total if total else 0.0

    # Per-level breakdown (for MATH500)
    by_level = defaultdict(list)
    for p in per_problem:
        if p["level"] is not None:
            by_level[str(p["level"])].append(p["acc5"])

    acc5_by_level = {
        lv: round(sum(vs) / len(vs), 6)
        for lv, vs in sorted(by_level.items())
    }

    summary = {
        "n_problems":    total,
        "n_correct":     sum(p["acc5"] for p in per_problem),
        "acc5":          round(total_acc5, 6),
        "acc5_by_level": acc5_by_level,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {scores_path}")
    print(f"[OK] {summary_path}")
    print(f"     ACC@5 = {total_acc5:.4f}  "
          f"({summary['n_correct']}/{total})")
    if acc5_by_level:
        for lv, v in acc5_by_level.items():
            print(f"       L{lv}: {v:.4f}")


if __name__ == "__main__":
    main()
