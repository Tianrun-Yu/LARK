#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the first \\boxed{...} answer from each model generation.

If the response contains \\boxed{...}, we take the content inside the first \\boxed{...}.
Otherwise, we treat that generation as incorrect.

Reads:  <eval_dir>/generations.jsonl
Writes: <eval_dir>/answers.jsonl
        {"problem_id": 0, "gold_answer": "...", "level": 1,
         "extracted": ["...", "", "...", "...", "..."]}

Usage:
  python extract_answer.py --eval_dir <path>
"""

import os
import re
import json
import argparse


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def extract_first_boxed(text: str) -> str:
    """
    Return the content of the first \\boxed{...} in text.
    Handles nested braces correctly.
    Returns empty string if no \\boxed{} is found.
    """
    match = re.search(r"\\boxed\{", text)
    if not match:
        return ""

    pos = match.end()
    depth = 1
    i = pos
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[pos: i - 1].strip()
    else:
        return text[pos:].strip()


def extract_answer(text: str) -> str:
    """
    ICPO-style extraction with fallbacks:
    1. Last \\boxed{...}
    2. Last number / fraction / decimal in text  (strip </think> block first)
    3. Empty string (failed)
    """
    # Strip <think>...</think> block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1. \boxed{}
    ans = extract_first_boxed(text)
    if ans:
        return ans

    # 2. Last number / fraction / decimal
    matches = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    if matches:
        return matches[-1]

    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract last \\boxed{} answer from each generation."
    )
    ap.add_argument("--eval_dir", required=True,
                    help="Directory containing generations.jsonl; "
                         "answers.jsonl is written here.")
    args = ap.parse_args()

    gen_path = os.path.join(args.eval_dir, "generations.jsonl")
    out_path = os.path.join(args.eval_dir, "answers.jsonl")

    if not os.path.isfile(gen_path):
        raise FileNotFoundError(f"generations.jsonl not found: {gen_path}")

    if os.path.isfile(out_path):
        print(f"[SKIP] already exists: {out_path}")
        return

    n_total  = 0
    n_found  = 0

    with open(gen_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            rec = json.loads(line)
            extracted = []
            for gen in rec["generations"]:
                ans = extract_answer(gen)
                extracted.append(ans)
                n_total += 1
                if ans:
                    n_found += 1

            out = {
                "problem_id":  rec["problem_id"],
                "gold_answer": rec.get("answer", ""),
                "level":       rec.get("level", None),
                "extracted":   extracted,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[OK] {out_path}")
    if n_total:
        print(f"     extracted: {n_found}/{n_total} ({100*n_found/n_total:.1f}%)")


if __name__ == "__main__":
    main()
