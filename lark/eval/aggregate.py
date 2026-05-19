#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate all summary.json files into a single CSV (Table 3 format).

Walks:
  <eval_root>/<student>/<q1_baseline>/<q2_method>/<benchmark>/summary.json

Produces:
  <eval_root>/../table3.csv

Columns:
  student, q1_baseline, q2_method,
  aime24, aime25, amc23,
  math500_L1, math500_L2, math500_L3, math500_L4, math500_L5, math500,
  gpqa_diamond

Usage:
  python aggregate.py \
    --eval_root ./results/eval \
    [--output_csv ./results/table3.csv]
"""

import os
import json
import csv
import argparse
from pathlib import Path

BENCHMARKS   = ["aime24", "aime25", "amc23", "math500", "gpqa_diamond"]
MATH_LEVELS  = ["1", "2", "3", "4", "5"]


def load_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate eval results into table3.csv."
    )
    ap.add_argument("--eval_root",  required=True,
                    help="Root of eval results, "
                         "e.g. ./results/eval")
    ap.add_argument("--output_csv", default=None,
                    help="Output CSV path "
                         "(default: <eval_root>/../table3.csv)")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_csv   = Path(args.output_csv) if args.output_csv \
                else eval_root.parent / "table3.csv"

    rows = []

    # Walk: eval_root/<student>/<q1_baseline>/<q2_method>
    for student_dir in sorted(eval_root.iterdir()):
        if not student_dir.is_dir():
            continue
        for q1_dir in sorted(student_dir.iterdir()):
            if not q1_dir.is_dir():
                continue
            for q2_dir in sorted(q1_dir.iterdir()):
                if not q2_dir.is_dir():
                    continue

                row = {
                    "student":     student_dir.name,
                    "q1_baseline": q1_dir.name,
                    "q2_method":   q2_dir.name,
                }

                for bm in BENCHMARKS:
                    summary_path = q2_dir / bm / "summary.json"

                    # MATH500 level columns (always fill, even when missing)
                    if bm == "math500":
                        if summary_path.is_file():
                            s   = load_summary(summary_path)
                            row["math500"] = round(s.get("acc5", 0.0), 4)
                            by_lv = s.get("acc5_by_level", {})
                            for lv in MATH_LEVELS:
                                key = f"math500_L{lv}"
                                row[key] = round(by_lv[lv], 4) if lv in by_lv else ""
                        else:
                            row["math500"] = ""
                            for lv in MATH_LEVELS:
                                row[f"math500_L{lv}"] = ""
                    else:
                        if summary_path.is_file():
                            s = load_summary(summary_path)
                            row[bm] = round(s.get("acc5", 0.0), 4)
                        else:
                            row[bm] = ""

                rows.append(row)

    # Column order matches Table 3 layout
    fieldnames = (
        ["student", "q1_baseline", "q2_method",
         "aime24", "aime25", "amc23",
         "math500_L1", "math500_L2", "math500_L3", "math500_L4", "math500_L5",
         "math500", "gpqa_diamond"]
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
