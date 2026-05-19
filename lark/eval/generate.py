#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACC@5 generation for evaluation benchmarks.
Backend: SGLang (preferred, fast) → transformers (fallback).

For each problem, generates N_SAMPLES=5 independent outputs.

Reads:  <benchmark_path>  – JSON/JSONL with fields: problem/prompt, answer
Writes: <output_dir>/generations.jsonl
"""

import os
import json
import time
import shutil
import tempfile
import argparse
from pathlib import Path

BENCHMARK_PROMPT_FILE = {
    "aime24":       "system_aime_amc.txt",
    "aime25":       "system_aime_amc.txt",
    "amc23":        "system_aime_amc.txt",
    "math500":      "system_math.txt",
    "gpqa_diamond": "system_gpqa.txt",
    "AIME":         "system_aime_amc.txt",
    "AMC":          "system_aime_amc.txt",
    "CPQA":         "system_gpqa.txt",
    "MATH-L1":      "system_math.txt",
    "MATH-L2":      "system_math.txt",
    "MATH-L3":      "system_math.txt",
    "MATH-L4":      "system_math.txt",
    "MATH-L5":      "system_math.txt",
    "TRAIN100":     "system_math.txt",
}

# Models that use the official Qwen-Math system prompt
QWEN_MATH_MODEL_KEYWORDS = ["qwen2.5-math", "qwen-math"]

def get_prompt_file(benchmark: str, model_path: str) -> str:
    """Return prompt file, using qwen-math official prompt if model is a math model."""
    model_lower = model_path.lower()
    if any(kw in model_lower for kw in QWEN_MATH_MODEL_KEYWORDS):
        # All benchmarks use the same official qwen-math system prompt
        return "system_qwen_math.txt"
    return BENCHMARK_PROMPT_FILE[benchmark]

N_SAMPLES = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark(path: str):
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    for item in data:
        if "problem" not in item and "prompt" in item:
            item["problem"] = item["prompt"]
        problems.append(item)
    return problems


def load_system_prompt(prompts_dir: str, filename: str) -> str:
    path = os.path.join(prompts_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_chat_prompts(problems, system_prompt: str, tokenizer) -> list:
    prompts = []
    for p in problems:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": p["problem"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)
    return prompts


def merge_lora_to_tmpdir(adapter_path: str, base_model_name: str) -> str:
    """Merge LoRA adapter into base model, save to a temp dir, return path."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[INFO] merging LoRA: {base_model_name} + {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model  = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    tmpdir = tempfile.mkdtemp(prefix="limo_merged_")
    print(f"[INFO] saving merged model → {tmpdir}  (will be deleted after generation)")
    merged.save_pretrained(tmpdir, safe_serialization=True)
    tokenizer.save_pretrained(tmpdir)
    del merged, model, base
    torch.cuda.empty_cache()
    return tmpdir


def truncate_after_first_boxed(text: str) -> str:
    """Truncate text right after the first complete \\boxed{...} (handling nested braces)."""
    import re
    match = re.search(r"\\boxed\{", text)
    if not match:
        return text  # no \boxed found, return as-is

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
        return text[:i]  # truncate right after closing }
    return text  # unclosed, return as-is


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ---------------------------------------------------------------------------
# SGLang backend
# ---------------------------------------------------------------------------

def generate_sglang(engine_model: str, prompts: list, n_samples: int, args) -> list:
    """Generate with SGLang Engine. Returns list[list[str]] shape [N, n_samples]."""
    import sglang as sgl

    print(f"[INFO] starting SGLang engine: {engine_model}")
    engine = sgl.Engine(
        model_path=engine_model,
        tp_size=args.tensor_parallel_size,
        disable_cuda_graph=True,   # avoid JIT nvcc compilation
        dtype="bfloat16",
    )

    sampling_params = {
        "temperature":      args.temperature,
        "top_p":            args.top_p,
        "max_new_tokens":   args.max_new_tokens,
        "repetition_penalty": 1.1,
        # Stop after first complete \boxed{...} — match closing } after \boxed{
        # Uses regex: \boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\} to handle one level of nesting
        "stop_regex": [r"\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"],
        "no_stop_trim": True,   # keep the matched \boxed{...} in output
    }

    # Repeat each prompt n_samples times → flat list
    N = len(prompts)
    repeated = [p for p in prompts for _ in range(n_samples)]

    print(f"[INFO] SGLang generating {N} problems × {n_samples} samples = {len(repeated)} sequences")
    t0 = time.time()

    # SGLang handles batching internally; send all at once
    raw_outputs = engine.generate(repeated, sampling_params)
    engine.shutdown()

    elapsed = time.time() - t0
    print(f"[INFO] SGLang done in {fmt_time(elapsed)}  ({elapsed/N:.1f}s/problem)")

    # raw_outputs is list of dicts {"text": "...", "meta_info": {...}}
    all_outputs = []
    for i in range(N):
        chunk = raw_outputs[i * n_samples: (i + 1) * n_samples]
        all_outputs.append([
            truncate_after_first_boxed(o["text"] if isinstance(o, dict) else o.outputs[0].text)
            for o in chunk
        ])
    return all_outputs


# ---------------------------------------------------------------------------
# Transformers fallback backend
# ---------------------------------------------------------------------------

def generate_transformers(model, tokenizer, prompts: list, n_samples: int,
                          args, device) -> list:
    """Generate with HuggingFace transformers. Returns list[list[str]]."""
    import torch
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    tokenizer.padding_side = "left"
    N = len(prompts)

    def generate_batch(batch_prompts):
        B = len(batch_prompts)
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            add_special_tokens=False,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        results = []
        for b in range(B):
            seqs = out_ids[b * n_samples: (b + 1) * n_samples]
            results.append([
                truncate_after_first_boxed(
                    tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                )
                for seq in seqs
            ])
        return results

    all_outputs = []
    batches = list(range(0, N, args.batch_size))

    iterator = tqdm(batches, desc="generating", unit="batch") if use_tqdm else batches
    t_start = time.time()

    for bi, start in enumerate(iterator):
        batch = prompts[start: start + args.batch_size]
        t_batch = time.time()
        try:
            results = generate_batch(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"\n  [OOM] batch={len(batch)} → retrying batch=1")
            results = []
            for p in batch:
                results.extend(generate_batch([p]))
        all_outputs.extend(results)

        done = min(start + args.batch_size, N)
        elapsed = time.time() - t_start
        speed   = elapsed / done if done > 0 else 0
        eta     = speed * (N - done)
        batch_t = time.time() - t_batch

        if use_tqdm:
            iterator.set_postfix(
                done=f"{done}/{N}",
                batch_t=fmt_time(batch_t),
                eta=fmt_time(eta),
            )
        else:
            pct = done / N * 100
            bar_len = 30
            filled  = int(bar_len * done / N)
            bar     = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{bar}] {done}/{N} ({pct:.1f}%)  "
                  f"batch={fmt_time(batch_t)}  elapsed={fmt_time(elapsed)}  eta={fmt_time(eta)}",
                  flush=True)

    total = time.time() - t_start
    print(f"\n[INFO] transformers done in {fmt_time(total)}  ({total/N:.1f}s/problem)")
    return all_outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate ACC@5 samples – SGLang (fast) or transformers fallback."
    )
    ap.add_argument("--model_path",    required=True)
    ap.add_argument("--benchmark",     required=True,
                    choices=list(BENCHMARK_PROMPT_FILE.keys()))
    ap.add_argument("--benchmark_path", required=True)
    ap.add_argument("--output_dir",    required=True)
    ap.add_argument("--prompts_dir",   default=None)
    ap.add_argument("--n_samples",     type=int,   default=N_SAMPLES)
    ap.add_argument("--temperature",   type=float, default=0.6)
    ap.add_argument("--top_p",         type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int,  default=8192)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--batch_size",    type=int,   default=4,
                    help="Batch size for transformers fallback (ignored by SGLang).")
    ap.add_argument("--backend",       choices=["auto", "sglang", "transformers"],
                    default="auto",
                    help="Inference backend. auto tries SGLang first.")
    ap.add_argument("--num_shards",   type=int, default=1,
                    help="Split problems across N shards for parallel generation.")
    ap.add_argument("--shard_id",     type=int, default=0,
                    help="Which shard to generate (0-indexed).")
    ap.add_argument("--merge_shards", action="store_true",
                    help="Merge shard outputs into final generations.jsonl.")
    args = ap.parse_args()

    if args.prompts_dir is None:
        args.prompts_dir = str(Path(__file__).parent / "prompts")

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Merge mode: combine shard outputs and return
    # ------------------------------------------------------------------
    if args.merge_shards:
        out_path = os.path.join(args.output_dir, "generations.jsonl")
        all_records = {}
        for sid in range(args.num_shards):
            sp = os.path.join(args.output_dir, f"generations_shard{sid}.jsonl")
            if not os.path.isfile(sp):
                print(f"[WARN] shard {sid} not found: {sp}")
                continue
            with open(sp, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    all_records[rec["problem_id"]] = rec
        with open(out_path, "w", encoding="utf-8") as w:
            for pid in sorted(all_records.keys()):
                w.write(json.dumps(all_records[pid], ensure_ascii=False) + "\n")
        print(f"[OK] merged {len(all_records)} problems → {out_path}")
        # Clean up shard files
        for sid in range(args.num_shards):
            sp = os.path.join(args.output_dir, f"generations_shard{sid}.jsonl")
            if os.path.isfile(sp):
                os.remove(sp)
        return

    # ------------------------------------------------------------------
    # Determine output path (shard or final)
    # ------------------------------------------------------------------
    if args.num_shards > 1:
        out_path = os.path.join(args.output_dir, f"generations_shard{args.shard_id}.jsonl")
    else:
        out_path = os.path.join(args.output_dir, "generations.jsonl")

    if os.path.isfile(out_path):
        print(f"[SKIP] already exists: {out_path}")
        return

    # ------------------------------------------------------------------
    # Detect adapter vs plain model
    # ------------------------------------------------------------------
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    is_lora = os.path.isfile(adapter_config_path)
    base_model_name = None
    if is_lora:
        with open(adapter_config_path) as f:
            base_model_name = json.load(f)["base_model_name_or_path"]
        print(f"[INFO] LoRA adapter detected  base={base_model_name}")

    # ------------------------------------------------------------------
    # Load benchmark and select shard subset
    # ------------------------------------------------------------------
    all_problems = load_benchmark(args.benchmark_path)
    if args.num_shards > 1:
        problems = [p for i, p in enumerate(all_problems) if i % args.num_shards == args.shard_id]
        # Keep original problem_ids for merging
        problem_ids = [i for i in range(len(all_problems)) if i % args.num_shards == args.shard_id]
        print(f"[INFO] benchmark={args.benchmark}  shard {args.shard_id}/{args.num_shards}  "
              f"n_problems={len(problems)}/{len(all_problems)}")
    else:
        problems = all_problems
        problem_ids = list(range(len(problems)))
        print(f"[INFO] benchmark={args.benchmark}  n_problems={len(problems)}")

    # ------------------------------------------------------------------
    # SGLang path
    # ------------------------------------------------------------------
    use_sglang = False
    if args.backend in ("auto", "sglang"):
        try:
            import sglang  # noqa: F401
            use_sglang = True
            print("[INFO] SGLang available → using SGLang backend")
        except ImportError:
            print("[INFO] SGLang not installed → falling back to transformers")

    tmpdir = None
    all_outputs = None

    if use_sglang:
        try:
            import multiprocessing as _mp
            from transformers import AutoTokenizer
            tok_path = args.model_path if not is_lora else base_model_name
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            prompt_file   = get_prompt_file(args.benchmark, args.model_path)
            system_prompt = load_system_prompt(args.prompts_dir, prompt_file)
            print(f"[INFO] system prompt: {prompt_file}")
            prompts = build_chat_prompts(problems, system_prompt, tokenizer)

            engine_model = args.model_path
            if is_lora:
                tmpdir = merge_lora_to_tmpdir(args.model_path, base_model_name)
                engine_model = tmpdir

            # Run SGLang in a child process so SIGQUIT doesn't kill us
            result_queue = _mp.Queue()
            def _sglang_worker(q, em, pr, ns, a):
                try:
                    out = generate_sglang(em, pr, ns, a)
                    q.put(("ok", out))
                except Exception as ex:
                    q.put(("err", str(ex)))

            proc = _mp.Process(target=_sglang_worker,
                               args=(result_queue, engine_model, prompts, args.n_samples, args))
            proc.start()
            proc.join()

            if proc.exitcode == 0:
                status, payload = result_queue.get()
                if status == "ok":
                    all_outputs = payload
                else:
                    raise RuntimeError(payload)
            else:
                raise RuntimeError(f"SGLang worker exited with code {proc.exitcode}")

        except Exception as e:
            print(f"[WARN] SGLang failed ({e}), falling back to transformers")
            use_sglang = False
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
                tmpdir = None
            all_outputs = None

    # ------------------------------------------------------------------
    # Transformers fallback
    # ------------------------------------------------------------------
    if not use_sglang or all_outputs is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print(f"[INFO] loading tokenizer from: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        fa2_kwargs = {}
        try:
            import flash_attn  # noqa: F401
            fa2_kwargs["attn_implementation"] = "flash_attention_2"
            print("[INFO] Flash Attention 2 enabled")
        except ImportError:
            pass

        if is_lora:
            print(f"[INFO] loading base model: {base_model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, dtype=torch.bfloat16,
                trust_remote_code=True, device_map="auto", **fa2_kwargs,
            )
            print(f"[INFO] loading LoRA adapter: {args.model_path}")
            model = PeftModel.from_pretrained(model, args.model_path)
        else:
            print(f"[INFO] loading model: {args.model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, dtype=torch.bfloat16,
                trust_remote_code=True, device_map="auto", **fa2_kwargs,
            )
        model.eval()
        device = next(model.parameters()).device

        prompt_file   = get_prompt_file(args.benchmark, args.model_path)
        system_prompt = load_system_prompt(args.prompts_dir, prompt_file)
        print(f"[INFO] system prompt: {prompt_file}")
        prompts = build_chat_prompts(problems, system_prompt, tokenizer)

        print(f"[INFO] transformers  n_samples={args.n_samples}  "
              f"batch_size={args.batch_size}  max_new_tokens={args.max_new_tokens}")
        all_outputs = generate_transformers(
            model, tokenizer, prompts, args.n_samples, args, device
        )

    # ------------------------------------------------------------------
    # Cleanup temp dir
    # ------------------------------------------------------------------
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"[INFO] removed temp dir {tmpdir}")

    # ------------------------------------------------------------------
    # Write generations.jsonl
    # ------------------------------------------------------------------
    with open(out_path, "w", encoding="utf-8") as w:
        for pid, (problem, generations) in zip(problem_ids, zip(problems, all_outputs)):
            record = {
                "problem_id":  pid,
                "problem":     problem.get("problem", ""),
                "answer":      problem.get("answer", ""),
                "level":       problem.get("level", None),
                "generations": generations,
            }
            w.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
