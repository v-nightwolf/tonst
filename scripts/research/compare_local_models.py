#!/usr/bin/env python3
"""
scripts/research/compare_local_models.py
------------------------
Compares candidate Ollama models against the current default
(llama3.2:1b) for tonst's LOCAL-MODEL tasks (enhanced redaction,
compression, history compaction) all at once -- reuses
benchmarks/benchmark_tonst.py's run_benchmark() directly (same synthetic
industries, same guard rails, same near-timeout/round-trip-failure
metrics) so results are directly comparable to every number already in
research/colab-benchmark-findings.md, not a separate ad-hoc test.

Why these candidates: llama3.2:1b accumulated a long list of
reliability bugs across this investigation (JSON-array parsing
failures, the `format: "json"` grammar-decoding hang, placeholder
re-casing corruption under compression/compaction) -- see
research/colab-benchmark-findings.md and
research/compression-model-replacement-plan.md for the full history.
Rather than assuming any of that is llama3.2:1b-specific, this runs
the SAME full pipeline (redaction + compression + compaction together,
`--enable-local-llm`-equivalent) against a few small instruction-tuned
alternatives generally reported as more reliable at structured/
instruction-following tasks, all runnable via Ollama with no new
serving infrastructure:
  - qwen2.5:1.5b   (Alibaba, ~1B-class, known for structured-output reliability)
  - phi3.5         (Microsoft, 3.8B, strong reasoning-per-parameter)
  - gemma2:2b      (Google, solid general instruction-following)

Note on scope: this compares models for ALL THREE local-model tasks at
once (matching how `--enable-local-llm` already runs them together),
which is useful as a first pass, but the guard-rail metrics below
(near_timeout_counts, round_trip_restoration_failures) don't
distinguish which of the three stages is responsible for a given
failure -- rerun a promising candidate with the isolated
--use-enhanced-redaction / --use-local-compression /
--use-history-compaction flags (already supported by benchmarks/benchmark_tonst.py)
if you need to know that.

Also note: this does NOT measure full_name/company/codename recall the
way scripts/research/gliner_sanity_check.py does for GLiNER -- benchmarks/benchmark_tonst.py's own
leak-check only covers email/card/phone/ip. Redaction quality here is
being observed only as a side effect (does redact_llm's guard rail
accept its output, i.e. does the model return parseable, uncorrupted
JSON), not measured for catch-rate. That's intentional: the redaction
backend decision is GLiNER's per research/gliner-sanity-check-findings.md,
independent of whatever wins here for compression/compaction.

Usage:
    python3 scripts/research/compare_local_models.py --iterations 60 --workers 1
    python3 scripts/research/compare_local_models.py --models llama3.2:1b,qwen2.5:1.5b --iterations 30

Must be run from the same directory as benchmarks/benchmark_tonst.py (imports it
directly). Requires `ollama` on PATH and the Ollama server already
running (`ollama serve`) -- this script will `ollama pull` each model
that isn't already present, which needs real internet access (works
fine from a normal terminal; will NOT work through a sandboxed
device-bridge shell whose egress policy blocks package/model
registries -- see research/gliner-sanity-check-findings.md's Setup
section for that same block encountered with pip/PyPI).
"""
from __future__ import annotations
# Run from anywhere: make the repo root and benchmarks/ importable without `pip install -e .`.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "benchmarks"))
import argparse
import json
import subprocess
import sys
import time
import traceback
from typing import Dict, List

from benchmark_tonst import run_benchmark, wait_for_ollama_ready

DEFAULT_MODELS = "llama3.2:1b,qwen2.5:1.5b,phi3.5,gemma2:2b"


def ensure_model_pulled(model: str) -> None:
    print(f"\n--- ollama pull {model} (no-op if already present) ---", flush=True)
    result = subprocess.run(["ollama", "pull", model])
    if result.returncode != 0:
        raise RuntimeError(f"`ollama pull {model}` failed with exit code {result.returncode}")


def summarize_for_table(model: str, overall: dict, wall_s: float, error: str | None) -> dict:
    if error:
        return {"model": model, "error": error}
    nt = overall["near_timeout_counts"]
    return {
        "model": model,
        "iterations": overall["iterations"],
        "percent_saved": overall["percent_saved"],
        "round_trip_restoration_failures": overall["round_trip_restoration_failures"],
        "near_timeout_total": nt["redaction"] + nt["compression"] + nt["compaction"],
        "near_timeout_by_stage": nt,
        "latency_ms_p50": overall["latency_overhead_ms"]["p50"],
        "latency_ms_mean": overall["latency_overhead_ms"]["mean"],
        "wall_clock_s": round(wall_s, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS,
                         help=f"Comma-separated Ollama model tags to compare. Default: {DEFAULT_MODELS}")
    parser.add_argument("--iterations", type=int, default=60,
                         help="Iterations per model (same shape as the staged Colab Step 1 run: 60, --workers 1, before scaling up).")
    parser.add_argument("--workers", type=int, default=1,
                         help="Default 1 (sequential) -- matches the staged, contention-free first pass used throughout this investigation.")
    parser.add_argument("--skip-pull", action="store_true", help="Skip `ollama pull` (use if all models are already present).")
    parser.add_argument("--output", type=str, default="model_comparison_report.json")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    full_reports: Dict[str, dict] = {}
    table_rows: List[dict] = []

    for model in models:
        print(f"\n{'=' * 60}\nModel: {model}\n{'=' * 60}", flush=True)
        t0 = time.perf_counter()
        try:
            if not args.skip_pull:
                ensure_model_pulled(model)
            wait_for_ollama_ready(model)
            report = run_benchmark(
                iterations=args.iterations,
                local_model=model,
                use_enhanced_redaction=True,
                use_local_compression=True,
                use_history_compaction=True,
                workers=args.workers,
                seed=42,  # same seed across all models -- identical synthetic cases, only the model differs
            )
            wall_s = time.perf_counter() - t0
            full_reports[model] = report
            table_rows.append(summarize_for_table(model, report["overall_results"], wall_s, None))
            print(json.dumps(table_rows[-1], indent=2), flush=True)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: one model's failure shouldn't kill the whole comparison
            wall_s = time.perf_counter() - t0
            err_str = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {model} failed: {err_str}", flush=True)
            traceback.print_exc()
            full_reports[model] = {"error": err_str}
            table_rows.append(summarize_for_table(model, {}, wall_s, err_str))

    with open(args.output, "w") as f:
        json.dump({"table": table_rows, "full_reports": full_reports}, f, indent=2)

    print("\n\n=== COMPARISON TABLE ===")
    print(json.dumps(table_rows, indent=2))
    print(f"\nFull per-model reports written to {args.output}")


if __name__ == "__main__":
    main()
