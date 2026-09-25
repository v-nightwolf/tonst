#!/usr/bin/env python3
"""
scripts/research/capture_restoration_failures.py
--------------------------------
Isolates ONE local-model stage (compression OR compaction) for ONE
model, and -- unlike benchmarks/benchmark_tonst.py's aggregate near_timeout /
round_trip_restoration_failures COUNTS -- captures the full transcript
of every failing case: which industry/shape, the original text, the
exact prompt sent to the "paid" API, and the raw final response. That's
enough to see exactly what the rewrite did to corrupt a placeholder,
not just that a failure happened.

Why this exists: scripts/research/compare_local_models.py's combined run (redaction +
compression + compaction together) found qwen2.5:1.5b and gemma2:2b
each producing 1 round_trip_restoration_failure out of 60 iterations
(see research/compression-model-replacement-plan.md), despite the
guard rails (placeholders_preserved / _no_corrupted_placeholders)
existing specifically to reject a corrupting rewrite before it's ever
used. An aggregate count can't say whether that's compression's fault
or compaction's, or show what actually went wrong -- this does both:
running with --use-enhanced-redaction OFF (mechanical/regex redaction
still runs -- see client.py's pipeline -- so real placeholders are
still present for compression/compaction to potentially corrupt) and
only ONE of use_local_compression/use_history_compaction ON, so any
corruption found is unambiguous about which stage caused it.

Compaction-only efficiency note: compact_history() only ever fires for
the unsupervised_multi_turn prompt shape (the only one that goes
through query_messages() -- see benchmarks/benchmark_tonst.py's
generate_benchmark_case). Cycling through all shapes like the main
benchmark does would waste most iterations on cases that can't
exercise compaction at all, so --stage compaction forces every
generated case to be unsupervised_multi_turn (still varying across all
6 industries) rather than mixing in shapes that are guaranteed
irrelevant to this stage.

Reproducibility caveat: same seed reproduces the same SYNTHETIC cases,
but not necessarily the same model output -- Ollama's default sampling
isn't fully deterministic run to run, so this may or may not surface
the exact same failure as the earlier combined run. Run with enough
iterations to catch a fresh example if the original doesn't recur.

Usage:
    python3 scripts/research/capture_restoration_failures.py --model gemma2:2b --stage compression --iterations 60
    python3 scripts/research/capture_restoration_failures.py --model gemma2:2b --stage compaction --iterations 60
    python3 scripts/research/capture_restoration_failures.py --model qwen2.5:1.5b --stage compression --iterations 60
    python3 scripts/research/capture_restoration_failures.py --model qwen2.5:1.5b --stage compaction --iterations 60

Must be run from the same directory as benchmarks/benchmark_tonst.py (imports it
directly).
"""
from __future__ import annotations
# Run from anywhere: make the repo root and benchmarks/ importable without `pip install -e .`.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "benchmarks"))
import argparse
import json
import random

from benchmark_tonst import INDUSTRY_CONFIGS, generate_benchmark_case, wait_for_ollama_ready
from tonst import TonstClient


def draw_case(seed: int, industry: str, mode: str, required_shape: str | None, max_attempts: int = 50):
    """generate_benchmark_case() picks its shape via an internal random
    draw among 2-3 options for the given mode -- when required_shape is
    set (used for --stage compaction, which only unsupervised_multi_turn
    can exercise), keep redrawing with a fresh rng until that shape
    comes up, rather than accepting whatever shape happened to be drawn.
    """
    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt * 7919)  # large prime step so retries don't collide with other industries' seeds
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(rng, industry, mode)
        if required_shape is None or shape == required_shape:
            return shape, flat_prompt, pii, parts, messages
    raise RuntimeError(f"could not draw shape={required_shape!r} for industry={industry!r} after {max_attempts} attempts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--stage", type=str, choices=["compression", "compaction"], required=True)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = args.output or f"restoration_failures_{args.model.replace(':', '_').replace('.', '_')}_{args.stage}.json"

    wait_for_ollama_ready(args.model)

    industries = list(INDUSTRY_CONFIGS.keys())
    use_compression = args.stage == "compression"
    use_compaction = args.stage == "compaction"
    required_shape = "unsupervised_multi_turn" if use_compaction else None

    failures = []

    for idx in range(args.iterations):
        industry = industries[idx % len(industries)]
        mode = "unsupervised" if use_compaction else ["supervised", "unsupervised"][(idx // len(industries)) % 2]
        shape, flat_prompt, pii, parts, messages = draw_case(args.seed + idx * 13, industry, mode, required_shape)

        thread_received = []

        def mock_paid_api(trimmed_prompt: str) -> str:
            thread_received.append(trimmed_prompt)
            tokens = [w for w in trimmed_prompt.split() if w.startswith("[[") and "]]" in w]
            return f"Processed request successfully. Reference tokens: {' '.join(tokens[:4])}"

        client = TonstClient(
            call_fn=mock_paid_api,
            use_local_compression=use_compression,
            use_enhanced_redaction=False,  # isolate to compression/compaction only -- mechanical regex redaction still runs underneath
            use_history_compaction=use_compaction,
            local_model=args.model,
            compaction_token_threshold=100,
        )

        if messages is not None:
            response, report = client.query_messages(messages, keep_last_n=3)
        else:
            response, report = client.query(flat_prompt)

        sent_prompt = thread_received[-1] if thread_received else None
        rt_fail = "[[" in response and "]]" in response

        if rt_fail:
            print(f"[FAILURE] iteration {idx} ({industry}/{shape})", flush=True)
            failures.append({
                "iteration": idx,
                "industry": industry,
                "mode": mode,
                "shape": shape,
                "original_flat_prompt": flat_prompt if messages is None else None,
                "original_messages": messages,
                "ground_truth_pii": pii,
                "sent_prompt_to_paid_api": sent_prompt,
                "final_response_to_user": response,
                "report": {
                    "original_tokens": report.original_tokens,
                    "sent_tokens": report.sent_tokens,
                    "redacted_fields": report.redacted_fields,
                    "redaction_ms": report.redaction_ms,
                    "compression_ms": report.compression_ms,
                    "compaction_ms": report.compaction_ms,
                },
            })
        elif (idx + 1) % 20 == 0:
            print(f"[{idx + 1}/{args.iterations}] checked, {len(failures)} failure(s) so far", flush=True)

    print(f"\nDone. {len(failures)} failure(s) out of {args.iterations} iterations ({args.model}, stage={args.stage}).")
    with open(output, "w") as f:
        json.dump({"model": args.model, "stage": args.stage, "iterations": args.iterations, "failures": failures}, f, indent=2)
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
