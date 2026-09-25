#!/usr/bin/env python3
"""
scripts/research/test_gliner_concurrency.py
-----------------------------
gliner_redact.py's module-level _MODEL_CACHE means every GlinerRedactor
instance across the whole process shares ONE loaded GLiNER model object
(that's the fix for the reload-cost bug -- see the cache comment in
gliner_redact.py). benchmarks/benchmark_tonst.py's real, production-shaped usage
calls .redact() from multiple ThreadPoolExecutor worker threads
concurrently (default --workers 4) against that SAME shared model. This
has never been tested -- every full-pipeline benchmark run so far used
--workers 1 specifically to sidestep the question, not answer it.

The concrete risk: GLiNER wraps a HuggingFace tokenizer (often a Rust-
backed "fast" tokenizer) plus a PyTorch model. A well-known failure mode
for Rust-backed tokenizers called concurrently from multiple Python
threads on the SAME tokenizer object is a `RuntimeError: Already
borrowed` (Rust's borrow checker enforces exclusive mutable access at
runtime, and Python's GIL does not serialize calls into extension code
enough to prevent two threads entering at once). A subtler, non-crashing
risk is silent cross-thread contamination or non-determinism in the
underlying forward pass. This script tests directly for BOTH:

1. Crash-style bugs: run many concurrent .redact() calls across many
   threads and multiple rounds (to catch an intermittent race, not just
   the first call) and check whether ANY exception was raised.
2. Correctness/determinism bugs: build a SEQUENTIAL baseline first (one
   thread, no concurrency) -- the ground truth for "what should this
   exact input produce" is GLiNER's own single-threaded answer, since
   the question here is determinism under concurrency, not GLiNER's
   accuracy. Then re-run the identical set of texts concurrently and
   diff every result against its own sequential baseline. Each text
   contains a UNIQUE numbered entity (name/company/codename) so any
   cross-thread contamination (thread A's result containing thread B's
   entity) is immediately visible, not just "recall dropped a bit".

Usage:
    cd ~/Desktop/tonst
    python3 scripts/research/test_gliner_concurrency.py
"""
# Run from anywhere: make the repo root and benchmarks/ importable without `pip install -e .`.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "benchmarks"))
import concurrent.futures
import re
import sys
import threading

sys.path.insert(0, ".")

from tonst.gliner_redact import GlinerRedactor

NUM_TEXTS = 40          # distinct, uniquely-identifiable cases
NUM_ROUNDS = 5          # repeat the full concurrent sweep this many times
                        # -- races are often intermittent, one pass can miss them
WORKERS = 16            # deliberately higher than benchmarks/benchmark_tonst.py's default 4,
                        # to make a real race more likely to surface, not less


def make_text(i: int) -> str:
    return (
        f"Ticket from Zephyrine{i} Quaddlebaum{i} at Nebulon{i} Systems "
        f"regarding Project Chimera{i}: please review and respond."
    )


def run_one(redactor: GlinerRedactor, i: int, text: str):
    result = redactor.redact(text)
    return i, result.mapping, result.redacted_text


def main():
    redactor = GlinerRedactor()
    print("Loading GLiNER (one-time)...", flush=True)
    redactor.is_available()
    print("Ready.\n", flush=True)

    texts = {i: make_text(i) for i in range(NUM_TEXTS)}

    # --- Step 1: sequential baseline (ground truth for determinism) ---
    print(f"Building sequential baseline over {NUM_TEXTS} distinct texts...", flush=True)
    baseline = {}
    for i, text in texts.items():
        _, mapping, redacted_text = run_one(redactor, i, text)
        baseline[i] = (mapping, redacted_text)
    print("Baseline built.\n", flush=True)

    # --- Step 2: concurrent hammering, multiple rounds ---
    exceptions = []
    mismatches = []
    cross_contamination = []
    total_calls = 0

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"Round {round_num}/{NUM_ROUNDS}: {NUM_TEXTS} concurrent calls "
              f"across {WORKERS} threads...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_one, redactor, i, text): i for i, text in texts.items()}
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                total_calls += 1
                try:
                    got_i, mapping, redacted_text = fut.result()
                except Exception as exc:
                    exceptions.append((round_num, i, repr(exc)))
                    continue

                expected_mapping, expected_redacted_text = baseline[i]
                if mapping != expected_mapping or redacted_text != expected_redacted_text:
                    mismatches.append({
                        "round": round_num,
                        "text_index": i,
                        "expected_mapping": expected_mapping,
                        "got_mapping": mapping,
                        "expected_redacted_text": expected_redacted_text,
                        "got_redacted_text": redacted_text,
                    })

                # Cross-contamination check: does this result contain any OTHER
                # text's unique numbered entity instead of/in addition to its own?
                # Uses a negative lookahead for a trailing digit so "Chimera1"
                # does NOT falsely match inside "Chimera11", "Chimera12", etc.
                # (a real bug in an earlier version of this script -- plain
                # substring containment treats "1" as a substring of "11",
                # which triggered ~150 false "contamination" hits that were
                # actually just each thread's own, correct entity).
                for other_i in texts:
                    if other_i == i:
                        continue
                    other_pattern = re.compile(rf"Chimera{other_i}(?!\d)")
                    if any(other_pattern.search(v) for v in mapping.values()):
                        cross_contamination.append((round_num, i, other_i))

    print(f"\nTotal concurrent calls: {total_calls}")
    print(f"Exceptions raised: {len(exceptions)}")
    for round_num, i, exc_repr in exceptions[:10]:
        print(f"    round={round_num} text_index={i} exception={exc_repr}")

    print(f"Mismatches vs. sequential baseline: {len(mismatches)}")
    for m in mismatches[:5]:
        print(f"    round={m['round']} text_index={m['text_index']}")
        print(f"      expected mapping: {m['expected_mapping']}")
        print(f"      got mapping:      {m['got_mapping']}")

    print(f"Cross-thread contamination events: {len(cross_contamination)}")
    for round_num, i, other_i in cross_contamination[:10]:
        print(f"    round={round_num}: text_index={i}'s result contained text_index={other_i}'s entity")

    if not exceptions and not mismatches and not cross_contamination:
        print(
            "\nNO CONCURRENCY ISSUES DETECTED: every concurrent call across "
            f"{NUM_ROUNDS} rounds x {NUM_TEXTS} texts x {WORKERS} threads "
            "raised no exception and matched its own sequential baseline "
            "exactly, with zero cross-thread contamination. This is evidence "
            "FOR thread-safety at this scale, not a formal guarantee -- races "
            "can still be load/hardware-dependent. Worth re-running once at a "
            "higher WORKERS count if you plan to deploy well above --workers 4."
        )
    else:
        print(
            "\nCONCURRENCY ISSUE DETECTED -- see counts above. Do NOT ship "
            "GLiNER redaction with workers > 1 until this is understood and "
            "fixed (e.g. wrapping predict_entities() in a lock, or giving "
            "each thread/worker its own model instance instead of sharing "
            "the module-level cache)."
        )


if __name__ == "__main__":
    main()
