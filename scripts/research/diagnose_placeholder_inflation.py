#!/usr/bin/env python3
"""
scripts/research/diagnose_placeholder_inflation.py
-----------------------------------
Run this on your Mac (needs Ollama running + gemma2:2b pulled, same as
the other diagnostics). Answers the question behind "why does the full
pipeline (17-19%) underperform the no-LLM baseline (22.79%) even though
mechanical trim already runs BEFORE compression?"

Confirmed by reading tonst/client.py directly: the pipeline order is
already redact -> mechanical_trim -> compress (step 2 then step 3 in
query()) -- so re-ordering isn't the fix, compression already only ever
sees already-trimmed text.

The real candidate: a redaction placeholder like [[NAME_a1b2c3d4]] is a
fixed-shape hex token that may tokenize to MORE tokens than the short
plain-text value it replaces (a first name, a short company name) --
so turning on enhanced (LLM-based) free-text redaction could be adding
tokens back in, before trim/compression ever get a chance to remove
them. This measures that directly, on real cases, with real token
counts -- not estimated.

Usage:
    cd ~/Desktop/tonst
    python3 scripts/research/diagnose_placeholder_inflation.py
"""
# Run from anywhere: make the repo root and benchmarks/ importable without `pip install -e .`.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "benchmarks"))
import random
import sys
import time

sys.path.insert(0, ".")

from tonst.redact import redact, redact_with_llm
from tonst.redact_llm import LLMRedactor
from tonst.trim import estimate_tokens, mechanical_trim
from benchmark_tonst import generate_benchmark_case

# Each (industry, mode, seed) explicitly chosen to hit a DIFFERENT shape --
# generate_benchmark_case()'s first random draw is the shape itself
# (rng.choice([...])), so reusing one seed across every case (the bug in
# the first version of this script) deterministically redraws the SAME
# shape every time regardless of industry. These seeds were verified
# directly against benchmarks/benchmark_tonst.py to cover all 5 real shapes at least
# once: the 2 supervised shapes, and critically all 3 unsupervised shapes
# -- including unsupervised_bloated_logs and unsupervised_verbose_dump,
# which are the two shapes that deliberately repeat a contact block/jargon
# lines twice specifically so mechanical trim has real duplication to cut.
# unsupervised_multi_turn (no such repetition by design) is included too,
# but no longer 3-out-of-4 of the sample by accident.
CASES = [
    ("Medical", "unsupervised", 1),   # -> unsupervised_multi_turn
    ("IT", "unsupervised", 5),        # -> unsupervised_bloated_logs
    ("Finance", "unsupervised", 6),   # -> unsupervised_verbose_dump
    ("Legal", "supervised", 1),       # -> supervised_few_shot
    ("Space", "supervised", 2),       # -> supervised_extraction
]


def main():
    redactor = LLMRedactor(timeout=8.0)
    print(f"LLMRedactor.is_available(): {redactor.is_available()}\n")

    totals = {"orig": 0, "regex": 0, "llm": 0, "regex_trim": 0, "llm_trim": 0}

    for industry, mode, seed in CASES:
        rng = random.Random(seed)
        shape, text, pii, parts, extra = generate_benchmark_case(rng, industry, mode)

        regex_only = redact(text)

        t0 = time.perf_counter()
        llm_result = redact_with_llm(text, redactor)
        dt = time.perf_counter() - t0

        orig_tok = estimate_tokens(text)
        regex_tok = estimate_tokens(regex_only.redacted_text)
        llm_tok = estimate_tokens(llm_result.redacted_text)

        trimmed_regex_tok = estimate_tokens(mechanical_trim(regex_only.redacted_text))
        trimmed_llm_tok = estimate_tokens(mechanical_trim(llm_result.redacted_text))

        free_text_fields = len(llm_result.mapping) - len(regex_only.mapping)

        print(f"=== {industry} / {mode} ({shape}) ===")
        print(f"  original tokens:                          {orig_tok}")
        print(f"  after regex-only redaction:                {regex_tok}  (fields={len(regex_only.mapping)})")
        print(f"  after regex+LLM redaction:                 {llm_tok}  (fields={len(llm_result.mapping)}, "
              f"+{free_text_fields} free-text, call={dt * 1000:.0f}ms)")
        print(f"  regex-only -> +mechanical trim:             {trimmed_regex_tok}  "
              f"({(1 - trimmed_regex_tok / orig_tok) * 100:.1f}% saved vs. original)")
        print(f"  regex+LLM -> +mechanical trim:              {trimmed_llm_tok}  "
              f"({(1 - trimmed_llm_tok / orig_tok) * 100:.1f}% saved vs. original)")
        delta = llm_tok - regex_tok
        print(f"  >>> enhanced redaction alone changed tokens by: {delta:+d} "
              f"(before trim/compress ever run)")
        print()

        totals["orig"] += orig_tok
        totals["regex"] += regex_tok
        totals["llm"] += llm_tok
        totals["regex_trim"] += trimmed_regex_tok
        totals["llm_trim"] += trimmed_llm_tok

    print("=" * 60)
    print("TOTALS across all sample cases")
    print("=" * 60)
    print(f"original:                  {totals['orig']}")
    print(f"regex-only + trim:         {totals['regex_trim']}  "
          f"({(1 - totals['regex_trim'] / totals['orig']) * 100:.1f}% saved)")
    print(f"regex+LLM redaction + trim:{totals['llm_trim']}  "
          f"({(1 - totals['llm_trim'] / totals['orig']) * 100:.1f}% saved)")
    print()
    if totals["llm_trim"] > totals["regex_trim"]:
        print(
            "CONFIRMED: enabling enhanced (LLM-based) free-text redaction, even before "
            "compression gets a chance to run, results in MORE tokens sent than regex-only "
            "redaction + mechanical trim. The placeholders it adds for names/companies/"
            "codenames cost more tokens than the plain text they replace. This is a real, "
            "structural tradeoff between free-text PII coverage and token savings -- not a "
            "compression bug, and not fixable by reordering trim vs. compress (trim already "
            "runs first)."
        )
    else:
        print(
            "NOT confirmed on these samples -- enhanced redaction didn't cost net tokens here. "
            "The full-pipeline underperformance vs. baseline may be concentrated in specific "
            "cases/industries rather than a general effect. Worth checking supervised cases "
            "separately, since they have much less redundant text for trim/compress to work "
            "with in the first place."
        )


if __name__ == "__main__":
    main()
