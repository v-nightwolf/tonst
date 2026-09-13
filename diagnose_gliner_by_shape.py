#!/usr/bin/env python3
"""
diagnose_gliner_by_shape.py
-----------------------------
Confirms (or refutes) a specific hypothesis about the ~18-20pp gap
between the standalone GLiNER sanity check (~91% weighted recall on
clean text) and the real in-pipeline benchmark_tonst.py numbers
(70.0% at 60 iterations, 72.78% at 180). diagnose_gliner_regex_interaction.py
already ruled out "regex running first disrupts GLiNER's context"
as the main cause (only a -2.59pp effect, mostly on codename) -- so
this checks the other real candidate: run_benchmark()'s
run_single_iteration() forces redaction_backend down to "regex"
whenever a case's messages is not None (the unsupervised_multi_turn
shape), REGARDLESS of which backend was actually selected. That shape
is ~1/6 of all cases by construction (unsupervised is half of all
cases, and multi_turn is 1 of 3 unsupervised shapes), and gets ZERO
enhanced free-text redaction every time -- regex structurally cannot
catch names/companies/codenames at all.

This replicates run_benchmark()'s EXACT case-generation scheme (same
seed formula, same industry/mode assignment per idx, default seed=42,
180 iterations) so the cases here are byte-for-byte identical to what
the real 180-iteration run used, then reports GLiNER free-text recall
BROKEN DOWN BY SHAPE -- so the multi_turn collapse (if real) is directly
visible instead of inferred from arithmetic.

Usage:
    cd ~/Desktop/tonst
    python3 diagnose_gliner_by_shape.py
"""
import random
import sys

sys.path.insert(0, ".")

from tonst.redact import redact
from tonst.gliner_redact import GlinerRedactor
from benchmark_tonst import generate_benchmark_case, INDUSTRY_CONFIGS

FREE_TEXT_FIELDS = ("full_name", "company", "codename")


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def entity_matches(predicted_text: str, ground_truth: str) -> bool:
    p, g = normalize(predicted_text), normalize(ground_truth)
    return g in p or p in g


def field_hit(mapping: dict, ground_truth_value: str) -> bool:
    return any(entity_matches(v, ground_truth_value) for v in mapping.values())


def main():
    redactor = GlinerRedactor()
    print("Loading GLiNER (one-time)...", flush=True)
    redactor.is_available()
    print("Ready.\n", flush=True)

    industries = list(INDUSTRY_CONFIGS.keys())
    modes = ["supervised", "unsupervised"]
    iterations = 180
    seed = 42  # matches run_benchmark()'s default exactly

    by_shape_hits = {}
    by_shape_totals = {}
    forced_regex_count = 0

    for idx in range(iterations):
        local_rng = random.Random(seed + idx)
        industry = industries[idx % len(industries)]
        mode = modes[(idx // len(industries)) % len(modes)]
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(local_rng, industry, mode)

        by_shape_hits.setdefault(shape, {f: 0 for f in FREE_TEXT_FIELDS})
        by_shape_totals.setdefault(shape, {f: 0 for f in FREE_TEXT_FIELDS})

        if messages is not None:
            # Mirrors run_single_iteration()'s effective_redaction_backend
            # logic exactly: messages-path cases force "regex" regardless
            # of the chosen backend, so free-text fields get NO enhanced
            # redaction at all here -- record as a guaranteed miss,
            # without spending a GLiNER call on it.
            forced_regex_count += 1
            for field in FREE_TEXT_FIELDS:
                by_shape_totals[shape][field] += 1
            continue

        regex_result = redact(flat_prompt)
        piped_result = redactor.redact(regex_result.redacted_text)

        for field in FREE_TEXT_FIELDS:
            gt_value = pii[field]
            by_shape_totals[shape][field] += 1
            if field_hit(piped_result.mapping, gt_value):
                by_shape_hits[shape][field] += 1

        if (idx + 1) % 20 == 0:
            print(f"[{idx + 1}/{iterations}] running...", flush=True)

    print(f"\n{forced_regex_count}/{iterations} cases ({100 * forced_regex_count / iterations:.1f}%) "
          f"were messages-path cases -- forced to regex-only, zero enhanced free-text redaction, "
          f"regardless of backend.\n")

    print(f"{'Shape':<28} {'n':>5} {'full_name':>11} {'company':>9} {'codename':>10} {'shape avg':>11}")
    grand_hits, grand_totals = 0, 0
    for shape in sorted(by_shape_totals):
        totals = by_shape_totals[shape]
        hits = by_shape_hits[shape]
        n = totals[FREE_TEXT_FIELDS[0]]
        per_field_pct = {f: (100 * hits[f] / totals[f] if totals[f] else 0.0) for f in FREE_TEXT_FIELDS}
        shape_total_hits = sum(hits.values())
        shape_total_n = sum(totals.values())
        shape_avg = 100 * shape_total_hits / shape_total_n if shape_total_n else 0.0
        grand_hits += shape_total_hits
        grand_totals += shape_total_n
        print(f"{shape:<28} {n:>5} {per_field_pct['full_name']:>10.2f}% {per_field_pct['company']:>8.2f}% "
              f"{per_field_pct['codename']:>9.2f}% {shape_avg:>10.2f}%")

    overall = 100 * grand_hits / grand_totals if grand_totals else 0.0
    print(f"\nOVERALL free-text recall across all {iterations} cases: {overall:.2f}%")
    print("(compare directly against the real benchmark's free_text_pii_recall_percent: "
          "70.0% at 60 iterations, 72.78% at 180)")


if __name__ == "__main__":
    main()
