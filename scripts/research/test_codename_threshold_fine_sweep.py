#!/usr/bin/env python3
"""
scripts/research/test_codename_threshold_fine_sweep.py
----------------------------------------
scripts/research/test_codename_threshold_sweep.py / scripts/research/test_codename_threshold_precision.py
tested a coarse grid (0.5 down to 0.05) and found: recall rises sharply
as threshold drops, but precision falls off a cliff starting around
0.15, and worse, some of that precision loss was a REAL correctness bug
(GLiNER catching the bracket-stripped inside of an already-placed regex
placeholder, which would corrupt it into a malformed nested token).
That bug is now FIXED in gliner_redact.py (_PLACEHOLDER_INNER_RE guard),
which changes the calculus: any remaining "unmatched" entity reported
here is now a genuine false positive (GLiNER tagging real but unrelated
text as if it were PII), not a placeholder-corruption artifact -- a much
more forgiving failure mode, since over-redacting ordinary text costs a
little compression/readability, not a round-trip/security bug.

This sweeps a FINER grid between the current default (0.30) and the
point where the coarse sweep started looking risky (0.15), through the
real, now-patched GlinerRedactor with the production 3-label set, to
find where the recall/precision curve actually bends -- rather than
jumping straight from 0.30 to 0.15 to 0.05 again.

Usage:
    cd ~/Desktop/tonst
    python3 scripts/research/test_codename_threshold_fine_sweep.py
"""
# Run from anywhere: make the repo root and benchmarks/ importable without `pip install -e .`.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "benchmarks"))
import random
import sys

sys.path.insert(0, ".")

from tonst.redact import redact
from tonst.gliner_redact import GlinerRedactor
from benchmark_tonst import generate_benchmark_case, INDUSTRY_CONFIGS

FREE_TEXT_FIELDS = ("full_name", "company", "codename")
THRESHOLDS = [0.30, 0.27, 0.25, 0.23, 0.22, 0.20, 0.18, 0.15]
SAMPLE_LIMIT = 6  # unmatched-entity examples to print per threshold


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def entity_matches(predicted_text: str, ground_truth: str) -> bool:
    p, g = normalize(predicted_text), normalize(ground_truth)
    return g in p or p in g


def field_hit(mapping: dict, ground_truth_value: str) -> bool:
    return any(entity_matches(v, ground_truth_value) for v in mapping.values())


def main():
    industries = list(INDUSTRY_CONFIGS.keys())
    modes = ["supervised", "unsupervised"]
    iterations = 180
    seed = 42

    cases = []
    for idx in range(iterations):
        local_rng = random.Random(seed + idx)
        industry = industries[idx % len(industries)]
        mode = modes[(idx // len(industries)) % len(modes)]
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(local_rng, industry, mode)
        if messages is not None:
            continue  # forced-regex-for-ollama-only path doesn't apply to a raw GlinerRedactor call anyway
        regex_result = redact(flat_prompt)
        cases.append((shape, pii, regex_result.redacted_text))

    print(f"Evaluating {len(cases)} non-messages-path cases across {len(THRESHOLDS)} thresholds "
          f"(post placeholder-guard fix).\n")
    print("Loading GLiNER (one-time)...", flush=True)
    GlinerRedactor(threshold=THRESHOLDS[0]).is_available()
    print("Ready.\n", flush=True)

    print(f"{'threshold':>9}  {'full_name':>10}  {'company':>9}  {'codename':>9}  "
          f"{'precision':>10}  {'entities/case':>14}  {'unmatched':>9}")
    all_unmatched_by_threshold = {}

    for threshold in THRESHOLDS:
        redactor = GlinerRedactor(threshold=threshold)
        hits = {f: 0 for f in FREE_TEXT_FIELDS}
        totals = {f: 0 for f in FREE_TEXT_FIELDS}
        total_entities = 0
        matched_entities = 0
        unmatched_samples = []

        for shape, pii, text in cases:
            result = redactor.redact(text)
            total_entities += result.entities_found
            gt_values = [pii[f] for f in FREE_TEXT_FIELDS]
            for field in FREE_TEXT_FIELDS:
                totals[field] += 1
                if field_hit(result.mapping, pii[field]):
                    hits[field] += 1
            for placeholder, span_value in result.mapping.items():
                if any(entity_matches(span_value, gt) for gt in gt_values):
                    matched_entities += 1
                else:
                    if len(unmatched_samples) < SAMPLE_LIMIT:
                        unmatched_samples.append((shape, placeholder, span_value, gt_values))

        pct = {f: (100 * hits[f] / totals[f] if totals[f] else 0.0) for f in FREE_TEXT_FIELDS}
        precision = 100 * matched_entities / total_entities if total_entities else 0.0
        mean_entities = total_entities / len(cases) if cases else 0.0
        unmatched = total_entities - matched_entities
        all_unmatched_by_threshold[threshold] = unmatched_samples

        print(f"{threshold:>9.2f}  {pct['full_name']:>9.2f}%  {pct['company']:>8.2f}%  "
              f"{pct['codename']:>8.2f}%  {precision:>9.2f}%  {mean_entities:>14.2f}  {unmatched:>9d}")

    print("\nSample unmatched (non-placeholder-inner) entities per threshold -- what's actually")
    print("getting over-tagged now that the placeholder-corruption bug is closed:\n")
    for threshold in THRESHOLDS:
        samples = all_unmatched_by_threshold[threshold]
        if not samples:
            continue
        print(f"=== threshold={threshold:.2f} ===")
        for shape, placeholder, span_value, gt_values in samples:
            print(f"    shape={shape:<22} tag={placeholder:<22} caught={span_value!r:<40} "
                  f"case_ground_truth={gt_values}")
        print()


if __name__ == "__main__":
    main()
