#!/usr/bin/env python3
"""
scripts/research/test_codename_threshold_sweep.py
-----------------------------------
scripts/research/test_codename_label_wording.py ruled out label rewording as a cheap fix
for supervised-shape codename recall: every generic alternative to the
current "project codename" label tied or lost to baseline, one collapsing
to 0% across every industry. This tests the other obvious lever: the
detection threshold (currently DEFAULT_THRESHOLD = 0.3 in
tonst/gliner_redact.py). Unlike the label-wording test, this goes through
the REAL GlinerRedactor with the REAL 3-label DEFAULT_LABELS set (matching
production exactly), varying only `threshold`, so we see both whether a
looser threshold recovers codename recall AND whether it costs anything
on full_name/company (which are already at 100% baseline -- a threshold
change can't improve them, only risk regressing them or flooding the
mapping with junk entities).

Also reports mean entities_found per case per threshold as a rough
proxy for over-triggering: a threshold that quadruples entity counts
without much recall gain is probably tagging noise, not more PII.

Usage:
    cd ~/Desktop/tonst
    python3 scripts/research/test_codename_threshold_sweep.py
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
THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]


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
            continue  # forced-regex path; not relevant here
        regex_result = redact(flat_prompt)
        cases.append((shape, pii, regex_result.redacted_text))

    print(f"Evaluating {len(cases)} non-messages-path cases across {len(THRESHOLDS)} thresholds.\n")
    print("Loading GLiNER (one-time)...", flush=True)
    GlinerRedactor(threshold=THRESHOLDS[0]).is_available()  # warms the shared module-level model cache
    print("Ready.\n", flush=True)

    print(f"{'threshold':>9}  {'full_name':>10}  {'company':>9}  {'codename':>9}  {'mean entities/case':>19}")
    for threshold in THRESHOLDS:
        redactor = GlinerRedactor(threshold=threshold)
        hits = {f: 0 for f in FREE_TEXT_FIELDS}
        totals = {f: 0 for f in FREE_TEXT_FIELDS}
        total_entities = 0

        for shape, pii, text in cases:
            result = redactor.redact(text)
            total_entities += result.entities_found
            for field in FREE_TEXT_FIELDS:
                totals[field] += 1
                if field_hit(result.mapping, pii[field]):
                    hits[field] += 1

        pct = {f: (100 * hits[f] / totals[f] if totals[f] else 0.0) for f in FREE_TEXT_FIELDS}
        mean_entities = total_entities / len(cases) if cases else 0.0
        print(f"{threshold:>9.2f}  {pct['full_name']:>9.2f}%  {pct['company']:>8.2f}%  "
              f"{pct['codename']:>8.2f}%  {mean_entities:>19.2f}")


if __name__ == "__main__":
    main()
