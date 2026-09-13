#!/usr/bin/env python3
"""
test_codename_threshold_precision.py
---------------------------------------
test_codename_threshold_sweep.py found that lowering GLiNER's detection
threshold from 0.3 to 0.05 nearly doubles codename recall (48.05% ->
79.22%) with full_name/company staying at or improving on 100% -- a much
stronger lever than any label rewording tested. But mean entities/case
also rose from 2.51 to 5.69, and that script only measured RECALL (does
the correct value show up somewhere in the mapping) -- it says nothing
about whether the extra entities are real or spurious.

This checks PRECISION at each threshold: for every entity GLiNER's
mapping actually produces, does its value match ANY of the case's 3
known ground-truth fields (full_name/company/codename)? An entity that
matches none of them is either (a) a genuinely spurious span -- GLiNER
tagging an ordinary word/phrase as if it were PII, which would get
needlessly redacted and could corrupt downstream text -- or (b) a
legitimate catch of something this benchmark's ground truth doesn't
track (e.g. a partial/duplicate span of a real entity). Prints a sample
of "unmatched" entities at the more promising thresholds so we can
actually see which one it is, instead of assuming.

Usage:
    cd ~/Desktop/tonst
    python3 test_codename_threshold_precision.py
"""
import random
import sys

sys.path.insert(0, ".")

from tonst.redact import redact
from tonst.gliner_redact import GlinerRedactor
from benchmark_tonst import generate_benchmark_case, INDUSTRY_CONFIGS

FREE_TEXT_FIELDS = ("full_name", "company", "codename")
THRESHOLDS = [0.30, 0.15, 0.10, 0.05]
SAMPLE_LIMIT = 12  # unmatched-entity examples to print per threshold


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def entity_matches(predicted_text: str, ground_truth: str) -> bool:
    p, g = normalize(predicted_text), normalize(ground_truth)
    return g in p or p in g


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
            continue
        regex_result = redact(flat_prompt)
        cases.append((shape, pii, regex_result.redacted_text))

    print(f"Evaluating {len(cases)} non-messages-path cases across {len(THRESHOLDS)} thresholds.\n")
    print("Loading GLiNER (one-time)...", flush=True)
    GlinerRedactor(threshold=THRESHOLDS[0]).is_available()
    print("Ready.\n", flush=True)

    for threshold in THRESHOLDS:
        redactor = GlinerRedactor(threshold=threshold)
        total_entities = 0
        matched_entities = 0
        unmatched_samples = []

        for shape, pii, text in cases:
            result = redactor.redact(text)
            gt_values = [pii[f] for f in FREE_TEXT_FIELDS]
            for placeholder, span_value in result.mapping.items():
                total_entities += 1
                if any(entity_matches(span_value, gt) for gt in gt_values):
                    matched_entities += 1
                else:
                    if len(unmatched_samples) < SAMPLE_LIMIT:
                        unmatched_samples.append((shape, placeholder, span_value, gt_values))

        unmatched = total_entities - matched_entities
        precision = 100 * matched_entities / total_entities if total_entities else 0.0
        print(f"=== threshold={threshold:.2f} ===")
        print(f"  total entities: {total_entities}   matched: {matched_entities}   "
              f"unmatched: {unmatched}   precision: {precision:.2f}%")
        if unmatched_samples:
            print(f"  sample unmatched entities (up to {SAMPLE_LIMIT}):")
            for shape, placeholder, span_value, gt_values in unmatched_samples:
                print(f"    shape={shape:<22} tag={placeholder:<22} caught={span_value!r:<40} "
                      f"case_ground_truth={gt_values}")
        print()


if __name__ == "__main__":
    main()
