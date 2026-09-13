#!/usr/bin/env python3
"""
diagnose_gliner_regex_interaction.py
-------------------------------------
Tests a specific hypothesis about why in-pipeline GLiNER recall
(72.78% free-text recall on a real 180-iteration benchmark_tonst.py
run, 2026-09-13) sits well below what the standalone per-field sanity
check would predict (full_name 100%, company 98.89%, codename 74.44%
-- a ~91% weighted average). This gap held essentially flat across a
3x sample-size increase (70.0% at 60 iterations, 72.78% at 180), which
rules out sample noise as the explanation -- something structural is
different between how the standalone check measured GLiNER and how
the real pipeline actually calls it.

The one real difference: the standalone check (gliner_sanity_check.py)
runs GLiNER on RAW, unredacted text. The real pipeline never does that
-- redact.redact_with_llm() ALWAYS runs regex first, so GLiNER only
ever sees text that already contains [[EMAIL_xxxxxxxx]]-style
placeholders wherever an email/phone/card/SSN/IP used to be, often
sitting right next to (or in the same sentence as) the free-text PII
GLiNER is supposed to catch. This script tests, directly, on matched
pairs of the SAME generated cases: does GLiNER's recall differ between
raw text and regex-pre-redacted text?

Uses the identical "loose" matching logic as gliner_sanity_check.py
(entity_matches/normalize) so numbers here are apples-to-apples with
the standalone check's own definition of recall.

Usage:
    cd ~/Desktop/tonst
    python3 diagnose_gliner_regex_interaction.py
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
    """Case/whitespace-insensitive containment either direction -- same
    definition gliner_sanity_check.py uses for "loose recall"."""
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
    n_per_cell = 15  # 6 industries x 2 modes x 15 = 180, same size as the real run

    raw_hits = {f: 0 for f in FREE_TEXT_FIELDS}
    piped_hits = {f: 0 for f in FREE_TEXT_FIELDS}
    totals = {f: 0 for f in FREE_TEXT_FIELDS}

    seed = 1000  # distinct range from other diagnostics/the real benchmark's seed=42
    done = 0
    total_cases = len(industries) * len(modes) * n_per_cell

    for industry in industries:
        for mode in modes:
            for _ in range(n_per_cell):
                rng = random.Random(seed)
                seed += 1
                shape, text, pii, parts, messages = generate_benchmark_case(rng, industry, mode)

                # Condition A: GLiNER on RAW text -- matches how
                # gliner_sanity_check.py actually measured recall.
                raw_result = redactor.redact(text)

                # Condition B: GLiNER on regex-redacted text -- matches
                # how the real pipeline calls it, via
                # redact.redact_with_llm() (regex always runs first).
                regex_result = redact(text)
                piped_result = redactor.redact(regex_result.redacted_text)

                for field in FREE_TEXT_FIELDS:
                    gt_value = pii[field]
                    totals[field] += 1
                    if field_hit(raw_result.mapping, gt_value):
                        raw_hits[field] += 1
                    if field_hit(piped_result.mapping, gt_value):
                        piped_hits[field] += 1

                done += 1
                if done % 20 == 0 or done == total_cases:
                    print(f"[{done}/{total_cases}] running...", flush=True)

    print(f"\n{'Field':<10} {'Raw-text recall':>16} {'Post-regex recall':>18} {'Delta':>8}")
    for field in FREE_TEXT_FIELDS:
        raw_pct = 100 * raw_hits[field] / totals[field]
        piped_pct = 100 * piped_hits[field] / totals[field]
        print(f"{field:<10} {raw_pct:>15.2f}% {piped_pct:>17.2f}% {piped_pct - raw_pct:>+7.2f}pp")

    raw_overall = 100 * sum(raw_hits.values()) / sum(totals.values())
    piped_overall = 100 * sum(piped_hits.values()) / sum(totals.values())
    print(f"{'OVERALL':<10} {raw_overall:>15.2f}% {piped_overall:>17.2f}% {piped_overall - raw_overall:>+7.2f}pp")

    if piped_overall < raw_overall - 5:
        print(
            "\nCONFIRMED (or at least strongly suggestive): running GLiNER on "
            "regex-pre-redacted text measurably hurts recall vs. raw text. The "
            "standalone sanity check's numbers describe GLiNER's ceiling on "
            "clean text, not its real in-pipeline behavior -- the regex "
            "placeholders sitting next to free-text PII appear to disrupt "
            "GLiNER's context modeling. Worth considering: run GLiNER on the "
            "ORIGINAL text before regex redaction runs, then merge the two "
            "mappings, instead of layering GLiNER on top of regex's output "
            "(the opposite order redact_with_llm() currently uses)."
        )
    else:
        print(
            "\nNOT confirmed here -- raw vs. post-regex recall are close for "
            "GLiNER specifically. The in-pipeline-vs-standalone gap likely has "
            "a different cause (e.g. worth double-checking benchmark_tonst.py's "
            "own ground-truth check against this script's field-level breakdown "
            "for a mismatch in what counts as a 'hit')."
        )


if __name__ == "__main__":
    main()
