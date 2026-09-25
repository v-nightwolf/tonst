#!/usr/bin/env python3
"""
scripts/research/test_codename_label_wording.py
---------------------------------
scripts/research/inspect_supervised_codename_misses.py found that GLiNER's codename recall
tracks whether the ground-truth string literally contains a "Project"-style
cue word, not whether it's contextually a codename. Confirmed against
benchmarks/benchmark_tonst.py's own per-industry codename pools: industries whose
codenames never say "Project" (healthcare: Protocol/Trial/Study, finance:
Strategy/Book/Ledger, aerospace: Mission/Payload/Vehicle) get gutted
codename recall, while industries that do use "Project" get near-perfect
recall -- with the exact same sentence template surrounding every case.

This is a label-wording problem, not a GLiNER capability problem: the
current zero-shot label is "project codename" (see DEFAULT_LABELS in
tonst/gliner_redact.py), and GLiNER's zero-shot matching leans on lexical
overlap between the label string and the candidate span. This script
tests several GENERIC, domain-neutral label rewordings (none of them
hardcode any of the benchmark's actual codename strings -- that would be
overfitting the eval, not fixing the redactor) against the full 180-case
seed=42 set, reporting per-industry and overall codename recall for each
candidate label, to find one that generalizes across naming conventions
instead of just "Project X".

Usage:
    cd ~/Desktop/tonst
    python3 scripts/research/test_codename_label_wording.py
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
from tonst.gliner_redact import GlinerRedactor, DEFAULT_MODEL, DEFAULT_THRESHOLD
from benchmark_tonst import generate_benchmark_case, INDUSTRY_CONFIGS

LABEL_CANDIDATES = {
    "baseline (current)": "project codename",
    "broader project/program": "internal project, program, or initiative codename",
    "generic confidential id": "confidential internal codename or reference identifier",
    "mission/study/protocol": "internal mission, trial, protocol, study, or project codename",
    "bare word": "codename",
}


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

    # Pre-generate all cases once (and their post-regex text) so every
    # label candidate is evaluated on the IDENTICAL set of cases -- a fair
    # apples-to-apples comparison, not re-randomized per candidate.
    cases = []
    for idx in range(iterations):
        local_rng = random.Random(seed + idx)
        industry = industries[idx % len(industries)]
        mode = modes[(idx // len(industries)) % len(modes)]
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(local_rng, industry, mode)
        if messages is not None:
            continue  # forced-regex path; not relevant to GLiNER label quality
        regex_result = redact(flat_prompt)
        cases.append((industry, shape, pii["codename"], regex_result.redacted_text))

    print(f"Evaluating {len(cases)} non-messages-path cases across {len(LABEL_CANDIDATES)} label candidates.\n")

    print("Loading GLiNER (one-time)...", flush=True)
    from gliner import GLiNER
    model = GLiNER.from_pretrained(DEFAULT_MODEL)
    print("Ready.\n", flush=True)

    for label_name, label_text in LABEL_CANDIDATES.items():
        per_industry_hits = {ind: 0 for ind in industries}
        per_industry_totals = {ind: 0 for ind in industries}

        for industry, shape, codename, text in cases:
            entities = model.predict_entities(text, [label_text], threshold=DEFAULT_THRESHOLD)
            mapping = {e.get("label", ""): e.get("text", "") for e in entities}
            hit = any(entity_matches(v, codename) for v in mapping.values())
            per_industry_totals[industry] += 1
            if hit:
                per_industry_hits[industry] += 1

        total_hits = sum(per_industry_hits.values())
        total_n = sum(per_industry_totals.values())
        overall = 100 * total_hits / total_n if total_n else 0.0

        print(f"Label: {label_name!r} -> {label_text!r}")
        for ind in industries:
            n = per_industry_totals[ind]
            pct = 100 * per_industry_hits[ind] / n if n else 0.0
            print(f"    {ind:<15} {pct:>6.2f}%  (n={n})")
        print(f"    {'OVERALL':<15} {overall:>6.2f}%  (n={total_n})\n")


if __name__ == "__main__":
    main()
