#!/usr/bin/env python3
"""
inspect_supervised_codename_misses.py
---------------------------------------
diagnose_gliner_by_shape.py found something new and separate from the
messages-path routing bug: codename recall in supervised_extraction
(24.32%) and supervised_few_shot (26.42%) is far below full_name/company
in the SAME rows (100%/100%) and far below codename recall in the
unsupervised shapes (91-100%). These supervised cases are NOT routed
through the forced-regex path, so this is GLiNER actually missing
codenames it's given a fair shot at.

This prints the actual text + ground-truth codename + what GLiNER's
mapping caught for a sample of MISSED supervised-shape cases, so we can
see the actual textual pattern instead of guessing at it.

Usage:
    cd ~/Desktop/tonst
    python3 inspect_supervised_codename_misses.py
"""
import random
import sys

sys.path.insert(0, ".")

from tonst.redact import redact
from tonst.gliner_redact import GlinerRedactor
from benchmark_tonst import generate_benchmark_case, INDUSTRY_CONFIGS


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
    seed = 42  # same scheme as diagnose_gliner_by_shape.py / the real benchmark

    misses = []
    hits_sample = []

    for idx in range(iterations):
        local_rng = random.Random(seed + idx)
        industry = industries[idx % len(industries)]
        mode = modes[(idx // len(industries)) % len(modes)]
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(local_rng, industry, mode)

        if shape not in ("supervised_extraction", "supervised_few_shot"):
            continue
        if messages is not None:
            continue  # forced-regex path, not relevant to this question

        regex_result = redact(flat_prompt)
        piped_result = redactor.redact(regex_result.redacted_text)

        codename = pii["codename"]
        hit = field_hit(piped_result.mapping, codename)
        record = {
            "idx": idx,
            "shape": shape,
            "codename": codename,
            "text_after_regex": regex_result.redacted_text,
            "gliner_mapping": piped_result.mapping,
        }
        if hit:
            if len(hits_sample) < 3:
                hits_sample.append(record)
        else:
            misses.append(record)

    print(f"Found {len(misses)} supervised-shape codename misses (of the sampled cases).\n")
    print("=" * 100)
    print("SAMPLE OF MISSES (up to 6):")
    print("=" * 100)
    for record in misses[:6]:
        print(f"\n--- idx={record['idx']} shape={record['shape']} ground_truth_codename={record['codename']!r} ---")
        # Print only the region of text around the codename mention, if findable, else the first 500 chars.
        text = record["text_after_regex"]
        pos = text.lower().find(record["codename"].lower())
        if pos >= 0:
            start = max(0, pos - 150)
            end = min(len(text), pos + len(record["codename"]) + 150)
            print(f"...context around codename mention...\n{text[start:end]!r}")
        else:
            print(f"(codename string not found verbatim in post-regex text -- first 300 chars:)\n{text[:300]!r}")
        print(f"GLiNER mapping produced: {record['gliner_mapping']}")

    print("\n" + "=" * 100)
    print("SAMPLE OF HITS (up to 3, for contrast):")
    print("=" * 100)
    for record in hits_sample:
        print(f"\n--- idx={record['idx']} shape={record['shape']} ground_truth_codename={record['codename']!r} ---")
        text = record["text_after_regex"]
        pos = text.lower().find(record["codename"].lower())
        if pos >= 0:
            start = max(0, pos - 150)
            end = min(len(text), pos + len(record["codename"]) + 150)
            print(f"...context around codename mention...\n{text[start:end]!r}")
        print(f"GLiNER mapping produced: {record['gliner_mapping']}")


if __name__ == "__main__":
    main()
