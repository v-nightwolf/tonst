#!/usr/bin/env python3
"""
gliner_sanity_check.py
-----------------------
Standalone sanity check: how well does a zero-shot GLiNER model catch
the free-text PII fields (full_name, company, codename) that tonst's
regex layer does NOT cover -- the exact gap redact_llm.py's Ollama-based
enhanced redaction was built for -- before committing to a GLiNER
integration.

Reuses the SAME synthetic-case generator as benchmark_tonst.py
(generate_benchmark_case / INDUSTRY_CONFIGS) rather than reimplementing
it, so results are directly comparable to the existing Ollama-based
numbers in research/colab-benchmark-findings.md, on the same 6
industries x 2 paradigms shape. Must be run from the same directory as
benchmark_tonst.py (imports it directly).

Why run this before integrating: redact_llm.py (Ollama + llama3.2:1b)
has an entire multi-session history of bugs that all trace back to it
being a GENERATIVE model coerced into producing parseable structured
output (greedy JSON-array regex failures, bare-object responses, the
`format: "json"` grammar-decoding hang, placeholder re-casing). GLiNER
is a token-classification model -- it returns spans/offsets into the
ORIGINAL text, never regenerates it -- so an entire category of those
bugs can't occur by construction. This script measures whether that
theoretical advantage holds up on tonst's actual synthetic data: recall
on full_name/company/codename per industry and per prompt shape, plus
real CPU latency (GLiNER doesn't need a GPU, which is the point -- this
sanity check is meant to run on a plain CPU machine/Colab CPU runtime,
no T4 required).

Usage:
    pip install gliner
    python3 gliner_sanity_check.py --iterations-per-cell 15

First run downloads the model checkpoint from Hugging Face (small model,
should be well under 1GB) -- needs real internet access. If you're
running this through a sandboxed shell (a Cowork device-bridge session,
a locked-down CI box) and pip/model-download fails with a 403 from a
proxy, that's a network policy blocking package/model registries, not a
bug in this script -- run it from a normal local terminal or in Colab
instead, both of which already work for the rest of this project's
Ollama benchmarking.
"""
from __future__ import annotations
import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List

from benchmark_tonst import INDUSTRY_CONFIGS, generate_benchmark_case

# Fields NOT covered by tonst's regex layer or by benchmark_tonst.py's
# own gt_keys leak-check (email/card/phone/ip) -- these are exactly what
# redact_llm.py's free-text LLM pass, and now GLiNER, exist to catch.
FREE_TEXT_FIELDS = ("full_name", "company", "codename")

# Default zero-shot labels handed to GLiNER at inference time. Chosen to
# map as directly as possible onto tonst's existing entity types without
# any per-industry hand-tuning. Overridable via --name-label/
# --company-label/--codename-label (see main()) -- the first sanity-check
# run (2026-09-11) found codename recall dropping to 18-33% on the
# realistic, low-structure prompt shapes (supervised + verbose_dump),
# vs. 88.57% on the one shape that literally spells out "Project: X" in
# the text. Before concluding GLiNER can't find codenames, it's worth
# testing whether a more descriptive label closes some of that gap on
# the hard shapes, since changing a label string is nearly free compared
# to changing model size.
DEFAULT_FIELD_TO_LABEL = {
    "full_name": "person name",
    "company": "company name",
    "codename": "project codename",
}


@dataclass
class FieldStats:
    loose_hits: int = 0   # ground-truth text found in SOME predicted span, regardless of that span's label
    strict_hits: int = 0  # ground-truth text found in a predicted span carrying the EXPECTED label
    total: int = 0

    def as_dict(self) -> dict:
        return {
            "loose_recall_percent": round(100.0 * self.loose_hits / self.total, 2) if self.total else 0.0,
            "strict_recall_percent": round(100.0 * self.strict_hits / self.total, 2) if self.total else 0.0,
            "total": self.total,
        }


def normalize(s: str) -> str:
    return " ".join(s.lower().split())


def entity_matches(predicted_text: str, ground_truth: str) -> bool:
    """
    Case/whitespace-insensitive containment either direction -- GLiNER
    sometimes returns a slightly wider or narrower span than the exact
    ground-truth string (e.g. trailing punctuation, or "at NimbusCloud"
    vs "NimbusCloud").
    """
    p, g = normalize(predicted_text), normalize(ground_truth)
    return g in p or p in g


def new_stats_map() -> Dict[str, FieldStats]:
    return {f: FieldStats() for f in FREE_TEXT_FIELDS}


def merge_field_report(stats_map: Dict[str, FieldStats]) -> dict:
    return {f: s.as_dict() for f, s in stats_map.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iterations-per-cell", type=int, default=15,
        help="Iterations per (industry x paradigm) cell -- default 15x6x2=180 total, half the size of the original 360-call Ollama benchmark.",
    )
    parser.add_argument("--model", type=str, default="urchade/gliner_small-v2.1")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="gliner_sanity_report.json")
    parser.add_argument("--name-label", type=str, default=DEFAULT_FIELD_TO_LABEL["full_name"],
                         help="Zero-shot label for full_name. Default is already ~100%% recall; not expected to need changing.")
    parser.add_argument("--company-label", type=str, default=DEFAULT_FIELD_TO_LABEL["company"],
                         help="Zero-shot label for company -- try alternatives like 'organization' if recall doesn't improve on other levers.")
    parser.add_argument("--codename-label", type=str, default=DEFAULT_FIELD_TO_LABEL["codename"],
                         help="Zero-shot label for codename -- the weak field (18-33%% on low-structure prompt shapes in the first run). Try e.g. 'confidential project or mission codename'.")
    args = parser.parse_args()

    field_to_label = {
        "full_name": args.name_label,
        "company": args.company_label,
        "codename": args.codename_label,
    }
    gliner_labels = list(field_to_label.values())

    print(f"Loading GLiNER model {args.model} (CPU) -- first run also downloads the checkpoint...", flush=True)
    print(f"Labels: {field_to_label}", flush=True)
    from gliner import GLiNER  # deferred import so --help works without the package installed

    t_load = time.perf_counter()
    model = GLiNER.from_pretrained(args.model)
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    industries = list(INDUSTRY_CONFIGS.keys())
    modes = ["supervised", "unsupervised"]

    overall_stats = new_stats_map()
    by_industry_stats: Dict[str, Dict[str, FieldStats]] = {ind: new_stats_map() for ind in industries}
    by_shape_stats: Dict[str, Dict[str, FieldStats]] = {}

    latencies_ms: List[float] = []
    per_industry_latency: Dict[str, List[float]] = {ind: [] for ind in industries}
    total_predicted_spans = 0
    unmatched_span_count = 0  # predicted spans matching none of the 3 ground-truth values this call -- rough over-generation signal, not a real precision score

    total_cases = len(industries) * len(modes) * args.iterations_per_cell
    done = 0
    seed_counter = args.seed

    for industry in industries:
        for mode in modes:
            for _ in range(args.iterations_per_cell):
                rng = random.Random(seed_counter)
                seed_counter += 1
                shape, flat_prompt, pii, _parts, _messages = generate_benchmark_case(rng, industry, mode)
                by_shape_stats.setdefault(shape, new_stats_map())

                t0 = time.perf_counter()
                entities = model.predict_entities(flat_prompt, gliner_labels, threshold=args.threshold)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                latencies_ms.append(elapsed_ms)
                per_industry_latency[industry].append(elapsed_ms)
                total_predicted_spans += len(entities)

                matched_any_gt = [False] * len(entities)
                for field_name in FREE_TEXT_FIELDS:
                    gt_value = pii[field_name]
                    expected_label = field_to_label[field_name]
                    loose_hit = False
                    strict_hit = False
                    for idx, ent in enumerate(entities):
                        if entity_matches(ent["text"], gt_value):
                            loose_hit = True
                            matched_any_gt[idx] = True
                            if ent.get("label") == expected_label:
                                strict_hit = True
                    for stats in (overall_stats[field_name], by_industry_stats[industry][field_name], by_shape_stats[shape][field_name]):
                        stats.total += 1
                        if loose_hit:
                            stats.loose_hits += 1
                        if strict_hit:
                            stats.strict_hits += 1

                unmatched_span_count += sum(1 for m in matched_any_gt if not m)

                done += 1
                if done % 20 == 0 or done == total_cases:
                    print(f"[{done}/{total_cases}] running...", flush=True)

    def pct(values: List[float], p: float) -> float:
        s = sorted(values)
        k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p / 100.0))))
        return round(s[k], 2)

    report = {
        "model": args.model,
        "threshold": args.threshold,
        "labels": field_to_label,
        "iterations_per_cell": args.iterations_per_cell,
        "total_cases": total_cases,
        "overall_recall_by_field": merge_field_report(overall_stats),
        "recall_by_industry": {ind: merge_field_report(fs) for ind, fs in by_industry_stats.items()},
        "recall_by_shape": {shape: merge_field_report(fs) for shape, fs in by_shape_stats.items()},
        "latency_ms": {
            "p50": pct(latencies_ms, 50),
            "p90": pct(latencies_ms, 90),
            "p99": pct(latencies_ms, 99),
            "mean": round(statistics.mean(latencies_ms), 2),
            "max": round(max(latencies_ms), 2),
        },
        "latency_by_industry_mean_ms": {ind: round(statistics.mean(v), 2) for ind, v in per_industry_latency.items()},
        "over_generation": {
            "total_predicted_spans": total_predicted_spans,
            "spans_matching_no_ground_truth_field": unmatched_span_count,
            "note": (
                "A span here that matches no ground-truth field isn't necessarily "
                "wrong -- GLiNER may be correctly tagging a real entity the "
                "synthetic generator doesn't track (e.g. a name mentioned only in "
                "jargon text). Treat this as a rough over-generation signal, not "
                "a true precision score -- getting a real precision number would "
                "require manually annotating a sample of these extra spans."
            ),
        },
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "labels": report["labels"],
        "overall_recall_by_field": report["overall_recall_by_field"],
        # recall_by_shape matters more than the overall/by-industry blend: the
        # first run found the highly-structured "Project: X"-style shape
        # scoring 88%+ on codename/company while the low-structure shapes
        # (closer to how real chat prompts read) scored 18-33%/52-65% --
        # printing this every run so that gap isn't accidentally averaged away.
        "recall_by_shape": report["recall_by_shape"],
        "latency_ms": report["latency_ms"],
    }, indent=2))
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
