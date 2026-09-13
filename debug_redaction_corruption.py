#!/usr/bin/env python3
"""
debug_redaction_corruption.py
------------------------------
Both gemma2:2b full-pipeline sanity runs (--workers 4 AND --workers 1)
showed a much higher round_trip_restoration_failure rate (10% and 55%
respectively) than the earlier direct guard-rail stress test
(stress_test_live_guardrails.py: 0/40 compression corruptions), and the
--workers 1 run (no contention at all) also showed avg_redacted_fields_
per_call nearly doubling (8.9 vs. the historical ~5.2-5.6 baseline for
5 regex fields + occasional free-text catches). That combination points
somewhere the earlier stress test never looked: LLMRedactor.redact()
itself, using gemma2:2b, may be over-generating entities -- finding
more spans per call than the 3 real free-text fields (full_name/
company/codename) -- which gives compression more placeholders to
preserve exactly per call and more surface area for one to get mangled.
This was never tested: the earlier stress test used a FIXED, hand-built
set of placeholders, never gemma2:2b's own redaction output feeding
into gemma2:2b's own compression.

This script reproduces the EXACT same benchmark cases run_benchmark()
generates for a given --iterations/--seed (same industry/mode cycling
logic copied directly from benchmark_tonst.py, idx 0..N-1) but
SEQUENTIALLY (no threading, so contention/--workers is not a factor)
and with instrumentation TonstClient doesn't expose:
  - monkeypatches tonst.redact_llm's model-call hook to log gemma's RAW
    redaction response text, not just the parsed entity count
  - monkeypatches tonst.local_model's HTTP session to log gemma's RAW
    compression response text
  - for every case, records report.redacted_fields (from
    OptimizationReport) alongside the raw texts
  - for every round_trip_restoration_failure, dumps the complete
    transcript: original text, raw redaction response, raw compression
    response, sent_prompt, final response -- so the actual failure mode
    is visible, not just a count.

Usage:
    python3 debug_redaction_corruption.py --model gemma2:2b --iterations 60
"""
from __future__ import annotations
import argparse
import json
import random
import time

from benchmark_tonst import INDUSTRY_CONFIGS, generate_benchmark_case, wait_for_ollama_ready
from tonst import TonstClient
import tonst.redact_llm as redact_llm_mod
import tonst.local_model as local_model_mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma2:2b")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = args.output or f"redaction_corruption_debug_{args.model.replace(':', '_').replace('.', '_')}.json"

    wait_for_ollama_ready(args.model)

    # --- Instrumentation: capture raw model text per call, not just the
    # parsed/guard-railed result each module normally returns. ---
    capture = {"redaction_raw": [], "compression_raw": []}

    real_redact_call = redact_llm_mod._default_ollama_call

    def logging_redact_call(prompt, model, timeout):
        raw = real_redact_call(prompt, model, timeout)
        capture["redaction_raw"].append(raw)
        return raw

    redact_llm_mod._default_ollama_call = logging_redact_call

    real_post = local_model_mod._SESSION.post

    def logging_post(*a, **kw):
        resp = real_post(*a, **kw)
        try:
            capture["compression_raw"].append(resp.json().get("response", ""))
        except Exception:
            capture["compression_raw"].append(None)
        return resp

    local_model_mod._SESSION.post = logging_post

    industries, modes = list(INDUSTRY_CONFIGS.keys()), ["supervised", "unsupervised"]

    results = []
    entity_overgen_count = 0

    for idx in range(args.iterations):
        local_rng = random.Random(args.seed + idx)
        industry = industries[idx % len(industries)]
        mode = modes[(idx // len(industries)) % len(modes)]
        shape, flat_prompt, pii, parts, messages = generate_benchmark_case(local_rng, industry, mode)

        capture["redaction_raw"].clear()
        capture["compression_raw"].clear()

        sent_holder = []

        def mock_paid_api(trimmed_prompt: str) -> str:
            sent_holder.append(trimmed_prompt)
            tokens = [w for w in trimmed_prompt.split() if w.startswith("[[") and "]]" in w]
            return f"Processed request successfully. Reference tokens: {' '.join(tokens[:4])}"

        client = TonstClient(
            call_fn=mock_paid_api,
            use_local_compression=True,
            use_enhanced_redaction=(True if messages is None else False),
            use_history_compaction=False,
            local_model=args.model,
            compaction_token_threshold=100,
        )

        t0 = time.perf_counter()
        if messages is not None:
            response, report = client.query_messages(messages, keep_last_n=3)
        elif parts is not None:
            response, report = client.query_structured(parts)
        else:
            response, report = client.query(flat_prompt)
        elapsed_s = time.perf_counter() - t0

        sent_prompt = sent_holder[-1] if sent_holder else ""
        rt_fail = "[[" in response and "]]" in response

        free_text_keys = ("full_name", "company", "codename")
        # crude over-generation signal: redacted_fields well above the
        # structured-field count (5: email/card/phone/ip/ssn) + the 3
        # real free-text fields (8 max legitimate placeholders per case,
        # some cases repeat a field like codename multiple times so a
        # few extra is normal -- but consistently far above 8 is not).
        overgenerated = report.redacted_fields > 10
        if overgenerated:
            entity_overgen_count += 1

        record = {
            "idx": idx,
            "industry": industry,
            "mode": mode,
            "shape": shape,
            "redacted_fields": report.redacted_fields,
            "redaction_ms": report.redaction_ms,
            "compression_ms": report.compression_ms,
            "elapsed_s": round(elapsed_s, 2),
            "rt_fail": rt_fail,
            "overgenerated": overgenerated,
        }

        status = "CORRUPTED" if rt_fail else ("OVERGEN" if overgenerated else "ok")
        print(f"[{idx}] {industry}/{mode}/{shape} redacted_fields={report.redacted_fields} rt_fail={rt_fail} -> {status}", flush=True)

        if rt_fail or overgenerated:
            record["ground_truth_pii"] = pii
            record["original_text"] = flat_prompt if flat_prompt else json.dumps(messages)
            record["redaction_raw_responses"] = list(capture["redaction_raw"])
            record["compression_raw_responses"] = list(capture["compression_raw"])
            record["sent_prompt"] = sent_prompt
            record["final_response"] = response

        results.append(record)

    redact_llm_mod._default_ollama_call = real_redact_call
    local_model_mod._SESSION.post = real_post

    n_fail = sum(1 for r in results if r["rt_fail"])
    n_overgen = sum(1 for r in results if r["overgenerated"])
    print(f"\n=== SUMMARY ({args.model}, {args.iterations} iterations, sequential, no contention) ===")
    print(f"round_trip_restoration_failures: {n_fail}/{args.iterations}")
    print(f"overgenerated (redacted_fields > 10): {n_overgen}/{args.iterations}")
    avg_fields = sum(r["redacted_fields"] for r in results) / len(results) if results else 0
    print(f"avg redacted_fields_per_call: {avg_fields:.2f}")

    with open(output, "w") as f:
        json.dump({"model": args.model, "iterations": args.iterations, "results": results}, f, indent=2)
    print(f"\nFull results (with raw model text for every failure/overgen case) written to {output}")


if __name__ == "__main__":
    main()
