#!/usr/bin/env python3
"""
scripts/research/stress_test_live_guardrails.py
--------------------------------
Targeted follow-up to scripts/research/capture_restoration_failures.py, which came back
with ZERO failures across 240 isolated compression/compaction
iterations for qwen2.5:1.5b and gemma2:2b. That result is inconclusive,
not clean: those isolated runs had --use-enhanced-redaction OFF, so the
only placeholders in the text were regex-created (email/card/phone/ip/
ssn) -- there was never a full_name/company/codename-style placeholder
present for compression/compaction to corrupt. The ORIGINAL guard-rail
bug this whole investigation found (see
research/colab-benchmark-findings.md, "compression/compaction
corrupting redaction placeholders") was specifically a model re-casing
a NAME-type placeholder's hex digest -- a shape the isolated runs
couldn't have reproduced.

This script tests the thing that actually matters for the real
pipeline: since GLiNER (not any generative model's own JSON-extraction
pass) is the planned redaction backend going forward, the relevant
question isn't "does qwen/gemma's OWN redaction interact badly with
its OWN compression" -- it's "does qwen/gemma's compression or
compaction corrupt a NAME/COMPANY/CODENAME-shaped placeholder", full
stop, regardless of what produced it. So this hand-constructs text
containing exactly those placeholder shapes (correctly formatted
`[[LABEL_xxxxxxxx]]` tokens, same format redact.py/redact_llm.py
produce) and calls the REAL local model repeatedly via the same
COMPRESSION_INSTRUCTION / COMPACTION_PROMPT prompts local_model.py and
compactor.py already use -- but bypasses their safe wrapper methods to
see the RAW model output before the guard rail filters it. That
separates two different questions that a black-box compress() call
can't distinguish:
  1. Does the raw model ever attempt to corrupt a placeholder at all?
  2. When it does, does placeholders_preserved()/_no_corrupted_placeholders()
     actually catch every instance (as designed), or does something slip
     through?
A "corruption attempted but caught" result is fine -- that's the guard
rail doing its job. A "corruption attempted and NOT caught" result is
the real problem, and is what actually explains a round_trip_restoration_failure.

Also samples the CPU% and RSS memory of the running 'ollama' process(es)
throughout each model's compression and compaction phases (separately),
via psutil, so the two candidate models' resource profiles can be
compared directly alongside their latency/corruption numbers. Requires
`pip install psutil`; if it's missing, the corruption test still runs
fine, just without resource numbers (a warning is printed instead of
failing).

Usage:
    pip install psutil   # once, if not already installed
    python3 scripts/research/stress_test_live_guardrails.py --model gemma2:2b --reps 20
    python3 scripts/research/stress_test_live_guardrails.py --model qwen2.5:1.5b --reps 20
"""
from __future__ import annotations
# Run from anywhere: make the repo root and benchmarks/ importable without `pip install -e .`.
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _ROOT)
_sys.path.insert(0, _os.path.join(_ROOT, "benchmarks"))
import argparse
import hashlib
import json
import threading
import time

import requests

try:
    import psutil
except ImportError:
    psutil = None

from tonst.local_model import COMPRESSION_INSTRUCTION, placeholders_preserved
from tonst.compactor import COMPACTION_PROMPT, _no_corrupted_placeholders

OLLAMA_URL = "http://localhost:11434/api/generate"


class ResourceMonitor:
    """Samples CPU% and RSS memory of every process whose name contains
    'ollama' (the server process itself, plus any per-model runner child
    process it spawns -- naming varies by Ollama version/platform, so we
    match broadly and sum across matches) at a fixed interval on a
    background thread, for as long as .running is True. Started/stopped
    around each model's test run so the two models' resource profiles
    aren't mixed together.

    Requires psutil (pip install psutil on the machine actually running
    Ollama -- not available in every sandboxed environment, so this
    degrades to a clear warning + zeroed-out stats rather than crashing
    if it's missing).
    """

    def __init__(self, poll_interval_s: float = 0.5):
        self.poll_interval_s = poll_interval_s
        self.samples: list[dict] = []  # each: {"t": elapsed_s, "cpu_percent": x, "rss_mb": y, "n_procs": n}
        self.running = False
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def _matching_procs(self):
        if psutil is None:
            return []
        procs = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                name = (p.info.get("name") or "").lower()
                if "ollama" in name:
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return procs

    def _run(self):
        procs = self._matching_procs()
        # Prime cpu_percent() -- first call after process discovery always
        # returns 0.0/garbage since there's no prior interval to measure against.
        for p in procs:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(self.poll_interval_s)
        while self.running:
            procs = self._matching_procs()
            total_cpu = 0.0
            total_rss_mb = 0.0
            n = 0
            for p in procs:
                try:
                    total_cpu += p.cpu_percent(None)
                    total_rss_mb += p.memory_info().rss / (1024 * 1024)
                    n += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self.samples.append({
                "t": round(time.time() - self._start_time, 2),
                "cpu_percent": round(total_cpu, 1),
                "rss_mb": round(total_rss_mb, 1),
                "n_procs": n,
            })
            time.sleep(self.poll_interval_s)

    def start(self):
        if psutil is None:
            return
        self.running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=self.poll_interval_s * 3)

    def summary(self) -> dict:
        if psutil is None:
            return {"available": False, "note": "psutil not installed -- run `pip install psutil` to capture CPU/memory."}
        if not self.samples:
            return {"available": False, "note": "no samples collected (no matching 'ollama' process found while running?)"}
        cpu_vals = [s["cpu_percent"] for s in self.samples]
        rss_vals = [s["rss_mb"] for s in self.samples]
        return {
            "available": True,
            "n_samples": len(self.samples),
            "cpu_percent_mean": round(sum(cpu_vals) / len(cpu_vals), 1),
            "cpu_percent_peak": round(max(cpu_vals), 1),
            "rss_mb_mean": round(sum(rss_vals) / len(rss_vals), 1),
            "rss_mb_peak": round(max(rss_vals), 1),
        }


def fake_placeholder(label: str, seed_text: str) -> str:
    """Same format redact.py/redact_llm.py produce: [[LABEL_ + 8 lowercase
    hex chars]]. Doesn't need to match a real hash's actual algorithm --
    only needs to be indistinguishable in SHAPE from a real one, since
    that's all placeholders_preserved()/_no_corrupted_placeholders() and
    the model itself can see.
    """
    digest = hashlib.sha256(seed_text.encode()).hexdigest()[:8]
    return f"[[{label}_{digest}]]"


NAME_PH = fake_placeholder("NAME", "Priya Malhotra")
COMPANY_PH = fake_placeholder("COMPANY", "NimbusCloud")
CODENAME_PH = fake_placeholder("CODENAME", "Project KubeShield")
EMAIL_PH = fake_placeholder("EMAIL", "priya.malhotra42@example.com")

COMPRESSION_SAMPLES = [
    (
        "compression_supervised_style",
        f"You are a Principal Cloud Infrastructure & SRE Support Assistant. "
        f"Lead engineer {NAME_PH} at {COMPANY_PH} (email: {EMAIL_PH}) submitted an urgent "
        f"ticket regarding {CODENAME_PH} connected to a production Kubernetes cluster outage. "
        f"Please classify the priority of this incident and draft a one-sentence response "
        f"explaining the immediate next steps the on-call team should take, given the urgency "
        f"and the customer-facing nature of the affected service.",
    ),
    (
        "compression_verbose_style",
        f"Hello team, I am dumping our raw notes from the war-room session for {CODENAME_PH} "
        f"at {COMPANY_PH}. Basically, {NAME_PH} (reachable at {EMAIL_PH}) noticed that whenever "
        f"telemetry packets originate from a specific host, our automated billing gateway "
        f"charges the card on file repeatedly due to a retry storm during this incident. "
        f"We need to know what immediate action {NAME_PH} should take, and whether this needs "
        f"to be escalated to the broader engineering organization before end of day.",
    ),
]

COMPACTION_SAMPLE = (
    f"user: [Turn 0] {NAME_PH} ({EMAIL_PH}) at {COMPANY_PH} on {CODENAME_PH}: host impacted.\n"
    f"assistant: [Turn 1] Logged diagnostics for {CODENAME_PH}.\n"
    f"user: [Turn 2] {NAME_PH} ({EMAIL_PH}) at {COMPANY_PH} on {CODENAME_PH}: host impacted.\n"
    f"assistant: [Turn 3] Logged diagnostics for {CODENAME_PH}.\n"
    f"user: [Turn 4] {NAME_PH} ({EMAIL_PH}) at {COMPANY_PH} on {CODENAME_PH}: host impacted.\n"
    f"assistant: [Turn 5] Logged diagnostics for {CODENAME_PH}.\n"
)


def raw_ollama_call(prompt: str, model: str, timeout: float = 20.0) -> str | None:
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.RequestException as exc:
        print(f"  [call failed: {exc}]")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--reps", type=int, default=20, help="Repetitions per sample text (model sampling isn't deterministic, so repeat to catch intermittent corruption).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = args.output or f"guardrail_stress_{args.model.replace(':', '_').replace('.', '_')}.json"

    if psutil is None:
        print("WARNING: psutil not installed -- CPU/memory will NOT be captured. Run `pip install psutil` and re-run for resource numbers.\n")

    results = {"model": args.model, "compression": [], "compaction": [], "resources": {}}

    print(f"=== Compression stress test: {args.model} ===")
    compression_monitor = ResourceMonitor()
    compression_wall_start = time.time()
    compression_monitor.start()
    for name, text in COMPRESSION_SAMPLES:
        for rep in range(args.reps):
            raw = raw_ollama_call(COMPRESSION_INSTRUCTION.format(text=text), args.model)
            if raw is None:
                continue
            raw = raw.strip()
            preserved = placeholders_preserved(text, raw)
            status = "OK (preserved)" if preserved else "CORRUPTED"
            print(f"  [{name} rep {rep}] {status}")
            if not preserved:
                results["compression"].append({"sample": name, "rep": rep, "original": text, "raw_model_output": raw})
    compression_monitor.stop()
    compression_wall_s = round(time.time() - compression_wall_start, 1)

    print(f"\n=== Compaction stress test: {args.model} ===")
    compaction_monitor = ResourceMonitor()
    compaction_wall_start = time.time()
    compaction_monitor.start()
    for rep in range(args.reps):
        raw = raw_ollama_call(COMPACTION_PROMPT.format(text=COMPACTION_SAMPLE), args.model)
        if raw is None:
            continue
        raw = raw.strip()
        ok = _no_corrupted_placeholders(COMPACTION_SAMPLE, raw)
        status = "OK (no corruption)" if ok else "CORRUPTED"
        print(f"  [compaction rep {rep}] {status}")
        if not ok:
            results["compaction"].append({"rep": rep, "original": COMPACTION_SAMPLE, "raw_model_output": raw})
    compaction_monitor.stop()
    compaction_wall_s = round(time.time() - compaction_wall_start, 1)

    n_compression_corrupt = len(results["compression"])
    n_compaction_corrupt = len(results["compaction"])
    total_compression_calls = len(COMPRESSION_SAMPLES) * args.reps
    total_compaction_calls = args.reps

    compression_res = compression_monitor.summary()
    compaction_res = compaction_monitor.summary()
    compression_res["wall_clock_s"] = compression_wall_s
    compaction_res["wall_clock_s"] = compaction_wall_s
    results["resources"] = {"compression": compression_res, "compaction": compaction_res}

    print(f"\n=== SUMMARY: {args.model} ===")
    print(f"Compression: {n_compression_corrupt}/{total_compression_calls} raw outputs corrupted a placeholder")
    print(f"Compaction:  {n_compaction_corrupt}/{total_compaction_calls} raw outputs corrupted a placeholder")
    print(
        "\nNote: 'corrupted' here means the RAW model output failed the guard rail check -- "
        "it does NOT mean a bad result would have shipped. local_model.py's compress() and "
        "compactor.py's summarize() both already reject output that fails this exact check and "
        "fall back to the original/no-summary. This script exists to measure the ATTEMPT rate "
        "directly, since the guard rail hides it in normal use."
    )
    if compression_res.get("available"):
        print(
            f"\nCompression resource usage ({compression_wall_s}s wall, {total_compression_calls} calls): "
            f"CPU mean {compression_res['cpu_percent_mean']}% / peak {compression_res['cpu_percent_peak']}%, "
            f"RSS mean {compression_res['rss_mb_mean']}MB / peak {compression_res['rss_mb_peak']}MB "
            f"(summed across {compression_res['n_samples']} samples of matching 'ollama' process(es))"
        )
    else:
        print(f"\nCompression resource usage: unavailable ({compression_res.get('note')})")
    if compaction_res.get("available"):
        print(
            f"Compaction resource usage  ({compaction_wall_s}s wall, {total_compaction_calls} calls): "
            f"CPU mean {compaction_res['cpu_percent_mean']}% / peak {compaction_res['cpu_percent_peak']}%, "
            f"RSS mean {compaction_res['rss_mb_mean']}MB / peak {compaction_res['rss_mb_peak']}MB"
        )
    else:
        print(f"Compaction resource usage: unavailable ({compaction_res.get('note')})")

    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results (including resource usage) written to {output}")


if __name__ == "__main__":
    main()
