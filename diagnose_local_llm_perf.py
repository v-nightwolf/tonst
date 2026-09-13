#!/usr/bin/env python3
"""
diagnose_local_llm_perf.py
---------------------------
Run this directly in a normal Terminal window on your Mac (NOT through any
sandboxed/bridge shell) with Ollama already running and gemma2:2b pulled.

Answers four questions before touching any tonst code:
  1. What quantization is gemma2:2b actually running at?
  2. Is Ollama using GPU (Metal) or falling back to CPU for it?
  3. Does calling Ollama repeatedly with the SAME instruction prefix get
     faster on calls 2+ (prefix/KV-cache reuse), or is every call cold?
  4. Does temperature=0 change latency, or just determinism?
  Bonus: how much of total latency is "thinking before the first word"
  (prefill) vs. "writing the answer" (decode)?

Usage:
    cd ~/Desktop/tonst
    python3 diagnose_local_llm_perf.py

Requires: `requests` (pip3 install requests --break-system-packages if missing)
"""
import json
import subprocess
import sys
import threading
import time
from typing import Optional

import requests

MODEL = "gemma2:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Same shape as tonst/local_model.py's COMPRESSION_INSTRUCTION -- static
# instruction first, variable text last -- so a real prefix-cache benefit
# (if Ollama gives one for your setup) should show up here too.
INSTRUCTION_TEMPLATE = (
    "Rewrite the following text to be as short as possible while preserving "
    "every fact, instruction, and constraint. Do not add commentary. "
    "Output only the rewritten text.\n\n---\n{text}"
)

SAMPLE_TEXTS = [
    "The quarterly billing cycle resets on the first day of each month, and "
    "any account that has not settled its outstanding balance within five "
    "business days of the reset will be flagged for manual review by the "
    "accounts receivable team.",
    "Customers on the Enterprise tier are entitled to a dedicated account "
    "manager, priority support response times of under one hour, and a "
    "custom service level agreement negotiated at the time of contract "
    "signing.",
    "When a support ticket is escalated to priority one status, the on-call "
    "engineer must acknowledge the page within ten minutes and provide a "
    "public status update within fifteen minutes of confirming the "
    "incident.",
    "All API keys issued to third-party integration partners are "
    "automatically rotated every ninety days, and any key that has not "
    "been used in the preceding thirty days is proactively revoked as a "
    "security precaution.",
]


def check_binary() -> None:
    try:
        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=5)
        print(f"[OK] ollama CLI found: {(out.stdout or out.stderr).strip()}")
    except FileNotFoundError:
        print("[FAIL] `ollama` not found on PATH. Run this in a normal Terminal window on your Mac, not inside any sandboxed/bridge shell.")
        sys.exit(1)
    except Exception as exc:
        print(f"[WARN] couldn't get ollama version: {exc}")


def check_quantization() -> None:
    print("\n=== 1. Quantization / model info (`ollama show gemma2:2b`) ===")
    try:
        out = subprocess.run(["ollama", "show", MODEL], capture_output=True, text=True, timeout=15)
        print(out.stdout or out.stderr)
    except Exception as exc:
        print(f"[WARN] `ollama show {MODEL}` failed: {exc}")


def check_gpu_during_call() -> None:
    print("\n=== 2. GPU vs CPU split (`ollama ps` while a call is in flight) ===")
    result_holder = {}

    def fire_call():
        t0 = time.perf_counter()
        try:
            requests.post(
                OLLAMA_URL,
                json={"model": MODEL, "prompt": "Write a 200 word essay about the ocean.", "stream": False},
                timeout=60,
            )
        except Exception as exc:
            result_holder["error"] = str(exc)
        result_holder["elapsed"] = time.perf_counter() - t0

    t = threading.Thread(target=fire_call)
    t.start()
    time.sleep(1.0)  # let the call actually start generating before we snapshot it
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10)
        print(out.stdout or out.stderr)
        print("(Look at the PROCESSOR column: '100% GPU' is good; any '% CPU' means it's not fully accelerated.)")
    except Exception as exc:
        print(f"[WARN] `ollama ps` failed: {exc}")
    t.join()
    print(f"(that background call took {result_holder.get('elapsed', 0):.2f}s total)")


def time_call(prompt: str, temperature: Optional[float] = None, stream: bool = False):
    payload = {"model": MODEL, "prompt": prompt, "stream": stream}
    if temperature is not None:
        payload["options"] = {"temperature": temperature}
    t0 = time.perf_counter()
    if stream:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60, stream=True)
        text = ""
        first_byte_s = None
        for line in resp.iter_lines():
            if not line:
                continue
            if first_byte_s is None:
                first_byte_s = time.perf_counter() - t0
            chunk = json.loads(line)
            text += chunk.get("response", "")
        elapsed = time.perf_counter() - t0
        return elapsed, first_byte_s, text
    else:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        elapsed = time.perf_counter() - t0
        return elapsed, None, resp.json().get("response", "")


def check_prefix_reuse() -> None:
    print("\n=== 3. Prefix/KV-cache reuse (same instruction, 4 different texts back-to-back) ===")
    times = []
    for i, text in enumerate(SAMPLE_TEXTS):
        prompt = INSTRUCTION_TEMPLATE.format(text=text)
        elapsed, _, _ = time_call(prompt)
        times.append(elapsed)
        print(f"  call {i + 1}: {elapsed * 1000:.0f}ms")
    if len(times) >= 2:
        rest_avg = sum(times[1:]) / len(times[1:])
        drop = (times[0] - rest_avg) / times[0] * 100
        print(f"  call 1 (cold) vs avg of calls 2-{len(times)}: {drop:+.1f}% change")
        print("  (a clearly faster call 2+ suggests real prefix-cache reuse; ~flat times means it isn't kicking in here)")


def check_temperature() -> None:
    print("\n=== 4. temperature=0 vs default: latency and output consistency ===")
    text = SAMPLE_TEXTS[0]
    prompt = INSTRUCTION_TEMPLATE.format(text=text)

    default_outputs, default_times = [], []
    for _ in range(3):
        elapsed, _, out = time_call(prompt, temperature=None)
        default_times.append(elapsed)
        default_outputs.append(out.strip())

    zero_outputs, zero_times = [], []
    for _ in range(3):
        elapsed, _, out = time_call(prompt, temperature=0.0)
        zero_times.append(elapsed)
        zero_outputs.append(out.strip())

    print(
        f"  default temp: avg {sum(default_times) / len(default_times) * 1000:.0f}ms, "
        f"{'IDENTICAL' if len(set(default_outputs)) == 1 else 'VARIED'} outputs across 3 runs"
    )
    print(
        f"  temperature=0: avg {sum(zero_times) / len(zero_times) * 1000:.0f}ms, "
        f"{'IDENTICAL' if len(set(zero_outputs)) == 1 else 'VARIED'} outputs across 3 runs"
    )


def check_prefill_vs_decode() -> None:
    print("\n=== Bonus: prefill (time-to-first-token) vs decode, via a streamed call ===")
    text = SAMPLE_TEXTS[1]
    prompt = INSTRUCTION_TEMPLATE.format(text=text)
    elapsed, ttft, out = time_call(prompt, stream=True)
    ttft = ttft or 0.0
    print(
        f"  time-to-first-token: {ttft * 1000:.0f}ms | total: {elapsed * 1000:.0f}ms | "
        f"decode-only: {(elapsed - ttft) * 1000:.0f}ms for ~{len(out.split())} words"
    )
    print(
        "  (if decode-only dominates, streaming truly can't help wall-clock time -- the model just "
        "takes that long to generate; if time-to-first-token is most of it, something is slow BEFORE "
        "generation even starts, e.g. model load/swap.)"
    )


if __name__ == "__main__":
    check_binary()
    check_quantization()
    check_gpu_during_call()
    check_prefix_reuse()
    check_temperature()
    check_prefill_vs_decode()
    print("\nDone. Paste this whole output back and we'll decide what's actually worth changing in tonst.")
