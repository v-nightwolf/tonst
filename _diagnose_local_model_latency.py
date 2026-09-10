"""
_diagnose_local_model_latency.py
---------------------------------
TEMPORARY diagnostic script -- not part of the library, not imported by
anything, safe to delete once you're done. Written to directly answer
the question behind the Colab benchmark's ~16s p50 latency_overhead_ms:
were LLMRedactor.redact() and LocalCompressor.compress() genuinely
timing out against a real Ollama, and if so, is it per-call model cost
or contention across concurrent workers?

Why this can't be answered from the cloud/sandbox side: there is no
Ollama installed or reachable here (`curl localhost:11434` refuses the
connection on this machine). This script is meant to run wherever
Ollama + the GPU actually are -- i.e. your Colab notebook, right after
`ollama serve` and `ollama pull llama3.2:1b` -- not here.

What it does:
    1. Enables DEBUG logging on tonst.redact_llm / tonst.local_model /
       tonst.compactor -- these three modules were just instrumented to
       log, per call, either "ok elapsed_ms=X" or a specific failure
       classification (TIMED OUT vs. FAILED/ConnectionError vs. bad
       HTTP status), instead of collapsing every failure into the same
       silent fail-soft result. This directly answers "is it timing out
       vs. something else" without guessing.
    2. Times ONE single, uncontended call to LLMRedactor.redact() and
       ONE to LocalCompressor.compress(), back to back, single-threaded
       -- no other load on Ollama. This isolates genuine per-call model
       cost from any contention effect.
    3. Fires N concurrent calls (default 4, matching the benchmark's
       worker count) through a thread pool, so you can compare
       uncontended latency (step 2) against contended latency. If
       uncontended is fast but contended is slow/timing out, the fix is
       concurrency (OLLAMA_NUM_PARALLEL vs. worker count), not the
       timeout value. If even the uncontended call is close to or past
       8s, the fix is a longer timeout tuned to that real number (and
       maybe a smaller/faster model).

Usage (in Colab, after Ollama is running and the model is pulled):
    python3 _diagnose_local_model_latency.py
    python3 _diagnose_local_model_latency.py --workers 8   # stress harder
"""

import argparse
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
# Quiet urllib3's own per-connection noise; we only want tonst's own
# outcome/timing lines from the loggers added in this diagnostic pass.
logging.getLogger("urllib3").setLevel(logging.WARNING)

from tonst.redact_llm import LLMRedactor
from tonst.local_model import LocalCompressor

SAMPLE_TEXT = (
    "Hi, my name is Arjun Rao and I work at NimbusFn. My email is "
    "arjun.rao@example.com and my phone is 555-0182. Please look into "
    "why my function's invocation count doubled overnight."
)


def time_one_redact(redactor: LLMRedactor) -> float:
    t0 = time.perf_counter()
    result = redactor.redact(SAMPLE_TEXT)
    dt = time.perf_counter() - t0
    print(f"  redact(): {dt*1000:.1f}ms  model_available={result.model_available}  entities_found={result.entities_found}")
    return dt


def time_one_compress(compressor: LocalCompressor) -> float:
    t0 = time.perf_counter()
    _, was_compressed = compressor.compress(SAMPLE_TEXT * 5)
    dt = time.perf_counter() - t0
    print(f"  compress(): {dt*1000:.1f}ms  was_compressed={was_compressed}")
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="concurrent calls for the contention test (default 4, matching the Colab benchmark's worker count)")
    parser.add_argument("--timeout", type=float, default=8.0, help="per-call timeout in seconds (default 8.0, matching the library default)")
    args = parser.parse_args()

    redactor = LLMRedactor(timeout=args.timeout)
    compressor = LocalCompressor(timeout=args.timeout)

    print(f"--- Step 0: is Ollama reachable at all? ---")
    print(f"LLMRedactor.is_available(): {redactor.is_available()}")
    print(f"LocalCompressor.is_available(): {compressor.is_available()}")
    print()

    print("--- Step 1: ONE uncontended redact() call ---")
    redact_solo_s = time_one_redact(redactor)
    print()

    print("--- Step 2: ONE uncontended compress() call ---")
    compress_solo_s = time_one_compress(compressor)
    print()

    print(f"--- Step 3: {args.workers} CONCURRENT redact() calls (contention test) ---")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        t0 = time.perf_counter()
        futures = [pool.submit(time_one_redact, LLMRedactor(timeout=args.timeout)) for _ in range(args.workers)]
        durations = [f.result() for f in futures]
        wall_s = time.perf_counter() - t0
    print(f"  wall clock for all {args.workers}: {wall_s*1000:.1f}ms  per-call: min={min(durations)*1000:.1f}ms max={max(durations)*1000:.1f}ms mean={statistics.mean(durations)*1000:.1f}ms")
    print()

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    # A call that lands within ~250ms of the configured timeout almost
    # certainly never got serviced at all -- it sat in Ollama's queue
    # for the whole window and was killed client-side. This is a much
    # stronger, more direct signal than comparing raw durations, and it
    # must be checked FIRST: a slow-but-real solo call (which happens
    # easily on CPU-only hardware) does not by itself mean contention
    # isn't ALSO happening -- the two are independent and can co-occur,
    # exactly as they did in the run that motivated this script.
    timed_out = [d for d in durations if d >= args.timeout - 0.25]
    fast_enough = [d for d in durations if d < args.timeout - 0.25]

    print(
        f"Solo baseline (no contention): redact={redact_solo_s*1000:.0f}ms, "
        f"compress={compress_solo_s*1000:.0f}ms."
    )
    if redact_solo_s > 1.0 or compress_solo_s > 1.0:
        print(
            "  That's meaningfully slower than you'd expect from a 1B model on a "
            "real GPU (usually well under 1s for a short prompt) -- if this is "
            "supposed to be GPU-accelerated hardware, check whether Ollama actually "
            "used the GPU (nvidia-smi during a call, or Ollama's own startup log) "
            "before assuming the timeout itself is wrong. On CPU-only hardware "
            "(no GPU at all), latency like this is expected, not a bug."
        )
    print()

    if timed_out:
        print(
            f"{len(timed_out)}/{len(durations)} concurrent calls hit the "
            f"{args.timeout:.1f}s timeout ceiling almost exactly "
            f"({[f'{d*1000:.0f}ms' for d in timed_out]}) while "
            f"{len(fast_enough)} succeeded quickly "
            f"({[f'{d*1000:.0f}ms' for d in fast_enough]} if any)."
        )
        print(
            "CONTENTION CONFIRMED: a call landing within ~250ms of the exact "
            "timeout value means it never got serviced at all -- it sat queued "
            "behind Ollama's real (often much lower than you'd guess) concurrent "
            "serving capacity for the entire window before being killed "
            "client-side. This holds regardless of whether the solo baseline "
            "above was fast or slow -- both can be true at once (a real, "
            "non-trivial per-call cost on top of hard queuing past a few "
            "concurrent requests). Fix: either raise OLLAMA_NUM_PARALLEL (and "
            "confirm your hardware can actually back that many concurrent "
            "contexts -- GPU VRAM or CPU cores, not just the setting) to cover "
            "your real worst-case simultaneous call count, or reduce concurrent "
            "workers to match Ollama's actual serving capacity, or raise the "
            "client timeout to something a queued-but-eventually-served request "
            "can survive."
        )
    else:
        print(
            f"All {len(durations)} concurrent calls completed without hitting "
            f"the timeout (max {max(durations)*1000:.0f}ms) -- contention at "
            f"this worker count ({args.workers}) doesn't look like the issue "
            "here. Try a higher --workers count if your real workload runs more "
            "concurrent requests than this."
        )


if __name__ == "__main__":
    main()
