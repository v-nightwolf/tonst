"""
cache_savings_demo_gemini_explicit.py
--------------------------------------
Measures REAL prompt-caching savings against Gemini's EXPLICIT
CachedContent path -- the deterministic sibling of
cache_savings_demo_gemini.py, which tests the IMPLICIT (automatic,
best-effort) path instead. Read that script's docstring first if you
haven't; this one exists because of what live-testing found there.

Why this script exists (found via live testing, Sept 2026): across 14
real implicit-caching calls in this project -- 8 before enabling
billing on the test project, 6 more after -- every single one missed
(cache_hit=False). That's consistent with Google's own "best-effort, no
guarantee" framing for implicit caching, but it means implicit caching
is a poor fit for anyone who needs the savings to actually show up,
not just theoretically exist. Switching to the explicit path in the
same test session got a real, guaranteed cache hit on the very first
try: 3/3 calls referencing an explicitly-created cache showed
cache_hit=True. See ROADMAP.md for the full writeup and numbers.

The tradeoff: explicit caching is heavier-weight. You create a
CachedContent resource first (a real API call, billed once at the
model's standard input rate -- see build_cached_content_resource() in
providers/gemini.py), then reference it by name in every later
generateContent call instead of resending the stable content. You're
also responsible for the resource's lifecycle: it accrues ongoing
STORAGE RENT per hour it exists (whether read again or not) until its
TTL expires or you delete it, which this script does automatically at
the end regardless of success or failure.

This script reuses cache_savings_demo_gemini.py's REFERENCE_DOC, MODEL,
and GEMINI_API_KEY rather than duplicating them -- same document, same
model, same 4,096-token minimum, so there is nothing provider- or
content-specific left to differ between the two demos except which
caching mechanism is being exercised.

Setup:
    1. Get an API key from https://aistudio.google.com/apikey
    2. IMPORTANT: enable BILLING on that key's Google AI Studio / Cloud
       project. A free-tier key gets a confirmed ZERO-TOKEN cache
       storage quota -- the cachedContents.create call below will fail
       with a 429 RESOURCE_EXHAUSTED error on a free-tier key. See
       providers/gemini.py's module docstring for the exact error.
    3. Put it in a .env file in this directory (gitignored):
           GEMINI_API_KEY=AI...
       or export it directly in your shell.
    4. python3 cache_savings_demo_gemini_explicit.py

What you should see (confirmed by an actual run, Sept 2026, with
billing enabled): the resource is created (populate cost ~4,824
tokens), then each of the 4 calls below shows cache_hit=True with
cache_read_input_tokens matching the populate size almost exactly
(~99.5% of that call's input coming from cache). Measured on that real
run, at gemini-3.6-flash's $0.75/1M published input rate: ~89.6% cost
savings per cached read (steady state), blending to ~67.3% across the
whole 1-populate + 3-read run -- the populate call itself is billed at
full price with no discount, so the blended figure climbs toward the
steady-state number the more times the cache gets reused within its
TTL.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import requests
import cache_savings_demo_gemini as base  # reuses REFERENCE_DOC, MODEL, GEMINI_API_KEY,
                                            # and triggers its dependency-check/.env loading
from tonst import PromptParts
from tonst.providers import gemini as gemini_provider

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
TTL_SECONDS = 600  # 10 minutes -- plenty for a short demo, negligible storage rent

# gemini-3.6-flash standard (non-cached) input rate, confirmed from
# ai.google.dev/gemini-api/docs/pricing, Sept 2026. Like GEMINI_PRICING
# in providers/gemini.py, this is an absolute USD figure that WILL go
# stale (Google's own page already lists a 2027-01-01 price change) --
# verify current pricing before relying on this for real budgeting.
INPUT_PRICE_PER_MILLION = 0.75

QUESTIONS = [
    "A customer named Priya says she was charged twice this month. In "
    "one sentence, what should the agent check first?",
    "A customer asks whether Acme is SOC 2 certified. In one sentence, "
    "how should the agent respond?",
    "A customer wants to delete their account today. In one sentence, "
    "what's the first step?",
    "A customer reports their custom domain SSL isn't working yet. In "
    "one sentence, what should the agent say?",
]


def _extract_text(response_json: dict) -> str:
    """
    Pulls the response text defensively. A candidate can legitimately
    come back with no `parts` (e.g. finishReason=MAX_TOKENS truncating
    before any visible output, or a safety block) -- crashing on that
    isn't a caching problem, so this reports it plainly instead.
    """
    try:
        candidate = response_json["candidates"][0]
        return candidate["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        finish_reason = response_json.get("candidates", [{}])[0].get("finishReason", "unknown")
        return f"(no text in response -- finishReason={finish_reason})"


def main():
    if not base.GEMINI_API_KEY:
        print(
            "No GEMINI_API_KEY (or GOOGLE_API_KEY) set (checked environment "
            "and ./.env). See this script's Setup docstring.\n"
        )
        sys.exit(1)

    parts = PromptParts(
        system="You are a Tier 2 support agent for Acme Cloud Hosting. "
        "Answer using only the reference guide provided.",
        stable_blocks=[base.REFERENCE_DOC],
        variable=QUESTIONS[0],
    )

    eligibility = gemini_provider.check_cache_eligibility(parts, model=base.MODEL)
    print(f"--- Cache eligibility check ---\n{eligibility.message}\n")
    if not eligibility.eligible:
        print("Stopping -- adjust REFERENCE_DOC (in cache_savings_demo_gemini.py) before spending API calls.")
        sys.exit(1)

    print(f"--- Step 1: create CachedContent (model={base.MODEL}, ttl={TTL_SECONDS}s) ---")
    resource_body = gemini_provider.build_cached_content_resource(parts, model=base.MODEL, ttl_seconds=TTL_SECONDS)
    create_resp = requests.post(f"{BASE_URL}/cachedContents", params={"key": base.GEMINI_API_KEY}, json=resource_body, timeout=30)
    if create_resp.status_code != 200:
        print(f"FAILED to create cache: {create_resp.status_code} {create_resp.text}")
        if "RESOURCE_EXHAUSTED" in create_resp.text or "FreeTier" in create_resp.text:
            print(
                "\nThis looks like the free-tier cache-quota limit -- see this "
                "script's Setup docstring, step 2."
            )
        sys.exit(1)
    cache_info = create_resp.json()
    cache_name = cache_info["name"]
    populate_tokens = cache_info.get("usageMetadata", {}).get("totalTokenCount", 0)
    print(f"Created {cache_name}\nPopulate cost: {populate_tokens} tokens @ standard rate (one-time, no discount)\n")

    results = []
    try:
        print(f"--- Step 2: firing {len(QUESTIONS)} calls referencing the cache, each with a DIFFERENT question ---\n")
        for i, question in enumerate(QUESTIONS, 1):
            body = gemini_provider.build_generate_request_from_cache(cache_name, question, max_output_tokens=100)
            t0 = time.time()
            resp = requests.post(
                f"{BASE_URL}/models/{base.MODEL}:generateContent",
                params={"key": base.GEMINI_API_KEY},
                headers={"content-type": "application/json"},
                json=body,
                timeout=30,
            )
            dt = time.time() - t0
            if resp.status_code != 200:
                print(f"Call {i} FAILED: {resp.status_code} {resp.text}")
                continue
            data = resp.json()
            usage = gemini_provider.parse_gemini_usage(data)
            text = _extract_text(data)
            results.append(usage)
            print(
                f"Call {i} ({dt:.2f}s): \"{text[:60]}\"\n"
                f"  input_tokens={usage.input_tokens}  cache_read_input_tokens={usage.cache_read_input_tokens}  "
                f"cache_hit={usage.cache_hit}  ({usage.percent_of_input_from_cache}% of this call's input from cache)"
            )
    finally:
        print(f"\n--- Step 3: cleaning up (deleting {cache_name}) so it stops accruing storage rent ---")
        del_resp = requests.delete(f"{BASE_URL}/{cache_name}", params={"key": base.GEMINI_API_KEY}, timeout=30)
        print(f"Delete status: {del_resp.status_code}")

    if not results:
        print("\nNo successful calls -- nothing to report.")
        return

    print("\n--- Verdict ---")
    hits = sum(1 for u in results if u.cache_hit)
    if hits == len(results):
        print(f"Confirmed: all {hits}/{len(results)} calls got a real, guaranteed cache hit.")
    else:
        print(f"{hits}/{len(results)} calls got a cache hit -- unexpected for the explicit path, worth investigating.")

    print("\n--- Aggregate, real measured numbers ---")
    total_processed = populate_tokens + sum(u.input_tokens + u.cache_read_input_tokens for u in results)
    print(f"Total tokens processed (populate + all calls): {total_processed}  (NOT reduced by caching -- same content sent + counted every time)")

    per_call_savings = [gemini_provider.estimated_implicit_cache_cost_savings_percent(u) for u in results]
    print(f"Per-call cost savings vs. no caching (steady state, ignores the one-time populate cost): {[f'{s:+.1f}%' for s in per_call_savings]}")

    overall_savings = gemini_provider.estimated_explicit_cache_cost_savings_percent(
        model=base.MODEL,
        cached_tokens=populate_tokens,
        num_requests=len(results),
        hours_cached=TTL_SECONDS / 3600,
        input_price_per_million=INPUT_PRICE_PER_MILLION,
    )
    print(
        f"\nBlended cost savings across this whole run (populate + storage rent for up to "
        f"{TTL_SECONDS}s + {len(results)} discounted reads, vs. {len(results)} full-price calls "
        f"-- i.e. what {len(results)} real answers would have cost with vs. without caching): "
        f"{overall_savings:+.1f}%"
    )
    print(
        "(Blended is lower than the per-call steady-state figure because the populate call "
        "and storage rent are pure overhead that doesn't answer any question -- they get "
        "amortized across however many reads happen, so the blended figure climbs toward the "
        "steady-state number as more reads happen within the same TTL window.)"
    )


if __name__ == "__main__":
    main()
