"""
providers/gemini.py
--------------------
Gemini context-caching support. Verified against Google's own docs
(ai.google.dev/gemini-api/docs/caching, ai.google.dev/api/caching,
ai.google.dev/gemini-api/docs/pricing), checked 2026-09-09.

Gemini is the most structurally different of the three providers this
tool supports, because it has TWO SEPARATE caching mechanisms with
different cost shapes -- not one mechanism with an optional explicit
mode, like OpenAI:

  - IMPLICIT caching: fully automatic, on by default, no code changes.
    Works the same way OpenAI's automatic caching does -- put stable
    content first (cache_structuring.structure_for_caching() already
    does this) and Google's backend opportunistically reuses it.
    Best-effort only: Google's own docs are explicit that there's no
    guaranteed savings. No developer-visible TTL; Google manages
    eviction. build_gemini_content_request() below is for this path.
  - EXPLICIT caching: a separate, developer-managed resource. You
    create a CachedContent object (build_cached_content_resource()
    below builds the request body for that), get back a resource name,
    then reference it in generateContent calls
    (build_generate_request_from_cache() below). This is heavier-weight
    than a request-time flag -- closer to "create an object with its
    own lifecycle" than "add a parameter."

The cost shape differs between the two, which is why there are TWO
separate cost-estimation functions below instead of one:

  - Implicit caching: same shape as Anthropic/OpenAI -- a discounted
    per-token rate on a cache hit, no separate write charge. Read
    estimated_implicit_cache_cost_savings_percent().
  - Explicit caching: NO per-token write premium the way Anthropic and
    OpenAI GPT-5.6+ have one. Instead, populating the cache is billed
    at the model's STANDARD input rate once, and then Google charges
    ongoing STORAGE RENT per hour the cache exists -- whether or not it
    is ever read again. Whether explicit caching saves money at all
    therefore depends on how many times you reuse it within its TTL
    window, not just on whether you turned the feature on. Read
    estimated_explicit_cache_cost_savings_percent().

Pricing note, flagged rather than silently assumed: the dollar figures
in GEMINI_PRICING below were pulled directly from Google's live pricing
page on 2026-09-09. Unlike the read/write MULTIPLIERS used for
Anthropic and OpenAI (ratios that are unlikely to change even if
headline prices do), these are absolute USD figures that WILL go stale
-- Google's own page already lists a scheduled price change for the
3.6/3.7/3.8 Flash line on 2027-01-01. Verify current pricing before
relying on this for real budgeting; treat GEMINI_PRICING as a
reasonable default, not a live source of truth.

Minimum-token caveat: ai.google.dev's caching-docs table and Firebase's
wrapper docs disagree on Flash-model minimums (2,048 vs. 1,024 tokens).
GEMINI_CACHE_MINIMUM_TOKENS below uses ai.google.dev's numbers, as the
primary-provider source rather than a downstream wrapper's -- but this
discrepancy is real and worth testing empirically before depending on
it precisely.

FREE-TIER CAVEAT (confirmed via live testing, Sept 2026 -- this is the
single most important caveat in this file): a free-tier API key/project
gets a ZERO-TOKEN cache storage quota. Confirmed directly from Google's
own API, calling cachedContents.create against a free-tier key:

    429 RESOURCE_EXHAUSTED: "TotalCachedContentStorageTokensPerModelFreeTier
    limit exceeded for model gemini-3.6-flash: limit=0, requested=4824"

This means BOTH caching mechanisms are unusable on a free-tier key --
not "less likely to hit," structurally zero capacity. It also explains
what would otherwise look like implicit caching randomly never
activating: 8 real API calls in testing, all correctly shaped and above
the real per-model minimum, produced 0 cache hits, because the account
had no cache storage available at all, at any size. Before concluding
caching "isn't working" for a given key, confirm billing is enabled on
that key's Google AI Studio / Cloud project -- that's a far more likely
explanation than a request-shaping bug or bad luck on a best-effort
mechanism.
"""

from __future__ import annotations
from typing import Optional

from ..cache_structuring import PromptParts, CacheUsageReport, CacheEligibility

# Per-model minimum tokens before Gemini's IMPLICIT cache can activate,
# or before an EXPLICIT CachedContent resource is accepted. Source:
# ai.google.dev/gemini-api/docs/caching, checked 2026-09-09 -- see the
# module docstring's caveat about a conflicting Firebase-docs figure.
CACHE_MINIMUM_TOKENS: dict[str, int] = {
    "gemini-2.5-flash": 2048,
    "gemini-2.5-pro": 2048,
    "gemini-3.1-pro-preview": 4096,
    "gemini-3.5-flash": 4096,
    "gemini-3.6-flash": 4096,
    "gemini-3.7-flash": 4096,
    "gemini-3.8-flash": 4096,
}
DEFAULT_CACHE_MINIMUM_TOKENS = 2048

# Cache-READ discount: confirmed uniformly 10% of standard input price
# (0.1x) across every model on Google's live pricing page as of
# 2026-09-09 -- unlike OpenAI, where the discount varies per model, one
# flat multiplier genuinely covers the whole Gemini lineup today. Kept
# as a named constant (not hardcoded inline) so a future per-model
# exception is a one-line table, not a rewrite, if Google ever
# introduces one.
CACHE_READ_MULTIPLIER = 0.1

# EXPLICIT caching only: ongoing storage rent, in USD per million tokens
# PER HOUR the cache exists, charged regardless of whether it's ever
# read again. This has no equivalent in Anthropic/OpenAI's pricing at
# all -- see module docstring. USD, not a multiplier -- will go stale;
# see the pricing-note caveat above.
GEMINI_STORAGE_USD_PER_MILLION_TOKEN_HOUR: dict[str, float] = {
    "gemini-2.5-flash": 1.00,
    "gemini-2.5-pro": 4.50,
    "gemini-3.5-flash": 1.00,
    "gemini-3.6-flash": 0.50,  # through 2026-12-31; 1.00 from 2027-01-01
    "gemini-3.7-flash": 0.50,
    "gemini-3.8-flash": 0.50,
}
DEFAULT_STORAGE_USD_PER_MILLION_TOKEN_HOUR = 1.00


def check_cache_eligibility(parts: PromptParts, model: str, token_estimator=None) -> CacheEligibility:
    """
    Same idea as cache_structuring.check_cache_eligibility() and
    providers.openai.check_cache_eligibility(), against Gemini's
    per-model minimum table above. Applies to both implicit and
    explicit caching -- the minimum is a property of the model, not of
    which mechanism you use.
    """
    if token_estimator is None:
        from ..trim import estimate_tokens as token_estimator

    stable_text = "\n\n".join([parts.system or ""] + list(parts.stable_blocks))
    stable_tokens = token_estimator(stable_text) if stable_text.strip() else 0
    minimum = CACHE_MINIMUM_TOKENS.get(model, DEFAULT_CACHE_MINIMUM_TOKENS)

    return CacheEligibility(
        eligible=stable_tokens >= minimum,
        stable_tokens_estimate=stable_tokens,
        minimum_required=minimum,
        model=model,
    )


def build_gemini_content_request(parts: PromptParts, model: str, max_output_tokens: int = 1000) -> dict:
    """
    IMPLICIT (automatic) caching path -- the one most callers want.
    Builds a generateContent-style request body with stable content
    ordered first, variable content last, and nothing else special: no
    marker, no separate resource to create or manage. Google's backend
    opportunistically caches the repeated prefix on its own; there is
    no guarantee it will, and no error if it doesn't.
    """
    parts_list = []
    if parts.system:
        parts_list.append({"text": parts.system.strip()})
    for block in parts.stable_blocks:
        if block and block.strip():
            parts_list.append({"text": block.strip()})
    if parts.variable and parts.variable.strip():
        parts_list.append({"text": parts.variable.strip()})

    return {
        "model": model,
        "contents": [{"role": "user", "parts": parts_list}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }


def build_cached_content_resource(parts: PromptParts, model: str, ttl_seconds: int = 3600) -> dict:
    """
    EXPLICIT caching path, step 1: the request body for
    cachedContents.create (SDK: client.caches.create()). Only the
    STABLE content belongs here -- system instructions and reusable
    reference material -- never the variable per-call question, since
    the whole point is that this resource gets reused unchanged across
    many later calls. Returns the resource's id via the API response;
    pass that id to build_generate_request_from_cache() for each actual
    call.

    ttl_seconds defaults to 3600 (1 hour), matching Google's own
    documented default when neither ttl nor expireTime is set. This is
    a genuinely different tradeoff than Anthropic/OpenAI's TTLs: a
    longer TTL here means more storage-rent hours billed whether or not
    you ever read it again, not just a longer window during which a
    discount is available.

    Two things fixed here after checking this against the live
    cachedContents.create schema (ai.google.dev/api/caching, Sept 2026):
    the `model` field is a full resource name ("models/{model}"), not a
    bare model id -- sending a bare id is a guaranteed request failure,
    not a caching-specific error -- and system instructions belong in
    their own `systemInstruction` field, not folded into `contents` as
    a fake user turn (contents is meant to hold only the reusable
    reference material being cached).
    """
    model_name = model if model.startswith("models/") else f"models/{model}"

    contents = []
    for block in parts.stable_blocks:
        if block and block.strip():
            contents.append({"role": "user", "parts": [{"text": block.strip()}]})

    resource: dict = {
        "model": model_name,
        "contents": contents,
        "ttl": f"{ttl_seconds}s",
    }
    if parts.system and parts.system.strip():
        resource["systemInstruction"] = {"parts": [{"text": parts.system.strip()}]}
    return resource


def build_generate_request_from_cache(
    cached_content_name: str, variable_text: str, max_output_tokens: int = 1000
) -> dict:
    """
    EXPLICIT caching path, step 2: a generateContent request that
    references an already-created CachedContent resource by name
    (e.g. "cachedContents/abc123", the id returned by creating the
    resource above) instead of resending the stable content at all.
    Only the variable/per-call text goes in `contents` here.
    """
    return {
        "cachedContent": cached_content_name,
        "contents": [{"role": "user", "parts": [{"text": variable_text.strip()}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens},
    }


def parse_gemini_usage(response_json: dict) -> CacheUsageReport:
    """
    Reads usageMetadata.cachedContentTokenCount from a real generateContent
    response (confirmed field name/casing; the Python/JS SDKs expose the
    same value as snake_case cached_content_token_count). Works for
    BOTH implicit and explicit caching -- the field reports actual cache
    reads either way; it does not distinguish which mechanism produced
    them.

    cache_creation_input_tokens is always 0 here, and that's correct,
    not a gap: implicit caching has no distinct write step to report,
    and explicit caching's population cost is billed through the
    separate cachedContents.create call (at the standard input rate),
    which does not appear in a generateContent response's usage at all.
    See estimated_explicit_cache_cost_savings_percent() for how to
    account for that population cost plus ongoing storage rent, which
    this dataclass shape cannot represent on its own.
    """
    usage = response_json.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount", 0)
    cached_tokens = usage.get("cachedContentTokenCount", 0)

    return CacheUsageReport(
        input_tokens=max(0, prompt_tokens - cached_tokens),
        output_tokens=usage.get("candidatesTokenCount", 0),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached_tokens,
    )


def estimated_implicit_cache_cost_savings_percent(usage: CacheUsageReport) -> float:
    """
    Cost estimate for the IMPLICIT caching path -- same shape as
    Anthropic/OpenAI's calculation (a flat per-token discount on the
    cached portion, no write premium), using Gemini's confirmed uniform
    10% cache-read rate (CACHE_READ_MULTIPLIER above). No model
    argument needed since, unlike OpenAI, this rate doesn't currently
    vary by model.
    """
    actual_cost = usage.input_tokens * 1.0 + usage.cache_read_input_tokens * CACHE_READ_MULTIPLIER
    total_tokens = usage.input_tokens + usage.cache_read_input_tokens
    no_cache_cost = total_tokens * 1.0
    if no_cache_cost == 0:
        return 0.0
    return round(100 * (no_cache_cost - actual_cost) / no_cache_cost, 1)


def estimated_explicit_cache_cost_savings_percent(
    model: str,
    cached_tokens: int,
    num_requests: int,
    hours_cached: float,
    input_price_per_million: float,
) -> float:
    """
    Cost estimate for the EXPLICIT CachedContent path. This is
    deliberately a DIFFERENT function signature from every other cost
    estimator in tonst, because explicit caching's cost shape is
    genuinely different -- it can't be computed from a single call's
    usage report the way implicit/Anthropic/OpenAI caching can, since
    the storage-rent component accrues over TIME, independent of how
    many times (if any) the cache gets read.

    Arguments:
      cached_tokens: size of the stable content stored in the cache.
      num_requests: how many generateContent calls will reuse this
        cached content within hours_cached -- the whole calculation
        hinges on this, since explicit caching can easily cost MORE
        than not caching at all if this is too low relative to
        hours_cached.
      hours_cached: how long the cache resource will exist (its TTL, in
        hours) -- storage rent is charged for this whole duration
        regardless of read count.
      input_price_per_million: the model's own standard (uncached)
        input price, in USD per million tokens. Required explicitly
        rather than defaulting to an internal table, because unlike the
        read/write MULTIPLIERS used elsewhere in tonst (ratios, stable
        over time), this needs a real, current dollar figure to weigh
        against the flat-rate storage rent -- see the module docstring
        pricing caveat. Pass your model's current published price.

    Returns the % cost difference vs. sending cached_tokens fresh (no
    caching) on every one of num_requests calls. Can be NEGATIVE --
    that's a real, expected outcome when num_requests is too low or
    hours_cached is too long for the reuse to pay for the storage rent,
    and is the entire point of exposing this as its own explicit
    calculation instead of assuming explicit caching always saves money.
    """
    storage_rate = GEMINI_STORAGE_USD_PER_MILLION_TOKEN_HOUR.get(
        model, DEFAULT_STORAGE_USD_PER_MILLION_TOKEN_HOUR
    )

    population_cost = (cached_tokens / 1_000_000) * input_price_per_million
    storage_cost = (cached_tokens / 1_000_000) * storage_rate * hours_cached
    read_cost = (cached_tokens / 1_000_000) * input_price_per_million * CACHE_READ_MULTIPLIER * num_requests

    actual_cost = population_cost + storage_cost + read_cost
    no_cache_cost = (cached_tokens / 1_000_000) * input_price_per_million * num_requests

    if no_cache_cost == 0:
        return 0.0
    return round(100 * (no_cache_cost - actual_cost) / no_cache_cost, 1)
