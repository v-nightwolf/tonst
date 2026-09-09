"""
providers/openai.py
--------------------
OpenAI prompt-caching support. Verified against OpenAI's own docs
(developers.openai.com/api/docs/guides/prompt-caching,
developers.openai.com/cookbook/examples/prompt_caching_201) plus
corroborating Azure Q&A and AWS Bedrock write-ups on GPT-5.6, checked
2026-09-09.

OpenAI's caching model is structurally different from Anthropic's
(tonst/cache_structuring.py) in ways that shape this module:

  - Caching is AUTOMATIC BY DEFAULT for every model that supports it --
    no explicit marker required, just correct ordering (stable content
    first, variable last). cache_structuring.structure_for_caching()
    already produces that ordering and needs no OpenAI-specific
    version; build_openai_cache_request() below exists mainly to hand
    back the right JSON shape (a `messages` list) for this provider.
  - GPT-5.6 and later ALSO support an OPTIONAL explicit mode
    (prompt_cache_breakpoint) that behaves more like Anthropic's
    cache_control -- and, new to this model generation, an explicit
    cache-WRITE premium (1.25x) that automatic-only OpenAI models never
    had. use_explicit_breakpoint=True opts into this; the default
    (False) uses the free, automatic path every model supports.
  - The minimum cacheable length is a FLAT 1,024 tokens across the
    lineup (unlike Anthropic's per-model 512-4096 range), with cache
    hits credited in multiples of 128 tokens beyond that floor.
  - The cache-READ discount is NOT one number the way Anthropic mostly
    has one -- it varies by model, from 50% off to 98.75% off.
    CACHE_READ_MULTIPLIER below is a real per-model table, not a
    constant to assume.
  - The usage field's JSON PATH differs by API surface: Chat Completions
    reports usage.prompt_tokens_details.cached_tokens; the newer
    Responses API reports usage.input_tokens_details.cached_tokens.
    parse_openai_usage() takes an explicit `endpoint` argument because
    there's no single path to guess from.

Caveat, flagged rather than hidden: the exact JSON path for GPT-5.6+'s
explicit cache-WRITE token count (read here as `cache_write_tokens`) is
corroborated by two secondary sources (an Azure Q&A thread and an AWS
Bedrock blog post) but was not directly confirmed against OpenAI's own
raw response schema during this module's research. Treat
cache_creation_input_tokens from parse_openai_usage() as best-effort on
GPT-5.6+ explicit mode. On every other OpenAI model it will correctly
read as 0 -- automatic caching has no distinct "populate the cache"
charge to report; the first call with a new prefix is simply a normal,
fully-priced call that happens to leave a cache entry behind.
"""

from __future__ import annotations
from typing import Optional

from ..cache_structuring import PromptParts, CacheUsageReport, CacheEligibility

# Flat across the current OpenAI lineup -- unlike Anthropic, there's
# currently only one number, not a per-model table.
CACHE_MINIMUM_TOKENS = 1024
CACHE_TOKEN_INCREMENT = 128  # hits are credited in multiples of this, above the floor

# Cache-READ discount by model, expressed the same way as Anthropic's
# multipliers (fraction of the base input price). This is NOT a single
# constant -- OpenAI's discount varies substantially across its own
# lineup, unlike Anthropic where most models share one rate. Unknown or
# future models fall back to DEFAULT_CACHE_READ_MULTIPLIER (the rate
# used by gpt-5-nano and the GPT-5.6+ line, the most common current
# figure).
CACHE_READ_MULTIPLIER: dict[str, float] = {
    "gpt-4o": 0.5,          # 50% off
    "gpt-4.1": 0.25,        # 75% off
    "gpt-5-nano": 0.1,      # 90% off
    "gpt-5.6": 0.1,         # 90% off
    "gpt-realtime": 0.0125,  # 98.75% off
}
DEFAULT_CACHE_READ_MULTIPLIER = 0.1

# Only GPT-5.6+'s OPTIONAL explicit mode has a write premium at all --
# the automatic path every other model uses has none, because there's
# no distinct "write" step being billed differently from a normal call.
CACHE_WRITE_MULTIPLIER_EXPLICIT = 1.25
EXPLICIT_CACHE_TTL = "30m"  # the only supported value as of GPT-5.6


def _cache_read_multiplier(model: str) -> float:
    return CACHE_READ_MULTIPLIER.get(model, DEFAULT_CACHE_READ_MULTIPLIER)


def check_cache_eligibility(parts: PromptParts, model: str, token_estimator=None) -> CacheEligibility:
    """
    Same idea as cache_structuring.check_cache_eligibility(), against
    OpenAI's flat 1,024-token minimum instead of Anthropic's per-model
    table. Still just tonst's chars/4 estimate -- a smoke test, not a
    substitute for reading real usage.*.cached_tokens off an actual
    response.
    """
    if token_estimator is None:
        from ..trim import estimate_tokens as token_estimator

    stable_text = "\n\n".join([parts.system or ""] + list(parts.stable_blocks))
    stable_tokens = token_estimator(stable_text) if stable_text.strip() else 0

    return CacheEligibility(
        eligible=stable_tokens >= CACHE_MINIMUM_TOKENS,
        stable_tokens_estimate=stable_tokens,
        minimum_required=CACHE_MINIMUM_TOKENS,
        model=model,
    )


def build_openai_cache_request(
    parts: PromptParts,
    model: str,
    max_tokens: int = 1000,
    use_explicit_breakpoint: bool = False,
) -> dict:
    """
    Builds an OpenAI Chat Completions-style request body with stable
    content ordered first, variable content last.

    For nearly every model, that ordering is ALL that's needed --
    OpenAI's automatic caching activates on it alone, no marker
    required, no extra cost to opt in. This is the default
    (use_explicit_breakpoint=False): stable + variable content are
    joined into one plain string, matching what automatic caching looks
    for.

    Pass use_explicit_breakpoint=True only for GPT-5.6+ models where you
    specifically want the newer explicit prompt_cache_breakpoint
    behavior -- e.g. because you need the guarantee it offers, or
    multiple breakpoints in one request -- and are prepared for its
    1.25x write premium on the first call. Most callers should leave
    this False and get caching for free.
    """
    messages = []
    if parts.system:
        messages.append({"role": "system", "content": parts.system.strip()})

    stable = [b.strip() for b in parts.stable_blocks if b and b.strip()]
    variable_text = (parts.variable or "").strip()

    if stable and use_explicit_breakpoint:
        content_blocks = [{"type": "text", "text": b} for b in stable]
        # Breakpoint on the LAST stable block, same placement logic as
        # Anthropic's cache_control -- everything up to and including it
        # is the cacheable prefix.
        content_blocks[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
        if variable_text:
            content_blocks.append({"type": "text", "text": variable_text})
        messages.append({"role": "user", "content": content_blocks})
        return {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
            "prompt_cache_options": {"ttl": EXPLICIT_CACHE_TTL},
        }

    # Automatic path: one plain string, stable-first. No marker, no
    # special request shape, no opt-in cost.
    user_text = "\n\n".join(stable + ([variable_text] if variable_text else []))
    messages.append({"role": "user", "content": user_text})
    return {"model": model, "max_tokens": max_tokens, "messages": messages}


def parse_openai_usage(response_json: dict, endpoint: str = "chat_completions") -> CacheUsageReport:
    """
    endpoint: "chat_completions" reads usage.prompt_tokens_details.cached_tokens
    (the format confirmed verbatim in OpenAI's own cookbook example).
    "responses" reads usage.input_tokens_details.cached_tokens instead --
    OpenAI uses a different JSON path on each API surface, so this
    function must be told which one it's reading; there's no shared
    shape to detect automatically.
    """
    usage = response_json.get("usage", {})
    if endpoint == "responses":
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        details = usage.get("input_tokens_details", {}) or {}
    else:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        details = usage.get("prompt_tokens_details", {}) or {}

    cached_tokens = details.get("cached_tokens", 0)
    # Best-effort field, GPT-5.6+ explicit mode only -- see module
    # docstring caveat. Always 0 on automatic-only models, correctly.
    write_tokens = details.get("cache_write_tokens", 0)

    return CacheUsageReport(
        input_tokens=max(0, prompt_tokens - cached_tokens - write_tokens),
        output_tokens=completion_tokens,
        cache_creation_input_tokens=write_tokens,
        cache_read_input_tokens=cached_tokens,
    )


def estimated_cost_savings_percent(usage: CacheUsageReport, model: str) -> float:
    """
    Estimates the % cost difference caching made on this call, against
    the baseline of the same token count sent with no caching at all --
    same principle as
    cache_structuring.CacheUsageReport.estimated_cost_savings_percent(),
    but using OpenAI's real per-model discount table (CACHE_READ_MULTIPLIER
    above) instead of Anthropic's. Not a method on CacheUsageReport
    itself, deliberately: that dataclass is a shared, provider-agnostic
    container, and each provider's pricing lives in that provider's own
    module rather than being baked into the shared type.
    """
    read_mult = _cache_read_multiplier(model)
    actual_cost = (
        usage.input_tokens * 1.0
        + usage.cache_creation_input_tokens * CACHE_WRITE_MULTIPLIER_EXPLICIT
        + usage.cache_read_input_tokens * read_mult
    )
    total_tokens = (
        usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    )
    no_cache_cost = total_tokens * 1.0
    if no_cache_cost == 0:
        return 0.0
    return round(100 * (no_cache_cost - actual_cost) / no_cache_cost, 1)
