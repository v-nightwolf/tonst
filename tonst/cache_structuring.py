"""
cache_structuring.py
---------------------
Structures a request so that provider-side prompt caching actually pays
off. Important to be precise about what this is NOT: prompt caching does
not skip the LLM call. Anthropic, OpenAI, and Gemini all cache the
*processed representation* of repeated prefix tokens so the model
doesn't have to reprocess them, but the model still runs and generates a
fresh response on every call -- caching only discounts (and speeds up)
the repeated portion of the input.

It only pays off if the request is shaped correctly:
  - Stable, reused content (system instructions, a reference document,
    tool descriptions written out as text) has to come FIRST.
  - That stable content has to be byte-for-byte IDENTICAL across calls --
    reordering it, changing its whitespace, or interleaving the variable
    question into it all silently break the match. There's no error when
    this happens; you just quietly stop getting the discount.
  - The variable part (the actual question, the newest turn) goes LAST.
  - The stable prefix also has to clear a PER-MODEL MINIMUM LENGTH --
    see CACHE_MINIMUM_TOKENS below. Below it, Anthropic silently skips
    caching entirely: no error, cache_creation_input_tokens and
    cache_read_input_tokens both stay 0. check_cache_eligibility() and
    the warning in build_anthropic_cache_request() exist specifically so
    this doesn't fail silently on you too.

Output shapes provided:
  - structure_for_caching() returns a flat, correctly-ordered string.
    This is enough for providers with automatic prefix caching (no
    explicit marker required) and works with TonstClient.query(str).
  - build_anthropic_cache_request() returns the actual Anthropic
    Messages API JSON body with an explicit `cache_control` breakpoint,
    which Anthropic requires -- correct ordering alone is not sufficient
    on that provider. Verified against Anthropic's prompt-caching docs
    (platform.claude.com/docs/en/build-with-claude/prompt-caching,
    checked 2026-09-08): cache_control goes on the LAST block whose
    prefix should be cached, as {"type": "ephemeral", "ttl": "5m"|"1h"}.
  - parse_anthropic_usage() reads the `usage` field of a REAL API
    response and reports whether a cache hit actually happened --
    the only ground truth for whether any of this worked.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import Optional

VALID_TTLS = ("5m", "1h")

# Anthropic's minimum cacheable-prefix length per model, in tokens, as of
# the docs checked 2026-09-08. A stable prefix shorter than this is
# processed normally with NO ERROR and NO CACHING -- the response's
# usage.cache_creation_input_tokens / usage.cache_read_input_tokens both
# come back 0. This table exists so tonst can warn about that ahead of
# time instead of a developer discovering it by silently getting no
# savings. Keyed on the model string as passed to the API; unknown/future
# models fall back to DEFAULT_CACHE_MINIMUM_TOKENS (the most common
# figure across the current lineup).
CACHE_MINIMUM_TOKENS: dict[str, int] = {
    "claude-haiku-4-5": 4096,
    "claude-haiku-3-5": 2048,
    "claude-opus-4-6": 4096,
    "claude-opus-4-5": 4096,
    "claude-opus-4-7": 2048,
    "claude-opus-4-8": 1024,
    "claude-opus-4-1": 1024,
    "claude-opus-4": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-sonnet-4-5": 1024,
    "claude-sonnet-4": 1024,
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-fable-5-1": 512,
    "claude-mythos-5": 512,
    "claude-mythos-5-1": 512,
    # "Claude Mythos Preview" is documented at 2,048 tokens, but its
    # exact API model-id string wasn't confirmed at the time this table
    # was written -- add it here once known, rather than guess a key
    # that would silently never match.
}
DEFAULT_CACHE_MINIMUM_TOKENS = 1024

# Anthropic's cache pricing multipliers, relative to that model's own
# base (uncached) input-token price of 1.0x. Verified against
# platform.claude.com/docs/en/build-with-claude/prompt-caching, checked
# 2026-09-09. A cache WRITE costs MORE than a fresh token -- you're
# paying a premium to populate the cache -- and only a cache READ is
# discounted. This is what makes "percent of input from cache" (see
# CacheUsageReport.percent_of_input_from_cache) different from an actual
# cost saving: caching never changes how many tokens get processed, so
# a real savings estimate has to weight each token type by its actual
# price, not just count what fraction came from cache.
CACHE_READ_MULTIPLIER_DEFAULT = 0.1  # 10% of base input price
CACHE_READ_MULTIPLIER_LOW_COST = 0.025  # Fable/Mythos family: 2.5% of base price
CACHE_WRITE_MULTIPLIER_5M = 1.25  # 125% of base input price
CACHE_WRITE_MULTIPLIER_1H = 2.0  # 200% of base input price

# Models with the lower 2.5% cache-read rate instead of the standard 10%.
_LOW_COST_CACHE_READ_MODELS = {
    "claude-fable-5",
    "claude-fable-5-1",
    "claude-mythos-5",
    "claude-mythos-5-1",
}


def _cache_read_multiplier(model: str) -> float:
    return (
        CACHE_READ_MULTIPLIER_LOW_COST
        if model in _LOW_COST_CACHE_READ_MODELS
        else CACHE_READ_MULTIPLIER_DEFAULT
    )


@dataclass
class PromptParts:
    """
    system: instructions that never change between calls in this
        conversation/session (e.g. "You are a support assistant...").
    stable_blocks: large reusable content that's IDENTICAL across many
        calls -- reference documents, few-shot examples, tool
        descriptions written out as text. List order is preserved; put
        the content least likely to ever change first, since everything
        up to and including the cache_control breakpoint must match
        exactly for a hit.
    variable: the part that's different on every call -- the user's
        actual question, or the newest turn. Always placed last.
    """
    system: Optional[str] = None
    stable_blocks: list = field(default_factory=list)
    variable: str = ""


def structure_for_caching(parts: PromptParts) -> str:
    """
    Returns a single string with stable content first, variable last.
    Byte-for-byte identical stable content across calls is what lets
    providers with automatic prefix caching (no explicit marker needed)
    actually get a cache hit -- reordering, whitespace changes, or
    interleaving the question into the stable section all silently
    defeat it. This is the ordering guarantee; it does not, by itself,
    set an explicit cache breakpoint (see build_anthropic_cache_request
    for that).
    """
    sections = []
    if parts.system:
        sections.append(parts.system.strip())
    sections.extend(b.strip() for b in parts.stable_blocks if b and b.strip())
    if parts.variable:
        sections.append(parts.variable.strip())
    return "\n\n".join(sections)


@dataclass
class CacheEligibility:
    """
    Heuristic estimate of whether `parts`'s stable content is long enough
    for `model` to actually cache it. This uses tonst's chars/4 token
    estimate (see trim.py), NOT the provider's real tokenizer -- treat it
    as a smoke test that catches the obvious "way too short" case, not
    an exact prediction. The only real ground truth is the `usage` field
    of an actual API response (see parse_anthropic_usage()).
    """
    eligible: bool
    stable_tokens_estimate: int
    minimum_required: int
    model: str

    @property
    def message(self) -> str:
        if self.eligible:
            return (
                f"Stable content is ~{self.stable_tokens_estimate} tokens "
                f"(est.), above {self.model}'s {self.minimum_required}-token "
                "minimum -- likely eligible for caching."
            )
        return (
            f"Stable content is only ~{self.stable_tokens_estimate} tokens "
            f"(est.), below {self.model}'s {self.minimum_required}-token "
            "minimum. This provider will NOT cache it: most providers raise "
            "no error here, they simply process the request without "
            "caching. Add more reusable content to stable_blocks/system, "
            "or don't bother shaping this request for caching at all."
        )


def check_cache_eligibility(parts: PromptParts, model: str, token_estimator=None) -> CacheEligibility:
    """
    Estimates whether the stable/cacheable portion of `parts` meets
    Anthropic's minimum prefix length for `model`. Meant to catch an
    obviously-too-short system prompt or reference snippet before
    spending an API call that silently won't cache -- not a substitute
    for checking real usage.cache_creation_input_tokens /
    usage.cache_read_input_tokens from an actual response.
    """
    if token_estimator is None:
        from .trim import estimate_tokens as token_estimator

    stable_text = "\n\n".join([parts.system or ""] + list(parts.stable_blocks))
    stable_tokens = token_estimator(stable_text) if stable_text.strip() else 0

    minimum = CACHE_MINIMUM_TOKENS.get(model, DEFAULT_CACHE_MINIMUM_TOKENS)
    return CacheEligibility(
        eligible=stable_tokens >= minimum,
        stable_tokens_estimate=stable_tokens,
        minimum_required=minimum,
        model=model,
    )


def build_anthropic_cache_request(
    parts: PromptParts,
    model: str,
    max_tokens: int = 1000,
    cache_ttl: str = "5m",
    warn_if_ineligible: bool = True,
) -> dict:
    """
    Builds the JSON body for a direct call to
    https://api.anthropic.com/v1/messages with an explicit cache
    breakpoint placed after the stable content. Anthropic requires this
    `cache_control` marker on the last stable block -- ordering alone
    (unlike OpenAI/Gemini's automatic prefix caching) does not trigger a
    cache hit on this provider.

    If `warn_if_ineligible` is True (default), this checks stable-content
    length against CACHE_MINIMUM_TOKENS and raises a UserWarning if it's
    likely too short to actually be cached. The request is still built
    and returned either way -- Anthropic itself doesn't error on a
    too-short cache_control block, it just silently skips caching, and
    tonst matches that (fail-soft), just with a visible warning instead
    of silence.

    If `parts.system` is set, it's cached too (as its own block with its
    own breakpoint) since a large, reused system prompt is one of the
    most common things worth caching.

    If there are no stable_blocks, this falls back to a plain string
    message body -- there's nothing to cache, so the block-array
    overhead (and the cost of writing a cache entry with no reuse ahead
    of it) isn't worth it.
    """
    if cache_ttl not in VALID_TTLS:
        raise ValueError(f"cache_ttl must be one of {VALID_TTLS}, got {cache_ttl!r}")

    if warn_if_ineligible:
        eligibility = check_cache_eligibility(parts, model)
        if not eligibility.eligible:
            warnings.warn(eligibility.message, UserWarning, stacklevel=2)

    body: dict = {"model": model, "max_tokens": max_tokens}

    if parts.system:
        body["system"] = [
            {
                "type": "text",
                "text": parts.system.strip(),
                "cache_control": {"type": "ephemeral", "ttl": cache_ttl},
            }
        ]

    stable = [b.strip() for b in parts.stable_blocks if b and b.strip()]
    variable_text = (parts.variable or "").strip()

    if stable:
        content_blocks = [{"type": "text", "text": b} for b in stable]
        # The breakpoint goes on the LAST stable block: everything up to
        # and including it becomes the cached prefix, so only one marker
        # is needed even if there are many stable blocks.
        content_blocks[-1]["cache_control"] = {"type": "ephemeral", "ttl": cache_ttl}
        if variable_text:
            content_blocks.append({"type": "text", "text": variable_text})
        body["messages"] = [{"role": "user", "content": content_blocks}]
    else:
        # Nothing stable to cache -- a plain string is simpler and avoids
        # block-array overhead for a one-off call.
        body["messages"] = [{"role": "user", "content": variable_text}]

    return body


@dataclass
class CacheUsageReport:
    """
    Parsed from a REAL Anthropic API response's `usage` field -- the
    only ground truth for whether caching actually happened on a given
    call. cache_creation_input_tokens > 0 means this call WROTE a new
    cache entry (expected on the first call with a given stable prefix).
    cache_read_input_tokens > 0 means this call GOT the discount by
    reading a previously-written cache entry.
    """
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_input_tokens > 0

    @property
    def cache_write(self) -> bool:
        return self.cache_creation_input_tokens > 0

    @property
    def percent_of_input_from_cache(self) -> float:
        total_input = (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        if total_input == 0:
            return 0.0
        return round(100 * self.cache_read_input_tokens / total_input, 1)

    def estimated_cost_savings_percent(self, model: str, cache_ttl: str = "5m") -> float:
        """
        Estimates the % COST difference caching made on this call,
        against the baseline of the exact same token count being sent
        with NO caching at all (everything billed at the standard 1x
        input rate). That baseline -- not "tokens saved" -- is the
        correct comparison, because caching never reduces how many
        tokens the model processes; see the module docstring and
        percent_of_input_from_cache above.

        This can come back NEGATIVE: a cache-WRITE call (the first call
        with a new stable prefix) costs MORE than not caching at all,
        since Anthropic charges a premium (1.25x for a 5-minute TTL,
        2x for 1-hour) to populate the cache. The saving only shows up
        on a later cache-READ call against that same prefix. Report
        both calls, not just the read, if you want an honest before/
        after picture -- see cache_savings_demo_anthropic.py (or the OpenAI/Gemini/generic siblings).

        Uses tonst's own CACHE_READ/WRITE multiplier constants above
        (verified against Anthropic's published pricing docs, not this
        library's own guess), applied to the REAL token counts parsed
        from an actual API response -- not an estimate.
        """
        read_mult = _cache_read_multiplier(model)
        write_mult = CACHE_WRITE_MULTIPLIER_1H if cache_ttl == "1h" else CACHE_WRITE_MULTIPLIER_5M

        actual_cost = (
            self.input_tokens * 1.0
            + self.cache_creation_input_tokens * write_mult
            + self.cache_read_input_tokens * read_mult
        )
        total_tokens = (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        no_cache_cost = total_tokens * 1.0  # same token count, all at the base rate

        if no_cache_cost == 0:
            return 0.0
        return round(100 * (no_cache_cost - actual_cost) / no_cache_cost, 1)


def parse_anthropic_usage(response_json: dict) -> CacheUsageReport:
    """
    Extracts cache-relevant token counts from a real Anthropic Messages
    API response. Pass the full decoded JSON body of the response (the
    function reads its "usage" key) -- this is the only way to actually
    confirm caching worked, as opposed to just building a
    correctly-shaped request and assuming it did.
    """
    usage = response_json.get("usage", {})
    return CacheUsageReport(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
    )
