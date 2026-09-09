"""
providers/generic.py
---------------------
A configurable adapter for ANY provider tonst doesn't have a dedicated
module for -- Mistral, Groq, Together AI, Fireworks, DeepSeek, xAI,
Cohere, a self-hosted vLLM/Ollama/TGI server, or whatever ships next
month. This is the actual answer to "make it work with any model, not
just these 3": tonst cannot ship a hand-written module for every
provider that exists or will ever exist (see providers/openai.py and
providers/gemini.py's docstrings for how much provider-specific detail
each dedicated module needed just for THREE providers -- different
marker rules, different minimums, different discount shapes, different
JSON paths, and in Gemini's case two totally different cost models).
Real evidence this problem doesn't stop at three: AWS Bedrock's
Converse API reports Claude's cache usage under camelCase field names
(cacheReadInputTokens, cacheWriteInputTokens) that are DIFFERENT from
Anthropic's own direct-API names, despite serving the same model --
so even "the same provider, a different door" can break a hardcoded
parser. See providers/presets.py for that exact case, built with this
module.

What genuinely generalizes to any provider, no config needed:
  - cache_structuring.structure_for_caching() -- stable-first,
    variable-last text ordering. This is universal: any provider doing
    prefix-based caching (the overwhelming majority -- it follows
    naturally from how transformer KV-caches work) benefits from
    correct ordering, and getting it "right" never hurts a provider
    that doesn't cache at all, or that ignores ordering entirely.
  - CacheUsageReport (cache_structuring.py) -- the one shared result
    shape every provider's usage, including a fully custom one, gets
    normalized into by the time you're done with parse_usage() below.

What does NOT generalize, and needs a few lines of config instead:
  - The provider's minimum cacheable prefix length, if any.
  - The cache-read discount (and write premium, if it has one).
  - Where in the response JSON the cached-token count actually lives --
    every provider checked so far (Anthropic, OpenAI x2 endpoints,
    Gemini, Bedrock Converse) puts it at a DIFFERENT path. There is no
    reason to expect the next one to match any of them, so this module
    asks for the path explicitly instead of guessing.

Usage: build one GenericCacheConfig per provider (and per pricing tier,
if a provider has more than one -- see providers/presets.py's Azure
PTU-M example), from whatever your provider's docs say, then reuse it:

    from tonst.providers.generic import GenericCacheConfig, parse_usage, estimated_cost_savings_percent

    my_provider = GenericCacheConfig(
        minimum_tokens=1024,
        cache_read_multiplier=0.5,             # whatever your provider's docs say
        cache_write_multiplier=1.0,            # 1.0 if there's no write premium
        usage_read_path="usage.cached_tokens", # wherever THEIR usage JSON puts it
        usage_input_path="usage.prompt_tokens",
        usage_output_path="usage.completion_tokens",
    )

    usage = parse_usage(response_json, my_provider)
    savings = estimated_cost_savings_percent(usage, my_provider)

If a provider doesn't do prefix caching at all, or you simply don't
know its numbers yet, the defaults (minimum_tokens=0,
cache_read_multiplier=1.0) make every function here a safe no-op:
eligibility is always True, savings always comes back 0% (never
wrong, never a fabricated discount) -- so wiring in a brand-new
provider before you've looked up its caching docs degrades gracefully
instead of silently lying about savings that were never confirmed.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

from ..cache_structuring import PromptParts, CacheUsageReport, CacheEligibility


def _get_path(data: dict, path: str, default: Any = 0) -> Any:
    """
    Reads a dot-separated path out of a nested dict, e.g.
    "usage.prompt_tokens_details.cached_tokens" -> data["usage"]["prompt_tokens_details"]["cached_tokens"].
    Returns `default` (never raises) if any segment is missing or the
    path is empty -- this is what lets an unconfigured field (e.g. a
    provider with no cache-write concept at all) safely resolve to 0
    instead of crashing parse_usage().
    """
    if not path:
        return default
    node: Any = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


@dataclass
class GenericCacheConfig:
    """
    Everything tonst needs to know about a provider's caching contract
    that it CANNOT infer generically. See the module docstring for how
    to fill this in from a provider's own docs.

    label: a human-readable name used only in CacheEligibility messages
        (e.g. "Mistral", "my self-hosted vLLM server") -- purely
        cosmetic, never parsed.
    minimum_tokens: the shortest stable prefix this provider will
        actually cache. 0 if unknown or if the provider has no minimum
        (some don't).
    cache_read_multiplier: price of a cached/read token, as a fraction
        of that provider's own standard input price (0.1 = 90% off).
        1.0 (the default) means "no discount configured yet" -- safe,
        not a guess.
    cache_write_multiplier: price of a token spent populating the
        cache, as a fraction of standard input price. 1.0 (the
        default) means no premium -- correct for providers whose
        caching is fully automatic with no distinct write cost (most
        of them), and a safe default for ones you haven't checked yet.
    usage_read_path / usage_write_path: dot-paths (see _get_path) into
        a real API response locating the cached-read and cache-write
        token counts. Leave usage_write_path="" (the default) for a
        provider with no write-cost concept -- it will correctly parse
        as always 0 rather than requiring a path that doesn't exist.
    usage_input_path / usage_output_path: dot-paths to the response's
        TOTAL prompt/input and completion/output token counts. tonst
        subtracts the read+write portions from usage_input_path's
        value to get the "genuinely fresh" input token count -- most
        providers report a combined total here, not just the uncached
        remainder, so don't pre-subtract before configuring this.
    """
    label: str = "custom provider"
    minimum_tokens: int = 0
    cache_read_multiplier: float = 1.0
    cache_write_multiplier: float = 1.0
    usage_read_path: str = ""
    usage_write_path: str = ""
    usage_input_path: str = "usage.prompt_tokens"
    usage_output_path: str = "usage.completion_tokens"


def check_cache_eligibility(parts: PromptParts, config: GenericCacheConfig, token_estimator=None) -> CacheEligibility:
    """
    Same idea as every other provider module's check_cache_eligibility()
    -- a chars/4 smoke test against config.minimum_tokens, not a
    substitute for reading real usage data back from an actual call.
    """
    if token_estimator is None:
        from ..trim import estimate_tokens as token_estimator

    stable_text = "\n\n".join([parts.system or ""] + list(parts.stable_blocks))
    stable_tokens = token_estimator(stable_text) if stable_text.strip() else 0

    return CacheEligibility(
        eligible=stable_tokens >= config.minimum_tokens,
        stable_tokens_estimate=stable_tokens,
        minimum_required=config.minimum_tokens,
        model=config.label,
    )


def build_generic_chat_request(parts: PromptParts, model: str, extra_fields: Optional[dict] = None) -> dict:
    """
    A reasonable DEFAULT request shape for the many providers that mimic
    OpenAI's Chat Completions format (Groq, Together AI, Fireworks,
    DeepSeek, Mistral's chat endpoint, self-hosted vLLM/Ollama/TGI
    OpenAI-compatible servers, and others) -- stable content ordered
    first, variable last, in a plain string. This is a convenience
    starting point, not a guarantee: if your provider's request shape
    genuinely differs (Cohere and Bedrock's own APIs do, for example),
    build the body yourself and use cache_structuring.structure_for_caching()
    directly instead, which makes no assumptions about JSON shape at all.

    extra_fields: merged into the returned body as-is -- use this for
    whatever provider-specific parameters you need (temperature, a
    provider-specific caching flag, etc.) without this function needing
    to know about them.
    """
    from ..cache_structuring import structure_for_caching

    body = {
        "model": model,
        "messages": [{"role": "user", "content": structure_for_caching(parts)}],
    }
    if extra_fields:
        body.update(extra_fields)
    return body


def parse_usage(response_json: dict, config: GenericCacheConfig) -> CacheUsageReport:
    """
    Extracts a normalized CacheUsageReport from ANY provider's response
    JSON, using the dot-paths in `config` instead of a hardcoded shape.
    This is what makes tonst's cost/eligibility reporting work for a
    provider it has never heard of -- the parsing logic here is
    identical for every provider; only the paths change.
    """
    total_input = _get_path(response_json, config.usage_input_path, 0)
    read_tokens = _get_path(response_json, config.usage_read_path, 0) if config.usage_read_path else 0
    write_tokens = _get_path(response_json, config.usage_write_path, 0) if config.usage_write_path else 0
    output_tokens = _get_path(response_json, config.usage_output_path, 0)

    return CacheUsageReport(
        input_tokens=max(0, total_input - read_tokens - write_tokens),
        output_tokens=output_tokens,
        cache_creation_input_tokens=write_tokens,
        cache_read_input_tokens=read_tokens,
    )


def estimated_cost_savings_percent(usage: CacheUsageReport, config: GenericCacheConfig) -> float:
    """
    Same shape and same baseline (this exact token count, sent with no
    caching at all, at the standard 1x rate) as every other provider
    module's cost-savings function -- but driven entirely by
    `config`'s multipliers instead of a hardcoded table, so it works
    for a provider tonst has never been told about by name.
    """
    actual_cost = (
        usage.input_tokens * 1.0
        + usage.cache_creation_input_tokens * config.cache_write_multiplier
        + usage.cache_read_input_tokens * config.cache_read_multiplier
    )
    total_tokens = usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
    no_cache_cost = total_tokens * 1.0
    if no_cache_cost == 0:
        return 0.0
    return round(100 * (no_cache_cost - actual_cost) / no_cache_cost, 1)
