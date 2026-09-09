"""
providers/presets.py
----------------------
Ready-made GenericCacheConfig instances (see providers/generic.py) for
platforms that DON'T get their own hand-written module, but that are
common enough -- and different enough from the direct API they wrap --
to be worth shipping a verified starting point instead of making every
caller re-derive it. Verified against official docs 2026-09-09.

Both presets below exist ONLY because the wrapping platform actually
changes something (the response shape, or the pricing) versus the
direct API it's built on. Where a platform passes a provider's response
through completely unchanged, there is nothing to add here -- reuse
that provider's own dedicated module directly. That distinction matters
enough to spell out per preset below, because it's easy to assume "a
wrapper always needs its own code" or "never does" -- the true answer,
confirmed by checking rather than guessing, is "it depends, and you
have to check."
"""

from __future__ import annotations
from .generic import GenericCacheConfig

# ---------------------------------------------------------------------
# AWS Bedrock -- Converse API, Claude models
# ---------------------------------------------------------------------
# Confirmed against AWS's own API reference
# (docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html)
# and the Converse API docs: usage fields are CAMELCASE and use
# DIFFERENT NAMES from Anthropic's own direct API -- notably
# cacheWriteInputTokens, not cache_creation_input_tokens. Pointing
# cache_structuring.parse_anthropic_usage() (built for the snake_case,
# direct-API shape) at a Converse response will silently read zeros for
# every cache field rather than error -- this preset exists specifically
# to avoid that silent failure mode.
#
# Pricing ratios confirmed the SAME as Anthropic direct (0.1x read,
# 1.25x write for a 5-minute TTL), just priced through Bedrock's own
# per-model, per-region rate card rather than Anthropic's.
#
# NOTE -- the other Bedrock API needs NO preset at all: calling Claude
# through Bedrock's InvokeModel API (sending Anthropic's own
# Messages-format request body) returns Anthropic's ORIGINAL snake_case
# field names completely unchanged. In that case, use
# cache_structuring.parse_anthropic_usage() directly -- this preset is
# for Converse only.
#
# minimum_tokens varies by the specific Claude version available on
# Bedrock (512-4,096, mirroring the range on cache_structuring.CACHE_MINIMUM_TOKENS)
# -- 1,024 below is a reasonable default, not a guarantee for your
# exact model; check which Claude version your Bedrock deployment uses.
BEDROCK_CONVERSE_CLAUDE = GenericCacheConfig(
    label="AWS Bedrock (Converse API, Claude)",
    minimum_tokens=1024,
    cache_read_multiplier=0.1,
    cache_write_multiplier=1.25,  # 5-minute TTL; use 2.0 to model a 1-hour TTL cache instead
    usage_read_path="usage.cacheReadInputTokens",
    usage_write_path="usage.cacheWriteInputTokens",
    usage_input_path="usage.inputTokens",
    usage_output_path="usage.outputTokens",
)

# ---------------------------------------------------------------------
# Azure OpenAI Service -- Provisioned Throughput (PTU-M) deployments
# ---------------------------------------------------------------------
# Confirmed against Microsoft Learn
# (learn.microsoft.com/en-us/azure/foundry/openai/how-to/prompt-caching):
# for a STANDARD Azure OpenAI deployment, the response field path is
# IDENTICAL to OpenAI's own direct API
# (usage.prompt_tokens_details.cached_tokens) -- so
# providers.openai.parse_openai_usage() already works completely
# unchanged there. NO preset is needed, or provided, for that case.
#
# This preset exists only for Azure's Provisioned-Throughput (PTU-M)
# tier specifically, which is an AZURE-ONLY pricing model with its own
# discount -- up to 100% off cached input tokens, a rate OpenAI's own
# direct per-model table (providers/openai.py) has no equivalent of.
# cache_read_multiplier=0.0 below is the BEST CASE Microsoft advertises
# for this tier -- confirm your specific deployment's actual discount
# before relying on this number; it is not guaranteed to be 100% for
# every PTU-M deployment.
AZURE_OPENAI_PTU_M = GenericCacheConfig(
    label="Azure OpenAI (Provisioned Throughput / PTU-M)",
    minimum_tokens=1024,
    cache_read_multiplier=0.0,
    cache_write_multiplier=1.0,
    usage_read_path="usage.prompt_tokens_details.cached_tokens",
    usage_write_path="usage.prompt_tokens_details.cache_write_tokens",
    usage_input_path="usage.prompt_tokens",
    usage_output_path="usage.completion_tokens",
)
