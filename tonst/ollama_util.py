"""
ollama_util.py
--------------
Shared helper for sizing Ollama's context window (`num_ctx`) per request.

Why this exists: Ollama runs a model with a default context window of only
2,048-4,096 tokens (depending on the Ollama version) regardless of what the
model supports, and when a prompt is longer it SILENTLY drops the start of
the prompt -- no error, no warning in the response. tonst's local-model
calls didn't set num_ctx, which was harmless while inputs were short, but:
  - compaction at the real default threshold (3,000 tokens) sends 3,000+
    tokens of history, so the oldest turns would be cut off before the
    model ever saw them -- a summary silently missing its beginning;
  - LLM redaction of a long prompt would only check the END of the text
    for PII;
  - local compression of a long prompt would "compress" a truncated copy.
Found while preparing the long-history live test (2026-09-25), before any
run hit it.

num_ctx_for() asks for just enough context for the prompt plus the output
budget, rounded up, never below Ollama's usual default and never above the
model's maximum (TONST_OLLAMA_MAX_CTX, default 8192 -- gemma2's limit).
A bigger window costs RAM (the KV cache grows with it), which matters on an
8 GB laptop, so it's sized per call rather than always maxed out. If a
prompt won't fit even at the maximum, fits_context() returns False and the
caller decides (compaction and compression refuse rather than work on a
truncated copy; redaction logs a warning).
"""

from __future__ import annotations
import math
import os

DEFAULT_MIN_CTX = 2048


def max_ctx() -> int:
    try:
        return max(DEFAULT_MIN_CTX, int(os.environ.get("TONST_OLLAMA_MAX_CTX", "8192")))
    except ValueError:
        return 8192


def estimate_prompt_tokens(prompt: str) -> int:
    # Deliberately pessimistic (chars/3, not the chars/4 used for cost
    # estimates): under-estimating here means silent truncation.
    return math.ceil(len(prompt) / 3)


def fits_context(prompt: str, num_predict: int) -> bool:
    return estimate_prompt_tokens(prompt) + num_predict + 128 <= max_ctx()


def num_ctx_for(prompt: str, num_predict: int) -> int:
    needed = estimate_prompt_tokens(prompt) + num_predict + 128
    rounded = int(math.ceil(needed / 1024.0) * 1024)
    return max(DEFAULT_MIN_CTX, min(max_ctx(), rounded))
