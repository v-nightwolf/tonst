"""
client.py
---------
The public SDK surface. A developer wraps their existing "call the paid
LLM API" function with TonstClient, and every call automatically
goes through, in order:

    1. Semantic cache check        (skip the paid call entirely on a hit)
    2. Local PII redaction         (strip sensitive data before it leaves)
    3. Mechanical trimming         (dedupe/whitespace/history truncation)
    4. Optional local-model compress (Ollama, off by default)
    5. The real paid API call      (only the trimmed/redacted prompt)
    6. Re-insert redacted values into the response
    7. Store the result in cache for next time

This file has no knowledge of which cloud provider you use -- you pass in
your own `call_fn(prompt: str) -> str`. That's what makes it provider-
agnostic (Claude, OpenAI, whatever).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

from .cache import SemanticCache
from .redact import redact, redact_with_llm
from .redact_llm import LLMRedactor
from .trim import mechanical_trim, estimate_tokens
from .local_model import LocalCompressor


@dataclass
class OptimizationReport:
    original_tokens: int
    sent_tokens: int
    cache_hit: bool
    redacted_fields: int
    locally_compressed: bool
    used_enhanced_redaction: bool = False

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.sent_tokens)

    @property
    def percent_saved(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round(100 * self.tokens_saved / self.original_tokens, 1)


class TonstClient:
    def __init__(
        self,
        call_fn: Callable[[str], str],
        use_local_compression: bool = False,
        use_enhanced_redaction: bool = False,
        cache_similarity_threshold: float = 0.90,
        local_model: str = "llama3.2:1b",
    ):
        self.call_fn = call_fn
        self.cache = SemanticCache(similarity_threshold=cache_similarity_threshold)
        self.use_local_compression = use_local_compression
        self.compressor: Optional[LocalCompressor] = (
            LocalCompressor(model=local_model) if use_local_compression else None
        )
        # Enhanced redaction catches free-text PII (names, addresses,
        # employers, codenames) that regex structurally cannot -- see
        # redact_llm.py. Off by default because it costs local latency and
        # needs Ollama running; regex-only redaction still always applies.
        self.use_enhanced_redaction = use_enhanced_redaction
        self.llm_redactor: Optional[LLMRedactor] = (
            LLMRedactor(model=local_model) if use_enhanced_redaction else None
        )

    def query(self, prompt: str) -> tuple[str, OptimizationReport]:
        original_tokens = estimate_tokens(prompt)

        # 1. Cache check happens on the *original* prompt, before any
        #    redaction/trimming, so semantically identical repeat questions
        #    match regardless of exact PII values.
        cached = self.cache.lookup(prompt)
        if cached is not None:
            report = OptimizationReport(
                original_tokens=original_tokens,
                sent_tokens=0,
                cache_hit=True,
                redacted_fields=0,
                locally_compressed=False,
            )
            return cached, report

        # 2. Redact sensitive fields locally: regex always, optionally
        #    layered with the local-LLM pass for free-text PII.
        if self.use_enhanced_redaction and self.llm_redactor is not None:
            redaction = redact_with_llm(prompt, self.llm_redactor)
        else:
            redaction = redact(prompt)

        # 3. Mechanical trim (safe, always applied).
        trimmed = mechanical_trim(redaction.redacted_text)

        # 4. Optional local-model compression (off by default; fails soft).
        locally_compressed = False
        if self.use_local_compression and self.compressor is not None:
            trimmed, locally_compressed = self.compressor.compress(trimmed)

        sent_tokens = estimate_tokens(trimmed)

        # 5. The actual paid call -- only ever sees the trimmed/redacted text.
        raw_response = self.call_fn(trimmed)

        # 6. Put real values back for the end user/app.
        final_response = redaction.restore(raw_response)

        # 7. Cache under the ORIGINAL prompt so future lookups match on it.
        self.cache.store(prompt, final_response)

        report = OptimizationReport(
            original_tokens=original_tokens,
            sent_tokens=sent_tokens,
            cache_hit=False,
            redacted_fields=len(redaction.mapping),
            locally_compressed=locally_compressed,
            used_enhanced_redaction=self.use_enhanced_redaction,
        )
        return final_response, report
