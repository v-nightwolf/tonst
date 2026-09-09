"""
client.py
---------
The public SDK surface. A developer wraps their existing "call the paid
LLM API" function with TonstClient, and every call automatically
goes through, in order:

    1. Local PII redaction          (strip sensitive data before it leaves)
    2. Mechanical trimming          (dedupe/whitespace/history truncation)
    3. Optional local-model compress (Ollama, off by default)
    4. The real paid API call       (only the trimmed/redacted prompt)
    5. Re-insert redacted values into the response

This file has no knowledge of which cloud provider you use -- you pass in
your own `call_fn(prompt: str) -> str`. That's what makes it provider-
agnostic (Claude, OpenAI, whatever).

Three entry points, for three input shapes:
    - query(prompt: str)                     -- one flat string.
    - query_structured(parts: PromptParts)    -- system/stable/variable,
      kept separate so prompt-caching structuring can order and (with
      redact_and_trim_parts()) cache-mark them correctly.
    - query_messages(messages: list[dict])    -- a role/content turn
      history, with a sliding window (and optionally local-model-based
      compaction of what falls outside it) applied before flattening
      into the same string pipeline as query().

Note: an earlier version of this pipeline also had a semantic response
cache as step 0 (skip the paid call entirely on a "close enough" repeat
question). That was deliberately removed -- see ROADMAP.md. Two reasons:
it required unbounded local storage growth to be useful at scale, and a
similarity-based cache can silently return a wrong answer to a question
that only superficially resembles a previous one, with no visible sign
of failure. Prompt-caching *structuring* (see cache_structuring.py) is
a different, safer thing: it never skips the real model call, it just
shapes the request so the *provider's own* caching (which always
re-verifies against the live model) can discount repeated prefix
content.

Every step here runs LOCALLY, sequentially, before the network call --
which means every step's wall-clock time is added to the total, not
overlapped with it. OptimizationReport times each step so that turning
on an optional heavier step (enhanced redaction, local compression, or
history compaction -- all three call a local model via Ollama) is an
informed latency tradeoff, not a guess.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from .redact import redact, redact_with_llm, restore_placeholders
from .redact_llm import LLMRedactor
from .trim import mechanical_trim, estimate_tokens, flatten_messages
from .local_model import LocalCompressor
from .cache_structuring import PromptParts, structure_for_caching
from .compactor import HistoryCompactor, compact_history


@dataclass
class OptimizationReport:
    original_tokens: int
    sent_tokens: int
    redacted_fields: int
    locally_compressed: bool
    used_enhanced_redaction: bool = False

    # History compaction (query_messages() only).
    history_compacted: bool = False
    history_turns_dropped: int = 0

    # Per-step wall-clock time in milliseconds. Everything here runs
    # locally and SEQUENTIALLY before call_ms's network/model call, so
    # these add up rather than overlap -- e.g. turning on enhanced
    # redaction, local compression, or history compaction (all three
    # call a local model via Ollama) can add real, visible latency on
    # top of the API call itself. structuring_ms is only nonzero for
    # query_structured(); compaction_ms only for query_messages().
    structuring_ms: float = 0.0
    compaction_ms: float = 0.0
    redaction_ms: float = 0.0
    trim_ms: float = 0.0
    compression_ms: float = 0.0
    call_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.sent_tokens)

    @property
    def percent_saved(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round(100 * self.tokens_saved / self.original_tokens, 1)

    @property
    def local_overhead_ms(self) -> float:
        """
        Everything this call spent BEFORE the actual paid API call --
        the latency cost of running tonst at all, separate from however
        long the model itself took to respond.
        """
        return (
            self.structuring_ms
            + self.compaction_ms
            + self.redaction_ms
            + self.trim_ms
            + self.compression_ms
        )


@dataclass
class StructuredRedactionResult:
    """
    Result of redacting+trimming a PromptParts while keeping the
    system/stable/variable split intact, instead of collapsing everything
    into one string first. Keeping the split matters because
    cache_structuring.build_anthropic_cache_request() needs the stable
    blocks separated out to place the cache_control breakpoint correctly.
    """
    parts: PromptParts
    # Merged placeholder -> original mapping across all parts, kept only
    # in memory on the local machine -- same contract as RedactionResult.
    mapping: dict = field(default_factory=dict)

    def restore(self, text: str) -> str:
        return restore_placeholders(text, self.mapping)


class TonstClient:
    def __init__(
        self,
        call_fn: Callable[[str], str],
        use_local_compression: bool = False,
        use_enhanced_redaction: bool = False,
        use_history_compaction: bool = False,
        local_model: str = "llama3.2:1b",
        compaction_token_threshold: int = 3000,
    ):
        self.call_fn = call_fn
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
        # History compaction: summarizes turns a sliding window would
        # otherwise silently drop, instead of just dropping them -- see
        # compactor.py. Off by default for the same reason as the two
        # above (local-model latency), plus it's the one optional step
        # whose fallback can genuinely lose information rather than just
        # missing an optimization.
        self.use_history_compaction = use_history_compaction
        self.history_compactor: Optional[HistoryCompactor] = (
            HistoryCompactor(model=local_model) if use_history_compaction else None
        )
        self.compaction_token_threshold = compaction_token_threshold

    def _redact(self, text: str):
        if self.use_enhanced_redaction and self.llm_redactor is not None:
            return redact_with_llm(text, self.llm_redactor)
        return redact(text)

    def query(self, prompt: str) -> tuple[str, OptimizationReport]:
        t_start = time.perf_counter()
        original_tokens = estimate_tokens(prompt)

        # 1. Redact sensitive fields locally: regex always, optionally
        #    layered with the local-LLM pass for free-text PII. This is
        #    the step most likely to cost real time when enhanced
        #    redaction is on, since that calls a local model via Ollama.
        t0 = time.perf_counter()
        redaction = self._redact(prompt)
        t1 = time.perf_counter()

        # 2. Mechanical trim (safe, always applied, effectively free).
        trimmed = mechanical_trim(redaction.redacted_text)
        t2 = time.perf_counter()

        # 3. Optional local-model compression (off by default; fails
        #    soft). The other step that calls a local model -- expect
        #    this to dominate total_ms whenever it's enabled.
        locally_compressed = False
        if self.use_local_compression and self.compressor is not None:
            trimmed, locally_compressed = self.compressor.compress(trimmed)
        t3 = time.perf_counter()

        sent_tokens = estimate_tokens(trimmed)

        # 4. The actual paid call -- only ever sees the trimmed/redacted
        #    text. call_ms is the network + model time, not tonst's own
        #    overhead; compare it against local_overhead_ms on the
        #    report to see the split.
        raw_response = self.call_fn(trimmed)
        t4 = time.perf_counter()

        # 5. Put real values back for the end user/app.
        final_response = redaction.restore(raw_response)
        t5 = time.perf_counter()

        report = OptimizationReport(
            original_tokens=original_tokens,
            sent_tokens=sent_tokens,
            redacted_fields=len(redaction.mapping),
            locally_compressed=locally_compressed,
            used_enhanced_redaction=self.use_enhanced_redaction,
            redaction_ms=(t1 - t0) * 1000,
            trim_ms=(t2 - t1) * 1000,
            compression_ms=(t3 - t2) * 1000,
            call_ms=(t4 - t3) * 1000,
            total_ms=(t5 - t_start) * 1000,
        )
        return final_response, report

    def redact_and_trim_parts(self, parts: PromptParts) -> StructuredRedactionResult:
        """
        Same redaction + mechanical trim as query(), but applied to each
        part of a PromptParts independently, so the system/stable/variable
        split survives instead of being collapsed into one string.

        Use this before cache_structuring.build_anthropic_cache_request()
        when you want the stable/cached blocks redacted. Because
        redact.py/redact_llm.py now produce deterministic placeholders
        (a hash of the original value, not a random UUID -- see
        ROADMAP.md), redacting the same stable content on a later call
        produces byte-for-byte identical output, which is required for
        the provider to recognize it as the same cached prefix. A random
        placeholder would silently defeat caching on every single call.
        """
        mapping: dict = {}

        def _process(text: str) -> str:
            if not text:
                return text
            redaction = self._redact(text)
            mapping.update(redaction.mapping)
            return mechanical_trim(redaction.redacted_text)

        processed = PromptParts(
            system=_process(parts.system) if parts.system else parts.system,
            stable_blocks=[_process(b) for b in parts.stable_blocks],
            variable=_process(parts.variable),
        )
        return StructuredRedactionResult(parts=processed, mapping=mapping)

    def query_structured(self, parts: PromptParts) -> tuple[str, OptimizationReport]:
        """
        Same pipeline as query(), but takes structured input (system
        instructions / stable reusable context / the actual variable
        question) instead of one flat string, and orders them stable-
        first, variable-last before anything else happens.

        Why this exists: provider-side prompt caching (Anthropic, OpenAI,
        Gemini) only discounts repeated content when it's (a) ordered
        first and (b) byte-for-byte identical across calls. A flat
        prompt string built by hand can't guarantee either -- it's easy
        to interleave the fresh question ahead of the reused system
        prompt, which silently defeats caching with no error. Keeping
        system/stable/variable separate until the last moment is what
        makes correct ordering automatic instead of something every
        caller has to get right themselves.

        This method still goes through call_fn(str) like query(), so it
        gets you correct ORDERING (which is all OpenAI/Gemini's automatic
        prefix caching needs) plus redaction/trimming/compression as
        usual. It does NOT set Anthropic's explicit cache_control
        breakpoint, because that requires a structured JSON request body,
        not a flat string -- call_fn's signature can't carry that. For a
        real Anthropic cache_control breakpoint, use
        redact_and_trim_parts() followed by
        cache_structuring.build_anthropic_cache_request() and call the
        Anthropic API directly instead of through call_fn.
        """
        t0 = time.perf_counter()
        ordered_prompt = structure_for_caching(parts)
        t1 = time.perf_counter()

        final_response, report = self.query(ordered_prompt)

        structuring_ms = (t1 - t0) * 1000
        return final_response, replace(
            report,
            structuring_ms=structuring_ms,
            total_ms=report.total_ms + structuring_ms,
        )

    def query_messages(
        self, messages: list[dict], keep_last_n: int = 6
    ) -> tuple[str, OptimizationReport]:
        """
        Entry point for callers who track conversation as a list of
        {"role", "content"} turns (the common chat-app shape) instead of
        one flat string. Always caps history to the system message(s)
        plus the last keep_last_n turns -- old turns are the single
        biggest silent token cost in chat apps. If
        use_history_compaction=True (constructor flag) and the dropped
        portion is large enough to be worth it, those older turns are
        summarized by a local model into one condensed message instead
        of being discarded outright; otherwise they're just dropped,
        same as trim.truncate_history().

        Ordering that matters: each message is redacted INDIVIDUALLY,
        before compaction runs -- never the other way around. A
        summarization step is a local rewrite, and (same principle as
        redact_and_trim_parts()) a rewrite step must never see real PII,
        only placeholders, or it risks reformatting/paraphrasing a
        sensitive value into a shape the redaction patterns won't
        recognize. The combined mapping from all messages is what
        restores real values in the final response, not query()'s own
        (necessarily empty, since the text it sees is already redacted)
        mapping.

        IMPORTANT: unlike this client's other fail-soft local-model
        steps (which only skip an optional optimization when the local
        model is unavailable), a failed compaction here still drops the
        older turns -- it falls back to plain truncation, not to "keep
        everything." This method trades some conversation memory for a
        hard cap on tokens; it is not lossless.
        """
        t_start = time.perf_counter()

        # Redact each message individually, BEFORE compaction sees any
        # of it. See docstring above for why the order is non-negotiable.
        redacted_messages = []
        mapping: dict = {}
        for m in messages:
            content = m.get("content", "")
            if content:
                redaction = self._redact(content)
                mapping.update(redaction.mapping)
                redacted_messages.append({**m, "content": redaction.redacted_text})
            else:
                redacted_messages.append(m)
        t_redact = time.perf_counter()

        result = compact_history(
            redacted_messages,
            compactor=self.history_compactor,
            keep_last_n=keep_last_n,
            token_threshold=self.compaction_token_threshold,
        )
        t_compact = time.perf_counter()

        # original_tokens should reflect the TRUE original size (before
        # redaction/compaction ever touched it), or "tokens saved" would
        # only capture mechanical trim's contribution and hide
        # compaction's -- usually the bigger win for long histories.
        original_tokens = estimate_tokens(flatten_messages(messages))

        flat_prompt = flatten_messages(result.messages)
        final_response, report = self.query(flat_prompt)

        # final_response was restored against query()'s OWN mapping,
        # which is empty (flat_prompt was already redacted, so query()'s
        # internal _redact() pass found nothing new). Apply the REAL
        # mapping collected above -- this is what actually puts real PII
        # back if the model echoed a placeholder in its response.
        final_response = restore_placeholders(final_response, mapping)

        redaction_ms = (t_redact - t_start) * 1000
        compaction_ms = (t_compact - t_redact) * 1000

        return final_response, replace(
            report,
            original_tokens=original_tokens,
            redacted_fields=len(mapping),
            history_compacted=result.compacted,
            history_turns_dropped=result.dropped_turns,
            redaction_ms=report.redaction_ms + redaction_ms,
            compaction_ms=compaction_ms,
            total_ms=report.total_ms + redaction_ms + compaction_ms,
        )
