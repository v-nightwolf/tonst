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
history compaction -- all three can call a local model) is an informed
latency tradeoff, not a guess.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

from .redact import redact, redact_with_llm, restore_placeholders, RedactionResult
from .redact_llm import LLMRedactor
from .trim import mechanical_trim, estimate_tokens, flatten_messages
from .local_model import LocalCompressor
from .cache_structuring import PromptParts, structure_for_caching
from .compactor import HistoryCompactor, compact_history

# Every redaction_backend value TonstClient accepts. "none" and "regex"
# need no local model at all; "ollama" and "gliner" each call a local
# model (a generative one via redact_llm.py, or an extractive one via
# gliner_redact.py, respectively) to catch free-text PII regex can't.
REDACTION_BACKENDS = {"none", "regex", "ollama", "gliner"}


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
    # redaction, local compression, or history compaction (each can
    # call a local model) can add real, visible latency on top of the
    # API call itself. structuring_ms is only nonzero for
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
        local_model: str = "gemma2:2b",
        compaction_token_threshold: int = 3000,
        redaction_backend: Optional[str] = None,
        redaction_model: Optional[str] = None,
        gliner_model: str = "urchade/gliner_medium-v2.1",
        compression_model: Optional[str] = None,
        compaction_model: Optional[str] = None,
    ):
        """
        redaction_backend lets a caller choose what catches free-text PII
        (names, addresses, employers, codenames) beyond the always-on...
        actually, beyond regex, which is NOT always-on anymore either --
        see "none" below. Not every caller needs the same coverage: an
        app already running on an enterprise/zero-retention LLM
        agreement, or one whose traffic structurally can't contain PII,
        pays real local latency for a safety margin it may not need.
        Independently, redaction_model / compression_model /
        compaction_model let each Ollama-backed stage use a DIFFERENT
        model instead of one shared `local_model` for everything.

        redaction_backend values:
          - None (default): infer from use_enhanced_redaction for
            backward compatibility -- "ollama" if True, else "regex".
          - "none": skip PII redaction entirely, including regex. Only
            mechanical trim/compression/compaction still run. This is
            NOT the safe default -- use it only when you're confident
            PII exposure genuinely isn't a concern for this traffic.
          - "regex": structured PII only (emails, cards, phones, SSNs,
            IPs) -- fast, dependency-free, catches nothing in free text.
          - "ollama": regex + a local generative model via redact_llm.py
            (LLMRedactor). Needs Ollama running; costs real latency
            (seconds, not milliseconds) -- see research/colab-benchmark-
            findings.md.
          - "gliner": regex + GLiNER, a small extractive/zero-shot NER
            model (gliner_redact.py). No GPU or Ollama needed; CPU
            latency around 150-250ms; structurally can't produce the
            JSON-parsing/hallucination failures a generative model can.
            See research/gliner-sanity-check-findings.md for the full
            recall comparison -- gliner_medium is the validated default
            (do not switch to gliner_large: tested, and it's worse, not
            better). Requires `pip install gliner`, only imported if
            this backend is actually selected -- not a hard dependency
            of tonst otherwise.

        redaction_model / compression_model / compaction_model each
        default to `local_model` when not given, so existing single-
        model callers are unaffected. Splitting them out matters in
        practice: a 2026-09-13 benchmark found gemma3:1b is fast and
        reliable for compression specifically, even though that same
        small model is unreliable for generative redaction (22.78%
        free-text recall) -- pairing redaction_backend="gliner" with
        compression_model="gemma3:1b" captures both findings at once
        instead of one shared model compromising on both jobs.
        """
        self.call_fn = call_fn
        self.use_local_compression = use_local_compression
        self.compressor: Optional[LocalCompressor] = (
            LocalCompressor(model=compression_model or local_model) if use_local_compression else None
        )

        # Resolve the redaction backend. Explicit redaction_backend wins;
        # otherwise fall back to the old boolean for callers who haven't
        # migrated. Validated eagerly so a typo'd backend name fails at
        # construction time, not silently mid-run.
        if redaction_backend is None:
            redaction_backend = "ollama" if use_enhanced_redaction else "regex"
        if redaction_backend not in REDACTION_BACKENDS:
            raise ValueError(
                f"redaction_backend must be one of {sorted(REDACTION_BACKENDS)}, got {redaction_backend!r}"
            )
        self.redaction_backend = redaction_backend
        # Kept for backward compat: OptimizationReport.used_enhanced_redaction
        # and any caller reading this attribute directly still get a
        # meaningful bool, generalized to "any beyond-regex backend ran"
        # rather than specifically "the Ollama one ran".
        self.use_enhanced_redaction = redaction_backend in ("ollama", "gliner")

        self.llm_redactor: Optional[LLMRedactor] = None
        self.gliner_redactor = None
        if redaction_backend == "ollama":
            self.llm_redactor = LLMRedactor(model=redaction_model or local_model)
        elif redaction_backend == "gliner":
            # Imported lazily so `gliner` and its ML dependencies (torch)
            # are only ever required when this backend is actually
            # selected, not for every tonst install.
            from .gliner_redact import GlinerRedactor
            self.gliner_redactor = GlinerRedactor(model=gliner_model)

        # History compaction: summarizes turns a sliding window would
        # otherwise silently drop, instead of just dropping them -- see
        # compactor.py. Off by default for the same reason as the two
        # above (local-model latency), plus it's the one optional step
        # whose fallback can genuinely lose information rather than just
        # missing an optimization.
        self.use_history_compaction = use_history_compaction
        self.history_compactor: Optional[HistoryCompactor] = (
            HistoryCompactor(model=compaction_model or local_model) if use_history_compaction else None
        )
        self.compaction_token_threshold = compaction_token_threshold

    def _redact(self, text: str):
        if self.redaction_backend == "none":
            return RedactionResult(redacted_text=text, mapping={})
        if self.redaction_backend == "regex":
            return redact(text)
        # "ollama" and "gliner" both expose a duck-type-compatible
        # .redact(text) -> object with .redacted_text/.mapping, so the
        # same regex-then-secondary-pass helper works for either one.
        secondary = self.llm_redactor if self.redaction_backend == "ollama" else self.gliner_redactor
        return redact_with_llm(text, secondary)

    def query(self, prompt: str) -> tuple[str, OptimizationReport]:
        t_start = time.perf_counter()
        original_tokens = estimate_tokens(prompt)

        # 1. Redact sensitive fields locally: regex always (unless
        #    redaction_backend="none"), optionally layered with a
        #    second-pass local model for free-text PII. This is the
        #    step most likely to cost real time when that second pass
        #    is on, since it calls a local model.
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
