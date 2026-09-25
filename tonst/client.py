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
      into the same string pipeline as query(). Pass rolling_state= for
      incremental, cache-friendly compaction (compactor.py).
    - query_rag(question, chunks)             -- retrieved RAG chunks:
      duplicates/near-duplicates removed and (optionally) filtered by
      relevance before they're placed in the variable part of the prompt.

Every entry point can also append its OptimizationReport to a local
savings log (savings_log.py, opt-in) -- `tonst stats` summarizes it.

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
from .compactor import HistoryCompactor, compact_history, compact_history_rolling, RollingSummary, run_fold_job
from .rag import optimize_chunks, format_context, chunk_text
from .savings_log import SavingsLog, redacted_types_from_mapping

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

    # Count of redacted fields by placeholder LABEL only ({"EMAIL": 2}) --
    # never values or placeholder hashes (see savings_log.py for why).
    redacted_types: dict = field(default_factory=dict)

    # History compaction (query_messages() only).
    history_compacted: bool = False
    history_turns_dropped: int = 0
    # Rolling compaction only (query_messages(rolling_state=...)):
    # summary_updated = a fold ran this call; summary_reused = the
    # existing summary was sent unchanged, with no local-model call.
    history_summary_updated: bool = False
    history_summary_reused: bool = False
    # background_summary=True only: a summary was started off the request
    # path this call (it takes effect from the next call).
    history_summary_scheduled: bool = False
    history_fold_failed: bool = False
    # Estimated tokens of history that was NOT sent and is NOT represented
    # in a summary either -- context the model simply never saw. These
    # are included in tokens_saved (they weren't sent), but they're a
    # loss, not a free saving; savings_log/`tonst stats` report them
    # separately so truncation isn't mistaken for optimization.
    history_tokens_lost: int = 0
    # compaction_cache_aware=True only: a summary was due by size but was
    # postponed because it wouldn't yet pay for re-caching the prompt.
    history_fold_postponed: bool = False

    # RAG (query_rag() only): retrieved chunks received vs. sent.
    chunks_in: int = 0
    chunks_sent: int = 0
    # True when relevance filtering was requested (top_k / min_relative_score /
    # max_tokens) but SKIPPED because no chunk matched the question
    # confidently -- only duplicates were removed.
    chunk_filter_skipped: bool = False

    # True only when BOTH original_tokens and sent_tokens came from a real
    # tokenizer (token_counter=...), not the chars/4 estimate.
    token_counts_exact: bool = False

    # Per-step wall-clock time in milliseconds. Everything here runs
    # locally and SEQUENTIALLY before call_ms's network/model call, so
    # these add up rather than overlap -- e.g. turning on enhanced
    # redaction, local compression, or history compaction (each can
    # call a local model) can add real, visible latency on top of the
    # API call itself. structuring_ms is only nonzero for
    # query_structured(); compaction_ms only for query_messages().
    structuring_ms: float = 0.0
    compaction_ms: float = 0.0
    # Time spent on optional exact token counting (token_counter=...);
    # 0 when counting uses the local chars/4 estimate.
    counting_ms: float = 0.0
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
            + self.counting_ms
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
        compaction_timeout: float = 60.0,
        compaction_summarizer: Optional[Callable] = None,
        compaction_cache_aware: bool = False,
        compaction_max_history_tokens: Optional[int] = None,
        compaction_cache_pricing="anthropic",
        redaction_backend: Optional[str] = None,
        redaction_model: Optional[str] = None,
        gliner_model: str = "urchade/gliner_medium-v2.1",
        compression_model: Optional[str] = None,
        compaction_model: Optional[str] = None,
        savings_log=None,
        app_name: Optional[str] = None,
        input_price_per_million: Optional[float] = None,
        token_counter: Optional[Callable[[str], Optional[int]]] = None,
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

        compaction_summarizer (optional): replaces the local model for
        history summaries -- any fn(prompt, model, timeout) -> str | None,
        e.g. summarizers.AnthropicSummarizer() (Claude Haiku). Summaries
        only ever see already-redacted text. Pair it with
        query_messages(..., background_summary=True) so its network
        latency never reaches the user.

        compaction_cache_aware (default False; rolling_state only): set
        True when your call_fn uses provider prompt caching (Anthropic
        cache_control, OpenAI automatic caching). Cached history is already
        cheap to re-send, and every new summary forces the provider to
        re-cache the rest of the prompt, so a summary only pays off if
        enough turns follow it. With this on, a due summary waits until it
        is expected to pay for itself (compactor.estimate_fold_payback):
        in an offline simulation it avoided summaries that raised cost by
        8-15% in 10-16-turn chats, and made no difference from ~20 turns on.
        Pass query_messages(..., expected_remaining_turns=N) if you know
        roughly how long the conversation will run.
        After each response, call rolling_state.observe_cache_usage(
        prompt_tokens, cached_tokens) with the provider's usage numbers:
        the estimate then uses the cache hit rate you actually get, not an
        assumed 100%. On Gemini, whose implicit cache hit 0% of a growing
        chat until ~16k tokens in a live run, this is the difference
        between -15% and +4.5% cost.
        compaction_cache_pricing: the provider's cache prices for that
        estimate -- "anthropic" (writes 1.25x, reads 0.1x), "gemini"
        (implicit caching: no write surcharge, reads 0.1x), or a
        (write_mult, read_mult) tuple.
        compaction_max_history_tokens (optional): with cache-aware
        compaction, summarize anyway once the prompt would exceed this --
        staying inside the context window comes first.

        compaction_timeout (default 60s): how long one local-model
        summarization call may take. Deliberately much longer than the 8s
        the other local-model steps use: a fold has to READ up to
        compaction_token_threshold tokens (3,000 by default), which on a
        CPU/laptop-class machine takes several seconds before generation
        even starts -- live testing on a MacBook Air timed out at 8s on a
        ~7k-token fold and needed ~5s for a ~1k-token one. With rolling
        compaction (rolling_state=...) folds are rare (4 in a 40-turn
        benchmark chat), so a long timeout is affordable. With the
        stateless default, compaction can run on EVERY turn once past the
        threshold -- lower this if that latency matters more to you.

        token_counter (off by default): a callable text -> token count
        used instead of the chars/4 estimate for original_tokens and
        sent_tokens -- e.g. token_count.AnthropicTokenCounter(model=...),
        which uses Anthropic's free count_tokens endpoint. A live test
        found real billed input ~1.8x the chars/4 estimate on
        tool-calling requests, so this matters when you want real
        numbers. PRIVACY: the counter is only ever given text that has
        ALREADY been redacted -- a remote counter must never see raw
        PII -- so with a counter set, original_tokens measures the
        redacted-but-not-yet-trimmed prompt (redaction's own small
        token change isn't counted as a saving or a cost). Each count
        is a network round trip, timed as counting_ms. If a count
        fails, the estimate is used and token_counts_exact is False.

        savings_log (off by default): True for the default location
        (~/.tonst/savings.jsonl, or $TONST_SAVINGS_LOG), a path string,
        or a SavingsLog instance. Every public query_* call then appends
        one line of metrics -- token counts, redaction counts by type,
        timings; never prompt text, PII values or placeholder hashes.
        app_name tags those lines for per-app attribution;
        input_price_per_million adds an ESTIMATED dollar figure (tokens
        saved x that price). Summarize with `tonst stats`.
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
            HistoryCompactor(
                model=compaction_model or local_model,
                timeout=compaction_timeout,
                model_call_fn=compaction_summarizer,
            )
            if use_history_compaction
            else None
        )
        self.compaction_token_threshold = compaction_token_threshold
        self.compaction_cache_aware = compaction_cache_aware
        self.compaction_max_history_tokens = compaction_max_history_tokens
        self.compaction_cache_pricing = compaction_cache_pricing
        self._compaction_summarizer = compaction_summarizer

        if savings_log is True:
            savings_log = SavingsLog()
        elif isinstance(savings_log, str):
            savings_log = SavingsLog(savings_log)
        elif savings_log in (None, False):
            savings_log = None
        elif not isinstance(savings_log, SavingsLog):
            raise TypeError("savings_log must be None, True, a path string, or a SavingsLog")
        self.token_counter = token_counter
        self._bg_executor = None   # created on first background_summary=True call
        self._bg_futures: list = []
        self.savings_log: Optional[SavingsLog] = savings_log
        self.app_name = app_name
        self.input_price_per_million = input_price_per_million

    def _count(self, text: str) -> tuple:
        """
        (tokens, exact, ms). Callers must only pass already-redacted text
        when a token_counter is set -- see the constructor docstring.
        """
        if self.token_counter is None:
            return estimate_tokens(text), False, 0.0
        t0 = time.perf_counter()
        try:
            n = self.token_counter(text)
        except Exception:  # noqa: BLE001 -- counting is optional; never break a call over it
            n = None
        ms = (time.perf_counter() - t0) * 1000
        if n is None:
            return estimate_tokens(text), False, ms
        return int(n), True, ms

    def _start_background_fold(self, job, state: RollingSummary) -> None:
        if self._bg_executor is None:
            from concurrent.futures import ThreadPoolExecutor
            # One worker: folds for a conversation are sequential by nature,
            # and a laptop-class machine can't usefully run two local
            # models at once anyway (see colab-benchmark-findings: contention).
            self._bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tonst-compaction")
        self._bg_futures = [f for f in self._bg_futures if not f.done()]
        self._bg_futures.append(self._bg_executor.submit(run_fold_job, job, self.history_compactor, state))

    def wait_for_background_work(self, timeout: Optional[float] = None) -> bool:
        """
        Blocks until any background summaries (query_messages(...,
        background_summary=True)) have finished. Returns False if the
        timeout expired first. Call before exiting, or before persisting a
        RollingSummary between requests.
        """
        from concurrent.futures import wait
        pending = [f for f in self._bg_futures if not f.done()]
        if not pending:
            return True
        done, not_done = wait(pending, timeout=timeout)
        self._bg_futures = list(not_done)
        return not not_done

    def _log(self, report: "OptimizationReport", method: str) -> None:
        if self.savings_log is not None:
            self.savings_log.record(
                report,
                method=method,
                app=self.app_name,
                input_price_per_million=self.input_price_per_million,
            )

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
        final_response, report = self._run(prompt)
        self._log(report, "query")
        return final_response, report

    def _run(self, prompt: str) -> tuple[str, OptimizationReport]:
        # The actual pipeline behind every query_* method. Kept separate
        # from query() so the wrappers (query_structured, query_messages,
        # query_rag) can adjust the report and log it exactly once.
        t_start = time.perf_counter()
        if self.token_counter is None:
            original_tokens, orig_exact, orig_ms = estimate_tokens(prompt), False, 0.0

        # 1. Redact sensitive fields locally: regex always (unless
        #    redaction_backend="none"), optionally layered with a
        #    second-pass local model for free-text PII. This is the
        #    step most likely to cost real time when that second pass
        #    is on, since it calls a local model.
        t0 = time.perf_counter()
        redaction = self._redact(prompt)
        t1 = time.perf_counter()

        if self.token_counter is not None:
            # Counted AFTER redaction, never before: a remote counter must not see raw PII.
            original_tokens, orig_exact, orig_ms = self._count(redaction.redacted_text)
            t1 = time.perf_counter()  # keep counting time out of trim_ms

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

        sent_tokens, sent_exact, sent_ms = self._count(trimmed)
        t3 = time.perf_counter()  # keep counting time out of call_ms

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
            redacted_types=redacted_types_from_mapping(redaction.mapping),
            locally_compressed=locally_compressed,
            used_enhanced_redaction=self.use_enhanced_redaction,
            redaction_ms=(t1 - t0) * 1000 - orig_ms,
            trim_ms=(t2 - t1) * 1000,
            compression_ms=(t3 - t2) * 1000 - sent_ms,
            call_ms=(t4 - t3) * 1000,
            counting_ms=orig_ms + sent_ms,
            token_counts_exact=orig_exact and sent_exact,
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

        final_response, report = self._run(ordered_prompt)

        structuring_ms = (t1 - t0) * 1000
        report = replace(
            report,
            structuring_ms=structuring_ms,
            total_ms=report.total_ms + structuring_ms,
        )
        self._log(report, "query_structured")
        return final_response, report

    def _summarizer_price_ratio(self) -> float:
        """Paid summarizer's input price relative to the main model's (0 for a local model)."""
        price = getattr(self._compaction_summarizer, "price_input", None)
        if not price:
            return 0.0
        # Main model price unknown: assume Sonnet-class ($3/M input).
        return float(price) / float(self.input_price_per_million or 3.0)

    def query_messages(
        self,
        messages: list[dict],
        keep_last_n: int = 6,
        rolling_state: Optional[RollingSummary] = None,
        background_summary: bool = False,
        expected_remaining_turns: Optional[int] = None,
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

        rolling_state (optional, recommended for live chats): pass a
        compactor.RollingSummary -- one per conversation, reused on every
        call, updated in place -- to switch to incremental compaction
        (compactor.compact_history_rolling()). Instead of re-summarizing
        all older turns on every call, turns leaving the window are kept
        verbatim until they reach compaction_token_threshold, then folded
        into the existing summary in one local-model call. Between folds
        the prompt only ever grows at the end, so the provider's prefix
        cache keeps hitting, and no local-model call happens at all.
        Prefer rolling_state over the stateless default whenever the
        provider caches prompts: in live testing (Anthropic, 24 turns) the
        stateless path cost MORE than no compaction at all ($0.0330 vs.
        $0.0295) and added 2.5-5 s to 7 of 24 turns, while rolling matched
        no-compaction on cost and on typical latency.

        background_summary (with rolling_state only): when a summary is
        due, run the local model on a background thread instead of making
        this call wait for it. It starts BEFORE the API call, so the two
        overlap, and the new summary is used from the next call on (this
        call still sends those turns verbatim). This removes the local
        model from response time entirely -- a blocking summary measured
        ~4.7 s on a MacBook Air. Call wait_for_background_work() before
        shutting down, or before persisting rolling_state.to_dict() if
        you store state between requests.

        expected_remaining_turns (optional, with compaction_cache_aware):
        how many more turns you expect this conversation to have. Without
        it, tonst assumes about half as many again as it has had so far.

        Ordering that matters: each message is redacted INDIVIDUALLY,
        before compaction runs -- never the other way around. A
        summarization step is a local rewrite, and (same principle as
        redact_and_trim_parts()) a rewrite step must never see real PII,
        only placeholders, or it risks reformatting/paraphrasing a
        sensitive value into a shape the redaction patterns won't
        recognize. The combined mapping from all messages is what
        restores real values in the final response, not query()'s own
        (necessarily empty, since the text it sees is already redacted)
        mapping. (Placeholders are deterministic, so a rolling summary
        built on an earlier call still restores correctly later.)

        IMPORTANT: unlike this client's other fail-soft local-model
        steps (which only skip an optional optimization when the local
        model is unavailable), a failed compaction here still drops the
        older turns -- it falls back to plain truncation, not to "keep
        everything." This method trades some conversation memory for a
        hard cap on tokens; it is not lossless.
        """
        if background_summary and rolling_state is None:
            raise ValueError("background_summary=True needs rolling_state (stateless compaction can't defer)")
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

        if rolling_state is not None:
            rolling = compact_history_rolling(
                redacted_messages,
                compactor=self.history_compactor,
                state=rolling_state,
                keep_last_n=keep_last_n,
                token_threshold=self.compaction_token_threshold,
                defer_fold=background_summary,
                cache_aware=self.compaction_cache_aware,
                expected_remaining_turns=expected_remaining_turns,
                summarizer_price_ratio=self._summarizer_price_ratio(),
                max_history_tokens=self.compaction_max_history_tokens,
                cache_pricing=self.compaction_cache_pricing,
            )
            if rolling.fold_job is not None:
                # Start the slow local-model call now, in parallel with the API call below.
                self._start_background_fold(rolling.fold_job, rolling_state)
            compacted_messages = rolling.messages
            history_fields = dict(
                history_summary_scheduled=rolling.fold_job is not None,
                history_compacted=rolling_state.summary is not None,
                history_turns_dropped=rolling.folded_turns + rolling.dropped_turns,
                history_summary_updated=rolling.summary_updated,
                history_summary_reused=rolling.summary_reused,
                history_fold_failed=rolling.fold_failed,
                history_tokens_lost=rolling.tokens_lost,
                history_fold_postponed=rolling.fold_postponed_for_cache,
            )
        else:
            result = compact_history(
                redacted_messages,
                compactor=self.history_compactor,
                keep_last_n=keep_last_n,
                token_threshold=self.compaction_token_threshold,
            )
            compacted_messages = result.messages
            history_fields = dict(
                history_compacted=result.compacted,
                history_turns_dropped=result.dropped_turns,
                history_tokens_lost=0 if result.compacted else result.older_tokens_estimate,
            )
        t_compact = time.perf_counter()

        # original_tokens should reflect the TRUE original size (before
        # redaction/compaction ever touched it), or "tokens saved" would
        # only capture mechanical trim's contribution and hide
        # compaction's -- usually the bigger win for long histories.
        if self.token_counter is None:
            original_tokens, orig_exact, orig_ms = estimate_tokens(flatten_messages(messages)), False, 0.0
        else:
            # Redacted history only -- a remote counter must never see raw PII.
            original_tokens, orig_exact, orig_ms = self._count(flatten_messages(redacted_messages))

        flat_prompt = flatten_messages(compacted_messages)
        final_response, report = self._run(flat_prompt)

        # final_response was restored against _run()'s OWN mapping,
        # which is empty (flat_prompt was already redacted, so its
        # internal _redact() pass found nothing new). Apply the REAL
        # mapping collected above -- this is what actually puts real PII
        # back if the model echoed a placeholder in its response.
        final_response = restore_placeholders(final_response, mapping)

        redaction_ms = (t_redact - t_start) * 1000
        compaction_ms = (t_compact - t_redact) * 1000

        report = replace(
            report,
            original_tokens=original_tokens,
            redacted_fields=len(mapping),
            redacted_types=redacted_types_from_mapping(mapping),
            redaction_ms=report.redaction_ms + redaction_ms,
            compaction_ms=compaction_ms,
            counting_ms=report.counting_ms + orig_ms,
            token_counts_exact=report.token_counts_exact and orig_exact,
            total_ms=report.total_ms + redaction_ms + compaction_ms + orig_ms,
            **history_fields,
        )
        self._log(report, "query_messages")
        return final_response, report

    def query_rag(
        self,
        question: str,
        chunks: list,
        system: Optional[str] = None,
        stable_blocks: Optional[list] = None,
        top_k: Optional[int] = None,
        min_relative_score: float = 0.0,
        max_tokens: Optional[int] = None,
        dedupe_threshold: float = 0.8,
        min_matched_terms: int = 2,
    ) -> tuple[str, OptimizationReport]:
        """
        Entry point for RAG: `chunks` is what your retriever returned
        (strings, or dicts with a "text"/"content" key). Duplicate and
        near-duplicate chunks are removed, and -- only if you ask via
        top_k / min_relative_score / max_tokens -- low-relevance chunks
        too (rag.optimize_chunks(); lexical scoring, and it never
        filters unless the best chunk shares at least min_matched_terms
        distinct words with the question).

        The kept chunks + the question form the VARIABLE part of the
        prompt; `system` and `stable_blocks` (instructions, reference
        material reused across questions) go first, so provider prefix
        caching still works. Everything is then redacted, trimmed and
        sent exactly like query_structured().

        original_tokens in the report counts ALL chunks as received, so
        the savings from dropped chunks are visible in percent_saved;
        chunks_in / chunks_sent show how many were cut.
        """
        t0 = time.perf_counter()
        selection = optimize_chunks(
            chunks,
            question,
            top_k=top_k,
            min_relative_score=min_relative_score,
            max_tokens=max_tokens,
            dedupe_threshold=dedupe_threshold,
            min_matched_terms=min_matched_terms,
        )
        parts = PromptParts(
            system=system,
            stable_blocks=list(stable_blocks or []),
            variable=format_context(selection.texts, question),
        )
        ordered_prompt = structure_for_caching(parts)
        t1 = time.perf_counter()

        unfiltered = structure_for_caching(
            PromptParts(
                system=system,
                stable_blocks=list(stable_blocks or []),
                variable=format_context([chunk_text(c) for c in chunks], question),
            )
        )

        if self.token_counter is None:
            orig_tokens, orig_exact, orig_ms = estimate_tokens(unfiltered), False, 0.0
        else:
            # The unfiltered prompt includes chunks that will never be sent, but
            # a remote counter must still only see redacted text -- so this
            # costs one extra redaction pass (cheap for regex; noticeable for
            # the gliner/ollama backends).
            tc = time.perf_counter()
            orig_tokens, orig_exact, _ = self._count(self._redact(unfiltered).redacted_text)
            orig_ms = (time.perf_counter() - tc) * 1000  # includes the extra redaction pass

        final_response, report = self._run(ordered_prompt)
        structuring_ms = (t1 - t0) * 1000
        report = replace(
            report,
            counting_ms=report.counting_ms + orig_ms,
            token_counts_exact=report.token_counts_exact and orig_exact,
            original_tokens=orig_tokens,
            chunks_in=len(chunks),
            chunks_sent=len(selection.chunks),
            chunk_filter_skipped=selection.fell_back,
            structuring_ms=structuring_ms,
            total_ms=report.total_ms + structuring_ms + orig_ms,
        )
        self._log(report, "query_rag")
        return final_response, report
