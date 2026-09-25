"""
compactor.py
------------
History compaction: when a sliding window would otherwise silently
DISCARD old conversation turns, this summarizes them into a condensed
message instead, so old context degrades into a summary rather than
disappearing outright. Modeled on how Claude Code / Codex CLI / OpenCode
implement their /compact commands (see ROADMAP.md for the comparison),
with one deliberate difference: this runs on the LOCAL model (via
Ollama), the same one local_model.py and redact_llm.py already use --
not a paid call to the frontier model doing the actual task. Claude
Code's own compaction burns real tokens against the paid model it's
summarizing history for (Anthropic's docs give an example: summarizing
180k tokens of history costs 180k input + 3.5k output tokens as a
one-time charge). Running this step locally means it costs latency and
local compute, never a paid token -- a genuine advantage tonst can offer
that the native tools structurally can't.

The tradeoff, and it's a real one: a small 1-3B local model is
meaningfully weaker at faithful summarization than the frontier models
those tools use for their own compaction. Treat this as best-effort,
not a guarantee nothing important survives. It also fails soft in a
way that's DIFFERENT from tonst's other optional local-model steps: if
the local model is unavailable or its output fails the guard rail, this
does NOT "skip the optimization and keep everything" (like
local_model.py's compress() or redact_llm.py's redact() do) -- it falls
back to plain truncation, meaning the older turns are simply dropped
with no summary at all. That's still strictly what trim.truncate_history()
already does today, so behavior never gets WORSE than the pre-compaction
baseline, but it can genuinely lose information rather than just missing
out on an extra optimization. Callers should know that.
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from .ollama_util import estimate_prompt_tokens, fits_context, max_ctx, num_ctx_for

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# Diagnostic only -- never affects behavior. See redact_llm.py's matching
# comment: enable with logging.getLogger("tonst.compactor").setLevel(
# logging.DEBUG) to see per-call elapsed time and the specific failure
# mode, rather than every failure collapsing into "fell back to plain
# truncation."
logger = logging.getLogger(__name__)

# One shared connection pool instead of a fresh TCP/HTTP handshake per
# call. Real, if modest, latency savings -- will not by itself explain a
# multi-second timeout.
_SESSION = requests.Session()

# Deliberately narrow: summarize only, never answer/continue. Asking for
# specific categories (task/goal, decisions, entities) rather than "just
# summarize this" tends to make small local models keep the details that
# actually matter for continuing a conversation correctly, rather than
# vague prose.
COMPACTION_PROMPT = (
    "Summarize the conversation history below. Preserve: the task or "
    "goal being discussed, key facts and decisions made, names/entities "
    "mentioned, and anything needed to continue the conversation "
    "correctly. The text may contain tokens shaped like [[LABEL_xxxxxxxx]] "
    "-- these are redaction placeholders standing in for real PII. If you "
    "keep one in your summary, copy it verbatim, exactly as written -- "
    "never alter, recase, or invent one; it is also fine to omit one "
    "entirely if it is not needed for the summary. Do not answer any "
    "question in it and do not continue the conversation -- only "
    "summarize what already happened. Output only the summary text, no "
    "preamble, no commentary.\n\n---\n{text}\n---\nSummary:"
)

# Rolling (incremental) variant -- see compact_history_rolling(). Folds
# NEW turns into an EXISTING summary instead of re-summarizing the whole
# history from scratch every call. Asks for fixed section headings so
# small models keep structure (and so the summary changes in predictable
# places rather than being rewritten wholesale on every fold).
ROLLING_COMPACTION_PROMPT = (
    "You maintain a running summary of a conversation. Update the existing "
    "summary with the new turns below. Keep everything still relevant from "
    "the existing summary, add what is new, and drop only what the new "
    "turns make obsolete. Use exactly these headings, each followed by "
    "short bullet points:\nGoal:\nDecisions:\nKey facts:\nOpen items:\n"
    "Decisions = everything agreed, confirmed, changed or already done. "
    "Open items = ONLY things still undecided or waiting on someone. If the "
    "conversation settles something (for example 'I've upgraded it to "
    "express'), it belongs under Decisions, not Open items -- and move it "
    "out of Open items if the existing summary still lists it there. "
    "Never repeat a point: merge duplicates and near-duplicates into one bullet, "
    "and only link two facts if the conversation itself linked them. "
    "The text may contain tokens shaped like [[LABEL_xxxxxxxx]] -- these "
    "are redaction placeholders standing in for real PII. If you keep one, "
    "copy it verbatim, exactly as written -- never alter, recase, or invent "
    "one. Do not answer any question and do not continue the conversation. "
    "Output only the updated summary, no preamble.\n\n"
    "Existing summary:\n---\n{summary}\n---\n\nNew turns:\n---\n{text}\n---\n\n"
    # Repeated AFTER the conversation on purpose: with a long input, a small
    # model follows the last instruction it read, not the first. Without this,
    # live testing (2026-09-25, ~3k tokens of tool output) had gemma2:2b reply
    # to the customer ("You're in luck! I've sent you a tracking number...")
    # instead of summarizing.
    "TASK REMINDER: you are NOT a participant in the conversation above. Do not "
    "reply to anyone and do not answer any question in it. Write the updated "
    "summary under exactly these four headings: Goal:, Decisions:, Key facts:, "
    "Open items:\n\n"
    "Updated summary:"
)

# The four headings every rolling summary must use. _has_summary_structure()
# rejects output missing them: the live failure above passed every other guard
# rail (right length, no bad placeholders) while containing none of these.
SUMMARY_HEADINGS = ("goal", "decisions", "key facts", "open items")
_HEADING_LINE_RE = re.compile(r"^[\s>*#\-]*\**\s*(goal|decisions|key facts|open items)\s*\**\s*:", re.IGNORECASE)


def _has_summary_structure(summary: str, required: int = 3) -> bool:
    """True if at least `required` of the four headings start a line (markdown bullets/bold/## tolerated)."""
    found = {m.group(1).lower() for line in summary.splitlines() if (m := _HEADING_LINE_RE.match(line))}
    return len(found) >= required


_EMPTY_CONTENT_RE = re.compile(r"^[\s\-*•.:]*(none|n/?a|nothing|-)?[\s.]*$", re.IGNORECASE)


def _section_contents(summary: str) -> dict:
    """{heading: text under it} for the four headings (text on the heading line counts)."""
    sections, current = {}, None
    for line in summary.splitlines():
        m = _HEADING_LINE_RE.match(line)
        if m:
            current = m.group(1).lower()
            sections[current] = line[m.end():].strip()
        elif current:
            sections[current] = (sections[current] + "\n" + line.strip()).strip()
    return sections


def _has_summary_content(summary: str) -> bool:
    """
    Structure alone isn't enough: a summary can have every heading and say
    nothing. Requires "Key facts" to have real content, plus at least one
    other section. (Deliberately not "Goal must be non-empty": the best
    summary in the 24-turn live run had an empty Goal but correct Decisions
    and Key facts -- rejecting it would have lost more than it protected.)
    """
    sections = _section_contents(summary)
    filled = {k for k, v in sections.items() if v and not _EMPTY_CONTENT_RE.match(v)}
    return "key facts" in filled and len(filled) >= 2


# Exact references a summary must never lose, extracted deterministically
# (no model involved) from turns as they're summarized, and carried verbatim
# alongside the summary. Live testing (2026-09-25, 24 turns, gemma2:2b) lost
# the case number CS-20931 from the summary; a support bot that forgets the
# case number it just gave out is a real failure. Deliberately narrow --
# identifiers, not facts: redaction placeholders, #-numbers (order #4471),
# ticket/case codes (CS-20931, PAY-311), and money amounts. Semantic facts
# ("the replacement is white") are still the summary's job.
_REFERENCE_PATTERNS = [
    re.compile(r"\[\[[A-Z_]+_[0-9a-f]{8}\]\]"),                      # redaction placeholders
    re.compile(r"(?<![\w#])#\d{3,}\b"),                               # #4471
    re.compile(r"\b[A-Z]{2,6}-\d{2,}\b"),                              # CS-20931, PAY-311
    re.compile(r"(?:₹|\$|€|£|\bRs\.?\s?|\bINR\s?|\bUSD\s?)\d[\d,]*(?:\.\d+)?"),  # ₹149, $3.50, Rs 999
    re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:rupees|dollars|euros)\b", re.IGNORECASE),  # 149 rupees
]
MAX_PINNED_REFERENCES = 40


def extract_references(text: str) -> list:
    """Unique exact references in order of first appearance (see _REFERENCE_PATTERNS)."""
    found = []
    for m in sorted(
        (m for pat in _REFERENCE_PATTERNS for m in pat.finditer(text or "")), key=lambda m: m.start()
    ):
        ref = m.group(0).strip()
        if ref not in found:
            found.append(ref)
    return found


def _merge_pinned(existing: list, new: list, limit: int = MAX_PINNED_REFERENCES) -> list:
    merged = list(existing)
    for ref in new:
        if ref not in merged:
            merged.append(ref)
    return merged[-limit:] if limit > 0 else merged


# Per-message cap on what the SUMMARIZER reads (the full text still goes to
# the paid model until it's summarized). Tool output -- JSON, logs, search
# results -- dominates long histories, a 2B model can't usefully condense raw
# JSON anyway, and a shorter input is both faster and far easier for it to
# follow. The head of each message is kept (the prose usually comes first).
SUMMARY_INPUT_CHARS_PER_MESSAGE = 700


def _clip_for_summary(content: str, limit: int = SUMMARY_INPUT_CHARS_PER_MESSAGE) -> str:
    if limit <= 0 or len(content) <= limit:
        return content
    return content[:limit].rstrip() + f" ...[{len(content) - limit} more characters of tool output/data omitted]"

# Lines a small model tends to echo back from the prompt around its answer:
# the --- fences that delimit sections, markdown code fences, and the
# "Summary:" / "Updated summary:" label itself. Live testing (2026-09-24)
# found gemma2:2b wrapping its summary in the prompt's --- fences, which
# then got sent to the paid model on every later turn.
_FENCE_LINE_RE = re.compile(r"^\s*(-{3,}|`{3,}\w*|(updated\s+)?summary:)\s*$", re.IGNORECASE)


def _clean_summary(raw: str) -> str:
    """Strip echoed fence/label lines from the start and end of a model summary."""
    lines = raw.strip().splitlines()
    while lines and (not lines[0].strip() or _FENCE_LINE_RE.match(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _FENCE_LINE_RE.match(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


# Well-formed placeholder as produced by redact.py/redact_llm.py (and any
# extraction-based redaction backend using the same [[LABEL_hexdigest]]
# shape): used to find the REAL placeholders in trusted source text.
_PLACEHOLDER_STRICT_RE = re.compile(r"\[\[[A-Z]+_[0-9a-f]{8}\]\]")
# Deliberately permissive: used to scan UNTRUSTED model output, so a
# corruption that no longer matches the strict pattern is still caught
# rather than silently waved through.
_PLACEHOLDER_LOOSE_RE = re.compile(r"\[\[.*?\]\]")


def _no_corrupted_placeholders(original: str, summary: str) -> bool:
    """
    Guard rail for HistoryCompactor.summarize(): deliberately weaker than
    local_model.py's placeholders_preserved() -- compaction is explicitly
    lossy by design (see module docstring), so a summary that drops a
    placeholder entirely is fine and expected. What's rejected is a
    [[...]]-shaped span in the summary that does NOT exactly match one of
    the real placeholders from `original` -- i.e. the model altered or
    invented one rather than just omitting it.
    """
    real_placeholders = set(_PLACEHOLDER_STRICT_RE.findall(original))
    for span in set(_PLACEHOLDER_LOOSE_RE.findall(summary)):
        if span not in real_placeholders:
            return False
    return True


def _default_ollama_call(prompt: str, model: str, timeout: float) -> Optional[str]:
    t0 = time.perf_counter()
    # Derived from the prompt itself (which already embeds older_text via
    # COMPACTION_PROMPT.format()) rather than threaded through as an extra
    # parameter -- this keeps _call_model's public 3-arg signature (prompt,
    # model, timeout) unchanged, since model_call_fn is an injectable
    # interface test_tonst.py relies on with exactly that shape.
    # Capped at ~1,200 tokens: summaries are bounded by max_summary_chars
    # (4,000 chars by default) anyway, and an uncapped output budget would
    # inflate num_ctx (and RAM) for no benefit.
    max_output_tokens = min(1200, max(64, int(len(prompt.split()) * 0.8)))
    if not fits_context(prompt, max_output_tokens):
        # Ollama would silently drop the START of the prompt -- i.e. the
        # oldest turns. Refuse instead of summarizing a truncated history;
        # the caller keeps the turns verbatim and retries/drops as usual.
        logger.warning(
            "compactor: prompt (~%d tokens) exceeds the local model's context (TONST_OLLAMA_MAX_CTX=%d); "
            "skipping this summary rather than letting Ollama truncate it -- lower compaction_token_threshold",
            estimate_prompt_tokens(prompt), max_ctx(),
        )
        return None
    try:
        resp = _SESSION.post(
            DEFAULT_OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # temperature=0: same rationale as local_model.py's compress() --
                # deterministic summaries, no measured latency cost (see
                # diagnose_local_llm_perf.py, 2026-09-13).
                # num_predict: defensive cap, same reasoning as the fix applied
                # to local_model.py's compress() on 2026-09-13 (that call had
                # NO cap at all and 51/360 iterations landed within 250ms of
                # the 8s timeout as a result). Compaction hasn't shown that
                # failure mode yet (0 near-timeouts in both benchmark runs so
                # far), but a summary is supposed to be well under the
                # original's length by design (guard rail requires < 60%), so
                # there's no reason to leave generation unbounded here either.
                # num_ctx: sized to the prompt -- see ollama_util.py. Without it
                # Ollama's small default window silently cuts off the oldest turns.
                "options": {
                    "temperature": 0.0,
                    "num_predict": max_output_tokens,
                    "num_ctx": num_ctx_for(prompt, max_output_tokens),
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("compactor ollama call ok model=%s elapsed_ms=%.1f", model, elapsed_ms)
        return resp.json().get("response", "")
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "compactor ollama call TIMED OUT model=%s configured_timeout=%.1fs elapsed_ms=%.1f",
            model, timeout, elapsed_ms,
        )
        return None
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "compactor ollama call FAILED model=%s elapsed_ms=%.1f error=%s: %s",
            model, elapsed_ms, type(exc).__name__, exc,
        )
        return None


class HistoryCompactor:
    def __init__(
        self,
        model: str = "gemma2:2b",
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 8.0,
        model_call_fn: Optional[Callable[[str, str, float], Optional[str]]] = None,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self._call_model = model_call_fn or _default_ollama_call

    def is_available(self) -> bool:
        try:
            resp = _SESSION.get(self.ollama_url.replace("/api/generate", "/api/tags"), timeout=1.5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def summarize(self, older_text: str) -> Optional[str]:
        """
        Returns a summary of `older_text`, or None if the local model is
        unavailable or its output fails the guard rail below. Callers
        must treat None as "could not compact" and fall back to plain
        truncation -- never guess at a summary.
        """
        if not older_text.strip():
            return None

        raw = self._call_model(COMPACTION_PROMPT.format(text=older_text), self.model, self.timeout)
        if raw is None:
            return None

        summary = _clean_summary(raw)
        if not summary:
            return None

        # Guard rail, same dual-bound shape as LocalCompressor.compress():
        # a "summary" that isn't meaningfully shorter than the source
        # didn't do its job (and may mean the model echoed/continued
        # rather than summarized); a suspiciously tiny one usually means
        # a misfire rather than genuine compaction.
        if len(summary) >= len(older_text) * 0.6:
            return None
        if len(summary) < 20:
            return None
        if not _no_corrupted_placeholders(older_text, summary):
            return None

        return summary

    def summarize_incremental(
        self, previous_summary: Optional[str], new_text: str, max_summary_chars: int = 4000
    ) -> Optional[str]:
        """
        Folds `new_text` (newly evicted turns) into `previous_summary`.
        Returns None on any failure -- same contract as summarize().

        The FIRST fold (no previous summary yet) uses the same structured
        prompt as every later fold, with "(none yet)" as the existing
        summary -- so the summary has the same Goal / Decisions / Key
        facts / Open items shape from the start. (An earlier version fell
        back to summarize()'s free-form prompt here, so the first summary
        came out as prose and only later ones had headings -- caught in
        live testing, 2026-09-24.)

        Guard rails:
          - the summary may grow by less than 60% of the new text's
            length (the same "must actually condense" ratio summarize()
            applies, measured on what's being added -- measuring against
            the whole source would wrongly reject a mature summary
            absorbing a short new batch), and must be at least 20 chars;
          - output must not exceed max_summary_chars, so a rolling
            summary can't grow without bound over a long session;
          - it must use the Goal / Decisions / Key facts / Open items
            headings (at least 3 of 4) -- rejects a model that replied to
            the conversation instead of summarizing it (live failure,
            2026-09-25);
          - no altered/invented placeholders, checked against BOTH the
            previous summary and the new turns (a placeholder that only
            ever lived in the old summary is still legitimate).
        """
        if not new_text.strip():
            return None
        previous_summary = previous_summary or ""

        prompt = ROLLING_COMPACTION_PROMPT.format(summary=previous_summary or "(none yet)", text=new_text)
        raw = self._call_model(prompt, self.model, self.timeout)
        if raw is None:
            return None
        summary = _clean_summary(raw)
        source = previous_summary + "\n" + new_text
        if not summary or len(summary) < 20:
            return None
        if len(summary) >= len(previous_summary) + len(new_text) * 0.6 or len(summary) > max_summary_chars:
            return None
        if not _has_summary_structure(summary):
            # e.g. the model replied to the conversation instead of summarizing it
            logger.warning("compactor: local model output is not a structured summary; rejected")
            return None
        if not _has_summary_content(summary):
            logger.warning("compactor: summary has the headings but no real Key facts; rejected")
            return None
        if not _no_corrupted_placeholders(source, summary):
            return None
        return summary


@dataclass
class CompactionResult:
    messages: list  # list[dict] -- system messages (if kept) + [optional summary message] + recent verbatim turns
    compacted: bool  # True only if a real summary was produced and used
    dropped_turns: int  # turns removed from the verbatim history (summarized OR discarded, either way)
    older_tokens_estimate: int  # estimated size of what was removed, before compaction
    summary_tokens_estimate: int = 0  # 0 whenever compacted is False


def compact_history(
    messages: list,
    compactor: Optional[HistoryCompactor],
    keep_last_n: int = 6,
    keep_system: bool = True,
    token_threshold: int = 3000,
    token_estimator=None,
) -> CompactionResult:
    """
    Always keeps the system message(s) (if keep_system) and the last
    keep_last_n turns verbatim -- identical selection to
    trim.truncate_history(). The difference: instead of silently
    discarding everything before that window, if the discarded portion
    is large enough to be worth summarizing (>= token_threshold,
    estimated) and a HistoryCompactor is available, it's condensed into
    one summary message inserted between the system message(s) and the
    verbatim recent turns -- so older context degrades into a summary
    instead of vanishing outright.

    Falls back to plain truncation (older turns simply dropped, no
    summary) when: no compactor was given, the older portion doesn't
    meet token_threshold (not worth the summarization cost/latency), the
    local model is unavailable, or its output fails the guard rail. This
    is never worse than trim.truncate_history()'s existing behavior --
    just not better, in those cases.

    Callers are expected to have already redacted `messages` (see
    TonstClient.query_messages()) -- this function has no idea what PII
    looks like and will happily hand raw content to the local model if
    it's given raw content. Redact first, always.
    """
    if token_estimator is None:
        from .trim import estimate_tokens as token_estimator

    if not messages:
        return CompactionResult(messages=messages, compacted=False, dropped_turns=0, older_tokens_estimate=0)

    system_msgs = [m for m in messages if m.get("role") == "system"] if keep_system else []
    other_msgs = [m for m in messages if m.get("role") != "system"]

    if len(other_msgs) <= keep_last_n:
        return CompactionResult(
            messages=system_msgs + other_msgs, compacted=False, dropped_turns=0, older_tokens_estimate=0
        )

    older = other_msgs[:-keep_last_n] if keep_last_n > 0 else other_msgs
    recent = other_msgs[-keep_last_n:] if keep_last_n > 0 else []

    older_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in older)
    older_tokens = token_estimator(older_text) if older_text.strip() else 0

    if compactor is None or older_tokens < token_threshold:
        return CompactionResult(
            messages=system_msgs + recent,
            compacted=False,
            dropped_turns=len(older),
            older_tokens_estimate=older_tokens,
        )

    summary = compactor.summarize(older_text)
    if summary is None:
        return CompactionResult(
            messages=system_msgs + recent,
            compacted=False,
            dropped_turns=len(older),
            older_tokens_estimate=older_tokens,
        )

    summary_message = {"role": "user", "content": f"[Summary of earlier conversation]\n{summary}"}
    return CompactionResult(
        messages=system_msgs + [summary_message] + recent,
        compacted=True,
        dropped_turns=len(older),
        older_tokens_estimate=older_tokens,
        summary_tokens_estimate=token_estimator(summary),
    )


# ---------------------------------------------------------------------
# Rolling (incremental) compaction
# ---------------------------------------------------------------------
#
# compact_history() above is stateless: every call re-summarizes ALL the
# older turns from scratch. In a live chat that has two costs:
#   1. a local-model call on every single turn once past the threshold;
#   2. the summary text comes out different every turn, so it can never
#      sit in a provider's cached prompt prefix -- and neither can
#      anything after it.
#
# compact_history_rolling() keeps a small caller-held state instead:
#   - turns that have been folded into the summary are never re-read;
#   - turns that fall out of the recent window are kept VERBATIM
#     ("pending") until they add up to token_threshold, and only then
#     folded into the summary in one local-model call;
#   - between folds, the prompt is [system][summary][pending...][recent...]
#     -- an append-only sequence, so the provider's prefix cache keeps
#     hitting on everything up to the newest turn.
# Token use is still bounded: summary + (< token_threshold of pending) +
# the recent window.

import hashlib  # noqa: E402  (kept next to the code that uses it)
import json as _json  # noqa: E402
import threading  # noqa: E402


def _fingerprint(messages: list) -> Optional[str]:
    if not messages:
        return None
    payload = _json.dumps(
        [(m.get("role"), m.get("content")) for m in messages], separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RollingSummary:
    """
    Caller-held state for compact_history_rolling(). Keep one per
    conversation and pass it back in on every call; it's updated in
    place. to_dict()/from_dict() make it easy to persist (a web server
    storing it next to the conversation, say). Holds only already-
    redacted text -- the summary is built from placeholder'd messages.
    """
    summary: Optional[str] = None
    summarized_count: int = 0          # non-system messages already folded in (or dropped)
    fingerprint: Optional[str] = None  # hash of exactly those messages, to detect edited history
    failed_folds: int = 0              # consecutive failed fold attempts for the CURRENT pending batch
    dropped_tokens: int = 0            # est. tokens of turns dropped WITHOUT being summarized (context lost)
    pinned: list = field(default_factory=list)  # exact references carried verbatim (see extract_references)
    # Observed provider cache hit rate (see observe_cache_usage); None until
    # there are CACHE_HIT_MIN_OBSERVATIONS responses to go on.
    cache_hit_rate: Optional[float] = None
    cache_observations: int = 0
    # Runtime-only (not persisted, not compared): background summarizing.
    fold_in_progress: bool = field(default=False, compare=False, repr=False)
    # Bumped on every reset, so a background job planned against an earlier
    # conversation can never be applied to a new one -- start_count and
    # previous_summary alone can't tell them apart (both start at 0 / None).
    generation: int = field(default=0, compare=False, repr=False)
    # What the in-flight background job covers, so an edit to those messages
    # (or a different conversation passed in) invalidates it.
    pending_fold_end: int = field(default=0, compare=False, repr=False)
    pending_fold_fingerprint: Optional[str] = field(default=None, compare=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    # Runtime-only bookkeeping for observe_cache_usage().
    _hit_ema: Optional[float] = field(default=None, compare=False, repr=False)
    _last_prompt_tokens: int = field(default=0, compare=False, repr=False)
    _prefix_changed: bool = field(default=False, compare=False, repr=False)
    _last_header: Optional[str] = field(default=None, compare=False, repr=False)

    def observe_cache_usage(self, prompt_tokens: int, cached_tokens: int) -> None:
        """
        Feed the provider's usage for each response back in, so cache-aware
        compaction works from the cache hit rate you're actually getting
        instead of assuming every repeated prefix is cached.
          Anthropic: prompt = input_tokens + cache_creation_input_tokens
                     + cache_read_input_tokens; cached = cache_read_input_tokens
          Gemini:    prompt = promptTokenCount; cached = cachedContentTokenCount
          OpenAI:    prompt = prompt_tokens; cached = prompt_tokens_details.cached_tokens
        The hit rate is cached tokens / the PREVIOUS request's prompt size,
        i.e. the share of what could have been reused that actually was.
        Turns right after the summary changed are skipped: a miss there is
        expected, not a sign the cache is unreliable.
        Why: a live Gemini run (2026-09-25) served 0% of a growing chat from
        its implicit cache until the prompt passed ~16k tokens, 35% overall.
        Assuming 0.1x reads there made cache-aware mode hold back summaries
        that would have paid off at once (+4.5% cost vs. -15% without it).
        """
        if prompt_tokens <= 0:
            return
        with self._lock:
            prev, changed = self._last_prompt_tokens, self._prefix_changed
            self._last_prompt_tokens, self._prefix_changed = int(prompt_tokens), False
            if not prev or changed:
                return
            rate = max(0.0, min(1.0, cached_tokens / prev))
            if self._hit_ema is None:
                self._hit_ema = rate
            else:
                self._hit_ema = CACHE_HIT_EMA_ALPHA * rate + (1 - CACHE_HIT_EMA_ALPHA) * self._hit_ema
            self.cache_observations += 1
            if self.cache_observations >= CACHE_HIT_MIN_OBSERVATIONS:
                self.cache_hit_rate = round(self._hit_ema, 4)

    def __post_init__(self):
        if self.cache_hit_rate is not None:
            self._hit_ema = float(self.cache_hit_rate)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "summarized_count": self.summarized_count,
            "fingerprint": self.fingerprint,
            "failed_folds": self.failed_folds,
            "dropped_tokens": self.dropped_tokens,
            "pinned": list(self.pinned),
            "cache_hit_rate": self.cache_hit_rate,
            "cache_observations": self.cache_observations,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RollingSummary":
        return cls(
            summary=d.get("summary"),
            summarized_count=int(d.get("summarized_count", 0)),
            fingerprint=d.get("fingerprint"),
            failed_folds=int(d.get("failed_folds", 0)),
            dropped_tokens=int(d.get("dropped_tokens", 0)),
            pinned=list(d.get("pinned") or []),
            cache_hit_rate=d.get("cache_hit_rate"),
            cache_observations=int(d.get("cache_observations", 0)),
        )


@dataclass
class RollingCompactionResult:
    messages: list
    summary_updated: bool = False   # a fold ran and succeeded this call
    summary_reused: bool = False    # an existing summary was sent unchanged (no local-model call)
    folded_turns: int = 0           # turns folded into the summary this call
    dropped_turns: int = 0          # turns dropped WITHOUT being summarized this call (fold failed / no compactor)
    pending_turns: int = 0          # out-of-window turns still kept verbatim, awaiting the next fold
    state_reset: bool = False       # history didn't match the state (edited/new conversation); rebuilt from scratch
    fold_failed: bool = False       # a fold was attempted this call and failed
    fold_will_retry: bool = False   # ...and its turns were kept verbatim for another attempt instead of dropped
    tokens_lost: int = 0            # est. tokens of this conversation's history dropped without a summary, so far
    # cache_aware=True only: a fold was due by size but was postponed because
    # rebuilding the provider's prompt cache would cost more than it saves
    # within the turns the conversation is expected to have left.
    fold_postponed_for_cache: bool = False
    fold_payback_turns: Optional[float] = None  # estimated turns for a fold to pay for itself (cache_aware only)
    # defer_fold=True only: a fold is due and has been handed back as a job
    # to run off the request path (run_fold_job); its turns stay verbatim
    # in `messages` until the job finishes.
    fold_job: Optional["FoldJob"] = None


@dataclass
class FoldJob:
    """
    One pending fold, captured at planning time so it can run later on
    another thread (see compact_history_rolling(defer_fold=True) and
    run_fold_job()). Contains only already-redacted text.
    """
    previous_summary: Optional[str]
    text: str
    tokens: int
    turns: int
    start_count: int
    end_count: int
    fingerprint_after: Optional[str]
    max_summary_chars: int
    max_fold_retries: int
    generation: int = 0
    references: list = field(default_factory=list)


def _apply_fold_outcome(state: "RollingSummary", job: FoldJob, new_summary: Optional[str],
                        have_compactor: bool, result: Optional["RollingCompactionResult"] = None) -> str:
    """
    Applies a fold's result to `state` (caller holds state._lock). Returns
    "folded", "retry" or "dropped". Shared by the blocking and background
    paths so both follow exactly the same retry-before-drop rules.
    """
    if new_summary is not None:
        state.summary = new_summary
        state.failed_folds = 0
        outcome = "folded"
    elif have_compactor and state.failed_folds < job.max_fold_retries:
        state.failed_folds += 1  # keep the turns verbatim; the next attempt needs more pending text
        return "retry"
    else:
        state.dropped_tokens += job.tokens
        state.failed_folds = 0
        outcome = "dropped"
    # Folded OR dropped, the batch's exact references survive either way.
    state.pinned = _merge_pinned(state.pinned, job.references)
    state.summarized_count = job.end_count
    state.fingerprint = job.fingerprint_after
    return outcome


def run_fold_job(job: FoldJob, compactor: Optional["HistoryCompactor"], state: "RollingSummary") -> str:
    """
    Runs a deferred fold (the slow local-model call) and applies it to
    `state`. Safe to call from a background thread: the model call runs
    WITHOUT holding the state lock, and the result is only applied if the
    state hasn't moved on in the meantime (otherwise "stale" -- e.g. the
    conversation was edited and the state reset). Always clears
    state.fold_in_progress. Returns "folded", "retry", "dropped" or "stale".
    """
    try:
        new_summary = None
        if compactor is not None:
            try:
                new_summary = compactor.summarize_incremental(
                    job.previous_summary, job.text, max_summary_chars=job.max_summary_chars
                )
            except Exception:  # noqa: BLE001 -- a background failure must never crash the app
                logger.warning("tonst background compaction failed", exc_info=True)
                new_summary = None
        with state._lock:
            if (
                state.generation != job.generation
                or state.summarized_count != job.start_count
                or state.summary != job.previous_summary
            ):
                return "stale"
            return _apply_fold_outcome(state, job, new_summary, compactor is not None)
    finally:
        with state._lock:
            state.fold_in_progress = False


def _summary_message_text(state: "RollingSummary") -> Optional[str]:
    """The message standing in for summarized/dropped turns: summary + pinned references."""
    parts = []
    if state.summary:
        parts.append(f"[Summary of earlier conversation]\n{state.summary}")
    if state.pinned:
        label = "Pinned references from earlier turns (exact)" if state.summary else \
            "[Earlier turns were trimmed] Pinned references from them (exact)"
        parts.append(f"{label}: {', '.join(state.pinned)}")
    return "\n".join(parts) if parts else None


# Anthropic prompt-caching price multipliers, relative to the base input
# price: writing a prefix into the cache costs 1.25x, reading it 0.1x.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10
# (write multiplier, read multiplier) per provider, relative to base input
# price. Gemini's implicit caching has no write surcharge and bills cached
# reads at 10% of input (Google pricing page, 2026-09-25). Pass a tuple
# for anything else.
CACHE_PRICING = {
    "anthropic": (CACHE_WRITE_MULT, CACHE_READ_MULT),
    "gemini": (1.0, 0.10),
}


def _cache_multipliers(cache_pricing) -> tuple:
    if isinstance(cache_pricing, str):
        if cache_pricing not in CACHE_PRICING:
            raise ValueError(f"unknown cache_pricing {cache_pricing!r}; use one of {sorted(CACHE_PRICING)} "
                             "or a (write_mult, read_mult) tuple")
        return CACHE_PRICING[cache_pricing]
    write, read = cache_pricing
    return float(write), float(read)
# A rolling summary is typically ~10% the size of what it folds in (live
# test: Claude Haiku folded ~7.8k input tokens into summaries of ~230-450
# tokens). Used only to ESTIMATE a fold's payback before it runs.
EST_SUMMARY_GROWTH = 0.10
# cache_aware without expected_remaining_turns: assume a conversation that
# has run N user turns has about N x this factor still to go. 0.5 was the
# best of 0.5/0.75/1.0 in an offline simulation of the live test's
# conversation (10-100 turns, Haiku summaries, Sonnet prices): it avoided
# the folds that never pay back in short chats (10-16 turns: 0% to +7%
# vs. +8% to +15% for folding at the plain threshold) and matched or beat
# the plain threshold from 40 turns on.
REMAINING_TURNS_FACTOR = 0.5
# observe_cache_usage(): smoothing of the observed hit rate, and how many
# usable observations before it replaces the assumption of full caching.
CACHE_HIT_EMA_ALPHA = 0.3
CACHE_HIT_MIN_OBSERVATIONS = 2
# Output tokens cost 5x input tokens on current Claude models.
SUMMARIZER_OUTPUT_PRICE_MULT = 5.0


def estimate_fold_payback(evict_tokens: int, kept_tokens: int, summary_tokens: int, summarizer_input_tokens: int = 0,
                          summarizer_price_ratio: float = 0.0, max_summary_tokens: Optional[int] = None,
                          cache_pricing="anthropic", cache_hit_rate: float = 1.0) -> float:
    """
    With prompt caching on, how many later turns a fold needs before it has
    paid for itself. All token counts should come from the same estimator;
    the result is a ratio, so a consistent under/over-estimate cancels out.

    - Without the fold, the next turn re-reads (summary + evicted + kept)
      from cache at 0.1x.
    - With it, the summary changes, so everything after the system prompt,
      i.e. (new summary + kept turns), is written to cache again at 1.25x
      instead of read at 0.1x: a one-off cost of 1.15 x (new summary + kept).
    - Plus the summarizer call itself, if it isn't free (a local model is):
      summarizer_price_ratio = its input price / the main model's input
      price (Claude Haiku 4.5 vs Sonnet 4.6: 1/3); output priced 5x input.
    - Every later turn then reads (evicted + old summary - new summary)
      fewer tokens at 0.1x: that's the per-turn saving.
    cache_pricing: "anthropic" (write 1.25x, read 0.1x -- the numbers
    above), "gemini" (implicit caching: no write surcharge, read 0.1x, so
    the one-off cost is 0.9x instead of 1.15x), or a (write, read) tuple.
    cache_hit_rate (default 1.0 = the repeated prefix is always cached):
    the share of the repeated prefix the provider actually serves from
    cache (RollingSummary.observe_cache_usage). A miss bills the prefix at
    the write price, so re-reading costs h x read + (1 - h) x write per
    token. At h = 0 there's nothing to lose: the fold pays back at once.
    Returns float('inf') if the fold would never pay back.
    """
    write_mult, read_mult = _cache_multipliers(cache_pricing)
    h = max(0.0, min(1.0, cache_hit_rate))
    read_mult = h * read_mult + (1 - h) * write_mult
    new_summary = summary_tokens + EST_SUMMARY_GROWTH * evict_tokens
    if max_summary_tokens:
        new_summary = min(new_summary, max(max_summary_tokens, summary_tokens))
    one_off = (write_mult - read_mult) * (new_summary + kept_tokens)
    if summarizer_price_ratio > 0:
        one_off += summarizer_price_ratio * (summarizer_input_tokens + SUMMARIZER_OUTPUT_PRICE_MULT * new_summary)
    per_turn = read_mult * (evict_tokens + summary_tokens - new_summary)
    if per_turn <= 0:
        return float("inf")
    return one_off / per_turn


def compact_history_rolling(
    messages: list,
    compactor: Optional[HistoryCompactor],
    state: RollingSummary,
    keep_last_n: int = 6,
    keep_system: bool = True,
    token_threshold: int = 3000,
    token_estimator=None,
    max_summary_chars: int = 4000,
    max_fold_retries: int = 1,
    defer_fold: bool = False,
    summary_input_chars: int = SUMMARY_INPUT_CHARS_PER_MESSAGE,
    cache_aware: bool = False,
    expected_remaining_turns: Optional[int] = None,
    summarizer_price_ratio: float = 0.0,
    max_history_tokens: Optional[int] = None,
    cache_pricing="anthropic",
) -> RollingCompactionResult:
    """
    Incremental, cache-friendly history compaction. `messages` is the
    FULL (already redacted) conversation so far, as with compact_history();
    `state` is updated in place.

    defer_fold (default False): when a fold is due, don't run the local
    model here. Instead return it as result.fold_job, keep the turns
    verbatim in this call's messages, and let the caller run it off the
    request path with run_fold_job() -- in parallel with the API call, or
    after it. The summary is then used from the next call on. This takes
    the local model out of the user's wait entirely: live testing on a
    MacBook Air measured a blocking fold adding ~4.7 s to that turn's
    response time. While a job is in flight no second one is scheduled
    (state.fold_in_progress). TonstClient.query_messages(...,
    background_summary=True) does all of this for you.

    summary_input_chars (default 700): each message is clipped to this
    many characters in what the SUMMARIZER reads (tool output, JSON and
    logs are cut, with a marker). The fold trigger and lost-token
    accounting still use full sizes, and the paid model keeps seeing the
    full text until it's summarized. 0 disables clipping.

    Failure behavior -- retry before dropping. If a fold fails (local
    model timed out or down, guard rail rejected the output), the turns
    are NOT dropped straight away: they stay in the prompt verbatim, and
    the fold is retried once the pending batch has grown by another
    token_threshold (so a struggling local model isn't hit on every
    single turn). Only after max_fold_retries further failures is the
    batch dropped, keeping the previous summary -- never worse than plain
    truncation, and pending text stays bounded at roughly
    (max_fold_retries + 1) x token_threshold. With no compactor at all
    there is nothing to retry, so the batch is dropped at the threshold.
    (The first version dropped on the first failure; live testing on a
    MacBook Air showed one cold-start timeout was enough to lose 14
    turns for good.)

    If the messages no longer match what `state` says was already
    summarized (the caller edited history, or passed a different
    conversation), the state is reset and rebuilt from scratch rather
    than silently producing a summary of the wrong conversation.

    cache_aware (default False): set this when the provider caches the
    prompt prefix (Anthropic cache_control, OpenAI automatic caching...).
    With caching, old history is already cheap -- re-read at 0.1x -- while
    every fold changes the summary and forces the rest of the prompt to be
    re-written to cache at 1.25x. A live 24-turn test (Sonnet 4.6) found
    folding at the plain token_threshold roughly break-even: one fold came
    3 turns before the end and never paid back. With cache_aware, a fold
    that is due by size only runs once estimate_fold_payback() says it
    pays for itself within the turns the conversation still has left:
    `expected_remaining_turns` if you know it (a scripted flow, an agent
    with a step budget), otherwise an estimate of half the user turns so
    far (REMAINING_TURNS_FACTOR). Short chats then skip folds that would
    only cost money; long ones fold as before. `summarizer_price_ratio` adds a paid summarizer's cost
    to the estimate (Claude Haiku 4.5 summarizing for Sonnet 4.6: 1/3;
    a local model: 0). `cache_pricing` sets the provider's cache prices:
    "anthropic" (default), "gemini", or a (write_mult, read_mult) tuple.

    max_history_tokens (optional): a hard cap. If the prompt would exceed
    it, a due fold runs regardless of payback -- staying inside the
    context window beats a small cache saving.
    """
    if token_estimator is None:
        from .trim import estimate_tokens as token_estimator

    with state._lock:
        return _compact_rolling_locked(
            messages, compactor, state, keep_last_n, keep_system, token_threshold, token_estimator,
            max_summary_chars, max_fold_retries, defer_fold, summary_input_chars,
            cache_aware, expected_remaining_turns, summarizer_price_ratio, max_history_tokens, cache_pricing,
        )


def _compact_rolling_locked(messages, compactor, state, keep_last_n, keep_system, token_threshold,
                            token_estimator, max_summary_chars, max_fold_retries, defer_fold,
                            summary_input_chars=SUMMARY_INPUT_CHARS_PER_MESSAGE, cache_aware=False,
                            expected_remaining_turns=None, summarizer_price_ratio=0.0,
                            max_history_tokens=None, cache_pricing="anthropic"):
    system_msgs = [m for m in messages if m.get("role") == "system"] if keep_system else []
    other = [m for m in messages if m.get("role") != "system"]
    result = RollingCompactionResult(messages=[])

    count = state.summarized_count
    if count > len(other) or _fingerprint(other[:count]) != state.fingerprint:
        if count or state.summary:
            result.state_reset = True
        state.summary, state.summarized_count, state.fingerprint = None, 0, None
        state.failed_folds, state.dropped_tokens = 0, 0
        state.pinned = []
        state.generation += 1
        count = 0

    # A background job is in flight: if the messages it's summarizing have
    # changed since it was planned, invalidate it (it will come back "stale").
    if state.fold_in_progress and (
        len(other) < state.pending_fold_end
        or _fingerprint(other[: state.pending_fold_end]) != state.pending_fold_fingerprint
    ):
        state.generation += 1
        state.pending_fold_end, state.pending_fold_fingerprint = 0, None
        result.state_reset = True

    unsummarized = other[count:]
    evictable = unsummarized[:-keep_last_n] if keep_last_n > 0 else list(unsummarized)
    if len(unsummarized) <= keep_last_n:
        evictable = []

    evict_full = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in evictable)
    evict_tokens = token_estimator(evict_full) if evict_full.strip() else 0
    # What the summarizer actually reads: long messages clipped (see _clip_for_summary).
    evict_text = "\n".join(
        f"{m.get('role', 'user')}: {_clip_for_summary(str(m.get('content', '')), summary_input_chars)}"
        for m in evictable
    )

    # Each failed attempt raises the bar for the next one by another
    # threshold's worth of pending text.
    fold_at = token_threshold * (1 + state.failed_folds)
    fold_due = bool(evictable) and evict_tokens >= fold_at
    # Only estimate payback when there's something a fold would actually do.
    if fold_due and cache_aware and compactor is not None and not state.fold_in_progress:
        header_now = _summary_message_text(state)
        summary_tokens = token_estimator(header_now) if header_now else 0
        kept = unsummarized[len(evictable):]
        kept_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in kept)
        kept_tokens = token_estimator(kept_text) if kept_text.strip() else 0
        payback = estimate_fold_payback(
            evict_tokens, kept_tokens, summary_tokens,
            summarizer_input_tokens=token_estimator(evict_text) + summary_tokens,
            summarizer_price_ratio=summarizer_price_ratio,
            max_summary_tokens=max_summary_chars // 4,
            cache_pricing=cache_pricing,
            cache_hit_rate=1.0 if state.cache_hit_rate is None else state.cache_hit_rate,
        )
        result.fold_payback_turns = round(payback, 1) if payback != float("inf") else None
        if expected_remaining_turns is not None:
            allowed = expected_remaining_turns
        else:
            # Unknown horizon: assume a conversation that has lasted N user
            # turns has about N/2 more to go (see REMAINING_TURNS_FACTOR).
            turns_so_far = sum(1 for m in other if m.get("role") == "user")
            allowed = REMAINING_TURNS_FACTOR * turns_so_far
        system_text = "\n".join(str(m.get("content", "")) for m in system_msgs)
        prompt_tokens = (token_estimator(system_text) if system_text.strip() else 0) + summary_tokens \
            + evict_tokens + kept_tokens
        over_cap = max_history_tokens is not None and prompt_tokens > max_history_tokens
        if payback > allowed and not over_cap:
            fold_due = False
            result.fold_postponed_for_cache = True
    if fold_due and defer_fold and compactor is not None:
        # Hand the fold back as a job; turns stay verbatim for now.
        if not state.fold_in_progress:
            state.fold_in_progress = True
            result.fold_job = FoldJob(
                previous_summary=state.summary, text=evict_text, tokens=evict_tokens, turns=len(evictable),
                start_count=count, end_count=count + len(evictable),
                fingerprint_after=_fingerprint(other[: count + len(evictable)]),
                max_summary_chars=max_summary_chars, max_fold_retries=max_fold_retries,
                generation=state.generation, references=extract_references(evict_text),
            )
            state.pending_fold_end = result.fold_job.end_count
            state.pending_fold_fingerprint = result.fold_job.fingerprint_after
        result.pending_turns = len(evictable)
        result.summary_reused = state.summary is not None
    elif fold_due:
        job = FoldJob(
            previous_summary=state.summary, text=evict_text, tokens=evict_tokens, turns=len(evictable),
            start_count=count, end_count=count + len(evictable),
            fingerprint_after=_fingerprint(other[: count + len(evictable)]),
            max_summary_chars=max_summary_chars, max_fold_retries=max_fold_retries,
            references=extract_references(evict_text),
        )
        new_summary = (
            compactor.summarize_incremental(state.summary, evict_text, max_summary_chars=max_summary_chars)
            if compactor is not None
            else None
        )
        outcome = _apply_fold_outcome(state, job, new_summary, compactor is not None)
        if outcome == "folded":
            result.summary_updated = True
            result.folded_turns = len(evictable)
        elif outcome == "retry":
            result.fold_failed = True
            result.fold_will_retry = True
            result.pending_turns = len(evictable)
            result.summary_reused = state.summary is not None
        else:
            result.fold_failed = compactor is not None
            result.dropped_turns = len(evictable)
        unsummarized = other[state.summarized_count:]
    else:
        result.pending_turns = len(evictable)
        result.summary_reused = state.summary is not None
    result.tokens_lost = state.dropped_tokens

    out = list(system_msgs)
    header = _summary_message_text(state)
    if header != state._last_header:
        state._prefix_changed = state._last_header is not None or header is not None
        state._last_header = header
    if header:
        out.append({"role": "user", "content": header})
    out.extend(unsummarized)
    result.messages = out
    return result
