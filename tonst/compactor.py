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
    try:
        resp = _SESSION.post(
            DEFAULT_OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
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

        summary = raw.strip()
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
