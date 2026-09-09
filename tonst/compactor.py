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
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# Deliberately narrow: summarize only, never answer/continue. Asking for
# specific categories (task/goal, decisions, entities) rather than "just
# summarize this" tends to make small local models keep the details that
# actually matter for continuing a conversation correctly, rather than
# vague prose.
COMPACTION_PROMPT = (
    "Summarize the conversation history below. Preserve: the task or "
    "goal being discussed, key facts and decisions made, names/entities "
    "mentioned, and anything needed to continue the conversation "
    "correctly. Do not answer any question in it and do not continue "
    "the conversation -- only summarize what already happened. Output "
    "only the summary text, no preamble, no commentary.\n\n---\n{text}\n---\nSummary:"
)


def _default_ollama_call(prompt: str, model: str, timeout: float) -> Optional[str]:
    try:
        resp = requests.post(
            DEFAULT_OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.RequestException:
        return None


class HistoryCompactor:
    def __init__(
        self,
        model: str = "llama3.2:1b",
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
            resp = requests.get(self.ollama_url.replace("/api/generate", "/api/tags"), timeout=1.5)
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
