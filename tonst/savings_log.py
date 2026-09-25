"""
savings_log.py
--------------
Opt-in, local, append-only log of what tonst did on each call -- tokens
in vs. tokens sent, how many PII fields were redacted (by TYPE only),
which optional steps ran, and how long they took -- plus a summarizer
and a `tonst stats` CLI on top of it.

Why this exists: every call already returns an OptimizationReport, but
a report that's printed once and thrown away can't answer "how much has
tonst saved us this month?" This persists exactly those numbers, locally,
in a plain JSONL file you own. Nothing is sent anywhere.

What is NEVER written to the log, by design:
  - Prompt or response text.
  - PII values.
  - Placeholder hashes. A placeholder like [[EMAIL_1a2b3c4d]] is a
    deterministic hash of the original value (that determinism is what
    makes prompt caching work -- see redact.py), which means a log full
    of them could be brute-forced back to real emails/phone numbers
    with a dictionary attack. Only the LABEL ("EMAIL") is counted.

Honesty about the numbers:
  - Token counts come from tonst's chars/4 estimate (trim.py), not the
    provider's tokenizer. Every entry says so (token_counts_are_estimates).
  - The dollar figure is only written if you pass input_price_per_million,
    and it is named estimated_cost_saved_usd for a reason: it's tokens
    saved x the price you gave, not a number read off a bill.
  - Provider-side prompt caching does not reduce tokens at all (it
    discounts them), so it never shows up in tokens_saved. If you have a
    real provider usage report (providers/*.parse_*_usage()), pass it as
    `usage=` and its real cache-read/cache-write token counts are
    recorded alongside, clearly separated from tonst's own estimates.

Scope note (open-core boundary): this is a developer-facing savings log.
It is deliberately NOT an audit/compliance record -- it has no
tamper-evidence, no retention policy, no signing, and it is written
best-effort (a failed write never fails your API call). Don't present
it to a compliance reviewer as proof of redaction.
"""

from __future__ import annotations
import json
import logging
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = os.path.join(os.path.expanduser("~"), ".tonst", "savings.jsonl")
SCHEMA_VERSION = 1

# [[LABEL_hexdigest]] -> LABEL. Labels may themselves contain underscores
# (e.g. CREDIT_CARD), so split on the LAST underscore only.
_PLACEHOLDER_RE = re.compile(r"^\[\[([A-Z0-9_]+)_[0-9a-f]+\]\]$")


def redacted_types_from_mapping(mapping: dict) -> dict:
    """
    {placeholder: original} -> {"EMAIL": 2, "NAME": 1, ...}. Reads only
    the placeholder's label; never touches the original values or the
    hash part.
    """
    counts: Counter = Counter()
    for placeholder in mapping or {}:
        m = _PLACEHOLDER_RE.match(str(placeholder))
        counts[m.group(1) if m else "OTHER"] += 1
    return dict(sorted(counts.items()))


class SavingsLog:
    """
    Append-only JSONL writer. Thread-safe within one process (one lock
    per instance); separate processes appending to the same file is
    fine on POSIX for lines this small, but not guaranteed everywhere --
    give each process its own path if that matters to you.

    Writes are best-effort: any OSError is logged at WARNING and
    swallowed, so a full disk or a read-only home directory never breaks
    the call it was trying to record.
    """

    def __init__(self, path: Optional[str] = None, app: Optional[str] = None):
        self.path = os.path.expanduser(path or os.environ.get("TONST_SAVINGS_LOG") or DEFAULT_LOG_PATH)
        self.app = app
        self._lock = threading.Lock()

    def record(
        self,
        report,
        *,
        method: str = "query",
        app: Optional[str] = None,
        tags: Optional[dict] = None,
        model: Optional[str] = None,
        input_price_per_million: Optional[float] = None,
        usage=None,
    ) -> Optional[dict]:
        """
        Appends one entry built from an OptimizationReport. Returns the
        entry dict that was written (or None if the write failed).

        tags: small free-form labels for your own attribution
            ({"team": "support", "feature": "ticket-triage"}). Don't put
            user data in here -- it's written verbatim.
        usage: optional CacheUsageReport from a real provider response.
        """
        entry = {
            "v": SCHEMA_VERSION,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app": app or self.app,
            "method": method,
            "model": model,
            "tags": tags or None,
            "original_tokens": report.original_tokens,
            "sent_tokens": report.sent_tokens,
            "tokens_saved": report.tokens_saved,
            "percent_saved": report.percent_saved,
            "token_counts_are_estimates": not getattr(report, "token_counts_exact", False),
            "redacted_fields": report.redacted_fields,
            "redacted_types": dict(getattr(report, "redacted_types", {}) or {}),
            "used_enhanced_redaction": report.used_enhanced_redaction,
            "locally_compressed": report.locally_compressed,
            "history_compacted": report.history_compacted,
            "history_turns_dropped": report.history_turns_dropped,
            "history_summary_updated": getattr(report, "history_summary_updated", False),
            "history_summary_reused": getattr(report, "history_summary_reused", False),
            "history_summary_scheduled": getattr(report, "history_summary_scheduled", False),
            "history_fold_failed": getattr(report, "history_fold_failed", False),
            "history_tokens_lost": getattr(report, "history_tokens_lost", 0),
            "history_fold_postponed": getattr(report, "history_fold_postponed", False),
            "chunks_in": getattr(report, "chunks_in", 0),
            "chunks_sent": getattr(report, "chunks_sent", 0),
            "chunk_filter_skipped": getattr(report, "chunk_filter_skipped", False),
            "local_overhead_ms": round(report.local_overhead_ms, 1),
            "call_ms": round(report.call_ms, 1),
            "total_ms": round(report.total_ms, 1),
        }
        if input_price_per_million is not None:
            entry["input_price_per_million"] = input_price_per_million
            entry["estimated_cost_saved_usd"] = round(
                report.tokens_saved / 1_000_000 * input_price_per_million, 8
            )
        pp = getattr(report, "provider_prompt_tokens", None)
        if usage is None and pp is not None:
            # From TonstClient(messages_fn=...) returning (text, usage): real prompt/cached counts.
            entry["provider_usage"] = {
                "input_tokens": pp, "output_tokens": 0,
                "cache_read_input_tokens": getattr(report, "provider_cached_tokens", 0) or 0,
                "cache_creation_input_tokens": 0,
            }
        if usage is not None:
            entry["provider_usage"] = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            }

        line = json.dumps(entry, separators=(",", ":"), sort_keys=True)
        try:
            with self._lock:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError as e:
            logger.warning("tonst savings log: could not write to %s (%s); continuing", self.path, e)
            return None
        return entry


def read_entries(path: Optional[str] = None) -> Iterable[dict]:
    """Yields parsed entries, skipping (and counting nothing for) corrupt lines."""
    path = os.path.expanduser(path or os.environ.get("TONST_SAVINGS_LOG") or DEFAULT_LOG_PATH)
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


@dataclass
class SavingsSummary:
    calls: int = 0
    calls_with_exact_counts: int = 0
    original_tokens: int = 0
    sent_tokens: int = 0
    redacted_fields: int = 0
    redacted_types: dict = field(default_factory=dict)
    estimated_cost_saved_usd: float = 0.0
    calls_with_price: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    calls_with_provider_usage: int = 0
    local_overhead_ms_total: float = 0.0
    overhead_ms_values: list = field(default_factory=list)
    history_tokens_lost: int = 0
    summary_folds: int = 0
    summary_reuses: int = 0
    fold_failures: int = 0
    chunk_filter_skips: int = 0
    by_app: dict = field(default_factory=dict)
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.sent_tokens)

    @property
    def percent_saved(self) -> float:
        if not self.original_tokens:
            return 0.0
        return round(100 * self.tokens_saved / self.original_tokens, 1)

    @property
    def avg_local_overhead_ms(self) -> float:
        return round(self.local_overhead_ms_total / self.calls, 1) if self.calls else 0.0

    @property
    def median_local_overhead_ms(self) -> float:
        v = sorted(self.overhead_ms_values)
        if not v:
            return 0.0
        mid = len(v) // 2
        return round(v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2, 1)

    @property
    def max_local_overhead_ms(self) -> float:
        return round(max(self.overhead_ms_values), 1) if self.overhead_ms_values else 0.0

    @property
    def tokens_saved_excluding_lost_history(self) -> int:
        return max(0, self.tokens_saved - self.history_tokens_lost)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "original_tokens": self.original_tokens,
            "sent_tokens": self.sent_tokens,
            "tokens_saved": self.tokens_saved,
            "percent_saved": self.percent_saved,
            "token_counts_are_estimates": self.calls_with_exact_counts < self.calls,
            "calls_with_exact_counts": self.calls_with_exact_counts,
            "redacted_fields": self.redacted_fields,
            "redacted_types": self.redacted_types,
            "estimated_cost_saved_usd": round(self.estimated_cost_saved_usd, 6),
            "calls_with_price": self.calls_with_price,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "calls_with_provider_usage": self.calls_with_provider_usage,
            "history_tokens_lost": self.history_tokens_lost,
            "tokens_saved_excluding_lost_history": self.tokens_saved_excluding_lost_history,
            "summary_folds": self.summary_folds,
            "summary_reuses": self.summary_reuses,
            "fold_failures": self.fold_failures,
            "chunk_filter_skips": self.chunk_filter_skips,
            "avg_local_overhead_ms": self.avg_local_overhead_ms,
            "median_local_overhead_ms": self.median_local_overhead_ms,
            "max_local_overhead_ms": self.max_local_overhead_ms,
            "by_app": self.by_app,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
        }


def summarize(path: Optional[str] = None, app: Optional[str] = None, since: Optional[str] = None) -> SavingsSummary:
    """
    Aggregates the log. `app` filters to one app name; `since` is an ISO
    date/datetime string (e.g. "2026-09-01") -- entries earlier than it
    are skipped. Timestamps are compared as UTC ISO strings, which sort
    correctly lexicographically.
    """
    s = SavingsSummary()
    types: Counter = Counter()
    for e in read_entries(path):
        if app is not None and e.get("app") != app:
            continue
        ts = e.get("ts") or ""
        if since and ts < since:
            continue
        s.calls += 1
        s.calls_with_exact_counts += e.get("token_counts_are_estimates") is False
        s.original_tokens += int(e.get("original_tokens", 0))
        s.sent_tokens += int(e.get("sent_tokens", 0))
        s.redacted_fields += int(e.get("redacted_fields", 0))
        types.update(e.get("redacted_types") or {})
        if "estimated_cost_saved_usd" in e:
            s.estimated_cost_saved_usd += float(e["estimated_cost_saved_usd"])
            s.calls_with_price += 1
        pu = e.get("provider_usage")
        if pu:
            s.calls_with_provider_usage += 1
            s.cache_read_input_tokens += int(pu.get("cache_read_input_tokens", 0))
            s.cache_creation_input_tokens += int(pu.get("cache_creation_input_tokens", 0))
        overhead = float(e.get("local_overhead_ms", 0.0))
        s.local_overhead_ms_total += overhead
        s.overhead_ms_values.append(overhead)
        s.history_tokens_lost += int(e.get("history_tokens_lost", 0))
        s.summary_folds += bool(e.get("history_summary_updated"))
        s.summary_reuses += bool(e.get("history_summary_reused"))
        s.fold_failures += bool(e.get("history_fold_failed"))
        s.chunk_filter_skips += bool(e.get("chunk_filter_skipped"))
        name = e.get("app") or "(no app)"
        a = s.by_app.setdefault(name, {"calls": 0, "original_tokens": 0, "sent_tokens": 0})
        a["calls"] += 1
        a["original_tokens"] += int(e.get("original_tokens", 0))
        a["sent_tokens"] += int(e.get("sent_tokens", 0))
        if ts:
            s.first_ts = ts if s.first_ts is None or ts < s.first_ts else s.first_ts
            s.last_ts = ts if s.last_ts is None or ts > s.last_ts else s.last_ts
    s.redacted_types = dict(sorted(types.items()))
    return s


def format_summary(s: SavingsSummary) -> str:
    """Human-readable text for `tonst stats`."""
    if s.calls == 0:
        return "No tonst calls logged yet."
    if s.calls_with_exact_counts == s.calls:
        how = "counted"
    elif s.calls_with_exact_counts:
        how = f"counted on {s.calls_with_exact_counts} of {s.calls} calls, rest estimated"
    else:
        how = "estimated"
    lines = [
        f"tonst savings  ({s.first_ts} -> {s.last_ts})",
        f"  calls:            {s.calls:,}",
        f"  tokens in:        {s.original_tokens:,}  ({how})",
        f"  tokens sent:      {s.sent_tokens:,}  ({how})",
        f"  tokens saved:     {s.tokens_saved:,}  ({s.percent_saved}%)",
    ]
    if s.history_tokens_lost:
        pct = round(100 * s.tokens_saved_excluding_lost_history / s.original_tokens, 1) if s.original_tokens else 0.0
        lines.append(
            f"    of which lost:  {s.history_tokens_lost:,}  (old chat history dropped without a summary --"
            " the model never saw it)"
        )
        lines.append(f"    saved excl. lost history: {s.tokens_saved_excluding_lost_history:,}  ({pct}%)")
    if s.calls_with_price:
        note = "" if s.calls_with_price == s.calls else f", priced on {s.calls_with_price} of {s.calls} calls"
        lines.append(f"  est. cost saved:  ${s.estimated_cost_saved_usd:,.4f}{note}")
    if s.calls_with_provider_usage:
        lines.append(
            f"  provider cache:   {s.cache_read_input_tokens:,} tokens read from cache, "
            f"{s.cache_creation_input_tokens:,} written (real usage, {s.calls_with_provider_usage} calls)"
        )
    types = ", ".join(f"{k} {v}" for k, v in s.redacted_types.items()) or "none"
    lines.append(f"  PII redacted:     {s.redacted_fields:,} fields ({types})")
    if s.summary_folds or s.summary_reuses or s.fold_failures:
        lines.append(
            f"  rolling summary:  {s.summary_folds} folds, {s.summary_reuses} reuses (no model call), "
            f"{s.fold_failures} failed folds"
        )
    if s.chunk_filter_skips:
        lines.append(
            f"  RAG:              relevance filtering skipped on {s.chunk_filter_skips} calls "
            "(no confident match; duplicates still removed)"
        )
    lines.append(
        f"  tonst overhead:   median {s.median_local_overhead_ms} ms, max {s.max_local_overhead_ms} ms "
        f"(mean {s.avg_local_overhead_ms} ms)"
    )
    if len(s.by_app) > 1:
        lines.append("  by app:")
        for name, a in sorted(s.by_app.items()):
            saved = max(0, a["original_tokens"] - a["sent_tokens"])
            pct = round(100 * saved / a["original_tokens"], 1) if a["original_tokens"] else 0.0
            lines.append(f"    {name}: {a['calls']:,} calls, {saved:,} tokens saved ({pct}%)")
    return "\n".join(lines)
