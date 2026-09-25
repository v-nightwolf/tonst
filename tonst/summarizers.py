"""
summarizers.py
--------------
Optional, stronger summarizers for history compaction -- an alternative
to the default local model (gemma2:2b via Ollama).

Why this exists: a 24-turn live test (2026-09-25) showed the local 2B model
keeps the gist but loses exact facts: by the end its summary had dropped the
replacement's colour, the delivery slot and the case number. A small, cheap
API model summarizes far more faithfully, and with background summaries
(query_messages(..., background_summary=True)) its latency never reaches the
user.

Privacy: compaction only ever runs on ALREADY-REDACTED text (TonstClient
redacts every message before compaction -- see client.query_messages), so a
remote summarizer sees placeholders like [[EMAIL_1a2b3c4d]], never the real
values. It does see the rest of the conversation, just as the main model
you're already calling does.

Cost: each summary is one small call -- a few thousand input tokens and a
few hundred output. At Claude Haiku 4.5 list prices ($1 / $5 per million
tokens) that's well under a cent per summary. Every call's usage is
accumulated on the instance (usage_total, cost_usd()) so it can be counted
against the savings rather than hidden.

Usage:
    from tonst import TonstClient, AnthropicSummarizer
    client = TonstClient(call_fn=..., use_history_compaction=True,
                         compaction_summarizer=AnthropicSummarizer())
"""

from __future__ import annotations
import logging
import os
import threading
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5-20251001"
# USD per million tokens, used only for cost_usd(). Update if pricing changes.
HAIKU_PRICE_INPUT = 1.00
HAIKU_PRICE_OUTPUT = 5.00


def _default_post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class AnthropicSummarizer:
    """
    A model_call_fn for HistoryCompactor: called as fn(prompt, model, timeout)
    and returning the summary text, or None on any failure (fail-soft --
    compaction then falls back exactly as it does when the local model is
    down). The `model` argument HistoryCompactor passes is ignored; this
    instance's own `model` is used.
    """

    def __init__(
        self,
        model: str = DEFAULT_SUMMARY_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        timeout: float = 60.0,
        price_input_per_million: float = HAIKU_PRICE_INPUT,
        price_output_per_million: float = HAIKU_PRICE_OUTPUT,
        post_fn: Optional[Callable[[str, dict, dict, float], dict]] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.price_input = price_input_per_million
        self.price_output = price_output_per_million
        self._post = post_fn or _default_post
        self._lock = threading.Lock()  # background summaries call this from another thread
        self.calls = 0
        self.failures = 0
        self.usage_total = {"input_tokens": 0, "output_tokens": 0}

    def __call__(self, prompt: str, model: str = "", timeout: float = 0.0) -> Optional[str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            data = self._post(MESSAGES_URL, headers, body, self.timeout)
        except Exception as e:  # noqa: BLE001 -- fail-soft by design
            logger.warning("AnthropicSummarizer call failed (%s); compaction will fall back", type(e).__name__)
            with self._lock:
                self.calls += 1
                self.failures += 1
            return None
        usage = data.get("usage") or {}
        with self._lock:
            self.calls += 1
            self.usage_total["input_tokens"] += int(usage.get("input_tokens", 0))
            self.usage_total["output_tokens"] += int(usage.get("output_tokens", 0))
        text = "".join(b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text")
        return text or None

    def cost_usd(self) -> float:
        return (
            self.usage_total["input_tokens"] * self.price_input
            + self.usage_total["output_tokens"] * self.price_output
        ) / 1_000_000


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-3.5-flash-lite"
# USD per million tokens (Google's pricing page, 2026-09-25), used only for
# cost_usd(). Thinking tokens bill as output.
GEMINI_LITE_PRICE_INPUT = 0.30
GEMINI_LITE_PRICE_OUTPUT = 2.50


def gemini_usage(data: dict) -> dict:
    """Normalise a generateContent usageMetadata block (camelCase or snake_case)."""
    u = data.get("usageMetadata") or data.get("usage_metadata") or {}

    def g(*names):
        for n in names:
            if u.get(n) is not None:
                return int(u[n])
        return 0

    return {
        "prompt_tokens": g("promptTokenCount", "prompt_token_count"),          # includes cached tokens
        "cached_tokens": g("cachedContentTokenCount", "cached_content_token_count"),
        "output_tokens": g("candidatesTokenCount", "candidates_token_count"),
        "thinking_tokens": g("thoughtsTokenCount", "thoughts_token_count"),
    }


class GeminiSummarizer:
    """
    Same contract as AnthropicSummarizer (fn(prompt, model, timeout) -> str
    | None, fail-soft, usage tracked), backed by a small Gemini model via
    generateContent. For apps whose main model is Gemini: summaries stay
    with the same provider. Only ever sees already-redacted text.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_SUMMARY_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 1500,
        timeout: float = 60.0,
        thinking_level: Optional[str] = "minimal",
        price_input_per_million: float = GEMINI_LITE_PRICE_INPUT,
        price_output_per_million: float = GEMINI_LITE_PRICE_OUTPUT,
        post_fn: Optional[Callable[[str, dict, dict, float], dict]] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.thinking_level = thinking_level
        self.price_input = price_input_per_million
        self.price_output = price_output_per_million
        self._post = post_fn or _default_post
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.usage_total = {"input_tokens": 0, "output_tokens": 0}

    def _body(self, prompt: str, with_thinking: bool) -> dict:
        gen = {"maxOutputTokens": self.max_tokens, "temperature": 0}
        if with_thinking and self.thinking_level:
            gen["thinkingConfig"] = {"thinkingLevel": self.thinking_level}
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}

    def __call__(self, prompt: str, model: str = "", timeout: float = 0.0) -> Optional[str]:
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        url = GEMINI_URL.format(model=self.model)
        data = None
        for with_thinking in (True, False):  # if the model rejects the thinking level, retry without it
            try:
                data = self._post(url, headers, self._body(prompt, with_thinking), self.timeout)
                break
            except Exception as e:  # noqa: BLE001 -- fail-soft by design
                bad_request = getattr(getattr(e, "response", None), "status_code", None) == 400
                if with_thinking and self.thinking_level and bad_request:
                    continue
                logger.warning("GeminiSummarizer call failed (%s); compaction will fall back", type(e).__name__)
                break
        if data is None:
            with self._lock:
                self.calls += 1
                self.failures += 1
            return None
        u = gemini_usage(data)
        with self._lock:
            self.calls += 1
            self.usage_total["input_tokens"] += u["prompt_tokens"]
            self.usage_total["output_tokens"] += u["output_tokens"] + u["thinking_tokens"]
        parts = ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        return text or None

    def cost_usd(self) -> float:
        return (
            self.usage_total["input_tokens"] * self.price_input
            + self.usage_total["output_tokens"] * self.price_output
        ) / 1_000_000
