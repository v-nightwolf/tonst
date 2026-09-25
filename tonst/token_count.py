"""
token_count.py
--------------
Optional, near-exact token counting, as an alternative to trim.py's
chars/4 estimate.

Why this exists: a live test against the real Anthropic API
(live_test_free_features.py, 2026-09-24) found real billed input was
~1.8x tonst's chars/4 estimate on tool-calling requests. Two reasons:
Anthropic adds a hidden tool-use system prompt (a few hundred tokens)
whenever tools are present, and JSON tool schemas produce more tokens per
character than prose. Tokenizers also differ by model -- Anthropic notes
Claude 4.7+ models produce ~30% more tokens than earlier ones for the
same text -- so no single chars-per-token ratio can be right everywhere.

Anthropic's /v1/messages/count_tokens endpoint is free (separate,
generous rate limits) and returns the count for the model you name.
Anthropic describes it as an estimate that "might differ by a small
amount" from billed input -- far closer than chars/4, but still verify
against a real `usage` field if a figure really matters.

Trade-off: each count is a network round trip (typically 100-300 ms).
TonstClient counts twice per call (original and sent), so turning this
on adds visible latency -- it's timed separately as counting_ms. Good
for calibration runs, sampling, or back-office jobs; think twice before
enabling it on every latency-sensitive production call.

Fail-soft, like everything else optional in tonst: if a count fails,
the caller falls back to the chars/4 estimate and the report says the
numbers are estimates.
"""

from __future__ import annotations
import logging
import os
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"


def _default_post(url: str, headers: dict, body: dict, timeout: float) -> dict:
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


class AnthropicTokenCounter:
    """
    count_text(text)   -> tokens for `text` sent as one user message
                          (includes a few tokens of message framing;
                          identical on both sides of a before/after
                          comparison, so differences stay accurate).
    count_request(body)-> tokens for a full request body
                          ({"system", "tools", "messages"}).
    count_tools(tools) -> tokens that `tools` add to a request, INCLUDING
                          the hidden tool-use system prompt: count with
                          tools minus count without. This is what tools
                          actually cost you per call.

    Every method returns None on failure (network error, bad key,
    unsupported content such as server tools) -- callers fall back to
    estimates. The instance is callable: counter(text) == count_text(text),
    so it can be passed straight to TonstClient(token_counter=...).

    post_fn is injectable for tests.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        post_fn: Optional[Callable[[str, dict, dict, float], dict]] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout
        self._post = post_fn or _default_post
        self._base_cache: dict = {}

    def count_request(self, body: dict) -> Optional[int]:
        payload = {"model": self.model, "messages": body.get("messages") or [{"role": "user", "content": "."}]}
        if body.get("system"):
            payload["system"] = body["system"]
        if body.get("tools"):
            payload["tools"] = body["tools"]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            result = self._post(COUNT_TOKENS_URL, headers, payload, self.timeout)
            return int(result["input_tokens"])
        except Exception as e:  # noqa: BLE001 -- fail-soft by design
            logger.warning("tonst token count failed (%s); falling back to estimate", type(e).__name__)
            return None

    def count_text(self, text: str) -> Optional[int]:
        return self.count_request({"messages": [{"role": "user", "content": text or "."}]})

    __call__ = count_text

    def count_tools(self, tools: list, probe: str = ".") -> Optional[int]:
        if not tools:
            return 0
        if probe not in self._base_cache:
            self._base_cache[probe] = self.count_request({"messages": [{"role": "user", "content": probe}]})
        base = self._base_cache[probe]
        with_tools = self.count_request({"messages": [{"role": "user", "content": probe}], "tools": tools})
        if base is None or with_tools is None:
            return None
        return max(0, with_tools - base)


GEMINI_COUNT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:countTokens"


class GeminiTokenCounter:
    """
    Gemini's countTokens endpoint (free). count_text(text) counts one user
    message; count_request(body) counts a full generateContent body
    (systemInstruction, tools, contents). Returns None on failure. Callable,
    so it can be passed to TonstClient(token_counter=...). Like the
    Anthropic counter it adds a network round trip per count.
    """

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        post_fn: Optional[Callable[[str, dict, dict, float], dict]] = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        self.timeout = timeout
        self._post = post_fn or _default_post

    def count_request(self, body: dict) -> Optional[int]:
        headers = {"x-goog-api-key": self.api_key, "content-type": "application/json"}
        req = {k: v for k, v in body.items() if k in ("contents", "systemInstruction", "tools", "toolConfig")}
        payload = {"generateContentRequest": {"model": f"models/{self.model}", **req}}
        try:
            data = self._post(GEMINI_COUNT_URL.format(model=self.model), headers, payload, self.timeout)
            return int(data["totalTokens"])
        except Exception as e:  # noqa: BLE001 -- fail-soft by design
            logger.warning("GeminiTokenCounter failed (%s); falling back to estimates", type(e).__name__)
            return None

    def count_text(self, text: str) -> Optional[int]:
        return self.count_request({"contents": [{"role": "user", "parts": [{"text": text}]}]})

    __call__ = count_text
