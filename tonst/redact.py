"""
redact.py
---------
Local, regex-based PII redaction. This runs entirely on the caller's own
machine/server -- nothing here ever touches the network. The point is to
strip sensitive fields BEFORE the prompt is sent to a paid cloud LLM, then
put the real values back into the response afterwards.

This is intentionally dependency-free (no spaCy/NER model) so it can run
on modest hardware. For production you'd likely pair this with a small
local model (via Ollama) for fuzzier redaction (e.g. free-text names),
but regex covers the highest-value, highest-confidence categories:
emails, phone numbers, card numbers, SSN-like IDs, and IP addresses.

Placeholders are DETERMINISTIC: a hash of the original value, not a
random UUID. This matters beyond just "same input -> same output" --
cache_structuring.py relies on stable/system content being byte-for-byte
identical across calls for provider-side prompt caching to work. If the
same email address redacted to a different random placeholder on every
call, a stable block containing it would never match its own previous
version, silently defeating caching every single time. A hash gives the
same placeholder for the same value, every call, while still not
revealing the original.
"""

from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field

# Order matters: more specific patterns first so they aren't partially
# swallowed by a looser pattern later in the list.
PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "PHONE": re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "SSN_LIKE": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _placeholder_for(label: str, original: str) -> str:
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:8]
    return f"[[{label}_{digest}]]"


def restore_placeholders(text: str, mapping: dict[str, str]) -> str:
    """
    Re-insert real values wherever a placeholder from `mapping` appears
    in `text`. Free function (not tied to a RedactionResult instance) so
    callers who accumulate a mapping across multiple redaction passes --
    e.g. TonstClient.query_messages(), which redacts several messages
    before combining them -- can restore against the combined mapping
    without constructing a throwaway RedactionResult just to call
    .restore() on it.
    """
    # Loop (bounded) rather than a single forward pass: if a real
    # value happens to itself contain another known placeholder token
    # -- defense in depth against a bug elsewhere producing a nested/
    # wrapped placeholder, e.g. the redact_llm.py entity-schema guard
    # rail fix (2026-09-11) -- a single pass can leave an inner
    # placeholder unrestored purely because of dict iteration order.
    # Looping until nothing changes makes restoration robust to that
    # regardless of ordering; the small iteration cap guarantees
    # termination even in a pathological mapping.
    for _ in range(5):
        new_text = text
        for placeholder, original in mapping.items():
            new_text = new_text.replace(placeholder, original)
        if new_text == text:
            break
        text = new_text
    return text


@dataclass
class RedactionResult:
    redacted_text: str
    # Maps placeholder token -> original value, kept ONLY in memory on
    # the local machine. Never sent to the cloud model or logged.
    mapping: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        """Re-insert real values into a model response that may echo placeholders."""
        return restore_placeholders(text, self.mapping)


def redact(text: str) -> RedactionResult:
    mapping: dict[str, str] = {}
    result_text = text

    for label, pattern in PATTERNS.items():
        def _sub(match: re.Match, label=label) -> str:
            original = match.group(0)
            # Skip short numeric noise being misfired as a card/phone number
            digits_only = re.sub(r"\D", "", original)
            if label in ("CREDIT_CARD", "PHONE", "SSN_LIKE") and len(digits_only) < 7:
                return original
            placeholder = _placeholder_for(label, original)
            mapping[placeholder] = original
            return placeholder

        result_text = pattern.sub(_sub, result_text)

    return RedactionResult(redacted_text=result_text, mapping=mapping)


def redact_with_llm(text: str, llm_redactor) -> RedactionResult:
    """
    Two-stage redaction: fast, deterministic regex first (catches emails,
    phones, cards, IPs), then the local-LLM pass on what's left (catches
    free-text names, addresses, employers, codenames -- see redact_llm.py).

    `llm_redactor` is a `tonst.redact_llm.LLMRedactor` instance, passed in
    rather than constructed here so callers control the model/timeout and
    tests can inject a fake one. If the local model isn't available, this
    silently degrades to regex-only redaction -- never raises.
    """
    regex_result = redact(text)
    llm_result = llm_redactor.redact(regex_result.redacted_text)

    combined_mapping = {**regex_result.mapping, **llm_result.mapping}
    return RedactionResult(redacted_text=llm_result.redacted_text, mapping=combined_mapping)
