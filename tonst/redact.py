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
"""

from __future__ import annotations
import re
import uuid
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


@dataclass
class RedactionResult:
    redacted_text: str
    # Maps placeholder token -> original value, kept ONLY in memory on
    # the local machine. Never sent to the cloud model or logged.
    mapping: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        """Re-insert real values into a model response that may echo placeholders."""
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text


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
            placeholder = f"[[{label}_{uuid.uuid4().hex[:8]}]]"
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
