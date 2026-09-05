"""
redact_llm.py
-------------
Upgrades redaction beyond regex. Every competing tool we looked at
(LLMShield, Helix, the WSO2 sample gateway) does PII redaction with
regex only -- which is fast and reliable for structured data (emails,
card numbers) but fundamentally can't catch free-text PII: a person's
name in a sentence, a home address, an internal project codename, a
patient's condition mentioned in prose. Regex has no way to know
"Priya Malhotra" is a name without a dictionary of every name in the
world.

This module runs a SMALL LOCAL MODEL (via Ollama) whose only job is to
find free-text PII spans and report them as structured data. It never
sees the network beyond localhost, and it never rewrites the prompt's
wording (that's local_model.py's job, kept deliberately separate) --
it only identifies spans to redact, the same mechanical placeholder
swap that redact.py already does for regex matches.

Design choices that matter:
- Strict output contract (JSON array only) with a guarded parser, so a
  local 1-3B model that occasionally misbehaves can't corrupt the
  pipeline -- malformed output is discarded, not guessed at.
- Fails soft exactly like local_model.py: if Ollama isn't running, or
  the model's output doesn't parse, the text passes through with only
  the regex-layer redaction applied. This step is additive, never load
  -bearing.
- Injectable model-call function (`model_call_fn`) so this is testable
  without a real Ollama instance -- see redact_llm_test in the demo.
"""

from __future__ import annotations
import json
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# Deliberately narrow instruction: find spans, don't rewrite, don't explain.
# The model is told the exact categories we want so it doesn't improvise
# (e.g. flagging "the invoice" as sensitive, which would over-redact).
REDACTION_PROMPT = """You detect personally identifiable information (PII) in text.

Find every span of free-text PII in the text below: full person names, \
home/mailing addresses, employer or company names when tied to a specific \
person, and specific project codenames. Do NOT flag emails, phone numbers, \
or card numbers -- those are handled separately.

Respond with ONLY a JSON array, nothing else. Each item: {{"text": "<exact \
substring from the input>", "type": "<NAME|ADDRESS|EMPLOYER|CODENAME>"}}. \
If nothing is found, respond with [].

Text:
---
{text}
---
JSON:"""


@dataclass
class LLMRedactionResult:
    redacted_text: str
    mapping: dict[str, str]
    model_available: bool
    entities_found: int


def _default_ollama_call(prompt: str, model: str, timeout: float) -> Optional[str]:
    try:
        resp = requests.post(
            DEFAULT_OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.RequestException:
        return None


def _extract_json_array(raw: str) -> list:
    """
    Small local models sometimes wrap JSON in prose or code fences despite
    instructions. Pull out the first [...] block rather than trusting the
    whole response to be clean JSON.
    """
    if not raw:
        return []
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, list):
            return []
        return parsed
    except (json.JSONDecodeError, ValueError):
        return []


class LLMRedactor:
    def __init__(
        self,
        model: str = "llama3.2:1b",
        timeout: float = 8.0,
        model_call_fn: Optional[Callable[[str, str, float], Optional[str]]] = None,
    ):
        self.model = model
        self.timeout = timeout
        # Injectable for testing -- production callers omit this and get
        # the real Ollama HTTP call.
        self._call_model = model_call_fn or _default_ollama_call

    def is_available(self) -> bool:
        try:
            resp = requests.get(DEFAULT_OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=1.5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def redact(self, text: str) -> LLMRedactionResult:
        raw = self._call_model(REDACTION_PROMPT.format(text=text), self.model, self.timeout)
        if raw is None:
            return LLMRedactionResult(redacted_text=text, mapping={}, model_available=False, entities_found=0)

        entities = _extract_json_array(raw)
        mapping: dict[str, str] = {}
        result_text = text

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            span = entity.get("text")
            label = entity.get("type", "PII")
            if not span or not isinstance(span, str):
                continue
            # Guard rail: only redact spans that actually appear verbatim in
            # the source text. A model that hallucinates a span that isn't
            # really there should not corrupt the output.
            if span not in result_text:
                continue
            placeholder = f"[[{label}_{uuid.uuid4().hex[:8]}]]"
            mapping[placeholder] = span
            # Replace only the first remaining occurrence per entity so
            # repeated identical spans (e.g. a name used twice) each get
            # their own placeholder-to-value mapping correctly restored.
            result_text = result_text.replace(span, placeholder, 1)

        return LLMRedactionResult(
            redacted_text=result_text,
            mapping=mapping,
            model_available=True,
            entities_found=len(mapping),
        )
