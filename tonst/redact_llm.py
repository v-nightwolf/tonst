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
- Placeholders are DETERMINISTIC (a hash of the exact span), matching
  redact.py -- the same free-text span redacts to the same placeholder
  on every call, which is required for cache_structuring.py's stable
  blocks to stay byte-identical across calls when they contain PII.
"""

from __future__ import annotations
import ast
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# Diagnostic only -- never affects behavior. Enable with
# logging.getLogger("tonst.redact_llm").setLevel(logging.DEBUG) to see
# per-call elapsed time and exactly why a call failed soft (timeout vs.
# connection refused vs. bad HTTP status vs. unparseable JSON), instead
# of every failure mode collapsing into the same silent "model_available=
# False" outcome. Added after a real benchmark showed this call
# clustered at almost exactly its 8.0s timeout across 360 runs with a
# 15ms spread -- the tight clustering IS the signature of a deterministic
# timeout, but distinguishing "always times out" from "always refused
# instantly" needs this, not just aggregate latency.
logger = logging.getLogger(__name__)

# One shared connection pool instead of a fresh TCP/HTTP handshake per
# call -- real, if modest, latency savings on repeated calls to the same
# local Ollama endpoint. This alone will not explain a multi-second
# timeout; it only removes per-call connection-setup overhead.
_SESSION = requests.Session()

# Deliberately narrow instruction: find spans, don't rewrite, don't explain.
# The model is told the exact categories we want so it doesn't improvise
# (e.g. flagging "the invoice" as sensitive, which would over-redact).
REDACTION_PROMPT = """You detect personally identifiable information (PII) in text.

Find every span of free-text PII in the text below: full person names, \
home/mailing addresses, employer or company names when tied to a specific \
person, and specific project codenames. Do NOT flag emails, phone numbers, \
or card numbers -- those are handled separately.

The text may already contain tokens shaped like [[LABEL_xxxxxxxx]] -- these \
are placeholders from an earlier redaction pass, not real text. Completely \
ignore them: never include one in your output, never treat it as PII, and \
never copy it (with or without surrounding words) into a "text" field.

Respond with ONLY a JSON array, nothing else. Each item: {{"text": "<exact \
substring from the input>", "type": "<NAME|ADDRESS|EMPLOYER|CODENAME>"}}. \
If nothing is found, respond with [].

Text:
---
{text}
---
JSON:"""

# Schema enforcement for the guard rail below -- the ONLY types this
# prompt ever asks for. A model response using anything else is either
# hallucinating a category or (found directly, 2026-09-11, gemma2:2b)
# re-flagging an ALREADY-REDACTED placeholder from the regex pass using
# an invented type like EMAIL/PHONE/SSN_LIKE/CREDIT_CARD/IP_ADDRESS.
ALLOWED_ENTITY_TYPES = {"NAME", "ADDRESS", "EMPLOYER", "CODENAME"}


@dataclass
class LLMRedactionResult:
    redacted_text: str
    mapping: dict[str, str]
    model_available: bool
    entities_found: int


def _placeholder_for(label: str, span: str) -> str:
    digest = hashlib.sha256(span.encode("utf-8")).hexdigest()[:8]
    return f"[[{label}_{digest}]]"


def _default_ollama_call(prompt: str, model: str, timeout: float) -> Optional[str]:
    t0 = time.perf_counter()
    try:
        resp = _SESSION.post(
            DEFAULT_OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": 300}},
            timeout=timeout,
        )
        resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("redact_llm ollama call ok model=%s elapsed_ms=%.1f", model, elapsed_ms)
        return resp.json().get("response", "")
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "redact_llm ollama call TIMED OUT model=%s configured_timeout=%.1fs elapsed_ms=%.1f",
            model, timeout, elapsed_ms,
        )
        return None
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.warning(
            "redact_llm ollama call FAILED model=%s elapsed_ms=%.1f error=%s: %s",
            model, elapsed_ms, type(exc).__name__, exc,
        )
        return None


def _iter_balanced_spans(raw: str, open_ch: str, close_ch: str):
    """
    Yields every top-level, BALANCED `open_ch...close_ch` substring of
    `raw`, left to right (e.g. every `[...]` span, or every `{...}`
    span, depending on which characters are passed). Unlike a single
    greedy regex match (the original approach here), this does not get
    confused by an unrelated bracket appearing earlier in a chatty
    response -- e.g. a small model prefacing its real answer with
    something like "The format is [name, employer]." before the actual
    JSON array. A greedy `\\[.*\\]` regex spans from that FIRST `[` to
    the LAST `]` in the whole string, which can swallow unrelated prose
    in between and produce unparseable garbage even though a perfectly
    valid array follows it. This instead finds each self-contained
    bracket pair as its own candidate, so a caller can try parsing each
    one in turn until one succeeds.
    """
    n = len(raw)
    i = 0
    while i < n:
        if raw[i] == open_ch:
            depth = 0
            start = i
            j = i
            closed = False
            while j < n:
                if raw[j] == open_ch:
                    depth += 1
                elif raw[j] == close_ch:
                    depth -= 1
                    if depth == 0:
                        yield raw[start : j + 1]
                        closed = True
                        break
                j += 1
            if not closed:
                return
            i = j + 1
        else:
            i += 1


def _try_parse(span: str):
    """Strict JSON first, then a Python-literal fallback (safe -- only
    evaluates literal data structures, never executes code). Returns
    None if neither parses."""
    try:
        return json.loads(span)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(span)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return None


def _extract_json_array(raw: str) -> list:
    """
    Small local models routinely don't follow "respond with ONLY a JSON
    array" to the letter. Observed failure modes worth handling
    explicitly (found via offline fixture testing AND a real local
    Ollama run, Sept 2026, after a real benchmark showed zero free-text
    entities caught across 360 calls -- see redact_llm's docstring and
    ROADMAP.md/colab-benchmark-findings.md):
      - A markdown code fence around the array (```json ... ```).
      - Some prose before or after the array ("Here is the JSON: [...]").
      - The array wrapped in an object instead of returned bare
        (e.g. {"entities": [...]}).
      - An earlier, UNRELATED bracket pair in the response (e.g. the
        model echoing part of its own instructions) that a naive greedy
        regex would merge with the real array into one unparseable blob.
      - Python-literal-style output (single-quoted strings) instead of
        strict JSON -- common in models trained on a lot of Python code.
      - The array wrapper dropped ENTIRELY when there's only one match:
        a bare object like `{"text": "Arjun", "type": "NAME"}` with no
        `[` `]` anywhere -- confirmed against a real (not simulated)
        llama3.2:1b response during local testing. Despite the prompt
        saying "Respond with ONLY a JSON array. Each item: {...}", a
        small model asked for a list of "items" will sometimes just
        return the one item's shape directly when there's a single
        match, rather than wrapping it.

    Strategy: try every top-level bracket-balanced `[...]` span first
    (see _iter_balanced_spans), in order. If none parse to a list, fall
    back to every top-level `{...}` span and treat a successfully
    parsed object as a one-item list -- this is what a caller wants
    when the model skipped the array wrapper for a single match. This
    means a real array or a real bare object anywhere in the response
    is found even if something before or after it is garbage.
    """
    if not raw:
        return []
    for span in _iter_balanced_spans(raw, "[", "]"):
        parsed = _try_parse(span)
        if isinstance(parsed, list):
            return parsed
    for span in _iter_balanced_spans(raw, "{", "}"):
        parsed = _try_parse(span)
        if isinstance(parsed, dict):
            return [parsed]
    return []


class LLMRedactor:
    def __init__(
        self,
        model: str = "gemma2:2b",
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
            resp = _SESSION.get(DEFAULT_OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=1.5)
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
            # Guard rail: reject anything outside the schema we actually
            # asked for. Root cause found 2026-09-11 (gemma2:2b, live
            # benchmark): redact_with_llm() runs regex redaction FIRST
            # (see redact.py), so this model is shown text that already
            # contains [[LABEL_hex]] placeholders -- and a chattier
            # model can "helpfully" re-report an existing placeholder as
            # its own entity, using an invented type never in the
            # allowed list. Rejecting anything outside the schema stops
            # that at the source rather than downstream.
            if not isinstance(label, str) or label.upper() not in ALLOWED_ENTITY_TYPES:
                continue
            # Guard rail: a genuine free-text PII span from the user's
            # own input never legitimately contains "[[" -- if the
            # model's "text" field does, it's quoting (or gluing words
            # onto) an ALREADY-REDACTED token from the earlier regex
            # pass, not real text. Left unchecked this wraps an existing
            # placeholder inside a brand-new one (mapping[new] = "[[OLD_
            # hex]]" instead of the real value), and restore_placeholders()
            # can leak the orphaned inner placeholder verbatim into the
            # final response -- this is the single check that would have
            # caught every corruption case found in the investigation.
            if "[[" in span:
                continue
            # Guard rail: only redact spans that actually appear in the
            # source text. A model that hallucinates a span that isn't
            # really there should not corrupt the output.
            if span not in result_text:
                # Small local models sometimes normalize a proper noun's
                # casing even when told to return the exact substring
                # (found via offline fixture testing, Sept 2026: a model
                # returning "arjun rao" for source text containing
                # "Arjun Rao" fails an exact match and gets silently
                # dropped, even though the span genuinely is present).
                # Fall back to a case-insensitive search before giving up
                # -- but always redact and hash the ACTUAL text found in
                # the source, never the model's re-cased version, so this
                # can never introduce text that wasn't really there, and
                # placeholders stay just as deterministic as before (same
                # source text -> same placeholder, regardless of which
                # casing the model happened to emit that run).
                match = re.search(re.escape(span), result_text, re.IGNORECASE)
                if not match:
                    continue
                span = match.group(0)
            placeholder = _placeholder_for(label, span)
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
