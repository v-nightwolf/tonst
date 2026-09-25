"""
local_model.py
--------------
Optional layer that uses a small local model (served by Ollama, e.g.
`llama3.2:1b` or `qwen2.5:1.5b`) to do the harder, semantic compression:
rewriting a verbose prompt into a shorter one that preserves meaning.

This is deliberately isolated behind a small interface and OFF by default,
because it's the riskiest piece: it costs local latency, needs Ollama
installed and running, and can (rarely) drop nuance a mechanical trim
would have kept. Mechanical trimming + caching should do most of the work;
this is the extra lever for teams with heavier prompts and capable hardware.

If Ollama isn't reachable, `compress()` fails soft and returns the
original text unchanged -- the pipeline should never break because the
optional local model wasn't available.
"""

from __future__ import annotations
import logging
import re
import time

import requests

from .ollama_util import fits_context, num_ctx_for

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

# Diagnostic only -- never affects behavior. See redact_llm.py's matching
# comment: enable with logging.getLogger("tonst.local_model").setLevel(
# logging.DEBUG) to see per-call elapsed time and the specific failure
# mode (timeout vs. connection refused vs. bad status), rather than every
# failure collapsing into the same silent "fell back to original text."
logger = logging.getLogger(__name__)

# One shared connection pool instead of a fresh TCP/HTTP handshake per
# call. Real, if modest, latency savings -- will not by itself explain a
# multi-second timeout.
_SESSION = requests.Session()

COMPRESSION_INSTRUCTION = (
    "Rewrite the following text to be as short as possible while preserving "
    "every fact, instruction, and constraint. Do not add commentary. "
    "The text may contain tokens shaped like [[LABEL_xxxxxxxx]] -- these are "
    "redaction placeholders standing in for real PII. Copy every one of them "
    "verbatim, exactly as written, character for character. Never alter, "
    "recase, or invent one. "
    "Output only the rewritten text.\n\n---\n{text}"
)

# Well-formed placeholder as produced by redact.py/redact_llm.py (and any
# extraction-based redaction backend using the same [[LABEL_hexdigest]]
# shape): used to find the REAL placeholders in trusted source text.
_PLACEHOLDER_STRICT_RE = re.compile(r"\[\[[A-Z]+_[0-9a-f]{8}\]\]")
# Deliberately permissive: used to scan UNTRUSTED model output. A
# corruption (e.g. a re-cased hex digest) no longer matches the strict
# pattern above -- scanning output with the strict pattern would let a
# corrupted span sail through un-flagged because it no longer "looks
# like" a placeholder to the strict regex. Any double-bracket span,
# well-formed or not, is a candidate that must exactly match a real one.
_PLACEHOLDER_LOOSE_RE = re.compile(r"\[\[.*?\]\]")


def placeholders_preserved(original: str, rewritten: str) -> bool:
    """
    Guard rail for LocalCompressor.compress(): compression promises to
    preserve every fact/instruction/constraint losslessly, so a
    redaction placeholder must survive completely unchanged. Rejects the
    rewrite in EITHER direction:
      - a real placeholder present in `original` is missing from `rewritten`
      - a bracket-shaped span appears in `rewritten` that isn't an exact
        copy of one of the real placeholders in `original` (catches a
        model mangling/re-casing a placeholder rather than dropping it
        cleanly).
    """
    real_placeholders = set(_PLACEHOLDER_STRICT_RE.findall(original))
    if not real_placeholders:
        return True
    for ph in real_placeholders:
        if ph not in rewritten:
            return False
    for span in set(_PLACEHOLDER_LOOSE_RE.findall(rewritten)):
        if span not in real_placeholders:
            return False
    return True


class LocalCompressor:
    def __init__(self, model: str = "gemma2:2b", ollama_url: str = DEFAULT_OLLAMA_URL, timeout: float = 8.0):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = _SESSION.get(self.ollama_url.replace("/api/generate", "/api/tags"), timeout=1.5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def compress(self, text: str) -> tuple[str, bool]:
        """Returns (possibly_compressed_text, was_compressed)."""
        t0 = time.perf_counter()
        # Bound worst-case generation length. Unlike redact_llm.py (which has
        # carried "num_predict": 300 since 2026-09-11 specifically to bound
        # generation), this call had NO cap at all -- if the model ever drifts
        # off-task (rambles, or partially answers an embedded instruction
        # instead of just rewriting it -- a documented failure mode for small
        # local models under compression-model-replacement-plan.md), nothing
        # stopped it from generating for 8+ seconds and blowing the timeout.
        # A benchmark run on 2026-09-13 (--workers 1, so no contention) showed
        # exactly this: 51/360 compression calls landed within 250ms of the
        # 8.0s timeout ceiling, each one silently falling back to uncompressed
        # text on timeout -- paying full latency for zero benefit. Since a
        # genuine compression is by definition supposed to be SHORTER than the
        # input, capping generation at ~2x the input's word count leaves ample
        # room for a real rewrite while bounding runaway generation.
        max_output_tokens = max(64, int(len(text.split()) * 2))
        prompt = COMPRESSION_INSTRUCTION.format(text=text)
        if not fits_context(prompt, max_output_tokens):
            # Ollama would silently truncate the input and we'd "compress" a
            # partial copy -- skip compression instead (fail-soft, as always).
            logger.warning("compress: text too long for the local model's context window; not compressing")
            return text, False
        try:
            resp = _SESSION.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    # temperature=0: makes compression deterministic. Verified via
                    # diagnose_local_llm_perf.py (2026-09-13) to cost nothing in
                    # latency (746ms vs 714ms avg, within noise) while turning
                    # VARIED outputs into IDENTICAL ones across repeated calls on
                    # the same input -- pure upside for guard-rail predictability
                    # and benchmark reproducibility.
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_output_tokens,
                        "num_ctx": num_ctx_for(prompt, max_output_tokens),  # see ollama_util.py
                    },
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug("local_model ollama call ok model=%s elapsed_ms=%.1f", self.model, elapsed_ms)
            compressed = resp.json().get("response", "").strip()
            # Guard rail: only accept the compression if it's actually shorter
            # and not suspiciously tiny (which usually means the local model
            # misfired rather than genuinely compressed).
            if (
                compressed
                and len(compressed) < len(text)
                and len(compressed) > len(text) * 0.15
                and placeholders_preserved(text, compressed)
            ):
                return compressed, True
            return text, False
        except requests.exceptions.Timeout:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "local_model ollama call TIMED OUT model=%s configured_timeout=%.1fs elapsed_ms=%.1f",
                self.model, self.timeout, elapsed_ms,
            )
            return text, False
        except requests.RequestException as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "local_model ollama call FAILED model=%s elapsed_ms=%.1f error=%s: %s",
                self.model, elapsed_ms, type(exc).__name__, exc,
            )
            return text, False
