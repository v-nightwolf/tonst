"""
gliner_redact.py
-----------------
Alternative free-text PII redaction backend using GLiNER -- a small,
extractive/zero-shot NER model -- instead of a generative local LLM via
Ollama (see redact_llm.py). GLiNER never rewrites text: it returns
spans/offsets into the ORIGINAL text, so it structurally cannot produce
the JSON-parsing/truncation/hallucination failure modes redact_llm.py
has had to guard against. It also runs on CPU, at a fraction of
Ollama's latency -- see research/gliner-sanity-check-findings.md for
the full comparison (gliner_medium: ~183ms mean latency, 98.89% company
recall, 74.44% strict codename recall vs. Ollama's ~1000ms+ and its own
JSON-parsing failure modes).

Model choice: `urchade/gliner_medium-v2.1` is the validated default --
deliberately NOT the largest available checkpoint. A direct head-to-
head against `gliner_large-v2.1` (2026-09-13) found model size is not a
monotonic lever past medium: company recall regressed from 98.89% to
86.11% and codename strict recall from 74.44% to 63.33%, while latency
got 2.6x worse. Do not "upgrade" this default to gliner_large without
re-running that comparison first -- it made things worse, not better.

Known limitation, not a bug: codename recall on the two "supervised"
prompt shapes (a direct task referencing a codename) sits at 58-60%
even with gliner_medium -- a real, accepted gap tracked in the research
doc, not something this module tries to paper over.

Kept as its own module (not merged into redact_llm.py), matching this
project's one-module-per-mechanism convention, and so that `gliner` +
its ML dependencies (torch) are only ever imported when this specific
backend is actually selected -- see client.py's lazy import of this
module, and _load()'s lazy import of `gliner` itself below.
"""

from __future__ import annotations
import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MODEL = "urchade/gliner_medium-v2.1"

# Process-wide model cache, keyed by model name. This matters a lot more
# than it looks: unlike Ollama (a separate server process that stays
# warm regardless of how many LLMRedactor/TonstClient objects come and
# go), GLiNER is an in-process model with no separate server -- if the
# loaded weights are cached only on a GlinerRedactor INSTANCE, every
# caller that constructs a fresh TonstClient per call/request (a common
# pattern, and exactly what benchmarks/benchmark_tonst.py's run_single_iteration()
# does) reloads the model from disk every single time. Found 2026-09-13
# via a live benchmark: redaction latency landed at a suspiciously flat
# ~6.1-6.4s p50 across every industry/shape (the signature of a fixed
# reload cost dominating, not variable per-case inference -- GLiNER's
# own isolated sanity check measured ~183-476ms mean depending on model
# size). Caching at module level instead of per-instance means the
# expensive load happens once per process, no matter how many
# GlinerRedactor/TonstClient instances are created afterward -- the
# correct equivalent of Ollama's already-persistent server process.
_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.Lock()

# Was 0.3 originally. Raised codename/company recall by lowering to
# 0.22, chosen from a fine-grained sweep (0.30 down to 0.15) run
# 2026-09-13 AFTER the placeholder-inner guard below was added (that
# fix is what makes lowering this safe at all -- before it, a lower
# threshold risked GLiNER catching a bracket-stripped placeholder and
# corrupting it into a malformed nested token). Findings from that
# sweep, all measured through this exact class with the production
# 3-label set on the same 154-case set:
#   - company recall reaches a clean 100% (from 98.70%) starting at
#     0.25 and holds there -- a free win, no new false positives.
#   - From 0.30 down through 0.22, precision barely moves (97.27% ->
#     96.38%) and the false positives are the SAME three recurring,
#     benign ones every time (a stray "Space"/"Project" label-word, one
#     email-fragment split) -- no new failure category, just gradual
#     codename recall gains (54.55% -> 57.79%).
#   - At 0.20, a genuinely new and less benign pattern appears for the
#     first time: generic scaffolding text like "Lead engineer" (a job
#     title) getting mistaken for a person's NAME. It recurs and
#     multiplies at 0.18/0.15, which is also where precision actually
#     starts falling (95.04% -> 93.09% -> 87.55%).
# 0.22 is the lowest value tested that stays inside the "same three
# known, benign false positives" zone -- one step before that new
# failure mode starts, not deep into the region where it dominates.
DEFAULT_THRESHOLD = 0.22

# Detects an ALREADY-PLACED placeholder's inner content (e.g. the
# "IP_ADDRESS_18061056" inside "[[IP_ADDRESS_18061056]]") anywhere it
# currently appears in the input text. Needed because of a real bug
# found 2026-09-13 while threshold-tuning: at lower thresholds GLiNER
# sometimes returns a span that is the INSIDE of an existing regex
# placeholder with the brackets stripped off (e.g. it returns
# "IP_ADDRESS_18061056", not "[[IP_ADDRESS_18061056]]"). The existing
# `"[[" in span` guard below doesn't catch that, because the brackets
# genuinely aren't part of the span GLiNER handed back -- so redact()
# would then do result_text.replace("IP_ADDRESS_18061056", new_placeholder),
# which mangles the ALREADY-PLACED placeholder into a malformed, doubly-
# nested token (e.g. "[[[[CODENAME_xxxx]]]]") that restore_placeholders()
# can't parse back at the end of the pipeline -- meaning the real PII
# behind it silently fails to restore into the final output. This regex
# lets redact() reject any GLiNER span that matches placeholder content
# already sitting in the text, regardless of whether the brackets came
# along with it.
_PLACEHOLDER_INNER_RE = re.compile(r"\[\[([^\[\]]+)\]\]")

# Simple, "canonical entity-type phrase" labels -- validated in
# research/gliner-sanity-check-findings.md. That doc's Run 3 found more
# descriptive label wording measurably HURTS recall, including on
# fields whose label wasn't even changed (GLiNER conditions on the full
# label set jointly at inference time). Do not "improve" these without
# controlled, single-variable re-testing against that doc's baselines.
DEFAULT_LABELS = {
    "full_name": "person name",
    "company": "company name",
    "codename": "project codename",
}
_LABEL_TO_TAG = {
    "person name": "NAME",
    "company name": "EMPLOYER",
    "project codename": "CODENAME",
}


@dataclass
class GlinerRedactionResult:
    redacted_text: str
    # Maps placeholder token -> original value, same contract as
    # redact.RedactionResult / redact_llm.LLMRedactionResult.
    mapping: dict[str, str] = field(default_factory=dict)
    model_available: bool = True
    entities_found: int = 0


def _placeholder_for(label: str, span: str) -> str:
    digest = hashlib.sha256(span.encode("utf-8")).hexdigest()[:8]
    return f"[[{label}_{digest}]]"


class GlinerRedactor:
    """
    Duck-type compatible with redact_llm.LLMRedactor: exposes
    is_available() and redact(text) -> an object with .redacted_text /
    .mapping. That's all redact.redact_with_llm(text, redactor) relies
    on, so it works unmodified with either backend -- no separate
    "redact_with_gliner" helper needed.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        labels: Optional[dict[str, str]] = None,
    ):
        self.model_name = model
        self.threshold = threshold
        self.labels = labels or DEFAULT_LABELS

    def _load(self):
        # Fast path: no lock needed just to READ an already-cached model
        # (dict reads are atomic under the GIL); the lock only guards
        # the load-and-populate path below, so steady-state calls after
        # the first never pay any lock overhead.
        cached = _MODEL_CACHE.get(self.model_name)
        if cached is not None:
            return cached
        with _MODEL_CACHE_LOCK:
            # Re-check: another thread may have loaded it while this one
            # was waiting on the lock.
            cached = _MODEL_CACHE.get(self.model_name)
            if cached is not None:
                return cached
            try:
                from gliner import GLiNER
            except ImportError as exc:
                raise ImportError(
                    "redaction_backend='gliner' was selected, but the `gliner` "
                    "package isn't installed. Run: pip install gliner"
                ) from exc
            model = GLiNER.from_pretrained(self.model_name)
            _MODEL_CACHE[self.model_name] = model
            return model

    def is_available(self) -> bool:
        try:
            self._load()
            return True
        except Exception:
            return False

    def redact(self, text: str) -> GlinerRedactionResult:
        try:
            model = self._load()
        except Exception:
            # Fails soft, exactly like LLMRedactor: if the model can't
            # load (missing package, bad model name, etc.), the text
            # passes through with only the regex-layer redaction that
            # already ran in redact_with_llm() before this is called.
            return GlinerRedactionResult(redacted_text=text, mapping={}, model_available=False, entities_found=0)

        gliner_labels = list(self.labels.values())
        try:
            entities = model.predict_entities(text, gliner_labels, threshold=self.threshold)
        except Exception:
            return GlinerRedactionResult(redacted_text=text, mapping={}, model_available=False, entities_found=0)

        mapping: dict[str, str] = {}
        result_text = text
        # See _PLACEHOLDER_INNER_RE's comment above: reject any GLiNER
        # span that is the inner content of a placeholder already
        # produced, even if GLiNER stripped the surrounding brackets.
        existing_placeholder_inner = set(_PLACEHOLDER_INNER_RE.findall(text))
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            span = entity.get("text")
            gliner_label = entity.get("label")
            if not span or not isinstance(span, str):
                continue
            # Same defense-in-depth guard as redact_llm.py: a genuine
            # free-text span never legitimately contains "[[" -- if it
            # does, it's overlapping an already-placed regex placeholder
            # (redact_with_llm() runs regex first), not real text.
            if "[[" in span or span in existing_placeholder_inner:
                continue
            # GLiNER is extractive (spans are sliced from the input by
            # offset), so unlike a generative model this should always
            # be an exact substring already -- this check is cheap
            # defense-in-depth, not a known failure mode here.
            if span not in result_text:
                continue
            tag = _LABEL_TO_TAG.get(gliner_label, "PII")
            placeholder = _placeholder_for(tag, span)
            mapping[placeholder] = span
            # Replace EVERY occurrence, not just the first -- see the
            # 2026-09-13 fix/postmortem in redact_llm.py for why a
            # count=1 limit is a real PII leak, not just a missed
            # optimization, when the same span repeats in the source.
            result_text = result_text.replace(span, placeholder)

        return GlinerRedactionResult(
            redacted_text=result_text,
            mapping=mapping,
            model_available=True,
            entities_found=len(mapping),
        )
