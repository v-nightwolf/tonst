"""
names.py
--------
One placeholder per PERSON, not per string.

The first answer-quality benchmark (2026-09-27, 100 prompts) showed the
biggest avoidable quality loss came from names: "Omar Haddad" and a later
bare "Omar" became two unrelated placeholders, so the model mixed up who
was who ("an email from Omar to Omar Haddad"), and because it only ever
saw the full-name placeholder it wrote "Hi Omar Haddad" and repeated the
full name everywhere.

apply_names() fixes both:

  "Omar Haddad"  -> [[NAME_1]]
  "Omar"         -> [[NAME_1.first]]      (same person, first name only)
  "Haddad"       -> [[NAME_1.last]]
  "Dr. Haddad"   -> Dr. [[NAME_1.last]]   (titles stay visible)

so the model can write "Hi [[NAME_1.first]]", which restores to "Hi Omar".

First and last names of every detected full name are also matched on
their own (capitalised, whole word) even when the detector didn't flag
them -- which raises recall ("Thanks, Daniel" after "Daniel Okafor" was
detected). Over-matching is safe: the value is restored afterwards.

The alias table lives on the PlaceholderFactory, so a TonstClient keeps
one identity per person across calls. If two different people share a
first name ("Omar Haddad", "Omar Khan"), a bare "Omar" is ambiguous and
is given its own placeholder instead of being guessed.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from .placeholders import PlaceholderFactory, default_factory

TITLES = {"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "sir", "madam", "shri", "smt"}

_AMBIGUOUS = object()


def split_title(span: str) -> tuple[str, str]:
    """'Dr. Mei Tanaka' -> ('Dr. ', 'Mei Tanaka'). Titles stay visible to the model."""
    tokens = span.split()
    i = 0
    while i < len(tokens) - 1 and tokens[i].rstrip(".").lower() in TITLES:
        i += 1
    if i == 0:
        return "", span.strip()
    title_text = span[: span.index(tokens[i])]
    return title_text, " ".join(tokens[i:])


def _word_re(word: str) -> re.Pattern:
    # whole word, not part of a longer word, not inside an existing placeholder
    return re.compile(r"(?<![\w\[])" + re.escape(word) + r"(?![\w\]])")


def _aliases(factory: PlaceholderFactory, local: dict) -> dict:
    """
    token -> (base placeholder, "first"|"last") or _AMBIGUOUS.
    A TonstClient's own factory remembers people across calls. The shared
    module-level default factory does NOT: it may serve unrelated callers
    in one process, and should neither link their people nor grow forever.
    """
    if factory is default_factory():
        return local
    if not hasattr(factory, "_name_aliases"):
        factory._name_aliases = {}
    return factory._name_aliases


def _register(aliases: dict, core: str, base: str) -> None:
    tokens = core.split()
    if len(tokens) < 2:
        return
    for token, part in ((tokens[0], "first"), (tokens[-1], "last")):
        if len(token) < 3 or not token[0].isupper():
            continue
        prev = aliases.get(token)
        if prev is None:
            aliases[token] = (base, part)
        elif prev is not _AMBIGUOUS and prev[0] != base:
            aliases[token] = _AMBIGUOUS


def part_placeholder(base: str, part: str) -> str:
    return base[:-2] + "." + part + "]]"


def apply_names(
    text: str,
    spans: Iterable[str],
    mapping: dict,
    factory: Optional[PlaceholderFactory] = None,
) -> str:
    """
    Replace person names in `text`. `spans` are name strings found by a
    detector (GLiNER, a local LLM); `mapping` is updated in place with
    placeholder -> original. Returns the new text.
    """
    factory = factory or default_factory()
    aliases = _aliases(factory, {})
    cores = []
    for span in spans:
        if not span or "[[" in span:
            continue
        _title, core = split_title(span)
        if core and core not in cores:
            cores.append(core)

    # Full names first, longest first, so "Omar Haddad" is placed before
    # a bare "Omar" is considered.
    multi = sorted((c for c in cores if len(c.split()) >= 2), key=len, reverse=True)
    single = [c for c in cores if len(c.split()) == 1]

    for core in multi:
        base = factory.make("NAME", core)
        _register(aliases, core, base)
        pattern = _word_re(core)
        if pattern.search(text):
            mapping[base] = core
            text = pattern.sub(base, text)
            # The model may write "Hi [[NAME_1.first]]" even when the prompt
            # only had the full name (the hint invites it), so the parts
            # must always be restorable.
            tokens = core.split()
            mapping.setdefault(part_placeholder(base, "first"), tokens[0])
            mapping.setdefault(part_placeholder(base, "last"), tokens[-1])

    # First / last names of every known person (this text or earlier calls
    # through the same factory), whether or not the detector flagged them.
    for token, entry in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if entry is _AMBIGUOUS:
            continue
        base, part = entry
        pattern = _word_re(token)
        if pattern.search(text):
            ph = part_placeholder(base, part)
            mapping[ph] = token
            text = pattern.sub(ph, text)

    # Remaining single-word names the detector found that aren't a known
    # person's first/last name: their own entity.
    for core in single:
        if not _word_re(core).search(text):
            continue
        base = factory.make("NAME", core)
        mapping[base] = core
        text = _word_re(core).sub(base, text)

    return text


# ---------------------------------------------------------------- recall helpers

_NAME = r"([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){0,2})"
_ROLES = (
    "manager|boss|colleague|coworker|co-worker|assistant|lead|director|VP|CEO|CTO|CFO|COO|founder|"
    "engineer|designer|recruiter|nurse|doctor|patient|client|customer|lawyer|attorney|accountant|"
    "landlord|tenant|intern|mentor|buddy|contact|champion|rep|representative|teammate|supervisor"
)
_CONTEXT_NAME_RES = [
    # "my manager Daniel Okafor", "our CTO Mei Tanaka"
    re.compile(r"\b(?i:my|our|your|their|his|her|the)\s+(?:" + _ROLES + r")\s*,?\s+" + _NAME),
    # "Dear Priya", "Hi Daniel Okafor", "Thanks, Omar", "Attn: Sofia Marchetti"
    re.compile(r"\b(?:Dear|Hi|Hello|Hey|Attn:?|Attention:?|Thanks|Regards|Cheers|cc:?)\s*,?\s+" + _NAME),
    # field labels at the start of a line: "Customer: Priya Nair", "- Employee: Omar Haddad"
    re.compile(r"(?m)^\s*[-*•]?\s*(?:Name|Customer|Client|Employee|Patient|Member|Contact|Owner|Manager|"
               r"Tenant|Candidate|Applicant|Signed|Signatory|Attn)\s*:\s*" + _NAME),
]
_NOT_NAMES = {
    "Team", "All", "Everyone", "Sir", "Madam", "There", "Again", "Folks", "Guys", "Customer", "Support",
    "Both", "You", "The", "A", "An", "Our", "Your", "My", "This", "That", "Re", "Fwd", "Hiring", "Sales",
    "Billing", "Management", "Hr", "Admin", "Legal", "Finance", "Engineering", "Ops", "Operations",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Today", "Tomorrow",
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December", "Please", "In", "On", "At", "For", "And", "Or", "But", "Is", "Was", "Will",
}


def context_name_spans(text: str) -> list[str]:
    """
    Names in positions where a person is almost certainly meant -- after a
    role ("my manager ..."), a greeting ("Dear ...") or a field label
    ("Customer: ..."). A safety net for detector misses: the 2026-09-27
    run showed GLiNER missing "my manager Daniel Okafor". Feed the result
    to apply_names() together with the detector's spans.
    """
    spans = []
    for pattern in _CONTEXT_NAME_RES:
        for m in pattern.finditer(text):
            words = [w for w in m.group(1).split() if w not in _NOT_NAMES]
            # stop at the first non-name word ("Priya Thanks" -> "Priya")
            name = []
            for w in m.group(1).split():
                if w in _NOT_NAMES:
                    break
                name.append(w)
            if name and words and name[0] == words[0]:
                spans.append(" ".join(name))
    return spans


def remember_entity(factory: Optional[PlaceholderFactory], label: str, span: str) -> None:
    """A client's factory remembers detected companies/codenames, so a later
    mention the detector misses is still replaced (the shared default
    factory never remembers)."""
    if factory is None or factory is default_factory():
        return
    if not hasattr(factory, "_known_entities"):
        factory._known_entities = {}
    factory._known_entities[span] = label


def apply_known_entities(text: str, mapping: dict, factory: Optional[PlaceholderFactory]) -> str:
    known = getattr(factory, "_known_entities", None) if factory is not None else None
    if not known:
        return text
    for span, label in sorted(known.items(), key=lambda kv: -len(kv[0])):
        pattern = _word_re(span)
        if pattern.search(text):
            ph = factory.make(label, span)
            mapping[ph] = span
            text = pattern.sub(ph, text)
    return text


# ---------------------------------------------------------------- not worth hiding

# Public AI/cloud services and their API hosts. Hiding "api.anthropic.com"
# from Anthropic protects nothing and cost answer quality: GLiNER tagged it
# as a company in a deploy script, so the model couldn't suggest
# ANTHROPIC_API_KEY or the anthropic-version header (2026-09-28).
PUBLIC_SERVICES = {
    "anthropic", "claude", "openai", "chatgpt", "gpt", "google", "gemini", "googleapis", "github", "gitlab",
    "aws", "amazon", "microsoft", "azure", "slack", "stripe", "twilio", "cloudflare", "vercel", "heroku",
    "docker", "kubernetes", "postgres", "postgresql", "mysql", "redis", "mongodb", "linux", "python",
}


def not_worth_hiding(span: str, label: str) -> bool:
    s = span.strip().lower()
    if label in ("CODENAME", "EMPLOYER") and len(s) <= 2:
        return True
    if label == "CODENAME" and re.fullmatch(r"v?\d+(\.\d+)*", s):  # "v1", "2.0" path/version segments
        return True
    parts = re.split(r"[\s./:-]+", s)
    return bool(parts) and all(p in PUBLIC_SERVICES or p in ("api", "www", "com", "io", "ai", "dev", "")
                               for p in parts) and any(p in PUBLIC_SERVICES for p in parts)
