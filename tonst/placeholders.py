"""
placeholders.py
---------------
Single source of truth for how redaction placeholders are built and
recognised. Every redaction backend (regex in redact.py, Ollama in
redact_llm.py, GLiNER in gliner_redact.py) asks a PlaceholderFactory for
the token that stands in for a sensitive value.

Two styles:

  "hash"      [[EMAIL_3f2a91c0]] -- the default. Deterministic: the same
              value always gets the same token, across calls and across
              processes, which is what provider-side prompt caching needs
              (see cache_structuring.py). The digest is an HMAC keyed with
              a secret that never leaves this machine. The earlier scheme
              (bare sha256 of the value) let anyone holding the prompt
              confirm a guess offline -- hash a suspected email or phone
              number and compare -- which undermines the whole point of
              redacting. With a keyed HMAC that check is impossible
              without the local key.

  "readable"  [[EMAIL_1]], [[NAME_2]] -- numbered per label, in order of
              first appearance, and stable for the lifetime of the
              factory (so a TonstClient keeps one numbering for a whole
              conversation). Easier for the model to reason about; carries
              no information about the value at all. Not stable across
              processes, so it can reduce prompt-cache hits for stable
              blocks that are redacted in separate processes.

The key for "hash" comes from, in order: the TONST_PLACEHOLDER_KEY
environment variable (any string), else ~/.tonst/placeholder.key
(created with 32 random bytes and 0600 permissions on first use), else --
if that file can't be written -- a random per-process key, with a warning,
since placeholders then stop being stable across processes.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
from typing import Optional

logger = logging.getLogger(__name__)

PLACEHOLDER_STYLES = ("hash", "readable")

DEFAULT_KEY_PATH = os.path.join(os.path.expanduser("~"), ".tonst", "placeholder.key")

# Well-formed placeholder in either style. Labels may contain digits and
# underscores (CREDIT_CARD, SSN_LIKE, IP_ADDRESS) -- the previous
# [A-Z]+_ pattern silently failed to recognise those labels.
# Optional ".first" / ".last": part of a person's name (see names.py).
PLACEHOLDER_STRICT_RE = re.compile(r"\[\[[A-Z][A-Z0-9_]*?_(?:[0-9a-f]{8}|[0-9]{1,6})(?:\.(?:first|last))?\]\]")
# Deliberately permissive: used to scan UNTRUSTED model output so that a
# mangled placeholder is still caught rather than waved through.
PLACEHOLDER_LOOSE_RE = re.compile(r"\[\[.*?\]\]")
# Same as STRICT, anchored, capturing the label.
PLACEHOLDER_LABEL_RE = re.compile(r"^\[\[([A-Z][A-Z0-9_]*?)_(?:[0-9a-f]{8}|[0-9]{1,6})(\.(?:first|last))?\]\]$")

_key_lock = threading.Lock()
_cached_key: Optional[bytes] = None


def _load_or_create_key(path: str = DEFAULT_KEY_PATH) -> bytes:
    env = os.environ.get("TONST_PLACEHOLDER_KEY")
    if env:
        return env.encode("utf-8")
    try:
        with open(path, "rb") as fh:
            key = fh.read()
        if len(key) >= 16:
            return key
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("tonst: could not read placeholder key at %s", path, exc_info=True)
    key = secrets.token_bytes(32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        return key
    except FileExistsError:
        # Another process created it between our read and write -- use theirs.
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        logger.warning(
            "tonst: could not persist placeholder key to %s; using a per-process key "
            "(placeholders will differ between processes, which reduces prompt-cache hits)",
            path,
        )
        return key


def local_key() -> bytes:
    global _cached_key
    if _cached_key is None:
        with _key_lock:
            if _cached_key is None:
                _cached_key = _load_or_create_key()
    return _cached_key


class PlaceholderFactory:
    def __init__(self, style: str = "hash", key: Optional[bytes] = None):
        if style not in PLACEHOLDER_STYLES:
            raise ValueError(f"placeholder style must be one of {PLACEHOLDER_STYLES}, got {style!r}")
        self.style = style
        self._key = key
        self._lock = threading.Lock()
        # readable style: (label, original) -> number, and next number per label
        self._assigned: dict[tuple[str, str], int] = {}
        self._next: dict[str, int] = {}

    def make(self, label: str, original: str) -> str:
        if self.style == "readable":
            with self._lock:
                n = self._assigned.get((label, original))
                if n is None:
                    n = self._next.get(label, 0) + 1
                    self._next[label] = n
                    self._assigned[(label, original)] = n
            return f"[[{label}_{n}]]"
        key = self._key if self._key is not None else local_key()
        digest = hmac.new(key, original.encode("utf-8"), hashlib.sha256).hexdigest()[:8]
        return f"[[{label}_{digest}]]"


# Sent ahead of any prompt that contains placeholders (TonstClient
# placeholder_hint=True, the default). Measured 2026-09-27 on Claude
# Sonnet 4.6: without it the model often rewrote [[NAME_1]] as NAME_1 or
# [NAME_1] or treated the tokens as unfilled template fields; with it,
# restoration failures went 4/10 -> 0/10 and must-have values 64% -> 100%.
PLACEHOLDER_HINT = (
    # The examples deliberately don't match a real placeholder ([[NAME_x]], not
    # [[NAME_1]]) so the note can never be confused with the data.
    "Note: tokens in double square brackets, like [[NAME_x]] or [[EMAIL_x]], stand in for private details. "
    "Treat each as the real value it represents and copy it into your answer exactly as written, "
    "brackets included, wherever that value belongs. [[NAME_x.first]] and [[NAME_x.last]] are the first and "
    "last name of the person [[NAME_x]]; use them where you'd use just a first or last name. "
    "Don't use double square brackets for anything else.\n\n"
)


_PH = r"\[\[(?:NAME|EMAIL|PHONE)_(?:[0-9a-f]{8}|\d{1,6})(?:\.(?:first|last))?\]\]"
_NAME_PH = r"\[\[NAME_(?:[0-9a-f]{8}|\d{1,6})(?:\.(?:first|last))?\]\]"
_CONTACT_PH = r"\[\[(?:EMAIL|PHONE)_(?:[0-9a-f]{8}|\d{1,6})\]\]"
# A contact placeholder that directly follows a person: "[[NAME_1]] ([[EMAIL_1]]@..., [[PHONE_1]])",
# "[[NAME_1]], [[EMAIL_1]]", "[[NAME_1]] on [[PHONE_1]]", "[[NAME_1]] <[[EMAIL_1]]>". Deliberately
# narrow: "Tell [[NAME_1]] to call [[PHONE_2]]" must NOT link the phone to that person.
_OWNED_RES = [
    re.compile(r"(" + _NAME_PH + r")\s*\(([^()\n]{0,120})\)"),
    re.compile(r"(" + _NAME_PH + r")\s*(?:<|,|:|\s(?:at|on|via|-))\s*(" + _CONTACT_PH + r")"),
]


def _base_name(ph: str) -> str:
    return re.sub(r"\.(?:first|last)\]\]$", "]]", ph)


def contact_owners(text: str) -> dict:
    """
    {contact placeholder: person placeholder} for emails/phones written right
    next to a person. Only links placeholders to placeholders -- no real
    value is involved. The 2026-09-27 run had a model tell a patient to call
    their own number, because it couldn't tell whose number it was.
    """
    owners: dict = {}
    for pattern in _OWNED_RES:
        for m in pattern.finditer(text):
            person = _base_name(m.group(1))
            for contact in re.findall(_CONTACT_PH, m.group(2)):
                owners.setdefault(contact, person)
    return owners


SECRET_HINT = (
    " [[SECRET_...]] tokens (e.g. [[SECRET_GITHUB_TOKEN_x]]) are passwords, keys or tokens of that kind the user "
    "pasted in plain text: treat them as exposed credentials (advise rotating them where relevant) and never "
    "repeat them."
)


def build_hint(text: str) -> str:
    """PLACEHOLDER_HINT plus, when known, which contact details belong to whom,
    and -- when secrets were withheld -- that they are exposed credentials (the
    model can then still give the "rotate this key" advice it gives on the
    raw text; missing it cost ~2 points per engineering case, 2026-09-28)."""
    owners = contact_owners(text)
    secret_note = SECRET_HINT if "[[SECRET_" in text else ""
    if not owners:
        if not secret_note:
            return PLACEHOLDER_HINT
        return PLACEHOLDER_HINT.rstrip("\n") + secret_note + "\n\n"
    by_person: dict = {}
    for contact, person in owners.items():
        by_person.setdefault(person, []).append(contact)
    parts = []
    for person, contacts in by_person.items():
        parts.append(f"{' and '.join(contacts)} {'is' if len(contacts) == 1 else 'are'} {person}'s own")
    # Worded as ownership, not just association: with "belongs to", Claude
    # still offered a customer's own email and phone as the support contact
    # (2026-09-27, 12-case run).
    return (PLACEHOLDER_HINT.rstrip("\n") + " Contact details: " + "; ".join(parts)
            + " -- never present them as anyone else's." + secret_note + "\n\n")


def contains_placeholder(text: str) -> bool:
    return bool(text) and PLACEHOLDER_STRICT_RE.search(text) is not None


_default_factory = PlaceholderFactory("hash")


def default_factory() -> PlaceholderFactory:
    return _default_factory


def make_placeholder(label: str, original: str, factory: Optional[PlaceholderFactory] = None) -> str:
    return (factory or _default_factory).make(label, original)
