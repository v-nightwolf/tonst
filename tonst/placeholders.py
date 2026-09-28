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
# The note is built from parts so a request only pays for the lines that
# apply to it (2026-09-28: the full note was ~110 tokens on every masked
# request). The examples deliberately don't match a real placeholder
# ([[NAME_x]], not [[NAME_1]]) so the note can never be confused with the data.
HINT_BASE = (
    "Note: tokens like [[EMAIL_x]] stand for real private values. Treat each as that value and copy it "
    "exactly, brackets included, where it belongs. Don't use double square brackets otherwise."
)
# Only when the text has a person placeholder: lets the model write
# "Hi [[NAME_1.first]]" instead of repeating the full name everywhere.
# Wording matters: "use them where only one is needed" (2026-09-28 short-note
# run) made Claude shorten owners to first names and write "Dr. [[NAME_1.first]]",
# dropping must-have full names; the original "where you'd use just a first or
# last name" didn't.
HINT_NAMES = " [[NAME_x.first]] and [[NAME_x.last]] are [[NAME_x]]'s first and last name; use them only where you'd use just a first or last name."
# The fullest fixed note (base + names). Kept as a name for callers and tests.
PLACEHOLDER_HINT = HINT_BASE + HINT_NAMES + "\n\n"


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
    " [[SECRET_...]] tokens are passwords or keys the user pasted: treat them as exposed credentials "
    "(advise rotating them where relevant) and never repeat them."
)


def hint_parts(text: str) -> tuple:
    """
    (fixed, variable) parts of the note for `text`.
    fixed    -- the base line, plus the names line when a person placeholder
                appears and the secrets line when a secret was withheld. It only
                changes when a new kind of value first appears, so in a chat it
                can sit in the (cached) system message.
    variable -- which contact details belong to whom ("" if none known). It
                grows as a conversation goes on, so chats send it with the
                latest message instead of in the cached prefix.
    """
    fixed = HINT_BASE
    if re.search(_NAME_PH, text):
        fixed += HINT_NAMES
    if "[[SECRET_" in text:
        fixed += SECRET_HINT
    owners = contact_owners(text)
    if not owners:
        return fixed, ""
    by_person: dict = {}
    for contact, person in owners.items():
        by_person.setdefault(person, []).append(contact)
    parts = [f"{' and '.join(cs)} {'is' if len(cs) == 1 else 'are'} {person}'s own" for person, cs in by_person.items()]
    # Worded as ownership, not just association: with "belongs to", Claude
    # still offered a customer's own email and phone as the support contact
    # (2026-09-27, 12-case run). "including the sender's": in the 2026-09-28
    # run Claude signed a sales email with the recipient's name and email.
    return fixed, "Contact details: " + "; ".join(parts) + " -- never present them as anyone else's, including the sender's."


def build_hint(text: str) -> str:
    """The whole note for a single prompt: only the lines that apply to `text`."""
    fixed, variable = hint_parts(text)
    return fixed + (" " + variable if variable else "") + "\n\n"


def strip_hint(text: str) -> str:
    """Remove a leading note added by build_hint() (used by tests and the benchmark's fake model)."""
    if text.startswith(HINT_BASE):
        i = text.find("\n\n")
        return text[i + 2:] if i != -1 else ""
    return text


def contains_placeholder(text: str) -> bool:
    return bool(text) and PLACEHOLDER_STRICT_RE.search(text) is not None


_default_factory = PlaceholderFactory("hash")


def default_factory() -> PlaceholderFactory:
    return _default_factory


def make_placeholder(label: str, original: str, factory: Optional[PlaceholderFactory] = None) -> str:
    return (factory or _default_factory).make(label, original)
