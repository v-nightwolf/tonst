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
revealing the original. The hash is an HMAC keyed with a secret that
stays on this machine (see placeholders.py), so a provider can't confirm
a guessed value by hashing it. placeholders.py also offers a "readable"
style ([[EMAIL_1]]).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .placeholders import PLACEHOLDER_STRICT_RE, PlaceholderFactory, make_placeholder

# Secrets: provider API keys, cloud credentials, tokens and private keys.
# Always on -- a leaked key is worse than a leaked email. Patterns with a
# named group "val" redact only that group (the value in `api_key=...`),
# leaving the surrounding text readable for the model.
# (label, pattern). The label names the KIND of credential -- [[SECRET_GITHUB_TOKEN_1]] --
# so the model can still give service-specific advice ("revoke it under GitHub >
# Settings > Developer settings"); with a bare [[SECRET_1]] it could only say
# "rotate it wherever it came from" (hold-out gh_token_leak, 2026-09-28). The
# kind is visible in the raw key's prefix anyway; the key itself never is.
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("SECRET_PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")),
    ("SECRET_ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("SECRET_OPENAI_KEY", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("SECRET_AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("SECRET_GITHUB_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("SECRET_GITHUB_TOKEN", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("SECRET_SLACK_TOKEN", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("SECRET_GOOGLE_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("SECRET_JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("SECRET", re.compile(
        # (?<![a-z]) rather than \b so env-style names match too: DB_PASSWORD, STRIPE_API_KEY
        r"(?i)(?<![a-z])(?:api[_-]?key|secret(?:[_-]?key)?|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|pwd)"
        r"\b[\"']?\s*[:=]\s*[\"']?(?P<val>[^\s\"',;]{8,})"
    )),
]

# Postal addresses (always on). Regex, not a model: the 2026-09-27
# benchmark found GLiNER's labels don't cover them and 100% leaked. Covers
# number-first streets (42 MG Road, 1180 Folsom Street, CA 94103; 221B Baker
# Street, London NW1 6XE), street-first forms (Rua Augusta 1520;
# Friedrichstraße 88, 10117 Berlin), French/Italian/Spanish number + Rue/Via,
# and Indian Flat/House ... City PIN. Over-matching is harmless (the text is
# restored); unusual formats can still be missed.
STREET = r"(?:Road|Rd|Street|St|Lane|Ln|Avenue|Ave|Boulevard|Blvd|Drive|Dr|Way|Place|Pl|Court|Ct|Terrace|Close|Crescent|Highway|Hwy|Parkway|Square|Sq|Marg|Nagar|Circle|Row)"
POST = r"(?:\s*,?\s*(?:[A-Z]{2}\s+)?\d{5}(?:-\d{3,4})?\b|\s*,?\s*\d{6}\b|\s*,?\s*\d{3}\s\d{3}\b|\s*,?\s*[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}\b)"
WORD = r"[A-ZÀ-Þ][a-zà-ÿ][A-Za-zà-ÿ'.\-]*"
PLACE = r"(?:,\s*" + WORD + r"(?:\s+" + WORD + r"){0,3})"
UNIT = r"(?:(?:Flat|Apt\.?|Apartment|Unit|Suite|House|Plot|Door)\s*(?:No\.?\s*)?[\w\-/]+,?\s+)"
ADDRESS_PATTERNS: list[re.Pattern] = [
    # [unit] 42 MG Road[, Area][, City] [postcode]
    re.compile(UNIT + r"?\d{1,5}[A-Za-z]?(?:[/-]\d{1,4})?,?\s+(?:[A-Z][\w'.\-]*\s+){1,4}" + STREET + r"\b\.?(?:\s+(?:North|South|East|West))?" + PLACE + r"{0,3}" + POST + r"?"),
    # Rua Augusta 1520 / Avenida Paulista 900 / Friedrichstraße 88 [, 10117 Berlin | , São Paulo 01304-001]
    re.compile(r"(?:\b(?:Rua|Avenida|Av\.|Calle|Via|Rue|Boulevard)\s+(?:[A-Z][\w'\-]*\s+){1,3}|\b[A-Z][a-zäöüß]+(?:straße|strasse|str\.|weg|platz|allee|gasse)\s+)\d{1,5}[a-z]?" + r"(?:,\s*(?:\d{4,5}(?:-\d{3})?\s+)?[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,2})?" + POST + r"?"),
    # 17 Rue de Rivoli, Paris
    re.compile(r"\b\d{1,5},?\s+(?:Rue|Avenue|Boulevard|Via|Calle|Place)\s+(?:[a-zà-ÿ']+\s+){0,2}" + WORD + PLACE + r"{0,2}" + POST + r"?"),
    # Flat 9B, Palm Residency, Sector 45, Gurugram 122003 (Indian styles ending in a 6-digit PIN)
    re.compile(r"\b(?:Flat|House|Plot|Door)\s*(?:No\.?\s*)?[\w\-/]+(?:,\s*[\w'.\- ]{2,40}?){1,5},\s*[A-Z][a-z]+\s*[-–]?\s*\d{6}\b"),
]


# Opt-in categories (pass extra_categories=... or TonstClient(extra_redaction=...)).
# Off by default because they can hurt answer quality: a model can't
# total up invoice amounts it can't see. The answer-quality benchmark is
# what decides whether they should become defaults.
EXTRA_PATTERNS: dict[str, list[re.Pattern]] = {
    "ACCOUNT_ID": [
        # the number after a keyword: "customer #928381", "policy number 48210"
        re.compile(
            r"(?i)\b(?:account|acct|customer|cust|client|member|policy|invoice|order|ticket|employee|patient|case|reference|ref)"
            r"(?:\s*(?:no\.?|number|num|id))?\s*[:#]?\s*#?(?P<val>[A-Z0-9][A-Z0-9-]{3,})\b"
        ),
        # self-describing codes on their own: INV-22243, PO-7781, CS-20931
        re.compile(r"\b(?:INV|ORD|ACC|ACCT|CUST|PO|TKT|CS|CASE|REF|SR|INC)-?\d{3,}\b"),
        # "#" followed by 4+ digits: parcel #777094
        re.compile(r"(?<![\w#])#(?P<val>\d{4,})\b"),
    ],
    "MONEY": [
        re.compile(
            r"(?:[$€£₹¥]|\bRs\.?\s?|\bINR\s?|\bUSD\s?|\bEUR\s?|\bGBP\s?)\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|m|M|bn|million|billion|lakh|crore)\b)?"
            r"|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|INR|dollars|euros|pounds|rupees)\b"
        ),
    ],
}
EXTRA_CATEGORIES = tuple(EXTRA_PATTERNS)

# Order matters: more specific patterns first so they aren't partially
# swallowed by a looser pattern later in the list.
PATTERNS: dict[str, re.Pattern] = {
    # Domain = dot-separated labels ending in a letter/digit, so a sentence's
    # full stop isn't swallowed ("...@northpeakcap.com." made a different
    # [[DOMAIN]] than "...@northpeakcap.com" and broke grouping, 2026-09-28).
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+"),
    # International numbers written with a leading + and 2-5 digit groups
    # (+91 98450 21733, +49 151 2384 9921). Runs before CREDIT_CARD, which
    # would otherwise swallow the longer ones and mislabel them. Emitted
    # with the PHONE label.
    "PHONE_INTL": re.compile(r"\+\d{1,3}(?:[-.\s]?\(?\d{2,5}\)?){2,4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    # US "(415) 555-0144": the generic pattern below starts at the digits and
    # left the "(" outside the placeholder; the model then dropped it
    # ("415) 555-0144" in an answer, hold-out 2026-09-28).
    "PHONE_PAREN": re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]\d{4}\b"),
    "PHONE": re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    # Trunk-prefix numbers written in two groups ("090000 12345", "0161 496 0000"
    # is covered above). Found by the hold-out set on 2026-09-28 -- so this
    # rule is no longer blind to that set.
    "PHONE_TRUNK": re.compile(r"(?<![\d-])0\d{3,5}[\s-]\d{5,6}\b"),
    "SSN_LIKE": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


_LABEL_FOR = {"PHONE_INTL": "PHONE", "PHONE_TRUNK": "PHONE", "PHONE_PAREN": "PHONE"}


def _placeholder_for(label: str, original: str, factory: Optional[PlaceholderFactory] = None) -> str:
    return make_placeholder(label, original, factory)


def restore_placeholders(text: str, mapping: dict[str, str], neutralize_unknown: bool = True) -> str:
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
    if not mapping:
        return text
    # Split emails ([[EMAIL_1]]@[[DOMAIN_1]], see redact()) collapse back to
    # the single EMAIL placeholder, which maps to the whole address -- so a
    # model that copies only [[EMAIL_1]] still restores a complete email.
    text = _SPLIT_EMAIL_RE.sub(lambda m: "[[" + m.group(1) + "]]", text)
    # Longest placeholders first, so [[NAME_1.first]] is never broken up
    # by [[NAME_1]] and NAME_10 is never matched as NAME_1.
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    for _ in range(5):
        new_text = text
        for placeholder, original in items:
            if placeholder not in new_text:
                continue
            new_text = _dedupe_leading_word(new_text, placeholder, original)
            new_text = new_text.replace(placeholder, original)
        if new_text == text:
            break
        text = new_text
    # A part the model made up for a person we only had one token for
    # ("[[NAME_3.first]]" when the text just said "Rahul"), seen in the
    # 2026-09-27 run: derive it from the whole value instead of leaving
    # the token in front of the user.
    def _part(m):
        base = "[[" + m.group(2) + "]]"
        if base not in mapping:
            return m.group(0)
        words = mapping[base].split()
        if m.group(3) == "last" and len(words) < 2:
            # A one-word name has no separate last name: "[[NAME_1.first]]
            # [[NAME_1.last]]" for "Sofia" must not become "Sofia Sofia".
            return ""
        return m.group("sp") + (words[0] if m.group(3) == "first" else words[-1])

    text = _PART_RE.sub(_part, text)
    # Placeholder-shaped tokens that are in no mapping were invented by the
    # model (e.g. "From: [[EMAIL_2]]" when no second email existed, seen in
    # the 2026-09-28 run). Show a neutral blank instead of a raw token.
    # Callers restoring with a PARTIAL mapping (TonstClient.query_messages()
    # restores twice) pass neutralize_unknown=False for the first pass.
    known_inner = {p[2:-2] for p in mapping if p.startswith("[[")}
    if neutralize_unknown:
        text = _INVENTED_RE.sub(
            lambda m: m.group(0) if m.group(1) in known_inner else "[" + m.group(2).lower().replace("_", " ") + "]",
            text,
        )
    # Second, tolerant pass for placeholders the model reformatted instead
    # of copying: [NAME_1], NAME_1, [[ NAME_1 ]]. Measured on Claude
    # Sonnet 4.6 (2026-09-27): without a hint it dropped one or both
    # brackets in 4 of 10 answers, which left raw tokens in front of the
    # user. Only exact inner ids from `mapping` are matched, on word
    # boundaries, so NAME_1 never matches inside NAME_10 or NAME_1.first.
    for placeholder, original in items:
        if not (placeholder.startswith("[[") and placeholder.endswith("]]")):
            continue
        inner = placeholder[2:-2]
        if inner not in text:
            continue
        x = re.escape(inner)
        pattern = re.compile(
            r"\[\[\s*" + x + r"\s*\]\]"                                        # [[ NAME_1 ]]
            r"|\[" + x + r"\]"                                                     # [NAME_1]
            r"|(?<![A-Za-z0-9_.])" + x + r"(?![A-Za-z0-9_]|\.(?:first|last)\b)"    # NAME_1
        )
        text = pattern.sub(lambda _m, o=original: o, text)
    return text


_INVENTED_RE = re.compile(r"\[\[\s*(([A-Z][A-Z0-9_]*?)_(?:[0-9a-f]{8}|\d{1,6})(?:\.(?:first|last))?)\s*\]\]")
_PART_RE = re.compile(r"(?P<sp>\s?)\[\[\s*(NAME_(?:[0-9a-f]{8}|\d{1,6}))\.(first|last)\s*\]\]")

# [[EMAIL_x]]@[[DOMAIN_y]], also with the brackets dropped by the model.
_SPLIT_EMAIL_RE = re.compile(
    r"\[{0,2}\s*(EMAIL_(?:[0-9a-f]{8}|\d{1,6}))\s*\]{0,2}\s*@\s*\[{0,2}\s*DOMAIN_(?:[0-9a-f]{8}|\d{1,6})\s*\]{0,2}"
)


def _dedupe_leading_word(text: str, placeholder: str, original: str) -> str:
    """
    "Project [[CODENAME_1]]" where the value is "Project Marigold" would
    restore to "Project Project Marigold" (seen in the 2026-09-27 run).
    Drop the repeated leading word before substituting.
    """
    words = original.split()
    if len(words) < 2:
        return text
    first = re.escape(words[0].rstrip(".,"))
    return re.sub(r"\b" + first + r"\.?\s+(?=" + re.escape(placeholder) + ")", "", text, flags=re.IGNORECASE)


# Tail of a streamed chunk that could still become (part of) a placeholder:
# an unfinished "[[...", a bare "NAME_1" / "EMAIL_1@DOMAIN_" being typed, or
# a complete [[EMAIL_n]] that may yet be followed by "@[[DOMAIN_n]]".
_STREAM_HOLD_RE = re.compile(r"(?:\[\[?[^\[\]\n]{0,60}\]?|[A-Za-z0-9_.@\[\]]{1,80})$")
_MAX_HOLD = 160


class StreamRestorer:
    """
    Restore placeholders in a streamed answer, chunk by chunk, without ever
    showing a half-received placeholder or restoring one that was split
    across chunks.

        restorer = result.stream_restorer()      # or StreamRestorer(mapping)
        for chunk in stream:
            print(restorer.feed(chunk), end="")
        print(restorer.flush(), end="")

    It holds back only the tail that could still turn into a placeholder
    (at most a few dozen characters), so text appears with almost no delay.
    Known limit: "Project " already shown before "[[CODENAME_1]]" arrives
    can't be un-shown, so the leading-word de-duplication of
    restore_placeholders() doesn't apply across chunk boundaries.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self._buf = ""

    def feed(self, chunk: str) -> str:
        self._buf += chunk or ""
        m = _STREAM_HOLD_RE.search(self._buf)
        hold = len(m.group(0)) if m else 0
        # Only keep holding while the tail can still be a placeholder; a
        # plain word ends at the next space or punctuation anyway.
        hold = min(hold, _MAX_HOLD)
        cut = len(self._buf) - hold
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return restore_placeholders(ready, self.mapping) if ready else ""

    def flush(self) -> str:
        out, self._buf = self._buf, ""
        return restore_placeholders(out, self.mapping) if out else ""


@dataclass
class RedactionResult:
    redacted_text: str
    # Maps placeholder token -> original value, kept ONLY in memory on
    # the local machine. Never sent to the cloud model or logged.
    mapping: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        """Re-insert real values into a model response that may echo placeholders."""
        return restore_placeholders(text, self.mapping)

    def stream_restorer(self) -> "StreamRestorer":
        """For streamed answers -- see StreamRestorer."""
        return StreamRestorer(self.mapping)


def _check_categories(extra_categories: Iterable[str]) -> tuple:
    extra = tuple(c.upper() for c in (extra_categories or ()))
    unknown = [c for c in extra if c not in EXTRA_PATTERNS]
    if unknown:
        raise ValueError(f"unknown redaction categories {unknown}; choose from {list(EXTRA_CATEGORIES)}")
    return extra


EMAIL_STYLES = ("split", "whole")


def redact(
    text: str,
    placeholders: Optional[PlaceholderFactory] = None,
    extra_categories: Iterable[str] = (),
    email_style: str = "split",
) -> RedactionResult:
    """
    Regex redaction. `placeholders` picks the placeholder style (default:
    keyed-hash, see placeholders.py); `extra_categories` switches on
    opt-in detectors from EXTRA_PATTERNS ("ACCOUNT_ID", "MONEY").
    """
    extra = _check_categories(extra_categories)
    if email_style not in EMAIL_STYLES:
        raise ValueError(f"email_style must be one of {EMAIL_STYLES}, got {email_style!r}")
    mapping: dict[str, str] = {}
    result_text = text

    def _apply(label: str, pattern: re.Pattern, text_in: str) -> str:
        def _sub(match: re.Match) -> str:
            whole = match.group(0)
            has_val = "val" in pattern.groupindex and match.group("val") is not None
            original = match.group("val") if has_val else whole
            # Never re-redact (part of) an existing placeholder.
            if "[[" in whole or "]]" in whole:
                return whole
            digits_only = re.sub(r"\D", "", original)
            # Skip short numeric noise being misfired as a card/phone number
            if label in ("CREDIT_CARD", "PHONE", "SSN_LIKE") and len(digits_only) < 7:
                return whole
            # An "ID" with no digit is almost always an ordinary word
            # ("order status", "customer service").
            if label == "ACCOUNT_ID" and not digits_only:
                return whole
            placeholder = _placeholder_for(label, original, placeholders)
            mapping[placeholder] = original
            if label == "EMAIL" and email_style == "split" and "@" in original:
                # "split" (default): [[EMAIL_1]]@[[DOMAIN_1]]. Both the local
                # part and the domain are hidden, but addresses at the same
                # domain share one [[DOMAIN_n]], so the model can still group
                # or compare them (the 2026-09-27 run: "group these emails by
                # domain" went 10 -> 1 with whole-email placeholders).
                # [[EMAIL_1]] maps to the WHOLE address, so it restores
                # correctly whether or not the model keeps the @[[DOMAIN]].
                domain = original.rsplit("@", 1)[1]
                domain_ph = _placeholder_for("DOMAIN", domain.lower(), placeholders)
                mapping[domain_ph] = domain
                return placeholder + "@" + domain_ph
            if has_val:
                start = match.start("val") - match.start()
                end = match.end("val") - match.start()
                return whole[:start] + placeholder + whole[end:]
            return placeholder

        return pattern.sub(_sub, text_in)

    for label, pattern in SECRET_PATTERNS:
        result_text = _apply(label, pattern, result_text)
    result_text = _apply("EMAIL", PATTERNS["EMAIL"], result_text)
    for pattern in ADDRESS_PATTERNS:
        result_text = _apply("ADDRESS", pattern, result_text)
    for label in extra:
        for pattern in EXTRA_PATTERNS[label]:
            result_text = _apply(label, pattern, result_text)
    for key, pattern in PATTERNS.items():
        if key == "EMAIL":
            continue
        result_text = _apply(_LABEL_FOR.get(key, key), pattern, result_text)

    return RedactionResult(redacted_text=result_text, mapping=mapping)


def redact_with_llm(
    text: str,
    llm_redactor,
    placeholders: Optional[PlaceholderFactory] = None,
    extra_categories: Iterable[str] = (),
    email_style: str = "split",
) -> RedactionResult:
    """
    Two-stage redaction: fast, deterministic regex first (catches emails,
    phones, cards, IPs), then the local-LLM pass on what's left (catches
    free-text names, addresses, employers, codenames -- see redact_llm.py).

    `llm_redactor` is a `tonst.redact_llm.LLMRedactor` instance, passed in
    rather than constructed here so callers control the model/timeout and
    tests can inject a fake one. If the local model isn't available, this
    silently degrades to regex-only redaction -- never raises.
    """
    regex_result = redact(text, placeholders=placeholders, extra_categories=extra_categories, email_style=email_style)
    # Only pass the factory when one was given, so duck-typed redactors
    # with the plain redact(text) signature (e.g. test fakes) keep working.
    if placeholders is not None:
        llm_result = llm_redactor.redact(regex_result.redacted_text, placeholders=placeholders)
    else:
        llm_result = llm_redactor.redact(regex_result.redacted_text)

    combined_mapping = {**regex_result.mapping, **llm_result.mapping}
    return RedactionResult(redacted_text=llm_result.redacted_text, mapping=combined_mapping)
