"""
Phase 3 tests: one entity per person (first/last-name parts), split emails,
restore de-duplication, secret handling. Each maps to a failure seen in the
100-prompt benchmark run of 2026-09-27.
"""

import logging

import pytest

import tonst.gliner_redact as gr
from tonst import TonstClient
from tonst.local_model import placeholders_preserved
from tonst.names import apply_names, split_title
from tonst.placeholders import PLACEHOLDER_HINT, PLACEHOLDER_STRICT_RE, PlaceholderFactory, default_factory
from tonst.redact import redact, restore_placeholders
from tonst.savings_log import redacted_types_from_mapping


# ---------------------------------------------------------------- names

def test_one_entity_per_person_with_parts():
    f = PlaceholderFactory("readable")
    mapping = {}
    text = "Omar Haddad met Mei Tanaka. Later, Omar called Dr. Tanaka."
    out = apply_names(text, ["Omar Haddad", "Mei Tanaka", "Omar"], mapping, f)
    assert out == "[[NAME_1]] met [[NAME_2]]. Later, [[NAME_1.first]] called Dr. [[NAME_2.last]]."
    assert restore_placeholders(out, mapping) == text


def test_first_name_found_even_if_detector_missed_it():
    f = PlaceholderFactory("readable")
    mapping = {}
    out = apply_names("Daniel Okafor wrote in. Thanks, Daniel!", ["Daniel Okafor"], mapping, f)
    assert out == "[[NAME_1]] wrote in. Thanks, [[NAME_1.first]]!"


def test_ambiguous_first_name_gets_its_own_entity():
    f = PlaceholderFactory("readable")
    mapping = {}
    out = apply_names("Omar Haddad and Omar Khan joined. Omar spoke first.",
                      ["Omar Haddad", "Omar Khan", "Omar"], mapping, f)
    assert "[[NAME_1]] and [[NAME_2]] joined." in out
    assert out.endswith("[[NAME_3]] spoke first.")


def test_titles_stay_visible():
    assert split_title("Dr. Mei Tanaka") == ("Dr. ", "Mei Tanaka")
    f = PlaceholderFactory("readable")
    mapping = {}
    assert apply_names("Ask Dr. Mei Tanaka", ["Dr. Mei Tanaka"], mapping, f) == "Ask Dr. [[NAME_1]]"


def test_client_factory_remembers_people_across_calls():
    f = PlaceholderFactory("readable")
    apply_names("Omar Haddad joined", ["Omar Haddad"], {}, f)
    mapping = {}
    assert apply_names("Thanks Omar", [], mapping, f) == "Thanks [[NAME_1.first]]"


def test_default_factory_does_not_link_unrelated_calls():
    apply_names("Priya Malhotra joined", ["Priya Malhotra"], {}, default_factory())
    mapping = {}
    assert apply_names("Priya called", [], mapping, default_factory()) == "Priya called"
    assert mapping == {}


def test_names_do_not_touch_words_inside_other_words():
    f = PlaceholderFactory("readable")
    out = apply_names("Grace Lee has a grace period; Leeds office.", ["Grace Lee"], {}, f)
    assert out == "[[NAME_1]] has a grace period; Leeds office."


class _FakeGliner:
    def __init__(self, entities):
        self.entities = entities

    def predict_entities(self, text, labels, threshold=0.5):
        return [e for e in self.entities if e["text"] in text]


def test_gliner_backend_uses_person_entities(monkeypatch):
    fake = _FakeGliner([{"text": "Omar Haddad", "label": "person name"},
                        {"text": "Omar", "label": "person name"},
                        {"text": "Veltrix Logistics", "label": "company name"}])
    monkeypatch.setattr(gr, "_MODEL_CACHE", {"fake": fake})
    red = gr.GlinerRedactor(model="fake")
    f = PlaceholderFactory("readable")
    res = red.redact("Omar Haddad of Veltrix Logistics. Omar will call.", placeholders=f)
    assert res.redacted_text == "[[NAME_1]] of [[EMPLOYER_1]]. [[NAME_1.first]] will call."


def test_client_gets_first_name_right_end_to_end(monkeypatch):
    fake = _FakeGliner([{"text": "Kavya Iyer", "label": "person name"}])
    monkeypatch.setattr(gr, "_MODEL_CACHE", {"urchade/gliner_medium-v2.1": fake})
    sent = []

    def model(prompt):
        sent.append(prompt)
        return "Hi [[NAME_1.first]], thanks for the demo. -- sent to [[NAME_1]]"

    client = TonstClient(call_fn=model, redaction_backend="gliner", placeholder_style="readable")
    out, rep = client.query("Write to Kavya Iyer about the demo.")
    assert "Kavya" not in sent[0]
    assert out == "Hi Kavya, thanks for the demo. -- sent to Kavya Iyer"
    assert rep.redacted_types == {"NAME": 1}


# ---------------------------------------------------------------- restore

@pytest.mark.parametrize("answer", ["Hi [[NAME_1.first]]", "Hi NAME_1.first", "Hi [NAME_1.first]"])
def test_restore_parts(answer):
    mapping = {"[[NAME_1]]": "Omar Haddad", "[[NAME_1.first]]": "Omar"}
    assert restore_placeholders(answer, mapping) == "Hi Omar"


def test_restore_full_and_part_together():
    mapping = {"[[NAME_1]]": "Omar Haddad", "[[NAME_1.first]]": "Omar"}
    assert restore_placeholders("NAME_1 (NAME_1.first)", mapping) == "Omar Haddad (Omar)"


def test_restore_does_not_repeat_leading_word():
    mapping = {"[[CODENAME_1]]": "Project Marigold"}
    assert restore_placeholders("Status of Project [[CODENAME_1]]:", mapping) == "Status of Project Marigold:"
    assert restore_placeholders("Status of [[CODENAME_1]]:", mapping) == "Status of Project Marigold:"


# ---------------------------------------------------------------- emails

def test_split_emails_share_domain_and_restore_whole():
    f = PlaceholderFactory("readable")
    res = redact("a.b@veltrix.io, c.d@veltrix.io, e@luma.in", placeholders=f)
    assert res.redacted_text == ("[[EMAIL_1]]@[[DOMAIN_1]], [[EMAIL_2]]@[[DOMAIN_1]], "
                                 "[[EMAIL_3]]@[[DOMAIN_2]]")
    for answer, expected in [
        ("[[EMAIL_1]]@[[DOMAIN_1]]", "a.b@veltrix.io"),
        ("write to [[EMAIL_1]]", "write to a.b@veltrix.io"),
        ("EMAIL_2@DOMAIN_1", "c.d@veltrix.io"),
        ("[[DOMAIN_1]] has 2 contacts", "veltrix.io has 2 contacts"),
    ]:
        assert res.restore(answer) == expected


def test_whole_email_style_still_available():
    res = redact("mail a@b.com", placeholders=PlaceholderFactory("readable"), email_style="whole")
    assert res.redacted_text == "mail [[EMAIL_1]]"
    with pytest.raises(ValueError):
        redact("x", email_style="pieces")


def test_counts_are_per_entity_not_per_part():
    mapping = {"[[EMAIL_1]]": "a@b.com", "[[DOMAIN_1]]": "b.com", "[[NAME_1]]": "Omar Haddad",
               "[[NAME_1.first]]": "Omar"}
    assert redacted_types_from_mapping(mapping) == {"EMAIL": 1, "NAME": 1}


# ---------------------------------------------------------------- secrets

KEY = "sk-ant-api03-" + "A" * 40


def test_secrets_counted_and_warned(caplog):
    client = TonstClient(call_fn=lambda p: "ok")
    with caplog.at_level(logging.WARNING, logger="tonst.client"):
        _, rep = client.query(f"why does {KEY} fail")
    assert rep.secrets_withheld == 1
    assert "NOT sent" in caplog.text


def test_secret_not_restored_when_asked():
    client = TonstClient(call_fn=lambda p: p.split("\n\n")[-1], restore_secrets=False)
    out, _ = client.query(f"Rewrite: key {KEY} leaked")
    assert KEY not in out and "[REDACTED]" in out


def test_secret_notice_appended():
    client = TonstClient(call_fn=lambda p: "Here's the fix.", secret_notice=True)
    out, _ = client.query(f"config: api_key={KEY}")
    assert out.startswith("Here's the fix.") and "contained 1 secret" in out
    out2, _ = client.query("no secrets here")
    assert out2 == "Here's the fix."


# ---------------------------------------------------------------- plumbing

def test_hint_explains_parts():
    assert "[[NAME_x.first]]" in PLACEHOLDER_HINT
    assert not PLACEHOLDER_STRICT_RE.search(PLACEHOLDER_HINT)


def test_part_placeholders_recognised_by_guards():
    assert PLACEHOLDER_STRICT_RE.fullmatch("[[NAME_1.first]]")
    assert PLACEHOLDER_STRICT_RE.fullmatch("[[NAME_3f2a91c0.last]]")
    original = "Ask [[NAME_1.first]] and [[EMAIL_1]]@[[DOMAIN_1]] about the renewal next week please."
    assert placeholders_preserved(original, "Ask [[NAME_1.first]], [[EMAIL_1]]@[[DOMAIN_1]] re renewal.")
    assert not placeholders_preserved(original, "Ask [[NAME_1.firstname]], [[EMAIL_1]]@[[DOMAIN_1]] re renewal.")


# ---------------------------------------------------------------- addresses

@pytest.mark.parametrize("addr", [
    "42 MG Road, Indiranagar, Bengaluru 560038",
    "1180 Folsom Street, San Francisco, CA 94103",
    "221B Baker Street, London NW1 6XE",
    "Rua Augusta 1520, São Paulo 01304-001",
    "Friedrichstraße 88, 10117 Berlin",
    "Flat 9B, Palm Residency, Sector 45, Gurugram 122003",
    "Unit 5, 88 King Street West, Toronto",
])
def test_addresses_hidden_and_restored(addr):
    text = f"Ship to {addr} please."
    res = redact(text, placeholders=PlaceholderFactory("readable"))
    assert res.redacted_text == "Ship to [[ADDRESS_1]] please."
    assert res.restore(res.redacted_text) == text


@pytest.mark.parametrize("text", [
    "Refund 149 rupees on 3 September", "The order ships in 3 business days",
    "version 3 Way handshake", "Section 4 Street performers",
])
def test_non_addresses_untouched(text):
    assert redact(text).redacted_text == text


# ---------------------------------------------------------------- recall helpers

from tonst.names import context_name_spans  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("Summarise this for my manager Daniel Okafor in 4 bullets", ["Daniel Okafor"]),
    ("Our CTO Mei Tanaka said", ["Mei Tanaka"]),
    ("- Customer: Priya Nair, Veltrix", ["Priya Nair"]),
    ("Dear Priya,", ["Priya"]),
    ("ask my manager Daniel Okafor Monday", ["Daniel Okafor"]),
    ("Hi Team, thanks", []),
    ("Attn: The Billing Team", []),
])
def test_context_names(text, expected):
    assert context_name_spans(text) == expected


def test_context_names_used_by_gliner_backend(monkeypatch):
    monkeypatch.setattr(gr, "_MODEL_CACHE", {"fake": _FakeGliner([])})  # detector finds nothing
    res = gr.GlinerRedactor(model="fake").redact(
        "Summarise for my manager Daniel Okafor. Daniel needs it today.", placeholders=PlaceholderFactory("readable"))
    assert res.redacted_text == "Summarise for my manager [[NAME_1]]. [[NAME_1.first]] needs it today."


def test_client_remembers_companies_across_calls(monkeypatch):
    fake = _FakeGliner([{"text": "Brightwell Health", "label": "company name"}])
    monkeypatch.setattr(gr, "_MODEL_CACHE", {"urchade/gliner_medium-v2.1": fake})
    sent = []
    client = TonstClient(call_fn=lambda p: (sent.append(p), "ok")[1], redaction_backend="gliner",
                         placeholder_style="readable")
    client.query("Brightwell Health called.")
    fake.entities = []  # the detector misses it the second time
    client.query("Book the Brightwell Health clinic for Friday.")
    assert "Brightwell" not in sent[1] and "[[EMPLOYER_1]]" in sent[1]


# ---------------------------------------------------------------- ownership hints

from tonst.placeholders import build_hint, contact_owners  # noqa: E402


def test_contact_owners_are_narrow():
    t = ("Customer [[NAME_1]] ([[EMAIL_1]]@[[DOMAIN_1]], [[PHONE_1]]) wrote. Tell [[NAME_1.first]] to call "
         "[[PHONE_2]]. Or ring [[NAME_3]] on [[PHONE_3]].")
    assert contact_owners(t) == {"[[EMAIL_1]]": "[[NAME_1]]", "[[PHONE_1]]": "[[NAME_1]]",
                                 "[[PHONE_3]]": "[[NAME_3]]"}


def test_hint_lists_owners_only_when_known():
    assert build_hint("no contacts here") == PLACEHOLDER_HINT
    hint = build_hint("Text [[NAME_2]] on [[PHONE_1]].")
    assert hint.endswith("Contact details: [[PHONE_1]] is [[NAME_2]]'s own -- never present them as anyone else's.\n\n")


def test_client_sends_owner_hint():
    sent = []
    client = TonstClient(call_fn=lambda p: (sent.append(p), "ok")[1], placeholder_style="readable")
    # regex-only backend: no names, so no owners -> plain hint
    client.query("Call +44 7700 900412 today")
    assert "Contact details" not in sent[0]


# ---------------------------------------------------------------- streaming

import random  # noqa: E402

from tonst import StreamRestorer  # noqa: E402


def test_stream_restorer_matches_whole_restore_for_any_chunking():
    mapping = {"[[NAME_1]]": "Omar Haddad", "[[NAME_1.first]]": "Omar",
               "[[EMAIL_1]]": "omar@veltrix.io", "[[DOMAIN_1]]": "veltrix.io"}
    full = "Hi [[NAME_1.first]], writing to [[EMAIL_1]]@[[DOMAIN_1]], cc [[NAME_1]]. NAME_1 agreed. Done."
    want = restore_placeholders(full, mapping)
    for seed in range(200):
        r = random.Random(seed)
        cuts = sorted(r.sample(range(1, len(full)), r.randint(1, 30)))
        parts = [full[a:b] for a, b in zip([0] + cuts, cuts + [len(full)])]
        s = StreamRestorer(mapping)
        out = "".join(s.feed(p) for p in parts) + s.flush()
        assert out == want, (seed, parts)


def test_stream_restorer_never_shows_half_a_placeholder():
    s = StreamRestorer({"[[NAME_1]]": "Omar"})
    shown = s.feed("Hello [[NA")
    assert "[[" not in shown and shown == "Hello "
    assert s.feed("ME_1]] there") == "Omar "
    assert s.flush() == "there"


# ---------------------------------------------------------------- found by the 12-case dry run

class _TruncatingGliner:
    """Mimics GLiNER's 384-token limit: only 'sees' the first ~200 words it is given."""

    def __init__(self, entities):
        self.entities = entities

    def predict_entities(self, text, labels, threshold=0.5):
        visible = " ".join(text.split(" ")[:200])
        return [e for e in self.entities if e["text"] in visible]


def test_gliner_scans_long_text_in_windows(monkeypatch):
    filler = " ".join(["word"] * 600)
    text = f"Start. {filler} The signature says Luma Retail at the end."
    monkeypatch.setattr(gr, "_MODEL_CACHE", {"fake": _TruncatingGliner([{"text": "Luma Retail", "label": "company name"}])})
    res = gr.GlinerRedactor(model="fake").redact(text, placeholders=PlaceholderFactory("readable"))
    assert "Luma Retail" not in res.redacted_text and "[[EMPLOYER_1]]" in res.redacted_text


def test_windows_cover_everything_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    wins = gr._windows(" ".join(words))
    seen = set(" ".join(wins).split())
    assert seen == set(words)
    assert all(len(w.split()) <= gr.WINDOW_WORDS for w in wins)
    assert gr._windows("short text") == ["short text"]


def test_trim_leaves_no_blank_runs_after_dedupe():
    from tonst.trim import mechanical_trim
    text = "A\n\nsig\n\nB\n\nsig\n\nC\n\nsig\n\n\n"
    out = mechanical_trim(text)
    assert "\n\n\n" not in out and out.endswith("C")



def test_made_up_part_of_single_name_restores():
    mapping = {"[[NAME_3]]": "Rahul", "[[NAME_1]]": "Omar Haddad"}
    assert restore_placeholders("mentor to [[NAME_3.first]] and [[NAME_1.last]]", mapping) == "mentor to Rahul and Haddad"
    # unknown entity: shown as a neutral blank, never as a raw token
    assert restore_placeholders("[[NAME_9.first]] unknown", mapping) == "[name] unknown"


def test_hint_no_longer_suggests_blank_fields():
    assert "[Your name]" not in PLACEHOLDER_HINT


# ---------------------------------------------------------------- found by the full v3 run

def test_email_does_not_swallow_sentence_period():
    res = redact("Contacts: a@northpeakcap.com, b@northpeakcap.com.", placeholders=PlaceholderFactory("readable"))
    assert res.redacted_text == "Contacts: [[EMAIL_1]]@[[DOMAIN_1]], [[EMAIL_2]]@[[DOMAIN_1]]."


def test_trunk_prefix_phone():
    assert redact("mobile 090000 12345.").redacted_text.startswith("mobile [[PHONE_")


def test_invented_placeholders_become_neutral_blanks():
    mapping = {"[[EMAIL_1]]": "a@b.com"}
    assert restore_placeholders("From: [[EMAIL_2]] To: [[EMAIL_1]]", mapping) == "From: [email] To: a@b.com"
    assert restore_placeholders("Hi [[NAME_7]]", mapping) == "Hi [name]"


def test_secrets_not_restored_by_default_and_hint_mentions_them():
    sent = []
    key = "sk-ant-api03-" + "B" * 40
    client = TonstClient(call_fn=lambda p: (sent.append(p), p.split("\n\n")[-1])[1])
    out, rep = client.query(f"Rewrite this review: you committed {key}")
    assert key not in out and rep.secrets_withheld == 1
    assert "exposed credentials" in sent[0]
    out2, _ = TonstClient(call_fn=lambda p: p.split("\n\n")[-1], restore_secrets=True).query(f"key {key}")
    assert key in out2


# ---------------------------------------------------------------- found by the v4 run

def test_single_name_has_no_separate_last_name():
    mapping = {"[[NAME_1]]": "Sofia"}
    assert restore_placeholders("Note for [[NAME_1.first]] [[NAME_1.last]]:", mapping) == "Note for Sofia:"


def test_parenthesised_us_phone_keeps_its_bracket():
    res = redact("call (415) 555-0144 today", placeholders=PlaceholderFactory("readable"))
    assert res.redacted_text == "call [[PHONE_1]] today"
    assert res.restore(res.redacted_text) == "call (415) 555-0144 today"


def test_secret_placeholders_name_the_kind():
    f = PlaceholderFactory("readable")
    res = redact("GITHUB_TOKEN=ghp_" + "a" * 36 + " and sk-ant-api03-" + "b" * 30, placeholders=f)
    assert "[[SECRET_GITHUB_TOKEN_1]]" in res.redacted_text and "[[SECRET_ANTHROPIC_KEY_1]]" in res.redacted_text
    assert redacted_types_from_mapping(res.mapping) == {"SECRET": 2}


def test_secrets_withheld_as_redacted():
    key = "sk-ant-api03-" + "C" * 40
    out, _ = TonstClient(call_fn=lambda p: p.split("\n\n")[-1]).query(f"key {key}")
    assert out == "key [REDACTED]"


def test_public_service_hosts_not_hidden(monkeypatch):
    fake = _FakeGliner([{"text": "api.anthropic.com", "label": "company name"},
                        {"text": "v1", "label": "project codename"},
                        {"text": "Veltrix Logistics", "label": "company name"}])
    monkeypatch.setattr(gr, "_MODEL_CACHE", {"fake": fake})
    res = gr.GlinerRedactor(model="fake").redact(
        "curl https://api.anthropic.com/v1/models for Veltrix Logistics", placeholders=PlaceholderFactory("readable"))
    assert res.redacted_text == "curl https://api.anthropic.com/v1/models for [[EMPLOYER_1]]"
