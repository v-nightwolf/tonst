"""
Phase 1 privacy tests for tonst: keyed placeholders, readable
placeholders, and the new secret / account-ID / money detectors.
"""

import hashlib
import os
import stat

import pytest

import tonst.placeholders as ph
from tonst import TonstClient
from tonst.local_model import placeholders_preserved
from tonst.placeholders import (
    PLACEHOLDER_STRICT_RE,
    PlaceholderFactory,
    _load_or_create_key,
)
from tonst.redact import redact, redact_with_llm, restore_placeholders


# ---------------------------------------------------------------- keyed hash

def test_hash_placeholder_is_not_bare_sha256():
    """The old scheme let anyone confirm a guessed value offline."""
    email = "john.smith@acme.com"
    result = redact(f"Contact {email} today", email_style="whole")
    (placeholder,) = result.mapping
    bare = hashlib.sha256(email.encode()).hexdigest()[:8]
    assert placeholder.startswith("[[EMAIL_")
    assert bare not in placeholder


def test_hash_placeholder_is_deterministic_and_key_dependent():
    a = PlaceholderFactory("hash", key=b"key-one-000000000")
    b = PlaceholderFactory("hash", key=b"key-two-000000000")
    assert a.make("EMAIL", "x@y.com") == a.make("EMAIL", "x@y.com")
    assert a.make("EMAIL", "x@y.com") != b.make("EMAIL", "x@y.com")


def test_key_file_created_private_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("TONST_PLACEHOLDER_KEY", raising=False)
    path = str(tmp_path / "sub" / "placeholder.key")
    first = _load_or_create_key(path)
    assert len(first) == 32
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert _load_or_create_key(path) == first


def test_env_key_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TONST_PLACEHOLDER_KEY", "from-env")
    assert _load_or_create_key(str(tmp_path / "k")) == b"from-env"
    assert not (tmp_path / "k").exists()


# ---------------------------------------------------------------- readable

def test_readable_numbering_is_consistent():
    f = PlaceholderFactory("readable")
    assert f.make("NAME", "John") == "[[NAME_1]]"
    assert f.make("NAME", "Ann") == "[[NAME_2]]"
    assert f.make("EMAIL", "a@b.com") == "[[EMAIL_1]]"
    assert f.make("NAME", "John") == "[[NAME_1]]"


def test_bad_style_rejected():
    with pytest.raises(ValueError):
        PlaceholderFactory("pretty")


def test_readable_round_trip_with_ten_plus_entities():
    # [[EMAIL_1]] must not be restored inside [[EMAIL_10]]
    f = PlaceholderFactory("readable")
    emails = [f"user{i}@example.com" for i in range(12)]
    result = redact(" ".join(emails), placeholders=f)
    assert "[[EMAIL_10]]" in result.redacted_text
    assert "user" not in result.redacted_text and "example.com" not in result.redacted_text
    assert result.restore(result.redacted_text) == " ".join(emails)


def test_client_readable_numbering_spans_calls():
    sent = []

    def fake_api(prompt):
        sent.append(prompt)
        return prompt

    client = TonstClient(call_fn=fake_api, placeholder_style="readable")
    out1, _ = client.query("Email a@b.com please")
    out2, _ = client.query("Now email c@d.com and a@b.com again")
    assert "[[EMAIL_1]]" in sent[0]
    assert "[[EMAIL_2]]" in sent[1] and "[[EMAIL_1]]" in sent[1]
    assert "a@b.com" in out1 and "c@d.com" in out2
    assert not any(a in sent[0] + sent[1] for a in ("a@b.com", "c@d.com", "b.com", "d.com"))


def test_readable_factory_reaches_secondary_redactor():
    seen = {}

    class FakeRedactor:
        def redact(self, text, placeholders=None):
            seen["factory"] = placeholders
            span = "Marcus"
            p = placeholders.make("NAME", span)
            return type("R", (), {"redacted_text": text.replace(span, p), "mapping": {p: span}})()

    f = PlaceholderFactory("readable")
    result = redact_with_llm("Marcus at m@x.io", FakeRedactor(), placeholders=f)
    assert seen["factory"] is f
    assert result.redacted_text == "[[NAME_1]] at [[EMAIL_1]]@[[DOMAIN_1]]"


def test_plain_signature_redactor_still_works():
    class Plain:
        def redact(self, text):
            return type("R", (), {"redacted_text": text, "mapping": {}})()

    result = redact_with_llm("mail m@x.io", Plain())
    assert "m@x.io" not in result.redacted_text


# ---------------------------------------------------------------- recognition

@pytest.mark.parametrize(
    "token",
    ["[[EMAIL_1a2b3c4d]]", "[[CREDIT_CARD_1a2b3c4d]]", "[[IP_ADDRESS_00ff00ff]]", "[[NAME_1]]", "[[SSN_LIKE_12]]"],
)
def test_strict_regex_recognises_all_labels(token):
    assert PLACEHOLDER_STRICT_RE.fullmatch(token)


def test_compression_guard_accepts_underscore_labels():
    """Previously any text with a [[CREDIT_CARD_..]] placeholder could never be compressed."""
    original = "Card [[CREDIT_CARD_1a2b3c4d]] was declined for [[NAME_1]], please retry the payment."
    rewritten = "Card [[CREDIT_CARD_1a2b3c4d]] declined for [[NAME_1]]; retry payment."
    assert placeholders_preserved(original, rewritten)
    assert not placeholders_preserved(original, "Card declined; retry payment.")


# ---------------------------------------------------------------- secrets

@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-" + "A" * 40,
        "sk-proj-" + "b" * 40,
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "c" * 36,
        "github_pat_" + "d" * 30,
        "xoxb-1234567890-abcdefghij",
        "AIza" + "e" * 35,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ],
)
def test_secrets_redacted_and_restored(secret):
    text = f"Use this key: {secret} for the deploy."
    result = redact(text)
    assert secret not in result.redacted_text
    assert "[[SECRET_" in result.redacted_text
    assert result.restore(result.redacted_text) == text


def test_private_key_block_redacted():
    block = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc123\n-----END RSA PRIVATE KEY-----"
    result = redact(f"key:\n{block}\nthanks")
    assert "MIIEowIBAAKCAQEA" not in result.redacted_text
    assert result.restore(result.redacted_text) == f"key:\n{block}\nthanks"


def test_assignment_redacts_only_the_value():
    result = redact('config: password = "Hunter2Hunter2" and api_key: abcd1234efgh')
    assert "Hunter2Hunter2" not in result.redacted_text
    assert "abcd1234efgh" not in result.redacted_text
    assert "password = \"[[SECRET_" in result.redacted_text
    assert "api_key: [[SECRET_" in result.redacted_text


@pytest.mark.parametrize(
    "text",
    ["The secret is out.", "Send a password reset link.", "Our API key rotation policy is monthly."],
)
def test_secret_words_alone_are_not_redacted(text):
    assert redact(text).redacted_text == text


# ---------------------------------------------------------------- opt-in categories

def test_money_and_accounts_off_by_default():
    text = "Customer #928381 disputes invoice INV-77391 for $14,821.32."
    assert redact(text).redacted_text == text


def test_money_and_accounts_opt_in():
    text = "Customer #928381 disputes invoice INV-77391 for $14,821.32 (about ₹12 lakh)."
    result = redact(text, extra_categories=["ACCOUNT_ID", "MONEY"])
    red = result.redacted_text
    for value in ["928381", "INV-77391", "$14,821.32", "₹12 lakh"]:
        assert value not in red, value
    assert red.startswith("Customer #[[ACCOUNT_ID_")
    assert "invoice [[ACCOUNT_ID_" in red
    assert result.restore(red) == text


@pytest.mark.parametrize(
    "text",
    ["Please contact customer service.", "What is the order status?", "The account team met today."],
)
def test_account_words_without_ids_untouched(text):
    assert redact(text, extra_categories=["ACCOUNT_ID"]).redacted_text == text


def test_money_with_trailing_currency_word():
    result = redact("Refund 149 rupees and 20 USD.", extra_categories=["MONEY"])
    assert "149 rupees" not in result.redacted_text and "20 USD" not in result.redacted_text


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        redact("x", extra_categories=["PASSPORT"])
    with pytest.raises(ValueError):
        TonstClient(call_fn=lambda p: p, extra_redaction=["PASSPORT"])


def test_client_extra_redaction():
    sent = []
    client = TonstClient(call_fn=lambda p: (sent.append(p), p)[1], extra_redaction=["MONEY"])
    out, _ = client.query("Summarise: the refund of $3,500.00 was approved.")
    assert "$3,500.00" not in sent[0]
    assert "$3,500.00" in out


def test_restore_placeholders_unchanged_api():
    mapping = {"[[NAME_1]]": "John", "[[NAME_10]]": "Priya"}
    assert restore_placeholders("[[NAME_10]] met [[NAME_1]]", mapping) == "Priya met John"


# ---------------------------------------------------------------- gaps found by the benchmark dry run

@pytest.mark.parametrize(
    "phone",
    ["+91 98450 21733", "+55 11 98822-4410", "+49 151 2384 9921", "+44 7700 900412", "+420 603 118 447"],
)
def test_international_phones_redacted_as_phone(phone):
    red = redact(f"call {phone} now").redacted_text
    assert red.startswith("call [[PHONE_") and red.endswith("]] now"), red


def test_env_style_secret_names():
    red = redact("DB_PASSWORD=Tr0ub4dor&3x\nSTRIPE_API_KEY: abcd1234efgh").redacted_text
    assert "Tr0ub4dor" not in red and "abcd1234efgh" not in red
    assert red.startswith("DB_PASSWORD=[[SECRET_")


def test_employee_and_patient_ids_opt_in():
    red = redact("employee ID 54550, patient no. 88213", extra_categories=["ACCOUNT_ID"]).redacted_text
    assert "54550" not in red and "88213" not in red


def test_card_numbers_still_cards():
    assert redact("card 4111 1111 1111 1111 ok").redacted_text.startswith("card [[CREDIT_CARD_")


def test_standalone_id_codes_opt_in():
    red = redact("invoiced as INV-22243; parcel #777094; we are #1", extra_categories=["ACCOUNT_ID"]).redacted_text
    assert "INV-22243" not in red and "777094" not in red
    assert red.endswith("we are #1")


def test_gliner_failure_is_loud_not_silent(monkeypatch, caplog):
    """If GLiNER can't load, redaction must say so -- not quietly drop to regex-only."""
    import builtins
    import logging
    import tonst.gliner_redact as gr

    real_import = builtins.__import__

    def no_gliner(name, *a, **k):
        if name == "gliner":
            raise ImportError("no gliner here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_gliner)
    monkeypatch.setattr(gr, "_MODEL_CACHE", {})
    monkeypatch.setattr(gr, "_WARNED", set())
    red = gr.GlinerRedactor(model="test/does-not-exist")
    with caplog.at_level(logging.WARNING, logger="tonst.gliner_redact"):
        res = red.redact("Priya Nair works at Veltrix Logistics")
    assert not res.model_available
    assert red.last_error is not None
    assert "NOT" in caplog.text


def test_benchmark_preflight_refuses_broken_gliner(monkeypatch, capsys):
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "experiments", "privacy_quality"))
    import run as bench
    import tonst.gliner_redact as gr

    class Broken:
        def __init__(self, *a, **k):
            self.last_error = RuntimeError("boom")

        def redact(self, text, placeholders=None):
            return gr.GlinerRedactionResult(redacted_text=text, mapping={}, model_available=False, entities_found=0)

    monkeypatch.setattr(gr, "GlinerRedactor", Broken)
    assert bench.main(["--fake", "--limit", "2", "--backend", "gliner"]) == 3
    assert "GLiNER check FAILED" in capsys.readouterr().out


# ---------------------------------------------------------------- from the first live benchmark run

@pytest.mark.parametrize(
    "answer",
    ["Dear NAME_1, thanks", "Dear [NAME_1], thanks", "Dear [[ NAME_1 ]], thanks", "Dear [[NAME_1]], thanks"],
)
def test_restore_tolerates_reformatted_placeholders(answer):
    assert restore_placeholders(answer, {"[[NAME_1]]": "Priya Nair"}) == "Dear Priya Nair, thanks"


def test_tolerant_restore_respects_boundaries():
    mapping = {"[[NAME_1]]": "Priya", "[[NAME_10]]": "Omar"}
    assert restore_placeholders("NAME_10 and NAME_1", mapping) == "Omar and Priya"
    assert restore_placeholders("MY_NAME_1X stays", mapping) == "MY_NAME_1X stays"


def test_tolerant_restore_hash_style():
    assert restore_placeholders("owner: EMPLOYER_d51ae5e6.", {"[[EMPLOYER_d51ae5e6]]": "Luma Retail"}) == "owner: Luma Retail."


def test_hint_added_only_when_placeholders_present():
    sent = []
    client = TonstClient(call_fn=lambda p: (sent.append(p), "ok")[1])
    client.query("What is 2+2?")
    client.query("Email a@b.com about it")
    assert not sent[0].startswith("Note: tokens")
    assert sent[1].startswith("Note: tokens in double square brackets")


def test_hint_can_be_turned_off():
    sent = []
    client = TonstClient(call_fn=lambda p: (sent.append(p), "ok")[1], placeholder_hint=False)
    client.query("Email a@b.com about it")
    assert not sent[0].startswith("Note:")


def test_hint_goes_in_system_message_for_messages_fn():
    seen = []
    client = TonstClient(messages_fn=lambda msgs: (seen.append(msgs), "ok")[1])
    client.query_messages([{"role": "system", "content": "Be brief."},
                           {"role": "user", "content": "Email a@b.com please"}])
    msgs = seen[0]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("Note: tokens") and msgs[0]["content"].endswith("Be brief.")
    assert sum(m["role"] == "system" for m in msgs) == 1
