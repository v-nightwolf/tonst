"""
test_tonst.py
-------------
Formal test suite -- run with: pytest test_tonst.py -v

This codifies every check we verified ad-hoc during development into a
suite anyone (including CI) can run and trust: redaction round-trips,
the hallucination guard rail, fail-soft behavior with no local model,
caching, trimming, and the full TonstClient pipeline end to end.
"""

import pytest
from tonst.redact import redact, redact_with_llm
from tonst.redact_llm import LLMRedactor
from tonst.trim import (
    estimate_tokens,
    strip_redundant_whitespace,
    dedupe_repeated_lines,
    truncate_history,
)
from tonst.cache import SemanticCache
from tonst.client import TonstClient


# ---------------------------------------------------------------------
# redact.py -- regex-based redaction
# ---------------------------------------------------------------------

def test_regex_redact_and_restore_round_trip():
    text = "Contact me at jane.doe@example.com or +1 415-555-0132, card 4111 1111 1111 1111"
    result = redact(text)
    assert "jane.doe@example.com" not in result.redacted_text
    assert "4111 1111 1111 1111" not in result.redacted_text
    assert result.restore(result.redacted_text) == text


def test_regex_redact_skips_short_numeric_noise():
    # A short number shouldn't be misfired as a card/phone number.
    text = "I'll be there at 5:30, room 42."
    result = redact(text)
    assert result.redacted_text == text
    assert len(result.mapping) == 0


# ---------------------------------------------------------------------
# redact_llm.py -- local-LLM redaction with fail-soft + guard rails
# ---------------------------------------------------------------------

def test_llm_redact_normal_case():
    text = "Please review this for Priya Malhotra at Initech regarding Project Nightingale."

    def fake_model(prompt, model, timeout):
        return (
            '[{"text": "Priya Malhotra", "type": "NAME"}, '
            '{"text": "Initech", "type": "EMPLOYER"}, '
            '{"text": "Project Nightingale", "type": "CODENAME"}]'
        )

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Priya Malhotra" not in result.redacted_text
    assert result.restore(result.redacted_text) == text


def test_llm_redact_ignores_hallucinated_spans():
    text = "Please review this for Priya Malhotra."

    def fake_model_hallucinate(prompt, model, timeout):
        return '[{"text": "Someone Not In The Text", "type": "NAME"}]'

    redactor = LLMRedactor(model_call_fn=fake_model_hallucinate)
    result = redact_with_llm(text, redactor)
    # The hallucinated span must never appear as a "redacted" value --
    # it was never actually in the source text.
    assert "Someone Not In The Text" not in result.mapping.values()


def test_llm_redact_fails_soft_on_malformed_output():
    text = "Please review this for Priya Malhotra."

    def fake_model_garbage(prompt, model, timeout):
        return "Sure! Here's the answer: Priya Malhotra is a name."

    redactor = LLMRedactor(model_call_fn=fake_model_garbage)
    result = redact_with_llm(text, redactor)
    # No crash, no partial corruption -- just falls back to regex-only.
    assert result.redacted_text  # non-empty, didn't raise


def test_llm_redact_fails_soft_when_model_unavailable():
    text = "Please review this for Priya Malhotra."

    def fake_model_unreachable(prompt, model, timeout):
        return None  # simulates Ollama not running

    redactor = LLMRedactor(model_call_fn=fake_model_unreachable)
    result = redact_with_llm(text, redactor)
    assert result.redacted_text == text  # regex found nothing here either


# ---------------------------------------------------------------------
# trim.py -- mechanical token reduction
# ---------------------------------------------------------------------

def test_estimate_tokens_roughly_chars_over_four():
    assert estimate_tokens("a" * 40) == 10


def test_strip_redundant_whitespace():
    assert strip_redundant_whitespace("a    b\n\n\n\nc") == "a b\n\nc"


def test_dedupe_repeated_lines():
    text = "hello\nhello\nworld"
    assert dedupe_repeated_lines(text) == "hello\nworld"


def test_truncate_history_keeps_system_and_recent_turns():
    messages = (
        [{"role": "system", "content": "sys"}]
        + [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    )
    trimmed = truncate_history(messages, keep_last_n=3)
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) == 4  # system + last 3
    assert trimmed[-1]["content"] == "msg9"


# ---------------------------------------------------------------------
# cache.py -- semantic cache
# ---------------------------------------------------------------------

def test_cache_miss_then_hit_on_similar_query():
    cache = SemanticCache(similarity_threshold=0.80)
    assert cache.lookup("What is prompt caching?") is None
    cache.store("What is prompt caching?", "It's a cost-saving technique.")
    # A near-identical rephrasing should hit.
    assert cache.lookup("What is prompt caching") == "It's a cost-saving technique."


def test_cache_miss_on_unrelated_query():
    cache = SemanticCache(similarity_threshold=0.90)
    cache.store("What is prompt caching?", "It's a cost-saving technique.")
    assert cache.lookup("What's the weather today?") is None


# ---------------------------------------------------------------------
# client.py -- full pipeline, end to end
# ---------------------------------------------------------------------

def test_client_redacts_before_calling_paid_api():
    received = {}

    def mock_api(prompt):
        received["prompt"] = prompt
        return "Thanks, we'll follow up."

    client = TonstClient(call_fn=mock_api)
    prompt = "My email is test@example.com, please help."
    response, report = client.query(prompt)

    assert "test@example.com" not in received["prompt"]
    assert report.redacted_fields == 1
    assert response == "Thanks, we'll follow up."


def test_client_cache_hit_costs_zero_tokens():
    def mock_api(prompt):
        return "cached-style response"

    client = TonstClient(call_fn=mock_api)
    client.query("What is prompt caching?")
    _, report = client.query("What is prompt caching")  # near-duplicate

    assert report.cache_hit is True
    assert report.sent_tokens == 0


def test_client_enhanced_redaction_catches_free_text_pii():
    received = {}

    def mock_api(prompt):
        received["prompt"] = prompt
        return "ok"

    def fake_model(prompt, model, timeout):
        return '[{"text": "Priya Malhotra", "type": "NAME"}]'

    client = TonstClient(call_fn=mock_api, use_enhanced_redaction=True)
    # Inject the fake model call so this test needs no real Ollama.
    client.llm_redactor._call_model = fake_model

    client.query("Hi, this is Priya Malhotra, please help.")
    assert "Priya Malhotra" not in received["prompt"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
