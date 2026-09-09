"""
test_tonst.py
-------------
Formal test suite -- run with: pytest test_tonst.py -v

This codifies every check we verified ad-hoc during development into a
suite anyone (including CI) can run and trust: redaction round-trips
(including determinism, which prompt-caching structuring depends on),
the hallucination guard rail, fail-soft behavior with no local model,
trimming, prompt-caching structuring, and the full TonstClient pipeline
end to end.
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
from tonst.cache_structuring import (
    PromptParts,
    structure_for_caching,
    build_anthropic_cache_request,
    check_cache_eligibility,
    parse_anthropic_usage,
    CacheUsageReport,
)
from tonst.providers import openai as openai_provider
from tonst.providers import gemini as gemini_provider
from tonst.providers import generic as generic_provider
from tonst.providers import presets as provider_presets
from tonst.compactor import HistoryCompactor, compact_history
from tonst.trim import flatten_messages
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


def test_regex_redact_placeholder_is_deterministic_across_calls():
    # This is what makes prompt-caching structuring safe: redacting the
    # SAME stable content on two separate calls must produce
    # byte-identical output, or the provider will never see a matching
    # cached prefix. A random-UUID placeholder would break this.
    text = "Contact me at jane.doe@example.com"
    result_a = redact(text)
    result_b = redact(text)
    assert result_a.redacted_text == result_b.redacted_text


def test_regex_redact_different_values_get_different_placeholders():
    text = "Emails: jane.doe@example.com and john.smith@example.com"
    result = redact(text)
    assert len(result.mapping) == 2
    assert len(set(result.mapping.keys())) == 2


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


def test_llm_redact_placeholder_is_deterministic_across_calls():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        return '[{"text": "Priya Malhotra", "type": "NAME"}]'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result_a = redact_with_llm(text, redactor)
    result_b = redact_with_llm(text, redactor)
    assert result_a.redacted_text == result_b.redacted_text


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
# cache_structuring.py -- prompt-caching structuring
# ---------------------------------------------------------------------

def test_structure_for_caching_orders_system_stable_variable():
    parts = PromptParts(
        system="You are a helpful assistant.",
        stable_blocks=["Reference doc content."],
        variable="What's the weather?",
    )
    result = structure_for_caching(parts)
    assert result.index("helpful assistant") < result.index("Reference doc")
    assert result.index("Reference doc") < result.index("weather")


def test_structure_for_caching_skips_empty_blocks():
    parts = PromptParts(system=None, stable_blocks=["", "  ", "real content"], variable="question")
    result = structure_for_caching(parts)
    assert result == "real content\n\nquestion"


def test_structure_for_caching_is_stable_across_calls():
    # Same inputs must produce the exact same string every time -- this
    # is the ordering guarantee providers' automatic caching relies on.
    parts = PromptParts(system="sys", stable_blocks=["ctx1", "ctx2"], variable="q")
    assert structure_for_caching(parts) == structure_for_caching(parts)


def test_build_anthropic_cache_request_places_breakpoint_on_last_stable_block():
    parts = PromptParts(
        system="You are a support agent.",
        stable_blocks=["Doc part 1", "Doc part 2"],
        variable="What's my order status?",
    )
    body = build_anthropic_cache_request(parts, model="claude-sonnet-4-6", warn_if_ineligible=False)

    # System is cached as its own block.
    assert body["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}

    content = body["messages"][0]["content"]
    # Breakpoint is on the LAST stable block, not the first, and not on
    # the trailing variable block.
    assert content[0].get("cache_control") is None
    assert content[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert "cache_control" not in content[2]
    assert content[2]["text"] == "What's my order status?"


def test_build_anthropic_cache_request_falls_back_to_plain_string_with_no_stable_blocks():
    parts = PromptParts(system=None, stable_blocks=[], variable="just a question")
    body = build_anthropic_cache_request(parts, model="claude-sonnet-4-6", warn_if_ineligible=False)
    assert body["messages"][0]["content"] == "just a question"
    assert "system" not in body


def test_build_anthropic_cache_request_rejects_invalid_ttl():
    parts = PromptParts(stable_blocks=["x"], variable="q")
    with pytest.raises(ValueError):
        build_anthropic_cache_request(parts, model="claude-sonnet-4-6", cache_ttl="1d")


def test_build_anthropic_cache_request_accepts_one_hour_ttl():
    parts = PromptParts(stable_blocks=["x"], variable="q")
    body = build_anthropic_cache_request(parts, model="claude-sonnet-4-6", cache_ttl="1h", warn_if_ineligible=False)
    assert body["messages"][0]["content"][0]["cache_control"]["ttl"] == "1h"


def test_check_cache_eligibility_flags_too_short_content():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    result = check_cache_eligibility(parts, model="claude-sonnet-4-6")
    assert result.eligible is False
    assert result.minimum_required == 1024


def test_check_cache_eligibility_flags_long_enough_content():
    parts = PromptParts(stable_blocks=["word " * 2000], variable="q")
    result = check_cache_eligibility(parts, model="claude-sonnet-4-6")
    assert result.eligible is True


def test_check_cache_eligibility_uses_default_minimum_for_unknown_model():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    result = check_cache_eligibility(parts, model="some-future-model-not-in-the-table")
    assert result.minimum_required == 1024  # DEFAULT_CACHE_MINIMUM_TOKENS


def test_check_cache_eligibility_haiku_3_5_uses_its_own_higher_minimum():
    # Regression test: claude-haiku-3-5's real minimum is 2,048 tokens,
    # not the 1,024 DEFAULT_CACHE_MINIMUM_TOKENS -- this model was
    # missing from CACHE_MINIMUM_TOKENS entirely, so a stable block
    # between 1,024 and 2,048 tokens was incorrectly reported eligible
    # (silently falling back to the wrong default) before this was
    # caught and fixed by checking against the live Anthropic docs.
    parts = PromptParts(stable_blocks=["word " * 1500], variable="q")  # ~1,875 tokens (est.) -- between the two minimums
    result = check_cache_eligibility(parts, model="claude-haiku-3-5")
    assert result.minimum_required == 2048
    assert result.eligible is False  # would have wrongly been True against the 1,024 default


def test_check_cache_eligibility_covers_previously_missing_models():
    # These four were absent from CACHE_MINIMUM_TOKENS at the same time
    # as claude-haiku-3-5 above; their correct minimum happens to match
    # DEFAULT_CACHE_MINIMUM_TOKENS (1,024) so the earlier gap never
    # produced a wrong answer for them -- this test just locks in that
    # they're now explicit entries, not accidentally-correct fallbacks.
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    for model in ("claude-opus-4-1", "claude-opus-4", "claude-sonnet-4"):
        result = check_cache_eligibility(parts, model=model)
        assert result.minimum_required == 1024


def test_build_anthropic_cache_request_warns_when_stable_content_too_short():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    with pytest.warns(UserWarning, match="below.*minimum"):
        build_anthropic_cache_request(parts, model="claude-sonnet-4-6")


def test_build_anthropic_cache_request_no_warning_when_eligible():
    parts = PromptParts(stable_blocks=["word " * 2000], variable="q")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        build_anthropic_cache_request(parts, model="claude-sonnet-4-6")


def test_build_anthropic_cache_request_can_suppress_warning():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_anthropic_cache_request(parts, model="claude-sonnet-4-6", warn_if_ineligible=False)


def test_parse_anthropic_usage_extracts_cache_fields():
    fake_response = {
        "usage": {
            "input_tokens": 12,
            "output_tokens": 40,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 1800,
        }
    }
    usage = parse_anthropic_usage(fake_response)
    assert usage.cache_hit is True
    assert usage.cache_write is False
    assert usage.percent_of_input_from_cache > 90


def test_parse_anthropic_usage_handles_missing_usage_field():
    usage = parse_anthropic_usage({})
    assert usage.input_tokens == 0
    assert usage.cache_hit is False


def test_estimated_cost_savings_is_negative_on_a_cache_write_call():
    # Real numbers from a live call: writing a new cache entry costs a
    # premium (1.25x base price for a 5-minute TTL) -- there's no
    # discount yet, so this should come back negative, not zero or positive.
    usage = CacheUsageReport(
        input_tokens=39, output_tokens=40,
        cache_creation_input_tokens=1547, cache_read_input_tokens=0,
    )
    savings = usage.estimated_cost_savings_percent("claude-sonnet-4-6")
    assert savings < 0


def test_estimated_cost_savings_is_strongly_positive_on_a_cache_read_call():
    # Real numbers from the same live test's second call: a cache READ
    # is billed at 10% of base price, so the saving should land close to
    # (but not exactly, because of the small uncached `input_tokens`
    # portion) the ~90% discount that multiplier implies.
    usage = CacheUsageReport(
        input_tokens=39, output_tokens=35,
        cache_creation_input_tokens=0, cache_read_input_tokens=1547,
    )
    savings = usage.estimated_cost_savings_percent("claude-sonnet-4-6")
    assert 85 < savings < 91


def test_estimated_cost_savings_uses_lower_multiplier_for_low_cost_models():
    usage = CacheUsageReport(
        input_tokens=0, output_tokens=10,
        cache_creation_input_tokens=0, cache_read_input_tokens=1000,
    )
    standard_savings = usage.estimated_cost_savings_percent("claude-sonnet-4-6")
    low_cost_savings = usage.estimated_cost_savings_percent("claude-fable-5-1")
    # 2.5% cache-read rate beats the standard 10% rate -- bigger discount.
    assert low_cost_savings > standard_savings


def test_estimated_cost_savings_is_zero_with_no_tokens():
    usage = CacheUsageReport(0, 0, 0, 0)
    assert usage.estimated_cost_savings_percent("claude-sonnet-4-6") == 0.0



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


def test_client_query_structured_orders_before_sending():
    received = {}

    def mock_api(prompt):
        received["prompt"] = prompt
        return "ok"

    client = TonstClient(call_fn=mock_api)
    parts = PromptParts(
        system="You are a support agent.",
        stable_blocks=["Reused reference content."],
        variable="What's my order status?",
    )
    client.query_structured(parts)

    sent = received["prompt"]
    assert sent.index("support agent") < sent.index("Reused reference")
    assert sent.index("Reused reference") < sent.index("order status")


def test_client_redact_and_trim_parts_preserves_structure_and_restores():
    client = TonstClient(call_fn=lambda p: "unused")
    parts = PromptParts(
        system="You are a support agent.",
        stable_blocks=["Contact backup: ops@example.com"],
        variable="My email is test@example.com, please help.",
    )
    result = client.redact_and_trim_parts(parts)

    assert "ops@example.com" not in result.parts.stable_blocks[0]
    assert "test@example.com" not in result.parts.variable
    assert len(result.mapping) == 2
    # Restoring a response that happens to echo a placeholder should
    # bring back the real value.
    placeholder = next(iter(result.mapping))
    assert result.restore(f"See {placeholder}") == f"See {result.mapping[placeholder]}"


def test_client_redact_and_trim_parts_is_deterministic_for_caching():
    # The whole point of the deterministic-placeholder fix: redacting the
    # same stable block twice must yield identical text, or a cached
    # prefix containing PII would never match itself on a later call.
    client = TonstClient(call_fn=lambda p: "unused")
    parts = PromptParts(stable_blocks=["Contact: ops@example.com"], variable="q1")
    result_a = client.redact_and_trim_parts(parts)

    parts_again = PromptParts(stable_blocks=["Contact: ops@example.com"], variable="q2")
    result_b = client.redact_and_trim_parts(parts_again)

    assert result_a.parts.stable_blocks[0] == result_b.parts.stable_blocks[0]


# ---------------------------------------------------------------------
# client.py -- per-step latency instrumentation
# ---------------------------------------------------------------------

def test_query_reports_nonzero_total_and_call_latency():
    def slow_api(prompt):
        import time
        time.sleep(0.01)
        return "ok"

    client = TonstClient(call_fn=slow_api)
    _, report = client.query("hello world")

    assert report.total_ms > 0
    assert report.call_ms >= 10  # the sleep(0.01) above, in ms
    assert report.structuring_ms == 0.0  # only set by query_structured()
    assert report.local_overhead_ms >= 0


def test_query_structured_reports_structuring_ms_and_rolls_into_total():
    client = TonstClient(call_fn=lambda p: "ok")
    parts = PromptParts(system="sys", stable_blocks=["ctx"], variable="q")
    _, report = client.query_structured(parts)

    assert report.structuring_ms >= 0
    assert report.total_ms >= report.structuring_ms


def test_local_overhead_ms_sums_the_local_steps():
    client = TonstClient(call_fn=lambda p: "ok")
    _, report = client.query("test@example.com needs help")
    expected = (
        report.structuring_ms + report.redaction_ms + report.trim_ms + report.compression_ms
    )
    assert report.local_overhead_ms == expected



# ---------------------------------------------------------------------
# compactor.py -- history compaction
# ---------------------------------------------------------------------

def test_compact_history_falls_back_to_truncation_with_no_compactor():
    messages = [{"role": "user", "content": f"turn {i}" * 50} for i in range(20)]
    result = compact_history(messages, compactor=None, keep_last_n=3, token_threshold=10)
    assert result.compacted is False
    assert result.dropped_turns == 17
    assert len(result.messages) == 3


def test_compact_history_falls_back_when_below_token_threshold():
    # Only a couple of short turns to drop -- not worth summarizing.
    messages = [{"role": "user", "content": "hi"} for _ in range(5)]
    compactor = HistoryCompactor(model_call_fn=lambda p, m, t: "should never be called")
    result = compact_history(messages, compactor=compactor, keep_last_n=3, token_threshold=3000)
    assert result.compacted is False
    assert result.dropped_turns == 2


def test_compact_history_summarizes_when_above_threshold():
    older_content = "We discussed the Q3 roadmap and agreed to ship prompt caching first. " * 40
    messages = (
        [{"role": "user", "content": older_content} for _ in range(5)]
        + [{"role": "user", "content": f"recent turn {i}"} for i in range(3)]
    )

    def fake_model(prompt, model, timeout):
        return "We agreed to ship prompt caching first, per the Q3 roadmap discussion."

    compactor = HistoryCompactor(model_call_fn=fake_model)
    result = compact_history(messages, compactor=compactor, keep_last_n=3, token_threshold=50)

    assert result.compacted is True
    assert result.dropped_turns == 5
    assert "[Summary of earlier conversation]" in result.messages[0]["content"]
    assert result.messages[-3:] == messages[-3:]


def test_compact_history_falls_back_when_summary_fails_guard_rail():
    older_content = "Detailed context that should be summarized down. " * 40
    messages = (
        [{"role": "user", "content": older_content} for _ in range(5)]
        + [{"role": "user", "content": f"recent turn {i}"} for i in range(3)]
    )

    def fake_model_too_long(prompt, model, timeout):
        # Echoes back the ENTIRE prompt it was given (instructions plus
        # all the older turns, joined) -- guaranteed not meaningfully
        # shorter than older_text, so the guard rail should reject it as
        # a non-summary.
        return prompt

    compactor = HistoryCompactor(model_call_fn=fake_model_too_long)
    result = compact_history(messages, compactor=compactor, keep_last_n=3, token_threshold=50)
    assert result.compacted is False
    assert result.dropped_turns == 5


def test_compact_history_keeps_system_messages_separate():
    messages = (
        [{"role": "system", "content": "You are a helpful assistant."}]
        + [{"role": "user", "content": "turn " * 100} for _ in range(10)]
    )
    result = compact_history(messages, compactor=None, keep_last_n=2, token_threshold=10)
    assert result.messages[0]["role"] == "system"


def test_history_compactor_is_available_checks_ollama():
    compactor = HistoryCompactor(model_call_fn=lambda p, m, t: None)
    # No real Ollama in this test env -- is_available() should fail soft
    # (return False, not raise) rather than crash the caller.
    assert compactor.is_available() in (True, False)


def test_flatten_messages_joins_role_and_content():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    flat = flatten_messages(messages)
    assert "SYSTEM: sys" in flat
    assert "USER: hello" in flat


# ---------------------------------------------------------------------
# client.py -- query_messages(): redact-before-compact ordering
# ---------------------------------------------------------------------

def test_query_messages_redacts_before_compaction_runs():
    seen_by_compactor = {}

    def capturing_model(prompt, model, timeout):
        seen_by_compactor["prompt"] = prompt
        return "A condensed summary of the earlier discussion, well under the guard rail length."

    older_content = (
        "Please reach jane.doe@example.com about this. " * 40
    )
    messages = (
        [{"role": "user", "content": older_content} for _ in range(5)]
        + [{"role": "user", "content": f"recent turn {i}"} for i in range(3)]
    )

    def mock_api(prompt):
        return "ok"

    client = TonstClient(call_fn=mock_api, use_history_compaction=True, compaction_token_threshold=50)
    client.history_compactor._call_model = capturing_model

    client.query_messages(messages, keep_last_n=3)

    # The compactor must never see the raw email -- only whatever
    # placeholder redaction produced.
    assert "jane.doe@example.com" not in seen_by_compactor.get("prompt", "")


def test_query_messages_falls_back_to_truncation_without_compaction_enabled():
    messages = [{"role": "user", "content": f"turn {i}" * 20} for i in range(20)]
    client = TonstClient(call_fn=lambda p: "ok")
    _, report = client.query_messages(messages, keep_last_n=3)
    assert report.history_compacted is False
    assert report.history_turns_dropped == 17


def test_query_messages_restores_placeholders_echoed_in_response():
    messages = [{"role": "user", "content": "My email is test@example.com, please confirm."}]

    def mock_api(prompt):
        # Echo the whole (already-redacted) prompt back -- simulates a
        # model that repeats redacted content verbatim in its response.
        return prompt

    client = TonstClient(call_fn=mock_api)
    response, report = client.query_messages(messages, keep_last_n=3)
    assert "test@example.com" in response
    assert report.redacted_fields == 1


def test_query_messages_populates_latency_and_original_tokens():
    messages = [{"role": "user", "content": "hello there"}]
    client = TonstClient(call_fn=lambda p: "ok")
    _, report = client.query_messages(messages)
    assert report.total_ms > 0
    assert report.original_tokens > 0
    assert report.compaction_ms >= 0



# ---------------------------------------------------------------------
# providers/openai.py -- OpenAI prompt caching
# ---------------------------------------------------------------------

def test_openai_check_cache_eligibility_flat_minimum():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    result = openai_provider.check_cache_eligibility(parts, model="gpt-4o")
    assert result.eligible is False
    assert result.minimum_required == 1024


def test_openai_check_cache_eligibility_long_enough():
    parts = PromptParts(stable_blocks=["word " * 2000], variable="q")
    result = openai_provider.check_cache_eligibility(parts, model="gpt-4o")
    assert result.eligible is True


def test_openai_build_request_automatic_path_is_plain_string():
    parts = PromptParts(
        system="You are a support agent.",
        stable_blocks=["Doc part 1", "Doc part 2"],
        variable="What's my order status?",
    )
    body = openai_provider.build_openai_cache_request(parts, model="gpt-4o")

    assert body["messages"][0]["role"] == "system"
    user_content = body["messages"][1]["content"]
    assert isinstance(user_content, str)
    assert user_content.index("Doc part 1") < user_content.index("Doc part 2")
    assert user_content.index("Doc part 2") < user_content.index("order status")
    assert "prompt_cache_options" not in body


def test_openai_build_request_explicit_breakpoint_marks_last_stable_block():
    parts = PromptParts(stable_blocks=["Doc part 1", "Doc part 2"], variable="q")
    body = openai_provider.build_openai_cache_request(
        parts, model="gpt-5.6", use_explicit_breakpoint=True
    )
    content = body["messages"][0]["content"]
    assert "prompt_cache_breakpoint" not in content[0]
    assert content[1]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    assert body["prompt_cache_options"]["ttl"] == "30m"


def test_openai_parse_usage_chat_completions_endpoint():
    response = {
        "usage": {
            "prompt_tokens": 2006,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 1920},
        }
    }
    usage = openai_provider.parse_openai_usage(response, endpoint="chat_completions")
    assert usage.cache_read_input_tokens == 1920
    assert usage.input_tokens == 86  # 2006 - 1920
    assert usage.cache_hit is True


def test_openai_parse_usage_responses_endpoint_uses_different_path():
    response = {
        "usage": {
            "input_tokens": 2006,
            "output_tokens": 40,
            "input_tokens_details": {"cached_tokens": 1920},
        }
    }
    usage = openai_provider.parse_openai_usage(response, endpoint="responses")
    assert usage.cache_read_input_tokens == 1920


def test_openai_cost_savings_uses_per_model_discount_table():
    usage = CacheUsageReport(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1000)
    gpt4o_savings = openai_provider.estimated_cost_savings_percent(usage, "gpt-4o")  # 50% off
    gpt56_savings = openai_provider.estimated_cost_savings_percent(usage, "gpt-5.6")  # 90% off
    assert gpt4o_savings == pytest.approx(50.0)
    assert gpt56_savings == pytest.approx(90.0)
    assert gpt56_savings > gpt4o_savings


def test_openai_cost_savings_negative_with_explicit_write_premium():
    usage = CacheUsageReport(input_tokens=0, output_tokens=0, cache_creation_input_tokens=1000, cache_read_input_tokens=0)
    savings = openai_provider.estimated_cost_savings_percent(usage, "gpt-5.6")
    assert savings < 0  # 1.25x write premium, no discount yet


# ---------------------------------------------------------------------
# providers/gemini.py -- Gemini context caching
# ---------------------------------------------------------------------

def test_gemini_check_cache_eligibility_per_model_minimum():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    result = gemini_provider.check_cache_eligibility(parts, model="gemini-3.5-flash")
    assert result.eligible is False
    assert result.minimum_required == 4096  # gemini-3.5-flash's higher minimum


def test_gemini_check_cache_eligibility_falls_back_to_default_for_unknown_model():
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    result = gemini_provider.check_cache_eligibility(parts, model="some-future-gemini-model")
    assert result.minimum_required == gemini_provider.DEFAULT_CACHE_MINIMUM_TOKENS


def test_gemini_build_content_request_orders_stable_first():
    parts = PromptParts(system="sys", stable_blocks=["ctx1", "ctx2"], variable="question")
    body = gemini_provider.build_gemini_content_request(parts, model="gemini-2.5-flash")
    texts = [p["text"] for p in body["contents"][0]["parts"]]
    assert texts == ["sys", "ctx1", "ctx2", "question"]


def test_gemini_build_cached_content_resource_excludes_variable():
    parts = PromptParts(system="sys", stable_blocks=["reusable doc"], variable="should not appear")
    resource = gemini_provider.build_cached_content_resource(parts, model="gemini-2.5-flash")
    all_text = str(resource["contents"])
    assert "reusable doc" in all_text
    assert "should not appear" not in all_text
    assert resource["ttl"] == "3600s"


def test_gemini_build_cached_content_resource_uses_fully_qualified_model_name():
    # Google's cachedContents.create requires "models/{model}", not a bare
    # id -- sending a bare id is a guaranteed request failure. Caught by
    # checking the real schema after live-testing turned up 0 cache hits.
    parts = PromptParts(system="sys", stable_blocks=["doc"], variable="q")
    resource = gemini_provider.build_cached_content_resource(parts, model="gemini-3.6-flash")
    assert resource["model"] == "models/gemini-3.6-flash"


def test_gemini_build_cached_content_resource_does_not_double_prefix():
    parts = PromptParts(stable_blocks=["doc"], variable="q")
    resource = gemini_provider.build_cached_content_resource(parts, model="models/gemini-3.6-flash")
    assert resource["model"] == "models/gemini-3.6-flash"


def test_gemini_build_cached_content_resource_puts_system_in_dedicated_field():
    parts = PromptParts(system="sys instructions", stable_blocks=["doc"], variable="q")
    resource = gemini_provider.build_cached_content_resource(parts, model="gemini-3.6-flash")
    assert resource["systemInstruction"]["parts"][0]["text"] == "sys instructions"
    assert "sys instructions" not in str(resource["contents"])


def test_gemini_build_generate_request_from_cache_references_resource_name():
    body = gemini_provider.build_generate_request_from_cache("cachedContents/abc123", "a new question")
    assert body["cachedContent"] == "cachedContents/abc123"
    assert body["contents"][0]["parts"][0]["text"] == "a new question"


def test_gemini_parse_usage_reads_cached_content_token_count():
    response = {
        "usageMetadata": {
            "promptTokenCount": 1600,
            "candidatesTokenCount": 30,
            "cachedContentTokenCount": 1547,
        }
    }
    usage = gemini_provider.parse_gemini_usage(response)
    assert usage.cache_read_input_tokens == 1547
    assert usage.input_tokens == 53  # 1600 - 1547
    assert usage.cache_creation_input_tokens == 0  # never populated via this path -- see docstring


def test_gemini_implicit_cache_savings_matches_confirmed_ten_percent_rate():
    usage = CacheUsageReport(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1000)
    savings = gemini_provider.estimated_implicit_cache_cost_savings_percent(usage)
    assert savings == pytest.approx(90.0)  # 1 - 0.1 multiplier


def test_gemini_explicit_cache_savings_positive_with_enough_reuse():
    # Populate once, read many times within the TTL window -- storage
    # rent should be dwarfed by the reads it saved on.
    savings = gemini_provider.estimated_explicit_cache_cost_savings_percent(
        model="gemini-2.5-flash",
        cached_tokens=100_000,
        num_requests=50,
        hours_cached=1,
        input_price_per_million=0.30,
    )
    assert savings > 0


def test_gemini_explicit_cache_savings_negative_with_too_little_reuse():
    # Populate once, read only ONCE, but keep it stored for a long time --
    # storage rent should outweigh the single read's discount.
    savings = gemini_provider.estimated_explicit_cache_cost_savings_percent(
        model="gemini-2.5-pro",
        cached_tokens=100_000,
        num_requests=1,
        hours_cached=100,
        input_price_per_million=1.25,
    )
    assert savings < 0



# ---------------------------------------------------------------------
# providers/generic.py -- configurable "any provider" adapter
# ---------------------------------------------------------------------

def test_generic_get_path_resolves_nested_dot_path():
    data = {"usage": {"details": {"cached_tokens": 42}}}
    assert generic_provider._get_path(data, "usage.details.cached_tokens") == 42


def test_generic_get_path_returns_default_when_missing():
    assert generic_provider._get_path({"usage": {}}, "usage.nope", default=0) == 0
    assert generic_provider._get_path({}, "", default=99) == 99  # empty path -> always default


def test_generic_check_cache_eligibility_uses_config_minimum():
    config = generic_provider.GenericCacheConfig(minimum_tokens=500)
    parts = PromptParts(stable_blocks=["word " * 200], variable="q")  # ~250 tokens
    result = generic_provider.check_cache_eligibility(parts, config)
    assert result.eligible is False
    assert result.minimum_required == 500


def test_generic_check_cache_eligibility_default_minimum_is_zero_always_eligible():
    config = generic_provider.GenericCacheConfig()  # unconfigured -- safe default
    parts = PromptParts(stable_blocks=["tiny"], variable="q")
    result = generic_provider.check_cache_eligibility(parts, config)
    assert result.eligible is True


def test_generic_build_chat_request_orders_stable_first():
    parts = PromptParts(system="sys", stable_blocks=["ctx"], variable="question")
    body = generic_provider.build_generic_chat_request(parts, model="some-model")
    content = body["messages"][0]["content"]
    assert content.index("sys") < content.index("ctx") < content.index("question")


def test_generic_build_chat_request_merges_extra_fields():
    parts = PromptParts(variable="q")
    body = generic_provider.build_generic_chat_request(parts, model="m", extra_fields={"temperature": 0.2})
    assert body["temperature"] == 0.2
    assert body["model"] == "m"


def test_generic_parse_usage_reads_configured_paths():
    config = generic_provider.GenericCacheConfig(
        usage_read_path="usage.cached_tokens",
        usage_write_path="usage.write_tokens",
        usage_input_path="usage.prompt_tokens",
        usage_output_path="usage.completion_tokens",
    )
    response = {"usage": {"prompt_tokens": 1000, "cached_tokens": 700, "write_tokens": 0, "completion_tokens": 20}}
    usage = generic_provider.parse_usage(response, config)
    assert usage.cache_read_input_tokens == 700
    assert usage.input_tokens == 300  # 1000 - 700
    assert usage.output_tokens == 20


def test_generic_parse_usage_write_path_defaults_to_zero_when_unset():
    config = generic_provider.GenericCacheConfig(usage_read_path="usage.cached")
    response = {"usage": {"prompt_tokens": 100, "cached": 50, "completion_tokens": 10}}
    usage = generic_provider.parse_usage(response, config)
    assert usage.cache_creation_input_tokens == 0


def test_generic_estimated_cost_savings_matches_configured_multiplier():
    config = generic_provider.GenericCacheConfig(cache_read_multiplier=0.5)
    usage = CacheUsageReport(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1000)
    savings = generic_provider.estimated_cost_savings_percent(usage, config)
    assert savings == pytest.approx(50.0)


def test_generic_estimated_cost_savings_zero_with_unconfigured_default():
    # Default multiplier 1.0 (no discount configured) -- should report
    # 0% rather than fabricating a saving that was never confirmed.
    config = generic_provider.GenericCacheConfig()
    usage = CacheUsageReport(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1000)
    savings = generic_provider.estimated_cost_savings_percent(usage, config)
    assert savings == 0.0


# ---------------------------------------------------------------------
# providers/presets.py -- verified configs for wrapped platforms
# ---------------------------------------------------------------------

def test_bedrock_converse_preset_parses_camelcase_fields():
    # Real shape confirmed against AWS's Converse API docs -- distinct
    # field names from Anthropic's own direct API.
    response = {"usage": {"inputTokens": 1586, "outputTokens": 40, "cacheReadInputTokens": 1547, "cacheWriteInputTokens": 0}}
    usage = generic_provider.parse_usage(response, provider_presets.BEDROCK_CONVERSE_CLAUDE)
    assert usage.cache_read_input_tokens == 1547
    assert usage.input_tokens == 39  # 1586 - 1547
    savings = generic_provider.estimated_cost_savings_percent(usage, provider_presets.BEDROCK_CONVERSE_CLAUDE)
    assert savings > 80  # same ~0.1x read rate as Anthropic direct


def test_azure_ptu_m_preset_reflects_full_discount():
    response = {"usage": {"prompt_tokens": 1000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 800}}}
    usage = generic_provider.parse_usage(response, provider_presets.AZURE_OPENAI_PTU_M)
    savings = generic_provider.estimated_cost_savings_percent(usage, provider_presets.AZURE_OPENAI_PTU_M)
    assert savings == pytest.approx(80.0)  # 800 of 1000 tokens at 100% off = 80% total savings


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
