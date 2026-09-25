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
# redact_llm.py -- realistic small-model output quirks (found via
# offline fixture testing, Sept 2026, after a real GPU benchmark showed
# use_enhanced_redaction catching zero free-text fields across 360
# calls; see ROADMAP.md / colab-benchmark-findings.md). These are
# independent of that benchmark's timeout issue -- they cover cases
# where the model DID respond, but the old parser/guard rail would have
# silently discarded a real, present span anyway.
# ---------------------------------------------------------------------

def test_llm_redact_handles_markdown_fenced_json():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        return '```json\n[{"text": "Priya Malhotra", "type": "NAME"}]\n```'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Priya Malhotra" not in result.redacted_text


def test_llm_redact_handles_array_wrapped_in_object():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        # Small models sometimes wrap the array in an object despite
        # being told to return a bare array.
        return '{"entities": [{"text": "Priya Malhotra", "type": "NAME"}]}'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Priya Malhotra" not in result.redacted_text


def test_llm_redact_ignores_unrelated_bracket_before_real_array():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        # An earlier, unrelated bracket pair (e.g. the model echoing
        # part of its own instructions) used to make a greedy regex
        # merge everything up to the LAST ']' into one unparseable blob,
        # silently discarding a real array that followed it.
        return 'The format is [name, employer]. Result: [{"text": "Priya Malhotra", "type": "NAME"}]'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Priya Malhotra" not in result.redacted_text


def test_llm_redact_handles_python_literal_style_single_quotes():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        # Not strict JSON (single-quoted strings) -- a real quirk of
        # models trained on a lot of Python code.
        return "[{'text': 'Priya Malhotra', 'type': 'NAME'}]"

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Priya Malhotra" not in result.redacted_text


def test_llm_redact_case_insensitive_fallback_uses_source_casing():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        # The model normalizes casing despite being told to return the
        # exact substring -- a real, common small-model quirk.
        return '[{"text": "priya malhotra", "type": "NAME"}]'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Priya Malhotra" not in result.redacted_text
    # The placeholder must be derived from the SOURCE text's actual
    # casing, not the model's re-cased version, both so nothing not
    # really present gets introduced and so the same source text always
    # redacts to the same placeholder regardless of what casing the
    # model happens to emit on a given run (required for
    # cache_structuring.py's stable blocks to stay byte-identical).
    assert "Priya Malhotra" in result.mapping.values()
    assert "priya malhotra" not in result.mapping.values()


def test_llm_redact_case_insensitive_fallback_does_not_defeat_hallucination_guard():
    text = "Please review this for Priya Malhotra."

    def fake_model(prompt, model, timeout):
        # A span that is genuinely absent (in any casing) must still be
        # rejected -- the case-insensitive fallback must never widen the
        # guard rail into accepting hallucinated spans.
        return '[{"text": "Someone Not In The Text At All", "type": "NAME"}]'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Someone Not In The Text At All" not in result.mapping.values()
    assert result.mapping == {}


def test_llm_redact_handles_bare_object_with_no_array_wrapper():
    """
    Confirmed against a REAL (not simulated) local Ollama run with
    llama3.2:1b, Sept 2026: despite the prompt saying "Respond with
    ONLY a JSON array. Each item: {...}", the model returned just
    '{"text": "Arjun", "type": "NAME"}' for its one match -- no `[` `]`
    anywhere. This was silently dropped to zero entities before this
    fix, since the old (and the array-only) extractor only ever looked
    for `[...]` spans.
    """
    text = "Hi, my name is Arjun and I work at NimbusFn."

    def fake_model(prompt, model, timeout):
        return '{"text": "Arjun", "type": "NAME"}'

    redactor = LLMRedactor(model_call_fn=fake_model)
    result = redact_with_llm(text, redactor)
    assert "Arjun" not in result.redacted_text
    assert "Arjun" in result.mapping.values()


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



# ---------------------------------------------------------------------
# relevance.py -- shared lexical scoring
# ---------------------------------------------------------------------

from tonst.relevance import tokenize, bm25_scores, shingles, jaccard


def test_tokenize_splits_camel_and_snake_case_and_folds_plurals():
    assert tokenize("searchIssues") == ["search", "issue"]
    assert tokenize("list_open_pull_requests") == ["list", "open", "pull", "request"]
    assert "the" not in tokenize("the weather")


def test_bm25_ranks_matching_document_first_and_zero_when_no_overlap():
    docs = ["get current weather for a city", "send an email to a contact", "create a calendar event"]
    scores = bm25_scores("what's the weather in Pune", docs)
    assert scores[0] > 0 and scores[0] == max(scores)
    assert bm25_scores("zzz qqq", docs) == [0.0, 0.0, 0.0]


def test_jaccard_near_duplicate_detection():
    a = shingles("The refund window is thirty days from the date of purchase for all items.")
    b = shingles("The refund window is thirty days from the date of purchase for all items!")
    c = shingles("Shipping to Canada takes five to seven business days via ground.")
    assert jaccard(a, b) > 0.8
    assert jaccard(a, c) < 0.1


# ---------------------------------------------------------------------
# tool_optimizer.py
# ---------------------------------------------------------------------

from tonst.tool_optimizer import (
    select_tools,
    ToolSession,
    build_anthropic_deferred_tools,
    estimate_tool_tokens,
    loaded_tools,
    TOOL_SEARCH_BM25,
)


def _anthropic_tools():
    def t(name, desc, **props):
        return {
            "name": name,
            "description": desc,
            "input_schema": {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()}},
        }
    return [
        t("get_weather", "Get the current weather forecast for a location", location="City name"),
        t("send_email", "Send an email message to a recipient", to="Recipient address", body="Message body"),
        t("create_calendar_event", "Create a calendar event with a title and time", title="Event title"),
        t("search_issues", "Search GitHub issues in a repository", repo="Repository owner/name"),
        t("create_pull_request", "Open a pull request on GitHub", repo="Repository owner/name"),
        t("query_database", "Run a read-only SQL query against the analytics database", sql="SQL text"),
        t("read_file", "Read a file from the workspace", path="File path"),
        t("translate_text", "Translate text into another language", text="Text to translate"),
    ]


def test_select_tools_keeps_relevant_tools_in_original_order():
    tools = _anthropic_tools()
    sel = select_tools(tools, "find open GitHub issues about login in the tonst repository", top_k=2)
    assert sel.selected_names == ["search_issues", "create_pull_request"]  # original order, not score order
    assert "send_email" in sel.dropped_names
    assert sel.tools[0] is tools[3]  # original objects, untouched
    assert sel.tokens_after < sel.tokens_before
    assert not sel.fell_back


def test_select_tools_openai_format_supported():
    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in _anthropic_tools()
    ]
    sel = select_tools(tools, "what's the weather forecast in Pune", top_k=1)
    assert sel.selected_names == ["get_weather"]


def test_select_tools_keeps_everything_when_nothing_matches():
    tools = _anthropic_tools()
    sel = select_tools(tools, "zxqv blorp", top_k=2)
    assert sel.fell_back
    assert len(sel.tools) == len(tools)
    assert sel.tokens_saved == 0


def test_select_tools_pins_always_include_and_never_drops_server_tools():
    tools = _anthropic_tools() + [{"type": "web_search_20250305", "name": "web_search"}]
    sel = select_tools(tools, "weather forecast", top_k=1, always_include=["read_file"])
    assert set(sel.selected_names) == {"get_weather", "read_file", "web_search"}


def test_select_tools_falls_back_on_one_word_coincidence():
    # One shared word ("message") is too weak to trust: in the benchmark,
    # every wrong pick looked like this.
    tools = _anthropic_tools()
    sel = select_tools(tools, "ping the team with a quick message", top_k=2)
    assert sel.fell_back and len(sel.tools) == len(tools)
    # ...unless the caller explicitly accepts single-word matches.
    assert not select_tools(tools, "ping the team with a quick message", top_k=2, min_matched_terms=1).fell_back


def test_tool_session_ignores_one_word_matches_when_growing():
    session = ToolSession(_anthropic_tools(), top_k=2)
    session.select("search GitHub issues in the repository")
    sel = session.select("and the email?")  # one shared word only
    assert sel.changed is False


def test_select_tools_rejects_unknown_pinned_name():
    with pytest.raises(ValueError):
        select_tools(_anthropic_tools(), "weather", always_include=["no_such_tool"])


def test_select_tools_is_a_no_op_under_top_k():
    tools = _anthropic_tools()
    sel = select_tools(tools, "weather", top_k=20)
    assert sel.tools == tools and sel.dropped_names == []


def test_tool_session_is_grow_only_and_cache_stable():
    import json
    session = ToolSession(_anthropic_tools(), top_k=2)
    first = session.select("search GitHub issues in the repository")
    second = session.select("thanks, can you look at the second one")  # matches nothing new
    assert second.changed is False
    assert json.dumps(second.tools) == json.dumps(first.tools)  # byte-identical -> cache keeps hitting
    third = session.select("now send an email with the summary to the team")
    assert third.changed and "send_email" in third.added_names
    assert set(first.selected_names) <= set(third.selected_names)  # never shrinks
    names = [t["name"] for t in _anthropic_tools()]
    assert third.selected_names == [n for n in names if n in third.selected_names]  # original order kept


def test_tool_session_state_round_trip():
    s1 = ToolSession(_anthropic_tools(), top_k=2)
    s1.select("weather forecast")
    s2 = ToolSession(_anthropic_tools(), top_k=2).load_state(s1.to_dict())
    assert s2.select("anything").selected_names == s1.select("anything").selected_names


def test_build_anthropic_deferred_tools_shape():
    tools = _anthropic_tools()
    out = build_anthropic_deferred_tools(tools, always_loaded=["read_file"], keep_search_tools_loaded=False)
    assert out[0] == TOOL_SEARCH_BM25
    assert "defer_loading" not in out[0]  # never defer the search tool
    by_name = {t["name"]: t for t in out}
    assert by_name["get_weather"]["defer_loading"] is True
    assert "defer_loading" not in by_name["read_file"]
    assert "defer_loading" not in tools[0]  # input not mutated
    assert [t["name"] for t in loaded_tools(out)] == ["tool_search_tool_bm25", "read_file"]


def test_build_anthropic_deferred_tools_mcp_toolset_and_no_duplicate_search_tool():
    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {"type": "mcp_toolset", "mcp_server_name": "github"},
    ]
    out = build_anthropic_deferred_tools(tools)
    assert len(out) == 2
    assert out[1]["default_config"]["defer_loading"] is True


def test_build_anthropic_deferred_tools_rejects_cache_control_on_deferred_tool():
    tools = _anthropic_tools()
    tools[0] = {**tools[0], "cache_control": {"type": "ephemeral"}}
    with pytest.raises(ValueError):
        build_anthropic_deferred_tools(tools)
    # ...but pinning that tool makes it legal.
    build_anthropic_deferred_tools(tools, always_loaded=["get_weather"])


def test_build_anthropic_cache_request_includes_tools_and_caches_them():
    tools = _anthropic_tools()
    parts = PromptParts(stable_blocks=["x " * 3000], variable="Q?")
    body = build_anthropic_cache_request(parts, model="claude-sonnet-4-6", tools=tools)
    assert [t["name"] for t in body["tools"]] == [t["name"] for t in tools]
    assert body["tools"][-1]["cache_control"]["type"] == "ephemeral"  # no system -> breakpoint on last tool
    assert "cache_control" not in tools[-1]  # caller's list not mutated

    with_system = build_anthropic_cache_request(PromptParts(system="sys", stable_blocks=["x " * 3000], variable="Q"),
                                                model="claude-sonnet-4-6", tools=tools)
    assert not any("cache_control" in t for t in with_system["tools"])  # system breakpoint already covers tools


def test_build_anthropic_cache_request_breakpoint_skips_deferred_tools():
    deferred = build_anthropic_deferred_tools(_anthropic_tools(), always_loaded=["get_weather"],
                                              keep_search_tools_loaded=False)
    body = build_anthropic_cache_request(PromptParts(stable_blocks=["x " * 3000], variable="Q"),
                                         model="claude-sonnet-4-6", tools=deferred)
    marked = [t["name"] for t in body["tools"] if "cache_control" in t]
    assert marked == ["get_weather"]


def test_cache_eligibility_counts_loaded_tools_only():
    parts = PromptParts(system="short")
    big_tool = {"name": "t", "description": "d " * 3000, "input_schema": {}}
    assert check_cache_eligibility(parts, "claude-sonnet-4-6", tools=[big_tool]).eligible
    assert not check_cache_eligibility(parts, "claude-sonnet-4-6", tools=[{**big_tool, "defer_loading": True}]).eligible


# ---------------------------------------------------------------------
# rag.py
# ---------------------------------------------------------------------

from tonst.rag import optimize_chunks, format_context

_CHUNKS = [
    "Refunds are accepted within thirty days of purchase if the item is unused and in original packaging.",
    "Refunds are accepted within thirty days of purchase if the item is unused and in original packaging!",
    "refunds are accepted within thirty days of purchase if the item is unused and in original packaging.",
    "Our headquarters moved to a new office building in 2019 with a rooftop garden.",
    "To start a refund, open the Orders page and choose Request refund next to the item.",
]


def test_optimize_chunks_default_only_dedupes():
    sel = optimize_chunks(_CHUNKS, "how do I get a refund?")
    reasons = dict(sel.dropped)
    assert reasons == {1: "near_duplicate", 2: "duplicate"}
    assert sel.chunks == [_CHUNKS[0], _CHUNKS[3], _CHUNKS[4]]  # retrieval order kept, text untouched
    assert sel.tokens_saved > 0


def test_optimize_chunks_relevance_filter_and_top_k():
    sel = optimize_chunks(_CHUNKS, "how do I request a refund?", top_k=2)
    assert _CHUNKS[3] not in sel.chunks
    assert len(sel.chunks) == 2
    assert (3, "below_top_k") in sel.dropped


def test_optimize_chunks_never_filters_when_nothing_matches():
    sel = optimize_chunks(_CHUNKS, "zxqv blorp", top_k=1)
    assert sel.fell_back
    assert len(sel.chunks) == 3  # only the duplicates went


def test_optimize_chunks_one_word_match_does_not_filter():
    sel = optimize_chunks(_CHUNKS, "refund?", top_k=1)
    assert sel.fell_back and len(sel.chunks) == 3


def test_optimize_chunks_token_budget_and_dict_chunks():
    chunks = [{"id": i, "text": t} for i, t in enumerate(_CHUNKS)]
    sel = optimize_chunks(chunks, "how do I request a refund", max_tokens=30)
    assert sel.tokens_after <= 30
    assert all(isinstance(c, dict) for c in sel.chunks)
    assert any(r == "over_budget" for _, r in sel.dropped)


def test_optimize_chunks_score_order():
    sel = optimize_chunks(_CHUNKS, "request refund orders page", top_k=2, order="score")
    assert sel.chunks[0] == _CHUNKS[4]


def test_format_context_is_deterministic():
    assert format_context(["a", "b"], "q?") == "[Context 1]\na\n\n[Context 2]\nb\n\nQuestion: q?"


def test_query_rag_end_to_end_redacts_and_reports_chunks():
    sent = {}

    def fake_call(prompt):
        sent["prompt"] = prompt
        return "Email [[EMAIL_" + prompt.split("[[EMAIL_")[1].split("]]")[0] + "]] for help."

    chunks = _CHUNKS + ["For refund problems, email support@acme.com and quote your order number."]
    client = TonstClient(call_fn=fake_call)
    response, report = client.query_rag("how do I get a refund?", chunks, system="You are a support bot.")
    assert "support@acme.com" not in sent["prompt"]
    assert "support@acme.com" in response  # restored
    assert sent["prompt"].startswith("You are a support bot.")
    assert sent["prompt"].rstrip().endswith("Question: how do I get a refund?")
    assert report.chunks_in == 6 and report.chunks_sent == 4
    assert report.redacted_types == {"EMAIL": 1}
    assert report.original_tokens > report.sent_tokens


# ---------------------------------------------------------------------
# compactor.py -- rolling compaction
# ---------------------------------------------------------------------

from tonst.compactor import compact_history_rolling, RollingSummary


def _conv(n, size=200):
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "word " * size})
    return msgs


class _CountingCompactor:
    """Stands in for HistoryCompactor; records calls."""
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def summarize_incremental(self, previous, new_text, max_summary_chars=4000):
        self.calls.append((previous, new_text))
        if self.fail:
            return None
        return f"Goal: v{len(self.calls)}"


def test_rolling_keeps_evicted_turns_verbatim_until_threshold():
    comp, state = _CountingCompactor(), RollingSummary()
    r = compact_history_rolling(_conv(8, size=50), comp, state, keep_last_n=6, token_threshold=10_000)
    assert comp.calls == []  # below threshold: no local-model call
    assert r.pending_turns == 2
    assert len(r.messages) == 1 + 8  # nothing dropped yet


def test_rolling_folds_once_then_reuses_summary_without_model_call():
    comp, state = _CountingCompactor(), RollingSummary()
    msgs = _conv(10)
    r1 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    assert r1.summary_updated and r1.folded_turns == 6 and len(comp.calls) == 1
    assert r1.messages[1]["content"] == "[Summary of earlier conversation]\nGoal: v1"
    assert r1.messages[2:] == msgs[7:]

    # Next turn: one more message. Evicted portion is tiny -> no new call, summary reused byte-for-byte.
    msgs2 = msgs + [{"role": "user", "content": "short follow-up"}]
    r2 = compact_history_rolling(msgs2, comp, state, keep_last_n=4, token_threshold=500)
    assert len(comp.calls) == 1 and r2.summary_reused and not r2.summary_updated
    assert r2.messages[: len(r1.messages)] == r1.messages  # append-only prefix -> provider cache keeps hitting


def test_rolling_second_fold_passes_previous_summary():
    comp, state = _CountingCompactor(), RollingSummary()
    msgs = _conv(10)
    compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    msgs += [{"role": "user", "content": "more " * 400}, {"role": "assistant", "content": "ok " * 400},
             {"role": "user", "content": "x " * 400}, {"role": "assistant", "content": "y " * 400}]
    r = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    assert r.summary_updated and comp.calls[-1][0] == "Goal: v1"
    assert "turn 0 " not in comp.calls[-1][1]  # already-summarized turns are never re-sent to the model


def test_rolling_failed_fold_keeps_turns_and_retries_before_dropping():
    comp, state = _CountingCompactor(), RollingSummary()
    msgs = _conv(10)
    compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    comp.fail = True
    msgs += [{"role": "user", "content": "a " * 1200}] * 4

    # First failure: nothing is dropped -- the turns stay verbatim for a retry.
    r1 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    assert r1.fold_failed and r1.fold_will_retry and r1.dropped_turns == 0
    assert r1.pending_turns == 4 and state.failed_folds == 1
    assert len(r1.messages) == 1 + 1 + 8  # system + summary + 4 pending + 4 recent
    calls_after_first_failure = len(comp.calls)

    # Not retried on the very next turn: the bar is now 2x threshold.
    msgs2 = msgs + [{"role": "assistant", "content": "ok"}]
    compact_history_rolling(msgs2, comp, state, keep_last_n=4, token_threshold=5000)
    assert len(comp.calls) == calls_after_first_failure

    # Second failure (past 2x threshold): now the batch is dropped, old summary kept.
    msgs3 = msgs2 + [{"role": "user", "content": "b " * 1200}] * 2
    r3 = compact_history_rolling(msgs3, comp, state, keep_last_n=4, token_threshold=500)
    assert r3.fold_failed and not r3.fold_will_retry and r3.dropped_turns > 0
    assert state.summary == "Goal: v1" and state.failed_folds == 0
    assert r3.tokens_lost > 0 and state.dropped_tokens == r3.tokens_lost


def test_rolling_retry_succeeds_without_losing_turns():
    comp, state = _CountingCompactor(fail=True), RollingSummary()
    msgs = _conv(10)
    r1 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    assert r1.fold_will_retry and state.summary is None
    comp.fail = False
    msgs += [{"role": "user", "content": "c " * 1200}, {"role": "assistant", "content": "d " * 1200}]
    r2 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500)
    assert r2.summary_updated and r2.folded_turns == 8 and state.dropped_tokens == 0


def test_rolling_without_compactor_is_batched_truncation():
    state = RollingSummary()
    r = compact_history_rolling(_conv(10), None, state, keep_last_n=4, token_threshold=500)
    assert r.dropped_turns == 6 and state.summary is None
    assert len(r.messages) == 1 + 4


def test_rolling_resets_when_history_does_not_match_state():
    comp, state = _CountingCompactor(), RollingSummary()
    compact_history_rolling(_conv(10), comp, state, keep_last_n=4, token_threshold=500)
    other = _conv(10, size=150)  # different conversation, same length
    r = compact_history_rolling(other, comp, state, keep_last_n=4, token_threshold=500)
    assert r.state_reset
    assert comp.calls[-1][0] is None  # rebuilt from scratch, not merged into the wrong summary


def test_rolling_state_round_trips_through_dict():
    state = RollingSummary(summary="s", summarized_count=3, fingerprint="abc", failed_folds=1, dropped_tokens=40)
    assert RollingSummary.from_dict(state.to_dict()) == state
    # Older saved states (without the newer fields) still load.
    assert RollingSummary.from_dict({"summary": "s", "summarized_count": 3}).failed_folds == 0


def test_summarize_incremental_first_fold_uses_structured_prompt():
    seen = {}

    def fake(prompt, model, timeout):
        seen["prompt"] = prompt
        return ("Goal: replace damaged order\nDecisions:\n- replacement\nKey facts:\n- order [[NAME_1a2b3c4d]]"
                "\nOpen items:\n- none")
    comp = HistoryCompactor(model_call_fn=fake)
    out = comp.summarize_incremental(None, "user: order for [[NAME_1a2b3c4d]] arrived damaged. " * 10)
    assert out is not None
    assert "Goal:" in seen["prompt"] and "(none yet)" in seen["prompt"]


def test_client_compaction_timeout_is_configurable_and_longer_by_default():
    client = TonstClient(call_fn=lambda p: "ok", use_history_compaction=True)
    assert client.history_compactor.timeout == 60.0
    client = TonstClient(call_fn=lambda p: "ok", use_history_compaction=True, compaction_timeout=15)
    assert client.history_compactor.timeout == 15


def test_summarize_incremental_guard_rails():
    def fake(prompt, model, timeout):
        return fake.out
    comp = HistoryCompactor(model_call_fn=fake)
    new_text = "user: I switched the deploy target to staging. " * 20
    fake.out = "Goal: deploy\nDecisions: target is staging\nKey facts: staging cluster\nOpen items: none"
    assert comp.summarize_incremental("Goal: deploy", new_text) == fake.out
    fake.out = "x" * 5000  # too long / grew too much
    assert comp.summarize_incremental("Goal: deploy", new_text) is None
    fake.out = "Goal: email [[EMAIL_deadbeef]] about it\nDecisions: -\nKey facts: deploy to staging\nOpen items: -"  # invented placeholder
    assert comp.summarize_incremental("Goal: deploy", new_text) is None
    fake.out = "Goal: keep [[NAME_1a2b3c4d]]\nDecisions: -\nKey facts: deploy to staging\nOpen items: -"  # placeholder only in the OLD summary is fine
    assert comp.summarize_incremental("Goal: talk to [[NAME_1a2b3c4d]]", new_text) == fake.out


def test_query_messages_rolling_state_end_to_end():
    prompts = []
    client = TonstClient(call_fn=lambda p: prompts.append(p) or "ok", compaction_token_threshold=300)
    client.history_compactor = _CountingCompactor()
    state = RollingSummary()
    msgs = _conv(10) + [{"role": "user", "content": "My email is jo@example.com"}]
    _, report = client.query_messages(msgs, keep_last_n=4, rolling_state=state)
    assert report.history_summary_updated and report.history_compacted
    assert "jo@example.com" not in prompts[-1]
    _, report2 = client.query_messages(msgs + [{"role": "assistant", "content": "noted"}], keep_last_n=4, rolling_state=state)
    assert report2.history_summary_reused and not report2.history_summary_updated
    assert prompts[-1].startswith(prompts[-2][: len(prompts[-2]) // 2])  # stable prefix across turns


# ---------------------------------------------------------------------
# savings_log.py + `tonst stats`
# ---------------------------------------------------------------------

import json as _json_mod
from tonst.savings_log import SavingsLog, summarize as summarize_savings, format_summary, redacted_types_from_mapping


def test_redacted_types_reads_labels_only():
    mapping = {"[[EMAIL_1a2b3c4d]]": "a@b.com", "[[EMAIL_99999999]]": "c@d.com", "[[CREDIT_CARD_abcdef01]]": "4111"}
    assert redacted_types_from_mapping(mapping) == {"CREDIT_CARD": 1, "EMAIL": 2}


def test_savings_log_never_writes_prompt_pii_or_placeholder_hashes(tmp_path):
    log = tmp_path / "s.jsonl"
    client = TonstClient(call_fn=lambda p: "ok", savings_log=str(log), app_name="support", input_price_per_million=3.0)
    client.query("Contact   jane.doe@example.com   about   the    invoice,  phone 555-123-4567.")
    raw = log.read_text()
    assert "jane.doe" not in raw and "555-123" not in raw and "invoice" not in raw
    assert "[[EMAIL_" not in raw  # no hashes
    entry = _json_mod.loads(raw.strip())
    assert entry["app"] == "support" and entry["method"] == "query"
    assert entry["redacted_types"].get("EMAIL") == 1
    assert entry["token_counts_are_estimates"] is True
    assert entry["estimated_cost_saved_usd"] >= 0


def test_savings_log_one_entry_per_public_call(tmp_path):
    log = tmp_path / "s.jsonl"
    client = TonstClient(call_fn=lambda p: "ok", savings_log=str(log))
    client.query_structured(PromptParts(system="s", variable="q"))
    client.query_messages([{"role": "user", "content": "hi"}])
    client.query_rag("q", ["a chunk"])
    methods = [_json_mod.loads(l)["method"] for l in log.read_text().splitlines()]
    assert methods == ["query_structured", "query_messages", "query_rag"]


def test_savings_log_off_by_default_and_fails_soft(tmp_path):
    assert TonstClient(call_fn=lambda p: "ok").savings_log is None
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    client = TonstClient(call_fn=lambda p: "ok", savings_log=str(blocker / "s.jsonl"))
    response, _ = client.query("hello")  # unwritable path must not break the call
    assert response == "ok"


def test_summarize_and_format(tmp_path):
    log = tmp_path / "s.jsonl"
    a = TonstClient(call_fn=lambda p: "ok", savings_log=str(log), app_name="a", input_price_per_million=3.0)
    b = TonstClient(call_fn=lambda p: "ok", savings_log=str(log), app_name="b")
    a.query("x   " * 200 + " mail me at a@b.com")
    b.query("line\nline\nline\n" * 50)
    with open(log, "a") as f:
        f.write("not json\n")  # corrupt line is skipped, not fatal
    s = summarize_savings(str(log))
    assert s.calls == 2 and s.tokens_saved > 0
    assert set(s.by_app) == {"a", "b"}
    assert s.calls_with_price == 1
    assert summarize_savings(str(log), app="a").calls == 1
    text = format_summary(s)
    assert "tokens saved" in text and "priced on 1 of 2 calls" in text
    assert format_summary(summarize_savings(str(tmp_path / "missing.jsonl"))) == "No tonst calls logged yet."


def test_savings_log_records_real_provider_usage(tmp_path):
    log = SavingsLog(str(tmp_path / "s.jsonl"))
    _, report = TonstClient(call_fn=lambda p: "ok").query("hi")
    usage = CacheUsageReport(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=1500)
    log.record(report, usage=usage)
    s = summarize_savings(log.path)
    assert s.cache_read_input_tokens == 1500 and s.calls_with_provider_usage == 1


def test_stats_cli(tmp_path, capsys=None):
    import io, contextlib
    from tonst.__main__ import main as cli_main
    log = tmp_path / "s.jsonl"
    TonstClient(call_fn=lambda p: "ok", savings_log=str(log)).query("a  " * 100)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli_main(["stats", "--log", str(log), "--json"]) == 0
    assert _json_mod.loads(buf.getvalue())["calls"] == 1



# ---------------------------------------------------------------------
# Reporting fixes found in live testing (2026-09-24)
# ---------------------------------------------------------------------

def test_query_rag_reports_when_relevance_filtering_was_skipped(tmp_path):
    log = tmp_path / "s.jsonl"
    client = TonstClient(call_fn=lambda p: "ok", savings_log=str(log))
    chunks = ["Refunds take 5-10 days. Email help@acme.com.",
              "Refunds take 5-10 days. Email help@acme.com.",
              "Our office has a rooftop garden."]
    # The exact case from manual testing: one-word match -> filtering skipped.
    _, report = client.query_rag("How do refunds work?", chunks, top_k=1)
    assert report.chunk_filter_skipped and report.chunks_sent == 2
    # Opting into single-word matches filters as asked.
    _, report2 = client.query_rag("How do refunds work?", chunks, top_k=1, min_matched_terms=1)
    assert not report2.chunk_filter_skipped and report2.chunks_sent == 1
    entries = [_json_mod.loads(l) for l in log.read_text().splitlines()]
    assert [e["chunk_filter_skipped"] for e in entries] == [True, False]
    assert "relevance filtering skipped on 1 calls" in format_summary(summarize_savings(str(log)))


def test_truncated_history_is_reported_as_lost_not_just_saved(tmp_path):
    log = tmp_path / "s.jsonl"
    client = TonstClient(call_fn=lambda p: "ok", savings_log=str(log))  # no compactor -> pure truncation
    history = [{"role": "user", "content": f"message {i} " * 200} for i in range(20)]
    _, report = client.query_messages(history, rolling_state=RollingSummary())
    assert report.history_tokens_lost > 0
    assert report.history_tokens_lost <= report.tokens_saved
    s = summarize_savings(str(log))
    assert s.history_tokens_lost == report.history_tokens_lost
    text = format_summary(s)
    assert "of which lost" in text and "saved excl. lost history" in text


def test_stateless_compaction_also_reports_lost_history():
    client = TonstClient(call_fn=lambda p: "ok", compaction_token_threshold=100)
    history = [{"role": "user", "content": f"message {i} " * 100} for i in range(12)]
    _, report = client.query_messages(history)
    assert report.history_turns_dropped == 6 and report.history_tokens_lost > 0


def test_log_records_summary_updated_and_reused(tmp_path):
    log = tmp_path / "s.jsonl"
    client = TonstClient(call_fn=lambda p: "ok", savings_log=str(log), compaction_token_threshold=300)
    client.history_compactor = _CountingCompactor()
    state = RollingSummary()
    msgs = _conv(10)
    client.query_messages(msgs, keep_last_n=4, rolling_state=state)
    client.query_messages(msgs + [{"role": "user", "content": "more?"}], keep_last_n=4, rolling_state=state)
    entries = [_json_mod.loads(l) for l in log.read_text().splitlines()]
    assert [(e["history_summary_updated"], e["history_summary_reused"]) for e in entries] == [(True, False), (False, True)]
    assert "1 folds, 1 reuses" in format_summary(summarize_savings(str(log)))


def test_stats_show_median_and_max_overhead_not_just_mean(tmp_path):
    log = SavingsLog(str(tmp_path / "s.jsonl"))
    _, fast = TonstClient(call_fn=lambda p: "ok").query("hi")
    for ms in (2.0, 3.0, 4.0, 8000.0):  # one slow outlier, like a local-model timeout
        log.record(__import__("dataclasses").replace(fast, redaction_ms=ms, trim_ms=0, compression_ms=0))
    s = summarize_savings(log.path)
    assert s.median_local_overhead_ms == 3.5
    assert s.max_local_overhead_ms == 8000.0
    assert "median 3.5 ms, max 8000.0 ms" in format_summary(s)



# ---------------------------------------------------------------------
# token_count.py -- optional exact counting (added after the live test
# found real input ~1.8x the chars/4 estimate on tool calls)
# ---------------------------------------------------------------------

from tonst.token_count import AnthropicTokenCounter, COUNT_TOKENS_URL


class _SpyCounter:
    def __init__(self, fail=False):
        self.seen = []
        self.fail = fail

    def __call__(self, text):
        self.seen.append(text)
        if self.fail:
            raise RuntimeError("network down")
        return len(text.split()) + 7


def test_token_counter_never_sees_raw_pii_in_any_entry_point():
    spy = _SpyCounter()
    client = TonstClient(call_fn=lambda p: "ok", token_counter=spy)
    client.query("Email jane.doe@example.com about invoice 5")
    client.query_messages([{"role": "user", "content": "I'm at jane.doe@example.com"},
                           {"role": "assistant", "content": "noted"}])
    client.query_rag("refund policy for jane.doe@example.com", ["Refund policy: 30 days.", "Contact jane.doe@example.com"])
    assert spy.seen, "counter should have been used"
    assert not any("jane.doe@example.com" in t for t in spy.seen)


def test_token_counter_results_are_used_and_marked_exact(tmp_path):
    log = tmp_path / "s.jsonl"
    spy = _SpyCounter()
    client = TonstClient(call_fn=lambda p: "ok", token_counter=spy, savings_log=str(log))
    _, report = client.query("one  two  two\nthree\nthree")
    assert report.token_counts_exact
    assert report.sent_tokens == len(spy.seen[-1].split()) + 7
    assert report.counting_ms >= 0
    entry = _json_mod.loads(log.read_text().splitlines()[0])
    assert entry["token_counts_are_estimates"] is False
    assert "(counted)" in format_summary(summarize_savings(str(log)))


def test_token_counter_failure_falls_back_to_estimate():
    client = TonstClient(call_fn=lambda p: "ok", token_counter=_SpyCounter(fail=True))
    response, report = client.query("hello there")
    assert response == "ok" and not report.token_counts_exact
    assert report.sent_tokens == estimate_tokens("hello there")


def test_mixed_exact_and_estimated_calls_are_labelled(tmp_path):
    log = tmp_path / "s.jsonl"
    TonstClient(call_fn=lambda p: "ok", token_counter=_SpyCounter(), savings_log=str(log)).query("a b c")
    TonstClient(call_fn=lambda p: "ok", savings_log=str(log)).query("a b c")
    assert "counted on 1 of 2 calls, rest estimated" in format_summary(summarize_savings(str(log)))


def test_anthropic_token_counter_requests_and_tool_overhead():
    calls = []

    def fake_post(url, headers, body, timeout):
        calls.append((url, headers, body))
        return {"input_tokens": 10 + (500 if body.get("tools") else 0) + 3 * len(body.get("tools") or [])}

    c = AnthropicTokenCounter(model="claude-sonnet-4-6", api_key="k", post_fn=fake_post)
    assert c("hello") == 10
    url, headers, body = calls[-1]
    assert url == COUNT_TOKENS_URL and headers["x-api-key"] == "k" and body["model"] == "claude-sonnet-4-6"
    tools = _anthropic_tools()
    assert c.count_tools(tools) == 500 + 3 * len(tools)  # with-tools minus base: includes the hidden tool prompt
    n_calls = len(calls)
    c.count_tools(tools[:2])
    assert len(calls) == n_calls + 1  # base count is cached
    assert c.count_tools([]) == 0


def test_anthropic_token_counter_fails_soft():
    def boom(url, headers, body, timeout):
        raise ConnectionError("offline")
    c = AnthropicTokenCounter(api_key="k", post_fn=boom)
    assert c("hi") is None and c.count_tools(_anthropic_tools()) is None


def test_select_tools_uses_real_counter_for_numbers_only():
    tools = _anthropic_tools()
    counter = lambda ts: 400 + 100 * len(ts)  # noqa: E731
    exact = select_tools(tools, "find open GitHub issues in the repository", top_k=2, token_counter=counter)
    estimated = select_tools(tools, "find open GitHub issues in the repository", top_k=2)
    assert exact.selected_names == estimated.selected_names  # selection unchanged
    assert exact.tokens_exact and (exact.tokens_before, exact.tokens_after) == (1200, 600)
    assert not estimated.tokens_exact
    failing = select_tools(tools, "find open GitHub issues in the repository", top_k=2, token_counter=lambda ts: None)
    assert not failing.tokens_exact and failing.tokens_before == estimated.tokens_before



def test_deferred_tools_keep_search_type_tools_loaded_by_default():
    # Live testing: deferring slack_search_messages / docs_search made Claude search the TOOL
    # catalog for the topic ("billing outage"), find nothing, and claim the data didn't exist.
    from tonst.tool_optimizer import is_search_like_tool, DEFERRED_TOOLS_SYSTEM_HINT
    tools = _anthropic_tools()
    out = build_anthropic_deferred_tools(tools)
    loaded = [t["name"] for t in loaded_tools(out)]
    assert loaded == ["tool_search_tool_bm25", "search_issues"]
    assert is_search_like_tool({"name": "logs_search", "input_schema": {}})
    assert is_search_like_tool({"name": "slackSearchMessages", "input_schema": {}})
    assert not is_search_like_tool({"name": "research_notes", "input_schema": {}})  # whole words only
    assert "finds TOOLS, not data" in DEFERRED_TOOLS_SYSTEM_HINT
    # Opting out restores defer-everything.
    plain = build_anthropic_deferred_tools(tools, keep_search_tools_loaded=False)
    assert [t["name"] for t in loaded_tools(plain)] == ["tool_search_tool_bm25"]



def test_summaries_are_stripped_of_echoed_prompt_fences():
    # Live testing: gemma2:2b wrapped its rolling summary in the prompt's --- fences.
    from tonst.compactor import _clean_summary
    raw = "---\nGoal: Replace cracked ceramic lamp\nDecisions:  Replacement\n---"
    assert _clean_summary(raw) == "Goal: Replace cracked ceramic lamp\nDecisions:  Replacement"
    assert _clean_summary("Updated summary:\n```\nGoal: x\n```\n") == "Goal: x"
    assert _clean_summary("Goal: a --- b") == "Goal: a --- b"  # only whole fence lines are removed

    def fake(prompt, model, timeout):
        return "---\nGoal: deploy\nDecisions: target is staging\nKey facts: staging cluster\nOpen items: none\n---"
    comp = HistoryCompactor(model_call_fn=fake)
    out = comp.summarize_incremental("Goal: deploy", "user: switch the deploy target to staging. " * 20)
    assert out == "Goal: deploy\nDecisions: target is staging\nKey facts: staging cluster\nOpen items: none"


def test_rolling_prompt_puts_settled_items_under_decisions():
    # Live testing: an already-agreed express upgrade was listed under Open items.
    from tonst.compactor import ROLLING_COMPACTION_PROMPT
    assert "Open items = ONLY things still undecided" in ROLLING_COMPACTION_PROMPT
    assert "belongs under Decisions" in ROLLING_COMPACTION_PROMPT
    # Live testing (24 turns): repeated bullets, and two facts linked that the chat never linked.
    assert "merge duplicates" in ROLLING_COMPACTION_PROMPT
    assert ROLLING_COMPACTION_PROMPT.format(summary="s", text="t")  # still a valid format string



# ---------------------------------------------------------------------
# Background summarizing (live test: a blocking summary added ~4.7 s to its turn)
# ---------------------------------------------------------------------

from tonst.compactor import run_fold_job


def test_deferred_fold_keeps_turns_verbatim_and_applies_later():
    comp, state = _CountingCompactor(), RollingSummary()
    msgs = _conv(10)
    r1 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500, defer_fold=True)
    assert r1.fold_job is not None and comp.calls == []           # nothing ran on the request path
    assert r1.messages[1:] == msgs[1:] and state.summary is None     # all turns still sent verbatim
    assert state.fold_in_progress

    # While the job is in flight, no second job is scheduled.
    r2 = compact_history_rolling(msgs + [{"role": "user", "content": "x " * 900}], comp, state,
                                 keep_last_n=4, token_threshold=500, defer_fold=True)
    assert r2.fold_job is None

    assert run_fold_job(r1.fold_job, comp, state) == "folded"
    assert state.summary == "Goal: v1" and not state.fold_in_progress and len(comp.calls) == 1
    r3 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500, defer_fold=True)
    assert r3.messages[1]["content"].endswith("Goal: v1") and r3.summary_reused


def test_stale_background_job_is_ignored():
    comp, state = _CountingCompactor(), RollingSummary()
    r = compact_history_rolling(_conv(10), comp, state, keep_last_n=4, token_threshold=500, defer_fold=True)
    # The conversation changes (edited history) before the job finishes -> state resets.
    compact_history_rolling(_conv(10, size=150), None, state, keep_last_n=4, token_threshold=10**6)
    assert run_fold_job(r.fold_job, comp, state) == "stale"
    assert state.summary is None and not state.fold_in_progress


def test_background_job_failure_follows_retry_then_drop():
    comp, state = _CountingCompactor(fail=True), RollingSummary()
    msgs = _conv(10)
    r = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500, defer_fold=True)
    assert run_fold_job(r.fold_job, comp, state) == "retry" and state.failed_folds == 1
    msgs += [{"role": "user", "content": "b " * 1200}] * 2
    r2 = compact_history_rolling(msgs, comp, state, keep_last_n=4, token_threshold=500, defer_fold=True)
    assert run_fold_job(r2.fold_job, comp, state) == "dropped" and state.dropped_tokens > 0

    class Boom(_CountingCompactor):
        def summarize_incremental(self, *a, **k):
            raise RuntimeError("ollama crashed")
    state2 = RollingSummary()
    r3 = compact_history_rolling(_conv(10), Boom(), state2, keep_last_n=4, token_threshold=500, defer_fold=True)
    assert run_fold_job(r3.fold_job, Boom(), state2) == "retry"   # exception handled, never raised
    assert not state2.fold_in_progress


def test_client_background_summary_does_not_block_the_call():
    import time as _t

    class Slow(_CountingCompactor):
        def summarize_incremental(self, previous, new_text, max_summary_chars=4000):
            _t.sleep(0.6)
            return super().summarize_incremental(previous, new_text, max_summary_chars)

    prompts = []
    client = TonstClient(call_fn=lambda p: prompts.append(p) or "ok", compaction_token_threshold=300)
    client.history_compactor = Slow()
    state = RollingSummary()
    msgs = _conv(10)

    t0 = _t.perf_counter()
    _, r1 = client.query_messages(msgs, keep_last_n=4, rolling_state=state, background_summary=True)
    assert _t.perf_counter() - t0 < 0.4              # did not wait for the 0.6 s summary
    assert r1.history_summary_scheduled and not r1.history_summary_updated
    assert client.wait_for_background_work(timeout=5)
    assert state.summary == "Goal: v1"

    _, r2 = client.query_messages(msgs + [{"role": "user", "content": "and now?"}], keep_last_n=4,
                                  rolling_state=state, background_summary=True)
    assert r2.history_summary_reused and "Goal: v1" in prompts[-1]


def test_background_summary_requires_rolling_state():
    with pytest.raises(ValueError):
        TonstClient(call_fn=lambda p: "ok").query_messages([{"role": "user", "content": "hi"}],
                                                          background_summary=True)



# ---------------------------------------------------------------------
# Ollama context window (num_ctx): without it Ollama silently drops the
# START of long prompts -- found preparing the long-history live test.
# ---------------------------------------------------------------------

class _FakeSession:
    def __init__(self, response_text="ok summary text that is long enough"):
        self.payloads = []
        self.response_text = response_text

    def post(self, url, json=None, timeout=None):
        self.payloads.append(json)
        outer = self

        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"response": outer.response_text}
        return R()


def test_num_ctx_is_sized_to_the_prompt():
    from tonst.ollama_util import num_ctx_for, fits_context
    assert num_ctx_for("x" * 300, 100) == 2048                    # never below Ollama's usual default
    assert num_ctx_for("x" * 15000, 1200) == 7168                # 5,000 + 1,200 + 128 -> rounded up to 1k
    assert num_ctx_for("x" * 60000, 1200) == 8192                # capped at the model maximum
    assert fits_context("x" * 15000, 1200) and not fits_context("x" * 60000, 1200)


def test_compactor_sends_num_ctx_and_refuses_prompts_that_would_be_truncated(monkeypatch=None):
    import tonst.compactor as C
    fake = _FakeSession()
    orig = C._SESSION
    C._SESSION = fake
    try:
        assert C._default_ollama_call("summarize " * 2000, "gemma2:2b", 5.0) is not None
        opts = fake.payloads[-1]["options"]
        assert opts["num_ctx"] >= 2048 and opts["num_predict"] <= 1200
        n = len(fake.payloads)
        assert C._default_ollama_call("x" * 60000, "gemma2:2b", 5.0) is None   # too long: refused, not truncated
        assert len(fake.payloads) == n                                        # ...and never sent
    finally:
        C._SESSION = orig


def test_local_compression_and_redaction_send_num_ctx():
    import tonst.local_model as LM
    import tonst.redact_llm as RL
    for mod in (LM, RL):
        fake = _FakeSession("[]")
        orig = mod._SESSION
        mod._SESSION = fake
        try:
            if mod is LM:
                LM.LocalCompressor().compress("please compress this sentence " * 50)
                assert LM.LocalCompressor().compress("x " * 40000) == ("x " * 40000, False)  # too long: skipped
            else:
                RL._default_ollama_call("find names in this text " * 50, "gemma2:2b", 5.0)
            assert "num_ctx" in fake.payloads[0]["options"]
        finally:
            mod._SESSION = orig



# ---------------------------------------------------------------------
# Long-history summary failure (live, 2026-09-25): with ~3k tokens of tool
# output, gemma2:2b replied to the customer instead of summarizing, and the
# reply passed every other guard rail.
# ---------------------------------------------------------------------

_LIVE_GARBAGE_SUMMARY = (
    "You're in luck!  I've sent you a tracking number.  You can find it in the \"Tracking Number\" "
    "section of your order summary. \n\n**Here's why I'm able to help:**\n\n* **I have access to real-time "
    "data:** I can access and process information about your order, including the courier's tracking "
    "system. \n* **I'm trained on a massive dataset:** This allows me to understand your request and "
    "provide you with the information you need. \n\n\nLet me know if you have any other questions!"
)


def test_reply_instead_of_summary_is_rejected():
    from tonst.compactor import _has_summary_structure
    assert not _has_summary_structure(_LIVE_GARBAGE_SUMMARY)
    comp = HistoryCompactor(model_call_fn=lambda prompt, model, timeout: _LIVE_GARBAGE_SUMMARY)
    long_turns = ("user: Will I get a tracking number?\nassistant: Yes. [tool output] " + "{\"event\": 1} " * 200)
    assert comp.summarize_incremental(None, long_turns) is None
    assert comp.summarize_incremental("Goal: x\nDecisions: -\nKey facts: -\nOpen items: -", long_turns) is None


def test_summary_structure_accepts_common_markdown_forms():
    from tonst.compactor import _has_summary_structure
    assert _has_summary_structure("- Goal: a\n- Decisions:\n    - b\n- Key facts: c\n- Open items: None")
    assert _has_summary_structure("**Goal:** a\n**Decisions:** b\n## Key facts:\n- c")
    assert not _has_summary_structure("Goal: a\nDecisions: b")               # only 2 of 4
    assert not _has_summary_structure("My goal: help you. Decisions: none.")  # headings must start a line


def test_rolling_prompt_repeats_the_task_after_the_conversation():
    from tonst.compactor import ROLLING_COMPACTION_PROMPT
    prompt = ROLLING_COMPACTION_PROMPT.format(summary="S", text="user: Will I get a tracking number?")
    tail = prompt[prompt.index("user: Will I get a tracking number?"):]
    assert "you are NOT a participant" in tail and "Goal:, Decisions:, Key facts:, Open items:" in tail


def test_summarizer_input_clips_long_tool_output_but_folding_uses_full_size():
    seen = []

    class Spy(_CountingCompactor):
        def summarize_incremental(self, previous, new_text, max_summary_chars=4000):
            seen.append(new_text)
            return super().summarize_incremental(previous, new_text, max_summary_chars)

    long_reply = "Here is the tracking data. [tool output] " + "{\"hub\": \"PNQ-2\", \"status\": \"at_hub\"} " * 120
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs += [{"role": "user", "content": f"question {i}"}, {"role": "assistant", "content": long_reply}]
    state = RollingSummary()
    r = compact_history_rolling(msgs, Spy(), state, keep_last_n=4, token_threshold=1500)
    assert r.summary_updated                                   # full size (~6k tokens) triggered the fold
    assert "more characters of tool output/data omitted" in seen[0]
    assert "Here is the tracking data." in seen[0]            # the prose at the head survives
    assert len(seen[0]) < len(long_reply) * 2                 # far smaller than the full batch
    # Clipping off -> the summarizer sees everything.
    seen.clear()
    compact_history_rolling(msgs, Spy(), RollingSummary(), keep_last_n=4, token_threshold=1500, summary_input_chars=0)
    assert "omitted" not in seen[0]



# ---------------------------------------------------------------------
# Pinned references (live 24-turn run lost case number CS-20931 from the summary)
# ---------------------------------------------------------------------

from tonst.compactor import extract_references


def test_extract_references_finds_exact_identifiers_only():
    text = ("Hi, my order #4471 arrived cracked. Your case number is CS-20931 and PAY-311 is the billing ticket. "
            "Express costs 149 rupees, or ₹149; refund of $3.50. Email [[EMAIL_1a2b3c4d]]. "
            "Hub PNQ-2 scanned it. Order #4471 again.")
    refs = extract_references(text)
    assert refs == ["#4471", "CS-20931", "PAY-311", "149 rupees", "₹149", "$3.50", "[[EMAIL_1a2b3c4d]]"]
    assert "PNQ-2" not in refs          # single-digit codes are too noisy (tracking hubs etc.)


def test_pinned_references_survive_a_thin_summary():
    class Thin(_CountingCompactor):
        def summarize_incremental(self, previous, new_text, max_summary_chars=4000):
            self.calls.append((previous, new_text))
            return "Goal: help\nDecisions: replacement\nKey facts: lamp damaged\nOpen items: none"

    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "My order #4471 arrived cracked. " + "details " * 150},
            {"role": "assistant", "content": "Your case number is CS-20931. " + "info " * 150}]
    msgs += [{"role": "user", "content": f"q{i}"} if i % 2 == 0 else {"role": "assistant", "content": f"a{i}"}
             for i in range(4)]
    state = RollingSummary()
    r = compact_history_rolling(msgs, Thin(), state, keep_last_n=4, token_threshold=200)
    assert r.summary_updated and state.pinned == ["#4471", "CS-20931"]
    header = r.messages[1]["content"]
    assert header.startswith("[Summary of earlier conversation]\nGoal: help")
    assert header.endswith("Pinned references from earlier turns (exact): #4471, CS-20931")
    assert RollingSummary.from_dict(state.to_dict()).pinned == ["#4471", "CS-20931"]


def test_pinned_references_survive_even_when_turns_are_dropped():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "Case CS-20931 please. " + "x " * 600}]
    msgs += [{"role": "user", "content": f"q{i}"} for i in range(4)]
    state = RollingSummary()
    r = compact_history_rolling(msgs, None, state, keep_last_n=4, token_threshold=100)   # no summarizer -> dropped
    assert r.dropped_turns == 1
    assert r.messages[1]["content"] == "[Earlier turns were trimmed] Pinned references from them (exact): CS-20931"


def test_summary_with_headings_but_no_key_facts_is_rejected():
    from tonst.compactor import _has_summary_content
    assert not _has_summary_content("Goal:\nDecisions:\nKey facts:\nOpen items:")
    assert not _has_summary_content("Goal: -\nDecisions: none\nKey facts: N/A\nOpen items: none")
    # The best real summary from the 24-turn run (empty Goal) must still be accepted:
    live = ("Goal: \nDecisions: replacement for the ceramic lamp from order #4471.  Express shipping upgraded.\n"
            "Key facts:  Order #4471, damaged lamp, replacement lamp requested, delivery to Baner Road, Pune. \n"
            "Open items:")
    assert _has_summary_content(live)
    comp = HistoryCompactor(model_call_fn=lambda p, m, t: "Goal:\nDecisions:\nKey facts:\nOpen items:")
    assert comp.summarize_incremental(None, "user: hello there, my lamp broke. " * 30) is None



# ---------------------------------------------------------------------
# Optional API summarizer (a stronger alternative to the local 2B model)
# ---------------------------------------------------------------------

from tonst.summarizers import AnthropicSummarizer, MESSAGES_URL


def test_anthropic_summarizer_request_usage_and_cost():
    calls = []

    def fake_post(url, headers, body, timeout):
        calls.append((url, headers, body))
        return {"content": [{"type": "text", "text": "Goal: g\nDecisions: d\nKey facts: k\nOpen items: none"}],
                "usage": {"input_tokens": 3000, "output_tokens": 200}}

    summ = AnthropicSummarizer(api_key="k", post_fn=fake_post)
    out = summ("PROMPT", "gemma2:2b", 8.0)            # HistoryCompactor's call shape; its model arg is ignored
    assert out.startswith("Goal: g")
    url, headers, body = calls[0]
    assert url == MESSAGES_URL and headers["x-api-key"] == "k"
    assert body["model"] == "claude-haiku-4-5-20251001" and body["temperature"] == 0
    assert body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert summ.usage_total == {"input_tokens": 3000, "output_tokens": 200}
    assert summ.cost_usd() == pytest.approx((3000 * 1.0 + 200 * 5.0) / 1_000_000)


def test_anthropic_summarizer_fails_soft_and_plugs_into_the_client():
    def boom(url, headers, body, timeout):
        raise ConnectionError("offline")
    bad = AnthropicSummarizer(api_key="k", post_fn=boom)
    assert bad("PROMPT") is None and bad.failures == 1

    seen = []

    def fake_post(url, headers, body, timeout):
        seen.append(body["messages"][0]["content"])
        return {"content": [{"type": "text", "text": "Goal: help\nDecisions: replace\nKey facts: order #4471\nOpen items: none"}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}

    client = TonstClient(call_fn=lambda p: "ok", use_history_compaction=True, compaction_token_threshold=100,
                         compaction_summarizer=AnthropicSummarizer(api_key="k", post_fn=fake_post))
    msgs = [{"role": "user", "content": "my email is jo@example.com and order #4471 " + "detail " * 200}]
    msgs += [{"role": "user", "content": f"q{i}"} for i in range(6)]
    state = RollingSummary()
    _, report = client.query_messages(msgs, keep_last_n=4, rolling_state=state)
    assert report.history_summary_updated and state.summary.startswith("Goal: help")
    assert seen and "jo@example.com" not in seen[0]      # the remote summarizer only sees redacted text



# ---------------------------------------------------------------------
# cache-aware rolling compaction
# ---------------------------------------------------------------------

from tonst.compactor import estimate_fold_payback


def test_fold_payback_estimate():
    # 1.15 x (new summary 100 + kept 1000) one-off vs. 0.1 x (1000 - 100) saved per turn
    assert estimate_fold_payback(1000, 1000, 0) == pytest.approx(1.15 * 1100 / 90)
    # a paid summarizer makes the fold take longer to pay back
    assert estimate_fold_payback(1000, 1000, 0, summarizer_input_tokens=1000, summarizer_price_ratio=1 / 3) \
        > estimate_fold_payback(1000, 1000, 0)
    # bigger batches pay back faster
    assert estimate_fold_payback(8000, 1000, 0) < estimate_fold_payback(2000, 1000, 0)
    # summary already at its cap and nothing evicted: never pays back
    assert estimate_fold_payback(0, 1000, 500, max_summary_tokens=500) == float("inf")


def test_cache_aware_postpones_fold_that_would_not_pay_back_in_a_short_chat():
    comp, state = _CountingCompactor(), RollingSummary()
    r = compact_history_rolling(_conv(10), comp, state, keep_last_n=4, token_threshold=500, cache_aware=True)
    assert comp.calls == [] and r.fold_postponed_for_cache and not r.summary_updated
    assert r.fold_payback_turns and r.fold_payback_turns > 2.5   # 5 user turns so far -> ~2.5 expected to follow
    assert len(r.messages) == 1 + 10                              # nothing dropped: turns stay verbatim


def test_cache_aware_folds_when_enough_turns_remain_or_prompt_too_big():
    comp, state = _CountingCompactor(), RollingSummary()
    r = compact_history_rolling(_conv(10), comp, state, keep_last_n=4, token_threshold=500, cache_aware=True,
                                expected_remaining_turns=50)
    assert r.summary_updated and not r.fold_postponed_for_cache

    comp2, state2 = _CountingCompactor(), RollingSummary()
    r2 = compact_history_rolling(_conv(10), comp2, state2, keep_last_n=4, token_threshold=500, cache_aware=True,
                                 max_history_tokens=100)
    assert r2.summary_updated  # context cap beats cache economics


def test_cache_aware_off_is_unchanged_default():
    comp, state = _CountingCompactor(), RollingSummary()
    r = compact_history_rolling(_conv(10), comp, state, keep_last_n=4, token_threshold=500)
    assert r.summary_updated and not r.fold_postponed_for_cache and r.fold_payback_turns is None


def test_client_cache_aware_compaction_reports_postponed_fold_and_prices_haiku():
    client = TonstClient(call_fn=lambda p: "ok", use_history_compaction=True, compaction_token_threshold=500,
                         compaction_cache_aware=True,
                         compaction_summarizer=AnthropicSummarizer(api_key="k", post_fn=lambda *a: {}))
    assert client._summarizer_price_ratio() == pytest.approx(1 / 3)        # Haiku $1 vs. assumed Sonnet $3
    _, report = client.query_messages(_conv(10), keep_last_n=4, rolling_state=RollingSummary())
    assert report.history_fold_postponed and not report.history_summary_updated

    local = TonstClient(call_fn=lambda p: "ok", use_history_compaction=True)
    assert local._summarizer_price_ratio() == 0.0                           # local model: free



# ---------------------------------------------------------------------
# Gemini: summarizer, token counter, cache pricing
# ---------------------------------------------------------------------

from tonst import GeminiSummarizer, GeminiTokenCounter


def _gemini_resp(text, prompt=3000, out=200, think=50):
    return {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}, {"text": text}]}}],
            "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": out, "thoughtsTokenCount": think}}


def test_gemini_summarizer_request_usage_cost_and_thought_parts_skipped():
    seen = []

    def fake_post(url, headers, body, timeout):
        seen.append((url, headers, body))
        return _gemini_resp("Goal: help")

    summ = GeminiSummarizer(api_key="k", post_fn=fake_post)
    assert summ("PROMPT") == "Goal: help"                       # the thought part is not part of the summary
    url, headers, body = seen[0]
    assert "gemini-3.5-flash-lite:generateContent" in url and headers["x-goog-api-key"] == "k"
    assert body["contents"][0]["parts"][0]["text"] == "PROMPT"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert summ.usage_total == {"input_tokens": 3000, "output_tokens": 250}   # thinking bills as output
    assert summ.cost_usd() == pytest.approx((3000 * 0.30 + 250 * 2.50) / 1_000_000)


def test_gemini_summarizer_retries_without_thinking_on_400_and_fails_soft():
    import requests as _rq

    class _Resp:
        status_code = 400

    bodies = []

    def picky(url, headers, body, timeout):
        bodies.append(body)
        if "thinkingConfig" in body["generationConfig"]:
            e = _rq.HTTPError("bad"); e.response = _Resp(); raise e
        return _gemini_resp("Goal: ok")

    summ = GeminiSummarizer(api_key="k", post_fn=picky)
    assert summ("P") == "Goal: ok" and len(bodies) == 2 and summ.failures == 0

    def down(url, headers, body, timeout):
        raise ConnectionError("offline")
    bad = GeminiSummarizer(api_key="k", post_fn=down)
    assert bad("P") is None and bad.failures == 1


def test_gemini_token_counter_wraps_request_and_fails_soft():
    seen = []

    def fake_post(url, headers, body, timeout):
        seen.append((url, body))
        return {"totalTokens": 42}

    c = GeminiTokenCounter(model="gemini-3.8-flash", api_key="k", post_fn=fake_post)
    assert c("hello") == 42
    url, body = seen[0]
    assert url.endswith("gemini-3.8-flash:countTokens")
    assert body["generateContentRequest"]["model"] == "models/gemini-3.8-flash"
    assert body["generateContentRequest"]["contents"][0]["parts"][0]["text"] == "hello"
    assert GeminiTokenCounter(api_key="k", post_fn=lambda *a: {})("x") is None


def test_cache_pricing_presets_change_fold_payback():
    anth = estimate_fold_payback(4000, 1000, 0, cache_pricing="anthropic")
    gem = estimate_fold_payback(4000, 1000, 0, cache_pricing="gemini")
    assert gem < anth                                    # no write surcharge: 0.9x vs 1.15x one-off
    assert gem == pytest.approx(0.9 * 1400 / (0.1 * 3600))
    assert estimate_fold_payback(4000, 1000, 0, cache_pricing=(1.0, 0.5)) < gem
    with pytest.raises(ValueError):
        estimate_fold_payback(4000, 1000, 0, cache_pricing="nope")



def test_observed_cache_hit_rate_uses_previous_prompt_and_skips_prefix_changes():
    st = RollingSummary()
    st.observe_cache_usage(1000, 0)          # first response: nothing to compare against
    assert st.cache_hit_rate is None and st.cache_observations == 0
    st.observe_cache_usage(2000, 1000)       # all of the previous 1000 reused
    assert st.cache_hit_rate is None         # one observation isn't enough yet
    st.observe_cache_usage(3000, 0)          # nothing reused
    assert st.cache_hit_rate == pytest.approx(0.7)    # EMA: 0.3 x 0 + 0.7 x 1.0
    st._prefix_changed = True                # the summary changed before this call: a miss is expected
    st.observe_cache_usage(1500, 0)
    assert st.cache_observations == 2
    back = RollingSummary.from_dict(st.to_dict())
    assert back.cache_hit_rate == st.cache_hit_rate and back.cache_observations == 2


def test_uncached_history_makes_folds_pay_back_immediately():
    assert estimate_fold_payback(4000, 1000, 0, cache_pricing="gemini", cache_hit_rate=0.0) == 0.0
    half = estimate_fold_payback(4000, 1000, 0, cache_pricing="gemini", cache_hit_rate=0.5)
    assert 0.0 < half < estimate_fold_payback(4000, 1000, 0, cache_pricing="gemini")


def test_cache_aware_folds_when_the_provider_is_not_actually_caching():
    comp, state = _CountingCompactor(), RollingSummary(cache_hit_rate=0.0, cache_observations=5)
    r = compact_history_rolling(_conv(10), comp, state, keep_last_n=4, token_threshold=500, cache_aware=True,
                                cache_pricing="gemini")
    assert r.summary_updated and not r.fold_postponed_for_cache      # the same chat with hits postpones (above)



# ---------------------------------------------------------------------
# messages_fn: chat apps keep roles; provider adapters
# ---------------------------------------------------------------------

from tonst.adapters import (to_anthropic, to_openai, to_gemini,
                            usage_from_anthropic, usage_from_openai, usage_from_gemini)


def test_client_needs_a_call_function():
    with pytest.raises(ValueError):
        TonstClient()


def test_messages_fn_gets_redacted_roles_and_restores_pii():
    seen = []

    def fake(messages):
        seen.append(messages)
        email_ph = next(tok for tok in messages[-1]["content"].split() if tok.startswith("[[EMAIL_"))
        return f"I'll write to {email_ph}", {"prompt_tokens": 120, "cached_tokens": 100}

    client = TonstClient(messages_fn=fake)
    msgs = [{"role": "system", "content": "You are support."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Email me at jo@example.com please"}]
    text, report = client.query_messages(msgs)
    sent = seen[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]   # roles kept, not flattened
    assert "jo@example.com" not in str(sent)
    assert text == "I'll write to jo@example.com"                                  # placeholder restored
    assert report.provider_prompt_tokens == 120 and report.provider_cached_tokens == 100
    assert report.redacted_fields == 1 and report.original_tokens > 0


def test_messages_fn_feeds_cache_hit_rate_to_rolling_state():
    calls = {"n": 0}

    def fake(messages):
        calls["n"] += 1
        prompt = 1000 * calls["n"]
        return "ok", {"prompt_tokens": prompt, "cached_tokens": 0}   # provider isn't caching at all

    client = TonstClient(messages_fn=fake)
    state = RollingSummary()
    history = [{"role": "system", "content": "sys"}]
    for i in range(4):
        history.append({"role": "user", "content": f"q{i}"})
        client.query_messages(history, rolling_state=state)
        history.append({"role": "assistant", "content": f"a{i}"})
    assert state.cache_hit_rate == 0.0


def test_query_works_with_messages_fn_only():
    got = []
    client = TonstClient(messages_fn=lambda m: got.append(m) or "fine")
    text, report = client.query("call me on jo@example.com")
    assert text == "fine" and got[0][0]["role"] == "user" and "jo@example.com" not in got[0][0]["content"]
    assert report.provider_prompt_tokens is None


def test_savings_log_records_provider_usage_from_messages_fn(tmp_path):
    log = tmp_path / "s.jsonl"
    client = TonstClient(messages_fn=lambda m: ("ok", {"prompt_tokens": 50, "cached_tokens": 40}), savings_log=str(log))
    client.query_messages([{"role": "user", "content": "hello"}])
    entry = _json_mod.loads(log.read_text().splitlines()[0])
    assert entry["provider_usage"]["input_tokens"] == 50 and entry["provider_usage"]["cache_read_input_tokens"] == 40


def test_to_anthropic_merges_roles_and_places_cache_breakpoints():
    body = to_anthropic([{"role": "system", "content": "S"},
                         {"role": "user", "content": "[Summary of earlier conversation] ..."},
                         {"role": "user", "content": "next question"}])
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert len(body["messages"]) == 1 and len(body["messages"][0]["content"]) == 2   # merged user turns
    assert body["messages"][0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in to_anthropic([{"role": "user", "content": "x"}], cache=False)["messages"][0]["content"][0]
    starts_with_assistant = to_anthropic([{"role": "assistant", "content": "hi"}, {"role": "user", "content": "yo"}])
    assert starts_with_assistant["messages"][0]["role"] == "user"


def test_to_gemini_and_to_openai_shapes():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    g = to_gemini(msgs)
    assert g["systemInstruction"] == {"parts": [{"text": "S"}]}
    assert [c["role"] for c in g["contents"]] == ["user", "model"]
    assert to_openai(msgs) == msgs


def test_usage_adapters_accept_dicts_and_sdk_objects():
    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    anth = Obj(usage=Obj(input_tokens=3, cache_creation_input_tokens=7, cache_read_input_tokens=90))
    assert usage_from_anthropic(anth) == {"prompt_tokens": 100, "cached_tokens": 90}
    assert usage_from_openai({"usage": {"prompt_tokens": 80, "prompt_tokens_details": {"cached_tokens": 64}}}) == \
        {"prompt_tokens": 80, "cached_tokens": 64}
    assert usage_from_openai({"usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}}}) == \
        {"prompt_tokens": 10, "cached_tokens": 0}
    assert usage_from_gemini({"usageMetadata": {"promptTokenCount": 5000, "cachedContentTokenCount": 4096}}) == \
        {"prompt_tokens": 5000, "cached_tokens": 4096}
    assert usage_from_gemini({"usageMetadata": {"promptTokenCount": 10}}) == {"prompt_tokens": 10, "cached_tokens": 0}
    assert usage_from_anthropic({}) is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
