"""
demo.py
-------
Runs the SDK against a MOCK paid-API function (no real network call, no
API key needed) so you can see the pipeline and the reported savings,
plus a walkthrough of prompt-caching structuring.

Swap `mock_paid_api_call` for a real call to Claude/OpenAI/etc. and this
becomes a working integration -- nothing else about the SDK changes.

Run: python3 demo.py
(No setup needed -- this bootstraps its own missing dependency below.)
"""

import json
import subprocess
import sys


def _ensure_dependencies():
    """
    Self-healing dependency check: if requests isn't installed, install
    it automatically rather than making the user run a separate setup
    step first. This only installs the one package this project
    declares in requirements.txt -- nothing else, nothing silent beyond
    a printed notice.
    """
    try:
        __import__("requests")
        return
    except ImportError:
        pass

    print("Installing missing dependency: requests>=2.25")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests>=2.25"])
    except subprocess.CalledProcessError:
        print(
            "\nAutomatic install failed (no internet connection, or pip is "
            "blocked on this machine).\n"
            "Please install manually and re-run:\n"
            f"    {sys.executable} -m pip install requests>=2.25"
        )
        sys.exit(1)
    print()


_ensure_dependencies()

from tonst import TonstClient, PromptParts, build_anthropic_cache_request, HistoryCompactor
from tonst.providers import openai as openai_provider
from tonst.providers import gemini as gemini_provider
from tonst.providers import generic as generic_provider
from tonst.providers import presets as provider_presets


def mock_paid_api_call(prompt: str) -> str:
    """
    Stands in for `anthropic.messages.create(...)` or similar.
    Just proves the SDK only ever hands this function the trimmed/redacted text.
    """
    print(f"    [PAID API RECEIVED] ({len(prompt)} chars): {prompt[:120]}...")
    return (
        "Thanks for reaching out. We've noted your account and will follow up "
        "with next steps shortly regarding your request."
    )


CHAT_HISTORY_BLOB = """You are a customer support assistant for Acme Cloud Hosting.
You are a customer support assistant for Acme Cloud Hosting.
Be polite and concise.
Be polite and concise.

Customer: Hi, I've been having trouble with my account.
Customer: Hi, I've been having trouble with my account.
Agent: I'm sorry to hear that, could you share more detail?
Customer: My email is jane.doe@example.com and my card ending isn't working,
full number is 4111 1111 1111 1111. Can you check my billing status? Also my
phone is +1 415-555-0132 in case you need to call me back.
"""


def demo_basic_pipeline():
    print("=== Basic pipeline: redact + trim before the paid call ===")
    client = TonstClient(call_fn=mock_paid_api_call, use_local_compression=False)

    response, report = client.query(CHAT_HISTORY_BLOB)
    print(f"Response returned to app: {response}\n")
    print(f"Original tokens (est.):   {report.original_tokens}")
    print(f"Tokens actually sent:     {report.sent_tokens}")
    print(f"Tokens saved:             {report.tokens_saved} ({report.percent_saved}%)")
    print(f"PII fields redacted:      {report.redacted_fields}")


def demo_cache_structuring():
    print("\n=== Prompt-caching structuring ===")
    print(
        "This does NOT skip the paid API call -- every call below still\n"
        "reaches the model. What it does is shape the request so a real\n"
        "provider's own caching can discount the repeated part, by (a)\n"
        "keeping stable content first / variable content last, and (b)\n"
        "making sure redacted PII inside the stable content is\n"
        "byte-identical every time, not a fresh random placeholder.\n"
    )

    client = TonstClient(call_fn=mock_paid_api_call)

    parts = PromptParts(
        system="You are a customer support assistant for Acme Cloud Hosting.",
        stable_blocks=[
            "Company policy: refunds within 30 days, escalate billing "
            "disputes to billing@acme-support.example.com.",
        ],
        variable="A customer named Jane (jane.doe@example.com) is asking "
        "about a refund. What should I tell her?",
    )

    print("--- Redacting the structured parts (system/stable/variable kept separate) ---")
    redacted = client.redact_and_trim_parts(parts)
    print(f"Stable block after redaction: {redacted.parts.stable_blocks[0]}")
    print(f"Variable part after redaction: {redacted.parts.variable}")

    print("\n--- Same input, redacted again: placeholders must match exactly ---")
    redacted_again = client.redact_and_trim_parts(parts)
    same = redacted.parts.stable_blocks[0] == redacted_again.parts.stable_blocks[0]
    print(f"Stable block identical across calls: {same}  <-- must be True for caching to work")

    print("\n--- Building the real Anthropic request body (cache_control breakpoint) ---")
    body = build_anthropic_cache_request(
        redacted.parts, model="claude-sonnet-4-6", max_tokens=300
    )
    print(json.dumps(body, indent=2))
    print(
        "\nNote the cache_control marker on the system block and on the "
        "LAST stable content block -- everything up to and including it "
        "is what Anthropic will treat as a cacheable prefix on repeat calls."
    )


def demo_history_compaction():
    print("\n=== History compaction (query_messages) ===")
    print(
        "Long chat histories get capped to the system message(s) + the\n"
        "last few turns. With use_history_compaction=True, whatever falls\n"
        "outside that window is summarized by a LOCAL model instead of\n"
        "just being dropped -- never a paid token spent on the summary.\n"
        "Here we fake the local model's response so this demo needs no\n"
        "Ollama install; swap in a real one to see genuine summarization.\n"
    )

    older_turn = (
        "Customer: My order #48213 hasn't arrived, my email is "
        "jane.doe@example.com. Agent: Checking now, one moment. "
    )
    conversation = (
        [{"role": "system", "content": "You are a customer support assistant."}]
        + [{"role": "user", "content": older_turn} for _ in range(6)]
        + [{"role": "user", "content": "Any update on my order?"}]
    )

    client = TonstClient(
        call_fn=mock_paid_api_call,
        use_history_compaction=True,
        compaction_token_threshold=50,
    )
    # Fake the local model so the demo runs with no Ollama dependency.
    client.history_compactor._call_model = lambda prompt, model, timeout: (
        "Customer asked about order #48213 status; agent was checking."
    )

    response, report = client.query_messages(conversation, keep_last_n=2)
    print(f"History compacted:        {report.history_compacted}")
    print(f"Older turns dropped:      {report.history_turns_dropped}")
    print(f"Compaction time (ms):     {report.compaction_ms:.2f}")
    print(f"Tokens saved:             {report.tokens_saved} ({report.percent_saved}%)")
    print(f"PII fields redacted:      {report.redacted_fields}  <-- redacted BEFORE the local model saw any of it")


def demo_multi_provider_caching():
    print("\n=== Multi-provider caching: OpenAI and Gemini ===")
    print(
        "Same PromptParts, three different providers, three genuinely\n"
        "different mechanics -- no single request shape or pricing\n"
        "formula covers all of them. See tonst/providers/ and the\n"
        "README's \"Multi-provider support\" section for the full\n"
        "comparison. No real API calls here (no OpenAI/Gemini key in\n"
        "this demo) -- request shapes are real, usage numbers are\n"
        "illustrative fakes shaped like each provider's actual response.\n"
    )

    parts = PromptParts(
        system="You are a customer support assistant for Acme Cloud Hosting.",
        stable_blocks=["Company policy: refunds within 30 days..."],
        variable="A customer is asking about a refund. What should I tell her?",
    )

    print("--- OpenAI: automatic caching needs only correct ordering ---")
    openai_body = openai_provider.build_openai_cache_request(parts, model="gpt-4o")
    print(f"Request 'messages' content is a single ordered string (no marker needed):")
    print(f"  {openai_body['messages'][1]['content'][:90]}...")
    fake_openai_response = {
        "usage": {
            "prompt_tokens": 1100, "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 1024},
        }
    }
    openai_usage = openai_provider.parse_openai_usage(fake_openai_response)
    openai_savings = openai_provider.estimated_cost_savings_percent(openai_usage, "gpt-4o")
    print(f"Parsed cache_read_input_tokens={openai_usage.cache_read_input_tokens}, "
          f"estimated cost savings on gpt-4o (50% discount model): {openai_savings:+.1f}%")

    print("\n--- Gemini: implicit (automatic) vs. explicit CachedContent ---")
    gemini_body = gemini_provider.build_gemini_content_request(parts, model="gemini-2.5-flash")
    print(f"Implicit path -- contents ordered stable-first, no marker: "
          f"{len(gemini_body['contents'][0]['parts'])} parts, no separate resource created.")
    fake_gemini_response = {
        "usageMetadata": {
            "promptTokenCount": 1100, "candidatesTokenCount": 35,
            "cachedContentTokenCount": 1024,
        }
    }
    gemini_usage = gemini_provider.parse_gemini_usage(fake_gemini_response)
    gemini_savings = gemini_provider.estimated_implicit_cache_cost_savings_percent(gemini_usage)
    print(f"Parsed cache_read_input_tokens={gemini_usage.cache_read_input_tokens}, "
          f"estimated cost savings (confirmed uniform 10% read rate): {gemini_savings:+.1f}%")

    explicit_savings = gemini_provider.estimated_explicit_cache_cost_savings_percent(
        model="gemini-2.5-flash", cached_tokens=100_000, num_requests=20,
        hours_cached=2, input_price_per_million=0.30,
    )
    print(f"Explicit CachedContent path -- a fundamentally different cost shape: "
          f"storage rent is charged per hour regardless of reads, so whether it\n"
          f"pays off depends on reuse. Reusing 100k cached tokens across 20 requests "
          f"in a 2-hour window: {explicit_savings:+.1f}% vs. no caching.")


def demo_generic_provider_caching():
    print("\n=== Generic caching support: any provider, not just the 3 named ones ===")
    print(
        "tonst can't ship a hand-written module for every provider that\n"
        "exists or will ever exist -- so beyond Anthropic/OpenAI/Gemini,\n"
        "GenericCacheConfig takes a provider's minimum length, pricing\n"
        "multipliers, and usage-JSON field paths as plain data instead\n"
        "of hardcoded code. Two real, verified examples below, plus one\n"
        "fully invented provider to show the escape hatch working for\n"
        "something tonst has literally never heard of.\n"
    )

    parts = PromptParts(
        system="You are a customer support assistant for Acme Cloud Hosting.",
        stable_blocks=["Company policy: refunds within 30 days..."],
        variable="A customer is asking about a refund. What should I tell her?",
    )

    print("--- Real example 1: AWS Bedrock's Converse API (Claude models) ---")
    print(
        "Same underlying model as tonst's Anthropic support, but Bedrock's\n"
        "Converse API reports cache usage under DIFFERENT, camelCase field\n"
        "names -- pointing tonst's Anthropic parser at this response would\n"
        "silently read zero. This preset (providers/presets.py) fixes that."
    )
    fake_bedrock_response = {
        "usage": {"inputTokens": 1586, "outputTokens": 40, "cacheReadInputTokens": 1547, "cacheWriteInputTokens": 0}
    }
    bedrock_usage = generic_provider.parse_usage(fake_bedrock_response, provider_presets.BEDROCK_CONVERSE_CLAUDE)
    bedrock_savings = generic_provider.estimated_cost_savings_percent(bedrock_usage, provider_presets.BEDROCK_CONVERSE_CLAUDE)
    print(f"Parsed cache_read_input_tokens={bedrock_usage.cache_read_input_tokens}, "
          f"estimated cost savings: {bedrock_savings:+.1f}%")

    print("\n--- Real example 2: Azure OpenAI, Provisioned Throughput (PTU-M) tier ---")
    print(
        "Standard Azure OpenAI deployments reuse tonst's OpenAI parser\n"
        "unchanged (identical field path) -- but the PTU-M pricing tier\n"
        "gets an Azure-specific discount OpenAI's own table has no idea\n"
        "about, so it gets its own preset instead."
    )
    fake_azure_response = {
        "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 800}}
    }
    azure_usage = generic_provider.parse_usage(fake_azure_response, provider_presets.AZURE_OPENAI_PTU_M)
    azure_savings = generic_provider.estimated_cost_savings_percent(azure_usage, provider_presets.AZURE_OPENAI_PTU_M)
    print(f"Parsed cache_read_input_tokens={azure_usage.cache_read_input_tokens}, "
          f"estimated cost savings: {azure_savings:+.1f}%")

    print("\n--- Made-up example: a provider tonst has never heard of ---")
    made_up_provider = generic_provider.GenericCacheConfig(
        label="Totally Fictional LLM Co.",
        minimum_tokens=800,
        cache_read_multiplier=0.3,   # invented -- whatever THEIR docs would say
        usage_read_path="cache_stats.tokens_from_cache",
        usage_input_path="token_usage.total_prompt",
        usage_output_path="token_usage.total_completion",
    )
    eligibility = generic_provider.check_cache_eligibility(parts, made_up_provider)
    print(eligibility.message)
    fake_response = {
        "token_usage": {"total_prompt": 1200, "total_completion": 30},
        "cache_stats": {"tokens_from_cache": 950},
    }
    made_up_usage = generic_provider.parse_usage(fake_response, made_up_provider)
    made_up_savings = generic_provider.estimated_cost_savings_percent(made_up_usage, made_up_provider)
    print(f"Parsed cache_read_input_tokens={made_up_usage.cache_read_input_tokens} from a made-up "
          f"response shape, estimated cost savings: {made_up_savings:+.1f}%")
    print(
        "\nNo tonst code changed to support this provider -- only a "
        f"{type(made_up_provider).__name__} describing it did."
    )


def main():
    demo_basic_pipeline()
    demo_cache_structuring()
    demo_history_compaction()
    demo_multi_provider_caching()
    demo_generic_provider_caching()


if __name__ == "__main__":
    main()
