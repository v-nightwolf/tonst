"""
cache_savings_demo_generic.py
--------------------------------
A TEMPLATE for measuring real prompt-caching savings against ANY
provider tonst doesn't have a dedicated script for -- Mistral, Groq,
Together AI, Fireworks, DeepSeek, xAI, Cohere, a self-hosted
vLLM/Ollama/TGI server, or whatever you're integrating tonst with.
This is the direct answer to "how do I test this against any model":
there is no single live endpoint to call generically, so instead of a
script that can only test one more named provider, this one is
runnable AS-IS out of the box (see DRY_RUN below) and documents exactly
what to change to point it at a REAL provider and a REAL key.

Unlike cache_savings_demo_anthropic.py / _openai.py / _gemini.py, this
script does NOT call a real API by default -- there is no "the generic
provider" to call. Run it unmodified and it exercises the full
tonst.providers.generic pipeline (eligibility check, request building,
usage parsing, cost-savings estimate) against a MOCKED response, so you
can see the pattern work end to end with zero setup. Then follow the
three numbered TODOs below to swap in your real provider.

Setup to test a REAL provider (leave DRY_RUN = True to skip all of this
and just see the pattern demonstrated):
    1. Fill in GenericCacheConfig with your provider's real numbers --
       check ITS docs for minimum_tokens, cache_read_multiplier,
       cache_write_multiplier (if any), and where cached-token counts
       live in ITS response JSON (usage_read_path etc.).
    2. Replace call_provider() below with a real HTTP call to your
       provider's endpoint and auth scheme.
    3. Set DRY_RUN = False, put your API key in a .env file or export
       it, and run: python3 cache_savings_demo_generic.py
"""

import os
import subprocess
import sys

DRY_RUN = True  # flip to False once you've filled in the three TODOs below


def _try_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _ensure_dependencies():
    if _try_import("requests"):
        return
    print("Installing missing dependency: requests>=2.25")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests>=2.25"])


def _load_dotenv_if_present():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_ensure_dependencies()
_load_dotenv_if_present()

import requests
from tonst import TonstClient, PromptParts
from tonst.providers import generic as generic_provider

MY_PROVIDER_API_KEY = os.environ.get("MY_PROVIDER_API_KEY", "")

# ---------------------------------------------------------------------
# TODO 1: describe your provider's caching contract here, from its docs.
# The values below are PLACEHOLDERS for the dry run -- replace every one
# with your real provider's real numbers before setting DRY_RUN = False.
# ---------------------------------------------------------------------
MY_PROVIDER = generic_provider.GenericCacheConfig(
    label="My Provider",
    minimum_tokens=1024,             # their documented minimum, or 0 if none/unknown
    cache_read_multiplier=0.5,       # e.g. 0.5 = 50% off cached tokens
    cache_write_multiplier=1.0,      # 1.0 if there's no write premium
    usage_read_path="usage.cached_tokens",      # wherever THEIR usage JSON puts it
    usage_write_path="",                        # "" if there's no write-cost concept
    usage_input_path="usage.prompt_tokens",
    usage_output_path="usage.completion_tokens",
)

# A moderately long, realistic reference doc -- same content used across
# tonst's other cache_savings_demo_*.py scripts. Real minimums vary by
# provider (check yours), so this may need to grow for a stricter one.
REFERENCE_DOC = """
Acme Cloud Hosting -- Tier 2 Support Reference Guide (Internal)

Section 1: Account & Billing
- Refunds are available within 30 days of the original charge for any
  plan tier. Refunds outside this window require a manager override and
  should be escalated to billing@acme-support.example.com with the
  account ID and reason.
- Failed payments trigger a 3-day grace period before service suspension.
  During the grace period, customers retain full access and receive one
  automated reminder email at day 1 and day 2.
- Plan downgrades take effect at the start of the next billing cycle;
  plan upgrades take effect immediately with a prorated charge.
- Enterprise accounts (50+ seats) are billed annually via invoice, not
  credit card, and go through the dedicated enterprise billing queue
  rather than self-service refunds.

Section 2: Technical Support Triage
- P1 (full outage): page on-call immediately, acknowledge within 15
  minutes, and post a status update every 30 minutes until resolved.
- P2 (degraded service, workaround exists): acknowledge within 2 hours,
  resolve or provide a workaround within 8 business hours.
- P3 (cosmetic or low-impact issue): acknowledge within 1 business day,
  no fixed resolution SLA but should not remain untriaged past a week.
- Always confirm the customer's account ID and the affected region
  before escalating -- most "outage" reports turn out to be a single
  misconfigured DNS record or an expired API key, not a platform issue.

Section 3: Common Customer Questions
- "Why was I charged twice?" -- almost always a plan change mid-cycle
  generating a prorated charge alongside the regular renewal. Check the
  invoice line items before assuming a billing error.
- "My site is down" -- check the status page first, then ask for the
  exact URL and error message; "down" covers everything from a full
  outage to a single broken image link.
- "Can I get a discount?" -- Tier 2 support cannot offer discretionary
  discounts. Route to the account management team for anything beyond
  the standard published pricing.
- "How do I delete my account?" -- confirm no active subscriptions
  first, then follow the account-closure checklist in the internal wiki;
  never delete an account directly from a support ticket.

Section 4: Escalation Contacts
- Billing disputes: billing@acme-support.example.com
- Security incidents: security@acme-support.example.com (P1 always,
  regardless of apparent severity)
- Enterprise account management: enterprise@acme-support.example.com
""".strip()


def call_provider(body: dict) -> dict:
    """
    TODO 2: replace this with a real call to your provider's endpoint.
    Only used when DRY_RUN = False. The example below assumes an
    OpenAI-compatible chat completions endpoint (true for many providers
    -- Groq, Together, Fireworks, DeepSeek, self-hosted vLLM/Ollama/TGI
    servers -- but check yours; some, like Cohere or Bedrock, use a
    genuinely different shape and need a different call here).
    """
    resp = requests.post(
        "https://api.my-provider.example.com/v1/chat/completions",  # TODO: your real endpoint
        headers={
            "Authorization": f"Bearer {MY_PROVIDER_API_KEY}",  # TODO: your real auth scheme
            "content-type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()


def call_provider_dry_run(body: dict) -> dict:
    """
    Fakes a response shaped like the config above expects, purely so
    this script demonstrates the full pipeline (eligibility -> request
    -> parse -> cost estimate) with zero setup. Delete this once you've
    wired up a real provider -- it exists only for the dry run.
    """
    prompt_len_estimate = len(str(body))
    return {
        "usage": {
            "prompt_tokens": 1200,
            "cached_tokens": 950,  # pretend most of the stable prefix was reused
            "completion_tokens": 25,
        }
    }


def main():
    if not DRY_RUN and not MY_PROVIDER_API_KEY:
        print(
            "DRY_RUN is False but no MY_PROVIDER_API_KEY is set. Either set\n"
            "DRY_RUN = True to see the pattern without a real key, or add:\n"
            "  export MY_PROVIDER_API_KEY=...\n"
            "  echo 'MY_PROVIDER_API_KEY=...' > .env   # gitignored\n"
        )
        sys.exit(1)

    if DRY_RUN:
        print(
            "=== DRY RUN -- no real API is being called ===\n"
            "This demonstrates the generic pipeline end to end using a fake\n"
            "response. Fill in the three TODOs in this file (GenericCacheConfig,\n"
            "call_provider(), and set DRY_RUN = False) to test your real "
            "provider.\n"
        )

    client = TonstClient(call_fn=lambda p: p)  # call_fn unused on this path

    parts = PromptParts(
        system="You are a Tier 2 support agent for Acme Cloud Hosting. "
        "Answer using only the reference guide provided.",
        stable_blocks=[REFERENCE_DOC],
        variable="A customer named Priya (priya.sharma@example.com) says "
        "she was charged twice this month. In one sentence, what should "
        "the agent check first?",
    )

    eligibility = generic_provider.check_cache_eligibility(parts, MY_PROVIDER)
    print(f"--- Cache eligibility check ({MY_PROVIDER.label}) ---\n{eligibility.message}\n")
    if not eligibility.eligible and not DRY_RUN:
        print("Stopping -- adjust REFERENCE_DOC to be longer before spending API calls.")
        sys.exit(1)

    redacted = client.redact_and_trim_parts(parts)
    print(f"PII fields redacted from structured parts: {len(redacted.mapping)}\n")

    # See generic.build_generic_chat_request()'s docstring: a reasonable
    # default for OpenAI-compatible providers, not a guarantee for every
    # provider's real request shape.
    body = generic_provider.build_generic_chat_request(redacted.parts, model="your-model-name")

    caller = call_provider_dry_run if DRY_RUN else call_provider

    print("--- Call 1 ---")
    response_1 = caller(body)
    usage_1 = generic_provider.parse_usage(response_1, MY_PROVIDER)
    print(f"input_tokens={usage_1.input_tokens}  cache_read_input_tokens={usage_1.cache_read_input_tokens}  cache_hit={usage_1.cache_hit}")

    print("\n--- Call 2, seconds later, IDENTICAL stable prefix ---")
    response_2 = caller(body)
    usage_2 = generic_provider.parse_usage(response_2, MY_PROVIDER)
    print(
        f"input_tokens={usage_2.input_tokens}  "
        f"cache_read_input_tokens={usage_2.cache_read_input_tokens}  "
        f"cache_hit={usage_2.cache_hit}  "
        f"({usage_2.percent_of_input_from_cache}% of this call's input came from cache)"
    )
    savings_2 = generic_provider.estimated_cost_savings_percent(usage_2, MY_PROVIDER)
    print(f"Estimated cost vs. no caching at all: {savings_2:+.1f}%  (using {MY_PROVIDER.label}'s configured multipliers)")

    print("\n--- Verdict ---")
    if DRY_RUN:
        print(
            "This was a dry run against a fake response -- the numbers above "
            "prove the PIPELINE works, not that your real provider's caching "
            "does. Fill in the three TODOs and set DRY_RUN = False to measure "
            "the real thing."
        )
    elif usage_2.cache_hit:
        print("Confirmed: the second call got a real cache hit from the live API.")
    else:
        print(
            "No cache hit on call 2. Double-check: does your provider actually "
            "support prefix caching at all? Is the stable prefix long enough "
            "(see the eligibility check above)? Is usage_read_path pointing at "
            "the right field in a REAL response from this provider (print "
            "response_2 and look) -- a wrong path silently reads 0, which looks "
            "identical to a genuine cache miss."
        )


if __name__ == "__main__":
    main()
