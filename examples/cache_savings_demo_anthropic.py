"""
examples/cache_savings_demo_anthropic.py
---------------------------------
Measures REAL prompt-caching savings against the actual api.anthropic.com
endpoint -- not a mock, not an estimate. This exists because tonst's own
estimate_tokens() (a chars/4 heuristic) has no idea whether a request
actually got cached; the only ground truth is the `usage` field of a real
API response (see cache_structuring.parse_anthropic_usage()).

This is ONE of several provider-specific "measure it for real" scripts --
tonst is a multi-provider tool (see tonst/providers/ and the README's
"Multi-provider support" / "Any other provider" sections), and there is
no single generic live-test script because there is no single generic
API to call: every provider needs its own base URL, auth header, and
request/response shape. This file is the Anthropic one specifically.
See the sibling scripts for the others, and for how to test a provider
tonst has no dedicated module for:
    - examples/cache_savings_demo_openai.py    -- OpenAI, api.openai.com
    - examples/cache_savings_demo_gemini.py    -- Google Gemini, generativelanguage.googleapis.com
    - examples/cache_savings_demo_generic.py   -- template for ANY OTHER provider,
      using tonst.providers.generic.GenericCacheConfig; runs out of the
      box in a mocked dry-run mode, with clear instructions for pointing
      it at a real provider and a real key.
All four follow the identical pattern: build an eligible request, call
the real API twice with an identical stable prefix, parse the real
`usage` field from both responses, and print the measured cost
difference -- only the provider-specific plumbing (URL, auth, request
shape, usage field path) changes between them.

This script also exists to sidestep a real gotcha: Anthropic requires a
MINIMUM stable-prefix length before it will cache anything at all
(1,024 tokens for claude-sonnet-4-6, the model used here) -- below that,
requests are processed normally with no error and no cache_creation/
cache_read tokens. tonst's own examples/demo.py uses a tiny stable block on
purpose (to keep the demo short) which means it would NOT actually get
cached if pointed at the real API. This script instead uses a
deliberately large, realistic reference document so the cache has a
real chance to activate, and prints check_cache_eligibility()'s estimate
before spending any API calls.

Setup (this script needs an ANTHROPIC key specifically -- see the
sibling scripts above for other providers):
    1. Get an API key from https://console.anthropic.com
    2. Put it in a .env file in the repository root (gitignored):
           ANTHROPIC_API_KEY=sk-ant-...
       or export it directly in your shell.
    3. python3 examples/cache_savings_demo_anthropic.py

What you should see: call 1 shows cache_creation_input_tokens > 0 (a new
cache entry was written) and cache_read_input_tokens == 0. Call 2, sent
seconds later with the IDENTICAL stable prefix, should show
cache_read_input_tokens > 0 -- that's the actual, measured discount.
"""
# Run from anywhere: make the repo root importable without `pip install -e .`.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import subprocess
import sys


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
    """
    Tiny, dependency-free .env loader -- just so ANTHROPIC_API_KEY can
    live in a gitignored file instead of being typed into a shell (and
    therefore into shell history) every time.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
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
from tonst import (
    TonstClient,
    PromptParts,
    build_anthropic_cache_request,
    check_cache_eligibility,
    parse_anthropic_usage,
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6"  # 1,024-token cache minimum -- see CACHE_MINIMUM_TOKENS

# Deliberately long and repetitive-but-realistic: a support-org reference
# doc, well past the 1,024-token minimum for MODEL above (~1,440 tokens
# by tonst's chars/4 estimate -- see examples/cache_savings_demo_gemini.py's
# REFERENCE_DOC comment for why that estimate runs a bit hot vs. the
# real tokenizer; either way this clears the minimum with real margin).
#
# NOTE (Sept 2026): this is a fresh scenario -- a serverless-functions
# platform's support reference guide -- swapped in for the original
# "Acme Cloud Hosting" doc so a re-test isn't just replaying the exact
# same cached text as before. Same structure and length class as the
# original, different domain and wording throughout.
REFERENCE_DOC = """
NimbusFn -- Serverless Functions Platform, Developer Support Reference (Internal)

Section 1: Account & Billing
- Refunds are available within 21 days of the original charge for any
  plan tier. Refunds outside this window require a manager override and
  should be escalated to billing@nimbusfn-support.example.com with the
  workspace ID and reason.
- Failed payments trigger a 5-day grace period before function
  invocations are paused. During the grace period, deployments keep
  running and the workspace owner receives one automated reminder email
  at day 2 and day 4.
- Plan downgrades take effect at the start of the next billing cycle;
  plan upgrades take effect immediately with a prorated charge.
- Team accounts (25+ members) are billed annually via invoice, not
  credit card, and go through the dedicated team billing queue rather
  than self-service refunds.

Section 2: Technical Support Triage
- P1 (functions not invoking platform-wide): page on-call immediately,
  acknowledge within 10 minutes, and post a status update every 20
  minutes until resolved.
- P2 (elevated cold-start latency or partial region outage): acknowledge
  within 1 hour, resolve or provide a workaround within 6 business
  hours.
- P3 (dashboard cosmetic issue, non-blocking): acknowledge within 1
  business day, no fixed resolution SLA but should not remain untriaged
  past a week.
- Always confirm the workspace ID, function name, and affected region
  before escalating -- most "my function isn't running" reports turn
  out to be an unhandled exception in the customer's own code, not a
  platform issue.

Section 3: Common Customer Questions
- "Why did my invocation count double?" -- almost always a retry policy
  configured on the trigger (queue or webhook) re-invoking after a
  timeout. Check the function's retry settings before assuming a
  billing error.
- "My function is timing out" -- check the configured timeout limit
  first, then ask for the exact function name and a request ID; a
  timeout can mean anything from a genuine slow dependency to a
  misconfigured 3-second limit on a function that needs 10.
- "Can I get a discount?" -- support cannot offer discretionary
  discounts. Route to the account management team for anything beyond
  the standard published pricing.
- "How do I delete my workspace?" -- confirm no active scheduled
  functions or paid add-ons first, then follow the workspace-closure
  checklist in the internal wiki; never delete a workspace directly
  from a support ticket.

Section 4: Escalation Contacts
- Billing disputes: billing@nimbusfn-support.example.com
- Security incidents: security@nimbusfn-support.example.com (P1 always,
  regardless of apparent severity)
- Team account management: teams@nimbusfn-support.example.com

Section 5: Plan Tiers & Feature Matrix
- Hobby ($0/mo): 1 workspace, community support only, 500K invocations
  included, shared compute, 128MB max memory per function, no custom
  domains for HTTP triggers.
- Starter ($19/mo): 5 workspaces, email support (next business day), 5M
  invocations included, dedicated compute pool, 512MB max memory, 3
  custom domains, deployment history retained for 7 days.
- Team ($79/mo): unlimited workspaces, priority email + chat support
  (4-hour response), 50M invocations included, 1GB max memory,
  unlimited custom domains, deployment history retained for 30 days,
  staging environments, SSO via SAML.
- Enterprise (custom pricing, annual invoice): everything in Team plus a
  named solutions engineer, custom SLA negotiation, dedicated
  infrastructure, deployment history retained for 1 year, audit
  logging, and a private Slack channel with the support team.
- Feature availability questions from customers should always be
  answered against the current published pricing page, not from memory
  -- the matrix above is reviewed quarterly and can lag a recent change.

Section 6: Outage Communication Protocol
- Every P1 incident gets a public status-page entry within 15 minutes
  of confirmation, regardless of how few workspaces appear affected.
- Status updates during an active P1 go out every 20 minutes on a fixed
  cadence, even if the update is just "still investigating, next update
  in 20 minutes" -- silence during an outage erodes trust faster than a
  slow-but-communicative resolution.
- Once resolved, a P1 always gets a post-incident summary within 24
  hours: what happened, customer impact, and what's changing to prevent
  recurrence. This is drafted by the on-call engineer and reviewed by
  the support lead before publishing.
- P2 and P3 issues are not posted to the public status page unless they
  affect a large enough share of workspaces that multiple independent
  tickets are expected; use judgment and escalate to the support lead
  if unsure whether an issue crosses that line.

Section 7: Data Retention & Workspace Closure
- Active workspace logs are retained for 30 days on Hobby/Starter and 90
  days on Team/Enterprise while the subscription is active, subject to
  the plan's own deployment-history window listed in Section 5.
- On voluntary workspace closure, function code and configuration are
  retained for 14 days in a recoverable state before permanent
  deletion, to cover accidental cancellations. Customers are told this
  explicitly during the closure flow.
- On involuntary closure (repeated failed payment beyond the grace
  period in Section 1), the same 14-day recoverable window applies
  before permanent deletion, and a reactivation link is included in
  every suspension notice email.
- Enterprise accounts under contract have their own data-retention
  terms specified in the master service agreement; check the contract
  before applying the default 14-day window to an enterprise closure.
""".strip()


def call_claude(body: dict) -> dict:
    """Direct call to the real Anthropic API. Returns the full decoded
    JSON response so usage.cache_* fields can be inspected."""
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()


def main():
    if not ANTHROPIC_API_KEY:
        print(
            "No ANTHROPIC_API_KEY set (checked environment and ./.env).\n"
            "This script needs a real key to measure real caching -- "
            "there's no mock mode, since the whole point is to observe\n"
            "actual usage.cache_creation_input_tokens / "
            "cache_read_input_tokens from api.anthropic.com.\n\n"
            "Add one of:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-...\n"
            "  echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   # gitignored\n"
        )
        sys.exit(1)

    client = TonstClient(call_fn=lambda p: p)  # call_fn unused on this path

    parts = PromptParts(
        system="You are a developer support agent for NimbusFn. "
        "Answer using only the reference guide provided.",
        stable_blocks=[REFERENCE_DOC],
        variable="A customer named Arjun (arjun.rao@example.com) says his "
        "function's invocation count doubled overnight. In one sentence, "
        "what should the agent check first?",
    )

    eligibility = check_cache_eligibility(parts, model=MODEL)
    print(f"--- Cache eligibility check ---\n{eligibility.message}\n")
    if not eligibility.eligible:
        print("Stopping -- adjust REFERENCE_DOC to be longer before spending API calls.")
        sys.exit(1)

    redacted = client.redact_and_trim_parts(parts)
    print(f"PII fields redacted from structured parts: {len(redacted.mapping)}\n")

    body = build_anthropic_cache_request(redacted.parts, model=MODEL, max_tokens=200)

    print("--- Call 1 (expect a cache WRITE: cache_creation_input_tokens > 0) ---")
    response_1 = call_claude(body)
    usage_1 = parse_anthropic_usage(response_1)
    text_1 = redacted.restore("".join(b["text"] for b in response_1["content"] if b["type"] == "text"))
    print(f"Response: {text_1}")
    print(
        f"input_tokens={usage_1.input_tokens}  "
        f"cache_creation_input_tokens={usage_1.cache_creation_input_tokens}  "
        f"cache_read_input_tokens={usage_1.cache_read_input_tokens}  "
        f"cache_write={usage_1.cache_write}  cache_hit={usage_1.cache_hit}"
    )
    savings_1 = usage_1.estimated_cost_savings_percent(MODEL)
    print(
        f"Estimated cost vs. no caching at all: {savings_1:+.1f}%  "
        "(negative is expected here -- writing a new cache entry costs a "
        "premium, 1.25x the base input price for a 5-minute TTL; the "
        "payoff comes on the next call, not this one)"
    )

    print("\n--- Call 2, seconds later, IDENTICAL stable prefix (expect a cache HIT) ---")
    response_2 = call_claude(body)
    usage_2 = parse_anthropic_usage(response_2)
    text_2 = redacted.restore("".join(b["text"] for b in response_2["content"] if b["type"] == "text"))
    print(f"Response: {text_2}")
    print(
        f"input_tokens={usage_2.input_tokens}  "
        f"cache_creation_input_tokens={usage_2.cache_creation_input_tokens}  "
        f"cache_read_input_tokens={usage_2.cache_read_input_tokens}  "
        f"cache_write={usage_2.cache_write}  cache_hit={usage_2.cache_hit}  "
        f"({usage_2.percent_of_input_from_cache}% of this call's input came from cache)"
    )
    savings_2 = usage_2.estimated_cost_savings_percent(MODEL)
    print(
        f"Estimated cost vs. no caching at all: {savings_2:+.1f}%  "
        "(this is the real payoff -- a cache READ is billed at 10% of "
        "the base input price for this model)"
    )

    print("\n--- Verdict ---")
    if usage_2.cache_hit:
        print("Confirmed: the second call got a real cache hit from the live API.")
        print(
            f"\nImportant distinction: the TOKEN COUNT processed was identical "
            f"both calls ({usage_2.input_tokens + usage_2.cache_read_input_tokens} "
            "tokens either way) -- caching never reduces how many tokens the "
            "model processes. What changed is the PRICE of the cached portion. "
            "Reusing this same stable prefix a third, fourth, fifth time would "
            f"repeat call 2's {savings_2:+.1f}% cost saving each time, while call "
            "1's one-time write premium is paid only once per cache TTL window."
        )
    else:
        print(
            "No cache hit on call 2. Either the >5-minute TTL expired between "
            "calls, the stable prefix wasn't actually byte-identical, or this "
            "model/account doesn't have caching enabled -- worth checking "
            "before trusting the caching feature to save anything in production."
        )

    print("\n" + "=" * 60)
    print("SUMMARY -- token & cost reduction, plainly stated")
    print("=" * 60)
    tokens_call1 = usage_1.input_tokens + usage_1.cache_creation_input_tokens
    tokens_call2 = usage_2.input_tokens + usage_2.cache_read_input_tokens
    print(f"Tokens sent, call 1 (cache write):  {tokens_call1}")
    print(f"Tokens sent, call 2 (cache read):    {tokens_call2}")
    token_delta = tokens_call1 - tokens_call2
    print(
        f"Token count change: {token_delta:+d}  "
        "(expected to be ~0 -- caching changes PRICE PER TOKEN, not how many "
        "tokens are sent; token count is not what this mechanism reduces)"
    )
    print(f"Cost change, call 1 vs. an uncached call of the same size:  {savings_1:+.1f}%")
    print(f"Cost change, call 2 vs. an uncached call of the same size:  {savings_2:+.1f}%")
    if usage_2.cache_hit:
        print(
            f"\nIn plain terms: call 2 cost {abs(savings_2):.1f}% LESS than paying "
            f"full price for the same {tokens_call2} tokens, because "
            f"{usage_2.percent_of_input_from_cache}% of its input was served from "
            "cache at Anthropic's confirmed 90% cache-read discount. Call 1 cost "
            f"{abs(savings_1):.1f}% MORE than full price -- that's the one-time "
            "cache-write premium, paid once per TTL window, not on every call."
        )
    else:
        print(
            "\nIn plain terms: no cache hit occurred on call 2, so neither call "
            "saved any money this run (0% real savings) -- see the Verdict above "
            "for the likely reason, and re-run to confirm before concluding "
            "caching doesn't work for this account."
        )


if __name__ == "__main__":
    main()
