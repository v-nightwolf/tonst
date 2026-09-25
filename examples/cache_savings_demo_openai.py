"""
examples/cache_savings_demo_openai.py
------------------------------
Measures REAL prompt-caching savings against the actual api.openai.com
endpoint -- the OpenAI sibling of examples/cache_savings_demo_anthropic.py. Same
pattern, different provider: build an eligible request, call the real
API twice with an identical stable prefix, parse the real `usage` field
from both responses, and print the measured cost difference using
tonst.providers.openai's real per-model discount table.

Worth knowing going in, because it changes what "success" looks like
compared to the Anthropic script: OpenAI's caching is AUTOMATIC, not
something this request opts into with a marker (see
tonst/providers/openai.py's docstring) -- so call 1 here is just a
normal, fully-priced call that HAPPENS to leave a cache entry behind
(there's no distinct "write" step to charge for on the automatic path,
unlike Anthropic's explicit cache_control). Only call 2's cached_tokens
count is the actual signal to watch.

Setup:
    1. Get an API key from https://platform.openai.com/api-keys
    2. Put it in a .env file in the repository root (gitignored):
           OPENAI_API_KEY=sk-...
       or export it directly in your shell.
    3. python3 examples/cache_savings_demo_openai.py

What you should see: call 1's cached_tokens should be 0 or small (the
prefix hasn't been seen before). Call 2, sent seconds later with the
IDENTICAL stable prefix, should show cached_tokens > 0, credited in
multiples of 128 tokens above the 1,024-token floor -- that's the real,
measured discount.
"""
# Run from anywhere: make the repo root importable without `pip install -e .`.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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
from tonst import TonstClient, PromptParts
from tonst.providers import openai as openai_provider

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o"  # flat 1,024-token cache minimum -- see providers.openai.CACHE_MINIMUM_TOKENS

# Reused verbatim from examples/cache_savings_demo_anthropic.py: the same
# ~1,560-token (chars/4 estimate) reference doc that measured 1,547 real
# input tokens against Anthropic's tokenizer -- comfortably above
# OpenAI's flat 1,024-token minimum too, real tokenizer differences
# notwithstanding.
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

Section 5: Plan Tiers & Feature Matrix
- Starter ($0/mo): 1 project, community support only, 99.5% uptime SLA,
  shared compute, 5GB storage, no custom domains.
- Pro ($29/mo): 10 projects, email support (next business day), 99.9%
  uptime SLA, dedicated compute pool, 100GB storage, 3 custom domains,
  daily automated backups retained for 7 days.
- Business ($99/mo): unlimited projects, priority email + chat support
  (4-hour response), 99.95% uptime SLA, dedicated compute, 1TB storage,
  unlimited custom domains, backups retained for 30 days, staging
  environments, SSO via SAML.
- Enterprise (custom pricing, annual invoice): everything in Business
  plus a named account manager, custom SLA negotiation, dedicated
  infrastructure, backups retained for 1 year, audit logging, and a
  private Slack channel with the support team.
- Feature availability questions from customers should always be
  answered against the current published pricing page, not from memory
  -- the matrix above is reviewed quarterly and can lag a recent change.

Section 6: Outage Communication Protocol
- Every P1 incident gets a public status-page entry within 15 minutes
  of confirmation, regardless of how few customers appear affected.
- Status updates during an active P1 go out every 30 minutes on a fixed
  cadence, even if the update is just "still investigating, next update
  in 30 minutes" -- silence during an outage erodes trust faster than a
  slow-but-communicative resolution.
- Once resolved, a P1 always gets a post-incident summary within 24
  hours: what happened, customer impact, and what's changing to prevent
  recurrence. This is drafted by the on-call engineer and reviewed by
  the support lead before publishing.
- P2 and P3 issues are not posted to the public status page unless they
  affect a large enough customer segment that multiple independent
  tickets are expected; use judgment and escalate to the support lead
  if unsure whether an issue crosses that line.

Section 7: Data Retention & Account Closure
- Active account data is retained indefinitely while the subscription
  is active, subject to the plan's own backup retention window listed
  in Section 5.
- On voluntary account closure, project data is retained for 30 days
  in a recoverable state before permanent deletion, to cover accidental
  cancellations. Customers are told this explicitly during the closure
  flow.
- On involuntary closure (repeated failed payment beyond the grace
  period in Section 1), the same 30-day recoverable window applies
  before permanent deletion, and a reactivation link is included in
  every suspension notice email.
- Enterprise accounts under contract have their own data-retention
  terms specified in the master service agreement; check the contract
  before applying the default 30-day window to an enterprise closure.

Section 8: Known Limitations (do not promise fixes on these)
- Custom domain SSL certificates can take up to 24 hours to provision
  after DNS is correctly pointed -- this is a known, expected delay, not
  a bug, and should be communicated as such rather than escalated.
- The platform does not currently support multi-region failover for
  Starter or Pro tiers; only Business and Enterprise get automatic
  regional failover. This is a known gap on the roadmap, not something
  support can work around per-customer.
- Bulk CSV import is capped at 50,000 rows per file; larger imports
  need to be split client-side. This limit is intentional (protects
  shared compute on lower tiers) and is not adjustable per-account
  except for Enterprise, where it's a contract term.
""".strip()


def call_openai(body: dict) -> dict:
    """Direct call to the real OpenAI API. Returns the full decoded JSON
    response so usage.prompt_tokens_details.cached_tokens can be read."""
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "content-type": "application/json",
        },
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()


def main():
    if not OPENAI_API_KEY:
        print(
            "No OPENAI_API_KEY set (checked environment and ./.env).\n"
            "This script needs a real key to measure real caching -- "
            "there's no mock mode, since the whole point is to observe\n"
            "actual usage.prompt_tokens_details.cached_tokens from "
            "api.openai.com.\n\n"
            "Add one of:\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "  echo 'OPENAI_API_KEY=sk-...' > .env   # gitignored\n"
        )
        sys.exit(1)

    client = TonstClient(call_fn=lambda p: p)  # call_fn unused on this path

    parts = PromptParts(
        system="You are a Tier 2 support agent for Acme Cloud Hosting. "
        "Answer using only the reference guide provided.",
        stable_blocks=[REFERENCE_DOC],
        variable="A customer named Priya (priya.sharma@example.com) says "
        "she was charged twice this month. In one sentence, what should "
        "the agent check first?",
    )

    eligibility = openai_provider.check_cache_eligibility(parts, model=MODEL)
    print(f"--- Cache eligibility check ---\n{eligibility.message}\n")
    if not eligibility.eligible:
        print("Stopping -- adjust REFERENCE_DOC to be longer before spending API calls.")
        sys.exit(1)

    redacted = client.redact_and_trim_parts(parts)
    print(f"PII fields redacted from structured parts: {len(redacted.mapping)}\n")

    # Automatic path -- no explicit breakpoint needed or used; ordering
    # alone (stable content first) is what makes this cacheable.
    body = openai_provider.build_openai_cache_request(redacted.parts, model=MODEL, max_tokens=200)
    # A routing hint that increases the odds two calls land on the same
    # backend, per OpenAI's own docs -- doesn't force a hit, but helps.
    body["prompt_cache_key"] = "tonst-cache-savings-demo"

    print("--- Call 1 (automatic caching -- no distinct 'write' cost; this is a normal-priced call) ---")
    response_1 = call_openai(body)
    usage_1 = openai_provider.parse_openai_usage(response_1)
    text_1 = redacted.restore(response_1["choices"][0]["message"]["content"])
    print(f"Response: {text_1}")
    print(
        f"input_tokens={usage_1.input_tokens}  "
        f"cache_read_input_tokens={usage_1.cache_read_input_tokens}  "
        f"cache_hit={usage_1.cache_hit}"
    )

    print("\n--- Call 2, seconds later, IDENTICAL stable prefix (expect cached_tokens > 0) ---")
    response_2 = call_openai(body)
    usage_2 = openai_provider.parse_openai_usage(response_2)
    text_2 = redacted.restore(response_2["choices"][0]["message"]["content"])
    print(f"Response: {text_2}")
    print(
        f"input_tokens={usage_2.input_tokens}  "
        f"cache_read_input_tokens={usage_2.cache_read_input_tokens}  "
        f"cache_hit={usage_2.cache_hit}  "
        f"({usage_2.percent_of_input_from_cache}% of this call's input came from cache)"
    )
    savings_2 = openai_provider.estimated_cost_savings_percent(usage_2, MODEL)
    print(
        f"Estimated cost vs. no caching at all: {savings_2:+.1f}%  "
        f"(using {MODEL}'s real discount rate from providers/openai.py's table)"
    )

    print("\n--- Verdict ---")
    if usage_2.cache_hit:
        print("Confirmed: the second call got a real cache hit from the live API.")
    else:
        print(
            "No cache hit on call 2. OpenAI's automatic caching is best-effort, "
            "not guaranteed -- possible causes: the cache entry expired (30-minute "
            "sliding window on the newest models, shorter/best-effort on others), "
            "the request landed on a different backend instance, or the stable "
            "prefix wasn't actually byte-identical. Worth re-running before "
            "concluding caching isn't working for this model/account."
        )


if __name__ == "__main__":
    main()
