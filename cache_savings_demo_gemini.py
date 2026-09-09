"""
cache_savings_demo_gemini.py
------------------------------
Measures REAL prompt-caching savings against the actual Gemini API
(generativelanguage.googleapis.com) -- the Gemini sibling of
cache_savings_demo_anthropic.py and cache_savings_demo_openai.py. Same
overall pattern: build an eligible request, call the real API twice
with an identical stable prefix, parse the real `usageMetadata` field
from both responses, and print the measured cost difference using
tonst.providers.gemini's confirmed uniform 10% cache-read rate.

This one tests Gemini's IMPLICIT (automatic) caching specifically --
see tonst/providers/gemini.py's docstring for why that's a meaningfully
different claim from Anthropic/OpenAI's caching. Google's own docs are
explicit that implicit caching is BEST-EFFORT, not guaranteed: there is
no marker to set, no error if it doesn't activate, and no promise it
will on any given call. A "no cache hit" result from this script is a
real, expected possible outcome, not necessarily evidence of a bug --
unlike the Anthropic/OpenAI scripts, where a correctly-shaped repeat
request is expected to hit close to 100% of the time.

Gemini's per-model minimum stable-prefix length is also higher than
Anthropic/OpenAI's (4,096 tokens for gemini-3.6-flash, the model used
here, vs. 1,024-1,536 for the other two) -- see providers/gemini.py's
CACHE_MINIMUM_TOKENS. REFERENCE_DOC below is deliberately longer than
the other two scripts' copies for exactly this reason.

Model note (found via live testing, Sept 2026): this script originally
targeted gemini-2.5-flash, which is still in providers/gemini.py's
tables but returns a 404 ("this model is no longer available to new
users") for accounts/keys created after Google's cutover -- the API
error itself names gemini-3.6-flash as the replacement, which is what
this script now calls. Existing projects with prior access to
gemini-2.5-flash may still be able to use it; new keys cannot.

Setup:
    1. Get an API key from https://aistudio.google.com/apikey
    2. IMPORTANT (found via live testing, Sept 2026): enable BILLING on
       that key's Google AI Studio / Cloud project. A free-tier key gets
       a confirmed ZERO-TOKEN cache storage quota -- caching (implicit
       AND explicit) cannot activate at all on one, regardless of how
       this script shapes the request. See providers/gemini.py's
       module docstring for the exact error this produces if you try
       explicit caching on a free-tier key.
    3. Put it in a .env file in this directory (gitignored):
           GEMINI_API_KEY=AI...
       or export it directly in your shell.
    4. python3 cache_savings_demo_gemini.py

What you should see (assuming billing is enabled -- see step 2): call
1's cachedContentTokenCount will likely be 0 (nothing to reuse yet).
Call 2, sent seconds later with the IDENTICAL stable prefix, MAY show
cachedContentTokenCount > 0 -- but because this is implicit/best-effort
caching, it's genuinely possible to see 0 on both calls even with
everything shaped correctly and billing enabled. That's Google's
documented behavior, not this script failing. If you see 0 on every
call including several retries, suspect the free-tier quota first --
it's a far more common cause than bad luck on a best-effort mechanism.
"""

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
from tonst.providers import gemini as gemini_provider

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
MODEL = "gemini-3.6-flash"  # 4,096-token cache minimum -- see providers.gemini.CACHE_MINIMUM_TOKENS
# NOTE: gemini-2.5-flash (2,048-token minimum) was the original choice here,
# but Google returns 404 "no longer available to new users" for it on new
# API keys as of Sept 2026 and names this model as the replacement.

# Longer than the Anthropic/OpenAI scripts' copy of this same document --
# Gemini's minimum for gemini-3.6-flash (4,096 tokens) is roughly 4x
# OpenAI's flat 1,024, so even the 12-section version (~2,540 tokens,
# enough for gemini-2.5-flash's old 2,048 minimum) isn't enough margin
# here. Sections 13-25 exist purely to clear the higher bar with
# real-tokenizer slack, not because Gemini's support org has more to
# say. NOTE (found via live testing, Sept 2026): the chars//4 estimator
# below is optimistic vs. Gemini's real tokenizer -- a first pass at
# ~4,549 est. tokens measured as only 4,026 REAL tokens against the
# live API (a ~12% gap), which is BELOW the 4,096 minimum and explains
# why no cache hit occurred despite the eligibility check saying
# "likely eligible." Sections 23-25 were added specifically to restore
# real margin -- current estimate is ~5,235 chars//4 tokens, projecting
# to roughly ~4,600+ real tokens, about 13% above the true minimum.
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

Section 9: Compliance & Certifications
- Acme Cloud Hosting maintains SOC 2 Type II certification, renewed
  annually; the current report is available under NDA to Business and
  Enterprise customers on request via the security@acme-support.example.com
  address, not through standard Tier 2 support tickets.
- GDPR data-processing agreements (DPAs) are available for any EU
  customer and must be countersigned before any EU personal data is
  processed on the platform -- Tier 2 support should route DPA requests
  to the legal team, never attempt to draft or approve terms directly.
- HIPAA-eligible hosting is only available on Enterprise plans with a
  signed Business Associate Agreement (BAA) in place; a customer asking
  about HIPAA on a lower tier should be told they need to upgrade and
  sign a BAA first, not that the feature is simply unavailable.
- PCI DSS compliance applies only to the payment-processing subsystem,
  not to customer-hosted applications -- a common point of confusion
  worth clarifying proactively when a customer asks about "PCI
  compliance" for their own site.

Section 10: API Rate Limits & Integration Partners
- The public API enforces 100 requests/minute on Starter, 1,000/minute
  on Pro, 10,000/minute on Business, and a negotiated limit on
  Enterprise -- rate-limit errors return HTTP 429 with a Retry-After
  header that customers are often not checking.
- Official integration partners (as of this document's last review):
  Zapier, Segment, Datadog, and PagerDuty. Integrations through any of
  these are supported end-to-end by Tier 2; anything built on a
  third-party unofficial connector is best-effort only.
- Webhook delivery retries up to 5 times with exponential backoff over
  roughly 24 hours before a delivery is marked permanently failed and
  surfaced in the account's webhook delivery log.
- API keys do not expire automatically, but are invalidated immediately
  if flagged for suspected leakage (e.g. found in a public repository
  scan) -- if a customer reports unexpected API behavior, check the key
  status before assuming a platform bug.

Section 11: Onboarding & Account Provisioning
- New Starter and Pro accounts are provisioned instantly on signup with
  no manual review step.
- Business accounts undergo an automated fraud-risk check that
  typically completes within 5 minutes; a small percentage are flagged
  for manual review, which can take up to 1 business day -- customers
  should be told this proactively rather than left wondering why
  their account isn't active yet.
- Enterprise accounts are provisioned by the solutions engineering team
  after a signed order form, typically within 2 business days of
  contract execution; Tier 2 support does not have the ability to
  expedite this and should route urgency requests to the assigned
  account manager.
- Account ownership transfers require verification from both the
  current and incoming owner's registered email addresses before Tier 2
  can process the change -- never process a transfer from a single
  party's request alone, regardless of how urgent it seems.

Section 12: Localization & Multi-Region Support
- The platform's control panel is available in English, Spanish,
  French, German, Japanese, and Portuguese; all other languages fall
  back to English automatically.
- Data residency options (EU, US, and APAC regions) are available on
  Business and Enterprise plans only; Starter and Pro accounts are
  hosted in the US region by default with no residency choice.
- Support is provided in English only for Tier 2; Enterprise customers
  with a dedicated account manager may have access to additional
  language support arranged separately, which Tier 2 should not
  attempt to replicate or promise to other tiers.
- Cross-region data transfer between a customer's own regions is
  self-service for Business and Enterprise, and disabled by default
  pending an explicit customer request due to the compliance
  implications covered in Section 9.

Section 13: Security Practices
- All data in transit uses TLS 1.2 or higher; TLS 1.0/1.1 are disabled
  platform-wide and cannot be re-enabled per-account, including for
  Enterprise customers with legacy integrations.
- Data at rest is encrypted using AES-256 on all storage tiers, with
  key rotation every 90 days managed entirely by the platform -- there
  is no customer-managed-key option today, which should be disclosed
  upfront to any customer asking about BYOK (bring your own key).
- Penetration testing is performed by a third-party firm twice yearly;
  summary findings (not full reports) are available to Enterprise
  customers under NDA via the same process as the SOC 2 report request
  in Section 9.
- Bug bounty reports go to security@acme-support.example.com and are
  never to be handled or acknowledged by Tier 2 support directly --
  forward immediately without attempting to reproduce or comment on
  severity.

Section 14: Disaster Recovery & Backups
- Cross-region backup replication runs automatically for Business and
  Enterprise plans; Starter and Pro backups (per the retention windows
  in Section 5) are single-region only.
- Recovery Time Objective (RTO) for a full regional failover is 4 hours
  for Enterprise with a signed DR addendum, and best-effort (no
  contractual number) for all other tiers.
- Customer-initiated restore requests from a backup snapshot go through
  a support ticket with account ID and desired restore point; Tier 2
  can approve restores within the plan's own retention window without
  escalation, but anything older requires a manager approval since it
  may involve pulling from cold storage.
- A failed backup job triggers an internal alert and an automatic retry
  within 1 hour; customers are not notified of a single failed-then-
  retried-successfully backup, only of a backup that fails twice in a
  row.

Section 15: Third-Party Audits & Certifications Detail
- The most recent SOC 2 Type II audit period and the current ISO 27001
  certificate number are both listed on the trust page, which Tier 2
  should point customers to rather than reciting from memory, since
  these are renewed on independent yearly cycles and this document is
  not the source of truth for exact dates.
- A customer's own auditor requesting a walkthrough or questionnaire
  response should be routed to security@acme-support.example.com, not
  answered ad hoc by Tier 2, even for questions that seem simple.
- Sub-processor list changes (new third-party vendors that touch
  customer data) are announced via email to all Business/Enterprise
  admins at least 30 days before taking effect, per the DPA terms
  referenced in Section 9.

Section 16: Data Export & Portability
- Customers can self-service export their project data at any time via
  the dashboard, in a documented JSON/CSV bundle format, regardless of
  plan tier -- this is never gated behind a support ticket.
- Full-account export requests (everything, not just one project) for
  Enterprise customers with very large datasets may need to go through
  the solutions engineering team for a manual bulk transfer instead of
  the self-service tool, if the export exceeds roughly 500GB.
- Exported data does not include billing history or internal audit
  logs; those are provided separately on request per the retention
  terms in Section 7, and only after identity verification.

Section 17: Service Credits & SLA Remediation
- SLA credit claims for missed uptime targets (Section 5) must be filed
  within 30 days of the qualifying incident; late claims are declined
  by default and require a manager exception to honor.
- Credits are issued as account credit against a future invoice, never
  as a cash refund, regardless of plan tier -- this should be stated
  clearly when a customer asks about "getting money back" for an
  outage.
- A single incident can only be claimed once per affected account, even
  if it technically breached multiple SLA thresholds (e.g., both uptime
  and a stated response-time SLA) -- credits are not stacked.

Section 18: Multi-Factor Authentication & Access Control
- MFA is optional but strongly recommended for all account tiers, and
  mandatory for any account with billing-admin or owner-level
  permissions on Business and Enterprise plans as of the most recent
  security policy update.
- Lost-MFA-device account recovery requires identity verification via
  the registered billing email plus one additional factor (a recent
  invoice number or the last four digits of the payment method on
  file) before Tier 2 can process an MFA reset -- never reset MFA off a
  single unverified request, regardless of how urgent the customer
  sounds.
- Role-based access control (viewer, editor, admin, owner) is available
  on Business and Enterprise; Starter and Pro accounts have a single
  implicit owner role with no sub-user permission tiers.

Section 19: Deprecation & End-of-Life Policy
- Deprecated API versions and platform features get a minimum 6-month
  sunset notice, published on the changelog and emailed to affected
  accounts based on detected usage, before removal.
- During the sunset window, deprecated endpoints continue to function
  normally but return a `Deprecation` response header; this is not an
  error and should not be treated as a bug report when customers ask
  about it.
- Emergency security-driven deprecations (rare) can bypass the standard
  6-month window with as little as 24 hours' notice; these are always
  accompanied by a direct email to every affected account, not just a
  changelog entry, given the shortened timeline.

Section 20: Support Channels & Response Hours
- Email support is available 24/7 for all tiers with the response-time
  targets listed per-tier in Section 5; live chat is available during
  business hours (9am-6pm in the customer's detected region) for
  Business and Enterprise only.
- Phone support is Enterprise-only, arranged through the named account
  manager, and is not a channel Tier 2 can offer or schedule directly
  for Business or lower tiers, even as a one-time exception.
- Community forum questions are not covered by any SLA and are answered
  on a best-effort basis by both staff and other customers; a forum
  post is never an acceptable substitute for a ticket when a customer
  needs a guaranteed response time.

Section 21: Custom Domains & DNS Configuration
- Custom domain verification requires either a TXT record or a CNAME,
  customer's choice; propagation can take up to 24 hours, consistent
  with the SSL provisioning delay noted in Section 8, and both delays
  often get reported together as a single "my domain isn't working"
  ticket.
- Wildcard subdomain support is available on Business and Enterprise
  only; Pro and Starter accounts can add individual subdomains but not
  a wildcard record.
- DNS changes made outside the platform's own DNS management (i.e., at
  a third-party registrar) are outside Tier 2's visibility -- always
  ask whether DNS is managed through Acme or externally before
  troubleshooting a domain issue.

Section 22: Internal Escalation Etiquette
- When escalating to engineering on-call, always include the account
  ID, affected region, a timestamp in UTC, and the exact error message
  or screenshot -- incomplete escalations are the single largest cause
  of delayed P1 response per the quarterly incident retrospective.
- Do not escalate a ticket twice through two different channels (e.g.,
  paging on-call AND opening an engineering Jira ticket) for the same
  issue; pick one path and note it in the ticket so on-call isn't
  duplicating triage effort.
- If a customer explicitly asks to speak with an engineer directly,
  explain that Tier 2 handles all first-line triage and engineering
  engagement happens through escalation, not direct customer contact,
  except for named Enterprise account managers who may loop in a
  solutions engineer by design.

Section 23: Third-Party Marketplace Add-Ons
- The marketplace lists add-ons built by both Acme and approved third
  parties; Acme-built add-ons carry the same support SLA as the core
  platform, while third-party add-ons are supported by their own
  publisher, not Tier 2 -- always check the publisher badge before
  troubleshooting an add-on issue as if it were a platform bug.
- Add-on billing is consolidated onto the customer's existing invoice
  regardless of publisher, but refund requests for a third-party add-on
  must be routed to that publisher's own support channel, listed on its
  marketplace listing page, not handled through Acme billing.
- Add-ons requesting elevated account permissions (beyond basic
  read-only project access) go through a manual review before being
  allowed into the marketplace; an add-on already listed there has
  already cleared this review and should not be treated as suspicious
  by default when a customer asks about it.

Section 24: Usage-Based Overage Billing
- Business and Enterprise plans include a base resource allotment
  (compute-hours and bandwidth); usage beyond the allotment is billed
  at the published per-unit overage rate on the next invoice, not
  blocked in real time, so a customer will not suddenly lose access
  mid-month for going over.
- Starter and Pro plans hard-cap at their listed resource allotment
  instead of billing overages; exceeding the cap throttles the
  account's compute rather than generating a surprise charge, and the
  dashboard shows a clear "approaching limit" banner before this
  happens.
- Overage disputes follow the same 30-day window as the SLA credit
  process in Section 17, and require the customer to point to the
  specific invoice line item in question rather than a general "this
  seems too high" claim, so the billing team has something concrete to
  audit.

Section 25: Internal Documentation Change Log Practices
- This reference guide is reviewed quarterly by the support-enablement
  team, with material changes (pricing, SLA numbers, escalation
  contacts) requiring sign-off from the relevant department owner
  before publishing an update.
- Minor wording clarifications do not require the full review cycle and
  can be merged by any Tier 2 lead, but any change touching a number a
  customer could rely on (a price, a percentage, an SLA hour count) is
  never treated as "minor," regardless of how small the change looks.
- Agents who spot outdated or contradictory guidance in this document
  should flag it in the enablement team's internal channel rather than
  silently working around it or telling customers something not
  written here, since undocumented tribal knowledge is exactly what
  this guide exists to prevent.
""".strip()


def call_gemini(body: dict) -> dict:
    """Direct call to the real Gemini API. Returns the full decoded JSON
    response so usageMetadata.cachedContentTokenCount can be read."""
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        params={"key": GEMINI_API_KEY},
        headers={"content-type": "application/json"},
        json=body,
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()


def main():
    if not GEMINI_API_KEY:
        print(
            "No GEMINI_API_KEY (or GOOGLE_API_KEY) set (checked environment "
            "and ./.env).\n"
            "This script needs a real key to measure real caching -- there's "
            "no mock mode, since the whole point is to observe actual\n"
            "usageMetadata.cachedContentTokenCount from the live Gemini API.\n\n"
            "Add one of:\n"
            "  export GEMINI_API_KEY=AI...\n"
            "  echo 'GEMINI_API_KEY=AI...' > .env   # gitignored\n"
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

    eligibility = gemini_provider.check_cache_eligibility(parts, model=MODEL)
    print(f"--- Cache eligibility check ---\n{eligibility.message}\n")
    if not eligibility.eligible:
        print("Stopping -- adjust REFERENCE_DOC to be longer before spending API calls.")
        sys.exit(1)

    redacted = client.redact_and_trim_parts(parts)
    print(f"PII fields redacted from structured parts: {len(redacted.mapping)}\n")

    # Implicit (automatic) path -- no separate resource to create, just
    # correct ordering. See providers/gemini.py for the explicit
    # CachedContent alternative, which needs its own create/reference
    # round-trip and isn't exercised by this script.
    body = gemini_provider.build_gemini_content_request(redacted.parts, model=MODEL, max_output_tokens=200)

    print("--- Call 1 (implicit caching -- best-effort, no guaranteed write step) ---")
    response_1 = call_gemini(body)
    usage_1 = gemini_provider.parse_gemini_usage(response_1)
    text_1 = redacted.restore(response_1["candidates"][0]["content"]["parts"][0]["text"])
    print(f"Response: {text_1}")
    print(f"input_tokens={usage_1.input_tokens}  cache_read_input_tokens={usage_1.cache_read_input_tokens}  cache_hit={usage_1.cache_hit}")

    print("\n--- Call 2, seconds later, IDENTICAL stable prefix (MAY show a cache hit -- not guaranteed) ---")
    response_2 = call_gemini(body)
    usage_2 = gemini_provider.parse_gemini_usage(response_2)
    text_2 = redacted.restore(response_2["candidates"][0]["content"]["parts"][0]["text"])
    print(f"Response: {text_2}")
    print(
        f"input_tokens={usage_2.input_tokens}  "
        f"cache_read_input_tokens={usage_2.cache_read_input_tokens}  "
        f"cache_hit={usage_2.cache_hit}  "
        f"({usage_2.percent_of_input_from_cache}% of this call's input came from cache)"
    )
    savings_2 = gemini_provider.estimated_implicit_cache_cost_savings_percent(usage_2)
    print(f"Estimated cost vs. no caching at all: {savings_2:+.1f}%  (using Gemini's confirmed uniform 10% cache-read rate)")

    print("\n--- Verdict ---")
    if usage_2.cache_hit:
        print("Confirmed: the second call got a real cache hit from the live API.")
    else:
        print(
            "No cache hit on call 2. Unlike Anthropic/OpenAI, this is a genuinely "
            "possible outcome even with everything shaped correctly -- Google's own "
            "docs describe implicit caching as best-effort with no guarantee. Try "
            "re-running immediately (send requests with similar prefixes close "
            "together, per Google's own guidance) before concluding caching isn't "
            "active for this model/account. For a GUARANTEED cache, use Gemini's "
            "explicit CachedContent path instead -- see providers/gemini.py's "
            "build_cached_content_resource() / build_generate_request_from_cache()."
        )


if __name__ == "__main__":
    main()
