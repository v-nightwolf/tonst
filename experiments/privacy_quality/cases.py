"""
cases.py -- the answer-quality benchmark's test prompts.

100 realistic work prompts (50 templates x 2 fictional entity bundles)
across 10 categories. Every person, company, address, number and key here
is invented. Each case records:

  prompt     the text a user would paste into an AI assistant
  sensitive  [(value, type)] -- the ground truth for the leakage check:
             every value that a privacy layer should keep from the provider
  expect     strings a good answer should contain after restoration
             (a name in a greeting, the right total, the right ID). Used
             for the objective part of the score, alongside the judge.

Types: NAME, EMAIL, PHONE, COMPANY, ADDRESS, ACCOUNT_ID, MONEY, SECRET,
IP_ADDRESS, CODENAME. ADDRESS has no detector yet, so it is expected to
leak -- the benchmark measures that honestly rather than skipping it.
"""

from __future__ import annotations

PEOPLE = [
    ("Priya", "Nair", "+91 98450 21733"),
    ("Daniel", "Okafor", "+1 415-555-0187"),
    ("Mei", "Tanaka", "+81 90-4418-2276"),
    ("Lucas", "Fernandes", "+55 11 98822-4410"),
    ("Anika", "Schulz", "+49 151 2384 9921"),
    ("Rahul", "Verma", "+91 99101 48823"),
    ("Grace", "Whitfield", "+44 7700 900412"),
    ("Omar", "Haddad", "+971 50 318 2267"),
    ("Sofia", "Marchetti", "+39 347 552 1904"),
    ("Ethan", "Brooks", "+1 646-555-0132"),
    ("Kavya", "Iyer", "+91 97390 66218"),
    ("Tomas", "Novak", "+420 603 118 447"),
]

COMPANIES = [
    ("Veltrix Logistics", "veltrix.io"),
    ("Brightwell Health", "brightwellhealth.com"),
    ("Northpeak Capital", "northpeakcap.com"),
    ("Luma Retail", "lumaretail.in"),
    ("Corvane Systems", "corvane.dev"),
    ("Halden & Rowe LLP", "haldenrowe.law"),
    ("Saffron Foods", "saffronfoods.co"),
    ("Orbis Telecom", "orbistel.net"),
    ("Kestrel Aerospace", "kestrel-aero.com"),
    ("Tidewater Insurance", "tidewaterins.com"),
]

ADDRESSES = [
    "42 MG Road, Indiranagar, Bengaluru 560038",
    "1180 Folsom Street, San Francisco, CA 94103",
    "7 Harrow Lane, Cambridge CB2 1TN",
    "Rua Augusta 1520, São Paulo 01304-001",
    "Friedrichstraße 88, 10117 Berlin",
    "Flat 9B, Palm Residency, Sector 45, Gurugram 122003",
]

CODENAMES = ["Project Bluefin", "Project Halcyon", "Project Ironleaf", "Project Nightjar", "Project Marigold"]
IPS = ["10.24.8.117", "172.19.4.62", "192.168.40.12", "10.8.33.201"]


def _fake_key(i: int) -> str:
    body = "".join("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"[(i * 7 + j * 13) % 57] for j in range(40))
    return f"sk-ant-api03-{body}"


def _fake_db_password(i: int) -> str:
    return ["Tr0ub4dor&3x", "Monsoon#2026!", "Blue$Falcon77", "Qx7!pLm29vRt", "Sunrise_Delta_88"][i % 5]


def _money(x: float, sym: str = "$") -> str:
    return f"{sym}{x:,.2f}"


def bundle(i: int) -> dict:
    """Deterministic set of fictional entities for instance i."""
    p1 = PEOPLE[i % len(PEOPLE)]
    p2 = PEOPLE[(i + 5) % len(PEOPLE)]
    p3 = PEOPLE[(i + 9) % len(PEOPLE)]
    c1, d1 = COMPANIES[i % len(COMPANIES)]
    c2, d2 = COMPANIES[(i + 3) % len(COMPANIES)]
    b = {
        "p1_first": p1[0], "p1_full": f"{p1[0]} {p1[1]}", "p1_phone": p1[2],
        "p1_email": f"{p1[0].lower()}.{p1[1].lower()}@{d1}",
        "p2_first": p2[0], "p2_full": f"{p2[0]} {p2[1]}", "p2_phone": p2[2],
        "p2_email": f"{p2[0].lower()}.{p2[1].lower()}@{d2}",
        "p3_first": p3[0], "p3_full": f"{p3[0]} {p3[1]}",
        "p3_email": f"{p3[0].lower()}@{d1}",
        "c1": c1, "c2": c2,
        "addr": ADDRESSES[i % len(ADDRESSES)],
        "acct": f"{48210 + i * 317}",
        "inv": f"INV-{20931 + i * 41}",
        "order": f"{771900 + i * 53}",
        "codename": CODENAMES[i % len(CODENAMES)],
        "ip": IPS[i % len(IPS)],
        "key": _fake_key(i),
        "dbpass": _fake_db_password(i),
    }
    # Line items for arithmetic cases.
    b["li1"] = 1200 + i * 37.5
    b["li2"] = 845.25 + i * 12
    b["li3"] = 310.0 + i * 5.75
    b["total"] = b["li1"] + b["li2"] + b["li3"]
    return b


# Each template: (category, id, prompt, sensitive slots, expect).
# Sensitive slots name keys in the bundle plus their type; expect is a
# list of bundle keys, literal strings, or callables(bundle) -> str.
TEMPLATES = [
    # ------------------------------------------------------------ support
    ("support", "complaint_reply",
     "Customer {p1_full} ({p1_email}, {p1_phone}) wrote in: order #{order} arrived with a cracked screen and "
     "they want a replacement, not a refund. They've been a customer of {c1} for six years. Draft a warm, "
     "concise reply from me ({p2_full}, support lead) confirming the replacement ships within 3 business days.",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("p1_phone", "PHONE"), ("order", "ACCOUNT_ID"),
      ("c1", "COMPANY"), ("p2_full", "NAME")],
     ["p1_first", "order", "p2_first"]),
    ("support", "refund_confirm",
     "Write a short email to {p1_full} at {p1_email} confirming we refunded {amt1} to the card ending 4417 for "
     "invoice {inv}. Mention it takes 5-7 business days to appear. Sign off as {p2_full}, {c1} Billing.",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("amt1", "MONEY"), ("inv", "ACCOUNT_ID"),
      ("p2_full", "NAME"), ("c1", "COMPANY")],
     ["p1_first", "amt1", "inv"]),
    ("support", "delivery_delay",
     "Our courier missed the delivery to {addr} for customer {p1_full} (phone {p1_phone}) twice. Write an "
     "apology SMS under 300 characters offering a new slot tomorrow 10am-1pm and a free shipping voucher.",
     [("addr", "ADDRESS"), ("p1_full", "NAME"), ("p1_phone", "PHONE")],
     ["p1_first"]),
    ("support", "account_locked",
     "{p1_full} from {c1} is locked out of their admin account (login {p1_email}, account {acct}) after too many "
     "password attempts. Write step-by-step instructions I can send them to verify identity and reset access.",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("p1_email", "EMAIL"), ("acct", "ACCOUNT_ID")],
     ["p1_first", "acct"]),
    ("support", "escalation",
     "Summarise this escalation for my manager {p2_full} in 4 bullet points:\n"
     "- Customer: {p1_full}, {c1}, account {acct}\n- Issue: billing charged twice for March ({amt1} each)\n"
     "- Contact tried: {p1_email}, {p1_phone}\n- Customer threatening to cancel a {amt2} annual contract",
     [("p2_full", "NAME"), ("p1_full", "NAME"), ("c1", "COMPANY"), ("acct", "ACCOUNT_ID"), ("amt1", "MONEY"),
      ("p1_email", "EMAIL"), ("p1_phone", "PHONE"), ("amt2", "MONEY")],
     ["p1_full", "acct"]),
    # ------------------------------------------------------------ sales
    ("sales", "demo_followup",
     "Write a follow-up email to {p1_full} ({p1_email}), Head of Ops at {c1}, after yesterday's demo. They "
     "liked the route-optimisation module and asked about SSO. Propose a call next Tuesday with our solutions "
     "engineer {p2_full}. Keep it under 150 words.",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("c1", "COMPANY"), ("p2_full", "NAME")],
     ["p1_first", "p2_full"]),
    ("sales", "renewal_quote",
     "{c1}'s contract (account {acct}) renews next month at {amt2}/year. Their champion {p1_full} asked for a "
     "10% discount. Write an internal note to my VP {p2_full} recommending we offer 7% for a two-year term, "
     "and state the resulting annual price.",
     [("c1", "COMPANY"), ("acct", "ACCOUNT_ID"), ("amt2", "MONEY"), ("p1_full", "NAME"), ("p2_full", "NAME")],
     ["p1_full", lambda b: _money(b["amt2_val"] * 0.93)]),
    ("sales", "intro_email",
     "My colleague {p2_full} ({p2_email}) knows {p1_full}, CTO of {c2}. Write a double-opt-in intro request "
     "{p2_first} can forward to {p1_first}, explaining that {c1} helps logistics teams cut fuel costs.",
     [("p2_full", "NAME"), ("p2_email", "EMAIL"), ("p1_full", "NAME"), ("c2", "COMPANY"), ("c1", "COMPANY")],
     ["p1_first", "c1"]),
    ("sales", "meeting_recap",
     "Recap for the CRM: call with {p1_full} and {p3_full} of {c1} today. Budget {amt2}, decision by end of "
     "quarter, blocker is security review (contact {p1_email}). Next step: send SOC 2 report. Write it as a "
     "5-line CRM note.",
     [("p1_full", "NAME"), ("p3_full", "NAME"), ("c1", "COMPANY"), ("amt2", "MONEY"), ("p1_email", "EMAIL")],
     ["p1_full", "amt2"]),
    ("sales", "discount_approval",
     "Draft a Slack message to {p2_full} asking approval for a {amt1} one-time discount on {c1}'s order "
     "#{order}. Justification: {p1_full} is moving 40 warehouses to us from a competitor.",
     [("p2_full", "NAME"), ("amt1", "MONEY"), ("c1", "COMPANY"), ("order", "ACCOUNT_ID"), ("p1_full", "NAME")],
     ["p2_first", "order", "amt1"]),
    # ------------------------------------------------------------ hr
    ("hr", "perf_review",
     "Turn these notes into a balanced performance-review paragraph for {p1_full} (employee ID {acct}), "
     "written by their manager {p2_full}: shipped {codename} two weeks early; strong mentor to {p3_first}; "
     "needs to improve written status updates; recommended for promotion to Senior Engineer.",
     [("p1_full", "NAME"), ("acct", "ACCOUNT_ID"), ("p2_full", "NAME"), ("codename", "CODENAME"),
      ("p3_first", "NAME")],
     ["p1_first", "codename"]),
    ("hr", "offer_letter",
     "Draft an offer letter for {p1_full}, {addr}, for the role of Product Designer at {c1}. Base salary "
     "{amt2} per year, start date 1 November, reporting to {p2_full}. Keep it one page.",
     [("p1_full", "NAME"), ("addr", "ADDRESS"), ("c1", "COMPANY"), ("amt2", "MONEY"), ("p2_full", "NAME")],
     ["p1_full", "amt2", "c1"]),
    ("hr", "leave_approval",
     "Reply to {p1_full} ({p1_email}) approving their leave from 14 to 25 October. Remind them to hand over "
     "{codename} on-call duties to {p3_full} and set an out-of-office.",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("codename", "CODENAME"), ("p3_full", "NAME")],
     ["p1_first", "p3_full"]),
    ("hr", "onboarding",
     "Create a first-week onboarding checklist for new hire {p1_full} joining {c1}'s data team. Their buddy is "
     "{p3_full}; laptop ships to {addr}; they need access to {codename} dashboards.",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("p3_full", "NAME"), ("addr", "ADDRESS"), ("codename", "CODENAME")],
     ["p3_full", "codename"]),
    ("hr", "exit_summary",
     "Summarise this exit interview in 3 themes for HR leadership. Employee: {p1_full}, 4 years at {c1}. Said "
     "manager {p2_full} rarely gave feedback, promotion to lead was promised twice and delayed, loved the team "
     "and {codename}. Leaving for {c2}.",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("p2_full", "NAME"), ("codename", "CODENAME"), ("c2", "COMPANY")],
     ["p2_full"]),
    # ------------------------------------------------------------ legal
    ("legal", "nda_summary",
     "Summarise the key terms of this NDA in plain English:\n\"This Mutual Non-Disclosure Agreement is entered "
     "into by {c1}, registered at {addr}, and {c2}. Confidential Information includes all details of "
     "{codename}. Obligations survive 3 years after termination. Signed for {c1} by {p1_full} and for {c2} by "
     "{p2_full}.\"",
     [("c1", "COMPANY"), ("addr", "ADDRESS"), ("c2", "COMPANY"), ("codename", "CODENAME"), ("p1_full", "NAME"),
      ("p2_full", "NAME")],
     ["c1", "c2", "codename"]),
    ("legal", "clause_extract",
     "From this contract excerpt, list the parties, the fee, the payment terms and the governing law as a "
     "table:\n\"{c1} shall pay {c2} a fixed fee of {amt2}, invoiced as {inv}, due 45 days from receipt. This "
     "agreement is governed by the laws of Singapore. Notices to {p1_full} at {p1_email}.\"",
     [("c1", "COMPANY"), ("c2", "COMPANY"), ("amt2", "MONEY"), ("inv", "ACCOUNT_ID"), ("p1_full", "NAME"),
      ("p1_email", "EMAIL")],
     ["c1", "c2", "amt2"]),
    ("legal", "vendor_risk",
     "Review this clause from {c2}'s vendor agreement and list the risks for us ({c1}): \"{c2} may subcontract "
     "any part of the services without notice and its total liability is capped at {amt1}.\" Our contact there "
     "is {p1_full}.",
     [("c2", "COMPANY"), ("c1", "COMPANY"), ("amt1", "MONEY"), ("p1_full", "NAME")],
     ["c2", "amt1"]),
    ("legal", "lease_summary",
     "Summarise this lease for tenant {p1_full}: premises at {addr}; landlord {c1}; monthly rent {amt1}; "
     "security deposit two months' rent; lock-in 11 months. State the deposit amount.",
     [("p1_full", "NAME"), ("addr", "ADDRESS"), ("c1", "COMPANY"), ("amt1", "MONEY")],
     ["p1_first", lambda b: _money(b["amt1_val"] * 2)]),
    ("legal", "dispute_letter",
     "Draft a firm but polite letter from {c1} to {c2} disputing invoice {inv} for {amt2}, because the "
     "services under {codename} were not delivered by the agreed date. Address it to {p2_full}.",
     [("c1", "COMPANY"), ("c2", "COMPANY"), ("inv", "ACCOUNT_ID"), ("amt2", "MONEY"), ("codename", "CODENAME"),
      ("p2_full", "NAME")],
     ["inv", "p2_full"]),
    # ------------------------------------------------------------ finance
    ("finance", "invoice_total",
     "Invoice {inv} from {c2} to {c1} (attn {p1_full}) has three lines: consulting {li1}, travel {li2}, "
     "software licences {li3}. What is the invoice total? Then write a one-line approval note.",
     [("inv", "ACCOUNT_ID"), ("c2", "COMPANY"), ("c1", "COMPANY"), ("p1_full", "NAME"), ("li1", "MONEY"),
      ("li2", "MONEY"), ("li3", "MONEY")],
     [lambda b: _money(b["total"]), "inv"]),
    ("finance", "payment_reminder",
     "Write a second payment reminder to {p1_full} ({p1_email}) at {c1}: invoice {inv} for {amt2} was due 30 "
     "days ago. Mention a 1.5% monthly late fee and state what the late fee is for one month.",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("c1", "COMPANY"), ("inv", "ACCOUNT_ID"), ("amt2", "MONEY")],
     ["inv", lambda b: _money(b["amt2_val"] * 0.015)]),
    ("finance", "expense_check",
     "{p1_full}'s expense report: hotel {li1}, flights {li2}, meals {li3}. Company policy caps a trip at "
     "{cap}. Is the report within policy? By how much is it over or under?",
     [("p1_full", "NAME"), ("li1", "MONEY"), ("li2", "MONEY"), ("li3", "MONEY"), ("cap", "MONEY")],
     [lambda b: _money(abs(b["cap_val"] - b["total"]))]),
    ("finance", "reconciliation",
     "Bank shows a payment of {li1} from {c1} (account {acct}) but our ledger has invoice {inv} at {li2} for "
     "them. What's the difference, and draft a note to {p1_full} ({p1_email}) asking about it.",
     [("li1", "MONEY"), ("c1", "COMPANY"), ("acct", "ACCOUNT_ID"), ("inv", "ACCOUNT_ID"), ("li2", "MONEY"),
      ("p1_full", "NAME"), ("p1_email", "EMAIL")],
     [lambda b: _money(abs(b["li1"] - b["li2"])), "inv"]),
    ("finance", "budget_variance",
     "{codename} budget was {amt2}; actual spend so far is {amt1} with two months left. Owner: {p1_full}. "
     "Write a 3-sentence status for the CFO including the remaining budget.",
     [("codename", "CODENAME"), ("amt2", "MONEY"), ("amt1", "MONEY"), ("p1_full", "NAME")],
     ["codename", lambda b: _money(b["amt2_val"] - b["amt1_val"])]),
    # ------------------------------------------------------------ health admin
    ("health_admin", "reschedule",
     "Write a text to patient {p1_full} ({p1_phone}) moving their physiotherapy appointment with Dr. "
     "{p2_full} from Thursday 4pm to Friday 11am at the {c1} clinic, {addr}.",
     [("p1_full", "NAME"), ("p1_phone", "PHONE"), ("p2_full", "NAME"), ("c1", "COMPANY"), ("addr", "ADDRESS")],
     ["p1_first", "p2_full"]),
    ("health_admin", "claim_followup",
     "Draft an email to {c2} claims team following up on claim for member {p1_full}, policy number {acct}, "
     "for {amt1} submitted 6 weeks ago. Reply-to {p2_email}.",
     [("c2", "COMPANY"), ("p1_full", "NAME"), ("acct", "ACCOUNT_ID"), ("amt1", "MONEY"), ("p2_email", "EMAIL")],
     ["acct", "p1_full"]),
    ("health_admin", "referral_note",
     "Write a brief referral cover note from Dr. {p2_full} at {c1} to the orthopaedics department for "
     "{p1_full}, DOB 12/03/1986, contact {p1_phone}, requesting a consultation about persistent knee pain.",
     [("p2_full", "NAME"), ("c1", "COMPANY"), ("p1_full", "NAME"), ("p1_phone", "PHONE")],
     ["p1_full", "p2_full"]),
    ("health_admin", "billing_query",
     "Patient {p1_full} (account {acct}) says she was billed {amt1} but insurance covers 80%. Work out what "
     "she should owe and write a reply explaining it.",
     [("p1_full", "NAME"), ("acct", "ACCOUNT_ID"), ("amt1", "MONEY")],
     [lambda b: _money(b["amt1_val"] * 0.2)]),
    ("health_admin", "shift_swap",
     "Nurses {p1_full} and {p3_full} want to swap their 20 October night shifts at {c1}. Write the approval "
     "message from ward manager {p2_full} and remind both to update the roster.",
     [("p1_full", "NAME"), ("p3_full", "NAME"), ("c1", "COMPANY"), ("p2_full", "NAME")],
     ["p1_first", "p3_first"]),
    # ------------------------------------------------------------ engineering
    ("engineering", "debug_config",
     "My service can't reach the database. Config:\n```\nDB_HOST={ip}\nDB_USER=svc_{codename_slug}\n"
     "DB_PASSWORD={dbpass}\nANTHROPIC_API_KEY={key}\nDB_PORT=5433\n```\nThe error is `connection refused on "
     "5432`. What's wrong?",
     [("ip", "IP_ADDRESS"), ("dbpass", "SECRET"), ("key", "SECRET")],
     ["5433"]),
    ("engineering", "postmortem",
     "Write a short postmortem summary: at 02:14 UTC {ip} (payments-db primary) ran out of disk; on-call "
     "{p1_full} failed over to the replica at 02:31; {p2_full} ({p2_email}) confirmed no data loss. Customer "
     "impact: {c1} saw 17 minutes of failed checkouts.",
     [("ip", "IP_ADDRESS"), ("p1_full", "NAME"), ("p2_full", "NAME"), ("p2_email", "EMAIL"), ("c1", "COMPANY")],
     ["p1_full", "17"]),
    ("engineering", "code_review",
     "Rewrite this code review comment to be kinder but still clear, addressed to {p1_first}: \"{p1_full}, "
     "this PR hard-codes the {codename} API key ({key}) and skips tests again. Not mergeable.\"",
     [("p1_full", "NAME"), ("codename", "CODENAME"), ("key", "SECRET")],
     ["p1_first"]),
    ("engineering", "log_analysis",
     "What pattern do you see in these auth logs?\n"
     "09:01 login_failed user={p1_email} ip={ip}\n09:01 login_failed user={p2_email} ip={ip}\n"
     "09:02 login_failed user={p3_email} ip={ip}\n09:02 login_ok user={p1_email} ip=10.0.0.5",
     [("p1_email", "EMAIL"), ("p2_email", "EMAIL"), ("p3_email", "EMAIL"), ("ip", "IP_ADDRESS")],
     ["ip"]),
    ("engineering", "deploy_question",
     "Our deploy script does `curl -H \"x-api-key: {key}\" https://api.anthropic.com/v1/models`. Rewrite it to "
     "read the key from an environment variable instead, and tell {p1_first} (who wrote it) why.",
     [("key", "SECRET")],
     ["p1_first"]),
    # ------------------------------------------------------------ meeting notes
    ("meetings", "action_items",
     "Extract action items with owners from these notes:\n{p1_full}: will send the {codename} timeline by "
     "Friday. {p2_full}: to get legal sign-off from {c2}. {p3_full}: fix the staging outage on {ip}. Budget "
     "stays at {amt2}.",
     [("p1_full", "NAME"), ("codename", "CODENAME"), ("p2_full", "NAME"), ("c2", "COMPANY"), ("p3_full", "NAME"),
      ("ip", "IP_ADDRESS"), ("amt2", "MONEY")],
     ["p1_full", "p2_full", "p3_full"]),
    ("meetings", "call_summary",
     "Summarise this call in 3 bullets: {p1_full} ({c1}) said their rollout of {codename} slipped to Q1 "
     "because {p2_full} left; they still want the {amt2} expansion; follow up at {p1_email}.",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("codename", "CODENAME"), ("p2_full", "NAME"), ("amt2", "MONEY"),
      ("p1_email", "EMAIL")],
     ["codename", "p1_full"]),
    ("meetings", "followup_email",
     "Write a follow-up email to {p1_full} and {p2_full} after today's steering meeting on {codename}: "
     "decisions were to keep {c1} as vendor and move go-live to 3 March. Ask {p2_first} to confirm headcount.",
     [("p1_full", "NAME"), ("p2_full", "NAME"), ("codename", "CODENAME"), ("c1", "COMPANY")],
     ["p2_first", "c1"]),
    ("meetings", "agenda",
     "Create a 45-minute agenda for Monday's review of {codename} with {p1_full} (product), {p2_full} "
     "(engineering) and {p3_full} (finance). Include 10 minutes on the {amt1} overspend.",
     [("codename", "CODENAME"), ("p1_full", "NAME"), ("p2_full", "NAME"), ("p3_full", "NAME"), ("amt1", "MONEY")],
     ["p1_full", "p3_full"]),
    ("meetings", "decision_log",
     "Write a decision-log entry: on 12 September {p1_full} decided to migrate {codename} from {c2}'s cloud to "
     "our own servers ({ip}) because of cost; {p2_full} disagreed citing staffing.",
     [("p1_full", "NAME"), ("codename", "CODENAME"), ("c2", "COMPANY"), ("ip", "IP_ADDRESS"), ("p2_full", "NAME")],
     ["p1_full", "p2_full"]),
    # ------------------------------------------------------------ data
    ("data", "top_customer",
     "Which customer spent the most and how much more than the lowest?\nname,email,spend\n"
     "{p1_full},{p1_email},{li1}\n{p2_full},{p2_email},{li2}\n{p3_full},{p3_email},{li3}",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("p2_full", "NAME"), ("p2_email", "EMAIL"), ("p3_full", "NAME"),
      ("p3_email", "EMAIL"), ("li1", "MONEY"), ("li2", "MONEY"), ("li3", "MONEY")],
     ["p1_full", lambda b: _money(b["li1"] - b["li3"])]),
    ("data", "dedupe",
     "These CRM rows may be duplicates. Which ones, and which record should we keep?\n"
     "1 | {p1_full} | {p1_email} | {c1}\n2 | {p1_first} {p1_last_initial}. | {p1_email} | {c1}\n"
     "3 | {p2_full} | {p2_email} | {c2}",
     [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("c1", "COMPANY"), ("p2_full", "NAME"), ("p2_email", "EMAIL"),
      ("c2", "COMPANY")],
     ["p1_email"]),
    ("data", "count_by_company",
     "Count contacts per company:\n{p1_full} - {c1}\n{p2_full} - {c2}\n{p3_full} - {c1}\n"
     "{p1_first} Jr. - {c2}\n",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("p2_full", "NAME"), ("c2", "COMPANY"), ("p3_full", "NAME")],
     ["c1", "c2"]),
    ("data", "overdue_list",
     "From this table, list overdue invoices and the total overdue amount:\n"
     "invoice | customer | amount | status\n{inv} | {c1} | {li1} | overdue\nINV-00017 | {c2} | {li2} | paid\n"
     "INV-00018 | {c2} | {li3} | overdue",
     [("inv", "ACCOUNT_ID"), ("c1", "COMPANY"), ("li1", "MONEY"), ("c2", "COMPANY"), ("li2", "MONEY"),
      ("li3", "MONEY")],
     ["inv", lambda b: _money(b["li1"] + b["li3"])]),
    ("data", "email_domains",
     "Group these emails by company domain and say which company has most contacts: {p1_email}, "
     "{p3_email}, {p2_email}, {p2_first_lower}@{c2_domain}.",
     [("p1_email", "EMAIL"), ("p3_email", "EMAIL"), ("p2_email", "EMAIL")],
     []),
    # ------------------------------------------------------------ rewrite / translate
    ("rewrite", "translate_hindi",
     "Translate into Hindi: \"Dear {p1_full}, your order #{order} from {c1} will be delivered to {addr} "
     "tomorrow. For questions call {p2_full} on {p2_phone}.\"",
     [("p1_full", "NAME"), ("order", "ACCOUNT_ID"), ("c1", "COMPANY"), ("addr", "ADDRESS"), ("p2_full", "NAME"),
      ("p2_phone", "PHONE")],
     ["order", "p2_phone"]),
    ("rewrite", "more_polite",
     "Make this more polite: \"{p1_first}, you still haven't sent the {codename} numbers. {p2_full} is "
     "waiting and {c1} will escalate. Send them to {p2_email} today.\"",
     [("p1_first", "NAME"), ("codename", "CODENAME"), ("p2_full", "NAME"), ("c1", "COMPANY"), ("p2_email", "EMAIL")],
     ["p1_first", "p2_email"]),
    ("rewrite", "shorten",
     "Shorten to two sentences: \"Hi team, as discussed with {p1_full} from {c1} on Tuesday, we are pausing "
     "{codename} until the security review by {p2_full} is complete, which should happen by the end of the "
     "month, after which we will restart and revisit the {amt2} budget.\"",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("codename", "CODENAME"), ("p2_full", "NAME"), ("amt2", "MONEY")],
     ["codename"]),
    ("rewrite", "bullets",
     "Convert to bullet points: {p1_full} will own onboarding for {c1}; {p2_full} handles billing questions at "
     "{p2_email}; {p3_full} is the escalation contact on {p1_phone}.",
     [("p1_full", "NAME"), ("c1", "COMPANY"), ("p2_full", "NAME"), ("p2_email", "EMAIL"), ("p3_full", "NAME"),
      ("p1_phone", "PHONE")],
     ["p1_full", "p2_email", "p3_full"]),
    ("rewrite", "formal_letter",
     "Turn this into a formal letter from {c1} to {p1_full}, {addr}: we're sorry we lost your parcel "
     "#{order}; we're refunding {amt1} and adding a {amt3} credit.",
     [("c1", "COMPANY"), ("p1_full", "NAME"), ("addr", "ADDRESS"), ("order", "ACCOUNT_ID"), ("amt1", "MONEY"),
      ("amt3", "MONEY")],
     ["p1_full", "order", "amt1"]),
]


def _prepare(b: dict, i: int) -> dict:
    """Add formatted money and derived slots to a bundle."""
    b = dict(b)
    b["amt1_val"] = 480.0 + i * 23.5
    b["amt2_val"] = 24000.0 + i * 1250
    b["cap_val"] = 2600.0 if i % 2 == 0 else 2200.0
    b["amt1"] = _money(b["amt1_val"])
    b["amt2"] = _money(b["amt2_val"])
    b["amt3"] = _money(25 + i)
    b["cap"] = _money(b["cap_val"])
    for k in ("li1", "li2", "li3"):
        b[k + "_val"] = b[k]
    b["li1"], b["li2"], b["li3"] = _money(b["li1_val"]), _money(b["li2_val"]), _money(b["li3_val"])
    b["codename_slug"] = b["codename"].split()[-1].lower()
    b["p1_last_initial"] = b["p1_full"].split()[-1][0]
    b["p2_first_lower"] = b["p2_first"].lower()
    b["c2_domain"] = b["p2_email"].split("@")[1]
    return b


# ---------------------------------------------------------------- heavy workloads
# Long, redundant inputs of the kind tonst's savings features were built
# for: an Outlook-style thread that re-quotes every earlier message (plus
# a signature and legal disclaimer on each), a log dump full of repeated
# heartbeat lines, a padded meeting transcript, and retrieval (RAG) with
# mirrored help-centre pages. They measure savings and privacy together.
# Each builder returns (prompt_or_rag, sensitive_slots, expect_spec).

DISCLAIMER = ("CONFIDENTIALITY NOTICE: This e-mail and any attachments are confidential and may be legally "
              "privileged. If you are not the intended recipient, please delete it and notify the sender.")


def _sig(full, company, phone):
    return f"Best regards,\n{full}\n{company}\nPhone: {phone}\n{DISCLAIMER}"


def heavy_email_thread(b):
    m1 = (f"From: {b['p1_full']} <{b['p1_email']}>\nSent: Monday 09:12\nTo: {b['p2_full']} <{b['p2_email']}>\n"
          f"Subject: Invoice {b['inv']} - duplicate charge\n\nHi {b['p2_first']},\n\n"
          f"We were charged twice for invoice {b['inv']} ({b['amt1']} each) on account {b['acct']}. "
          f"Can you reverse one of the charges?\n\n" + _sig(b['p1_full'], b['c1'], b['p1_phone']))
    m2 = (f"From: {b['p2_full']} <{b['p2_email']}>\nSent: Monday 11:40\nTo: {b['p1_full']} <{b['p1_email']}>\n"
          f"Subject: RE: Invoice {b['inv']} - duplicate charge\n\nHi {b['p1_first']},\n\n"
          f"Thanks for flagging. I've asked our billing team ({b['p3_full']}) to check. Could you send the bank "
          f"statement line?\n\n" + _sig(b['p2_full'], b['c2'], b['p2_phone']))
    m3 = (f"From: {b['p1_full']} <{b['p1_email']}>\nSent: Tuesday 08:05\nTo: {b['p2_full']} <{b['p2_email']}>\n"
          f"Subject: RE: RE: Invoice {b['inv']} - duplicate charge\n\nHi {b['p2_first']},\n\n"
          f"Attached. Both debits are dated 3 September. Please confirm by Friday; our month-end close depends "
          f"on it.\n\n" + _sig(b['p1_full'], b['c1'], b['p1_phone']))
    thread = "\n\n".join([m3, m2, m1, m2, m1, m1])  # each reply re-quotes everything below it
    prompt = ("Summarise this email thread in three bullets, then draft my reply as "
              f"{b['p2_full']} confirming the refund of one {b['amt1']} charge by Thursday.\n\n" + thread)
    sensitive = [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("p2_full", "NAME"), ("p2_email", "EMAIL"),
                 ("p3_full", "NAME"), ("p1_phone", "PHONE"), ("p2_phone", "PHONE"), ("c1", "COMPANY"),
                 ("c2", "COMPANY"), ("inv", "ACCOUNT_ID"), ("acct", "ACCOUNT_ID"), ("amt1", "MONEY")]
    return prompt, sensitive, ["p1_first", "inv"]


def heavy_log_dump(b):
    lines = []
    for n in range(40):
        lines.append(f"2026-09-20T02:{n:02d}:00Z INFO  healthcheck ok service=payments host={b['ip']}")
        if n in (7, 19, 31):
            lines.append(f"2026-09-20T02:{n:02d}:14Z ERROR charge_failed user={b['p1_email']} "
                         f"card_bin=411111 reason=issuer_timeout host={b['ip']}")
        if n in (12, 33):
            lines.append(f"2026-09-20T02:{n:02d}:41Z ERROR charge_failed user={b['p2_email']} "
                         f"card_bin=550000 reason=issuer_timeout host={b['ip']}")
        lines.append(f"2026-09-20T02:{n:02d}:30Z INFO  healthcheck ok service=payments host={b['ip']}")
    prompt = ("Here are payments logs from last night. What is causing the errors, which users were affected, "
              "and what should on-call do?\n\n" + "\n".join(lines))
    sensitive = [("p1_email", "EMAIL"), ("p2_email", "EMAIL"), ("ip", "IP_ADDRESS")]
    return prompt, sensitive, ["p1_email", "p2_email", "issuer_timeout"]


def heavy_transcript(b):
    pad = "\n\n\n"
    turns = [
        f"{b['p1_full']}:    Okay,   let's start.   Main topic is the {b['codename']} launch.",
        "[crosstalk]",
        f"{b['p2_full']}:    Engineering is two weeks behind because {b['c2']} delivered the API late.",
        "[crosstalk]",
        f"{b['p3_full']}:    Finance can hold the {b['amt2']} budget if we slip no more than a month.",
        "[inaudible]",
        f"{b['p1_full']}:    Then {b['p2_first']}, please send a revised plan by Wednesday.",
        "[crosstalk]",
        f"{b['p2_full']}:    Will do. I'll also ask {b['c2']} for a written delivery commitment.",
        "[inaudible]",
        f"{b['p3_full']}:    I'll update the forecast once the plan is in.",
    ]
    body = pad.join(turns * 3)  # transcripts often repeat after reconnects
    prompt = "Extract the decisions and action items (with owners and dates) from this transcript:\n\n" + body
    sensitive = [("p1_full", "NAME"), ("p2_full", "NAME"), ("p3_full", "NAME"), ("codename", "CODENAME"),
                 ("c2", "COMPANY"), ("amt2", "MONEY")]
    return prompt, sensitive, ["p2_full", "Wednesday"]


def heavy_rag_support(b):
    refund = ("Refunds: damaged items can be replaced or refunded within 30 days. Replacements ship within 3 "
              "business days. Refunds reach the original card in 5-7 business days.")
    chunks = [
        refund,
        "Shipping: standard delivery takes 3-5 business days; express takes 1-2.",
        refund.replace("Refunds:", "Refund policy (mirror):"),
        "Warranty: screens are covered for 12 months against manufacturing defects.",
        refund,
        "Accounts: you can change your delivery address until the order ships.",
        "Warranty: screens are covered for 12 months against manufacturing defects.",
        "Gift cards cannot be refunded or exchanged for cash.",
    ]
    question = (f"Customer {b['p1_full']} ({b['p1_email']}, {b['p1_phone']}) says order #{b['order']} arrived with "
                f"a cracked screen. What are their options and how fast? Draft a short reply to {b['p1_first']}.")
    rag = {"question": question, "chunks": chunks, "top_k": 4}
    sensitive = [("p1_full", "NAME"), ("p1_email", "EMAIL"), ("p1_phone", "PHONE"), ("order", "ACCOUNT_ID")]
    return rag, sensitive, ["p1_first", "3 business days"]


def heavy_rag_contract(b):
    c = [
        f"Clause 4.1 Fees. {b['c1']} shall pay {b['c2']} {b['amt2']} per year, invoiced quarterly.",
        f"Clause 4.2 Late payment. Overdue amounts accrue interest at 1% per month.",
        f"Clause 9.1 Termination. Either party may terminate with 90 days' written notice to the other's "
        f"signatory ({b['p1_full']} for {b['c1']}, {b['p2_full']} for {b['c2']}).",
        f"Clause 4.1 Fees. {b['c1']} shall pay {b['c2']} {b['amt2']} per year, invoiced quarterly.",
        f"Clause 12 Governing law. This agreement is governed by the laws of Singapore.",
        f"Clause 9.1 Termination. Either party may terminate with 90 days' written notice to the other's "
        f"signatory ({b['p1_full']} for {b['c1']}, {b['p2_full']} for {b['c2']}).",
        "Clause 7 Confidentiality. Obligations survive three years after termination.",
    ]
    question = (f"If {b['c1']} wants to terminate, who must they notify, how much notice is needed, and what "
                f"does the contract say about fees still owed?")
    rag = {"question": question, "chunks": c, "top_k": 4}
    sensitive = [("c1", "COMPANY"), ("c2", "COMPANY"), ("amt2", "MONEY"), ("p1_full", "NAME"), ("p2_full", "NAME")]
    return rag, sensitive, ["p2_full", "90 days"]


HEAVY = [
    ("email_thread", heavy_email_thread),
    ("log_dump", heavy_log_dump),
    ("transcript", heavy_transcript),
    ("rag_support", heavy_rag_support),
    ("rag_contract", heavy_rag_contract),
]


def rag_prompt(rag: dict) -> str:
    """What a caller without tonst would send: every retrieved chunk, then the question."""
    blocks = [f"[Context {n}]\n{t.strip()}" for n, t in enumerate(rag["chunks"], 1)]
    return "\n\n".join(blocks + [f"Question: {rag['question'].strip()}"])


# ---------------------------------------------------------------- hold-out cases
# Written AFTER the Phase 3 detector/name rules, with people, companies,
# phone/address formats and phrasings that were NOT used while tuning them
# (and not looked at through the detectors before the benchmark run). If
# these score clearly worse than the main set, the rules are overfitted.

HOLDOUT = [
    ("followup_colleague",
     "Loop in my colleague Chloé Dubois (chloe.dubois@bluepinemedia.fr) on the Ardent Robotics renewal and write a "
     "short note from me, Aarav Mehta, explaining that Kwame Mensah from their side asked for a 3-year term.",
     [("Chloé Dubois", "NAME"), ("chloe.dubois@bluepinemedia.fr", "EMAIL"), ("Ardent Robotics", "COMPANY"),
      ("Aarav Mehta", "NAME"), ("Kwame Mensah", "NAME")],
     ["Chloé", "Kwame Mensah"]),
    ("us_phone_formats",
     "Two voicemails to return: Isabella Rossi at (415) 555-0144 about her Helio Pharma order, and Liam O'Connor on "
     "+1.212.555.0199 who wants to cancel. Write a call-back plan with what to say to each.",
     [("Isabella Rossi", "NAME"), ("(415) 555-0144", "PHONE"), ("Helio Pharma", "COMPANY"),
      ("Liam O'Connor", "NAME"), ("+1.212.555.0199", "PHONE")],
     ["Isabella", "Liam"]),
    ("brooklyn_delivery",
     "Delivery exception for Yuki Sato: the parcel for Apartment 3C, 55 Water St, Brooklyn, NY 11201 was left at the "
     "wrong door. Draft an apology and ask whether we should re-send to the same address.",
     [("Yuki Sato", "NAME"), ("Apartment 3C, 55 Water St, Brooklyn, NY 11201", "ADDRESS")],
     ["Yuki"]),
    ("hyderabad_kyc",
     "Summarise this KYC note: Fatima Zahra, Plot 12, Jubilee Hills, Hyderabad 500033, director of Solenne "
     "Cosmetics Pvt Ltd; verified by Noah Fischer on 3 Oct; mobile 090000 12345.",
     [("Fatima Zahra", "NAME"), ("Plot 12, Jubilee Hills, Hyderabad 500033", "ADDRESS"),
      ("Solenne Cosmetics", "COMPANY"), ("Noah Fischer", "NAME"), ("090000 12345", "PHONE")],
     ["Noah Fischer"]),
    ("possessive_review",
     "Aarav's quarterly review: Aarav Mehta closed the Tarmac Freight deal, but Chloé felt his handover notes were "
     "thin. Write two balanced paragraphs for his manager, Kwame.",
     [("Aarav Mehta", "NAME"), ("Tarmac Freight", "COMPANY")],
     ["Aarav", "Tarmac Freight"]),
    ("lowercase_emails",
     "Which of these belong to the same organisation, and who is the odd one out? isabella.rossi@heliopharma.com, "
     "noah.fischer@heliopharma.com, y.sato@quarryvale.co.uk",
     [("isabella.rossi@heliopharma.com", "EMAIL"), ("noah.fischer@heliopharma.com", "EMAIL"),
      ("y.sato@quarryvale.co.uk", "EMAIL")],
     ["y.sato@quarryvale.co.uk"]),
    ("vendor_chain",
     "Quarry & Vale Partners subcontracted the audit to Bluepine Media without telling us. Draft a firm email from "
     "Fatima Zahra to their partner Liam O'Connor asking for the subcontract terms by Friday.",
     [("Quarry & Vale Partners", "COMPANY"), ("Bluepine Media", "COMPANY"), ("Fatima Zahra", "NAME"),
      ("Liam O'Connor", "NAME")],
     ["Liam", "Friday"]),
    ("gh_token_leak",
     "I accidentally pushed this to a public repo: `GITHUB_TOKEN=ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789`. What "
     "should Noah Fischer (our security lead) do right now, in order?",
     [("ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "SECRET"), ("Noah Fischer", "NAME")],
     ["Noah"]),
    ("meeting_nicknames",
     "Notes: Kwame Mensah (Ardent Robotics) and Isabella Rossi agreed the pilot starts 1 Nov. Kwame owns the "
     "hardware list; Isabella owns legal review. Turn this into a follow-up email to both.",
     [("Kwame Mensah", "NAME"), ("Ardent Robotics", "COMPANY"), ("Isabella Rossi", "NAME")],
     ["Kwame", "Isabella", "1 Nov"]),
    ("paris_invoice_address",
     "Update the billing address for Solenne Cosmetics to 17 Rue de Rivoli, 75001 Paris and write a one-line "
     "confirmation to Chloé Dubois.",
     [("Solenne Cosmetics", "COMPANY"), ("17 Rue de Rivoli, 75001 Paris", "ADDRESS"), ("Chloé Dubois", "NAME")],
     ["Chloé"]),
]


def build_cases() -> list[dict]:
    cases = []
    for t_index, (category, tid, template, sensitive_slots, expect_spec) in enumerate(TEMPLATES):
        for k in range(2):
            i = t_index * 2 + k
            raw = bundle(i)
            b = _prepare(raw, i)
            numeric = {"li1": raw["li1"], "li2": raw["li2"], "li3": raw["li3"], "total": raw["total"]}
            fmt = {key: val for key, val in b.items()}
            prompt = template.format(**fmt)
            sensitive = []
            seen = set()
            for slot, typ in sensitive_slots:
                val = str(b[slot])
                if val in prompt and val not in seen:
                    sensitive.append((val, typ))
                    seen.add(val)
            calc = {**b, **numeric}
            expect = []
            for e in expect_spec:
                expect.append(e(calc) if callable(e) else (str(b[e]) if e in b else e))
            cases.append({
                "id": f"{category}.{tid}.{k + 1}",
                "category": category,
                "prompt": prompt,
                "sensitive": sensitive,
                "expect": expect,
            })
    for h_index, (hid, builder) in enumerate(HEAVY):
        for k in range(2):
            i = 100 + h_index * 2 + k
            b = _prepare(bundle(i), i)
            content, sensitive_slots, expect_spec = builder(b)
            case = {"id": f"heavy.{hid}.{k + 1}", "category": "heavy"}
            if isinstance(content, dict):
                case["rag"] = content
                case["prompt"] = rag_prompt(content)
            else:
                case["prompt"] = content
            seen = set()
            case["sensitive"] = []
            for slot, typ in sensitive_slots:
                val = str(b[slot])
                if val in case["prompt"] and val not in seen:
                    case["sensitive"].append((val, typ))
                    seen.add(val)
            case["expect"] = [str(b[e]) if e in b else e for e in expect_spec]
            cases.append(case)
    for hid, prompt, sensitive, expect in HOLDOUT:
        cases.append({"id": f"holdout.{hid}", "category": "holdout", "prompt": prompt,
                      "sensitive": [(v, t) for v, t in sensitive if v in prompt], "expect": expect})
    return cases


CASES = build_cases()

if __name__ == "__main__":
    import collections
    print(len(CASES), "cases")
    print(collections.Counter(c["category"] for c in CASES))
    print(CASES[0]["prompt"])
