"""
benchmark_free_features.py
--------------------------
Offline, deterministic benchmark for the three free features whose value
can be measured without a paid API call:

  1. Tool filtering (tool_optimizer.select_tools / ToolSession):
     token savings AND recall -- how often a tool the task actually
     needs got filtered out. Recall is the number that matters; saving
     tokens by dropping the tool the model needed is a failure, not a
     saving.
  2. RAG chunk optimization (rag.optimize_chunks): token savings and
     whether the answer-bearing chunk survived.
  3. Rolling vs. stateless history compaction: local-model calls and
     how often the prompt prefix stays stable between turns (a proxy
     for provider prefix-cache hits).
  4. Cache-aware compaction: a cost simulation of folding at the plain
     token threshold vs. compaction_cache_aware, under Anthropic prompt
     caching, for conversations of 8-100 turns (calibrated on the live
     24-turn run -- see bench_cache_aware_compaction).

Honest scope: the tool catalog, tasks, help-center corpus and chat are
hand-built to look like real workloads (GitHub/Slack/Jira/filesystem/
database-style tools; a help-center knowledge base with mirrored pages),
NOT captured from real traffic. The numbers show how the mechanisms
behave, including where they fail (the paraphrased-task group is there
on purpose). Token counts use tonst's chars/4 estimate.

Run:  python benchmark_free_features.py   (no network, no Ollama needed)
"""

from __future__ import annotations
import json

from tonst.tool_optimizer import select_tools, ToolSession, estimate_tool_tokens
from tonst.rag import optimize_chunks
from tonst.compactor import compact_history, compact_history_rolling, RollingSummary
from tonst.trim import flatten_messages, estimate_tokens


# ---------------------------------------------------------------------
# 1. Tools
# ---------------------------------------------------------------------

def _tool(name, desc, **params):
    props = {}
    for p, d in params.items():
        props[p] = {"type": "string", "description": d}
    return {
        "name": name,
        "description": desc,
        "input_schema": {"type": "object", "properties": props, "required": list(params)[:1]},
    }


TOOLS = [
    # GitHub-style
    _tool("github_search_issues", "Search issues and pull requests in a GitHub repository by keyword, label or state.",
          repo="Repository in owner/name form", query="Search keywords", state="open, closed or all"),
    _tool("github_get_issue", "Get the full details, body and comments of a single GitHub issue.",
          repo="Repository in owner/name form", number="Issue number"),
    _tool("github_create_issue", "Create a new issue in a GitHub repository with a title, body and labels.",
          repo="Repository in owner/name form", title="Issue title", body="Markdown body", labels="Comma-separated labels"),
    _tool("github_comment_issue", "Add a comment to an existing GitHub issue or pull request.",
          repo="Repository in owner/name form", number="Issue or PR number", body="Comment text"),
    _tool("github_create_pull_request", "Open a pull request from a branch into the base branch.",
          repo="Repository in owner/name form", head="Source branch", base="Target branch", title="PR title"),
    _tool("github_list_pull_requests", "List pull requests in a repository, optionally filtered by state or author.",
          repo="Repository in owner/name form", state="open, closed or all", author="GitHub username"),
    _tool("github_merge_pull_request", "Merge an approved pull request using merge, squash or rebase.",
          repo="Repository in owner/name form", number="PR number", method="merge, squash or rebase"),
    _tool("github_get_file", "Read the contents of a file at a given path and git ref in a repository.",
          repo="Repository in owner/name form", path="File path", ref="Branch, tag or commit SHA"),
    _tool("github_list_commits", "List recent commits on a branch with author, date and message.",
          repo="Repository in owner/name form", branch="Branch name", since="ISO date"),
    _tool("github_get_workflow_runs", "Get recent CI workflow runs and their pass/fail status for a repository.",
          repo="Repository in owner/name form", workflow="Workflow file name", branch="Branch name"),
    # Slack-style
    _tool("slack_send_message", "Post a message to a Slack channel or direct message.",
          channel="Channel name or ID", text="Message text"),
    _tool("slack_search_messages", "Search Slack message history across channels by keyword.",
          query="Search keywords", channel="Optional channel to restrict to"),
    _tool("slack_list_channels", "List Slack channels the bot can see, with member counts.",
          prefix="Optional name prefix filter"),
    _tool("slack_get_thread", "Get all replies in a Slack thread.",
          channel="Channel ID", thread_ts="Timestamp of the parent message"),
    # Jira-style
    _tool("jira_create_ticket", "Create a Jira ticket in a project with summary, description, type and priority.",
          project="Project key", summary="Ticket summary", issue_type="Bug, Task or Story", priority="Priority level"),
    _tool("jira_search_tickets", "Search Jira tickets with JQL or keywords.",
          jql="JQL query or keywords"),
    _tool("jira_update_ticket", "Update fields on a Jira ticket such as status, assignee or priority.",
          key="Ticket key like PROJ-123", status="New status", assignee="Assignee username"),
    _tool("jira_add_comment", "Add a comment to a Jira ticket.",
          key="Ticket key like PROJ-123", body="Comment text"),
    # Filesystem / code
    _tool("fs_read_file", "Read a text file from the local workspace.", path="File path relative to the workspace"),
    _tool("fs_write_file", "Write or overwrite a text file in the local workspace.", path="File path", content="New file content"),
    _tool("fs_list_directory", "List files and folders in a workspace directory.", path="Directory path"),
    _tool("fs_search_code", "Search source code in the workspace for a string or regex.", pattern="Search pattern", glob="Optional file glob"),
    _tool("run_tests", "Run the project's test suite and return failures.", target="Optional test file or test name"),
    _tool("run_shell", "Run a shell command in the workspace and return its output.", command="Command to run"),
    # Data
    _tool("sql_query", "Run a read-only SQL query against the analytics Postgres database.", sql="SQL SELECT statement"),
    _tool("sql_list_tables", "List tables and their columns in the analytics database.", schema="Schema name"),
    _tool("metrics_get_timeseries", "Fetch a monitoring metric time series such as latency, error rate or CPU.",
          metric="Metric name", service="Service name", window="Time window like 1h or 7d"),
    _tool("logs_search", "Search application logs for a service by text and time range.",
          service="Service name", query="Text to search for", window="Time window"),
    # Docs / calendar / email
    _tool("docs_search", "Search the internal documentation wiki for pages matching keywords.", query="Search keywords"),
    _tool("docs_get_page", "Get the full content of an internal documentation page.", page_id="Page ID"),
    _tool("calendar_create_event", "Create a calendar event and invite attendees.",
          title="Event title", start="Start time", attendees="Comma-separated emails"),
    _tool("calendar_find_free_time", "Find free time slots shared by a set of people.",
          attendees="Comma-separated emails", duration="Meeting length in minutes"),
    _tool("email_send", "Send an email to one or more recipients.", to="Recipient emails", subject="Subject line", body="Email body"),
    _tool("email_search", "Search the mailbox for emails by sender, subject or keyword.", query="Search keywords"),
    _tool("web_fetch", "Fetch a public web page and return its text.", url="Page URL"),
    _tool("translate_text", "Translate text into a target language.", text="Text to translate", target_language="Language code"),
]

# (task, tools it genuinely needs, group). "direct" tasks share vocabulary
# with the tool descriptions; "paraphrased" ones deliberately don't, to
# measure the known BM25 weakness honestly.
TASKS = [
    ("Find open GitHub issues about the login timeout in acme/web", ["github_search_issues"], "direct"),
    ("Read issue 482 in acme/web and summarize the comments", ["github_get_issue"], "direct"),
    ("Create a GitHub issue in acme/api titled 'Rate limiter drops requests' with the bug label", ["github_create_issue"], "direct"),
    ("Open a pull request from feature/cache into main in acme/api titled 'Add response cache'",
     ["github_create_pull_request"], "direct"),
    ("Merge pull request 77 in acme/api using squash", ["github_merge_pull_request"], "direct"),
    ("Did the CI workflow runs pass on the main branch of acme/web today?", ["github_get_workflow_runs"], "direct"),
    ("Post a message in the #deploys Slack channel saying the release is out", ["slack_send_message"], "direct"),
    ("Search Slack messages for discussion of the billing outage", ["slack_search_messages"], "direct"),
    ("Create a Jira bug ticket in project PAY for the refund rounding error, high priority", ["jira_create_ticket"], "direct"),
    ("Update Jira ticket PAY-311 status to Done and assign it to priya", ["jira_update_ticket"], "direct"),
    ("Read the file src/config.py in the workspace", ["fs_read_file"], "direct"),
    ("Search the source code for calls to retry_with_backoff", ["fs_search_code"], "direct"),
    ("Run the test suite and tell me what fails", ["run_tests"], "direct"),
    ("Run a SQL query counting signups per day last week in the analytics database", ["sql_query"], "direct"),
    ("Show the p95 latency metric for the checkout service over the last 24h", ["metrics_get_timeseries"], "direct"),
    ("Search the logs of the payments service for 'timeout' in the last hour", ["logs_search"], "direct"),
    ("Search the internal documentation for the on-call runbook", ["docs_search"], "direct"),
    ("Find free time for a 30 minute meeting with ana@acme.com and raj@acme.com", ["calendar_find_free_time"], "direct"),
    ("Send an email to finance@acme.com with the subject 'Q3 invoice' saying the invoice is attached and due Friday",
     ["email_send"], "direct"),
    ("Translate this release note into Spanish: 'Version 2.4 adds dark mode and fixes the login timeout.'",
     ["translate_text"], "direct"),
    # multi-tool
    ("Search GitHub issues about flaky tests in acme/web and create a Jira ticket for the worst one",
     ["github_search_issues", "jira_create_ticket"], "direct"),
    ("Check the error rate metric for the api service and search its logs for exceptions",
     ["metrics_get_timeseries", "logs_search"], "direct"),
    ("Read the file docs/CHANGELOG.md and post a summary message to the #releases Slack channel",
     ["fs_read_file", "slack_send_message"], "direct"),
    ("List the tables in the analytics database and then run a SQL query counting the rows in the orders table",
     ["sql_list_tables", "sql_query"], "direct"),
    # paraphrased: no shared vocabulary with the right tool on purpose
    ("Ping the team in #eng that v2.4 shipped", ["slack_send_message"], "paraphrased"),
    ("Is the build green on acme/web main?", ["github_get_workflow_runs"], "paraphrased"),
    ("Book 30 minutes with ana@acme.com tomorrow at 3pm", ["calendar_create_event"], "paraphrased"),
    ("How many people bought something yesterday?", ["sql_query"], "paraphrased"),
    ("Put this in French: 'Your order has shipped and will arrive on Monday.'", ["translate_text"], "paraphrased"),
    ("What broke after last night's deploy?", ["logs_search", "metrics_get_timeseries"], "paraphrased"),
]
# Every task above now contains everything needed to act (the first live run
# found several that didn't -- an email with no body, "translate this" with no
# text -- and Claude correctly asked for the missing content, which was scored
# as a miss). A few tasks have a second, genuinely reasonable FIRST step; the
# live test counts these as correct, and says so in its output. Used only by
# live_test_free_features.py -- the offline benchmark measures whether the
# needed tools were KEPT, not which one the model calls first.
ACCEPTABLE_FIRST_STEPS = {
    "Run a SQL query counting signups per day last week in the analytics database": ["sql_list_tables"],
    "How many people bought something yesterday?": ["sql_list_tables"],
    "Book 30 minutes with ana@acme.com tomorrow at 3pm": ["calendar_find_free_time"],
    "What broke after last night's deploy?": ["github_list_commits", "github_get_workflow_runs"],
}


def bench_tools(top_k: int) -> dict:
    full = estimate_tool_tokens(TOOLS)
    rows = {"direct": [], "paraphrased": []}
    for task, needed, group in TASKS:
        sel = select_tools(TOOLS, task, top_k=top_k)
        kept = set(sel.selected_names)
        rows[group].append({
            "hit": all(n in kept for n in needed),
            "tokens_after": sel.tokens_after,
            "fell_back": sel.fell_back,
        })
    out = {"top_k": top_k, "tool_count": len(TOOLS), "tool_tokens_full": full}
    for group, r in rows.items():
        n = len(r)
        out[group] = {
            "tasks": n,
            "recall_percent": round(100 * sum(x["hit"] for x in r) / n, 1),
            "avg_tokens_sent": round(sum(x["tokens_after"] for x in r) / n),
            "avg_token_saving_percent": round(100 * (1 - sum(x["tokens_after"] for x in r) / (n * full)), 1),
            "fell_back_count": sum(x["fell_back"] for x in r),
        }
    return out


def bench_tool_session() -> dict:
    # One multi-turn incident-response conversation.
    turns = [
        "The checkout service is slow; show the latency metric for checkout over the last hour",
        "thanks. anything odd there?",
        "Search the checkout service logs for timeout errors",
        "ok, looks like the database. what does that mean?",
        "Create a Jira bug ticket in project PAY for the checkout timeouts",
        "and post a message in the #incidents Slack channel linking the ticket",
        "great, thanks",
    ]
    session = ToolSession(TOOLS, top_k=5)
    per_turn, changed = [], 0
    for t in turns:
        sel = session.select(t)
        per_turn.append(sel.tokens_after)
        changed += sel.changed
    full = estimate_tool_tokens(TOOLS)
    return {
        "turns": len(turns),
        "tool_list_changed_on_turns": changed,          # includes turn 1
        "cache_stable_turns": len(turns) - changed,
        "tools_sent_final": len(session.active),
        "avg_tokens_sent": round(sum(per_turn) / len(per_turn)),
        "tool_tokens_full": full,
        "avg_token_saving_percent": round(100 * (1 - sum(per_turn) / (len(per_turn) * full)), 1),
    }


# ---------------------------------------------------------------------
# 2. RAG
# ---------------------------------------------------------------------

KB = {
    "refund-policy": "Refunds are available within 30 days of delivery for unused items in their original packaging. "
                     "Digital products and gift cards cannot be refunded. Refunds go back to the original payment method "
                     "within 5 to 10 business days after the returned item is inspected at our warehouse.",
    "start-return": "To start a return, open Orders, choose the item and click Request return. Print the prepaid label, "
                    "pack the item securely and drop it at any courier partner location within 7 days.",
    "shipping-times": "Standard shipping takes 3 to 5 business days within India and 7 to 12 business days internationally. "
                      "Express shipping takes 1 to 2 business days in metro cities.",
    "shipping-costs": "Shipping is free on orders above 999 rupees. Below that, standard shipping costs 79 rupees and "
                      "express shipping costs 149 rupees.",
    "account-password": "To reset your password, click Forgot password on the sign-in page and follow the link sent to "
                        "your registered email. The link expires after 30 minutes.",
    "payment-methods": "We accept UPI, credit and debit cards, net banking and cash on delivery for orders under 5,000 rupees.",
    "warranty": "Electronics carry a one-year manufacturer warranty. Warranty claims are handled by the brand's service "
                "centre; keep your invoice as proof of purchase.",
    "cancel-order": "You can cancel an order from the Orders page until it has been shipped. Once shipped, cancel by "
                    "refusing delivery or starting a return after it arrives.",
}

# What a generous top-10 retriever typically hands back: the right pages,
# the same pages again from a mirrored/older copy, and loosely related filler.
RAG_QUERIES = [
    ("How long do refunds take to reach my card?", "refund-policy",
     ["refund-policy", "refund-policy@mirror", "start-return", "start-return@v1", "cancel-order",
      "payment-methods", "shipping-times", "warranty", "refund-policy@mirror2", "shipping-costs"]),
    ("How much does express shipping cost?", "shipping-costs",
     ["shipping-costs", "shipping-times", "shipping-costs@mirror", "shipping-times@mirror", "payment-methods",
      "cancel-order", "start-return", "refund-policy", "warranty", "account-password"]),
    ("I forgot my password, how do I reset it?", "account-password",
     ["account-password", "account-password@mirror", "payment-methods", "cancel-order", "warranty",
      "shipping-times", "refund-policy", "start-return", "shipping-costs", "account-password@v1"]),
    ("Can I cancel my order after it ships?", "cancel-order",
     ["cancel-order", "start-return", "cancel-order@mirror", "refund-policy", "shipping-times",
      "start-return@v1", "payment-methods", "warranty", "shipping-costs", "account-password"]),
]


def _variant(key: str) -> str:
    base, _, tag = key.partition("@")
    text = KB[base]
    if tag.startswith("mirror"):
        return text.replace("  ", " ") + " "  # byte-different, same content (whitespace)
    if tag == "v1":
        # older revision: one sentence changed, rest identical -> near-duplicate
        sentences = text.split(". ")
        sentences[-1] = sentences[-1].replace("within", "in about")
        return ". ".join(sentences) + " (Updated 2025.)"
    return text


def bench_rag() -> dict:
    results = []
    for question, answer_key, retrieved in RAG_QUERIES:
        chunks = [{"id": k, "text": _variant(k)} for k in retrieved]
        answer_text = KB[answer_key]
        for label, kwargs in [("dedupe_only", {}), ("dedupe+top_k=4", {"top_k": 4}), ("dedupe+top_k=2", {"top_k": 2})]:
            sel = optimize_chunks(chunks, question, **kwargs)
            kept_ids = [c["id"].split("@")[0] for c in sel.chunks]
            results.append({
                "config": label,
                "saving_percent": round(100 * sel.tokens_saved / sel.tokens_before, 1),
                "answer_kept": answer_key in kept_ids,
                "chunks_sent": len(sel.chunks),
            })
    out = {}
    for label in ("dedupe_only", "dedupe+top_k=4", "dedupe+top_k=2"):
        r = [x for x in results if x["config"] == label]
        out[label] = {
            "queries": len(r),
            "avg_chunks_sent_of_10": round(sum(x["chunks_sent"] for x in r) / len(r), 1),
            "avg_token_saving_percent": round(sum(x["saving_percent"] for x in r) / len(r), 1),
            "answer_chunk_kept_percent": round(100 * sum(x["answer_kept"] for x in r) / len(r), 1),
        }
    return out


# ---------------------------------------------------------------------
# 3. Rolling vs. stateless compaction
# ---------------------------------------------------------------------

class _FakeCompactor:
    """Deterministic stand-in for the local model; counts calls."""
    def __init__(self):
        self.calls = 0

    def summarize(self, older_text):
        self.calls += 1
        return f"Summary covering {len(older_text)} chars of history."

    def summarize_incremental(self, previous, new_text, max_summary_chars=4000):
        self.calls += 1
        return (previous or "") + f"\n- folded {len(new_text)} chars"


def bench_compaction(turns: int = 40, keep_last_n: int = 6, threshold: int = 600) -> dict:
    """
    Raw tokens are the wrong yardstick here: rolling mode deliberately
    keeps more text verbatim (evicted turns wait for a batch fold), but
    in exchange the prompt only grows at the end, so most of it is a
    repeat of the previous request's prefix. With provider prefix
    caching, repeated prefix tokens are billed at a fraction of the
    normal price. cost_units models that: cached tokens x read_price +
    uncached tokens x 1.0, where "cached" = the longest common prefix
    with the previous turn's prompt. Shown at 10% (Anthropic's standard
    cache-read price; OpenAI's newer models go lower) and 50% (OpenAI's
    least generous discount). Ignores cache minimum lengths and TTLs.
    """
    import os

    msgs = [{"role": "system", "content": "You are a helpful support assistant."}]
    comps = {"stateless": _FakeCompactor(), "rolling": _FakeCompactor()}
    state = RollingSummary()
    prev = {k: None for k in comps}
    stable = {k: 0 for k in comps}
    sent = {k: 0 for k in comps}
    cached = {k: 0 for k in comps}
    for i in range(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                     "content": f"Turn {i}: " + ("details about the order and the delivery issue " * 6)})
        prompts = {
            "stateless": flatten_messages(
                compact_history(msgs, comps["stateless"], keep_last_n=keep_last_n, token_threshold=threshold).messages),
            "rolling": flatten_messages(
                compact_history_rolling(msgs, comps["rolling"], state, keep_last_n=keep_last_n,
                                        token_threshold=threshold).messages),
        }
        for name, text in prompts.items():
            if prev[name] is not None:
                common = os.path.commonprefix([prev[name], text])
                cached[name] += len(common) // 4
                if text.startswith(prev[name]):
                    stable[name] += 1
            prev[name] = text
            sent[name] += estimate_tokens(text)

    def cost(name, read_price):
        return round(cached[name] * read_price + (sent[name] - cached[name]))

    full = sum(estimate_tokens(flatten_messages(msgs[: 2 + k])) for k in range(turns))
    out = {"turns": turns, "keep_last_n": keep_last_n, "token_threshold": threshold,
           "no_compaction_total_tokens": full}
    for name in comps:
        out[name] = {
            "local_model_calls": comps[name].calls,
            "prefix_stable_turns": stable[name],
            "total_tokens_sent": sent[name],
            "tokens_reusable_from_cache": cached[name],
            "cost_units_at_10pct_cache_price": cost(name, 0.10),
            "cost_units_at_50pct_cache_price": cost(name, 0.50),
        }
    return out


def _sim_conversation_cost(turns: int, **rolling_kwargs) -> dict:
    """
    Cost of one scripted support chat (the live test's long conversation:
    ~500 tokens of tool output per reply) under Anthropic prompt caching,
    Sonnet 4.6 prices. none=True means no compaction; any other keyword
    arguments go to compact_history_rolling(). Cache model: the longest common message prefix with the
    previous request is read at 0.1x, the rest written at 1.25x. The
    summarizer is a fake that returns Haiku-sized summaries (~8% of what
    it reads, capped at ~2,000 chars) and is billed at Haiku prices;
    summaries apply one turn after they start (background mode). Token
    counts are tonst's estimate / 0.63 -- the estimate-to-real ratio the
    live run measured on this conversation.
    """
    from live_test_free_features import _conversation, _handbook
    from tonst.compactor import HistoryCompactor, run_fold_job

    est_over_real = 0.63

    def real(text):
        return estimate_tokens(text) / est_over_real

    class HaikuSized:
        def __init__(self):
            self.prev, self.inp, self.out = 0, 0.0, 0.0

        def __call__(self, prompt, model="", timeout=0.0):
            target = min(2000, int(self.prev + 0.08 * max(0, len(prompt) - self.prev - 1500)))
            body = ("replacement approved, express shipping. " * 60)[: max(40, target - 150)]
            s = f"Goal: resolve damaged lamp order #4471\nDecisions:\n- {body}\nKey facts:\n- ceramic lamp, Pune\nOpen items:\n- none"
            self.prev = len(s)
            self.inp += real(prompt)
            self.out += real(s)
            return s

    no_compaction = rolling_kwargs.pop("none", False)
    # Provider prices: (write multiplier, read multiplier, main input $/M,
    # summarizer input $/M, summarizer output $/M). Default: Anthropic
    # (Sonnet 4.6 main, Haiku 4.5 summaries).
    write_mult, read_mult, price_in, sum_in, sum_out = rolling_kwargs.pop("prices", (1.25, 0.10, 3.0, 1.0, 5.0))
    full, system = _conversation(turns, True), _handbook()
    summ = HaikuSized()
    comp = HistoryCompactor(model_call_fn=summ)
    state, prev, api, folds, pending, postponed = RollingSummary(), None, 0.0, 0, [], 0
    for t in range(turns):
        history = full[: 2 * t + 1]
        for item in list(pending):
            folds += run_fold_job(item, comp, state) == "folded"
            pending.remove(item)
        if no_compaction:
            msgs = history
        else:
            r = compact_history_rolling(history, comp, state, keep_last_n=4, token_threshold=3000,
                                        defer_fold=True, **rolling_kwargs)
            msgs = r.messages
            postponed += r.fold_postponed_for_cache
            if r.fold_job:
                pending.append(r.fold_job)
        sent = [system] + [m["content"] for m in msgs]
        k = 0
        if prev:
            while k < min(len(prev), len(sent)) and prev[k] == sent[k]:
                k += 1
        api += (sum(real(x) for x in sent[:k]) * read_mult + sum(real(x) for x in sent[k:]) * write_mult) * price_in / 1e6
        prev = sent
    summarizer = (summ.inp * sum_in + summ.out * sum_out) / 1e6
    return {"cost_usd": round(api + summarizer, 4), "summarizer_usd": round(summarizer, 4), "folds": folds,
            "turns_postponed": postponed}


GEMINI_SIM_PRICES = (1.0, 0.10, 0.75, 0.30, 2.50)  # implicit caching; 3.8 Flash main, 3.5 Flash-Lite summaries


def bench_cache_aware_compaction(turn_counts=(8, 10, 12, 16, 20, 24, 30, 40, 60, 100), provider="anthropic") -> dict:
    """
    Plain threshold folding vs. compaction_cache_aware.
    provider="anthropic": Sonnet 4.6 + Haiku summaries. Sanity check against
    the live 24-turn run: it measured -2.7% for plain threshold folding with
    Haiku; this simulation gives about -4%.
    provider="gemini": Gemini 3.8 Flash + 3.5 Flash-Lite summaries, implicit
    caching modeled as always hitting (in reality it's best effort, and
    prompts under the model's cache minimum never hit).
    """
    if provider == "gemini":
        prices, ratio, pricing = GEMINI_SIM_PRICES, 0.30 / 0.75, "gemini"
    else:
        prices, ratio, pricing = (1.25, 0.10, 3.0, 1.0, 5.0), 1 / 3, "anthropic"
    out = {}
    for n in turn_counts:
        base = _sim_conversation_cost(n, none=True, prices=prices)["cost_usd"]
        plain = _sim_conversation_cost(n, prices=prices)
        aware = _sim_conversation_cost(n, prices=prices, cache_aware=True, summarizer_price_ratio=ratio,
                                       cache_pricing=pricing)
        out[f"{n}_turns"] = {
            "no_compaction_usd": base,
            "threshold": {**plain, "vs_none_percent": round(100 * (plain["cost_usd"] - base) / base, 1)},
            "cache_aware": {**aware, "vs_none_percent": round(100 * (aware["cost_usd"] - base) / base, 1)},
        }
    return out


if __name__ == "__main__":
    report = {
        "tools_top_k_5": bench_tools(5),
        "tools_top_k_8": bench_tools(8),
        "tool_session": bench_tool_session(),
        "rag": bench_rag(),
        "compaction": bench_compaction(),
        "cache_aware_compaction": bench_cache_aware_compaction(),
        "cache_aware_compaction_gemini": bench_cache_aware_compaction(provider="gemini"),
    }
    print(json.dumps(report, indent=2))
