"""
live_test_free_features.py
--------------------------
Measures the free features against the REAL Anthropic API -- real billed
token counts from each response's `usage` field, not tonst's chars/4
estimates. benchmark_free_features.py answers "does the mechanism
behave?"; this answers "does it actually save money on a real provider,
and does the model still do the right thing?"

Part 1 -- tools (the question that matters most):
    Sends each task from benchmark_free_features.TASKS up to four ways:
      all             -- all 36 tool definitions, no optimization
      filtered        -- tonst.select_tools(top_k=5)
      deferred        -- tonst.build_anthropic_deferred_tools() with its
                         defaults (search-type tools kept loaded) plus
                         DEFERRED_TOOLS_SYSTEM_HINT in the system prompt
      deferred_plain  -- every tool deferred, no hint (what the first
                         full live run tested; kept to show the fix)
    and records the REAL input tokens billed plus what Claude did first.
    Each call gets one outcome:
      correct     -- called a tool the task needs
      acceptable  -- called a reasonable first-step tool listed in
                     ACCEPTABLE_FIRST_STEPS (e.g. listing tables before
                     querying)
      asked       -- no tool call; asked the user a question instead
      wrong_tool  -- called a tool that doesn't fit
      no_tool     -- no tool call and no question (e.g. "no such tool")
    success = correct + acceptable. A saving that lowers success is a
    failure, so both are reported side by side.

Part 2 -- rolling vs. stateless history compaction (cost AND latency):
    Replays one scripted support conversation turn by turn, three times:
    with no compaction (full history -- the baseline), with
    compact_history() (stateless) and with compact_history_rolling().
    Every turn records local_ms (tonst's compaction step, which runs
    BEFORE the API call and so adds directly to response time) and
    api_ms (the real API round trip), and the results report p50/p95/max
    per mode plus how many turns the local model ran on. Each turn is sent as a real request with
    Anthropic's multi-turn caching pattern (system prompt cached, plus a
    cache breakpoint on the last block of the conversation), and the real
    cache_read / cache_creation / uncached input tokens are recorded.
    Uses your local Ollama model for the summaries if it's running
    (gemma2:2b by default); otherwise both modes fall back to plain
    truncation, which still tests the caching behavior. Assistant turns
    are scripted, so both modes see the identical conversation, and
    max_tokens is tiny: only input-side cost is being measured.

Part 3 -- estimate accuracy (printed alongside the above):
    tonst's savings log uses chars/4 token estimates by default. Every
    call here also compares that estimate with the real billed input,
    and (for the all/filtered modes) checks Anthropic's free
    count_tokens endpoint against the billed number -- that's what
    tonst's optional AnthropicTokenCounter uses. (count_tokens doesn't
    support the tool search tool, so deferred mode isn't counted.)

Misses are diagnosable: for every call where Claude didn't call a
correct tool, the stop reason, Claude's text reply and (in deferred
mode) the tools its search returned are saved and printed.

Setup (same as cache_savings_demo_anthropic.py):
    1. ANTHROPIC_API_KEY in a .env file in this directory (gitignored),
       or exported in your shell.
    2. Optional, for real summaries in part 2: `ollama run gemma2:2b "hi"`
       first so the model is loaded.
    3. python3 live_test_free_features.py            (asks before spending)
       python3 live_test_free_features.py --part tools --tasks 10
       python3 live_test_free_features.py --part compaction --no-ollama

Cost: roughly $1.30 at claude-sonnet-4-6 list prices for everything
(~120 small tool calls + ~24 conversation calls); a cost estimate is
printed and confirmed before anything is sent. Results are written to
live_test_results.json.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time


def _load_dotenv_if_present():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv_if_present()

import requests  # noqa: E402  (tonst's own dependency)

from tonst import (  # noqa: E402
    select_tools,
    build_anthropic_deferred_tools,
    estimate_tool_tokens,
    compact_history,
    compact_history_rolling,
    run_fold_job,
    RollingSummary,
    HistoryCompactor,
    AnthropicTokenCounter,
    AnthropicSummarizer,
    DEFERRED_TOOLS_SYSTEM_HINT,
)
from tonst.compactor import _summary_message_text  # noqa: E402
from tonst.trim import estimate_tokens, flatten_messages  # noqa: E402
from benchmark_free_features import TOOLS, TASKS, ACCEPTABLE_FIRST_STEPS  # noqa: E402

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"

# Sonnet 4.6 list prices, USD per million tokens, used ONLY to print
# dollar figures. Cache writes (5-minute TTL) bill at 1.25x input and
# cache reads at 0.1x -- see cache_structuring.py. Update if pricing
# changes; the token counts themselves are always real.
PRICE_INPUT = 3.00
PRICE_OUTPUT = 15.00
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.10

TOOL_SYSTEM = (
    "You are an operations assistant with access to tools. For every request, call the single most "
    "appropriate tool to act on it. Always respond with a tool call, not with text."
)


# ---------------------------------------------------------------------
# API plumbing
# ---------------------------------------------------------------------

def _post(body: dict, api_key: str) -> dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    for attempt in range(5):
        resp = requests.post(API_URL, headers=headers, json=body, timeout=120)
        if resp.status_code in (429, 500, 529):
            wait = 2 ** attempt * 2
            print(f"    (API {resp.status_code}, retrying in {wait}s)")
            time.sleep(wait)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:600]}")
        return resp.json()
    raise RuntimeError("API kept returning 429/5xx; try again later")


def _usage(resp: dict) -> dict:
    u = resp.get("usage") or {}
    return {
        "input_tokens": int(u.get("input_tokens", 0)),
        "output_tokens": int(u.get("output_tokens", 0)),
        "cache_creation_input_tokens": int(u.get("cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(u.get("cache_read_input_tokens", 0) or 0),
    }


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _cost_usd(u: dict) -> float:
    return (
        u["input_tokens"] * PRICE_INPUT
        + u["cache_creation_input_tokens"] * PRICE_INPUT * CACHE_WRITE_MULT
        + u["cache_read_input_tokens"] * PRICE_INPUT * CACHE_READ_MULT
        + u["output_tokens"] * PRICE_OUTPUT
    ) / 1_000_000


def _total_input(u: dict) -> int:
    return u["input_tokens"] + u["cache_creation_input_tokens"] + u["cache_read_input_tokens"]


# ---------------------------------------------------------------------
# Part 1: tools
# ---------------------------------------------------------------------

def _found_tool_names(obj) -> list:
    """Tool names returned by tool search (tool_reference blocks), found defensively anywhere in a block."""
    out = []
    if isinstance(obj, dict):
        if obj.get("type") == "tool_reference" and obj.get("tool_name"):
            out.append(obj["tool_name"])
        for v in obj.values():
            out.extend(_found_tool_names(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_found_tool_names(v))
    return out


def _tool_call(model: str, tools: list, task: str, api_key: str, system: str = TOOL_SYSTEM) -> tuple:
    """Returns (first custom tool name or None, searches made, usage summed over any
    pause_turn continuations, detail dict for diagnosing misses)."""
    messages = [{"role": "user", "content": task}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    searches = 0
    detail = {"stop_reason": None, "text": "", "search_queries": [], "search_found": []}
    for _ in range(3):
        resp = _post(
            {"model": model, "max_tokens": 600, "system": system, "tools": tools, "messages": messages},
            api_key,
        )
        usage = _add_usage(usage, _usage(resp))
        blocks = resp.get("content") or []
        detail["stop_reason"] = resp.get("stop_reason")
        for b in blocks:
            if b.get("type") == "server_tool_use":
                searches += 1
                detail["search_queries"].append(json.dumps(b.get("input", {}))[:200])
            elif b.get("type") == "tool_search_tool_result":
                detail["search_found"].extend(_found_tool_names(b))
            elif b.get("type") == "text":
                detail["text"] = (detail["text"] + " " + b.get("text", "")).strip()[:500]
        for b in blocks:
            if b.get("type") == "tool_use":
                return b.get("name"), searches, usage, detail
        if resp.get("stop_reason") == "pause_turn":
            # Server-side tool loop paused; send the partial turn back to continue it.
            messages = messages + [{"role": "assistant", "content": blocks}]
            continue
        break
    return None, searches, usage, detail


MODES = ("all", "filtered", "deferred", "deferred_plain")


def _outcome(task: str, needed: list, name, detail: dict) -> str:
    if name in needed:
        return "correct"
    if name is not None and name in ACCEPTABLE_FIRST_STEPS.get(task, []):
        return "acceptable"
    if name is not None:
        return "wrong_tool"
    if detail.get("stop_reason") == "end_turn" and "?" in (detail.get("text") or ""):
        return "asked"
    return "no_tool"


def run_tools(model: str, api_key: str, n_tasks: int, modes=MODES) -> dict:
    tasks = TASKS[:n_tasks]
    deferred = build_anthropic_deferred_tools(TOOLS)
    deferred_plain = build_anthropic_deferred_tools(TOOLS, keep_search_tools_loaded=False)
    counter = AnthropicTokenCounter(model=model, api_key=api_key)
    rows = []
    print(f"\nPart 1: tools -- {len(tasks)} tasks x {len(modes)} modes ({', '.join(modes)}), "
          f"{len(TOOLS)} tools, model {model}")
    for n, (task, needed, group) in enumerate(tasks, 1):
        variants = {
            "all": (TOOLS, TOOL_SYSTEM),
            "filtered": (select_tools(TOOLS, task, top_k=5).tools, TOOL_SYSTEM),
            "deferred": (deferred, TOOL_SYSTEM + " " + DEFERRED_TOOLS_SYSTEM_HINT),
            "deferred_plain": (deferred_plain, TOOL_SYSTEM),
        }
        line = []
        for mode in modes:
            tools, system = variants[mode]
            name, searches, usage, detail = _tool_call(model, tools, task, api_key, system=system)
            outcome = _outcome(task, needed, name, detail)
            ok = outcome in ("correct", "acceptable")
            est = estimate_tool_tokens([t for t in tools if not t.get("defer_loading")]) + estimate_tokens(
                system + task
            )
            counted = None
            if not mode.startswith("deferred"):  # count_tokens doesn't support the tool search tool
                counted = counter.count_request(
                    {"system": system, "tools": tools, "messages": [{"role": "user", "content": task}]}
                )
            rows.append({
                "task": task, "group": group, "mode": mode, "tool_called": name, "outcome": outcome,
                "correct": ok, "needed": needed, "tools_sent": len(tools),
                "tools_loaded": len([t for t in tools if not t.get("defer_loading")]),
                "searches": searches, "usage": usage,
                "estimated_input_tokens": est, "count_tokens_endpoint": counted,
                "detail": detail if not ok else {"stop_reason": detail["stop_reason"],
                                                 "search_found": detail["search_found"]},
            })
            tag = {"correct": "OK  ", "acceptable": "OK* ", "asked": "ASK ", "wrong_tool": "WRNG",
                   "no_tool": "NONE"}[outcome]
            line.append(f"{mode}={tag}({_total_input(usage)})")
        print(f"  [{n:2}/{len(tasks)}] {group[:4]}  " + "  ".join(line) + f"  | {task[:40]}")

    summary = {}
    for mode in modes:
        for group in ("direct", "paraphrased", "overall"):
            r = [x for x in rows if x["mode"] == mode and (group == "overall" or x["group"] == group)]
            if not r:
                continue
            tot = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
            for x in r:
                tot = _add_usage(tot, x["usage"])
            entry = {
                "tasks": len(r),
                "success_percent": round(100 * sum(x["correct"] for x in r) / len(r), 1),
                "outcomes": {o: sum(1 for x in r if x["outcome"] == o)
                             for o in ("correct", "acceptable", "asked", "wrong_tool", "no_tool")},
                "avg_real_input_tokens": round(sum(_total_input(x["usage"]) for x in r) / len(r)),
                "avg_output_tokens": round(sum(x["usage"]["output_tokens"] for x in r) / len(r)),
                "cost_usd": round(_cost_usd(tot), 4),
            }
            if mode.startswith("deferred"):
                entry["avg_searches"] = round(sum(x["searches"] for x in r) / len(r), 2)
            summary.setdefault(mode, {})[group] = entry

    # Paired view: on the tasks sending ALL tools got right, how often does each other mode also get it right?
    if "all" in modes:
        by_task = {}
        for x in rows:
            by_task.setdefault(x["task"], {})[x["mode"]] = x
        base = [t for t, v in by_task.items() if v.get("all", {}).get("correct")]
        summary["paired_vs_all"] = {
            m: f"{sum(1 for t in base if by_task[t].get(m, {}).get('correct'))}/{len(base)}"
            for m in modes if m != "all"
        }

    ratios = [
        _total_input(x["usage"]) / x["estimated_input_tokens"]
        for x in rows if not x["mode"].startswith("deferred") and x["estimated_input_tokens"]
    ]
    counted = [
        (x["count_tokens_endpoint"], _total_input(x["usage"]))
        for x in rows if x["count_tokens_endpoint"] is not None and _total_input(x["usage"])
    ]
    summary["estimate_check"] = {
        "real_over_estimated_input_tokens_avg": round(sum(ratios) / len(ratios), 2) if ratios else None,
        "note": "1.0 = chars/4 estimate matched the real billed input; >1 means tonst under-estimates",
        "count_tokens_calls_compared": len(counted),
        "count_tokens_exact_matches": sum(1 for c, b in counted if c == b),
        "count_tokens_max_abs_diff": max((abs(c - b) for c, b in counted), default=None),
    }
    summary["misses"] = [
        {"mode": x["mode"], "outcome": x["outcome"], "task": x["task"], "needed": x["needed"],
         "tool_called": x["tool_called"], **x["detail"]}
        for x in rows if not x["correct"]
    ]
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------
# Part 2: rolling vs. stateless compaction
# ---------------------------------------------------------------------

def _handbook() -> str:
    # A realistic, static support handbook, comfortably above Sonnet 4.6's
    # 1,024-token cache minimum so the system prompt is cacheable.
    sections = {
        "Orders": [
            "Orders can be cancelled from the Orders page until they are handed to the courier.",
            "Once shipped, a cancellation becomes a return; the customer refuses delivery or starts a return.",
            "Order numbers are four or more digits prefixed with #; always confirm the number before acting.",
            "Split shipments show one tracking number per parcel; check every parcel before calling an order lost.",
            "Address changes are possible only before the label is printed; after that, ask the courier to reroute.",
        ],
        "Damaged items": [
            "Ask for a photo of the item and of the outer box before approving any claim.",
            "If the outer box is crushed, file a courier damage claim in addition to helping the customer.",
            "Customers choose between replacement and refund; never push one over the other.",
            "Replacements ship by the same method as the original order unless the customer upgrades.",
            "Fragile categories (glassware, ceramics, electronics screens) are replaced without requiring a return.",
        ],
        "Refunds": [
            "Refunds go to the original payment method within 5 to 10 business days after inspection.",
            "Cash on delivery orders are refunded by bank transfer; collect account details through the secure form only.",
            "Partial refunds are allowed for missing accessories; the amount comes from the accessory price list.",
            "Refunds above 20,000 rupees require a team lead's approval before they are issued.",
            "Never promise a refund date earlier than the payment provider's own processing time.",
        ],
        "Shipping": [
            "Standard shipping takes 3 to 5 business days in India and 7 to 12 internationally.",
            "Express shipping takes 1 to 2 business days in metro cities and costs 149 rupees.",
            "Upgrading a replacement to express is free when the original order arrived damaged.",
            "Deliveries can be scheduled for an evening slot (6pm to 9pm) in Pune, Mumbai, Bengaluru and Delhi.",
            "Couriers attempt delivery three times before returning a parcel to the warehouse.",
        ],
        "Returns process": [
            "Return labels are emailed within an hour of approval and are valid for 7 days.",
            "Items must be packed in any sturdy box; the original packaging is preferred but not required.",
            "Courier pickup can be booked instead of drop-off for items heavier than 10 kg.",
            "Returned items are inspected within 3 business days of reaching the warehouse.",
            "If inspection finds the item used or incomplete, contact the customer before rejecting the return.",
        ],
        "Payments": [
            "Accepted methods are UPI, credit and debit cards, net banking and cash on delivery under 5,000 rupees.",
            "Failed card payments are never retried automatically; the customer must retry from the Orders page.",
            "EMI orders that are returned are refunded to the card, and the bank cancels the remaining installments.",
            "Gift card balances are non-refundable but can be restored if an order paid by gift card is cancelled.",
            "Never ask a customer to share a card number, CVV or OTP in chat, email or over the phone.",
        ],
        "Tone and escalation": [
            "Acknowledge the problem in the first reply and state the next concrete step.",
            "Escalate to a team lead after two failed delivery attempts on a replacement.",
            "Security or fraud concerns always go to the trust team, regardless of order value.",
            "Summarize what was agreed at the end of every resolved conversation.",
            "Do not share internal ticket IDs with customers; give them the public case number instead.",
        ],
    }
    out = ["Acme Store -- Customer Support Handbook (internal, v7)\n"]
    for title, items in sections.items():
        out.append(f"\n## {title}")
        for i, item in enumerate(items, 1):
            out.append(f"{i}. {item} This applies to all customer tiers unless a team lead documents an exception in the case notes.")
    return "\n".join(out)


def _tool_output(i: int) -> str:
    """~2,000 characters (~500 tokens) of realistic, deterministic tool output -- the kind of
    content (logs, JSON, search results) that makes real agent and support histories long."""
    lines = [f"$ tracking_lookup --order 4471 --page {i}", "{"]
    for j in range(19):
        lines.append(
            f'  "event_{j}": {{"ts": "2026-09-{10 + (i + j) % 18:02d}T{(i * 7 + j) % 24:02d}:{(j * 13) % 60:02d}Z", '
            f'"hub": "{["PNQ-2", "BOM-1", "BLR-4", "DEL-3"][(i + j) % 4]}", "status": '
            f'"{["in_transit", "at_hub", "scan_ok", "out_for_delivery", "exception"][(i * 3 + j) % 5]}", '
            f'"latency_ms": {100 + (i * 37 + j * 11) % 900}}},'
        )
    lines.append("}")
    return "\n".join(lines)


def _conversation(turns: int, long_history: bool = False) -> list:
    facts = [
        ("Hi, my order #4471 arrived today and the lamp inside is cracked.", "Sorry about that! Could you share a photo of the lamp and the outer box?"),
        ("Sent both photos. The box is crushed on one corner.", "Thanks, that confirms courier damage. I'll file a claim with them. Would you like a replacement or a refund?"),
        ("A replacement please, not a refund.", "Noted: replacement for the ceramic lamp from order #4471. Since it's fragile, you don't need to return the broken one."),
        ("Great. Is my address still the Pune one ending in Baner Road?", "Yes, the replacement will go to the Baner Road address in Pune on file."),
        ("Can you send it express? I need it before the weekend.", "Express is free here because the original arrived damaged. I've upgraded the replacement to express."),
        ("I'm only home after 6pm on weekdays.", "I've requested the evening slot (6pm to 9pm) with the courier for your delivery."),
        ("Will I get a tracking number?", "Yes, you'll get it by SMS and email as soon as the parcel is handed over, usually within a day."),
        ("Also, the matching lampshade from the same order is fine, right?", "Right, only the lamp base is being replaced; the lampshade stays with you."),
        ("What happens if the replacement also arrives damaged?", "We'd replace it again, and after two failed attempts a team lead takes over your case personally."),
        ("Okay. Can I get the case number for reference?", "Your public case number is CS-20931. Please quote it if you contact us again."),
        ("Thanks. One more thing: can I change the colour to white?", "White is in stock at the same price, so I've switched the replacement to the white lamp."),
        ("Perfect, is anything else needed from me?", "Nothing else. To summarize: white replacement lamp, express, evening slot, Pune address, case CS-20931."),
        ("Actually, I just got a delivery SMS -- is that the replacement?", "Yes, that's the replacement parcel; it's out for delivery in today's evening slot."),
        ("It came! The white lamp is perfect.", "Wonderful, glad it arrived safely. I've closed the courier claim on our side."),
    ]
    msgs = []
    for i in range(turns):
        u, a = facts[i % len(facts)]
        if i >= len(facts):
            u, a = f"(follow-up {i}) {u}", f"(follow-up {i}) {a}"
        if long_history:
            a = f"{a}\n\n[tool output]\n{_tool_output(i)}"
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def _to_anthropic_messages(msgs: list) -> list:
    """Merge same-role neighbours into block lists (roles must alternate), start with a user turn,
    and put the moving cache breakpoint on the very last block."""
    out = []
    for m in msgs:
        if m.get("role") == "system":
            continue
        block = {"type": "text", "text": m["content"]}
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"].append(block)
        else:
            out.append({"role": m["role"], "content": [block]})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(Earlier conversation omitted.)"}]})
    if out:
        out[-1]["content"][-1] = {**out[-1]["content"][-1], "cache_control": {"type": "ephemeral"}}
    return out


COMPACTION_MODES = ("none", "stateless", "rolling", "rolling_bg", "rolling_bg_haiku", "rolling_bg_haiku_aware")
DEFAULT_COMPACTION_MODES = ("none", "stateless", "rolling")
# Long history: stateless would make the local model re-read thousands of
# tokens on nearly every turn (tens of seconds each on a laptop), and the
# short-chat runs already showed it's the worst option -- so the default
# compares no compaction against rolling, blocking vs. background.
DEFAULT_LONG_COMPACTION_MODES = ("none", "rolling_bg", "rolling_bg_haiku")

# Facts the scripted support conversation establishes, with the words that show
# a summary kept them. Scored only for facts whose turns were actually
# summarized (facts still in the verbatim window don't count either way).
FACTS_TO_KEEP = {
    "order #4471": ["4471"],
    "replacement, not refund": ["replacement"],
    "express shipping": ["express"],
    "Pune / Baner Road address": ["baner", "pune"],
    "evening delivery slot": ["evening", "6pm"],
    "tracking by SMS/email": ["tracking"],
    "case CS-20931": ["cs-20931"],
    "white lamp": ["white"],
}


def _fact_recall(summarized_text: str, kept_text: str) -> dict:
    src, kept = summarized_text.lower(), (kept_text or "").lower()
    present = {f: kws for f, kws in FACTS_TO_KEEP.items() if any(k in src for k in kws)}
    lost = [f for f, kws in present.items() if not any(k in kept for k in kws)]
    return {"facts_in_summarized_turns": len(present), "facts_kept": len(present) - len(lost), "lost": lost}


def _pct(values: list, q: float) -> float:
    """Nearest-rank percentile; 0 for an empty list."""
    if not values:
        return 0.0
    v = sorted(values)
    k = max(0, min(len(v) - 1, int(round(q / 100 * len(v) + 0.5)) - 1))
    return round(v[k], 1)


def _latency_summary(per_turn: list) -> dict:
    local = [t["local_ms"] for t in per_turn]
    api = [t["api_ms"] for t in per_turn]
    total = [t["local_ms"] + t["api_ms"] for t in per_turn]
    busy = [x for x in local if x >= 50]  # turns where the local model actually ran (tonst's own work is ~ms)
    return {
        "api_ms_p50": _pct(api, 50), "api_ms_p95": _pct(api, 95), "api_ms_max": _pct(api, 100),
        "local_ms_p50": _pct(local, 50), "local_ms_max": _pct(local, 100),
        "turns_with_local_model_work": len(busy),
        "local_ms_per_busy_turn_avg": round(sum(busy) / len(busy), 1) if busy else 0.0,
        "total_ms_p50": _pct(total, 50), "total_ms_p95": _pct(total, 95), "total_ms_max": _pct(total, 100),
        "local_share_of_total_time_percent": round(100 * sum(local) / sum(total), 1) if sum(total) else 0.0,
    }


def run_compaction(model: str, api_key: str, turns: int, threshold: int, use_ollama: bool, local_model: str,
                   modes=DEFAULT_COMPACTION_MODES, long_history: bool = False) -> dict:
    """
    Latency matters as much as cost here: compaction runs BEFORE the API
    call, in the same request, so any time the local model spends is added
    straight onto that turn's response time. Every turn records local_ms
    (tonst's compaction step) and api_ms (the real API round trip) so the
    two can be compared, per mode:
      none      -- full history every turn, no compaction (baseline: does a
                   longer prompt make the API itself slower?)
      stateless -- compact_history()
      rolling   -- compact_history_rolling(), summary made before the call
      rolling_bg -- compact_history_rolling(defer_fold=True): the summary runs
                   on a background thread in parallel with the API call and
                   is used from the next turn (TonstClient's
                   background_summary=True)
      rolling_bg_haiku -- same, but summaries come from Claude Haiku
                   (AnthropicSummarizer) instead of the local model; its
                   API cost is ADDED to this mode's cost
      rolling_bg_haiku_aware -- same as rolling_bg_haiku, plus
                   cache_aware=True: a due summary waits until it's
                   expected to pay for re-caching the prompt
    Every rolling mode also reports fact recall: of the known facts in the
    turns that got summarized, how many survive in the summary + pinned
    references.
    long_history adds ~500 tokens of tool output to every assistant turn,
    so history reaches ~10k+ tokens -- where compaction is supposed to pay.
    """
    import threading
    compactor = None
    if use_ollama:
        c = HistoryCompactor(model=local_model, timeout=90)
        if c.is_available():
            compactor = c
        else:
            print("  (Ollama not reachable -- compaction modes will use plain truncation instead of summaries)")
    handbook = _handbook()
    full = _conversation(turns, long_history)
    run_id = int(time.time())
    print(f"\nPart 2: compaction -- {turns} user turns{' (LONG history: ~500-token tool output per reply)' if long_history else ''}"
          f" x {len(modes)} modes ({', '.join(modes)}), keep_last_n=4, threshold={threshold} tokens, "
          f"summaries={'local ' + local_model if compactor else 'off (truncation)'}")

    results = {}
    for mode in modes:
        summarizer = None
        mode_compactor = compactor
        if mode.startswith("rolling_bg_haiku"):
            summarizer = AnthropicSummarizer(api_key=api_key)
            mode_compactor = HistoryCompactor(model_call_fn=summarizer)
        # A per-run, per-mode tag at the top of the system prompt keeps the
        # modes (and earlier runs) from sharing cache entries with each other.
        system = [{"type": "text", "text": f"[live-test {mode} {run_id}]\n" + handbook,
                   "cache_control": {"type": "ephemeral"}}]
        state = RollingSummary()
        total = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        per_turn, folds, est_total, postponed = [], 0, 0, 0
        bg_threads, bg_outcomes, bg_busy_at_start = [], [], 0
        for t in range(turns):
            history = full[: 2 * t + 1]  # up to and including this turn's user message
            scheduled = False
            if mode.startswith("rolling_bg") and state.fold_in_progress:
                bg_busy_at_start += 1  # the previous background summary hadn't finished yet
            t0 = time.perf_counter()
            if mode == "none":
                sent_msgs, did_fold = history, False
            elif mode == "stateless":
                r = compact_history(history, compactor, keep_last_n=4, token_threshold=threshold)
                sent_msgs, did_fold = r.messages, r.compacted
            elif mode == "rolling":
                r = compact_history_rolling(history, mode_compactor, state, keep_last_n=4, token_threshold=threshold)
                sent_msgs, did_fold = r.messages, r.summary_updated
            else:  # rolling_bg, rolling_bg_haiku
                aware = mode == "rolling_bg_haiku_aware"
                r = compact_history_rolling(history, mode_compactor, state, keep_last_n=4, token_threshold=threshold,
                                            defer_fold=True, cache_aware=aware,
                                            summarizer_price_ratio=(summarizer.price_input / PRICE_INPUT) if aware else 0.0)
                sent_msgs, did_fold = r.messages, False
                postponed += r.fold_postponed_for_cache
                if r.fold_job is not None:
                    scheduled = True
                    th = threading.Thread(
                        target=lambda job=r.fold_job, c=mode_compactor: bg_outcomes.append(run_fold_job(job, c, state)),
                        daemon=True,
                    )
                    th.start()  # runs in parallel with the API call below
                    bg_threads.append(th)
            local_ms = (time.perf_counter() - t0) * 1000
            folds += int(did_fold)

            t1 = time.perf_counter()
            resp = _post({"model": model, "max_tokens": 16, "system": system,
                          "messages": _to_anthropic_messages(sent_msgs)}, api_key)
            api_ms = (time.perf_counter() - t1) * 1000

            u = _usage(resp)
            total = _add_usage(total, u)
            if mode.startswith("rolling"):
                state.observe_cache_usage(_total_input(u), u["cache_read_input_tokens"])
            est_total += estimate_tokens(system[0]["text"]) + estimate_tokens(flatten_messages(sent_msgs))
            per_turn.append({**u, "local_ms": round(local_ms, 1), "api_ms": round(api_ms, 1), "summarized": did_fold,
                             "summary_started_in_background": scheduled})
            note = "  <- summary made (blocking)" if did_fold else ("  <- summary started in background" if scheduled else "")
            if mode.startswith("rolling_bg") and r.fold_postponed_for_cache:
                note = f"  <- summary postponed (payback ~{r.fold_payback_turns} turns)"
            print(f"  {mode:10} turn {t + 1:2}: local {local_ms:7.0f} ms  api {api_ms:6.0f} ms  |  uncached "
                  f"{u['input_tokens']:4}  write {u['cache_creation_input_tokens']:5}  read "
                  f"{u['cache_read_input_tokens']:5}{note}")
        for th in bg_threads:
            th.join(timeout=180)
        if mode.startswith("rolling_bg"):
            folds = sum(1 for o in bg_outcomes if o == "folded")
        results[mode] = {
            "usage_total": total,
            "cost_usd": round(_cost_usd(total), 4),
            "total_input_tokens": _total_input(total),
            "local_summaries_made": folds,
            "local_compaction_ms_total": round(sum(t["local_ms"] for t in per_turn)),
            "latency": _latency_summary(per_turn),
            "estimated_over_real_input": round(est_total / _total_input(total), 2) if _total_input(total) else None,
            "final_summary": _summary_message_text(state) if mode.startswith("rolling") else None,
            "per_turn": per_turn,
        }
        if mode.startswith("rolling"):
            summarized = flatten_messages([m for m in full[: 2 * turns - 1] if m.get("role") != "system"]
                                          [: state.summarized_count])
            results[mode]["fact_recall"] = _fact_recall(summarized, _summary_message_text(state) or "")
            results[mode]["observed_cache_hit_rate"] = state.cache_hit_rate
        if summarizer is not None:
            results[mode]["summarizer"] = {
                "model": summarizer.model, "calls": summarizer.calls, "failures": summarizer.failures,
                "usage": summarizer.usage_total, "cost_usd": round(summarizer.cost_usd(), 4),
            }
            results[mode]["api_cost_usd"] = results[mode]["cost_usd"]
            results[mode]["cost_usd"] = round(results[mode]["cost_usd"] + summarizer.cost_usd(), 4)
        if mode.startswith("rolling_bg"):
            results[mode]["background"] = {
                "summaries_started": len(bg_threads),
                "outcomes": bg_outcomes,
                "turns_where_previous_summary_still_running": bg_busy_at_start,
                "turns_summary_postponed_for_cache": postponed,
            }
    return results


# ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--part", choices=["tools", "compaction", "all"], default="all")
    p.add_argument("--tasks", type=int, default=len(TASKS), help=f"tool tasks to run (max {len(TASKS)})")
    p.add_argument("--modes", default=",".join(MODES),
                   help=f"comma-separated tool modes to compare (default: {','.join(MODES)})")
    p.add_argument("--turns", type=int, default=None,
                   help="user turns in the compaction conversation (default 12, or 24 with --long)")
    p.add_argument("--threshold", type=int, default=None,
                   help="compaction token threshold for part 2 (default 150, or 3000 -- tonst's real default -- with --long)")
    p.add_argument("--long", action="store_true",
                   help="part 2 with long history: ~500 tokens of tool output per assistant reply")
    p.add_argument("--compaction-modes", default=None,
                   help=f"comma-separated, from {','.join(COMPACTION_MODES)} "
                        f"(default {','.join(DEFAULT_COMPACTION_MODES)}; with --long {','.join(DEFAULT_LONG_COMPACTION_MODES)})")
    p.add_argument("--model", default=os.environ.get("TONST_LIVE_MODEL", DEFAULT_MODEL))
    p.add_argument("--local-model", default="gemma2:2b")
    p.add_argument("--no-ollama", action="store_true", help="skip local summaries in part 2 (truncation only)")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation before spending")
    args = p.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set. Put it in a .env file next to this script or export it.")
        return 1

    n_tasks = max(1, min(args.tasks, len(TASKS)))
    threshold = args.threshold if args.threshold is not None else (3000 if args.long else 150)
    if args.turns is None:
        args.turns = 24 if args.long else 12
    if args.compaction_modes:
        c_modes = tuple(m.strip() for m in args.compaction_modes.split(",") if m.strip())
    else:
        c_modes = DEFAULT_LONG_COMPACTION_MODES if args.long else DEFAULT_COMPACTION_MODES
    if set(c_modes) - set(COMPACTION_MODES):
        print(f"Unknown compaction mode(s): {sorted(set(c_modes) - set(COMPACTION_MODES))}")
        return 1
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    unknown = set(modes) - set(MODES)
    if unknown:
        print(f"Unknown mode(s): {sorted(unknown)}. Choose from {', '.join(MODES)}.")
        return 1
    est = 0.0
    if args.part in ("tools", "all"):
        est += n_tasks * len(modes) * (2500 * PRICE_INPUT + 150 * PRICE_OUTPUT) / 1_000_000
    if args.part in ("compaction", "all"):
        # Calibrated on the real runs. Short chats: ~$0.0013 per call. Long history: each
        # call re-reads everything so far, so cost grows with the turn number -- fitted to
        # the 12-turn --long run ($0.22 for 3 modes): ~$0.0012 + $0.00075 x turn per call.
        if args.long:
            est += len(c_modes) * sum(0.0012 + 0.00075 * t for t in range(1, args.turns + 1))
            # a few Haiku summaries per Haiku mode, well under a cent each
            est += sum(0.005 * max(1, args.turns // 5) for m in c_modes if m.startswith("rolling_bg_haiku"))
        else:
            est += args.turns * len(c_modes) * 0.0015
    print(f"Model: {args.model}. Rough cost estimate: ~${est:.2f} (list prices; real usage is printed per call).")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 0

    out = {"model": args.model, "ran_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.part in ("tools", "all"):
        out["tools"] = run_tools(args.model, api_key, n_tasks, modes)
    if args.part in ("compaction", "all"):
        out["compaction"] = run_compaction(args.model, api_key, args.turns, threshold,
                                           not args.no_ollama, args.local_model, c_modes, args.long)

    print("\n================ RESULTS ================")
    if "tools" in out:
        s = out["tools"]["summary"]
        print("Tools (real billed input per request; success = called a correct or reasonable first tool):")
        for mode in [m for m in MODES if m in s]:
            o = s[mode]["overall"]
            oc = o["outcomes"]
            extra = f"  searches/call {o['avg_searches']}" if "avg_searches" in o else ""
            print(f"  {mode:14} success {o['success_percent']:5}%  input {o['avg_real_input_tokens']:5}  "
                  f"output {o['avg_output_tokens']:4}  cost ${o['cost_usd']:.4f}{extra}")
            print(f"  {'':14} correct {oc['correct']}, acceptable {oc['acceptable']}, asked {oc['asked']}, "
                  f"wrong tool {oc['wrong_tool']}, no tool {oc['no_tool']}")
            for g in ("direct", "paraphrased"):
                if g in s[mode]:
                    gg = s[mode][g]
                    print(f"      {g:11} success {gg['success_percent']:5}%  input {gg['avg_real_input_tokens']:5}")
        if "paired_vs_all" in s:
            print("  on the tasks where sending ALL tools succeeded, the other modes also succeeded on: "
                  + ", ".join(f"{m} {v}" for m, v in s["paired_vs_all"].items()))
        ec = s["estimate_check"]
        print(f"  chars/4 estimate check: real/estimated = {ec['real_over_estimated_input_tokens_avg']}")
        if ec["count_tokens_calls_compared"]:
            print(f"  count_tokens endpoint vs billed: {ec['count_tokens_exact_matches']}/"
                  f"{ec['count_tokens_calls_compared']} exact, max difference {ec['count_tokens_max_abs_diff']} tokens")
        if s["misses"]:
            print(f"  misses ({len(s['misses'])}) -- why Claude didn't call a correct tool:")
            for m in s["misses"]:
                print(f"    [{m['mode']}] ({m['outcome']}) {m['task'][:60]}")
                print(f"        needed {m['needed']}, called {m['tool_called']}, stop_reason {m.get('stop_reason')}")
                if m.get("search_found") or m.get("search_queries"):
                    print(f"        searched {m.get('search_queries')} -> found {m.get('search_found')}")
                if m.get("text"):
                    print(f"        Claude said: {m['text'][:300]}")
    if "compaction" in out:
        c = out["compaction"]
        print("Compaction -- cost (real usage over the whole conversation):")
        for mode in [m for m in COMPACTION_MODES if m in c]:
            r = c[mode]
            u = r["usage_total"]
            vs = f" ({100 * (r['cost_usd'] - c['none']['cost_usd']) / c['none']['cost_usd']:+.1f}% vs none)" \
                if "none" in c and mode != "none" and c["none"]["cost_usd"] else ""
            print(f"  {mode:22} cost ${r['cost_usd']:.4f}{vs}  uncached {u['input_tokens']:6}  cache write "
                  f"{u['cache_creation_input_tokens']:6}  cache read {u['cache_read_input_tokens']:6}  "
                  f"summaries {r['local_summaries_made']:2}  estimate/real {r['estimated_over_real_input']}")
        print("Compaction -- latency per turn (local = tonst compaction before the call; api = real round trip):")
        for mode in [m for m in COMPACTION_MODES if m in c]:
            L = c[mode]["latency"]
            print(f"  {mode:22} api p50 {L['api_ms_p50']:6.0f} / p95 {L['api_ms_p95']:6.0f} ms   "
                  f"total p50 {L['total_ms_p50']:6.0f} / p95 {L['total_ms_p95']:6.0f} / max {L['total_ms_max']:6.0f} ms   "
                  f"local model ran on {L['turns_with_local_model_work']:2} turns, "
                  f"avg {L['local_ms_per_busy_turn_avg']:5.0f} ms each ({L['local_share_of_total_time_percent']}% of all time)")
        for mode in ("rolling_bg", "rolling_bg_haiku", "rolling_bg_haiku_aware"):
            if mode in c:
                b = c[mode]["background"]
                print(f"  {mode}: {b['summaries_started']} background summaries started, outcomes {b['outcomes']}; "
                      f"previous one still running at the start of {b['turns_where_previous_summary_still_running']} turns; "
                      f"postponed for cache on {b.get('turns_summary_postponed_for_cache', 0)} turns")
        for mode in ("rolling_bg_haiku", "rolling_bg_haiku_aware"):
            if mode in c and "summarizer" in c[mode]:
                sm = c[mode]["summarizer"]
                print(f"  {mode} summarizer ({sm['model']}): {sm['calls']} calls, {sm['failures']} failed, "
                      f"cost ${sm['cost_usd']:.4f} (included in that mode's cost above)")
        print("Compaction -- fact recall (facts from summarized turns that survive in summary + pinned references):")
        for mode in [m for m in COMPACTION_MODES if m in c and "fact_recall" in c[m]]:
            fr = c[mode]["fact_recall"]
            lost = f"  lost: {', '.join(fr['lost'])}" if fr["lost"] else ""
            print(f"  {mode:22} {fr['facts_kept']}/{fr['facts_in_summarized_turns']} kept{lost}")
        for mode in ("rolling", "rolling_bg", "rolling_bg_haiku", "rolling_bg_haiku_aware"):
            if mode in c and c[mode]["final_summary"]:
                print(f"  final {mode} summary (check it for accuracy):\n    "
                      + c[mode]["final_summary"].replace("\n", "\n    "))

    with open("live_test_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nFull per-call results: live_test_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
