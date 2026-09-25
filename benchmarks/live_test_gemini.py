"""
benchmarks/live_test_gemini.py
-------------------
The same live checks as benchmarks/live_test_free_features.py, against Google's
Gemini API instead of Anthropic's. Costs real money (the script prints an
upper-bound estimate and asks before spending).

Why a separate run matters: Gemini's economics differ in ways that change
tonst's conclusions.
  - Caching is implicit (automatic, best effort) with NO write surcharge;
    cached input bills at 10% of the input price. Anthropic charges 1.25x
    to write the cache, which is what made compaction near break-even there.
  - The minimum cacheable prompt is larger (4,096 tokens on newer Flash
    models vs. 1,024 on Sonnet 4.6), so small prompts may never cache.
  - Gemini 3.x models always think; thinking tokens bill as output.

Part 1 -- tools: every task in TASKS with ALL 36 tool definitions vs. only
  the ones select_tools() picks (top 5). Deferred tool loading is an
  Anthropic API feature, so it isn't tested here. Scored the same way as
  the Anthropic run (correct / acceptable / asked / wrong_tool / no_tool).
  Also measures: implicit cache hits, the free countTokens endpoint vs.
  billed prompt tokens, how long select_tools() and a countTokens call
  take.

Part 2 -- compaction (long history, ~500 tokens of tool output per reply):
    none                 -- full history every turn
    rolling_bg_lite      -- rolling compaction, background summaries by a
                            small Gemini model (GeminiSummarizer)
    rolling_bg_lite_aware -- the same with cache_aware=True and Gemini cache
                            pricing (no write surcharge)
    rolling_bg_local     -- (optional) background summaries by the local
                            Ollama model
  Per turn: tonst's own time (local_ms) and the API round trip (api_ms), so
  end-to-end latency is visible; cost includes thinking tokens and the
  summarizer's own calls; fact recall as in the Anthropic run.

Setup: GEMINI_API_KEY in .env (or the environment).
Run:   python3 benchmarks/live_test_gemini.py                      # both parts, gemini-3.8-flash
       python3 benchmarks/live_test_gemini.py --part tools --tasks 5
       python3 benchmarks/live_test_gemini.py --model gemini-3.1-pro-preview
Results: live_test_gemini_results.json
"""

from __future__ import annotations
# Run from anywhere: make the repo root (and this folder) importable without `pip install -e .`.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import argparse
import json
import os
import sys
import threading
import time

# Importing the Anthropic live script loads .env and gives us the shared
# fixtures (tasks, conversation, handbook, fact list), so both providers
# are tested on identical inputs.
from live_test_free_features import (
    TOOL_SYSTEM,
    FACTS_TO_KEEP,
    _handbook,
    _conversation,
    _fact_recall,
    _latency_summary,
)
from benchmark_free_features import TOOLS, TASKS, ACCEPTABLE_FIRST_STEPS

import requests

from tonst import (
    select_tools,
    estimate_tool_tokens,
    compact_history_rolling,
    run_fold_job,
    RollingSummary,
    HistoryCompactor,
    GeminiSummarizer,
    GeminiTokenCounter,
)
from tonst.compactor import _summary_message_text
from tonst.summarizers import gemini_usage
from tonst.trim import estimate_tokens, flatten_messages

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-3.8-flash"

# USD per million tokens: (input, cached input, output incl. thinking).
# From Google's pricing page on 2026-09-25 (the 3.6-3.8 Flash prices are
# promotional through 2026-12-31 and double after). Used only to print
# dollar figures; token counts are always the API's own.
PRICES = {
    "gemini-3.8-flash": (0.75, 0.075, 3.75),
    "gemini-3.7-flash": (0.75, 0.075, 3.75),
    "gemini-3.6-flash": (0.75, 0.075, 3.75),
    "gemini-3.5-flash": (1.50, 0.15, 9.00),
    "gemini-3-flash-preview": (0.50, 0.05, 3.00),
    "gemini-3.1-pro-preview": (2.00, 0.20, 12.00),
    "gemini-3.1-flash-lite": (0.25, 0.025, 1.50),
}

_thinking_supported = {}  # model -> False once the API has rejected thinkingConfig


# ---------------------------------------------------------------------
# API plumbing
# ---------------------------------------------------------------------

def _post(model: str, body: dict, api_key: str, thinking_level) -> tuple:
    """POST generateContent; returns (response json, api_ms). Retries 429/5xx,
    and drops thinkingConfig once if the model rejects it."""
    headers = {"x-goog-api-key": api_key, "content-type": "application/json"}
    body = json.loads(json.dumps(body))
    gen = body.setdefault("generationConfig", {})
    if thinking_level and _thinking_supported.get(model, True):
        gen["thinkingConfig"] = {"thinkingLevel": thinking_level}
    for attempt in range(6):
        t0 = time.perf_counter()
        resp = requests.post(API.format(model=model), headers=headers, json=body, timeout=180)
        ms = (time.perf_counter() - t0) * 1000
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt * 2
            print(f"    (API {resp.status_code}, retrying in {wait}s)")
            time.sleep(wait)
            continue
        if resp.status_code == 400 and "thinking" in resp.text.lower() and "thinkingConfig" in gen:
            print(f"    (model rejected thinkingLevel={thinking_level!r}; continuing with the model's default)")
            _thinking_supported[model] = False
            gen.pop("thinkingConfig", None)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:800]}")
        return resp.json(), ms
    raise RuntimeError("API kept returning 429/5xx; try again later")


def _zero() -> dict:
    return {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}


def _add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def _cost(u: dict, prices: tuple) -> float:
    p_in, p_cached, p_out = prices
    uncached = max(0, u["prompt_tokens"] - u["cached_tokens"])
    return (uncached * p_in + u["cached_tokens"] * p_cached
            + (u["output_tokens"] + u["thinking_tokens"]) * p_out) / 1_000_000


def _input_cost(u: dict, prices: tuple) -> float:
    return _cost({**u, "output_tokens": 0, "thinking_tokens": 0}, prices)


def _parts(resp: dict) -> list:
    cands = resp.get("candidates") or [{}]
    return (cands[0].get("content") or {}).get("parts") or []


def _finish(resp: dict):
    return (resp.get("candidates") or [{}])[0].get("finishReason")


def _declarations(tools: list) -> list:
    return [{"functionDeclarations": [
        {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}
        for t in tools
    ]}]


def _to_contents(msgs: list) -> list:
    """{"role","content"} messages -> Gemini contents: assistant->model, same-role
    neighbours merged, must start with a user turn. System messages are dropped
    (the system prompt goes in systemInstruction)."""
    out = []
    for m in msgs:
        if m.get("role") == "system":
            continue
        role = "model" if m["role"] == "assistant" else "user"
        part = {"text": m["content"]}
        if out and out[-1]["role"] == role:
            out[-1]["parts"].append(part)
        else:
            out.append({"role": role, "parts": [part]})
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "parts": [{"text": "(Earlier conversation omitted.)"}]})
    return out


# ---------------------------------------------------------------------
# Part 1: tools
# ---------------------------------------------------------------------

TOOL_MODES = ("all", "filtered")


def _outcome(task: str, needed: list, name, text: str) -> str:
    if name in needed:
        return "correct"
    if name is not None and name in ACCEPTABLE_FIRST_STEPS.get(task, []):
        return "acceptable"
    if name is not None:
        return "wrong_tool"
    if "?" in (text or ""):
        return "asked"
    return "no_tool"


def run_tools(model: str, api_key: str, n_tasks: int, thinking: str, prices: tuple) -> dict:
    tasks = TASKS[:n_tasks]
    counter = GeminiTokenCounter(model=model, api_key=api_key)
    rows = []
    print(f"\nPart 1: tools -- {len(tasks)} tasks x {len(TOOL_MODES)} modes, {len(TOOLS)} tools, "
          f"model {model}, thinking {thinking}")
    for n, (task, needed, group) in enumerate(tasks, 1):
        t0 = time.perf_counter()
        filtered = select_tools(TOOLS, task, top_k=5).tools
        select_ms = (time.perf_counter() - t0) * 1000
        line = []
        for mode in TOOL_MODES:
            tools = TOOLS if mode == "all" else filtered
            body = {
                "systemInstruction": {"parts": [{"text": TOOL_SYSTEM}]},
                "tools": _declarations(tools),
                "contents": [{"role": "user", "parts": [{"text": task}]}],
                "generationConfig": {"maxOutputTokens": 2048, "temperature": 0},
            }
            resp, api_ms = _post(model, body, api_key, thinking)
            u = gemini_usage(resp)
            parts = _parts(resp)
            name = next((p["functionCall"].get("name") for p in parts if p.get("functionCall")), None)
            text = " ".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))[:500]
            outcome = _outcome(task, needed, name, text)
            ok = outcome in ("correct", "acceptable")
            t1 = time.perf_counter()
            counted = counter.count_request(body)
            count_ms = (time.perf_counter() - t1) * 1000
            rows.append({
                "task": task, "group": group, "mode": mode, "tool_called": name, "outcome": outcome,
                "correct": ok, "needed": needed, "tools_sent": len(tools), "usage": u,
                "cost_usd": _cost(u, prices),
                "estimated_input_tokens": estimate_tool_tokens(tools) + estimate_tokens(TOOL_SYSTEM + task),
                "count_tokens_endpoint": counted, "count_tokens_ms": round(count_ms, 1),
                "select_tools_ms": round(select_ms, 2) if mode == "filtered" else 0.0,
                "api_ms": round(api_ms, 1), "finish_reason": _finish(resp),
                "text": text if not ok else "",
            })
            tag = {"correct": "OK  ", "acceptable": "OK* ", "asked": "ASK ", "wrong_tool": "WRNG",
                   "no_tool": "NONE"}[outcome]
            line.append(f"{mode}={tag}({u['prompt_tokens']}, cached {u['cached_tokens']})")
        print(f"  [{n:2}/{len(tasks)}] {group[:4]}  " + "  ".join(line) + f"  | {task[:40]}")

    summary = {}
    for mode in TOOL_MODES:
        for group in ("direct", "paraphrased", "overall"):
            r = [x for x in rows if x["mode"] == mode and (group == "overall" or x["group"] == group)]
            if not r:
                continue
            tot = _zero()
            for x in r:
                tot = _add(tot, x["usage"])
            api = [x["api_ms"] for x in r]
            summary.setdefault(mode, {})[group] = {
                "tasks": len(r),
                "success_percent": round(100 * sum(x["correct"] for x in r) / len(r), 1),
                "outcomes": {o: sum(1 for x in r if x["outcome"] == o)
                             for o in ("correct", "acceptable", "asked", "wrong_tool", "no_tool")},
                "avg_prompt_tokens": round(tot["prompt_tokens"] / len(r)),
                "cached_share_percent": round(100 * tot["cached_tokens"] / tot["prompt_tokens"], 1)
                if tot["prompt_tokens"] else 0.0,
                "avg_output_tokens": round(tot["output_tokens"] / len(r)),
                "avg_thinking_tokens": round(tot["thinking_tokens"] / len(r)),
                "cost_usd": round(_cost(tot, prices), 4),
                "input_cost_usd": round(_input_cost(tot, prices), 4),
                "api_ms_p50": sorted(api)[len(api) // 2],
            }
    by_task = {}
    for x in rows:
        by_task.setdefault(x["task"], {})[x["mode"]] = x
    base = [t for t, v in by_task.items() if v.get("all", {}).get("correct")]
    summary["paired_vs_all"] = {
        "filtered": f"{sum(1 for t in base if by_task[t].get('filtered', {}).get('correct'))}/{len(base)}"}
    ratios = [x["usage"]["prompt_tokens"] / x["estimated_input_tokens"] for x in rows
              if x["estimated_input_tokens"] and x["usage"]["prompt_tokens"]]
    counted = [(x["count_tokens_endpoint"], x["usage"]["prompt_tokens"]) for x in rows
               if x["count_tokens_endpoint"] is not None and x["usage"]["prompt_tokens"]]
    count_ms = sorted(x["count_tokens_ms"] for x in rows if x["count_tokens_endpoint"] is not None)
    sel_ms = sorted(x["select_tools_ms"] for x in rows if x["mode"] == "filtered")
    summary["estimate_check"] = {
        "real_over_estimated_input_tokens_avg": round(sum(ratios) / len(ratios), 2) if ratios else None,
        "count_tokens_calls_compared": len(counted),
        "count_tokens_exact_matches": sum(1 for c, b in counted if c == b),
        "count_tokens_max_abs_diff": max((abs(c - b) for c, b in counted), default=None),
        "count_tokens_ms_p50": count_ms[len(count_ms) // 2] if count_ms else None,
        "select_tools_ms_p50": sel_ms[len(sel_ms) // 2] if sel_ms else None,
        "select_tools_ms_max": sel_ms[-1] if sel_ms else None,
    }
    summary["misses"] = [
        {"mode": x["mode"], "outcome": x["outcome"], "task": x["task"], "needed": x["needed"],
         "tool_called": x["tool_called"], "finish_reason": x["finish_reason"], "text": x["text"]}
        for x in rows if not x["correct"]
    ]
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------
# Part 2: compaction
# ---------------------------------------------------------------------

COMPACTION_MODES = ("none", "rolling_bg_lite", "rolling_bg_lite_aware", "rolling_bg_local")
DEFAULT_COMPACTION_MODES = ("none", "rolling_bg_lite", "rolling_bg_lite_aware")


def _save(out: dict) -> None:
    with open("live_test_gemini_results.json", "w") as f:
        json.dump(out, f, indent=2)


def run_compaction(model: str, api_key: str, turns: int, threshold: int, thinking: str, prices: tuple,
                   modes, summary_model: str, local_model: str, on_mode_done=None) -> dict:
    handbook = _handbook()
    full = _conversation(turns, True)
    run_id = int(time.time())
    print(f"\nPart 2: compaction -- {turns} user turns, long history, {len(modes)} modes ({', '.join(modes)}), "
          f"keep_last_n=4, threshold={threshold}, model {model}, summaries by {summary_model}")
    results = {}
    for mode in modes:
        summarizer, compactor = None, None
        if mode.startswith("rolling_bg_lite"):
            summarizer = GeminiSummarizer(model=summary_model, api_key=api_key)
            compactor = HistoryCompactor(model_call_fn=summarizer)
        elif mode == "rolling_bg_local":
            c = HistoryCompactor(model=local_model, timeout=90)
            compactor = c if c.is_available() else None
            if compactor is None:
                print("  (Ollama not reachable -- rolling_bg_local will fall back to plain truncation)")
        aware = mode == "rolling_bg_lite_aware"
        ratio = (summarizer.price_input / prices[0]) if (aware and summarizer) else 0.0
        # A per-run, per-mode tag at the top keeps modes from sharing implicit cache entries.
        system = f"[live-test {mode} {run_id}]\n" + handbook
        state = RollingSummary()
        total = _zero()
        per_turn, threads, outcomes = [], [], []
        busy_at_start, postponed = 0, 0
        for t in range(turns):
            history = full[: 2 * t + 1]
            scheduled = False
            note = ""
            t0 = time.perf_counter()
            if mode == "none":
                sent = history
            else:
                if state.fold_in_progress:
                    busy_at_start += 1
                r = compact_history_rolling(
                    history, compactor, state, keep_last_n=4, token_threshold=threshold, defer_fold=True,
                    cache_aware=aware, summarizer_price_ratio=ratio, cache_pricing="gemini",
                )
                sent = r.messages
                postponed += r.fold_postponed_for_cache
                if r.fold_postponed_for_cache:
                    note = f"  <- summary postponed (payback ~{r.fold_payback_turns} turns)"
                if r.fold_job is not None:
                    scheduled = True
                    note = "  <- summary started in background"
                    th = threading.Thread(
                        target=lambda job=r.fold_job, c=compactor: outcomes.append(run_fold_job(job, c, state)),
                        daemon=True)
                    th.start()
                    threads.append(th)
            local_ms = (time.perf_counter() - t0) * 1000
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": _to_contents(sent),
                "generationConfig": {"maxOutputTokens": 512, "temperature": 0},
            }
            resp, api_ms = _post(model, body, api_key, thinking)
            u = gemini_usage(resp)
            total = _add(total, u)
            if mode != "none":
                state.observe_cache_usage(u["prompt_tokens"], u["cached_tokens"])
            per_turn.append({**u, "local_ms": round(local_ms, 1), "api_ms": round(api_ms, 1),
                             "summarized": False, "summary_started_in_background": scheduled})
            print(f"  {mode:22} turn {t + 1:2}: local {local_ms:6.1f} ms  api {api_ms:6.0f} ms  |  prompt "
                  f"{u['prompt_tokens']:6}  cached {u['cached_tokens']:6}  out {u['output_tokens']:4}  "
                  f"think {u['thinking_tokens']:4}{note}")
        for th in threads:
            th.join(timeout=180)
        res = {
            "usage_total": total,
            "api_cost_usd": round(_cost(total, prices), 4),
            "input_cost_usd": round(_input_cost(total, prices), 4),
            "cached_share_percent": round(100 * total["cached_tokens"] / total["prompt_tokens"], 1)
            if total["prompt_tokens"] else 0.0,
            "latency": _latency_summary(per_turn),
            "per_turn": per_turn,
        }
        res["cost_usd"] = res["api_cost_usd"]
        if mode != "none":
            summarized = flatten_messages([m for m in full[: 2 * turns - 1]][: state.summarized_count])
            res["final_summary"] = _summary_message_text(state)
            res["fact_recall"] = _fact_recall(summarized, res["final_summary"] or "")
            res["observed_cache_hit_rate"] = state.cache_hit_rate
            res["background"] = {"summaries_started": len(threads), "outcomes": outcomes,
                                 "turns_where_previous_summary_still_running": busy_at_start,
                                 "turns_summary_postponed_for_cache": postponed}
        if summarizer is not None:
            res["summarizer"] = {"model": summarizer.model, "calls": summarizer.calls,
                                 "failures": summarizer.failures, "usage": summarizer.usage_total,
                                 "cost_usd": round(summarizer.cost_usd(), 4)}
            res["cost_usd"] = round(res["api_cost_usd"] + summarizer.cost_usd(), 4)
        results[mode] = res
        if on_mode_done:
            on_mode_done(results)  # saved after every mode, so a mid-run failure keeps what finished
    return results


# ---------------------------------------------------------------------

def _preflight(model: str, api_key: str, thinking: str) -> dict:
    """One tiny call: checks the key and model, and shows what usage the API reports."""
    resp, ms = _post(model, {"contents": [{"role": "user", "parts": [{"text": "Reply with the word OK."}]}],
                             "generationConfig": {"maxOutputTokens": 256}}, api_key, thinking)
    u = gemini_usage(resp)
    text = " ".join(p.get("text", "") for p in _parts(resp) if not p.get("thought")).strip()
    print(f"Preflight: {model} answered {text[:30]!r} in {ms:.0f} ms; usage reported: {resp.get('usageMetadata')}")
    if not _thinking_supported.get(model, True):
        print("  note: this model rejected the thinking level; runs use its default thinking")
    return u


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live Gemini benchmark for tonst's free features")
    p.add_argument("--part", choices=("tools", "compaction", "all"), default="all")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--thinking", default="low",
                   help="thinkingLevel for the main model (default low; '' = the model's default)")
    p.add_argument("--tasks", type=int, default=len(TASKS))
    p.add_argument("--turns", type=int, default=24)
    p.add_argument("--threshold", type=int, default=3000)
    p.add_argument("--compaction-modes", default=",".join(DEFAULT_COMPACTION_MODES),
                   help=f"comma-separated, from {','.join(COMPACTION_MODES)}")
    p.add_argument("--summary-model", default="gemini-3.5-flash-lite")
    p.add_argument("--local-model", default="gemma2:2b")
    p.add_argument("--price", default=None,
                   help="input,cached,output USD per million, for a model not in the built-in table")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = p.parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY in .env or the environment.")
        return 1
    if args.price:
        prices = tuple(float(x) for x in args.price.split(","))
    elif args.model in PRICES:
        prices = PRICES[args.model]
    else:
        print(f"No built-in prices for {args.model}; pass --price input,cached,output (USD per million).")
        return 1
    modes = tuple(m.strip() for m in args.compaction_modes.split(",") if m.strip())
    if set(modes) - set(COMPACTION_MODES):
        print(f"Unknown compaction mode(s): {sorted(set(modes) - set(COMPACTION_MODES))}")
        return 1
    thinking = args.thinking or None
    n_tasks = max(1, min(args.tasks, len(TASKS)))

    _preflight(args.model, api_key, thinking)

    # Upper-bound estimate: real tokens ~ chars/4 x 1.8 for tool JSON and / 0.63 for the
    # JSON-heavy chat (both measured on the Anthropic run), no cache hits at all, and
    # ~600 output+thinking tokens per call.
    p_in, _, p_out = prices
    est = 0.0
    if args.part in ("tools", "all"):
        all_tok = (estimate_tool_tokens(TOOLS) + 150) * 1.8
        filt_tok = (estimate_tool_tokens(TOOLS[:5]) + 150) * 1.8
        est += n_tasks * ((all_tok + filt_tok) * p_in + 2 * 600 * p_out) / 1e6
    if args.part in ("compaction", "all"):
        full = _conversation(args.turns, True)
        sys_tok = estimate_tokens(_handbook())
        prompt_tok = sum((sys_tok + estimate_tokens(flatten_messages(full[: 2 * t + 1]))) / 0.63
                         for t in range(args.turns))
        est += len(modes) * (prompt_tok * p_in + args.turns * 600 * p_out) / 1e6
        est += sum(0.003 for m in modes if m.startswith("rolling_bg_lite")) * max(1, args.turns // 6)
    print(f"Model {args.model} (${p_in}/M in, ${p_out}/M out). Upper-bound cost estimate: ~${est:.2f} "
          f"(assumes no cache hits; real usage is printed per call).")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 0

    out = {"provider": "gemini", "model": args.model, "thinking_level": thinking,
           "thinking_supported": _thinking_supported.get(args.model, True),
           "prices_usd_per_million": {"input": prices[0], "cached": prices[1], "output": prices[2]},
           "ran_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.part in ("tools", "all"):
        out["tools"] = run_tools(args.model, api_key, n_tasks, thinking, prices)
        _save(out)
    if args.part in ("compaction", "all"):
        out["compaction"] = run_compaction(args.model, api_key, args.turns, args.threshold, thinking, prices,
                                           modes, args.summary_model, args.local_model,
                                           on_mode_done=lambda r: _save({**out, "compaction": r}))

    print("\n================ SUMMARY ================")
    if "tools" in out:
        s = out["tools"]["summary"]
        for mode in TOOL_MODES:
            o = s[mode]["overall"]
            print(f"  tools {mode:9} success {o['success_percent']:5}%  {o['outcomes']}  avg prompt "
                  f"{o['avg_prompt_tokens']:5}  cached {o['cached_share_percent']:5}%  thinking/call "
                  f"{o['avg_thinking_tokens']:4}  cost ${o['cost_usd']:.4f} (input ${o['input_cost_usd']:.4f})  "
                  f"api p50 {o['api_ms_p50']:.0f} ms")
        print(f"  filtered also right on tasks 'all' got right: {s['paired_vs_all']['filtered']}")
        e = s["estimate_check"]
        print(f"  countTokens vs billed prompt: {e['count_tokens_exact_matches']}/{e['count_tokens_calls_compared']} "
              f"exact (max diff {e['count_tokens_max_abs_diff']}); a count takes ~{e['count_tokens_ms_p50']} ms; "
              f"chars/4 is {e['real_over_estimated_input_tokens_avg']}x low; select_tools p50 "
              f"{e['select_tools_ms_p50']} ms (max {e['select_tools_ms_max']})")
        for m in s["misses"]:
            print(f"    miss [{m['mode']}] {m['outcome']}: {m['task'][:60]} -> {m['tool_called']} "
                  f"(needed {m['needed']}) {m['text'][:120]}")
    if "compaction" in out:
        c = out["compaction"]
        base = c.get("none", {}).get("cost_usd")
        for mode, r in c.items():
            vs = f" ({100 * (r['cost_usd'] - base) / base:+.1f}% vs none)" if base and mode != "none" else ""
            L = r["latency"]
            print(f"  compaction {mode:22} cost ${r['cost_usd']:.4f}{vs}  input ${r['input_cost_usd']:.4f}  "
                  f"cached {r['cached_share_percent']}%  total p50 {L['total_ms_p50']:.0f} / p95 "
                  f"{L['total_ms_p95']:.0f} ms  tonst max {L['local_ms_max']} ms")
            if "fact_recall" in r:
                fr = r["fact_recall"]
                lost = f", lost: {', '.join(fr['lost'])}" if fr["lost"] else ""
                print(f"      facts {fr['facts_kept']}/{fr['facts_in_summarized_turns']}{lost}; "
                      f"observed cache hit rate {r.get('observed_cache_hit_rate')}; background {r['background']}")
            if "summarizer" in r:
                sm = r["summarizer"]
                print(f"      summarizer {sm['model']}: {sm['calls']} calls, {sm['failures']} failed, "
                      f"${sm['cost_usd']:.4f} (included)")

    _save(out)
    print("\nFull per-call results: live_test_gemini_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
