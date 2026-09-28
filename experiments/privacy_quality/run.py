"""
run.py -- does masking sensitive data change how good the AI's answers are?

For every case in cases.py and every provider, this sends the prompt:

  original      as-is, straight to the provider (the baseline)
  hash          through tonst, [[NAME_3f2a91c0]]-style placeholders
  readable      through tonst, [[NAME_1]]-style placeholders
  readable_hint readable + one sentence telling the model the [[...]]
                tokens are placeholders to keep verbatim
  readable_hint_extra  readable_hint + ACCOUNT_ID and MONEY redaction

and then measures, per variant:

  leakage       share of the case's sensitive values that still reached the
                provider (by type), from the exact text that was sent
  restoration   answers with a placeholder left over after restoring
                ([[...]] or a bare NAME_1 the model un-bracketed)
  expected      share of must-have values (the customer's name, the right
                total, the right invoice number) in the final answer
  judge         a blind side-by-side: the original answer vs. the masked
                one, same provider, order randomised, scored 1-10 each
  tokens / cost from the providers' own usage numbers

Usage (from the repo root, with ANTHROPIC_API_KEY and GEMINI_API_KEY in .env):

  python experiments/privacy_quality/run.py --fake --limit 5     # offline dry run, no API calls
  python experiments/privacy_quality/run.py --limit 10           # small paid run (asks first)
  python experiments/privacy_quality/run.py                      # full run (asks first)

Results go to experiments/privacy_quality/results/ (gitignored): one JSONL
line per call (so an interrupted run resumes where it stopped) and a
summary JSON + Markdown report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def _load_dotenv():
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
# Hugging Face download bars redraw over the "Proceed?" question and hide it.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from tonst import TonstClient  # noqa: E402
from cases import CASES  # noqa: E402

# ---------------------------------------------------------------- config

# secret_notice stays off (the library default): it appends a line meant for
# the app's UI, and the judge graded it as a stray artifact in the answer
# (2026-09-27, 12-case run). Apps show report.secrets_withheld instead.
VARIANTS = {
    "original": None,
    "hash": {"style": "hash", "extra": (), "hint": False},
    "readable": {"style": "readable", "extra": (), "hint": False},
    "hash_hint": {"style": "hash", "extra": (), "hint": True},
    "readable_hint": {"style": "readable", "extra": (), "hint": True},
    "readable_hint_extra": {"style": "readable", "extra": ("ACCOUNT_ID", "MONEY"), "hint": True},
}
# The no-hint variants were settled by the 2026-09-27 run (Claude needs the
# hint); the default now compares the two hinted styles. Pass --variants to
# run any of the others.
DEFAULT_VARIANTS = ["original", "hash_hint", "readable_hint"]

from tonst.placeholders import PLACEHOLDER_HINT  # noqa: E402  (the same note TonstClient sends)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODELS = {"anthropic": "claude-sonnet-4-6", "gemini": "gemini-3.8-flash"}
DEFAULT_JUDGE = "claude-sonnet-4-6"

# USD per million tokens (input, output); used only to print dollar figures.
PRICES = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.1-flash-lite": (0.25, 1.50),
}

JUDGE_PROMPT = """You are grading two answers to the same request. Judge only how well each answer does the task: \
correctness (right people, numbers, IDs, facts from the request), completeness, and usefulness. Ignore small \
wording differences. An answer that contains leftover template tokens like [[NAME_1]] or NAME_1 instead of a real \
value, or that uses the wrong name/number, should lose points.

REQUEST:
<<<
{request}
>>>

ANSWER A:
<<<
{a}
>>>

ANSWER B:
<<<
{b}
>>>

Reply with JSON only: {{"score_a": <1-10>, "score_b": <1-10>, "better": "A" | "B" | "tie", "reason": "<one sentence>"}}"""

PLACEHOLDER_SHAPED_RE = re.compile(r"\[\[\s*[A-Z][A-Z0-9_]*?_(?:\d{1,6}|[0-9a-f]{8})(?:\.(?:first|last))?\s*\]\]")
BARE_PLACEHOLDER_RE = re.compile(r"\b[A-Z][A-Z_]*?_(?:\d{1,6}|[0-9a-f]{8})\b")
LABELS = ("NAME", "EMAIL", "DOMAIN", "PHONE", "EMPLOYER", "CODENAME", "SECRET", "ACCOUNT_ID", "MONEY", "CREDIT_CARD",
          "SSN_LIKE", "IP_ADDRESS", "PII")


# ---------------------------------------------------------------- providers

class Usage(dict):
    pass


def _post_json(url, headers, body, timeout=180):
    import requests
    for attempt in range(6):
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        if resp.status_code in (429, 500, 502, 503, 504, 529):
            wait = 2 ** attempt * 2
            print(f"      (API {resp.status_code}, retrying in {wait}s)", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()
    raise RuntimeError("API kept failing with 429/5xx; try again later")


def call_anthropic(model, prompt, max_tokens=1200):
    body = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    headers = {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    data = _post_json(ANTHROPIC_URL, headers, body)
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage", {})
    return text, {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0)}


def call_gemini(model, prompt, max_tokens=1200):
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens * 3, "thinkingConfig": {"thinkingLevel": "low"}},
    }
    headers = {"x-goog-api-key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
               "content-type": "application/json"}
    try:
        data = _post_json(GEMINI_URL.format(model=model), headers, body)
    except RuntimeError as exc:
        if "thinking" not in str(exc).lower():
            raise
        body["generationConfig"].pop("thinkingConfig")
        data = _post_json(GEMINI_URL.format(model=model), headers, body)
    cands = data.get("candidates") or [{}]
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = data.get("usageMetadata", {})
    out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return text, {"input": u.get("promptTokenCount", 0), "output": out}


def fake_call(model, prompt, max_tokens=1200):
    """Offline stand-in: 'answers' by echoing the request, so placeholders
    flow through and restoration is exercised. No network."""
    text = "Draft:\n" + prompt.replace(PLACEHOLDER_HINT, "")
    return text, {"input": len(prompt) // 4, "output": len(text) // 4}


def fake_judge(model, prompt, max_tokens=300):
    return json.dumps({"score_a": 7, "score_b": 7, "better": "tie", "reason": "fake judge"}), {"input": 0, "output": 0}


PROVIDER_FNS = {"anthropic": call_anthropic, "gemini": call_gemini}


# ---------------------------------------------------------------- scoring

def _numbers(text):
    out = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def expect_hit(expected: str, answer: str) -> bool:
    if expected.startswith(("$", "₹", "€", "£")):
        try:
            target = float(re.sub(r"[^\d.]", "", expected))
        except ValueError:
            return expected in answer
        return any(abs(n - target) < 0.011 for n in _numbers(answer))
    return expected.lower() in answer.lower()


def leftovers(answer: str) -> list:
    # Only tonst-shaped tokens count. Models also write their own
    # [[YOUR NAME]] template fields and bash has [[ -z "$X" ]]; those are
    # not restoration failures (full run 2026-09-27: 9 of 11 flagged cases
    # were exactly that).
    found = PLACEHOLDER_SHAPED_RE.findall(answer)
    rest = PLACEHOLDER_SHAPED_RE.sub(" ", answer)
    for m in BARE_PLACEHOLDER_RE.findall(rest):
        if m.split("_")[0] in LABELS or any(m.startswith(label + "_") for label in LABELS):
            found.append(m)
    return found


def leaked(sensitive, sent: str) -> list:
    low = sent.lower()
    return [(v, t) for v, t in sensitive if v.lower() in low]


# ---------------------------------------------------------------- running

def run_one(case, provider, variant, model, call, backend):
    cfg = VARIANTS[variant]
    t0 = time.perf_counter()
    if cfg is None:
        # Without tonst: the prompt as written (for RAG cases, every
        # retrieved chunk followed by the question).
        sent = case["prompt"]
        answer, usage = call(model, sent)
        raw = answer
        report = None
    else:
        captured = {}

        def call_fn(p):
            captured["sent"] = p
            text, u = call(model, p)
            captured["raw"], captured["usage"] = text, u
            return text

        client = TonstClient(
            call_fn=call_fn,
            redaction_backend=backend,
            placeholder_style=cfg["style"],
            extra_redaction=cfg["extra"],
            placeholder_hint=cfg["hint"],
        )
        if "rag" in case:
            rag = case["rag"]
            answer, rep = client.query_rag(rag["question"], rag["chunks"], top_k=rag.get("top_k"))
        else:
            answer, rep = client.query(case["prompt"])
        sent, raw, usage = captured["sent"], captured["raw"], captured["usage"]
        report = {"redacted_fields": getattr(rep, "redacted_fields", None),
                  "redacted_types": getattr(rep, "redacted_types", None),
                  "secrets_withheld": getattr(rep, "secrets_withheld", None),
                  "chunks_in": getattr(rep, "chunks_in", None), "chunks_sent": getattr(rep, "chunks_sent", None)}
    ms = (time.perf_counter() - t0) * 1000
    leaks = leaked(case["sensitive"], sent) if cfg is not None else []
    return {
        "case": case["id"], "category": case["category"], "provider": provider, "model": model,
        "variant": variant, "sent": sent, "raw_answer": raw, "answer": answer, "usage": usage,
        "ms": round(ms), "leaks": leaks, "n_sensitive": len(case["sensitive"]),
        "leftovers": leftovers(answer) if cfg is not None else [],
        "expect_hits": [expect_hit(e, answer) for e in case["expect"]],
        "report": report,
    }


def judge_pair(case, orig, masked, judge_model, judge_call, rng):
    flip = rng.random() < 0.5
    a, b = (masked, orig) if flip else (orig, masked)
    prompt = JUDGE_PROMPT.format(request=case["prompt"], a=a["answer"], b=b["answer"])
    text, usage = judge_call(judge_model, prompt, 300)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        v = json.loads(m.group(0)) if m else {}
        sa, sb, better = float(v["score_a"]), float(v["score_b"]), v.get("better", "tie")
    except (ValueError, KeyError, TypeError):
        return {"error": text[:200], "usage": usage}
    s_orig, s_mask = (sb, sa) if flip else (sa, sb)
    if better == "tie":
        verdict = "tie"
    else:
        masked_is = "A" if flip else "B"
        verdict = "masked_better" if better == masked_is else "original_better"
    return {"case": case["id"], "provider": masked["provider"], "variant": masked["variant"],
            "score_original": s_orig, "score_masked": s_mask, "verdict": verdict,
            "reason": v.get("reason", ""), "usage": usage, "judge_model": judge_model}


def preflight_backend(backend: str) -> bool:
    """Refuse to run if the chosen redaction backend isn't actually working --
    otherwise the masked variants silently measure regex-only redaction."""
    if backend != "gliner":
        return True
    from tonst.gliner_redact import GlinerRedactor

    probe = "Priya Nair from Veltrix Logistics called about Project Bluefin."
    red = GlinerRedactor()
    res = red.redact(probe)
    if res.model_available and res.entities_found > 0:
        print(f"GLiNER check OK: {res.redacted_text}")
        return True
    print("GLiNER check FAILED: it found no names/companies in a test sentence, so masked results would be")
    print("regex-only. Fix GLiNER first, or run with --backend regex to measure regex redaction on purpose.")
    if red.last_error is not None:
        import traceback
        traceback.print_exception(type(red.last_error), red.last_error, red.last_error.__traceback__)
    return False


def judge_model_for(provider: str, judge_arg: str, models: dict) -> str:
    """--judge-model cross: each provider's answers are graded by the OTHER
    provider's model, so neither grades its own work (models tend to prefer
    their own style). Otherwise one model grades everything."""
    if judge_arg == "cross":
        return models["gemini"] if provider == "anthropic" else models["anthropic"]
    return judge_arg


def call_for_model(model: str):
    return call_gemini if model.startswith("gemini") else call_anthropic


def estimate_cost(cases, providers, variants, models, judge, judge_model):
    avg_in = sum(len(c["prompt"]) for c in cases) / max(1, len(cases)) / 3.5 + 20
    avg_out = 380
    total = 0.0
    lines = []
    for p in providers:
        pin, pout = PRICES.get(models[p], (3.0, 15.0))
        n = len(cases) * len(variants)
        c = n * (avg_in * pin + avg_out * pout) / 1e6
        total += c
        lines.append(f"  {p} ({models[p]}): {n} calls ~ ${c:.2f}")
    if judge:
        for p in providers:
            jm = judge_model_for(p, judge_model, models)
            pin, pout = PRICES.get(jm, (3.0, 15.0))
            n = len(cases) * (len([v for v in variants if v != "original"]))
            # Gemini judges also spend thinking tokens (billed as output)
            out = 90 + (250 if jm.startswith("gemini") else 0)
            c = n * ((avg_in + 2 * avg_out + 250) * pin + out * pout) / 1e6
            total += c
            lines.append(f"  judge of {p} answers ({jm}): {n} calls ~ ${c:.2f}")
    return total, lines


def summarise(records, judgments):
    by = defaultdict(list)
    for r in records:
        by[(r["provider"], r["variant"])].append(r)
    orig_tokens = {(r["provider"], r["case"]): r["usage"]["input"] for r in records if r["variant"] == "original"}
    jby = defaultdict(list)
    for j in judgments:
        if "verdict" in j:
            jby[(j["provider"], j["variant"])].append(j)
    rows = []
    for (provider, variant), rs in sorted(by.items()):
        n = len(rs)
        exp = [h for r in rs for h in r["expect_hits"]]
        row = {
            "provider": provider, "variant": variant, "cases": n,
            "expected_hit_rate": round(sum(exp) / len(exp), 3) if exp else None,
            "cost_usd": 0.0,
        }
        pin, pout = PRICES.get(rs[0]["model"], (0, 0))
        row["cost_usd"] = round(sum(r["usage"]["input"] * pin + r["usage"]["output"] * pout for r in rs) / 1e6, 4)
        if variant != "original":
            tot_sens = sum(r["n_sensitive"] for r in rs)
            leak_by_type = defaultdict(lambda: [0, 0])
            for r in rs:
                case = next(c for c in CASES if c["id"] == r["case"])
                leaked_vals = {v for v, _ in r["leaks"]}
                for v, t in case["sensitive"]:
                    leak_by_type[t][1] += 1
                    if v in leaked_vals:
                        leak_by_type[t][0] += 1
            row["leak_rate"] = round(sum(len(r["leaks"]) for r in rs) / tot_sens, 3) if tot_sens else 0
            row["leak_rate_by_type"] = {t: round(a / b, 2) for t, (a, b) in sorted(leak_by_type.items())}
            # recomputed from the saved answer so an improved check applies to old runs too
            row["restoration_failures"] = sum(1 for r in rs if leftovers(r["answer"]))
            deltas = [r["usage"]["input"] - orig_tokens[(provider, r["case"])] for r in rs
                      if (provider, r["case"]) in orig_tokens]
            row["avg_input_token_delta"] = round(sum(deltas) / len(deltas), 1) if deltas else None
            # Savings on the long, redundant "heavy" cases, where tonst's
            # trimming / RAG de-duplication apply (vs the same cases sent as-is).
            heavy = [r for r in rs if r["category"] == "heavy" and (provider, r["case"]) in orig_tokens]
            if heavy:
                orig_by_case = {r["case"]: r for r in records
                                if r["variant"] == "original" and r["provider"] == provider}
                o_in = sum(orig_by_case[r["case"]]["usage"]["input"] for r in heavy)
                m_in = sum(r["usage"]["input"] for r in heavy)
                o_cost = sum(orig_by_case[r["case"]]["usage"]["input"] * pin
                             + orig_by_case[r["case"]]["usage"]["output"] * pout for r in heavy)
                m_cost = sum(r["usage"]["input"] * pin + r["usage"]["output"] * pout for r in heavy)
                row["heavy_cases"] = len(heavy)
                row["heavy_input_tokens_change_pct"] = round(100 * (m_in - o_in) / o_in, 1) if o_in else None
                row["heavy_cost_change_pct"] = round(100 * (m_cost - o_cost) / o_cost, 1) if o_cost else None
            js = jby.get((provider, variant), [])
            ho = [r for r in rs if r["category"] == "holdout"]
            if ho:
                tot = sum(r["n_sensitive"] for r in ho)
                hexp = [h for r in ho for h in r["expect_hits"]]
                row["holdout_cases"] = len(ho)
                row["holdout_leak_rate"] = round(sum(len(r["leaks"]) for r in ho) / tot, 3) if tot else 0
                row["holdout_leaks"] = sorted({f"{t}: {v}" for r in ho for v, t in r["leaks"]})
                row["holdout_expected_hit_rate"] = round(sum(hexp) / len(hexp), 3) if hexp else None
                hoj = [j for j in js if j["case"].startswith("holdout.")]
                if hoj:
                    row["holdout_judge_original"] = round(sum(j["score_original"] for j in hoj) / len(hoj), 2)
                    row["holdout_judge_masked"] = round(sum(j["score_masked"] for j in hoj) / len(hoj), 2)
            hj = [j for j in js if j["case"].startswith("heavy.")]
            if hj:
                row["heavy_judge_original"] = round(sum(j["score_original"] for j in hj) / len(hj), 2)
                row["heavy_judge_masked"] = round(sum(j["score_masked"] for j in hj) / len(hj), 2)
            if js:
                row["judged"] = len(js)
                row["judge_score_original"] = round(sum(j["score_original"] for j in js) / len(js), 2)
                row["judge_score_masked"] = round(sum(j["score_masked"] for j in js) / len(js), 2)
                row["masked_better"] = sum(j["verdict"] == "masked_better" for j in js)
                row["tie"] = sum(j["verdict"] == "tie" for j in js)
                row["original_better"] = sum(j["verdict"] == "original_better" for j in js)
        rows.append(row)
    return rows


def markdown(rows):
    out = ["| provider | variant | expected values | leak rate | restore failures | judge (orig → masked) | "
           "masked better / tie / worse | Δ input tokens | cost |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        judge = (f"{r['judge_score_original']} → {r['judge_score_masked']}" if "judged" in r else "—")
        wtl = (f"{r['masked_better']} / {r['tie']} / {r['original_better']}" if "judged" in r else "—")
        out.append(
            f"| {r['provider']} | {r['variant']} | {r['expected_hit_rate']:.0%} | "
            f"{r.get('leak_rate', 0):.0%} | {r.get('restoration_failures', '—')} | {judge} | {wtl} | "
            f"{r.get('avg_input_token_delta', '—')} | ${r['cost_usd']:.2f} |"
        )
    heavy_rows = [r for r in rows if "heavy_cases" in r]
    if heavy_rows:
        out += ["", "Heavy workloads only (long threads, logs, transcripts, RAG with duplicate chunks), "
                "vs the same prompts sent as-is:", "",
                "| provider | variant | cases | input tokens | cost | judge (orig → masked) |", "|---|---|---|---|---|---|"]
        for r in heavy_rows:
            hj = (f"{r['heavy_judge_original']} → {r['heavy_judge_masked']}" if "heavy_judge_masked" in r else "—")
            out.append(f"| {r['provider']} | {r['variant']} | {r['heavy_cases']} | "
                       f"{r['heavy_input_tokens_change_pct']:+.1f}% | {r['heavy_cost_change_pct']:+.1f}% | {hj} |")
    ho_rows = [r for r in rows if "holdout_cases" in r]
    if ho_rows:
        out += ["", "Hold-out cases only (new people, companies and formats not used while tuning the rules):", "",
                "| provider | variant | cases | leak rate | must-have values | judge (orig → masked) | what leaked |",
                "|---|---|---|---|---|---|---|"]
        for r in ho_rows:
            hj = (f"{r['holdout_judge_original']} → {r['holdout_judge_masked']}"
                  if "holdout_judge_masked" in r else "—")
            ev = f"{r['holdout_expected_hit_rate']:.0%}" if r.get("holdout_expected_hit_rate") is not None else "—"
            leaks = "; ".join(r["holdout_leaks"]) or "nothing"
            out.append(f"| {r['provider']} | {r['variant']} | {r['holdout_cases']} | {r['holdout_leak_rate']:.0%} | "
                       f"{ev} | {hj} | {leaks} |")
    types = sorted({t for r in rows for t in r.get("leak_rate_by_type", {})})
    if types:
        out += ["", "Leak rate by type (share of values of that type that reached the provider):", "",
                "| provider | variant | " + " | ".join(types) + " |", "|---|---|" + "---|" * len(types)]
        for r in rows:
            if "leak_rate_by_type" in r:
                out.append(f"| {r['provider']} | {r['variant']} | " +
                           " | ".join(f"{r['leak_rate_by_type'].get(t, 0):.0%}" for t in types) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--providers", default="anthropic,gemini")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N cases (spread across categories)")
    ap.add_argument("--categories", default=None, help="comma-separated category filter")
    ap.add_argument("--backend", default="gliner", choices=["gliner", "regex", "ollama"],
                    help="redaction backend for masked variants (gliner catches names/companies)")
    ap.add_argument("--anthropic-model", default=DEFAULT_MODELS["anthropic"])
    ap.add_argument("--gemini-model", default=DEFAULT_MODELS["gemini"])
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE,
                    help="model that grades answers, e.g. claude-sonnet-4-6 or gemini-3.8-flash; "
                         "'cross' = Gemini grades Claude's answers and Claude grades Gemini's")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--fake", action="store_true", help="offline dry run: no API calls, no cost")
    ap.add_argument("--yes", action="store_true", help="don't ask before spending")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--run-name", default=None, help="results file prefix (default: timestamp, or 'fake')")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v}; choose from {list(VARIANTS)}")
    judge = not args.no_judge
    if judge and "original" not in variants:
        variants = ["original"] + variants
    models = {"anthropic": args.anthropic_model, "gemini": args.gemini_model}

    cases = CASES
    if args.categories:
        cats = set(args.categories.split(","))
        cases = [c for c in cases if c["category"] in cats]
    if args.limit:
        # round-robin across categories so a small run still covers every kind of task
        by_cat = defaultdict(list)
        for c in cases:
            by_cat[c["category"]].append(c)
        picked, i = [], 0
        while len(picked) < min(args.limit, len(cases)):
            for cat in by_cat:
                if i < len(by_cat[cat]) and len(picked) < args.limit:
                    picked.append(by_cat[cat][i])
            i += 1
        cases = picked

    if any(VARIANTS[v] for v in variants) and not preflight_backend(args.backend):
        return 3

    if args.fake:
        calls = {p: fake_call for p in providers}
        judge_call = fake_judge
    else:
        missing = [k for k, p in (("ANTHROPIC_API_KEY", "anthropic"), ("GEMINI_API_KEY", "gemini"))
                   if p in providers and not os.environ.get(k) and not (k == "GEMINI_API_KEY" and os.environ.get("GOOGLE_API_KEY"))]
        if judge:
            jms = {judge_model_for(p, args.judge_model, models) for p in providers}
            if any(not m.startswith("gemini") for m in jms) and not os.environ.get("ANTHROPIC_API_KEY"):
                missing.append("ANTHROPIC_API_KEY (judge)")
            if any(m.startswith("gemini") for m in jms) and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                missing.append("GEMINI_API_KEY (judge)")
        if missing:
            print("Missing API keys:", ", ".join(missing), "- put them in .env at the repo root.")
            return 2
        calls = {p: PROVIDER_FNS[p] for p in providers}
        judge_call = None  # chosen per judge model below
        total, lines = estimate_cost(cases, providers, variants, models, judge, args.judge_model)
        print(f"{len(cases)} cases x {len(variants)} variants x {len(providers)} providers "
              f"(masking backend: {args.backend})")
        print("\n".join(lines))
        print(f"  estimated total ~ ${total:.2f} (list prices; already-finished calls in a resumed run are skipped)")
        sys.stdout.flush()
        if not args.yes and input("\nProceed? Type y and press Enter [y/N]: ").strip().lower() != "y":
            return 1

    os.makedirs(args.out, exist_ok=True)
    name = args.run_name or ("fake" if args.fake else time.strftime("%Y%m%d-%H%M%S"))
    if args.fake and not args.run_name:
        # A plain --fake run always starts fresh: resuming would mix answers
        # produced by older code into the tables (seen 2026-09-27).
        for f in os.listdir(args.out):
            if f.startswith("fake."):
                os.remove(os.path.join(args.out, f))
    calls_path = os.path.join(args.out, f"{name}.calls.jsonl")
    # Judgments are stored per judge model, so re-running an existing run
    # with a different --judge-model re-grades the saved answers (no new
    # answer calls) and the two judges can be compared.
    jtag = "" if args.judge_model == DEFAULT_JUDGE else "-" + re.sub(r"[^A-Za-z0-9.]+", "-", args.judge_model)
    judge_path = os.path.join(args.out, f"{name}.judge{jtag}.jsonl")
    done = {}
    if os.path.exists(calls_path):
        with open(calls_path) as fh:
            for line in fh:
                r = json.loads(line)
                done[(r["case"], r["provider"], r["variant"])] = r
    judged = {}
    if os.path.exists(judge_path):
        with open(judge_path) as fh:
            for line in fh:
                j = json.loads(line)
                if "verdict" in j:
                    judged[(j["case"], j["provider"], j["variant"])] = j

    rng = random.Random(args.seed)
    records = list(done.values())
    judgments = list(judged.values())
    errors = 0
    with open(calls_path, "a") as cf, open(judge_path, "a") as jf:
        for n, case in enumerate(cases, 1):
            for provider in providers:
                for variant in variants:
                    key = (case["id"], provider, variant)
                    if key in done:
                        continue
                    try:
                        rec = run_one(case, provider, variant, models[provider], calls[provider], args.backend)
                    except Exception as exc:  # keep going; the run can be resumed
                        errors += 1
                        print(f"  ! {key}: {exc}", flush=True)
                        continue
                    done[key] = rec
                    records.append(rec)
                    cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cf.flush()
                if judge and (case["id"], provider, "original") in done:
                    for variant in variants:
                        key = (case["id"], provider, variant)
                        if variant == "original" or key in judged or key not in done:
                            continue
                        try:
                            jm = judge_model_for(provider, args.judge_model, models)
                            j = judge_pair(case, done[(case["id"], provider, "original")], done[key],
                                           jm, judge_call or call_for_model(jm), rng)
                        except Exception as exc:
                            errors += 1
                            print(f"  ! judge {key}: {exc}", flush=True)
                            continue
                        j.setdefault("case", case["id"])
                        j.setdefault("provider", provider)
                        j.setdefault("variant", variant)
                        judged[key] = j
                        judgments.append(j)
                        jf.write(json.dumps(j, ensure_ascii=False) + "\n")
                        jf.flush()
            print(f"[{n}/{len(cases)}] {case['id']}", flush=True)

    rows = summarise([r for r in records if r["case"] in {c["id"] for c in cases}],
                     [j for j in judgments if j.get("case") in {c["id"] for c in cases}])
    judge_cost = 0.0
    if judge and not args.fake:
        judge_cost = sum(
            (j["usage"]["input"] * PRICES.get(j.get("judge_model", ""), (3.0, 15.0))[0]
             + j["usage"]["output"] * PRICES.get(j.get("judge_model", ""), (3.0, 15.0))[1])
            for j in judgments if "usage" in j) / 1e6
    summary = {"run": name, "cases": len(cases), "providers": providers, "variants": variants,
               "backend": args.backend, "models": models, "judge_model": args.judge_model if judge else None,
               "judge_cost_usd": round(judge_cost, 4), "errors": errors, "rows": rows}
    with open(os.path.join(args.out, f"{name}{jtag}.summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    md = markdown(rows)
    with open(os.path.join(args.out, f"{name}{jtag}.report.md"), "w") as fh:
        fh.write(f"# Answer-quality benchmark: {name}\n\n{len(cases)} cases, backend {args.backend}, "
                 f"judge {args.judge_model if judge else 'off'}\n\n{md}\n")
    print()
    print(md)
    if judge and not args.fake:
        print(f"\njudge cost ~ ${judge_cost:.2f}")
    if errors:
        print(f"\n{errors} calls failed; re-run the same command with --run-name {name} to retry just those.")
    print(f"\nSaved to {args.out}/{name}.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
