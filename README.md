![tests](https://github.com/v-nightwolf/tonst/actions/workflows/tests.yml/badge.svg)

# tonst — Token Optimization & Security Tool (Proof of Concept)

A drop-in wrapper around any paid LLM API call that reduces token spend and
keeps sensitive data off the cloud model, by doing the work **locally**
before the request ever leaves the machine.

```
Your app
   │
   ▼
TonstClient.query(prompt)
   │
   ├─ 1. Local PII redaction (regex always-on by default, optionally + a
   │     free-text pass -- GLiNER or Ollama, your choice -- see
   │     "Why enhanced redaction matters" below)
   ├─ 2. Mechanical trim (dedupe, whitespace, history truncation)
   ├─ 3. Optional local-model compression (Ollama, off by default)
   │
   ▼
Your existing paid API call (Claude/OpenAI/etc.) — only sees the
trimmed, redacted prompt
   │
   ▼
Re-insert real values into the response → return to your app
```

For requests built from reusable parts (a system prompt, reference docs,
tool descriptions) rather than one flat string, `TonstClient.query_structured()`
and `cache_structuring.py` shape the request so a **real provider's own
prompt caching** can discount the repeated portion — see "Prompt-caching
structuring" below.

For chat-style conversations tracked as a list of `{"role", "content"}`
turns, `TonstClient.query_messages()` caps history to a sliding window
and, optionally, condenses what falls outside it into a summary instead
of dropping it outright — see "History compaction" below. Pass a
`RollingSummary` to make that compaction incremental and cache-friendly
— see "Rolling compaction".

For agents and tool-calling apps, `tool_optimizer.py` cuts the tokens
spent on tool/MCP definitions (Anthropic's native deferred loading, or
local relevance filtering for any provider) — see "Tool and MCP
definition optimization". For RAG pipelines, `TonstClient.query_rag()`
drops duplicate and (optionally) low-relevance retrieved chunks before
they reach the prompt — see "RAG context optimization". And every call
can be logged to a local savings file you summarize with `tonst stats` —
see "Savings log".

## Whitepaper

A full technical whitepaper — methodology, benchmarks, and real measured
cost/privacy results — is available:

- **Live version:** [Beyond the Prompt](https://claude.ai/artifact/2hcKTcfwBzAWev1PUGRv2x)
- **Permanent citable record (DOI):** [10.5281/zenodo.22745266](https://doi.org/10.5281/zenodo.22745266)

## Real results (not simulated)

Across **384 live API benchmark iterations** spanning 6 enterprise verticals (Medical, Space, Electronics, Finance, IT, Legal), `tonst` cuts prompt payload volume by **up to 23.68% locally** (running the full pipeline: GLiNER redaction + mechanical trim + compression) and drives a **53.00% net reduction in API cost** via provider prompt caching—all while maintaining **100.0% structured PII recall with zero privacy leaks**.

| Mechanism | Scope & Scale | Peak Savings | Workload Average | Key Reliability / Safety Metric |
|---|---|---|---|---|
| **Local Trim + GLiNER Redaction + Compression** | 360 Runs (`claude-3-5-sonnet-20241022` pricing baseline) | **23.68% token drop** (Space) | **21.09% token drop** | 100.0% Structured PII Recall (0 leaks), 87.78% Free-Text PII Recall |
| **Provider Prompt Caching** | 24 Calls (`gemini-3.1-flash-lite`) | **72.00% read discount** | **53.00% net cost drop** | Zero cache-key leakage |
| **Tool / MCP definition filtering** (`select_tools`, top 5 of 36) | 30 tasks × 2 providers (`claude-sonnet-4-6`, `gemini-3.8-flash`) | **−70% cost** on direct tasks (Gemini) | **−51% (Claude) / −55% (Gemini) cost** | Same success as sending all tools: 30/30 Claude, 29/30 Gemini (same miss either way) |
| **Rolling history compaction** (background summaries) | 20–24-turn support chats, both providers | **−15.1% cost, −43% tokens** (Gemini, 20 turns) | −2.7% (Claude + Haiku, 24 turns, cache already cheap) | 8/8 facts kept with Haiku / Flash-Lite summaries; no added latency |

---

### Free features on live APIs (Claude Sonnet 4.6 + Gemini 3.8 Flash, September 2026)

These are real API calls with billed usage, from `live_test_free_features.py`
(Anthropic) and `live_test_gemini.py` (Gemini). The same 30 tool tasks and
the same scripted support conversation were used on both providers. Details
and every intermediate run are in "Tool and MCP definition optimization",
"Rolling compaction" and ROADMAP.md.

**Tool filtering: 36 tool definitions, 30 tasks (24 direct, 6 paraphrased)**

| | Claude Sonnet 4.6 | Gemini 3.8 Flash (thinking low) |
|---|---|---|
| Success, all tools sent | 30/30 | 29/30 |
| Success, `select_tools` (top 5) | **30/30** | **29/30** (29/29 of the tasks "all" got right) |
| Cost, all → filtered | $0.40 → **$0.19 (−51%)** | $0.0738 → **$0.0331 (−55%)** |
| Direct tasks only | −64% | −70% |
| Anthropic deferred loading (tool search) | 29/30, −18% | n/a (Anthropic-only) |

- Paraphrased requests that tonst isn't confident about fall back to sending every tool: no saving, but no risk either.
- Gemini's implicit cache never hit on these prompts (~2.9k tokens, under Flash's 4,096-token minimum), so filtering was the only saving available there.

**Rolling compaction: long support chat (~500 tokens of tool output per reply)**

| | Turns | No compaction | With compaction | Facts kept |
|---|---|---|---|---|
| Gemini 3.8 Flash, Flash-Lite summaries | 20 | $0.1329 | **$0.1128 (−15.1%)**, tokens −43% | 8/8 |
| Gemini, cache-aware (observed hit rate) | 20 | $0.1329 | $0.1206 (−9.3%) | 8/8 |
| Claude Sonnet 4.6, Haiku summaries | 24 | $0.186 | $0.181 (−2.7%), tokens −41% | 8/8 |
| Claude, local gemma2:2b summaries | 24 | $0.186 | $0.189 (+1.6%) | 5/8 |
| Claude, cache-aware, short chat | 12 | $0.0719 | $0.0719 (0%; summarizing anyway: +14.9%) | — |

- **Claude:** prompt caching already makes old history cheap (0.1× on reads), so compaction is near break-even until ~20 turns. The offline simulation puts it at −18% by 40 turns and −55% by 100.
- **Gemini:** implicit caching is best-effort. It served 0% of this chat until ~18k tokens, so compaction paid off sooner.
- **Cache-aware mode** reads the cache hit rate the provider actually reports and adjusts to either case.

**Accuracy and latency**

| | Claude | Gemini |
|---|---|---|
| chars/4 estimate vs. real prompt tokens | 1.77× low (tool JSON + hidden tool prompt) | within 2% |
| Provider token-count endpoint vs. billed | 60/60 exact | 60/60 exact, ~320 ms per count |
| `select_tools()` time | — | 2.3 ms p50 (max 4.5 ms) |
| tonst's own compaction work per turn | ≤ 6 ms | ≤ 3 ms |
| Background summaries on the request path | 0 ms | 0 ms (p95 response time fell from 4.0 s to 2.7 s with smaller prompts) |
| Blocking local summary (gemma2:2b on a MacBook Air) | 4.7–12 s on the turn it runs | — |

---

### Mechanical Trimming & Privacy Benchmark (360-Iteration Suite)

To evaluate the full local pipeline (not just mechanical trimming in
isolation), `tonst` was benchmarked across the same 360-run test matrix
using Anthropic's `claude-3-5-sonnet-20241022` pricing baseline ($3.00
per 1M input tokens): six domain verticals, 60 iterations each, split
evenly between **Supervised** (strictly structured fields with explicit
PII keys) and **Unsupervised** (unstructured free-text narrative inputs
containing embedded PII, duplicate instructions, and redundant
whitespace) -- now run with the actual recommended production
configuration: `redaction_backend="gliner"` (`gliner_medium`, threshold
0.22) for free-text PII, plus `gemma3:1b` for local compression,
`--workers 1`. This replaces an earlier version of this table that only
measured mechanical trim + regex-only redaction (no enhanced backend, no
compression) -- see `research/gliner-sanity-check-findings.md` for that
superseded run's numbers. This run was produced on a Google Colab T4 GPU
instance via `tonst_gliner_full_benchmark.ipynb`, using
`benchmark_tonst.py`'s built-in mocked "paid API" call -- so the token
and PII-recall numbers below are real, but the dollar figure is still an
estimate against a hardcoded $3/M rate, not a real invoice (see
`research/colab-benchmark-findings.md`'s "Cost benchmarking" section).

| Metric | Supervised Paradigm | Unsupervised Paradigm | Total / Combined |
|---|---|---|---|
| **Iterations** | 180 | 180 | **360** |
| **Original Tokens** | 32,189 | 40,077 | **72,266** |
| **Tokens Sent to API** | 29,743 | 27,279 | **57,022** |
| **Tokens Saved** | 2,446 (**7.60%**) | 12,798 (**31.93%**) | **15,244 (21.09%)** |
| **Supervised PII Recall** | 100.0% | 100.0% | **100.0% (0 Leaks)** |
| **Free-Text PII Recall** | 77.04% | 98.52% | **87.78%** |
| **Restoration Failures** | 0 | 0 | **0** |
| **Mean Local Overhead** | 1,814.5 ms | 2,702.1 ms | **2,258.3 ms** |
| **Estimated Cost Saved** | — | — | **$0.045732 USD** |

**Industry Performance Breakdown**

| Industry | Supervised Savings | Unsupervised Savings | Overall Token Savings (%) | Total Tokens Saved |
|---|---|---|---|---|
| **Space** | 12.52% | **32.63%** | **23.68%** | 2,930 |
| **Medical** | 6.28% | 36.54% | **23.13%** | 2,794 |
| **Finance** | 9.63% | 31.71% | **21.96%** | 2,669 |
| **Legal** | 5.38% | 30.67% | **19.51%** | 2,353 |
| **IT** | 5.78% | 30.58% | **19.43%** | 2,258 |
| **Electronics** | 5.82% | 29.34% | **18.70%** | 2,240 |

**Technical Privacy & Latency Findings**
* **Zero Privacy Leaks & Perfect Restoration**: Across all 360 runs, `tonst` achieved **100.0% recall on structured PII fields** with zero unredacted values reaching the API endpoint and zero round-trip placeholder restoration failures.
* **Free-Text PII Sensitivity**: `redaction_backend="gliner"` + `gemma3:1b` compression together caught **87.78%** of free-text PII overall (77.04% supervised / 98.52% unsupervised) — up from a 57.50% regex-only baseline (the number this table showed before GLiNER was wired in; still the right comparison for "what does enabling an enhanced backend actually buy you"). The residual gap is the known, accepted GLiNER limitation on the two "supervised" prompt shapes specifically (codename recall ~58-60% there, see "Why enhanced redaction matters" above) — not a bug, and not something regex-only redaction could have caught at all.
* **Local Latency Overhead**: Running local GLiNER redaction + `gemma3:1b` compression adds a mean local processing delay of 2,258.3-2,389.8 ms before API dispatch across two independent 360-iteration Colab runs (p50: 2,022.8-2,089.1 ms, p90: 4,031.7-4,380.1 ms, p99: 5,213.7-5,624.5 ms) — measured on a Google Colab T4 instance, where GLiNER itself ran on CPU (this step doesn't use the GPU; see "Why enhanced redaction matters" above for a second measurement from different hardware that came out meaningfully faster). Token, recall, leak, and restoration numbers were bit-for-bit identical across both runs — only latency varied, consistent with shared-cloud-instance noise rather than any code change. A separate `--workers 4` run of this same pipeline (also replicated twice) confirmed correctness holds under concurrency (identical recall/leak/restoration numbers on every run) but is NOT free on latency — see `research/gliner-sanity-check-findings.md` for the full breakdown.

---

### Multi-Industry Prompt Caching Benchmark (`gemini-3.1-flash-lite`)

To evaluate real-world provider prompt caching savings, `tonst` executed a 24-call empirical test suite against Google's `gemini-3.1-flash-lite` model across 6 industry domains. Each domain was evaluated over a 4-call sequence consisting of 1 initial cache-write call followed by 3 consecutive cache-read calls, with local deterministic PII redaction applied prior to payload construction.

| Industry | Target Entity / Domain | Baseline Spend | Real Spend (`tonst`) | Net Savings (%) |
|---|---|---|---|---|
| **IT** | NimbusCloud | $0.00616 | $0.00286 | **53.60%** |
| **Space** | OrbitalVanguard Aero | $0.00622 | $0.00290 | **53.40%** |
| **Legal** | Sterling & Vance LLP | $0.00630 | $0.00297 | **52.81%** |
| **Electronics** | OmniChip Design | $0.00631 | $0.00298 | **52.77%** |
| **Medical** | Apex Clinical Trials | $0.00631 | $0.00298 | **52.73%** |
| **Finance** | Vanguard Citadel Custody | $0.00631 | $0.00299 | **52.68%** |
| **Total Workload** | **24 Total API Calls** | **$0.03761** | **$0.01768** | **53.00%** |

**Economic Lifecycle Analysis**
* **Read-Discount vs. Net Savings**: While individual cached read calls achieve a **68.00% to 72.00% token discount**, the overall workload net cost reduction stabilizes at **53.00%**. This reflects the write-premium amortized across initial cache creation.
* **Deterministic Placeholder Stability**: Because `tonst` uses deterministic hashing for PII placeholders rather than random identifiers, prompt prefixes remain byte-for-byte identical across calls, preventing cache key invalidation while keeping sensitive enterprise data off provider servers.

---

### Single-Prompt Verification Baseline

Single-prompt verification on `api.anthropic.com` demonstrates how mechanical trimming scales between minimal queries and realistic, verbose operational prompts (cost figures use `claude-sonnet-4-6`'s published standard input rate, $3 / million tokens as of this writing — verify current pricing before relying on this for real budgeting, per the same caveat that applies throughout this README):

| Metric | Minimal Clean Prompt | Realistic Bloated Prompt |
|---|---|---|
| Original tokens | 26 | 186 |
| Tokens sent to API | 26 | 136 |
| **Tokens saved** | 0 (0.0%) | **50 (26.9%)** |
| Estimated cost, no tonst | $0.000078 | $0.000558 |
| Estimated cost, with tonst | $0.000078 | $0.000408 |
| **Cost saved** | $0.00 (0.0%) | **$0.00015 (26.9%)** |
| PII fields redacted | 1 | 3 |

The 0% result on the short prompt is intentionally included here, not hidden — `tonst` doesn't manufacture savings where none exist. Real prompts with any duplication, verbose history, or repeated instructions (the overwhelming majority of real chat-app traffic) see meaningful reduction; a single already-minimal prompt does not, and shouldn't.

**Why cost tracks tokens 1:1 here, unlike prompt-caching:** This test measures plain trimming — literally sending fewer tokens at the same standard price, no discount or premium multiplier involved. Prompt caching is a meaningfully different mechanism (see "Prompt-caching structuring" below), where token count never drops and the entire saving comes from a cheaper price per token on a cache hit instead. Both are real cost reductions; they just come from different places, and `tonst` reports both correctly rather than treating "tokens saved" as a universal proxy for "money saved."

At real traffic volumes the fractions of a cent above add up: an app sending 100,000 requests/day with this same 50-token, 26.9% overhead would save an estimated **$15.00/day, or about $5,475/year**, on this one mechanical trimming pass alone — before any prompt-caching savings on top of it.

Reproduce this yourself with `real_api_demo.py` (see below).

**Live Header Verification**
* **Anthropic (`api.anthropic.com`)**: Real call execution returns `cache_read_input_tokens` and `cache_creation_input_tokens` in the raw usage response header, confirming that cache breakpoints (`cache_control`) successfully shift tokens from full input pricing ($3.00/1M) to cached read pricing ($0.30/1M).
* **Gemini (`generativelanguage.googleapis.com`)**: Live test responses confirm `cachedContentTokenCount` matching the exact token length of the prefix payload, verifying that zero cached tokens were billed at standard rates.

## Install

```bash
pip install -r requirements.txt
```

This installs the one hard dependency the code needs: `requests` (for
talking to a local Ollama instance and, in `real_api_demo.py`, the
Anthropic API directly). If you see `ModuleNotFoundError: No module
named 'requests'` when running the demo, this step was skipped — run it
and re-try.

If you plan to use `redaction_backend="gliner"` (recommended — see "Why
enhanced redaction matters" below), also install its optional extra:

```bash
pip install -e ".[gliner]"
```

This pulls in `gliner` plus its transitive ML dependencies (`torch`,
`transformers`, `huggingface_hub`) — expect a noticeably larger install
than the base package. Every other `redaction_backend` value
(`"none"`/`"regex"`/`"ollama"`) works without it.

Full packaging (once you're ready to `pip install` it as a real package):

```bash
pip install tonst
```

(Not yet published to PyPI — this POC ships as source. `pip install -e .`
from this directory already works, since `pyproject.toml` is in place;
or copy the `tonst/` folder directly into your project.)

## Why this shape

- **Nothing here requires you to run a server.** The redaction and
  trimming logic run in-process, wherever your app already runs (your own
  backend, not a third-party gateway). The only optional server-side piece
  in a real product would be a lightweight usage/billing dashboard — no
  inference, no GPUs.
- **Redaction + restoration is provider-agnostic and reversible.** Sensitive
  fields are swapped for placeholders before the call, and swapped back
  after — the cloud model never sees the real value, and your app never
  sees a placeholder.
- **Redaction placeholders are deterministic, not random.** The same PII
  value always redacts to the exact same placeholder (a hash of the value,
  not a random UUID). This is required for prompt-caching structuring:
  a stable block containing PII has to be byte-for-byte identical across
  calls for a provider's cache to recognize it as the same prefix — a
  random placeholder would silently defeat caching on every call.
- **The optional local-model step (`local_model.py`) is isolated and fails
  soft.** If Ollama isn't installed or running, the pipeline just skips
  that step rather than breaking. This matches the real-world constraint
  that not every deployment machine can run a local model well.

## Prompt-caching structuring

Provider-side prompt caching (Anthropic, OpenAI, Gemini) does **not** skip
the LLM call. It caches the *processed representation* of repeated prefix
tokens so the model doesn't reprocess them — the model still runs and
generates a fresh response every time. The discount only applies if the
request is shaped correctly:

- Stable, reused content (system instructions, reference docs, tool
  descriptions) has to come **first**, and be **byte-for-byte identical**
  across calls.
- The variable part (the actual question) goes **last**.
- On Anthropic specifically, an explicit `cache_control` marker is
  required on the last stable block — ordering alone isn't enough (unlike
  OpenAI/Gemini's automatic prefix caching, which needs only correct
  ordering).

`cache_structuring.py` provides:

- `PromptParts(system, stable_blocks, variable)` — a small container that
  keeps these apart instead of one flat string, so ordering can be
  enforced automatically instead of relying on every caller to get it
  right by hand.
- `structure_for_caching(parts)` — returns a correctly-ordered flat
  string. Works with `TonstClient.query()`'s existing `call_fn(str)`
  interface and with any provider's automatic prefix caching.
- `build_anthropic_cache_request(parts, model, ...)` — returns the actual
  Anthropic Messages API JSON body with the `cache_control` breakpoint
  placed correctly. Use this (instead of `call_fn`) when you want a real
  cache breakpoint, since that needs a structured request body that a
  flat string can't carry.
- `check_cache_eligibility(parts, model)` / the automatic warning inside
  `build_anthropic_cache_request()` — Anthropic requires a **minimum
  stable-prefix length per model** (1,024 tokens for `claude-sonnet-4-6`,
  ranging 512–4,096 across their lineup) before it will cache anything at
  all. Below that minimum there's no error — the request just silently
  processes without caching, with `cache_creation_input_tokens` and
  `cache_read_input_tokens` both coming back 0. This check exists so that
  failure is visible (a `UserWarning`) instead of a developer discovering
  it by getting no savings and not knowing why.
- `parse_anthropic_usage(response_json)` — reads the real `usage` field
  from an actual API response and reports whether a cache write or cache
  hit happened. This is the **only** ground truth for whether caching
  worked; a correctly-shaped request is necessary but not sufficient —
  see the "Testing against a real provider" section below for a live measurement against the
  real API.

```python
from tonst import TonstClient, PromptParts, build_anthropic_cache_request

client = TonstClient(call_fn=my_api_call)

parts = PromptParts(
    system="You are a customer support assistant for Acme Cloud Hosting.",
    stable_blocks=["Company policy: refunds within 30 days..."],
    variable="A customer is asking about a refund. What should I tell her?",
)

# Ordering only (works through call_fn, any provider):
response, report = client.query_structured(parts)

# Or, for a real Anthropic cache_control breakpoint, redact the parts
# first (keeping them separate) and build the request directly:
redacted = client.redact_and_trim_parts(parts)
body = build_anthropic_cache_request(redacted.parts, model="claude-sonnet-4-6")
# body is ready to pass to anthropic.messages.create(**body)
```

## History compaction

Long-running chats accumulate old turns that a fixed-size window would
otherwise just discard. `TonstClient.query_messages()` (off by default —
opt in with `use_history_compaction=True`) can condense whatever falls
outside the window into a single summary message instead of dropping it,
using the same local model (via Ollama) that `local_model.py` and
`redact_llm.py` already use.

This is the same idea as Claude Code's `/compact`, Codex CLI's and
OpenCode's history compaction — but with one deliberate difference:
**it runs on the local model, not the paid one.** Claude Code's own
compaction burns real tokens against the frontier model it's
summarizing history for — Anthropic's own docs give an example where
summarizing 180k tokens of history costs a one-time charge of 180k
input + 3.5k output tokens. Running this step locally instead means it
costs latency and local compute, never a paid token.

The honest tradeoff: a small 1–3B local model is meaningfully weaker at
faithful summarization than the frontier models those tools use for
their own compaction. This is best-effort, not a guarantee nothing
important survives — and unlike tonst's other optional local-model
steps, its fail-soft path is NOT "skip the optimization and keep
everything." If the local model is unavailable or its summary fails a
length guard rail (too long to have summarized anything, or
suspiciously short), the older turns are simply dropped with no
summary — identical to what `trim.truncate_history()` already does.
Behavior never gets worse than that pre-compaction baseline, but a
successful compaction genuinely trades some conversation memory for a
hard cap on tokens; it is not lossless.

Compaction only ever sees content that's already been redacted,
message by message, before it runs — a summarization step is a local
rewrite, and (same principle behind `redact_and_trim_parts()`) a
rewrite step must never see raw PII, only placeholders.

```python
client = TonstClient(call_fn=my_api_call, use_history_compaction=True)

response, report = client.query_messages(conversation_history, keep_last_n=6)
print(report.history_compacted, report.history_turns_dropped, report.compaction_ms)
```

`compactor.py` also exposes `HistoryCompactor` and `compact_history()`
directly if you want compaction without the rest of the pipeline.

## Rolling compaction

`compact_history()` above is stateless: every call re-summarizes **all**
the older turns from scratch. In a live chat that means a local-model
call on every turn once you're past the threshold, and a summary whose
text changes every turn — so it can never sit in the provider's cached
prompt prefix, and neither can anything after it.

Pass a `RollingSummary` (one per conversation, reused on every call,
updated in place) to switch `query_messages()` to incremental mode:

- Turns leaving the recent window are kept **verbatim** until they add
  up to `compaction_token_threshold`, then folded into the existing
  summary in **one** local-model call. Already-summarized turns are never
  re-read.
- Between folds the prompt is `[system][summary][older turns...][recent
  turns...]` and only ever grows at the end, so the provider's prefix
  cache keeps hitting — and no local model runs at all.
- The summary uses fixed headings (Goal / Decisions / Key facts / Open
  items) from the very first fold, is capped in size
  (`max_summary_chars`), and goes through the same placeholder guard
  rail as regular compaction.
- A failed fold (a timeout, say) does **not** drop anything straight
  away. The turns stay in the prompt verbatim and the fold is retried
  once another threshold's worth of text has piled up. Only after a
  second failure is that batch dropped, keeping the previous summary, so
  it's never worse than plain truncation. If the history no longer
  matches the state (edited, or a different conversation), the state
  resets instead of merging into the wrong summary.
- A fold has to *read* up to `compaction_token_threshold` tokens, which
  takes seconds on a laptop. The client's `compaction_timeout` defaults
  to 60s (other local-model steps use 8s). Live testing on a MacBook Air:
  a ~7k-token fold timed out at 8s; a ~1k-token fold took ~5s, and the
  next turn reused the summary in 0 ms. Load the model first
  (`ollama run gemma2:2b "hi"`) to avoid paying model-load time on the
  first fold.

```python
from tonst import TonstClient, RollingSummary

client = TonstClient(call_fn=my_api_call, use_history_compaction=True)
state = RollingSummary()            # store state.to_dict() with the conversation

response, report = client.query_messages(history, keep_last_n=6, rolling_state=state)
print(report.history_summary_updated, report.history_summary_reused)
```

The state holds only already-redacted text. `compact_history_rolling()`
is also usable on its own.

**Live result** (2026-09-24, `claude-sonnet-4-6`, 12-turn support chat,
gemma2:2b summaries, Anthropic multi-turn caching, same conversation in
both modes):

| | Stateless | Rolling |
|---|---|---|
| Local summaries | 4 (17.0 s) | **1 (7.2 s)** |
| Cache writes (billed 1.25×) | 2,784 | **2,078** |
| Cache reads (billed 0.1×) | 16,785 | 18,362 |
| Total input tokens | 19,605 | 20,476 (+4%) |
| Cost | $0.0185 | **$0.0163 (−12%)** |

- **The mechanism works.** With rolling compaction, cache reads grew
  every turn (1,544 → 1,818): the conversation itself was served from
  cache. Stateless reads stayed flat at the 1,524-token system prompt,
  and every turn re-wrote 80–180 tokens of history at the write price.
  The single fold (turn 7) reset the cached history once, as designed.
- **Why 12% and not more:** the 1,500-token system prompt is cached
  identically in both modes and dominates this short chat. On the
  conversation history alone, rolling was about 47% cheaper
  (~530 tokens written + ~1,600 read, vs. ~1,240 written). The gap grows
  with longer conversations and longer messages.
- **Summary quality (gemma2:2b):** the right structure and the core
  facts (order number, damage, replacement, address), but an
  already-agreed express upgrade was listed under Open items, and two
  minor facts were dropped. Fine for keeping the thread of a
  conversation; don't rely on it for exact commitments. Since then the
  prompt spells out that settled items belong under Decisions, and
  echoed `---` fence lines are stripped from summaries.
- chars/4 was within 10% of billed input for this prose conversation
  (vs. 1.77× low for JSON tool definitions).

**Cost and latency against no compaction at all** (24 turns, threshold
600, `live_test_free_features.py --part compaction`, 2026-09-24):

| | No compaction | Stateless | Rolling |
|---|---|---|---|
| Cost | $0.0295 | $0.0330 | $0.0291 |
| API time, p50 / p95 | 1.66 / 2.87 s | 1.68 / 2.31 s | 1.68 / 2.57 s |
| Total time, p50 / p95 / max | 1.66 / 2.87 / 3.06 s | 1.84 / **5.53** / 7.18 s | 1.68 / 2.73 / 6.25 s |
| Local model ran on | — | 7 turns, avg 3.2 s | 1 turn, 4.7 s |

- **tonst's own code adds ~0.1 ms per turn.** The only real added latency
  is the local model, and it lands on that turn's response, because the
  summary is made before the API call.
- **The API call wasn't slower with a longer prompt** (~1.7 s p50 in every
  mode), so compaction doesn't make the API faster either.
- **Use rolling, not stateless, whenever the provider caches prompts.**
  Stateless cost *more* than not compacting at all (it breaks the cache
  every turn) and added 2.5–5 s to 7 of 24 turns. Rolling matched no
  compaction on cost and on typical latency.
- **For short chats, compaction doesn't save money.** With prompt
  caching, a growing history of ~1,000 tokens is re-read at 10% of the
  price, which is already cheap. Compaction pays off when history gets
  long (tool outputs, long agent runs, long support threads), and it's
  needed when a conversation would overflow the model's context window.
  That's why the default `compaction_token_threshold` is 3,000 tokens:
  short chats are never compacted. To measure the long case, run
  `live_test_free_features.py --part compaction --long` (~500 tokens of
  tool output per reply, history reaching ~13k tokens, compared at the
  real 3,000-token default).

**Long history** (`--part compaction --long`, 12 turns, ~500-token
tracking-tool output in every reply, history reaching ~12,600 real tokens,
real 3,000-token threshold, 2026-09-25):

| | No compaction | Rolling | Rolling, background |
|---|---|---|---|
| Cost | $0.0719 | $0.0687 (−4%) | $0.0791 (+10%) |
| Slowest turn | 2.6 s | **13.9 s** (12.3 s summary) | 2.8 s |

- **Background summaries removed the local model from response time
  entirely.** The blocking summary took 12.3 s on a MacBook Air; in the
  background, no turn was slower than no compaction.
- **With prompt caching, the saving is on cache *reads* only.** Every new
  turn adds ~1,000 new tokens that are written at full price in every
  mode; compaction shrinks only the re-read history (billed at 10%).
  After the summary, turns were ~26% cheaper, but the summary's one-time
  cache rewrite takes ~2–3 turns to pay back. This run summarized at
  turn 9 of 12, and the background summary landed 2 turns later with a
  bigger rewrite. So for cost, compaction is a modest, long-conversation
  win. Its main job is keeping long conversations inside the model's
  context window.
- **The summary itself failed, and nothing caught it.** Given ~3k tokens
  of tool output, gemma2:2b *replied to the customer* ("You're in luck!
  I've sent you a tracking number...") instead of summarizing. That
  passed every guard rail and would have put false claims into the paid
  model's context. Fixed three ways: summaries must now use the Goal /
  Decisions / Key facts / Open items headings (at least 3 of 4) or
  they're rejected, falling back to keeping the turns verbatim; the task
  is repeated *after* the conversation in the prompt; and each message
  is clipped to 700 characters in what the summarizer reads
  (`summary_input_chars`), keeping the prose and cutting raw JSON/logs a
  2B model can't condense anyway. The paid model still sees full text
  until it's summarized.
- chars/4 was ~0.7× real tokens on this JSON-heavy history, the same
  pattern as tool definitions. Use `token_counter=` for real numbers.

**24 turns, long history** (~24k tokens by the end, same setup; after
the summary-failure fixes, 2026-09-25):

| | No compaction | Rolling | Rolling, background |
|---|---|---|---|
| Cost | $0.186 | **$0.147 (−21%)** | $0.168 (−10%) |
| Tokens sent | 315k | 159k (−50%) | 183k (−42%) |
| Total time p50 / p95 / max | 1.8 / 2.6 / 3.8 s | 1.9 / **10.9 / 12.4** s | **1.7 / 2.0** / 9.5 s* |

\*an API-side spike on one turn; tonst's own step on it took 0.7 ms.

The saving grew with length (−4% at 12 turns, −21% at 24), and
background mode kept response time at or below no compaction. No summary
replied to the customer any more, and one bad attempt was rejected and
retried as designed. **But the summaries lost facts.** By turn 24 the
local model's summary no longer had the replacement's colour, the
evening delivery slot, or the case number. That led to three additions:

- **Pinned references.** As turns are summarized (or dropped), exact
  identifiers are extracted with patterns, not a model: redaction
  placeholders, `#`-numbers (order #4471), ticket/case codes (CS-20931),
  money amounts. They're carried verbatim under the summary
  ("Pinned references from earlier turns (exact): #4471, CS-20931"), so
  the references a support or agent flow depends on survive whatever the
  model writes. Semantic facts ("the replacement is white") are still
  the summary's job.
- **A content check.** A summary with the headings but no real Key facts
  is rejected. (Not "Goal must be filled": the best real summary in the
  24-turn run had an empty Goal but correct Decisions and Key facts.)
- **A stronger summarizer when fidelity matters:**

  ```python
  from tonst import TonstClient, AnthropicSummarizer
  client = TonstClient(call_fn=..., use_history_compaction=True,
                       compaction_summarizer=AnthropicSummarizer())   # Claude Haiku 4.5
  client.query_messages(history, rolling_state=state, background_summary=True)
  ```

  It only ever sees already-redacted text, costs well under a cent per
  summary (usage is tracked on the instance so it can be counted against
  the savings), and in background mode its latency never reaches the
  user. The local model stays the default. `live_test_free_features.py
  --long` now compares no compaction, local background and Haiku
  background, and scores **fact recall**: of the known facts in turns
  that were summarized, how many survive in summary + pinned references.

**Local vs. Haiku summaries, 24 turns, background mode** (2026-09-25):

| | No compaction | Local (gemma2:2b) | Claude Haiku 4.5 |
|---|---|---|---|
| Cost (Haiku's own cost included) | $0.186 | $0.189 (+1.6%) | $0.181 (−2.7%) |
| Tokens sent | 315k | 207k | 186k |
| Facts kept from summarized turns | — | 5/8 | **8/8** |
| Summaries | — | 2 folded, 1 retry; ~5 turns late | 3/3 folded, on time |

Haiku fixed fidelity, for about a cent over the whole chat. Latency was
the same in all three (p50 ~1.8 s).

**Cost was close to break-even, and that's how prompt caching works.**
Cached history is re-read at 0.1× the input price, and every new summary
changes the prompt, so everything after the system prompt gets written to
cache again at 1.25×. A summary pays for itself only after several more
turns. In this run the last one came 3 turns before the end and never
did. Without caching, the same token cut (315k → 186k) would have saved
about 41%.

### Cache-aware compaction

```python
client = TonstClient(call_fn=..., use_history_compaction=True,
                     compaction_summarizer=AnthropicSummarizer(),
                     compaction_cache_aware=True)       # your call_fn uses prompt caching
client.query_messages(history, rolling_state=state, background_summary=True,
                      expected_remaining_turns=None)    # pass it if you know it
```

With `compaction_cache_aware=True`, a summary that is due by size waits
until it's expected to pay for itself (`estimate_fold_payback`). The
estimate weighs the one-off cache rewrite (1.15× the new summary plus the
recent turns) and, for a paid summarizer, its call against the cache
reads saved on each later turn. It then compares the payback with the
turns left: `expected_remaining_turns` if you pass it, otherwise half as
many again as the conversation has had so far.
`compaction_max_history_tokens` forces a summary regardless, to stay
inside the context window. Reports show `history_fold_postponed`.

Offline simulation of the live test's conversation (Haiku summaries,
Sonnet 4.6 prices, cache rewrite modeled; `benchmark_free_features.py`,
`cache_aware_compaction`). It's calibrated on the live run: it gives
−4.2% at 24 turns, where the live run measured −2.7%.

| Turns | 10 | 12 | 16 | 20 | 24 | 30 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| Summarize at threshold | +14.9% | +7.7% | +9.2% | −2.3% | −4.2% | −10.6% | −18.2% | −36.0% | −54.7% |
| Cache-aware | **0%** | **0%** | +7.3% | −2.9% | −4.2% | −10.6% | −18.1% | −35.9% | −54.7% |

**Live check, 12 turns** (2026-09-25): no compaction $0.0719; summary
at threshold (Haiku) $0.0826 (**+14.9%**: its summary landed 2 turns
later, with only one turn left to pay back); cache-aware **$0.0719
(0.0%)**. It postponed the summary on 4 turns and made no Haiku call.

So cache-aware compaction removes the loss in short chats, where a
summary would only cost money, and changes nothing in long ones. The
bigger picture holds either way: **with prompt caching, compaction saves
money only past ~20 turns of this size (−18% at 40, −55% at 100).**
Before that, it's for fidelity and context-window headroom.

### Gemini

The same pieces work with Gemini:

```python
from tonst import TonstClient, GeminiSummarizer, GeminiTokenCounter
client = TonstClient(call_fn=..., use_history_compaction=True,
                     compaction_summarizer=GeminiSummarizer(),      # gemini-3.5-flash-lite
                     compaction_cache_aware=True,
                     compaction_cache_pricing="gemini")             # implicit caching: no write surcharge
```

`select_tools()` accepts Gemini function declarations as they are
(`name` / `description` / `parameters`). `GeminiTokenCounter` uses the
free countTokens endpoint.

The offline simulation with Gemini prices (3.8 Flash main model, 3.5
Flash-Lite summaries, implicit caching assumed to always hit) comes out
almost the same as for Anthropic. Summarizing at the threshold costs +8%
at 12 turns and saves −17% at 40 and −54% at 100; cache-aware mode is 0%
at 12 turns. There's no write surcharge, but uncached tokens and the
summarizer are relatively more expensive, so it about evens out. The
real unknown is how reliably implicit caching hits, and that's what
`live_test_gemini.py` measures: tools (all vs. filtered), compaction
(none / Flash-Lite summaries / cache-aware), cache-hit share,
countTokens accuracy and latency.

**Live tools run, Gemini 3.8 Flash** (thinking low, 30 tasks, 2026-09-25):

| | All 36 tools | select_tools (top 5) |
|---|---|---|
| Success | 29/30 | 29/30 (29/29 of the tasks "all" got right) |
| Avg prompt tokens | 2,887 | 1,003 (direct tasks: 532) |
| Cost | $0.0738 | **$0.0331 (−55%; direct tasks −70%)** |
| API p50 | 2.28 s | 2.25 s |

- Both modes missed the same task ("What broke after last night's deploy?" → Slack search).
- All 6 paraphrased tasks, and one direct one, fell back to sending every tool, so they saved nothing but lost nothing either.
- **No implicit cache hits at all**: every prompt was under Gemini Flash's 4,096-token cache minimum.
- chars/4 was within 2% of real tokens here, versus 1.77× low on Claude. Gemini adds no hidden tool-use prompt.
- countTokens matched billed tokens on 60/60 calls, but took ~320 ms (up to ~730 ms) each, so it shouldn't sit on the request path.
- `select_tools()` took 2.3 ms (max 4.5 ms).
- Whole run: $0.11.

**Live compaction run, Gemini 3.8 Flash** (20 turns, long history,
Flash-Lite summaries, 2026-09-25):

| | No compaction | Summaries (Flash-Lite) | Cache-aware (before fix) |
|---|---|---|---|
| Cost (summarizer included) | $0.1329 | **$0.1128 (−15.1%)** | $0.1389 (+4.5%) |
| Prompt tokens | 244k | 138k (−43%) | 171k |
| Served from Gemini's cache | 35% | 0% | 0% |
| Facts kept | — | 8/8 | 8/8 |
| Total time p50 / p95 | 2.25 / 4.00 s | 2.15 / 2.70 s | 2.21 / 3.11 s |

- **Gemini's implicit cache is best-effort.** With no compaction, it served nothing from turn 4 (past 4,096 tokens) through turn 15. It only started hitting at ~18k tokens, and then only in a 16k block.
- **Summaries paid off straight away**, because there was little cache to lose. Flash-Lite kept all 8 facts for $0.0027.
- **Cache-aware mode assumed caching was working** and held the summary back for 6 turns, so it cost more than no compaction.
- **Fix: cache-aware mode now uses the observed hit rate.** Call `rolling_state.observe_cache_usage(prompt_tokens, cached_tokens)` after each response. The payback estimate then prices a missed prefix at the write price, and at a 0% hit rate a summary pays back immediately.
  - Replaying this run's usage, the fixed logic summarizes at turn 9, like plain rolling mode (−15%).
  - With a cache that always hits (Anthropic), it behaves exactly as before.
- tonst's own time stayed ≤2.5 ms per turn. The smaller prompts cut API p95 from 4.0 s to 2.7 s.

### Background summaries (no added latency)

Rolling compaction already keeps recent turns verbatim until a summary is
due, so the summary doesn't have to be made *before* the API call:

```python
response, report = client.query_messages(history, rolling_state=state, background_summary=True)
# report.history_summary_scheduled -> a summary started on a background thread
client.wait_for_background_work()   # before exiting, or before persisting state.to_dict()
```

When a summary is due, the local model starts on a background thread *in
parallel with* the API call, and the new summary is used from the next
call on (this call still sends those turns verbatim). The user never
waits for the local model. The job is planned against a snapshot of the
state. If the conversation changes before it finishes (an edit, or a
different conversation passed in), the result is discarded as stale
rather than applied to the wrong history. A background failure follows
the same retry-then-drop rule as a blocking one and never raises into
your app. One worker thread per client, because a laptop can't usefully
run two local models at once. Stateless compaction can't do this, since
it needs a fresh summary on every turn.

**Local model context window.** Ollama gives a model a small context
window by default (2,048–4,096 tokens) and *silently* drops the start of
anything longer. At the 3,000-token default threshold, that would have
cut the oldest turns out of every summary. tonst now sizes `num_ctx` to
each request (`tonst/ollama_util.py`, capped by `TONST_OLLAMA_MAX_CTX`,
default 8,192). A summary or compression that still wouldn't fit is
skipped rather than run on a truncated copy, and LLM redaction logs a
warning that the start of the text wasn't checked.

## Tool and MCP definition optimization

With many tools, or a few MCP servers, the tool definitions alone can
cost tens of thousands of input tokens per request before the model
reads the task. `tonst/tool_optimizer.py` offers two strategies:

**Anthropic: native deferred loading.** `build_anthropic_deferred_tools()`
adds Anthropic's tool search tool and marks every tool `defer_loading:
true` except the few you pin, and except search-type tools
(`*_search_*`, `*lookup*`), which stay loaded by default. Add
`DEFERRED_TOOLS_SYSTEM_HINT` to your system prompt. Both come from a
live-test failure: with search tools deferred, Claude searched the tool
catalog for the *topic* ("billing outage") instead of the capability,
found nothing, and told the user the data didn't exist. Deferred tools are still sent, but stay
out of the model's context and out of the cached prefix until the model
searches for one. Nothing is filtered away. MCP servers are deferred via
`mcp_toolset.default_config`. A deferred tool that also carries
`cache_control` is refused, because the API rejects it with a 400.
(Checked against Anthropic's tool-search docs, 2026-09-24.)

```python
from tonst import build_anthropic_deferred_tools, build_anthropic_cache_request

tools = build_anthropic_deferred_tools(all_tools, always_loaded=["read_file", "search_code"])
body = build_anthropic_cache_request(parts, model="claude-sonnet-4-6", tools=tools)
```

`build_anthropic_cache_request(tools=...)` now places tools in the
request. The system-prompt breakpoint already covers them; with no
system prompt, the breakpoint goes on the last non-deferred tool.

**Any provider: local relevance filtering.** `select_tools()` keeps the
`top_k` tools most relevant to the request (BM25 over names, descriptions
and parameter descriptions; no model, no dependency) plus anything pinned
in `always_include` and any provider/server tool it can't score. It
accepts Anthropic and OpenAI tool formats and returns your **original**
tool dicts in their **original** order. It never rewrites a
description. If the best-matching tool shares fewer than two distinct
words with the request (`min_matched_terms=2`), the match is treated as a
coincidence and nothing is filtered (`fell_back=True`). In the offline
benchmark below, every wrong pick had exactly one shared word ("book a
*slot*" matching a free-time-*slots* tool), and this rule took recall on
paraphrased requests from 50% to 100%. Lexical matching still can't see
synonyms ("ping the team" vs. a Slack tool). Those requests just get no
savings, and native deferred loading doesn't have this limitation.

Filtering fights prompt caching: tools sit at the front of the prefix,
so a different tool list every turn invalidates the cache for everything
after it. Use `ToolSession` in multi-turn runs. It selects once, then only
**adds** tools a later turn clearly needs, never removing or reordering.
Turns where nothing is added send byte-identical tools (`changed=False`).

```python
from tonst import ToolSession

session = ToolSession(all_tools, top_k=8, always_include=["read_file"])
sel = session.select(user_message)          # every turn
print(sel.tokens_saved, sel.added_names, sel.changed)
```

## RAG context optimization

`TonstClient.query_rag(question, chunks)` takes the chunks your
retriever already returned (strings or `{"text": ...}` dicts). It
removes exact and near-duplicates (word 5-gram similarity; the
retriever's earlier chunk wins), then, only if you ask with `top_k`,
`min_relative_score` or `max_tokens`, drops low-relevance chunks
(BM25 against the question). Kept chunks and the question go in the
**variable** part of the prompt, after your cacheable `system` and
`stable_blocks`. The result is then redacted, trimmed and sent like any
other call.

```python
response, report = client.query_rag(
    question, retrieved_chunks,
    system="Answer only from the context.", top_k=5,
)
print(report.chunks_in, report.chunks_sent, report.percent_saved)
```

It only **selects** whole chunks and never rewrites one, so it can't
reshape PII past the redactor. If no chunk shares a word with the
question, relevance filtering is skipped (lexical scoring can't see
synonyms, so "can't tell" never becomes "drop all the context").
Per-chunk local-model compression is deliberately not included: the
Sept 2026 ablation measured local compression at about +1.9 points of
savings for about 2s per call, and per chunk that latency multiplies.
`rag.optimize_chunks()` is usable on its own.

## Free-feature benchmark (offline)

`benchmark_free_features.py` runs without a network or Ollama and writes
`free_features_benchmark.json`. The workloads are **hand-built to look
realistic, not captured from real traffic**: 36 GitHub/Slack/Jira/
filesystem/database-style tools, 30 tasks with known required tools (6
deliberately paraphrased to share no vocabulary), a help-center corpus
with mirrored and older duplicate pages, and a 40-turn chat. Token
counts are chars/4 estimates.

| Feature | Result |
|---|---|
| Tool filtering, `top_k=5` (36 tools, ~2.7k tokens) | Direct requests: **100% recall, 83% fewer tool tokens**. Paraphrased: 100% recall via fallback, 0% savings. |
| Tool filtering, `top_k=8` | Direct: 100% recall, 78% fewer tool tokens. |
| `ToolSession`, 7-turn incident chat | 87.5% fewer tool tokens; tool list changed on 3 of 7 turns (4 cache-stable). |
| RAG, dedupe only (10 retrieved chunks) | 19% fewer context tokens, answer chunk kept 4/4. |
| RAG, dedupe + `top_k=4` / `top_k=2` | 63% / 81% fewer, answer chunk kept 4/4 (only 4 queries: small sample). |
| Rolling vs. stateless compaction (40 turns) | Local-model calls **4 vs. 26**; prefix-stable turns 35 vs. 5; raw tokens sent 27.0k vs. 17.9k (rolling sends *more*); cost with prefix caching: **6.8k vs. 16.2k units at a 10% cache price**, 15.8k vs. 16.9k at 50%. |

To measure the same things against the real Anthropic API (billed
tokens, real cache reads, and whether Claude still picks the right tool),
run `python3 live_test_free_features.py` with `ANTHROPIC_API_KEY` set.

**Live results** (2026-09-24, `claude-sonnet-4-6`, all 30 tasks: 24
direct, 6 paraphrased; real billed tokens; second full run, after the
test-task and deferred-mode fixes described below):

| Mode | Succeeded | Real billed input / call | Output / call | Cost (30 calls) |
|---|---|---|---|---|
| All 36 tools | 30/30 | 3,872 | 108 | $0.40 |
| **`select_tools(top_k=5)`** | **30/30** | **1,668 (−57%)** | 99 | **$0.19 (−51%)** |
| Deferred, tonst defaults + hint | 29/30 | 2,787 (−28%) | 168 | $0.33 (−18%) |
| Deferred, everything deferred | 25/30 | 2,210 (−43%) | 217 | $0.30 (−25%) |

"Succeeded" = Claude's first action was a tool the task needs, or a
reasonable first step such as listing tables before querying.

- **`select_tools` matched sending every tool (30/30) at half the cost.**
  On direct requests cost fell 64% ($0.114 vs. $0.314). On the 6
  paraphrased requests the confidence fallback sent all tools, so those
  cost the same as the baseline, and nothing needed was ever dropped.
- **The deferred fix worked, but deferral still doesn't pay at this
  size.** Keeping search-type tools loaded and adding the system hint
  took deferred loading from 25/30 to 29/30 and eliminated the "search
  found nothing, so the data doesn't exist" failures. Its one miss:
  asked to *read* issue 482, Claude used the already-loaded
  `github_search_issues` instead of searching for `github_get_issue`.
  It cost *more* than everything-deferred, though, and only 18% less
  than sending all tools: 8 tools stay loaded, and whenever a search is
  needed (22 of 30 calls) the prompt is re-read, averaging ~3,300 input
  tokens, close to sending everything.
- **The first full run's gap was mostly test design.** Several tasks
  lacked what the tool needed (an email with no body, "translate this"
  with no text), and Claude rightly asked for it. With those fixed,
  all-tools went from 23/30 to 30/30. The script now scores "asked a
  question" as its own outcome.
- **Token counting:** Anthropic's `count_tokens` matched billed input
  **exactly on 60/60 calls** in both full runs. tonst's chars/4
  estimate was 1.77× too low (see "Exact token counts").

**Which mode to use.** Up to ~50 tools, use `select_tools()` (or
`ToolSession` for multi-turn). It was as accurate as sending every tool
at half the cost, and cheaper than either deferred variant. Deferred
loading makes sense only when even a filtered list would be large
(hundreds of tools, several MCP servers). There, use tonst's defaults
(search-type tools loaded, plus `DEFERRED_TOOLS_SYSTEM_HINT`) and pin
your most-used tools with `always_loaded`. Re-run
`live_test_free_features.py` on your own tool list before relying on
either.

The compaction row is the clearest example of why raw tokens mislead:
rolling mode sends more tokens, but most of them repeat the previous
request's prefix. That makes it much cheaper where cache reads are
heavily discounted, and about break-even where they're only 50% off.
The model assumes the provider caches the longest common prefix and
ignores cache minimum lengths and TTLs, so treat it as directional.

## Exact token counts (optional)

tonst estimates tokens as characters ÷ 4. That's fine for rough prose,
but the live test found real billed input **~1.8×** the estimate on
tool-calling requests: JSON schemas are token-dense, providers add
hidden prompts, and tokenizers differ by model (Anthropic notes Claude
4.7+ produce about 30% more tokens for the same text). For real numbers,
pass a counter:

```python
from tonst import TonstClient, AnthropicTokenCounter, select_tools

counter = AnthropicTokenCounter(model="claude-sonnet-4-6")   # uses ANTHROPIC_API_KEY
client = TonstClient(call_fn=my_api_call, token_counter=counter, savings_log=True)
sel = select_tools(all_tools, request, token_counter=counter.count_tools)  # includes the hidden tool prompt
```

It uses Anthropic's free `count_tokens` endpoint, which Anthropic calls
a close estimate of billed input. The live test script checks it against
real billed tokens. Things to know:

- **Privacy:** the counter only ever receives already-redacted text.
  A remote counter must never see raw PII, and a test enforces this for
  every entry point. With a counter set, `original_tokens` therefore
  measures the redacted, pre-trim prompt.
- **Latency:** each count is a network round trip (~100–300 ms), timed
  separately as `counting_ms`. Use it for calibration or sampling
  rather than on every latency-sensitive call.
- **Fail-soft:** if a count fails, the estimate is used, the report says
  `token_counts_exact=False`, and `tonst stats` labels the numbers
  "estimated" or "counted on N of M calls".

## Savings log

Opt in, and every `query*` call appends one line of metrics to a local
JSONL file (`~/.tonst/savings.jsonl` by default, or `$TONST_SAVINGS_LOG`):
tokens in and tokens sent, redaction counts **by type**, which optional
steps ran, and timings. Nothing is sent anywhere.

```python
client = TonstClient(call_fn=my_api_call, savings_log=True,
                     app_name="support-bot", input_price_per_million=3.0)
```

```
$ tonst stats            # or: python -m tonst stats [--app X] [--since 2026-09-01] [--json]
tonst savings  (2026-09-24T10:02:11+00:00 -> 2026-09-24T16:40:52+00:00)
  calls:            1,204
  tokens in:        3,912,440  (estimated)
  tokens sent:      2,870,115  (estimated)
  tokens saved:     1,042,325  (26.6%)
  est. cost saved:  $3.1270
  PII redacted:     2,311 fields (EMAIL 1,402, NAME 610, PHONE 299)
  avg tonst overhead: 4.2 ms/call
```

*(Illustrative output, not a measured result.)*

What is **never** logged: prompt or response text, PII values, or
placeholder hashes. Placeholders are deterministic hashes of the
original value, so a log full of them could be brute-forced back to
real emails and phone numbers. Only the label (`EMAIL`) is counted.

**Dropped history is reported as lost, not just "saved".** Old turns
that were dropped without a summary weren't sent, so they count toward
`tokens_saved`. But the model never saw that context, so `tonst stats`
shows them on their own line ("of which lost") with a "saved excl.
lost history" figure next to it. Otherwise truncation would look like
optimization. The stats also show median and max tonst overhead, not
just the mean, because one slow local-model call (a fold, a timeout)
can dominate an average of otherwise millisecond-scale calls.
`query_rag()` reports `chunk_filter_skipped` when relevance filtering
was requested but skipped for lack of a confident match. Short
questions often hit this; pass `min_matched_terms=1` to filter anyway.

Honesty about the numbers: token counts use the chars/4 estimate and
are marked as estimates. The dollar figure appears only if you give a
price, and is named `estimated_cost_saved_usd` because it's tokens saved
× your price, not a bill. Provider prompt caching doesn't reduce tokens
(it discounts them), so it never shows up in `tokens_saved`. To record
real cache usage, pass a parsed provider usage report to
`SavingsLog.record(report, usage=...)`.

This is a developer savings log, not an audit record. It has no
tamper-evidence or retention policy, and writes are best-effort (an
unwritable path never breaks your API call).

## Performance: every local step adds to the total, sequentially

Redaction, trimming, and optional compression/compaction all run
**locally, one after another, before** the network call — their
wall-clock time adds up rather than overlapping with it. Three steps
can add real, visible latency, all for the same reason (they call a
local model via Ollama) and all off by default:

- **Enhanced redaction** (`use_enhanced_redaction=True`) to catch
  free-text PII.
- **Local compression** (`use_local_compression=True`) to rewrite the
  prompt shorter.
- **History compaction** (`use_history_compaction=True`, only on
  `query_messages()`) to summarize old turns instead of dropping them.

`OptimizationReport` now times every step (`redaction_ms`, `trim_ms`,
`compression_ms`, `call_ms`, `structuring_ms` for `query_structured()`,
`compaction_ms` for `query_messages()`, and `total_ms`), plus a
`local_overhead_ms` property summing everything that happened before
the actual API call. Check this before enabling any optional step in a
latency-sensitive path — the token/cost savings need to be worth what
they add to response time:

```python
response, report = client.query(prompt)
print(f"tonst overhead: {report.local_overhead_ms:.1f}ms, API call: {report.call_ms:.1f}ms")
```

## Files

| File | Purpose |
|---|---|
| `tonst/cache_structuring.py` | `PromptParts` + helpers that order a request stable-first/variable-last and build a real Anthropic `cache_control` request body — see "Prompt-caching structuring" above. |
| `tonst/redact.py` | Regex-based PII detection + reversible, deterministic redaction, plus `redact_with_llm()` to layer in the enhanced pass below. |
| `tonst/redact_llm.py` | **The differentiator.** Local-LLM-based redaction for free-text PII (names, addresses, employers, codenames) that regex structurally cannot catch. Strict JSON contract, hallucination guard rail, fails soft if Ollama isn't running. |
| `tonst/gliner_redact.py` | **The recommended differentiator.** Extractive/zero-shot NER redaction for the same free-text PII, via GLiNER instead of a generative model -- no GPU or Ollama needed, ~150-250ms latency, structurally can't hallucinate a span. See `redaction_backend="gliner"` in "Why enhanced redaction matters" below. |
| `tonst/trim.py` | Token estimation, whitespace/duplicate cleanup, chat-history truncation. |
| `tonst/compactor.py` | `HistoryCompactor` + `compact_history()` — local-model summarization of conversation history that falls outside the sliding window, instead of discarding it. See "History compaction" above. |
| `tonst/providers/openai.py` | OpenAI prompt-caching: automatic-ordering request building, the optional GPT-5.6+ explicit mode, per-model discount table, and usage parsing for both the Chat Completions and Responses API JSON shapes. See "Multi-provider support" below. |
| `tonst/providers/gemini.py` | Gemini context caching: both the automatic *implicit* path and the resource-based *explicit* `CachedContent` path, each with its own cost model — see "Multi-provider support" below. |
| `tonst/providers/generic.py` | `GenericCacheConfig` + config-driven request/usage/cost functions for **any provider tonst doesn't have a dedicated module for** — the actual answer to "works with any model." See "Any other provider" below. |
| `tonst/providers/presets.py` | Two real, verified `GenericCacheConfig` presets built with `generic.py`: AWS Bedrock's Converse API (different field names from direct Anthropic) and Azure OpenAI's Provisioned-Throughput tier (different pricing from direct OpenAI). |
| `tonst/local_model.py` | Optional Ollama-backed semantic compression, off by default. |
| `tonst/tool_optimizer.py` | Tool/MCP definition optimization: Anthropic deferred loading (`build_anthropic_deferred_tools()`), local relevance filtering for any provider (`select_tools()`), and cache-stable multi-turn filtering (`ToolSession`). See "Tool and MCP definition optimization". |
| `tonst/rag.py` | `optimize_chunks()`: de-duplication and optional relevance/budget filtering of retrieved RAG chunks. Used by `TonstClient.query_rag()`. |
| `tonst/relevance.py` | Dependency-free BM25 scoring and near-duplicate detection, shared by the two modules above. |
| `tonst/savings_log.py` / `tonst/__main__.py` | Opt-in local savings log and the `tonst stats` command. See "Savings log". |
| `live_test_free_features.py` | Live test against the real Anthropic API: real billed tokens for all tools vs. `select_tools()` vs. Anthropic deferred loading (plus whether Claude still calls the right tool, with the stop reason, reply text and tool-search results saved for every miss), real cache reads for rolling vs. stateless compaction, and how close the chars/4 estimate and the `count_tokens` endpoint are to billed tokens. Asks before spending; about $1 at Sonnet 4.6 prices. Results in `live_test_results.json`. |
| `tonst/summarizers.py` | `AnthropicSummarizer`: optional Claude Haiku summarizer for history compaction (redacted text only, usage/cost tracked). See "Rolling compaction". |
| `tonst/ollama_util.py` | Sizes Ollama's context window (`num_ctx`) per request so long prompts aren't silently truncated. See "Local model context window". |
| `tonst/token_count.py` | `AnthropicTokenCounter`: optional exact token counts via Anthropic's free `count_tokens` endpoint. See "Exact token counts". |
| `benchmark_free_features.py` | Offline benchmark for tool filtering, RAG chunk optimization and rolling compaction; results in `free_features_benchmark.json`. See "Free-feature benchmark". |
| `tonst/client.py` | `TonstClient` — the public SDK surface that ties it all together. |
| `demo.py` | Runnable demo against a mocked paid API call — no API key or network needed. Covers both the basic pipeline and prompt-caching structuring. |
| `real_api_demo.py` | Real integration test against the actual `api.anthropic.com` endpoint — see "Running the real API test" below. |
| `cache_savings_demo_anthropic.py` | Measures REAL prompt-caching savings against `api.anthropic.com` using a realistically large reference document (clears the per-model minimum), printing actual `cache_creation_input_tokens` / `cache_read_input_tokens` from two consecutive calls. |
| `cache_savings_demo_openai.py` | The same live-measurement pattern against `api.openai.com`, using `providers/openai.py`'s automatic-caching request shape and per-model discount table. |
| `cache_savings_demo_gemini.py` | The same pattern against the real Gemini API, testing the *implicit* (automatic, best-effort) caching path via `providers/gemini.py`. |
| `cache_savings_demo_gemini_explicit.py` | Gemini's *explicit* `CachedContent` path — deterministic, not best-effort. Creates a cache resource, then references it across several calls. Exists because live testing found implicit caching missed 18/18 real calls while explicit hit 3/3 (later 4/4) the moment billing was enabled; see `providers/gemini.py`'s docstring and `ROADMAP.md` for the full numbers. |
| `cache_savings_demo_generic.py` | A runnable **template** for testing prompt caching against any provider tonst has no dedicated script for — works out of the box in a mocked dry-run mode; three clearly marked edits point it at a real provider and a real key. |

## Why enhanced redaction matters (and why competitors don't have it)

Every comparable open-source tool we found (LLMShield, Helix AI Gateway,
the WSO2 AI Gateway sample) does PII redaction with regex only. Regex is
fast and reliable for *structured* data — emails, card numbers, phone
numbers — but it has no way to know "Priya Malhotra" is a person's name,
or that "Project Nightingale" is a confidential codename, without some
form of semantic understanding.

`TonstClient` closes that gap with a `redaction_backend` parameter, so
you pick how much coverage you need instead of one fixed behavior:

```python
client = TonstClient(call_fn=my_api_call, redaction_backend="gliner")
```

| `redaction_backend` | What it catches | Local model needed | Notes |
|---|---|---|---|
| `"none"` | Nothing — not even regex | — | Only for traffic you're confident carries no PII |
| `"regex"` (default) | Structured PII only (emails, cards, phones, SSNs, IPs) | — | Fast, dependency-free, catches nothing in free text |
| `"gliner"` | Regex + free-text PII (names, employers, codenames) via GLiNER | CPU only, no Ollama | ~150ms-1.3s latency, hardware-dependent (see below); structurally can't hallucinate |
| `"ollama"` | Regex + free-text PII via a local generative model | Ollama running | Seconds, not milliseconds — see `research/colab-benchmark-findings.md` |

**`gliner` is the recommended enhanced backend.** GLiNER
(`gliner_redact.py`) is a small, extractive/zero-shot NER model: it
returns spans/offsets into the *original* text rather than generating
new text, so it structurally cannot produce the JSON-parsing/
truncation/hallucination failures a generative model can, and it needs
no GPU or separately-running service. Install it with:

```bash
pip install tonst[gliner]      # or: pip install -e ".[gliner]" from this repo
```

(`gliner` and its transitive ML dependencies — `torch`, `transformers`,
`huggingface_hub` — are only ever imported if `redaction_backend="gliner"`
is actually selected; every other backend works without installing it.)

Validated end to end at two scales: an initial 180-iteration run
(Apple Silicon Mac, CPU-only, `--workers 1`) and a full 360-iteration
replication on a Google Colab T4 GPU instance (`--workers 1`) that came
back consistent — **87.41%** and **87.78%** free-text PII recall
respectively, both with **100%** recall on the supervised/structured-
field paradigm, **zero round-trip restoration failures**, and **zero
PII leaks**. The one known, accepted gap: codename recall on the two
"supervised" prompt shapes sits around 58-60%, because GLiNER's
zero-shot label matching leans on lexical overlap between the label and
the span (a codename literally containing a cue word like "Project" is
caught reliably; one that doesn't — e.g. "Study NEURO-Vanguard",
"Ledger Settlement-X" — is caught less often). Full methodology,
per-run numbers, and the threshold/label-wording experiments that ruled
out cheaper fixes are in `research/gliner-sanity-check-findings.md`.

GLiNER's own absolute latency turned out to be hardware- and even
session-dependent, not a fixed number: the same `gliner_medium` model
averaged ~269ms per call on the Mac's Apple Silicon CPU vs. 1,308ms and
1,438ms on two independent Colab sessions (`gliner_redact.py` doesn't
move the model onto CUDA, so the T4 GPU sitting alongside it on Colab
isn't actually used for this step -- both runs were CPU-bound, and
Colab's shared virtual CPU is both slower and more variable than Apple
Silicon for this workload). Budget from a measurement on your actual
target hardware rather than any single number in isolation.

A `--workers 4` (production-default) run of the same full pipeline on
Colab -- replicated twice -- confirmed correctness holds under
concurrency -- identical recall/leak/restoration-failure numbers to
the `--workers 1` run on every run, settling the question
`research/gliner-sanity-check-findings.md` had flagged as open. It is
**not** free on latency, though: both redaction and compression slowed
down substantially under 4-way contention (mean redaction latency rose
3.2-3.4x across the two runs, with 13.9% of calls landing within 250ms
of the harness's timeout-tracking threshold, both times) -- correct,
but not the naive 4x throughput speedup one might expect. Full numbers
in the research doc.

`redact_llm.py` (the `"ollama"` backend) remains available for cases
that need a generative model's broader judgment and can tolerate its
latency and occasional hallucination-guard-rail rejections. Both
enhanced backends degrade gracefully if their local model isn't
available — the pipeline never breaks, and neither ever silently trusts
a flagged span that doesn't verbatim-match the source text (see the
guard rails in `redact_llm.py` and `gliner_redact.py`).

Independently of the backend, `redaction_model` / `compression_model` /
`compaction_model` let each Ollama-backed stage use a different model
instead of one shared model compromising on every job — see the
`TonstClient.__init__` docstring in `client.py` for details.

The old `use_enhanced_redaction=True` boolean still works (it now maps
to `redaction_backend="ollama"` for backward compatibility) but new code
should use `redaction_backend` directly.

## Running the demo

```bash
python3 demo.py
```

You should see PII stripped before the "paid API" ever saw it, and a
walkthrough of prompt-caching structuring including the actual JSON body
with `cache_control` breakpoints.

## Running the test suite

```bash
pip install -r requirements-dev.txt
pytest test_tonst.py -v
```

183 tests covering redaction round-trips (including placeholder
determinism, which caching depends on), the hallucination guard rail,
fail-soft behavior when Ollama isn't running, trimming, prompt-caching
structuring, history compaction (including the redact-before-compact
ordering, the summary guard rail and rolling compaction), tool/MCP
definition optimization, RAG chunk optimization, the savings log, and
the full `TonstClient` pipeline end to end. Runs automatically on every push via GitHub
Actions (`.github/workflows/tests.yml`) across Python 3.9–3.12.

## Running the real API test

`real_api_demo.py` wires `TonstClient` to the actual `api.anthropic.com`
endpoint — not a mock. Confirmed in testing: without a key it reaches
the real API and fails with a clean `authentication_error`, proving the
request format (endpoint, headers, JSON body) is correct end-to-end.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com
python3 real_api_demo.py
```

With no key set, you'll see the same clean auth failure this was tested
with — that's expected and confirms the integration is wired correctly.
Add your real key to see an actual response, redaction, and token savings.

## Testing against a real provider

`real_api_demo.py` and `demo.py` prove the pipeline is wired correctly,
but neither one tells you whether prompt caching actually saved
anything — that requires a stable prefix past the per-model minimum
(see "Prompt-caching structuring" above) and reading the real `usage`
field back from two consecutive calls. There is at least one live-test
script per provider, because there is no single generic API to call —
each needs its own base URL, auth scheme, and request/response shape:

| Script | Provider | Needs |
|---|---|---|
| `cache_savings_demo_anthropic.py` | Anthropic (`api.anthropic.com`) | `ANTHROPIC_API_KEY` |
| `cache_savings_demo_openai.py` | OpenAI (`api.openai.com`) | `OPENAI_API_KEY` |
| `cache_savings_demo_gemini.py` | Gemini (`generativelanguage.googleapis.com`), *implicit* caching (automatic, best-effort) | `GEMINI_API_KEY` |
| `cache_savings_demo_gemini_explicit.py` | Gemini, *explicit* `CachedContent` caching (deterministic — a guaranteed hit, not best-effort) | `GEMINI_API_KEY` + billing enabled on that key's project |
| `cache_savings_demo_generic.py` | **Any other provider** — Mistral, Groq, Together, DeepSeek, a self-hosted server, etc. | Runs with no key at all in its default dry-run mode; see below to point it at a real one. |

All five follow the identical overall pattern — build an eligible
request, call the real API against an identical stable prefix, parse
the real `usage` field from the response(s), print the measured cost
difference — only the provider-specific plumbing (and, for Gemini,
which of its two caching mechanisms) changes. Pick the one for your
provider and run it, e.g.:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or: echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env   (gitignored)
python3 cache_savings_demo_anthropic.py
```

Expect call 1 to show a cache write (a new cache entry created) and
call 2 — sent seconds later with the identical stable prefix — to show
a cache read. If call 2 doesn't show a hit, something real may be wrong
(TTL expired, prefix wasn't actually byte-identical, or caching isn't
enabled for that model/account) and is worth chasing down before
relying on this feature in production — **except on Gemini**, where a
miss on the implicit path is a genuinely possible, documented, best-
effort outcome, not necessarily a bug; see `cache_savings_demo_gemini.py`'s
own docstring.

**Gemini specifically: check billing before chasing anything else, and
consider using the explicit script instead.** Confirmed via live
testing (Sept 2026): a free-tier Gemini API key gets a zero-token cache
storage quota, so caching (implicit *and* explicit) cannot activate at
all, no matter how correctly the request is shaped. If every call shows
`cache_hit=False` including several retries, enable billing on that
key's Google AI Studio / Cloud project before suspecting anything else
— see `providers/gemini.py`'s module docstring for the exact error this
produces. Separately, even with billing enabled, implicit caching
missed on every one of 18 real test calls in this project — if you need
the savings to reliably show up rather than just theoretically exist,
run `cache_savings_demo_gemini_explicit.py` instead, which got a
guaranteed hit on 3/3 calls in the same testing.

**Anthropic — confirmed against the real API on 2026-09-09**
(`claude-sonnet-4-6`, 5-minute TTL, a ~1,560-token reference-doc stable
prefix):

| | Call 1 (cache write) | Call 2 (cache read) |
|---|---|---|
| Tokens processed | 1,586 | 1,586 |
| `cache_creation_input_tokens` | 1,547 | 0 |
| `cache_read_input_tokens` | 0 | 1,547 |
| Estimated cost vs. no caching | **-24.4%** (a premium) | **+87.8%** (the payoff) |

The first call with any new stable prefix actually costs *more* than not
caching (Anthropic charges a 1.25x premium on a 5-minute-TTL cache
write), and the saving only shows up from the second call onward, when
that prefix is read from cache at 10% of the base input price.
`CacheUsageReport.estimated_cost_savings_percent(model)` computes this
— a true cost estimate using Anthropic's published cache pricing
multipliers, not a token-count proxy for it.

**Reconfirmed with redaction in the loop (2026-09-13,
`tonst_gliner_full_benchmark.ipynb`).** The standalone test above
exercises caching alone; a second real-API run combined it with GLiNER
redaction on a 1,418-token reference document containing five real PII
fields (two names, an employer, an email, a project codename). GLiNER
caught all five before the request was built. Call 1 (write): 1,417
tokens processed, `cache_creation_input_tokens=1336`. Call 2 (read),
identical prefix: `cache_read_input_tokens=1336` (94.3% of that call's
input served from cache) for a **+84.9%** cost saving. A final
assertion confirmed none of the five raw PII values reached the actual
request body — the redact-then-cache pipeline holds end to end against
the real API, not just each half in isolation.

**Gemini — confirmed against the real API on 2026-09-09**
(`gemini-3.6-flash`, explicit `CachedContent` path, billing enabled, a
~4,600-token reference-doc stable prefix). Implicit caching was also
tested and is deliberately excluded from this table: it missed on all
18 real calls attempted across this project, both before and after
billing was enabled — a genuinely possible outcome per Google's own
"best-effort, no guarantee" framing, not a tonst bug, but not something
you can build a cost projection on either. The explicit path, in
contrast, hit on every call:

| | Populate (one-time) | Read calls (4/4 hit) |
|---|---|---|
| Tokens processed | 4,824 | 4,824 cached + ~22 new, per call |
| `cachedContentTokenCount` | 0 | 4,824 |
| `cache_hit` | false (nothing to reuse yet) | **true**, every call |
| Estimated cost vs. no caching | standard rate, no discount | **+89.6%** per call (steady state) |

Blended across the whole run (the one-time populate cost + storage rent
+ 4 discounted reads, vs. 4 full-price calls): **+62.2%** — lower than
the 89.6% steady-state figure because the populate call and storage
rent are pure overhead that answers no question by itself; the blended
number climbs toward 89.6% the more times a populated cache gets reused
within its TTL. See `ROADMAP.md` for the full investigation, including
the free-tier cache-quota discovery that explained the initial 0%
results before billing was enabled.

**Both providers confirm the same core nuance: caching does not reduce
token count at all.** Every call above processed its full stable prefix
regardless of cache hit or miss — only the price per token on the
cached portion changes.

### Testing against any other provider

For a provider none of the named provider scripts cover,
`cache_savings_demo_generic.py` is a runnable template, not a
throwaway example: it uses `tonst.providers.generic.GenericCacheConfig`
(see "Any other provider" above) and works out of the box with
`DRY_RUN = True` (the default) against a mocked response, so you can
see the whole pipeline — eligibility check, request building, usage
parsing, cost estimate — run end to end before writing a single line of
integration code. Three clearly marked edits (fill in your provider's
real `GenericCacheConfig` numbers, replace the placeholder HTTP call
with a real one, flip `DRY_RUN = False`) turn it into a real live test
against your provider and your key:

```bash
python3 cache_savings_demo_generic.py   # dry run, no key needed, works immediately
# then edit the 3 TODOs in the file, set DRY_RUN = False, and:
export MY_PROVIDER_API_KEY=...
python3 cache_savings_demo_generic.py   # now calling your real provider
```

## Multi-provider support

Everything above this section is Anthropic-specific
(`build_anthropic_cache_request`, `parse_anthropic_usage`, the
per-model minimum table). `tonst/providers/openai.py` and
`tonst/providers/gemini.py` bring the same idea — structure the request
correctly, then measure the real cost effect from real usage data — to
the other two major providers. They are **not** built behind one shared
interface: the three providers' caching mechanics differ enough
(especially Gemini's) that forcing a common abstraction would hide real
differences a caller needs to know about, not simplify anything. The
one thing they do share is `CacheUsageReport` — a plain data container
each provider's own `parse_*_usage()` fills in, so callers get a
consistent shape back regardless of which provider they used.

| | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| Marker required? | Always (`cache_control`) | No — automatic by default. GPT-5.6+ has an optional explicit mode. | No — automatic ("implicit") by default. A separate, heavier `CachedContent` resource ("explicit") also exists. |
| Minimum cacheable length | 512–4,096 tokens, per model | Flat 1,024 tokens, all models | 2,048–4,096 tokens, per model |
| Cache-read discount | 0.1x (0.025x for Fable/Mythos) | Varies by model: 0.5x down to 0.0125x | Confirmed uniform 0.1x across the lineup |
| Cache-write premium | 1.25x (5m TTL) / 2x (1h TTL) | Only on GPT-5.6+ explicit mode: 1.25x. Automatic path: none. | None on either path — see below. |
| Ongoing storage cost | None | None | **Explicit path only**: a flat $/million-tokens/hour rent, charged whether or not the cache is ever read again |
| Usage field (JSON path) | `usage.cache_read_input_tokens` / `cache_creation_input_tokens` | `usage.prompt_tokens_details.cached_tokens` (Chat Completions) or `usage.input_tokens_details.cached_tokens` (Responses) — different per endpoint | `usageMetadata.cachedContentTokenCount` |

Gemini's explicit `CachedContent` path is genuinely different from
every other mechanism in this table, which is why it gets its own cost
function (`estimated_explicit_cache_cost_savings_percent()`) instead of
reusing the read/write-multiplier shape the others share: populating it
costs the standard input rate once, then a storage rent accrues **per
hour it exists, whether or not it's ever read again**. Whether it saves
money at all depends on how many times you reuse it inside its TTL
window — the function can (correctly) return a negative number if you
don't reuse it enough to cover the rent.

```python
from tonst import PromptParts
from tonst.providers import openai as openai_provider
from tonst.providers import gemini as gemini_provider

parts = PromptParts(system="...", stable_blocks=["..."], variable="...")

# OpenAI -- automatic, no marker needed for most models:
body = openai_provider.build_openai_cache_request(parts, model="gpt-4o")

# Gemini -- implicit (automatic) path:
body = gemini_provider.build_gemini_content_request(parts, model="gemini-2.5-flash")

# Gemini -- explicit path (create once, reference by id on every later call):
cache_resource = gemini_provider.build_cached_content_resource(parts, model="gemini-2.5-flash")
# ... call cachedContents.create with cache_resource, get back a name, then:
body = gemini_provider.build_generate_request_from_cache(cache_resource_name, parts.variable)
```

## Any other provider

Anthropic, OpenAI, and Gemini are three providers, not "any model."
Mistral, Groq, Together AI, Fireworks, DeepSeek, xAI, Cohere, a
self-hosted vLLM/Ollama/TGI server, and whatever ships next month are
all real gaps a hardcoded three-provider tool would have — and real
evidence the gap doesn't stop at three: AWS Bedrock's Converse API
reports Claude's own cache usage under different, camelCase field names
than Anthropic's direct API, despite serving the identical model. A
wrapped platform can break a hardcoded parser even when the underlying
provider is one tonst already supports.

`tonst/providers/generic.py` is the actual answer: a `GenericCacheConfig`
that takes a provider's minimum cacheable length, cache-read/write price
multipliers, and the dot-path to its usage JSON's cached-token count as
plain data, instead of tonst needing hardcoded, provider-specific code
for every one. `structure_for_caching()` (already provider-agnostic —
see above) handles the request-shaping half for free; `generic.py`
handles the usage-parsing and cost-estimate half for anything else,
once you tell it four things from that provider's own docs:

```python
from tonst.providers.generic import GenericCacheConfig, parse_usage, estimated_cost_savings_percent

my_provider = GenericCacheConfig(
    label="Mistral",                        # cosmetic, for eligibility messages
    minimum_tokens=1024,                     # whatever their docs say
    cache_read_multiplier=0.5,               # e.g. 50% off cached tokens
    cache_write_multiplier=1.0,              # 1.0 if there's no write premium
    usage_read_path="usage.cached_tokens",   # wherever THEIR usage JSON puts it
    usage_input_path="usage.prompt_tokens",
    usage_output_path="usage.completion_tokens",
)

usage = parse_usage(response_json, my_provider)
savings = estimated_cost_savings_percent(usage, my_provider)
```

If you don't yet know a provider's real numbers, the defaults
(`minimum_tokens=0`, `cache_read_multiplier=1.0`) make every function a
safe no-op — eligibility is always `True`, savings always comes back
`0%`. Wiring in a new provider before checking its caching docs degrades
gracefully instead of silently fabricating a discount that was never
confirmed.

`tonst/providers/presets.py` ships two real configs built this way,
each verified against official docs on 2026-09-09:
- **AWS Bedrock, Converse API, Claude models** — needs its own preset
  because Bedrock reports cache usage under different field names
  (`cacheReadInputTokens`, `cacheWriteInputTokens`) than Anthropic's
  direct API. (If you call Bedrock's *InvokeModel* API with Anthropic's
  own request format instead, the response is unchanged snake_case —
  just use `parse_anthropic_usage()` directly there; no preset needed.)
- **Azure OpenAI, Provisioned Throughput (PTU-M) tier** — needs its own
  preset only because this specific tier gets an Azure-only discount
  (up to 100% off cached tokens) that OpenAI's own per-model table has
  no equivalent of. A standard Azure deployment uses the exact same
  field path as OpenAI direct, so `providers/openai.py` already works
  there unchanged — also no preset needed.

That "sometimes you need one, sometimes you don't, and you have to
check" is the actual lesson from building this: don't assume every
wrapped platform needs new code, and don't assume none of them do.

**What's been live-tested vs. what hasn't:** the Anthropic path above
was confirmed against the real API (see "Measuring real prompt-caching
savings"). The OpenAI and Gemini modules were built directly from each
provider's own current documentation and are logically consistent with
it, but have **not** been run against a real OpenAI or Gemini API key
yet — see "Testing against a real provider" above for the scripts that exist per provider, and the generic template for anything else. Flagged
honestly rather than glossed over, a few specific things are worth
verifying empirically before depending on them in production:
- The exact JSON path for OpenAI's GPT-5.6+ explicit cache-write token
  count (`cache_write_tokens`) was corroborated by two secondary
  sources (an Azure Q&A thread, an AWS Bedrock write-up), not confirmed
  against OpenAI's own raw response schema directly.
- Gemini's per-model minimum-token thresholds conflict between
  `ai.google.dev`'s own caching docs and the Firebase AI Logic wrapper
  docs (2,048 vs. 1,024 tokens for Flash models). `CACHE_MINIMUM_TOKENS`
  in `providers/gemini.py` uses the primary `ai.google.dev` numbers.
- All dollar figures in `providers/gemini.py`
  (`GEMINI_STORAGE_USD_PER_MILLION_TOKEN_HOUR`) are absolute USD prices
  pulled from Google's live pricing page on 2026-09-09, not ratios —
  unlike Anthropic/OpenAI's multipliers, these **will** go stale as
  pricing changes (Google's own page already lists a scheduled change
  for 2027-01-01). Verify current pricing before relying on this for
  real budgeting.

## Wiring in a real provider

Replace the mock in `demo.py`:

```python
import anthropic
client = anthropic.Anthropic()

def real_call(prompt: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

opt_client = TonstClient(call_fn=real_call)
response, report = opt_client.query(user_prompt)
```

## What's genuinely production-ready vs. what's a stub

**Solid enough to build on:**
- Redaction/restoration round-trip logic (tested, deterministic).
- Prompt-caching structuring ordering logic and the Anthropic request-body
  shape (tested against the documented `cache_control` format).
- Fail-soft design for the optional local-model step.

**Deliberately simplified for the POC — replace before shipping:**
- `estimate_tokens()` uses a chars/4 heuristic. Use the real provider tokenizer
  (`tiktoken`, Anthropic's token counting endpoint, etc.) for accurate billing math.
- `redact.py` only catches high-confidence patterns (email, phone, card, IP,
  SSN-like). Free-text PII (names, addresses in prose) needs either a local
  NER model or a local LLM prompted specifically for redaction (see
  `redact_llm.py` for the latter).
- `build_anthropic_cache_request()` only covers Anthropic today. OpenAI and
  Gemini don't need an explicit marker (their automatic prefix caching only
  needs correct ordering, which `structure_for_caching()` already provides),
  but a dedicated OpenAI/Gemini request-builder isn't written yet.
- `check_cache_eligibility()`'s token estimate uses the same chars/4
  heuristic as `estimate_tokens()` — good enough to catch an obviously
  too-short prompt, not an exact prediction against the real tokenizer.
  the real `usage` check in the relevant `cache_savings_demo_*.py` script is the real test.
- `OptimizationReport`'s per-step timings (`redaction_ms`, `trim_ms`,
  `compression_ms`, `call_ms`, `structuring_ms`, `total_ms`) are measured
  with `time.perf_counter()` around each pipeline step — accurate for
  this process, but not a substitute for real load testing under
  concurrency.
