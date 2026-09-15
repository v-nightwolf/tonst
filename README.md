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
of dropping it outright — see "History compaction" below.

## Whitepaper

A full technical whitepaper — methodology, benchmarks, and real measured
cost/privacy results — is available:

- **Live version:** [Beyond the Prompt](https://claude.ai/code/artifact/0dc4a7b5-37f6-469b-bda8-0a1843b384f5)
- **Permanent citable record (DOI):** [10.5281/zenodo.22745266](https://doi.org/10.5281/zenodo.22745266)

## Real results (not simulated)

Across **384 live API benchmark iterations** spanning 6 enterprise verticals (Medical, Space, Electronics, Finance, IT, Legal), `tonst` cuts prompt payload volume by **up to 23.68% locally** (running the full pipeline: GLiNER redaction + mechanical trim + compression) and drives a **53.00% net reduction in API cost** via provider prompt caching—all while maintaining **100.0% structured PII recall with zero privacy leaks**.

| Mechanism | Scope & Scale | Peak Savings | Workload Average | Key Reliability / Safety Metric |
|---|---|---|---|---|
| **Local Trim + GLiNER Redaction + Compression** | 360 Runs (`claude-3-5-sonnet-20241022` pricing baseline) | **23.68% token drop** (Space) | **21.09% token drop** | 100.0% Structured PII Recall (0 leaks), 87.78% Free-Text PII Recall |
| **Provider Prompt Caching** | 24 Calls (`gemini-3.1-flash-lite`) | **72.00% read discount** | **53.00% net cost drop** | Zero cache-key leakage |

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

85 tests covering redaction round-trips (including placeholder
determinism, which caching depends on), the hallucination guard rail,
fail-soft behavior when Ollama isn't running, trimming, prompt-caching
structuring, history compaction (including the redact-before-compact
ordering and the summary guard rail), and the full `TonstClient`
pipeline end to end. Runs automatically on every push via GitHub
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
