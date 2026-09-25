# Results and benchmarks

Every measured result for tonst, with how it was measured. Each section says whether its numbers came from real provider APIs, a local pipeline run, or a simulation.

[← Back to the README](../README.md)

## Measured results

Three kinds of measurement appear below, and each table says which it is:

- **Live API**: real calls to Anthropic or Gemini, with the provider's billed usage. This covers the prompt-caching benchmark (24 Gemini calls), the September 2026 tool-filtering and compaction runs, and the single-prompt check.
- **Local pipeline**: tonst's own redaction, trimming and compression run for real on 360 prompts, with the paid API call mocked. Token counts, PII recall and latency are measured. Dollar figures are estimates at a fixed $3 per million input tokens.
- **Simulation**: the offline cost model in `benchmarks/benchmark_free_features.py`, calibrated against the live runs. It's used only where a live run would be too expensive, and is labelled wherever it appears.

| Mechanism | Scope & Scale | Peak Savings | Workload Average | Key Reliability / Safety Metric |
|---|---|---|---|---|
| **Local Trim + GLiNER Redaction + Compression** | 360 local-pipeline runs, API mocked (`claude-3-5-sonnet-20241022` pricing baseline) | **23.68% token drop** (Space) | **21.09% token drop** | 100.0% Structured PII Recall (0 leaks), 87.78% Free-Text PII Recall |
| **Provider Prompt Caching** | 24 live calls (`gemini-3.1-flash-lite`) | **72.00% read discount** | **53.00% net cost drop** | Zero cache-key leakage |
| **Tool / MCP definition filtering** (`select_tools`, top 5 of 36) | 30 tasks × 2 providers (`claude-sonnet-4-6`, `gemini-3.8-flash`) | **−70% cost** on direct tasks (Gemini) | **−51% (Claude) / −55% (Gemini) cost** | Same success as sending all tools: 30/30 Claude, 29/30 Gemini (same miss either way) |
| **Rolling history compaction** (background summaries) | 20–24-turn support chats, both providers | **−15.1% cost, −43% tokens** (Gemini, 20 turns) | −2.7% (Claude + Haiku, 24 turns, cache already cheap) | 8/8 facts kept with Haiku / Flash-Lite summaries; no added latency |

---

### Free features on live APIs (Claude Sonnet 4.6 + Gemini 3.8 Flash, September 2026)

These are real API calls with billed usage, from `benchmarks/live_test_free_features.py`
(Anthropic) and `benchmarks/live_test_gemini.py` (Gemini). The same 30 tool tasks and
the same scripted support conversation were used on both providers. Details
and every intermediate run are in [Tool definitions](tools-and-rag.md#tool-and-mcp-definition-optimization),
[Rolling compaction](compaction.md#rolling-compaction) and [ROADMAP.md](../ROADMAP.md).

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

### Mechanical Trimming & Privacy Benchmark (360-Iteration Suite, local pipeline)

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
`benchmarks/benchmark_tonst.py`'s built-in mocked "paid API" call -- so the token
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
* **Free-Text PII Sensitivity**: `redaction_backend="gliner"` + `gemma3:1b` compression together caught **87.78%** of free-text PII overall (77.04% supervised / 98.52% unsupervised) — up from a 57.50% regex-only baseline (the number this table showed before GLiNER was wired in; still the right comparison for "what does enabling an enhanced backend actually buy you"). The residual gap is the known, accepted GLiNER limitation on the two "supervised" prompt shapes specifically (codename recall ~58-60% there, see [Choosing a redaction backend](redaction.md#choosing-a-redaction-backend)) — not a bug, and not something regex-only redaction could have caught at all.
* **Local Latency Overhead**: Running local GLiNER redaction + `gemma3:1b` compression adds a mean local processing delay of 2,258.3-2,389.8 ms before API dispatch across two independent 360-iteration Colab runs (p50: 2,022.8-2,089.1 ms, p90: 4,031.7-4,380.1 ms, p99: 5,213.7-5,624.5 ms) — measured on a Google Colab T4 instance, where GLiNER itself ran on CPU (this step doesn't use the GPU; see [Choosing a redaction backend](redaction.md#choosing-a-redaction-backend) for a second measurement from different hardware that came out meaningfully faster). Token, recall, leak, and restoration numbers were bit-for-bit identical across both runs — only latency varied, consistent with shared-cloud-instance noise rather than any code change. A separate `--workers 4` run of this same pipeline (also replicated twice) confirmed correctness holds under concurrency (identical recall/leak/restoration numbers on every run) but is NOT free on latency — see `research/gliner-sanity-check-findings.md` for the full breakdown.

---

### Multi-Industry Prompt Caching Benchmark (`gemini-3.1-flash-lite`, live API)

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

### Single-Prompt Verification Baseline (live API)

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

**Why cost tracks tokens 1:1 here, unlike prompt-caching:** This test measures plain trimming — literally sending fewer tokens at the same standard price, no discount or premium multiplier involved. Prompt caching is a meaningfully different mechanism (see [Prompt-caching structuring](caching-and-providers.md#prompt-caching-structuring)), where token count never drops and the entire saving comes from a cheaper price per token on a cache hit instead. Both are real cost reductions; they just come from different places, and `tonst` reports both correctly rather than treating "tokens saved" as a universal proxy for "money saved."

At real traffic volumes the fractions of a cent above add up: an app sending 100,000 requests/day with this same 50-token, 26.9% overhead would save an estimated **$15.00/day, or about $5,475/year**, on this one mechanical trimming pass alone — before any prompt-caching savings on top of it.

Reproduce this yourself with `examples/real_api_demo.py` (see below).

**Live Header Verification**
* **Anthropic (`api.anthropic.com`)**: Real call execution returns `cache_read_input_tokens` and `cache_creation_input_tokens` in the raw usage response header, confirming that cache breakpoints (`cache_control`) successfully shift tokens from full input pricing ($3.00/1M) to cached read pricing ($0.30/1M).
* **Gemini (`generativelanguage.googleapis.com`)**: Live test responses confirm `cachedContentTokenCount` matching the exact token length of the prefix payload, verifying that zero cached tokens were billed at standard rates.

## Free-feature benchmark (offline)

`benchmarks/benchmark_free_features.py` runs without a network or Ollama and writes
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
run `python3 benchmarks/live_test_free_features.py` with `ANTHROPIC_API_KEY` set.

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
  estimate was 1.77× too low (see [Exact token counts](measurement.md#exact-token-counts)).

**Which mode to use.** Up to ~50 tools, use `select_tools()` (or
`ToolSession` for multi-turn). It was as accurate as sending every tool
at half the cost, and cheaper than either deferred variant. Deferred
loading makes sense only when even a filtered list would be large
(hundreds of tools, several MCP servers). There, use tonst's defaults
(search-type tools loaded, plus `DEFERRED_TOOLS_SYSTEM_HINT`) and pin
your most-used tools with `always_loaded`. Re-run
`benchmarks/live_test_free_features.py` on your own tool list before relying on
either.

The compaction row is the clearest example of why raw tokens mislead:
rolling mode sends more tokens, but most of them repeat the previous
request's prefix. That makes it much cheaper where cache reads are
heavily discounted, and about break-even where they're only 50% off.
The model assumes the provider caches the longest common prefix and
ignores cache minimum lengths and TTLs, so treat it as directional.

## Performance: local steps run in sequence

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
