# Testing, demos and repository layout

How to run the unit tests, demos, offline benchmark and live API tests, and what each file in the repository is for.

[← Back to the README](../README.md)

## Running the demo

```bash
python3 examples/demo.py
```

You should see PII stripped before the "paid API" ever saw it, and a
walkthrough of prompt-caching structuring including the actual JSON body
with `cache_control` breakpoints.

## Running the test suite

```bash
pip install -r requirements-dev.txt
pytest test_tonst.py -v
```

191 tests covering redaction round-trips (including placeholder
determinism, which caching depends on), the hallucination guard rail,
fail-soft behavior when Ollama isn't running, trimming, prompt-caching
structuring, history compaction (including the redact-before-compact
ordering, the summary guard rail and rolling compaction), tool/MCP
definition optimization, RAG chunk optimization, the savings log, the provider adapters and `messages_fn`, and
the full `TonstClient` pipeline end to end. Runs automatically on every push via GitHub
Actions (`.github/workflows/tests.yml`) across Python 3.9–3.12.

## Running the real API test

`examples/real_api_demo.py` wires `TonstClient` to the actual `api.anthropic.com`
endpoint — not a mock. Confirmed in testing: without a key it reaches
the real API and fails with a clean `authentication_error`, proving the
request format (endpoint, headers, JSON body) is correct end-to-end.

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # from console.anthropic.com
python3 examples/real_api_demo.py
```

With no key set, you'll see the same clean auth failure this was tested
with — that's expected and confirms the integration is wired correctly.
Add your real key to see an actual response, redaction, and token savings.

## Files

| File | Purpose |
|---|---|
| `tonst/cache_structuring.py` | `PromptParts` + helpers that order a request stable-first/variable-last and build a real Anthropic `cache_control` request body — see [Prompt-caching structuring](caching-and-providers.md#prompt-caching-structuring). |
| `tonst/redact.py` | Regex-based PII detection + reversible, deterministic redaction, plus `redact_with_llm()` to layer in the enhanced pass below. |
| `tonst/redact_llm.py` | **The differentiator.** Local-LLM-based redaction for free-text PII (names, addresses, employers, codenames) that regex structurally cannot catch. Strict JSON contract, hallucination guard rail, fails soft if Ollama isn't running. |
| `tonst/gliner_redact.py` | **The recommended differentiator.** Extractive/zero-shot NER redaction for the same free-text PII, via GLiNER instead of a generative model -- no GPU or Ollama needed, ~150-250ms latency, structurally can't hallucinate a span. See `redaction_backend="gliner"` in [Choosing a redaction backend](redaction.md#choosing-a-redaction-backend). |
| `tonst/trim.py` | Token estimation, whitespace/duplicate cleanup, chat-history truncation. |
| `tonst/compactor.py` | `HistoryCompactor` + `compact_history()` — local-model summarization of conversation history that falls outside the sliding window, instead of discarding it. See [History compaction](compaction.md). |
| `tonst/providers/openai.py` | OpenAI prompt-caching: automatic-ordering request building, the optional GPT-5.6+ explicit mode, per-model discount table, and usage parsing for both the Chat Completions and Responses API JSON shapes. See [Multi-provider support](caching-and-providers.md#multi-provider-support). |
| `tonst/providers/gemini.py` | Gemini context caching: both the automatic *implicit* path and the resource-based *explicit* `CachedContent` path, each with its own cost model — see [Multi-provider support](caching-and-providers.md#multi-provider-support). |
| `tonst/providers/generic.py` | `GenericCacheConfig` + config-driven request/usage/cost functions for **any provider tonst doesn't have a dedicated module for** — the actual answer to "works with any model." See [Any other provider](caching-and-providers.md#any-other-provider). |
| `tonst/providers/presets.py` | Two real, verified `GenericCacheConfig` presets built with `generic.py`: AWS Bedrock's Converse API (different field names from direct Anthropic) and Azure OpenAI's Provisioned-Throughput tier (different pricing from direct OpenAI). |
| `tonst/local_model.py` | Optional Ollama-backed semantic compression, off by default. |
| `tonst/tool_optimizer.py` | Tool/MCP definition optimization: Anthropic deferred loading (`build_anthropic_deferred_tools()`), local relevance filtering for any provider (`select_tools()`), and cache-stable multi-turn filtering (`ToolSession`). See [docs/tools-and-rag.md](tools-and-rag.md). |
| `tonst/rag.py` | `optimize_chunks()`: de-duplication and optional relevance/budget filtering of retrieved RAG chunks. Used by `TonstClient.query_rag()`. |
| `tonst/relevance.py` | Dependency-free BM25 scoring and near-duplicate detection, shared by the two modules above. |
| `tonst/savings_log.py` / `tonst/__main__.py` | Opt-in local savings log and the `tonst stats` command. See [Savings log](measurement.md#savings-log). |
| `benchmarks/live_test_free_features.py` | Live test against the real Anthropic API: real billed tokens for all tools vs. `select_tools()` vs. Anthropic deferred loading (plus whether Claude still calls the right tool, with the stop reason, reply text and tool-search results saved for every miss), real cache reads for rolling vs. stateless compaction, and how close the chars/4 estimate and the `count_tokens` endpoint are to billed tokens. Asks before spending; about $1 at Sonnet 4.6 prices. Results in `live_test_results.json`. |
| `tonst/summarizers.py` | `AnthropicSummarizer` (Claude Haiku) and `GeminiSummarizer` (Gemini Flash-Lite): optional API summarizers for history compaction (redacted text only, usage/cost tracked). See [Rolling compaction](compaction.md#rolling-compaction). |
| `tonst/ollama_util.py` | Sizes Ollama's context window (`num_ctx`) per request so long prompts aren't silently truncated.  |
| `tonst/token_count.py` | `AnthropicTokenCounter` and `GeminiTokenCounter`: optional exact token counts via each provider's free counting endpoint. See [Exact token counts](measurement.md#exact-token-counts). |
| `tonst/adapters.py` | Converts tonst's message list to Anthropic / OpenAI / Gemini request shapes (with prompt-caching markers for Anthropic) and reads each provider's usage block back, for `TonstClient(messages_fn=...)`. |
| `benchmarks/live_test_gemini.py` | The same live checks against the Gemini API: all tools vs. `select_tools()`, rolling compaction with Gemini Flash-Lite summaries and cache-aware mode, implicit cache hits, countTokens accuracy and latency. Prints an upper-bound cost estimate and asks first. Results in `live_test_gemini_results.json`. |
| `benchmarks/benchmark_free_features.py` | Offline benchmark for tool filtering, RAG chunk optimization and rolling compaction; plus the cache-aware compaction cost simulation for Anthropic and Gemini prices; results in `free_features_benchmark.json`. See [Free-feature benchmark](results.md#free-feature-benchmark-offline). |
| `tonst/client.py` | `TonstClient` — the public SDK surface that ties it all together. |
| `examples/demo.py` | Runnable demo against a mocked paid API call — no API key or network needed. Covers both the basic pipeline and prompt-caching structuring. |
| `examples/real_api_demo.py` | Real integration test against the actual `api.anthropic.com` endpoint — see [Running the real API test](#running-the-real-api-test). |
| `examples/cache_savings_demo_anthropic.py` | Measures REAL prompt-caching savings against `api.anthropic.com` using a realistically large reference document (clears the per-model minimum), printing actual `cache_creation_input_tokens` / `cache_read_input_tokens` from two consecutive calls. |
| `examples/cache_savings_demo_openai.py` | The same live-measurement pattern against `api.openai.com`, using `providers/openai.py`'s automatic-caching request shape and per-model discount table. |
| `examples/cache_savings_demo_gemini.py` | The same pattern against the real Gemini API, testing the *implicit* (automatic, best-effort) caching path via `providers/gemini.py`. |
| `examples/cache_savings_demo_gemini_explicit.py` | Gemini's *explicit* `CachedContent` path — deterministic, not best-effort. Creates a cache resource, then references it across several calls. Exists because live testing found implicit caching missed 18/18 real calls while explicit hit 3/3 (later 4/4) the moment billing was enabled; see `providers/gemini.py`'s docstring and `ROADMAP.md` for the full numbers. |
| `examples/cache_savings_demo_generic.py` | A runnable **template** for testing prompt caching against any provider tonst has no dedicated script for — works out of the box in a mocked dry-run mode; three clearly marked edits point it at a real provider and a real key. |
