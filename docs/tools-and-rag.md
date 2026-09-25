# Tool definitions and RAG context

Sending the model only the tool definitions and retrieved chunks a request needs.

[← Back to the README](../README.md)

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
