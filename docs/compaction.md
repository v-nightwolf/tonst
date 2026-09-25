# History compaction

Keeping long conversations inside a token budget without silently losing what was said.

[← Back to the README](../README.md)

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
600, `benchmarks/live_test_free_features.py --part compaction`, 2026-09-24):

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
  `benchmarks/live_test_free_features.py --part compaction --long` (~500 tokens of
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
  user. The local model stays the default. `benchmarks/live_test_free_features.py
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
Sonnet 4.6 prices, cache rewrite modeled; `benchmarks/benchmark_free_features.py`,
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
`benchmarks/live_test_gemini.py` measures: tools (all vs. filtered), compaction
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
