# Roadmap & Open Discussion

This file tracks bigger decisions that aren't settled yet — things we're
watching for signal on rather than committing to. It's written so it can be
pasted directly into a pinned GitHub issue once the repo has enough traffic
for that issue to actually get seen.

## Recently decided

### Semantic response caching — removed

An earlier version of this pipeline included a semantic cache: if a new
question was "close enough" in meaning to a previous one, it returned the
cached answer with zero tokens spent and no model call at all. Removed for
two reasons: (1) it requires unbounded local storage growth to stay useful
at real scale, and (2) similarity-based matching can silently return a
wrong or stale answer to a question that only superficially resembles a
previous one, with no visible sign of failure. Not planned to come back in
this form.

### Prompt-caching structuring — shipped

`tonst/cache_structuring.py` + `TonstClient.query_structured()` /
`redact_and_trim_parts()`. Important distinction from the removed cache
above: this never skips the real model call. It shapes a request (stable
content first and byte-identical across calls, variable content last, an
explicit `cache_control` breakpoint for Anthropic) so the *provider's own*
caching — which always re-verifies against the live model — can discount
the repeated portion. See the README's "Prompt-caching structuring"
section.

### Conversation history management — shipped

Two complementary pieces, both off by default:

- `trim.truncate_history()` — the plain fallback, always available: keep
  the system message(s) plus the last N turns, drop the rest.
- `tonst/compactor.py` (`HistoryCompactor` + `compact_history()`) +
  `TonstClient.query_messages(use_history_compaction=True)` — condense
  the dropped turns into one summary message instead of discarding them,
  using the same local model (via Ollama) as `local_model.py` and
  `redact_llm.py`.

This was researched directly against how Claude Code's `/compact`, Codex
CLI, and OpenCode implement history compaction. The key finding that
shaped the design: those tools' compaction runs against the *paid*
frontier model doing the actual task — Anthropic's own docs cite an
example where summarizing 180k tokens of history costs a one-time 180k
input + 3.5k output token charge. Running the equivalent step on a local
model instead means it costs latency and local compute, never a paid
token — a genuine structural advantage over the native tools, not just a
different implementation of the same idea.

The honest tradeoff, stated plainly rather than glossed over: a small
1–3B local model summarizes less faithfully than the frontier models
those tools use for their own compaction. This is why the fail-soft
behavior here is deliberately different from tonst's other optional
local-model steps — a failed or too-long/too-short summary (caught by a
length guard rail) falls back to plain truncation rather than "keep
everything," because dropping is the existing, already-accepted
baseline behavior, and a bad summary is worse than no summary. See the
README's "History compaction" section for the full design and the
redact-before-compact ordering (compaction never sees raw PII, only
already-redacted content, for the same reason a rewrite step in general
must not).

### Multi-provider caching support (beyond Anthropic) — shipped

`tonst/providers/openai.py` and `tonst/providers/gemini.py`, alongside
the existing Anthropic implementation in `cache_structuring.py`. Built
in response to the stated goal of launching tonst as a provider-agnostic
tool, not an Anthropic-only one — direct research into OpenAI's and
Gemini's actual caching mechanics (not assumed from memory) surfaced
real structural differences that shaped the design:

- OpenAI is automatic-by-default (ordering only, no marker) for nearly
  every model, with an optional explicit mode on GPT-5.6+ that behaves
  more like Anthropic's `cache_control` — including, new to that model
  generation, a write premium OpenAI never had before. Its cache-read
  discount is NOT one number: it ranges from 50% off to 98.75% off
  depending on the model, unlike Anthropic where one rate covers most
  of the lineup.
- Gemini has TWO separate caching mechanisms with different cost
  shapes, not one mechanism with an optional mode: an automatic
  "implicit" cache (same shape as OpenAI/Anthropic's read discount, no
  write cost), and a separate, developer-managed "explicit"
  `CachedContent` resource that bills the standard input rate once to
  populate, then an ongoing storage rent per hour it exists — charged
  whether or not it's ever read again. This is a genuinely different
  cost shape from anything Anthropic or OpenAI do, which is why it gets
  its own dedicated cost function rather than reusing the
  read/write-multiplier shape the other paths share.

Deliberately NOT built behind one shared interface across providers —
see `tonst/providers/__init__.py`'s docstring for why forcing a common
abstraction here would hide real differences instead of simplifying
anything. `CacheUsageReport` (already existing, from the Anthropic work)
is the one piece every provider's usage-parsing function shares, giving
callers a consistent shape to read regardless of which provider they
used.

**Honestly flagged, not yet closed:** unlike the Anthropic path (real,
live-tested against `api.anthropic.com` — see README), the OpenAI and
Gemini modules are built directly from each provider's current official
docs but have not yet been run against a real OpenAI or Gemini API key.
A few specific details are corroborated only by secondary sources or
carry a documented discrepancy between primary sources — see the
README's "Multi-provider support" section for the itemized list
(OpenAI's `cache_write_tokens` field path, Gemini's conflicting
minimum-token figures, and the fact that Gemini's storage-rent dollar
figures will go stale over time in a way the other providers'
multiplier-based pricing won't). Worth live-testing with real keys
before depending on either in production.

**Known asymmetry, not yet cleaned up:** Anthropic's implementation
still lives in `tonst/cache_structuring.py` rather than
`tonst/providers/anthropic.py`, for backward compatibility with the
existing public API built before this package existed. A future pass
could move it and keep thin backward-compatible re-exports — tracked
here rather than done opportunistically, to avoid an unplanned breaking
change to the public API in the same pass that added two new providers.

### Generic "any provider" adapter — shipped

`tonst/providers/generic.py` (`GenericCacheConfig` + config-driven
`check_cache_eligibility()` / `parse_usage()` / `estimated_cost_savings_percent()`)
plus `tonst/providers/presets.py` (real, verified configs for AWS
Bedrock's Converse API and Azure OpenAI's PTU-M tier). Built in direct
response to "make it work with any model, not just these 3" — three
hardcoded provider modules is still a hardcoded list, and Mistral,
Groq, Together, Fireworks, DeepSeek, xAI, Cohere, and any self-hosted
OpenAI-compatible server are all real gaps a three-provider tool would
have. The generic module is the honest answer to that: it can't know a
new provider's real pricing/minimums in advance, so it asks for them as
plain config instead of guessing, and defaults to a safe no-op
(0% savings, always eligible) rather than fabricating a discount that
was never confirmed for an unconfigured provider.

The two shipped presets aren't toy examples — they came out of
verifying a real, useful, and non-obvious fact: **whether a wrapped
platform needs its own config isn't decidable in general, only per
platform.** AWS Bedrock's Converse API reports Claude's cache usage
under different camelCase field names than Anthropic's own direct API
(same underlying model, different wrapper) — needs a preset. Bedrock's
other API (InvokeModel) passes Anthropic's response through completely
unchanged — needs nothing, reuse `cache_structuring.parse_anthropic_usage()`
directly. Azure OpenAI's standard tier matches OpenAI direct exactly —
needs nothing; its Provisioned-Throughput tier has an Azure-only
discount OpenAI's own table doesn't have — needs a preset. This
"sometimes yes, sometimes no, and you have to check" finding is now
documented in the README precisely so a future contributor doesn't
assume either blanket answer when adding platform #6, #7, #8.

A caveat carried over from the dedicated OpenAI/Gemini modules applies
here too: none of `generic.py`'s worked examples (Bedrock, Azure PTU-M)
have been run against a real API key yet — built from official docs,
logically consistent, not yet live-verified the way the core Anthropic
path was.

### Per-provider live-test scripts — shipped

`cache_savings_demo.py` was renamed to `examples/cache_savings_demo_anthropic.py`
and given three siblings: `examples/cache_savings_demo_openai.py`,
`examples/cache_savings_demo_gemini.py`, and `examples/cache_savings_demo_generic.py` —
built in response to feedback that the original single script's setup
instructions read as *the* way to test caching, when it was always
Anthropic-specific. There is deliberately no single generic live-test
script, because there is no single generic API to call — every provider
needs its own base URL, auth scheme, and request/response shape.
`examples/cache_savings_demo_generic.py` is the closest thing: a template that
runs out of the box in a mocked dry-run mode (proving the pipeline
logic end to end with zero setup) and documents exactly which three
edits point it at a real provider and a real key. See the README's
"Testing against a real provider" section for the full table.

Verified logically (eligibility check, request building, and usage
parsing all run correctly against realistic inputs — see the README)
for OpenAI and Gemini, but **not yet live-tested against a real key**
for either — this session's sandboxed environment could only reach
`api.anthropic.com`, not `api.openai.com` or the Gemini API, so
reachability itself is unconfirmed from here too. Worth running for
real with actual keys before fully trusting the numbers they'd report.

### CACHE_MINIMUM_TOKENS gap for Haiku 3.5 — found and fixed

Re-verified `cache_structuring.py`'s per-model minimum table against the
live Anthropic docs (in response to being asked "did we implement this
logic" from the docs page directly) and found `claude-haiku-3-5` was
missing from `CACHE_MINIMUM_TOKENS` entirely. Its real minimum is 2,048
tokens; being absent meant it silently fell back to
`DEFAULT_CACHE_MINIMUM_TOKENS` (1,024) — so a stable prefix between
1,024 and 2,048 tokens would have been reported `eligible=True` by
`check_cache_eligibility()` when Anthropic would actually process it
without caching at all, with no error to reveal the mismatch. Fixed by
adding the missing entry, along with three others (`claude-opus-4-1`,
`claude-opus-4`, `claude-sonnet-4`) that were also absent but happened
to match the 1,024 default already, so they never produced a wrong
answer — added anyway so the table is complete rather than
accidentally-correct in places. Two regression tests added:
`test_check_cache_eligibility_haiku_3_5_uses_its_own_higher_minimum`
(the actual bug) and `test_check_cache_eligibility_covers_previously_missing_models`
(locks in the other three as explicit entries). 82/82 tests passing.

**Still open**: "Claude Mythos Preview" (documented minimum: 2,048
tokens) is not in the table — its exact API model-id string wasn't
confirmed, so a guessed key would risk silently never matching rather
than fixing anything. Add it once the real model string is known.

The same re-check against the live docs also surfaced several
not-yet-implemented mechanics worth a future pass: multiple/tool
`cache_control` breakpoints (Anthropic allows up to 4 per request; tonst
places at most 2 today), a top-level automatic-caching flag, the
detailed `cache_creation` TTL breakdown in usage responses, the
multi-turn "advance the breakpoint forward each turn" pattern, and
`max_tokens=0` cache pre-warming. None of these are bugs — they're
scope tonst hasn't covered yet — tracked here rather than assumed done.

### Live-tested examples/cache_savings_demo_gemini.py with a real key — three findings

Ran the Gemini demo against a real API key end to end (2-call demo, a
6-call intensive aggregate run, and a direct explicit-caching probe).
Zero cache hits across all 8 real calls, which turned out to have three
separate causes, found one at a time by actually testing rather than
assuming:

1. **Wrong model, script-level bug.** The script targeted
   `gemini-2.5-flash`, which Google now returns a 404 for on new API
   keys ("no longer available to new users"). Fixed: switched to
   `gemini-3.6-flash` (the replacement Google's own error names) and
   expanded `REFERENCE_DOC` to clear that model's higher 4,096-token
   minimum.
2. **Token estimator gap, real correctness issue.** The `chars // 4`
   heuristic used by `check_cache_eligibility()` said ~4,549 tokens for
   a prompt Google's real tokenizer measured at only 4,026 — a ~12%
   overestimate that put the actual request under the true minimum
   while our own check said "likely eligible." Padded the reference
   document further for real margin (confirmed 4,606 real tokens on
   the next run), but the estimator itself is still optimistic near a
   threshold — worth a proper fix (e.g. a stricter safety margin, or
   a real tokenizer call when one's available) rather than just more
   padding. Not yet fixed at the estimator level, only worked around
   in this one demo.
3. **Free-tier cache quota is zero — the actual root cause.** Even
   with real tokens confirmed above the minimum, 8/8 calls still missed.
   A direct probe of the EXPLICIT caching path (`cachedContents.create`)
   settled it with a decisive error instead of more guessing:

       429 RESOURCE_EXHAUSTED: "TotalCachedContentStorageTokensPerModelFreeTier
       limit exceeded for model gemini-3.6-flash: limit=0, requested=4824"

   A free-tier key/project gets a confirmed zero-token cache storage
   quota — both implicit and explicit caching are structurally
   unusable on one, not just unlucky. This is now documented directly
   in `providers/gemini.py`'s module docstring and in the demo script's
   setup instructions, since it's a far more likely explanation for
   "caching isn't working" than a request-shaping bug or best-effort
   randomness.

   Along the way, also found and fixed a real bug in
   `build_cached_content_resource()`: the `model` field needs Google's
   full resource-name format (`"models/{model}"`), not a bare model
   id — confirmed against the live `cachedContents.create` schema.
   Sending the bare id would have failed regardless of billing tier.
   Three regression tests added (fully-qualified name, no double-
   prefixing if already qualified, system instructions in their own
   `systemInstruction` field instead of folded into `contents`). 85/85
   tests passing.

   **Resolved**: re-ran after enabling billing. Implicit caching still
   showed 0/6 hits — confirms it really is best-effort even with
   quota available, not just a free-tier artifact. Switched to the
   EXPLICIT path (deterministic, not best-effort) and got a real,
   guaranteed cache hit: created one `CachedContent` resource (4,824
   tokens, billed once at standard rate), then 3 calls referencing it
   each showed `cache_hit=True`, `cache_read_input_tokens=4824`
   (~99.5% of that call's input from cache). Computed real costs from
   the actual measured tokens at gemini-3.6-flash's published $0.75/1M
   input rate and confirmed 10% cache-read multiplier:
   - **Steady-state savings per cached read: 89.6%** — matches the
     predicted ~90% (1 — 0.1 multiplier) almost exactly.
   - **Blended savings across the whole run: 53.0% vs. 3 full-price
     calls** (i.e. what 3 real answers would have cost with vs.
     without caching) — lower than steady-state because the
     one-time populate cost and storage rent are pure overhead that
     answers no question by themselves; they get amortized across
     however many reads happen. A follow-up 4-read run measured 62.2%,
     confirming the trend: the blended figure climbs toward the 89.6%
     steady-state ceiling as more reads happen within the same TTL
     window.

     **Correction (caught on a follow-up run, Sept 2026)**: this entry
     originally reported 67.3% for the 3-read run, comparing against
     **4** full-price calls (treating the populate call as if it also
     answered a question). That was wrong -- the populate call
     produces no answer, so the honest no-cache baseline for N real
     answers is N full-price calls, not N+1. `estimated_explicit_cache_cost_savings_percent()`
     itself was always correct (its own tests already assumed N, not
     N+1); the error was in this writeup and in the demo script's
     printed explanation, both now fixed to match the function's real
     semantics.
   - **Token count: unchanged, as always** — every call still
     processed the full ~4,824-token prefix; only price per token
     changed on the cached portion. This is the same pattern confirmed
     earlier on Anthropic in this project, now confirmed on Gemini too.

   Practical implication for the tool: on Gemini, prefer the EXPLICIT
   path over implicit when a guaranteed, measurable saving matters —
   implicit caching missed 100% of the time in this testing (18 calls
   across three test sessions -- 8 before billing was enabled, 8 more
   immediately after, and 2 more on a later re-check -- all misses),
   consistent with Google's own "best-effort, no guarantee" framing,
   whereas explicit hit 100% of the time (3/3) once billing was on.

   **Promoted to the permanent repo**: the throwaway diagnostic script
   that produced this result was cleaned up (fixed a `KeyError` where
   it assumed a response always has `candidates[0].content.parts` —
   a truncated/reasoning-heavy response can legitimately have neither)
   and shipped as `examples/cache_savings_demo_gemini_explicit.py`, the deterministic
   sibling of `examples/cache_savings_demo_gemini.py`. It imports that sibling's
   `REFERENCE_DOC`/`MODEL`/`GEMINI_API_KEY` rather than duplicating them.
   README's "Testing against a real provider" section and files table
   updated accordingly.

### Four free-tier features — shipped (Sept 2026)

Chosen from a feasibility review of seven proposed ideas (see the
project's research notes). The other three were not built: a
semantic/exact response cache (already removed, see above), local-first
model routing (a different product that changes answer quality; the
local-model benchmarks argue against it on typical hardware), and a
prompt-injection scanner (a crowded category, and keyword detection
gives false confidence).
All four are free/MIT by design: they drive adoption. The paid tier
stays centered on the audit/compliance trail, team policy and support.

- **Rolling compaction** (`compact_history_rolling()`, `RollingSummary`,
  `query_messages(rolling_state=...)`). Fixes a real gap in the
  stateless compactor: it re-summarized every older turn on every call,
  paying local-model latency each turn and producing a summary that
  changed every turn, so it could never be cached. Rolling mode keeps
  evicted turns verbatim until they reach the threshold, then folds them
  into the existing summary once. Between folds the prompt is
  append-only.
- **Tool/MCP definition optimization** (`tool_optimizer.py`). Anthropic
  now supports deferred tool loading natively (tool search tool +
  `defer_loading`), so on Anthropic tonst just builds that request
  correctly. For other providers it filters locally with BM25.
  `ToolSession` is grow-only, because per-turn filtering otherwise
  invalidates the prompt cache (tools sit at the front of the prefix).
  Tool descriptions are never rewritten by the local model.
- **RAG context optimization** (`rag.py`, `query_rag()`). Deduplicates
  and optionally filters retrieved chunks by relevance or token budget.
  It only selects whole chunks and never rewrites one. Per-chunk local
  compression was left out on purpose, based on the ablation numbers.
- **Savings log + `tonst stats`** (`savings_log.py`). Opt-in local JSONL
  of per-call metrics. It never records prompt text, PII values or
  placeholder hashes (hashes are brute-forceable). Token counts are
  marked as estimates. This is the data layer for the hosted dashboard.
  The open-core line: this free log is deliberately *not* an audit
  record (no tamper-evidence or retention).

**Offline benchmark** (`benchmarks/benchmark_free_features.py`, hand-built but
realistic workloads, not real traffic; full table in the README):
tool filtering reached 100% recall and 83% fewer tool tokens on direct
requests (`top_k=5`, 36 tools). RAG dedupe alone cut 19% of context
tokens, or 63% with `top_k=4`, keeping the answer chunk 4/4 (small
sample). Rolling compaction made 4 local-model calls vs. 26 for
stateless, and cost less than half as much with prefix caching at a 10%
cache price, despite sending more raw tokens.

**A real flaw the benchmark caught, and fixed:** the first version of
`select_tools()` had only 50% recall on paraphrased requests. A
one-word coincidental match ("book a *slot*" vs. a free-time-*slots*
tool) beat the "nothing matched, keep everything" fallback and silently
dropped the needed tool. Fixed with `min_matched_terms=2`: the best
match must share at least two distinct words with the request, or
nothing is filtered. Recall on paraphrased requests went to 100%. The
same rule was applied to `ToolSession` growth and RAG filtering.
135/135 tests.

**Seven fixes from hands-on testing on a MacBook Air (same day):**
(1) compaction timeout is now its own setting, `compaction_timeout`,
default 60s (a ~7k-token fold timed out at the shared 8s); (2) a failed
rolling fold keeps the turns verbatim and retries once before dropping
(the first version lost 14 turns to one cold-start timeout); (3) the
first fold now uses the structured Goal/Decisions/Key facts/Open items
prompt (it used to produce free-form prose); (4) `query_rag()` reports
`chunk_filter_skipped`; (5) the savings log records summary
updated/reused/failed; (6) dropped-without-summary history is reported
as `history_tokens_lost` and shown separately in `tonst stats` instead
of silently inflating "saved"; (7) `tonst stats` shows median/max
overhead (one 8s timeout had pushed the mean to 1.3s). 143/143 tests.
Live measured: a ~1k-token fold took ~5s with gemma2:2b and the next
turn reused it in 0 ms. The summary kept every key fact of the test
conversation.

`benchmarks/live_test_free_features.py` added: the real-API test for tools (all
vs. filtered vs. deferred, with a correct-tool check), rolling vs.
stateless compaction cache reads, and estimate accuracy.

**First live API result (5 direct tasks, Sonnet 4.6):**
- `select_tools(top_k=5)`: 5/5 correct, 72% less billed input, 64% lower cost.
- Deferred loading: 4/5 correct, 41% less input, 26% lower cost; roughly
  2× output tokens from the search step. One miss (searched once, then
  answered without a tool call). The script didn't save why, so it now
  records the stop reason, reply text and search results for every miss.
- Real billed input was 1.79× the chars/4 estimate: hidden tool-use
  prompt plus token-dense JSON. Added optional `AnthropicTokenCounter`
  (free `count_tokens` endpoint) for `TonstClient(token_counter=...)`
  and `select_tools(token_counter=...)`. It only ever sees redacted
  text (enforced by a test), is fail-soft, and its time is reported as
  `counting_ms`. 150/150 tests.

**Full live tools run (30 tasks, Sonnet 4.6):**
- All tools: 23/30, $0.39.
- `select_tools`: 23/30, $0.19. It kept 22 of the 23 all-tools successes; the other was a reasonable list-tables-first.
- Everything deferred: 18/30, $0.27, with 2× output tokens.
- Deferred failure mode: Claude sent the tool search the TOPIC ("billing outage", "on-call runbook") instead of the capability, found nothing, and told the user the data didn't exist.
- Fix: `build_anthropic_deferred_tools(keep_search_tools_loaded=True)` by default, plus `DEFERRED_TOOLS_SYSTEM_HINT`.
- Test fixes: underspecified tasks now include their content; outcomes are scored as correct / acceptable / asked / wrong tool / no tool; `ACCEPTABLE_FIRST_STEPS` covers reasonable alternatives; the live script compares `deferred` (fixed) and `deferred_plain` (old).
- `count_tokens` matched billed input on 60/60 calls; chars/4 was 1.77× low.
- Guidance: `select_tools` up to ~50 tools; deferred (with the fixes) for large catalogs. 151/151 tests.
- **Re-run after the fixes (30 tasks):**
  - all tools: 30/30, $0.40
  - `select_tools`: 30/30, $0.19 (−51%; −64% on direct requests)
  - deferred with tonst defaults + hint: 29/30, $0.33 (−18%)
  - everything deferred: 25/30, $0.30
  - The fix removed every "search found nothing" failure. Its one new miss: it used the loaded `github_search_issues` to "read issue 482" instead of searching for `github_get_issue`.
  - Deferral doesn't pay at 36 tools: calls that need a search average ~3,300 input tokens.
  - count_tokens matched billed input 60/60 again.
- **Conclusion:** `select_tools` is the recommended default for catalogs up to ~50 tools; deferred is for very large catalogs only.

**Live compaction run (12 turns, Sonnet 4.6, gemma2:2b):**
- Rolling: $0.0163 vs. stateless $0.0185 (−12% overall; about −47% on the history portion, since the cached system prompt dominates a short chat).
- 1 local summary vs. 4 (7 s vs. 17 s).
- Rolling sent 4% more raw tokens but cost less, as the offline benchmark predicted.
- Cache reads grew every turn in rolling mode and stayed flat in stateless mode.
- Summary: right structure and core facts, but a settled decision (express upgrade) was listed under Open items, and `---` fences were echoed from the prompt.
- Fixed: `_clean_summary()` strips echoed fence/label lines, and the prompt defines Decisions vs. Open items explicitly. 153/153 tests.
- chars/4 was within 10% for prose (1.1×), so the 1.77× miss is JSON-specific.
- Worth measuring next: a longer chat (`--turns 24`) to see the gap grow.

**Latency + cost vs. no compaction (24 turns, threshold 600):**
- Costs: none $0.0295, stateless $0.0330, rolling $0.0291.
- tonst's own code: ~0.1 ms/turn. API p50 ~1.7 s in every mode, so prompt length didn't change API latency.
- Stateless added 2.5-5 s on 7/24 turns (p95 total 5.5 s vs. 2.9 s). Rolling added 4.7 s on 1 turn.
- **Conclusions:** (1) with provider caching, stateless is counterproductive, so recommend rolling only; (2) compaction doesn't save money on short chats (cached history re-reads are cheap), and pays only for long histories / context limits, hence the 3,000-token default threshold.
- Test costs reconciled: the script's per-run costs sum to $2.40 for all live testing, matching the API console.

**Background summaries:** `compact_history_rolling(defer_fold=True)` returns a `FoldJob`; `run_fold_job()` applies it thread-safely. `query_messages(background_summary=True)` runs the summary on a background thread in parallel with the API call, removing the local model from response time. The next call uses the summary. `wait_for_background_work()` is provided.
- Stale jobs are discarded: a generation counter plus a fingerprint of the covered messages. A test caught that start_count/previous_summary alone couldn't tell a new conversation from the old one.

**Ollama num_ctx fix (found before it bit):** Ollama's default context (2k-4k) silently drops the START of longer prompts, so summaries at the 3,000-token threshold would have lost the oldest turns.
- `tonst/ollama_util.py` sizes num_ctx per request, capped by TONST_OLLAMA_MAX_CTX (8192).
- Compaction and compression refuse over-long prompts; LLM redaction warns.
- `benchmarks/live_test_free_features.py --long` added: ~500-token tool outputs, history ~13k tokens at the real 3,000 threshold; modes none / rolling / rolling_bg.
- 161/161 tests.

**Long-history live run (12 turns, ~12.6k tokens):**
- Cost: none $0.0719, rolling $0.0687 (−4%), rolling_bg $0.0791 (+10%; the summary landed 2 turns later with a bigger cache rewrite).
- Latency: the blocking summary took 12.3 s (turn total 13.9 s); background had zero added latency (max 2.8 s).
- Cost saving is on cache reads only (~26%/turn after the summary; ~2-3 turns to pay back the rewrite). Compaction is mainly for context limits, with a modest cost win on long chats.
- **Critical finding:** gemma2:2b replied to the customer instead of summarizing ~3k tokens of tool output. It passed all guard rails and would have injected false claims.
- Fixed: a heading-structure guard (at least 3 of 4 headings, else rejected); the task repeated after the conversation; each message clipped to 700 chars in the summarizer input. 165/165 tests.
- Next: `--long` now defaults to 24 turns (~$0.76, estimate calibrated on the real 12-turn cost); re-run to check summary quality and cost over a longer chat.

**24-turn long-history run (after the fixes):**
- Cost: none $0.186, rolling $0.147 (−21%), rolling_bg $0.168 (−10%). Tokens −50% / −42%.
- Latency: rolling p95 10.9 s; rolling_bg p95 2.0 s (one 9.5 s API-side spike, tonst 0.7 ms).
- No garbage summaries; one attempt was rejected and retried correctly.
- But the local summary lost the white colour, the evening slot and case CS-20931.
- Fixes: **pinned references** (deterministic extraction of placeholders, #-numbers, ticket codes and amounts; carried verbatim; kept even when turns are dropped; persisted in `RollingSummary.to_dict`), a **content check** (Key facts must be non-empty), and an optional **`AnthropicSummarizer`** (Haiku 4.5, `TonstClient(compaction_summarizer=...)`, redacted text only, usage/cost tracked).
- Live script: `--long` defaults to none / rolling_bg / rolling_bg_haiku; the Haiku cost is added to its mode; a new **fact recall** metric scores facts from summarized turns. 171/171 tests.

**Local vs. Haiku summaries (24 turns, background, 2026-09-25):**
- Cost: none $0.186, local $0.189 (+1.6%), Haiku $0.181 (−2.7%, Haiku's $0.011 included).
- Fact recall: local 5/8, **Haiku 8/8**. Pinned references carried #4471 and CS-20931 in both.
- Latency was the same in all three modes.
- Why cost barely moved: with prompt caching, old history costs 0.1× to re-read, and each summary forces a 1.25× cache rewrite of everything after the system prompt. The last summary (3 turns before the end) never paid back.
- Fix: **cache-aware compaction** (`compaction_cache_aware=True`, `estimate_fold_payback`, `expected_remaining_turns`, `compaction_max_history_tokens`, report field `history_fold_postponed`). A due summary waits until its estimated payback fits the turns left, which defaults to half the turns so far.
- Offline simulation (in the benchmark, calibrated to the live run): it removes the +8–15% loss in 10–12-turn chats and is neutral from 20 turns on.
- With caching, compaction saves money only past ~20 turns (−18% at 40, −55% at 100); below that it's for fidelity and context headroom.
- Live script: new mode `rolling_bg_haiku_aware`. 176/176 tests.
- **Live 12-turn check:** none $0.0719; Haiku at threshold $0.0826 (+14.9%; its summary landed 2 turns later, 1 turn before the end); cache-aware $0.0719 (0.0%, postponed on 4 turns, no Haiku call). The simulation predicted +7.7% / 0%, so the real loss from an early summary was larger than modeled.

**Gemini (3.8 Flash, thinking low) — tools, 30 tasks (2026-09-25):**
- all 29/30 $0.0738; filtered 29/30 $0.0331 (−55%, direct −70%); 29/29 paired.
- Same miss in both modes. Paraphrased tasks fell back to all tools (safe, no saving).
- 0% implicit cache hits: prompts (~2.9k tokens) were under Flash's 4,096 minimum.
- chars/4 within 2% on Gemini (1.77× low on Claude).
- countTokens exact 60/60 but ~320 ms per call; select_tools 2.3 ms. Run cost $0.11.
- Added: `GeminiSummarizer`, `GeminiTokenCounter`, `compaction_cache_pricing` ("anthropic"/"gemini"/tuple), `benchmarks/live_test_gemini.py`, Gemini provider in the cost simulation. 180/180 tests.

**Gemini compaction, 20 turns (2026-09-25):**
- none $0.1329 (35% cached); Flash-Lite summaries $0.1128 (−15.1%, 8/8 facts, p95 2.7 s vs 4.0 s); cache-aware $0.1389 (+4.5%).
- Gemini's implicit cache hit 0% from turn 4 to turn 15, and only started at ~18k tokens.
- Cache-aware mode wrongly assumed caching and postponed the summary for 6 turns.
- Fix: `RollingSummary.observe_cache_usage(prompt, cached)` keeps an observed hit rate (share of the previous prompt reused; skips turns after a summary change; EMA 0.3; needs 2 observations; persisted). `estimate_fold_payback(cache_hit_rate=)` prices misses at the write price.
- Replay of this run: now folds at turn 9, same as plain rolling (−15%). Unchanged when the cache always hits. 183/183 tests.
- Confirmed live: cache-aware $0.1206 (−9.3% vs. none), 0% observed hit rate, folds at turns 9 and 15, 0 postponed, 8/8 facts.

**Still not measured on real traffic:** tool savings on a real
MCP-heavy agent, dedupe rates on a real retriever, and real cache-hit
rates for rolling compaction against a live provider.

## Open discussion topics

### More token-reduction techniques — planned, not yet built

Next up, in rough priority order: schema compaction (terser field names/
structure on outgoing structured-output requests, expanded back
transparently) and a text-vs-image tile-cost router (pick whichever
encoding costs fewer tokens for a given chunk of context, given a
legibility floor). Tool/function-definition trimming and RAG context
optimization have shipped (see above).

### Java support — interested?

**Status: not planned, actively watching for demand.**

tonst is Python-only today. A Java port isn't ruled out, but it isn't queued
up either — this is a "tell us if you need it" item, not a "someday maybe"
that quietly rots.

**What would trigger it:**
- 2–3 concrete requests for a Java/JVM version, or
- Comments/issues referencing Spring or other enterprise-Java use cases
  where tonst's approach (local redaction + prompt-caching structuring in
  front of a paid LLM call) would apply

If you're reading this because you want Java support: comment on the pinned
issue (or open a new one) and say what you're building. That's the signal
we're waiting for.

**If/when it happens:** scope v1 to just the redaction + caching-structuring
core (the genuinely differentiated part — see `redact_llm.py` and
`cache_structuring.py` in the Python version), not a full port of
everything. A smaller, focused Java v1 validates demand faster than
porting the whole surface area up front.

---

*(Add future open-discussion topics below this line.)*
