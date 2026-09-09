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

`cache_savings_demo.py` was renamed to `cache_savings_demo_anthropic.py`
and given three siblings: `cache_savings_demo_openai.py`,
`cache_savings_demo_gemini.py`, and `cache_savings_demo_generic.py` —
built in response to feedback that the original single script's setup
instructions read as *the* way to test caching, when it was always
Anthropic-specific. There is deliberately no single generic live-test
script, because there is no single generic API to call — every provider
needs its own base URL, auth scheme, and request/response shape.
`cache_savings_demo_generic.py` is the closest thing: a template that
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

### Live-tested cache_savings_demo_gemini.py with a real key — three findings

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
   and shipped as `cache_savings_demo_gemini_explicit.py`, the deterministic
   sibling of `cache_savings_demo_gemini.py`. It imports that sibling's
   `REFERENCE_DOC`/`MODEL`/`GEMINI_API_KEY` rather than duplicating them.
   README's "Testing against a real provider" section and files table
   updated accordingly.

## Open discussion topics

### More token-reduction techniques — planned, not yet built

Next up, in rough priority order: schema compaction (terser field names/
structure on outgoing structured-output requests, expanded back
transparently), a text-vs-image tile-cost router (pick whichever encoding
costs fewer tokens for a given chunk of context, given a legibility floor),
tool/function-definition trimming for agentic workloads, and RAG-style
context retrieval (inject only the relevant chunk of a reference doc
instead of the whole thing, starting with simple keyword-based selection).

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
