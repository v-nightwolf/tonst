# Prompt caching and providers

Structuring requests so the provider's own prompt cache discounts the repeated part, per provider.

[← Back to the README](../README.md)

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
  see the [Testing against a real provider](#testing-against-a-real-provider) section below for a live measurement against the
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

## Testing against a real provider

`real_api_demo.py` and `demo.py` prove the pipeline is wired correctly,
but neither one tells you whether prompt caching actually saved
anything — that requires a stable prefix past the per-model minimum
(see [Prompt-caching structuring](#prompt-caching-structuring)) and reading the real `usage`
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
(see [Any other provider](#any-other-provider)) and works out of the box with
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

**What's been live-tested vs. what hasn't:** the Anthropic path was
confirmed against the real API (see [Testing against a real
provider](#testing-against-a-real-provider)). Gemini has been tested live
too: the implicit-caching benchmark in [results](results.md#measured-results)
(`gemini-3.1-flash-lite`, 24 calls) and the September 2026 tool and
compaction runs (`gemini-3.8-flash`, `live_test_gemini.py`). The OpenAI
module was built from OpenAI's current documentation and is consistent
with it, but has **not** been run against a real OpenAI key yet. A few
specific things are worth verifying before depending on them in
production:
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
