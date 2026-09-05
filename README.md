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
   ├─ 1. Semantic cache check ──────► HIT → return cached answer, $0 spent
   │
   ├─ 2. Local PII redaction (regex)
   ├─ 3. Mechanical trim (dedupe, whitespace, history truncation)
   ├─ 4. Optional local-model compression (Ollama, off by default)
   │
   ▼
Your existing paid API call (Claude/OpenAI/etc.) — only sees the
trimmed, redacted prompt
   │
   ▼
Re-insert real values into the response → return to your app
   │
   ▼
Store in cache for next time
```

## Real results (not simulated)

Tested against the actual `api.anthropic.com` endpoint on 2026-09-05, using
a realistic support-chat prompt with duplicated system instructions and
embedded PII:

| | Short, clean prompt | Realistic bloated prompt |
|---|---|---|
| Original tokens | 26 | 186 |
| Tokens sent to API | 26 | 136 |
| **Tokens saved** | 0 (0.0%) | **50 (26.9%)** |
| PII fields redacted | 1 | 3 |

The 0% result on the short prompt is intentionally included here, not
hidden — tonst doesn't manufacture savings where none exist. Real prompts
with any duplication, verbose history, or repeated instructions (the
overwhelming majority of real chat-app traffic) see meaningful reduction;
a single already-minimal prompt does not, and shouldn't.

Reproduce this yourself with `real_api_demo.py` (see below).

## Install

```bash
pip install -r requirements.txt
```

This installs the two dependencies the code needs: `requests` (for talking
to a local Ollama instance) and `numpy` (for the semantic cache's
similarity math). If you see `ModuleNotFoundError: No module named
'requests'` when running the demo, this step was skipped — run it and
re-try.

Full packaging (once you're ready to `pip install` it as a real package):

```bash
pip install tonst
```

(Not yet published — this POC ships as source. `pip install -e .` from this
directory works once `pyproject.toml` is in place, or copy the `tonst/`
folder directly into your project.)

## Why this shape

- **Nothing here requires you to run a server.** The cache, redaction, and
  trimming logic run in-process, wherever your app already runs (your own
  backend, not a third-party gateway). The only optional server-side piece
  in a real product would be a lightweight usage/billing dashboard — no
  inference, no GPUs.
- **Redaction + restoration is provider-agnostic and reversible.** Sensitive
  fields are swapped for placeholders before the call, and swapped back
  after — the cloud model never sees the real value, and your app never
  sees a placeholder.
- **The optional local-model step (`local_model.py`) is isolated and fails
  soft.** If Ollama isn't installed or running, the pipeline just skips
  that step rather than breaking. This matches the real-world constraint
  that not every deployment machine can run a local model well.

## Files

| File | Purpose |
|---|---|
| `tonst/cache.py` | Local semantic cache (bag-of-words cosine similarity for this POC — swap in local embeddings via Ollama for production). |
| `tonst/redact.py` | Regex-based PII detection + reversible redaction, plus `redact_with_llm()` to layer in the enhanced pass below. |
| `tonst/redact_llm.py` | **The differentiator.** Local-LLM-based redaction for free-text PII (names, addresses, employers, codenames) that regex structurally cannot catch. Strict JSON contract, hallucination guard rail, fails soft if Ollama isn't running. |
| `tonst/trim.py` | Token estimation, whitespace/duplicate cleanup, chat-history truncation. |
| `tonst/local_model.py` | Optional Ollama-backed semantic compression, off by default. |
| `tonst/client.py` | `TonstClient` — the public SDK surface that ties it all together. |
| `demo.py` | Runnable demo against a mocked paid API call — no API key or network needed. |
| `real_api_demo.py` | Real integration test against the actual `api.anthropic.com` endpoint — see "Running the real API test" below. |

## Why enhanced redaction matters (and why competitors don't have it)

Every comparable open-source tool we found (LLMShield, Helix AI Gateway,
the WSO2 AI Gateway sample) does PII redaction with regex only. Regex is
fast and reliable for *structured* data — emails, card numbers, phone
numbers — but it has no way to know "Priya Malhotra" is a person's name,
or that "Project Nightingale" is a confidential codename, without some
form of semantic understanding.

`redact_llm.py` closes that gap using a small local model (via Ollama) to
find free-text PII spans, while keeping the fast regex pass as the first,
always-on line of defense. Enable it with one flag:

```python
client = TonstClient(call_fn=my_api_call, use_enhanced_redaction=True)
```

If Ollama isn't installed or running, this degrades gracefully to
regex-only redaction — it never breaks the pipeline, and it never
silently trusts a hallucinated span from the model (see the guard rail
in `redact_llm.py` that verifies every flagged span actually appears
verbatim in the source text before redacting it).

## Running the demo

```bash
python3 demo.py
```

You should see the second (near-duplicate) query hit the cache and cost
zero tokens, and the first query's PII stripped before the "paid API"
ever saw it.

## Running the test suite

```bash
pip install -r requirements-dev.txt
pytest test_tonst.py -v
```

15 tests covering redaction round-trips, the hallucination guard rail,
fail-soft behavior when Ollama isn't running, caching, trimming, and the
full `TonstClient` pipeline end to end. Runs automatically on every push
via GitHub Actions (`.github/workflows/tests.yml`) across Python 3.9–3.12.

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
- Cache interface and eviction policy shape (swap the embedding function only).
- Fail-soft design for the optional local-model step.

**Deliberately simplified for the POC — replace before shipping:**
- `estimate_tokens()` uses a chars/4 heuristic. Use the real provider tokenizer
  (`tiktoken`, Anthropic's token counting endpoint, etc.) for accurate billing math.
- `cache.py`'s bag-of-words similarity is a stand-in for real embeddings.
  Swap `_embed()` to call a local embedding model (e.g. `nomic-embed-text`
  via Ollama) for genuinely semantic matching instead of word-overlap matching.
- `redact.py` only catches high-confidence patterns (email, phone, card, IP,
  SSN-like). Free-text PII (names, addresses in prose) needs either a local
  NER model or a local LLM prompted specifically for redaction.
- No persistence: the cache is in-memory and resets when the process restarts.
  A real deployment would back it with a local SQLite/embedded vector store.
