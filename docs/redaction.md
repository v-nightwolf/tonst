# PII redaction

How tonst keeps personal data away from the cloud model, and how to choose a redaction backend.

[← Back to the README](../README.md)

## Design principles

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

## Choosing a redaction backend

Several open-source LLM gateways we looked at (LLMShield, Helix AI
Gateway, the WSO2 AI Gateway sample) redact PII with regex only.
Dedicated PII toolkits such as Microsoft Presidio and LLM Guard do use
NER models; tonst's aim is to run that kind of detection in-process, as
one step of the request pipeline, with placeholders that restore
automatically and stay cache-stable. Regex is fast and reliable for
*structured* data — emails, card numbers, phone
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


## Placeholders and extra detectors (tonst)

**Keyed placeholders.** The hash in `[[EMAIL_3f2a91c0]]` is an HMAC keyed
with a secret that never leaves your machine: `TONST_PLACEHOLDER_KEY` if
set, otherwise `~/.tonst/placeholder.key` (created on first use, 0600).
A bare hash of the value would let anyone holding the prompt confirm a
guessed email or phone number offline; a keyed hash doesn't.

**Readable placeholders.** `TonstClient(placeholder_style="readable")`
produces `[[EMAIL_1]]`, `[[NAME_2]]`, numbered in order of first appearance
and kept for the life of the client, so a whole conversation uses one
numbering. Unlike the hash style they aren't stable across processes.

**Secrets (always on).** Anthropic/OpenAI keys, AWS access key IDs, GitHub
and Slack tokens, Google API keys, JWTs, PEM private keys, and the value in
`password=` / `api_key:` / `client_secret=`-style assignments →
`[[SECRET_…]]`.

**Opt-in categories.** `TonstClient(extra_redaction=["ACCOUNT_ID", "MONEY"])`
(or `redact(text, extra_categories=[...])`):

- `ACCOUNT_ID` — the number after account / customer / client / member /
  policy / invoice / order / ticket (must contain a digit).
- `MONEY` — amounts with a currency symbol or code (`$14,821.32`, `₹12 lakh`,
  `149 rupees`, `20 USD`).

Off by default because the model then can't see or calculate with those
values; the answer-quality benchmark decides whether that trade-off is worth it.

## People, emails and secrets (tonst, Phase 3)

These come from reading the answers in the 100-prompt benchmark
(`experiments/privacy_quality`), where they were the main remaining causes
of lower-quality answers.

**One placeholder per person.** Names found by GLiNER or the local LLM go
through `tonst/names.py`: "Omar Haddad" → `[[NAME_1]]`, a later "Omar" →
`[[NAME_1.first]]`, "Dr. Haddad" → `Dr. [[NAME_1.last]]`. The model can write
"Hi [[NAME_1.first]]", which restores to "Hi Omar". First and last names of
every detected person are also matched on their own, which catches mentions
the detector missed. A TonstClient remembers people across calls; if two
people share a first name, a bare first name gets its own placeholder rather
than a guess.

**Split emails.** `a.b@veltrix.io` → `[[EMAIL_1]]@[[DOMAIN_1]]`: both parts are
hidden, but addresses at the same domain share `[[DOMAIN_1]]`, so "group
these by company" still works. `[[EMAIL_1]]` alone restores the whole address.
`TonstClient(email_style="whole")` restores the old one-token form.

**Secrets.** API keys, tokens and passwords are never sent.
`report.secrets_withheld` counts them and a warning is logged.
`secret_notice=True` appends a one-line notice to the answer (the provider
can no longer warn the user itself); `restore_secrets=False` keeps the real
value out of the answer too.

**Restore fixes.** "Project [[CODENAME_1]]" with the value "Project Marigold"
no longer becomes "Project Project Marigold". `[[NAME_1.first]]` is never
broken up by `[[NAME_1]]`.

## Addresses, recall safety nets, contact owners, streaming (tonst, Phase 3)

**Addresses (always on).** Regex patterns for number-first streets
("1180 Folsom Street, San Francisco, CA 94103", "221B Baker Street, London
NW1 6XE"), street-first forms ("Rua Augusta 1520", "Friedrichstraße 88, 10117
Berlin"), "17 Rue de Rivoli, Paris" style, and Indian "Flat/House/Plot ...,
City PIN" → `[[ADDRESS_n]]`. Unusual formats can still be missed.

**Name safety net.** With the GLiNER or Ollama backend, names in positions
that almost always mean a person — after a role ("my manager Daniel Okafor"),
a greeting ("Dear Priya") or a field label ("Customer: Priya Nair") — are
added to whatever the detector found.

**Remembered entities.** A TonstClient remembers companies and codenames it
has detected, so a later mention the detector misses is still hidden.

**Contact owners.** When an email or phone sits right next to a person
("[[NAME_1]] ([[EMAIL_1]]@..., [[PHONE_1]])", "[[NAME_3]] on [[PHONE_3]]"), the
hint adds "Contact details: [[PHONE_1]] belongs to [[NAME_1]]". Only
placeholders are linked; deliberately narrow, so "tell [[NAME_1]] to call
[[PHONE_2]]" is not linked.

**Streaming.** `StreamRestorer(mapping)` (or `result.stream_restorer()`)
restores a streamed answer chunk by chunk, holding back only a tail that
could still become a placeholder. Known limit: the "Project Project"
de-duplication can't apply across chunk boundaries.
