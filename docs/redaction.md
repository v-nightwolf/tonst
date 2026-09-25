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
