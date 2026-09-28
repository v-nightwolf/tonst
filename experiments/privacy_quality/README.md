# Answer-quality benchmark

Does hiding sensitive data from the AI make its answers worse? This runs
120 realistic work prompts (support replies, contracts, HR notes, invoices,
config files with keys, meeting notes, small tables, translations) through
Claude and Gemini, once as-is and once through tonst, and compares.

All people, companies, numbers and keys in `cases.py` are invented.

## Run

From the repo root, with `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in `.env`
and GLiNER installed (`pip install -e ".[gliner]"`):

```bash
# free dry run: fake model, no network (checks the plumbing and leakage)
python experiments/privacy_quality/run.py --fake --limit 10

# small paid run first (~10 cases, prints a cost estimate and asks)
python experiments/privacy_quality/run.py --limit 10

# full run
python experiments/privacy_quality/run.py
```

A run can be interrupted and resumed: re-run with `--run-name <name>` and
finished calls are skipped. Options: `--providers anthropic`,
`--variants original,readable_hint`, `--categories finance,data`,
`--backend regex|gliner|ollama`, `--no-judge`.

## Variants

| variant | what the provider sees |
|---|---|
| `original` | the prompt as written (baseline) |
| `hash` | tonst with `[[NAME_3f2a91c0]]` placeholders |
| `readable` | tonst with `[[NAME_1]]` placeholders |
| `readable_hint` | readable + one sentence saying the `[[...]]` tokens are placeholders to copy verbatim |
| `hash_hint` | hash + the hint (stable placeholders, best for prompt caching) |
| `readable_hint_extra` | readable_hint + account/invoice IDs and money amounts hidden too |

Default: `original,hash_hint,readable_hint` (the no-hint variants were settled
by the first full run: Claude needs the hint). Hint variants also turn on
`secret_notice`.

The `heavy` category (10 cases: re-quoted email threads, log dumps, padded
transcripts, RAG with duplicate chunks) runs through the same client with
tonst's savings features, and the report adds a table of input-token and
cost change on those cases.

The `holdout` category (10 cases) was written after the Phase 3 detector and
name rules, with new people, companies, phone and address formats and
phrasings, so it shows whether those rules generalise or were fitted to the
main set. The report has a separate hold-out table listing exactly what leaked.

## What's measured

- **Leak rate** — share of each case's sensitive values that reached the
  provider, by type, read from the exact text sent. Addresses have no
  detector yet, so they are expected to leak.
- **Restore failures** — answers with a placeholder left over after
  restoring (`[[NAME_1]]`, or `NAME_1` if the model dropped the brackets).
- **Expected values** — share of must-have values in the final answer:
  the customer's name in a reply, the right invoice total, the right ID.
  Money is compared as a number.
- **Judge** — Claude compares the original answer with the masked one for
  the same provider, blind and in random order, scoring each 1–10.
- **Δ input tokens** and **cost** from the providers' own usage numbers.

Results land in `results/` (gitignored): `<run>.calls.jsonl`,
`<run>.judge.jsonl`, `<run>.summary.json` and `<run>.report.md`.

## Results: tonst 0.2.0 (2026-09-28)

Release run, `--run-name final`: 120 cases (100 main, 10 heavy, 10 hold-out) ×
Claude Sonnet 4.6 and Gemini 3.8 Flash, masking backend GLiNER, judge Gemini
3.8 Flash. Total cost about $2.50. The tables below are the runner's own report
(all 120 cases); the main-100 split and the comparison with the previous
development run (v3, same cases, before the note to the model was shortened)
are computed from the saved results.

**Main 100 prompts, judge score change (masked − as-is):**

| | Claude hash (default) | Claude readable | Gemini hash (default) | Gemini readable |
|---|---|---|---|---|
| Release run | −0.08 | −0.47 | −0.38 | −0.31 |
| v3 (development) | −0.46 | −0.50 | −0.42 | −0.55 |

**Heavy cases, input tokens vs. as-is (hash / readable):**

| | Claude | Gemini |
|---|---|---|
| Re-quoted email thread | −41% / −51% | −33% / −50% |
| Padded meeting transcript | −44% / −51% | −34% / −47% |
| Log dump (an email on every line) | +9% / −2% | +1% / −16% |
| Small RAG sets | −6% to +5% / −13% to −4% | +12% to +15% / −3% to −1% |

**Reading the leak table:** `MONEY` and `ACCOUNT_ID` are visible by default
(hiding them is opt-in via `extra_redaction`, because models can't calculate
with hidden amounts), so their "leaks" are by design. Every other category was
0% except one company, "the Brightwell Health clinic", in one prompt (2 of 480
masked prompts).

**Known limits seen in the answers:** hidden names can't be transliterated
(e.g. into Devanagari), gender can't be inferred from a placeholder, and Claude
occasionally mixes up or comments on whose contact details are whose.

### Runner report

120 cases, backend gliner, judge gemini-3.8-flash

| provider | variant | expected values | leak rate | restore failures | judge (orig → masked) | masked better / tie / worse | Δ input tokens | cost |
|---|---|---|---|---|---|---|---|---|
| anthropic | hash_hint | 100% | 18% | 0 | 9.2 → 9.11 | 33 / 32 / 55 | 94.9 | $0.55 |
| anthropic | original | 98% | 0% | — | — | — | — | $0.59 |
| anthropic | readable_hint | 99% | 18% | 0 | 9.29 → 8.72 | 20 / 35 / 65 | 66.0 | $0.50 |
| gemini | hash_hint | 100% | 18% | 0 | 9.7 → 9.3 | 18 / 45 / 57 | 111.7 | $0.14 |
| gemini | original | 99% | 0% | — | — | — | — | $0.15 |
| gemini | readable_hint | 100% | 18% | 0 | 9.67 → 9.35 | 17 / 56 / 47 | 63.8 | $0.13 |

Heavy workloads only (long threads, logs, transcripts, RAG with duplicate chunks), vs the same prompts sent as-is:

| provider | variant | cases | input tokens | cost | judge (orig → masked) |
|---|---|---|---|---|---|
| anthropic | hash_hint | 10 | -8.4% | -8.2% | 9.8 → 9.2 |
| anthropic | readable_hint | 10 | -18.8% | -16.3% | 9.8 → 8.4 |
| gemini | hash_hint | 10 | -6.7% | -10.0% | 10.0 → 9.45 |
| gemini | readable_hint | 10 | -23.2% | -28.0% | 10.0 → 9.55 |

Hold-out cases only (new people, companies and formats not used while tuning the rules):

| provider | variant | cases | leak rate | must-have values | judge (orig → masked) | what leaked |
|---|---|---|---|---|---|---|
| anthropic | hash_hint | 10 | 0% | 100% | 8.7 → 9.0 | nothing |
| anthropic | readable_hint | 10 | 0% | 100% | 9.2 → 8.6 | nothing |
| gemini | hash_hint | 10 | 0% | 94% | 9.6 → 9.2 | nothing |
| gemini | readable_hint | 10 | 0% | 94% | 9.5 → 9.3 | nothing |

Leak rate by type (share of values of that type that reached the provider):

| provider | variant | ACCOUNT_ID | ADDRESS | CODENAME | COMPANY | EMAIL | IP_ADDRESS | MONEY | NAME | PHONE | SECRET |
|---|---|---|---|---|---|---|---|---|---|---|---|
| anthropic | hash_hint | 57% | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 0% |
| anthropic | readable_hint | 57% | 0% | 0% | 1% | 0% | 0% | 100% | 0% | 0% | 0% |
| gemini | hash_hint | 57% | 0% | 0% | 0% | 0% | 0% | 100% | 0% | 0% | 0% |
| gemini | readable_hint | 57% | 0% | 0% | 1% | 0% | 0% | 100% | 0% | 0% | 0% |
