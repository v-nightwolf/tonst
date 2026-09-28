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
