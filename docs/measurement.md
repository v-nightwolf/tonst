# Measuring savings

Exact token counts and the local savings log behind `tonst stats`.

[← Back to the README](../README.md)

## Exact token counts

tonst estimates tokens as characters ÷ 4. That's fine for rough prose,
but the live test found real billed input **~1.8×** the estimate on
tool-calling requests: JSON schemas are token-dense, providers add
hidden prompts, and tokenizers differ by model (Anthropic notes Claude
4.7+ produce about 30% more tokens for the same text). For real numbers,
pass a counter:

```python
from tonst import TonstClient, AnthropicTokenCounter, select_tools

counter = AnthropicTokenCounter(model="claude-sonnet-4-6")   # uses ANTHROPIC_API_KEY
client = TonstClient(call_fn=my_api_call, token_counter=counter, savings_log=True)
sel = select_tools(all_tools, request, token_counter=counter.count_tools)  # includes the hidden tool prompt
```

It uses Anthropic's free `count_tokens` endpoint, which Anthropic calls
a close estimate of billed input. The live test script checks it against
real billed tokens. Things to know:

- **Privacy:** the counter only ever receives already-redacted text.
  A remote counter must never see raw PII, and a test enforces this for
  every entry point. With a counter set, `original_tokens` therefore
  measures the redacted, pre-trim prompt.
- **Latency:** each count is a network round trip (~100–300 ms), timed
  separately as `counting_ms`. Use it for calibration or sampling
  rather than on every latency-sensitive call.
- **Fail-soft:** if a count fails, the estimate is used, the report says
  `token_counts_exact=False`, and `tonst stats` labels the numbers
  "estimated" or "counted on N of M calls".

## Savings log

Opt in, and every `query*` call appends one line of metrics to a local
JSONL file (`~/.tonst/savings.jsonl` by default, or `$TONST_SAVINGS_LOG`):
tokens in and tokens sent, redaction counts **by type**, which optional
steps ran, and timings. Nothing is sent anywhere.

```python
client = TonstClient(call_fn=my_api_call, savings_log=True,
                     app_name="support-bot", input_price_per_million=3.0)
```

```
$ tonst stats            # or: python -m tonst stats [--app X] [--since 2026-09-01] [--json]
tonst savings  (2026-09-24T10:02:11+00:00 -> 2026-09-24T16:40:52+00:00)
  calls:            1,204
  tokens in:        3,912,440  (estimated)
  tokens sent:      2,870,115  (estimated)
  tokens saved:     1,042,325  (26.6%)
  est. cost saved:  $3.1270
  PII redacted:     2,311 fields (EMAIL 1,402, NAME 610, PHONE 299)
  avg tonst overhead: 4.2 ms/call
```

*(Illustrative output, not a measured result.)*

What is **never** logged: prompt or response text, PII values, or
placeholder hashes. Placeholders are deterministic hashes of the
original value, so a log full of them could be brute-forced back to
real emails and phone numbers. Only the label (`EMAIL`) is counted.

**Dropped history is reported as lost, not just "saved".** Old turns
that were dropped without a summary weren't sent, so they count toward
`tokens_saved`. But the model never saw that context, so `tonst stats`
shows them on their own line ("of which lost") with a "saved excl.
lost history" figure next to it. Otherwise truncation would look like
optimization. The stats also show median and max tonst overhead, not
just the mean, because one slow local-model call (a fold, a timeout)
can dominate an average of otherwise millisecond-scale calls.
`query_rag()` reports `chunk_filter_skipped` when relevance filtering
was requested but skipped for lack of a confident match. Short
questions often hit this; pass `min_matched_terms=1` to filter anyway.

Honesty about the numbers: token counts use the chars/4 estimate and
are marked as estimates. The dollar figure appears only if you give a
price, and is named `estimated_cost_saved_usd` because it's tokens saved
× your price, not a bill. Provider prompt caching doesn't reduce tokens
(it discounts them), so it never shows up in `tokens_saved`. To record
real cache usage, pass a parsed provider usage report to
`SavingsLog.record(report, usage=...)`.

This is a developer savings log, not an audit record. It has no
tamper-evidence or retention policy, and writes are best-effort (an
unwritable path never breaks your API call).
