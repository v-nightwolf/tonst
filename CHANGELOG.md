# Changelog

## 0.2.0 — 2026-09-28

Redaction overhaul, measured with a new answer-quality benchmark
(`experiments/privacy_quality/`). All savings features are unchanged.

### Added
- **Answer-quality benchmark**: 120 realistic prompts sent as-is and redacted to
  Claude and Gemini, graded blind; reports leaks, restoration failures,
  must-have values, score change, tokens and cost.
- **Keyed placeholders**: hashes are HMAC-keyed with a local key
  (`~/.tonst/placeholder.key` or `TONST_PLACEHOLDER_KEY`), so a provider can't
  confirm a guessed value by hashing it. Still stable across calls, so prompt
  caching keeps working. `placeholder_style="readable"` gives `[[NAME_1]]`.
- **One placeholder per person**: `[[NAME_1]]`, `[[NAME_1.first]]`,
  `[[NAME_1.last]]`; titles stay visible.
- **Split emails**: `[[EMAIL_1]]@[[DOMAIN_1]]` (`email_style="whole"` for the old shape).
- **Note to the model** (`placeholder_hint=True`): explains the placeholders,
  who owns which contact details, and that secrets are exposed credentials.
  Sent only when something was hidden, with only the lines that apply:
  ~45 tokens, up to ~115 with names, contacts and secrets. In chats the fixed
  part goes in the system message (cacheable) and the contact line rides on
  the latest user message.
- **New detectors**: API keys and other secrets (always on), postal addresses,
  more phone formats; opt-in `extra_redaction=["ACCOUNT_ID", "MONEY"]`.
- **Secrets withheld from answers** by default (`[REDACTED]`,
  `report.secrets_withheld`); `restore_secrets=True` and `secret_notice=True`
  to change that.
- **Tolerant restore** of reformatted placeholders, neutral text for invented
  ones, and `StreamRestorer` for streamed answers.
- Recall safety nets for names after roles/greetings/field labels; companies
  and codenames remembered per client; public services (e.g. `api.anthropic.com`)
  left visible.

### Fixed
- Placeholder regex missed labels with underscores (`CREDIT_CARD`, `SSN_LIKE`,
  `IP_ADDRESS`), so those values weren't always restored.
- GLiNER only read about the first 300 words; it now scans in overlapping windows.
- GLiNER load failures were silent; they now log a loud warning, and the
  `[gliner]` extra includes `sentencepiece` and `protobuf`.
- An email followed by a full stop swallowed the full stop.
- Trimming left blank lines behind after removing duplicates.

### Changed
- Placeholders look different from 0.1.0 (keyed hashes, split emails).
  Nothing is stored between runs, so no migration is needed.
- Requires Python 3.10+.
