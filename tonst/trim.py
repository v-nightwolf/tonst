"""
trim.py
-------
Cheap, local, mechanical token-reduction. No model call needed for these --
they're safe to always apply because they don't change meaning.

estimate_tokens() uses a rough chars/4 heuristic (close enough for cost
estimates without pulling in a tokenizer library). Swap in tiktoken or the
provider's own tokenizer for production-grade accuracy.
"""

from __future__ import annotations
import re


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def strip_redundant_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedupe_repeated_lines(text: str) -> str:
    """Drop exact-duplicate lines (common in accumulated chat histories/logs)."""
    seen = set()
    out_lines = []
    for line in text.split("\n"):
        key = line.strip()
        if key and key in seen:
            continue
        seen.add(key)
        out_lines.append(line)
    return "\n".join(out_lines)


def truncate_history(messages: list[dict], keep_last_n: int = 6, keep_system: bool = True) -> list[dict]:
    """
    Keep the system prompt (if any) plus only the most recent N turns.
    Old turns are the single biggest silent token cost in chat apps.
    """
    if not messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"] if keep_system else []
    other_msgs = [m for m in messages if m.get("role") != "system"]
    trimmed = other_msgs[-keep_last_n:]
    return system_msgs + trimmed


def mechanical_trim(text: str) -> str:
    text = strip_redundant_whitespace(text)
    text = dedupe_repeated_lines(text)
    return text
