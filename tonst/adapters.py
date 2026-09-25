"""
adapters.py
-----------
Small converters between tonst's plain message list and each provider's
request/response shapes, for TonstClient(messages_fn=...).

tonst hands messages_fn the final, redacted conversation as
[{"role": "system" | "user" | "assistant", "content": str}, ...].
These helpers turn that into what each API expects, and turn the API's
usage block back into the {"prompt_tokens", "cached_tokens"} dict tonst
reads (so the report and cache-aware compaction see real numbers).

    import anthropic
    from tonst import TonstClient
    from tonst.adapters import to_anthropic, usage_from_anthropic

    api = anthropic.Anthropic()

    def call_claude(messages):
        resp = api.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
                                   **to_anthropic(messages))
        return resp.content[0].text, usage_from_anthropic(resp)

    client = TonstClient(messages_fn=call_claude)

Every helper accepts either a plain dict (raw JSON) or an SDK response
object, and none of them imports a provider SDK.
"""

from __future__ import annotations
from typing import Optional


def _get(obj, *path, default=None):
    """obj.a.b or obj["a"]["b"], whichever exists."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _merge_turns(messages: list, assistant_role: str) -> tuple:
    """(system text, [(role, [texts])]) with same-role neighbours merged and a user turn first."""
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content"))
    turns = []
    for m in messages:
        role = m.get("role")
        if role == "system" or not m.get("content"):
            continue
        role = assistant_role if role == "assistant" else "user"
        if turns and turns[-1][0] == role:
            turns[-1][1].append(m["content"])
        else:
            turns.append((role, [m["content"]]))
    if turns and turns[0][0] != "user":
        turns.insert(0, ("user", ["(Earlier conversation omitted.)"]))
    return system, turns


def to_anthropic(messages: list, cache: bool = True) -> dict:
    """
    Keyword arguments for Anthropic's Messages API: {"system", "messages"}.
    Roles must alternate there, so same-role neighbours are merged (tonst's
    summary message is a user turn and can sit next to one). cache=True puts
    a cache_control breakpoint on the system prompt and a moving one on the
    last block, which is how a multi-turn chat gets cached turn after turn.
    Prompts under the model's minimum cacheable length are simply not cached.
    """
    system, turns = _merge_turns(messages, "assistant")
    out_msgs = [{"role": r, "content": [{"type": "text", "text": t} for t in texts]} for r, texts in turns]
    if cache and out_msgs:
        last = out_msgs[-1]["content"][-1]
        out_msgs[-1]["content"][-1] = {**last, "cache_control": {"type": "ephemeral"}}
    body = {"messages": out_msgs}
    if system:
        block = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        body["system"] = [block]
    return body


def to_openai(messages: list) -> list:
    """Chat Completions `messages`. OpenAI caches repeated prefixes automatically; no markers needed."""
    return [{"role": m["role"], "content": m["content"]} for m in messages if m.get("content")]


def to_gemini(messages: list) -> dict:
    """
    generateContent REST body fields: {"systemInstruction", "contents"}
    (role "model" for the assistant). Gemini 2.5+ caches repeated prefixes
    implicitly; it's best-effort and needs a few thousand tokens.
    """
    system, turns = _merge_turns(messages, "model")
    body = {"contents": [{"role": r, "parts": [{"text": t} for t in texts]} for r, texts in turns]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def usage_from_anthropic(resp) -> Optional[dict]:
    u = _get(resp, "usage")
    if u is None:
        return None
    read = int(_get(u, "cache_read_input_tokens", default=0))
    total = int(_get(u, "input_tokens", default=0)) + int(_get(u, "cache_creation_input_tokens", default=0)) + read
    return {"prompt_tokens": total, "cached_tokens": read}


def usage_from_openai(resp) -> Optional[dict]:
    u = _get(resp, "usage")
    if u is None:
        return None
    if _get(u, "prompt_tokens") is not None:   # Chat Completions
        return {"prompt_tokens": int(_get(u, "prompt_tokens")),
                "cached_tokens": int(_get(u, "prompt_tokens_details", "cached_tokens", default=0))}
    return {"prompt_tokens": int(_get(u, "input_tokens", default=0)),   # Responses API
            "cached_tokens": int(_get(u, "input_tokens_details", "cached_tokens", default=0))}


def usage_from_gemini(resp) -> Optional[dict]:
    u = _get(resp, "usageMetadata") or _get(resp, "usage_metadata")
    if u is None:
        return None
    prompt = _get(u, "promptTokenCount") if _get(u, "promptTokenCount") is not None else _get(u, "prompt_token_count", default=0)
    cached = _get(u, "cachedContentTokenCount") if _get(u, "cachedContentTokenCount") is not None \
        else _get(u, "cached_content_token_count", default=0)
    return {"prompt_tokens": int(prompt), "cached_tokens": int(cached or 0)}
