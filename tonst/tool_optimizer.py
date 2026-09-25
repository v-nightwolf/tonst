"""
tool_optimizer.py
-----------------
Cuts the tokens spent on tool / function definitions in agentic and
tool-calling workloads. With many tools (or a few MCP servers), the
definitions alone can run to tens of thousands of input tokens on every
request, before the model has read a word of the actual task.

Two strategies, because providers differ:

1. Provider-native deferred loading (Anthropic) -- build_anthropic_deferred_tools()
   Anthropic's tool search tool lets you mark tools `defer_loading: true`:
   their full definitions are still sent, but they're kept OUT of the
   model's context (and out of the cached prompt prefix) until the model
   searches for and loads one. This is the best option on Anthropic --
   nothing is ever filtered away, so a tool can't be "missing" -- and
   this module just builds the request shape correctly:
     - adds the tool search tool (bm25 or regex variant), never deferred;
     - defers every custom tool except the ones you pin as always-loaded
       (Anthropic recommends keeping your 3-5 most-used tools loaded);
     - defers whole MCP servers via mcp_toolset.default_config;
     - refuses a deferred tool that also carries cache_control, which
       the API rejects with a 400.
   Verified against platform.claude.com/docs/en/agents-and-tools/tool-use/
   tool-search-tool (checked 2026-09-24): type strings
   tool_search_tool_regex_20251119 / tool_search_tool_bm25_20251119, no
   beta header, at least one non-deferred tool required, supported on
   current Claude models (not Opus 4.1 and earlier).

2. Local relevance filtering (any provider) -- select_tools() / ToolSession
   For providers without native deferred loading, send only the tools
   relevant to the request, chosen locally with BM25 over each tool's
   name, description and parameter descriptions (relevance.py). The
   ORIGINAL tool dicts are returned untouched and in their ORIGINAL
   order -- never rewritten, reformatted or summarized -- so the same
   selection always serializes to the same bytes.

   Honest risks, and how they're handled:
     - Filtering can drop a tool the model actually needed. Pin anything
       critical with always_include. If the best-matching tool shares
       fewer than min_matched_terms (default 2) distinct words with the
       request, the match is treated as a coincidence and NOTHING is
       filtered (fell_back=True) -- "can't tell" never becomes "drop
       everything". In benchmarks/benchmark_free_features.py this rule took recall
       on deliberately paraphrased requests from 50% to 100%, at the
       cost of sending the full tool list for those requests.
     - Lexical matching can't see synonyms ("ping the team" vs. a
       "send a Slack message" tool). Where the provider supports it,
       native deferred loading (strategy 1) doesn't have this problem.
     - Filtering fights prompt caching. Tool definitions sit at the very
       front of the prompt prefix on Anthropic (and early on others), so
       a DIFFERENT tool list every turn invalidates the cache for
       everything after it -- which can cost more than the filtering
       saved. ToolSession exists for this: it selects once per
       conversation and only ever GROWS the set (never shrinks or
       reorders it), so the tool block stays byte-identical across turns
       and changes at most a handful of times per session.

Deliberately NOT done: rewriting or summarizing tool descriptions with the
local model. A subtly wrong description produces wrong tool calls, and a
1-3B local model isn't reliable enough for that to be worth the risk.
"""

from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .relevance import bm25_scores, matched_terms
from .trim import estimate_tokens

TOOL_SEARCH_BM25 = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
TOOL_SEARCH_REGEX = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
_SEARCH_TOOLS = {"bm25": TOOL_SEARCH_BM25, "regex": TOOL_SEARCH_REGEX}

# Tool "type" values that mean "a normal tool you defined yourself", as
# opposed to a provider/server tool (web search, code execution, tool
# search, an MCP toolset...) that tonst can't score and must never drop.
_CUSTOM_TYPES = {None, "custom", "function"}


# ---------------------------------------------------------------------
# Reading tool definitions in either common format
# ---------------------------------------------------------------------
#   Anthropic:            {"name", "description", "input_schema"}
#   OpenAI Chat:          {"type": "function", "function": {"name", "description", "parameters"}}
#   OpenAI Responses:     {"type": "function", "name", "description", "parameters"}

def _fn(tool: dict) -> dict:
    return tool.get("function") if isinstance(tool.get("function"), dict) else tool


def is_custom_tool(tool: dict) -> bool:
    return tool.get("type") in _CUSTOM_TYPES


def tool_name(tool: dict) -> str:
    if tool.get("type") == "mcp_toolset":
        return f"mcp:{tool.get('mcp_server_name', '')}"
    return str(_fn(tool).get("name", ""))


def _schema(tool: dict) -> dict:
    f = _fn(tool)
    return f.get("input_schema") or f.get("parameters") or {}


def _schema_text(schema, depth: int = 0) -> list[str]:
    # Property names + descriptions, recursively (bounded), so a question
    # mentioning "repository" matches a tool whose only mention of it is
    # a parameter description.
    out: list[str] = []
    if not isinstance(schema, dict) or depth > 4:
        return out
    for name, prop in (schema.get("properties") or {}).items():
        out.append(str(name))
        if isinstance(prop, dict):
            if prop.get("description"):
                out.append(str(prop["description"]))
            out.extend(_schema_text(prop, depth + 1))
            if isinstance(prop.get("items"), dict):
                out.extend(_schema_text(prop["items"], depth + 1))
    return out


def tool_search_text(tool: dict) -> str:
    """The text a tool is matched against: name, description, parameters."""
    f = _fn(tool)
    parts = [tool_name(tool), str(f.get("description", ""))]
    parts.extend(_schema_text(_schema(tool)))
    return " ".join(p for p in parts if p)


def estimate_tool_tokens(tools: Iterable[dict]) -> int:
    """chars/4 estimate of the tool definitions as they'd be serialized."""
    tools = list(tools)
    if not tools:
        return 0
    return estimate_tokens(json.dumps(tools, separators=(",", ":"), sort_keys=True))


# ---------------------------------------------------------------------
# Strategy 2: local relevance filtering (any provider)
# ---------------------------------------------------------------------

@dataclass
class ToolSelection:
    tools: list                      # the tool dicts to send, original objects, original order
    selected_names: list
    dropped_names: list
    tokens_before: int
    tokens_after: int
    fell_back: bool = False          # True if nothing was filtered because relevance couldn't be judged
    reason: str = ""
    added_names: list = field(default_factory=list)  # ToolSession only: tools newly added this turn
    changed: bool = True             # ToolSession only: False when the tool list is byte-identical to last turn
    tokens_exact: bool = False       # True when tokens_before/after came from a real counter, not chars/4

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


def _measure(tools: list, token_counter) -> tuple:
    """(tokens, exact). token_counter(tools) -> int or None, e.g.
    token_count.AnthropicTokenCounter(...).count_tools -- which also
    includes the hidden tool-use system prompt the provider adds. Falls
    back to the chars/4 estimate if there's no counter or it fails."""
    if token_counter is not None:
        try:
            n = token_counter(tools)
        except Exception:  # noqa: BLE001
            n = None
        if n is not None:
            return int(n), True
    return estimate_tool_tokens(tools), False


def _validate_names(tools: list, names: Iterable[str], arg: str) -> set:
    names = set(names or ())
    known = {tool_name(t) for t in tools}
    unknown = names - known
    if unknown:
        raise ValueError(f"{arg} names not found among the tools given: {sorted(unknown)}")
    return names


def _rank(tools: list, query: str) -> list[tuple[float, int]]:
    """(score, index) for every scoreable tool, best first; stable on ties."""
    idx = [i for i, t in enumerate(tools) if is_custom_tool(t)]
    scores = bm25_scores(query, [tool_search_text(tools[i]) for i in idx])
    return sorted(zip(scores, idx), key=lambda p: (-p[0], p[1]))


def select_tools(
    tools: list,
    query: str,
    top_k: int = 8,
    always_include: Iterable[str] = (),
    min_score: float = 0.0,
    min_matched_terms: int = 2,
    token_counter=None,
) -> ToolSelection:
    """
    Keeps the top_k custom tools most relevant to `query`, plus every
    tool named in always_include, plus every non-custom (provider/server)
    tool, which can't be scored and is never dropped. Returns the ORIGINAL
    dicts in their ORIGINAL order.

    `query` is whatever best describes what the model is about to do:
    the user's latest message, or the task description for an agent run.

    Nothing is filtered when there are no more custom tools than top_k,
    or when the best-scoring tool shares fewer than min_matched_terms
    distinct words with the request (no match, or only a coincidental
    one-word match -- see the module docstring). Set min_matched_terms=1
    to filter on any overlap at all (more savings, more risk).

    token_counter (optional): measure tokens_before/tokens_after with a
    real tokenizer instead of chars/4 -- e.g.
    AnthropicTokenCounter(model=...).count_tools. A live test found the
    real cost of tool definitions ~1.4-2x the chars/4 estimate (JSON is
    token-dense, and providers add a hidden tool-use prompt). Only
    changes the reported numbers, never the selection. Tool definitions
    contain no user data, so sending them to a counting endpoint is safe.
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    tools = list(tools)
    pinned = _validate_names(tools, always_include, "always_include")
    before, exact = _measure(tools, token_counter)
    all_names = [tool_name(t) for t in tools]

    def _all(reason: str, fell_back: bool = False) -> ToolSelection:
        return ToolSelection(tools, all_names, [], before, before, fell_back=fell_back, reason=reason,
                             tokens_exact=exact)

    candidates = [t for t in tools if is_custom_tool(t) and tool_name(t) not in pinned]
    if len(candidates) <= top_k:
        return _all("tool count already within top_k; nothing filtered")

    ranked = [(s, i) for s, i in _rank(tools, query) if tool_name(tools[i]) not in pinned]
    if (
        not ranked
        or ranked[0][0] <= 0.0
        or matched_terms(query, tool_search_text(tools[ranked[0][1]])) < min_matched_terms
    ):
        return _all("no confident tool match for this request; kept all tools rather than guess", fell_back=True)

    keep_idx = {i for s, i in ranked[:top_k] if s > min_score}
    keep_idx |= {i for i, t in enumerate(tools) if not is_custom_tool(t) or tool_name(t) in pinned}
    kept = [t for i, t in enumerate(tools) if i in keep_idx]
    after, after_exact = _measure(kept, token_counter)
    if exact != after_exact:  # never mix a real count with an estimate
        before, after, exact = estimate_tool_tokens(tools), estimate_tool_tokens(kept), False
    return ToolSelection(
        tools=kept,
        selected_names=[tool_name(t) for t in kept],
        dropped_names=[tool_name(t) for i, t in enumerate(tools) if i not in keep_idx],
        tokens_before=before,
        tokens_after=after,
        reason=f"kept top {top_k} by relevance",
        tokens_exact=exact,
    )


class ToolSession:
    """
    Cache-friendly tool filtering for a multi-turn conversation or agent
    run. The first select() picks the relevant tools; every later select()
    can only ADD tools that the new turn clearly needs (and that aren't
    already in the set) -- it never removes or reorders any. Since the
    returned list always keeps the tools' original relative order, an
    unchanged set serializes byte-identically, so the provider's prompt
    cache keeps hitting on turns where nothing was added (changed=False).

    State is two small lists (see to_dict()/from_dict()), so a web
    server can persist it per conversation.
    """

    def __init__(
        self,
        tools: list,
        top_k: int = 8,
        always_include: Iterable[str] = (),
        add_per_turn: int = 3,
        min_matched_terms: int = 2,
        token_counter=None,
    ):
        self.tools = list(tools)
        self.top_k = top_k
        self.always_include = sorted(_validate_names(self.tools, always_include, "always_include"))
        self.add_per_turn = add_per_turn
        self.min_matched_terms = min_matched_terms
        self.token_counter = token_counter
        self._full_measure: Optional[tuple] = None  # all tools never change: measure once
        self.active: Optional[set] = None  # names currently sent; None until the first select()

    def _ordered(self, names: set) -> list:
        return [t for t in self.tools if tool_name(t) in names]

    def select(self, query: str) -> ToolSelection:
        if self._full_measure is None:
            self._full_measure = _measure(self.tools, self.token_counter)
        before, exact = self._full_measure
        if self.active is None:
            first = select_tools(
                self.tools, query, top_k=self.top_k, always_include=self.always_include,
                min_matched_terms=self.min_matched_terms, token_counter=self.token_counter,
            )
            self.active = set(first.selected_names)
            first.added_names = list(first.selected_names)
            return first

        # Only confident matches grow the set: a one-word coincidence
        # would add a useless tool AND break the cache for no reason.
        ranked = [
            (s, i) for s, i in _rank(self.tools, query)
            if s > 0.0 and matched_terms(query, tool_search_text(self.tools[i])) >= self.min_matched_terms
        ]
        additions = []
        for s, i in ranked[: self.top_k]:
            name = tool_name(self.tools[i])
            if name not in self.active:
                additions.append(name)
            if len(additions) >= self.add_per_turn:
                break
        self.active |= set(additions)
        kept = self._ordered(self.active)
        after, after_exact = _measure(kept, self.token_counter)
        if exact != after_exact:
            before, after, exact = estimate_tool_tokens(self.tools), estimate_tool_tokens(kept), False
        return ToolSelection(
            tools=kept,
            selected_names=[tool_name(t) for t in kept],
            dropped_names=[tool_name(t) for t in self.tools if tool_name(t) not in self.active],
            tokens_before=before,
            tokens_after=after,
            tokens_exact=exact,
            reason="grew tool set" if additions else "tool set unchanged (cache-stable)",
            added_names=additions,
            changed=bool(additions),
        )

    def to_dict(self) -> dict:
        return {"active": sorted(self.active) if self.active is not None else None}

    def load_state(self, state: dict) -> "ToolSession":
        active = state.get("active")
        self.active = set(active) if active is not None else None
        return self


# ---------------------------------------------------------------------
# Strategy 1: Anthropic native deferred loading
# ---------------------------------------------------------------------

# Name words that mark a tool whose own job is searching/looking up DATA.
_SEARCH_LIKE_NAME_TERMS = {"search", "lookup"}

# Add this to your system prompt when using deferred tools. In live testing
# (2026-09-24, Sonnet 4.6, 36 tools all deferred), Claude several times sent
# the tool search tool the TOPIC it wanted to find ("billing outage",
# "on-call runbook") instead of the CAPABILITY it needed ("search slack
# messages"), found no tool, and then told the user the data didn't exist.
DEFERRED_TOOLS_SYSTEM_HINT = (
    "Some tools are not loaded yet. The tool search tool finds TOOLS, not data: search for "
    "what the tool does (for example 'search slack messages' or 'query database'), never "
    "for the content you are looking for. If a tool search returns nothing, try a broader "
    "description of the capability before concluding that no suitable tool exists."
)


def is_search_like_tool(tool: dict) -> bool:
    """True for tools whose NAME says they search/look up data (e.g. slack_search_messages, logs_search)."""
    from .relevance import tokenize
    return is_custom_tool(tool) and bool(_SEARCH_LIKE_NAME_TERMS & set(tokenize(tool_name(tool))))


def build_anthropic_deferred_tools(
    tools: list,
    always_loaded: Iterable[str] = (),
    search: str = "bm25",
    keep_search_tools_loaded: bool = True,
) -> list:
    """
    Returns a NEW tools list for the Anthropic Messages API with the tool
    search tool added and every custom tool deferred except those named
    in always_loaded (and, by default, search-type tools -- see below).
    Input dicts are not modified.

    Also add DEFERRED_TOOLS_SYSTEM_HINT to your system prompt.

    - keep_search_tools_loaded (default True): tools whose name contains
      "search" or "lookup" (slack_search_messages, logs_search, ...) stay
      loaded. Live testing found deferring them is the main way deferred
      loading fails: asked to "search Slack for the billing outage",
      Claude searched the TOOL catalog for "billing outage", found
      nothing, and told the user the data didn't exist. 3 of its 5 extra
      misses (vs. sending all tools) in a 30-task run were this. Set False
      to defer them too.
    - search: "bm25" (natural-language queries) or "regex".
    - Anthropic recommends keeping your 3-5 most-used tools loaded as
      well -- pass them in always_loaded.
    - Server/provider tools (web search, code execution, ...) are left
      exactly as given -- never deferred.
    - mcp_toolset entries get default_config.defer_loading = True unless
      you've already set it; pin individual MCP tools yourself via that
      toolset's `configs` ({"tool_name": {"defer_loading": false}}).
    - An existing tool search tool in the input is kept and not
      duplicated.

    Raises ValueError for an unknown always_loaded name, an unknown
    search variant, or a tool that would end up both deferred and
    carrying cache_control (the API rejects that with a 400).
    """
    if search not in _SEARCH_TOOLS:
        raise ValueError(f"search must be one of {sorted(_SEARCH_TOOLS)}, got {search!r}")
    pinned = _validate_names(tools, always_loaded, "always_loaded")
    if keep_search_tools_loaded:
        pinned |= {tool_name(t) for t in tools if is_search_like_tool(t)}

    out: list = []
    has_search = any(str(t.get("type", "")).startswith("tool_search_tool_") for t in tools)
    if not has_search:
        out.append(dict(_SEARCH_TOOLS[search]))

    for t in tools:
        t = copy.deepcopy(t)
        if t.get("type") == "mcp_toolset":
            cfg = t.setdefault("default_config", {})
            cfg.setdefault("defer_loading", True)
        elif is_custom_tool(t):
            if tool_name(t) in pinned:
                t.pop("defer_loading", None)
            else:
                if "cache_control" in t:
                    raise ValueError(
                        f"tool {tool_name(t)!r} carries cache_control, so it can't be deferred "
                        "(Anthropic returns a 400). Pin it with always_loaded, or move the "
                        "breakpoint to a non-deferred tool."
                    )
                t["defer_loading"] = True
        out.append(t)
    return out


def loaded_tools(tools: list) -> list:
    """The tools that are actually in the model's context (not deferred)."""
    return [t for t in tools if not t.get("defer_loading")]
