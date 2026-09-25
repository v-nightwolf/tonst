"""
rag.py
------
Trims the retrieved context in a RAG pipeline before it goes into the
prompt. Retrieved chunks are often the single biggest token cost in a
retrieval-heavy app, and a lot of it is waste: the same passage
retrieved twice from overlapping splits, near-identical boilerplate from
mirrored docs, and low-relevance filler that only made the top-k because
k was generous.

optimize_chunks() works on the chunks your retriever already returned --
tonst does not do retrieval itself. It only ever SELECTS whole chunks;
it never rewrites, paraphrases or compresses a chunk's text. That keeps
it safe to run before redaction (nothing is transformed, so no PII can
be reshaped into a form redaction misses -- see pipeline-flow notes in
ROADMAP.md) and makes it deterministic.

Steps, all local, dependency-free, sub-millisecond for typical inputs:
  1. Exact duplicates removed (case/whitespace-insensitive).
  2. Near-duplicates removed: word 5-gram Jaccard similarity above
     dedupe_threshold (default 0.8) against an already-kept chunk. The
     earlier chunk in retrieval order wins, since retrievers put their
     best match first.
  3. Relevance filter (optional): BM25 against the question, keeping
     top_k and/or dropping chunks scoring below min_relative_score x the
     best chunk's score. If the best chunk shares fewer than
     min_matched_terms (default 2) distinct words with the question,
     nothing is filtered by relevance (fell_back=True): lexical scoring
     can't see synonyms, and a one-word coincidence or "can't tell"
     must not turn into "drop all the context".
  4. Token budget (optional): max_tokens keeps the highest-scoring
     chunks that fit.
Output keeps the retriever's original order by default.

Deliberately NOT done here: running each chunk through the local
compression model. The Sept 2026 ablation measured local compression at
only ~+1.9 percentage points of token savings for ~2s of latency per
call; per chunk, that latency multiplies. Query-aware compression
(e.g. Microsoft's LongLLMLingua) is a better fit for that job if you
need it.

Retrieved chunks change with every question, so they belong in the
VARIABLE part of a prompt (after the cacheable system prompt / stable
reference material), never in stable_blocks -- TonstClient.query_rag()
does this for you.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

from .relevance import bm25_scores, jaccard, matched_terms, normalize_for_compare, shingles
from .trim import estimate_tokens

Chunk = Union[str, dict]


def chunk_text(chunk: Chunk) -> str:
    """Accepts a plain string or a dict with a "text" (or "content") key."""
    if isinstance(chunk, str):
        return chunk
    return str(chunk.get("text", chunk.get("content", "")))


@dataclass
class ChunkSelection:
    chunks: list                      # kept chunks (original objects), in output order
    dropped: list = field(default_factory=list)  # (index, reason) for each dropped chunk
    tokens_before: int = 0
    tokens_after: int = 0
    fell_back: bool = False           # relevance filtering skipped because no chunk matched the question

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    @property
    def texts(self) -> list:
        return [chunk_text(c) for c in self.chunks]


def optimize_chunks(
    chunks: list,
    question: str,
    top_k: Optional[int] = None,
    min_relative_score: float = 0.0,
    max_tokens: Optional[int] = None,
    dedupe_threshold: float = 0.8,
    order: str = "original",
    min_matched_terms: int = 2,
) -> ChunkSelection:
    """
    See module docstring. Defaults are conservative: with no top_k,
    min_relative_score or max_tokens given, only exact and near-duplicate
    chunks are removed.

    order: "original" (retriever order, default) or "score" (most
    relevant first).
    """
    if order not in ("original", "score"):
        raise ValueError("order must be 'original' or 'score'")
    if not 0.0 <= dedupe_threshold <= 1.0:
        raise ValueError("dedupe_threshold must be between 0 and 1")

    texts = [chunk_text(c) for c in chunks]
    sel = ChunkSelection(chunks=[], tokens_before=sum(estimate_tokens(t) for t in texts if t.strip()))

    # 1-2. Dedupe, earliest (best-ranked by the retriever) wins.
    kept: list[int] = []
    seen_exact: set = set()
    kept_shingles: list = []
    for i, t in enumerate(texts):
        key = normalize_for_compare(t)
        if not key:
            sel.dropped.append((i, "empty"))
            continue
        if key in seen_exact:
            sel.dropped.append((i, "duplicate"))
            continue
        sh = shingles(t)
        if dedupe_threshold < 1.0 and any(jaccard(sh, other) >= dedupe_threshold for other in kept_shingles):
            sel.dropped.append((i, "near_duplicate"))
            continue
        seen_exact.add(key)
        kept_shingles.append(sh)
        kept.append(i)

    # 3. Relevance.
    scores = dict(zip(kept, bm25_scores(question, [texts[i] for i in kept])))
    best = max(scores.values(), default=0.0)
    filtering = top_k is not None or min_relative_score > 0 or max_tokens is not None
    best_i = min((i for i in kept if scores[i] == best), default=None)
    confident = best > 0.0 and best_i is not None and matched_terms(question, texts[best_i]) >= min_matched_terms
    if filtering and not confident:
        sel.fell_back = True
    elif filtering:
        ranked = sorted(kept, key=lambda i: (-scores[i], i))
        if min_relative_score > 0:
            floor = best * min_relative_score
            for i in [i for i in ranked if scores[i] < floor]:
                sel.dropped.append((i, "low_relevance"))
            ranked = [i for i in ranked if scores[i] >= floor]
        if top_k is not None and len(ranked) > top_k:
            for i in ranked[top_k:]:
                sel.dropped.append((i, "below_top_k"))
            ranked = ranked[:top_k]
        # 4. Token budget, greedily by score.
        if max_tokens is not None:
            used, fitted = 0, []
            for i in ranked:
                cost = estimate_tokens(texts[i])
                if used + cost <= max_tokens:
                    fitted.append(i)
                    used += cost
                else:
                    sel.dropped.append((i, "over_budget"))
            ranked = fitted
        kept = ranked

    if order == "original" or sel.fell_back or not filtering:
        kept = sorted(kept) if order == "original" else sorted(kept, key=lambda i: (-scores.get(i, 0.0), i))
    sel.dropped.sort()
    sel.chunks = [chunks[i] for i in kept]
    sel.tokens_after = sum(estimate_tokens(texts[i]) for i in kept)
    return sel


def format_context(chunk_texts: list, question: str) -> str:
    """
    Default layout for the variable part of a RAG prompt: numbered
    context blocks, then the question. Deterministic for the same input.
    """
    blocks = [f"[Context {n}]\n{t.strip()}" for n, t in enumerate(chunk_texts, 1)]
    return "\n\n".join(blocks + [f"Question: {question.strip()}"])
