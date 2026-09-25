"""
relevance.py
------------
Small, dependency-free lexical relevance scoring shared by
tool_optimizer.py (which tool definitions matter for this turn?) and
rag.py (which retrieved chunks matter for this question?).

Deliberately lexical (BM25 over word tokens), not embedding-based:
  - No new dependency, no model download, no GPU, microseconds per call.
    Everything else in tonst's "always-on" path has the same property,
    and relevance filtering should too.
  - Deterministic: the same input always produces the same selection,
    which is what keeps a filtered tool list or chunk list byte-stable
    across calls (and therefore cacheable by the provider).
  - Honest limitation: BM25 only sees shared words. A question phrased
    with synonyms that never appear in a tool description or chunk
    ("forecast" vs. "weather") scores zero. Callers in this package
    therefore treat "nothing scored above zero" as "can't tell, keep
    everything" rather than "nothing is relevant, drop everything".

Also includes a cheap near-duplicate check (word-shingle Jaccard
similarity), used by rag.py to drop retrieved chunks that are the same
passage retrieved twice from overlapping splits or mirrored documents.
"""

from __future__ import annotations
import math
import re
from collections import Counter

# Short, conservative stopword list -- only words that carry no topical
# signal in either a tool description or a retrieved passage. Kept small
# on purpose: an over-aggressive list starts dropping words that DO
# matter for tool matching ("get", "list", "send" are verbs tools are
# named after, so they are NOT in here).
STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from as
    is are was were be been being am do does did done have has had
    this that these those it its it's i me my we our you your he she
    they them their what which who whom whose when where why how
    can could should would will shall may might must please
    about into over under than so such not no yes just also very
    there here some any all each every more most other
    """.split()
)

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    # Minimal plural folding only ("issues" -> "issue", "files" -> "file").
    # Anything smarter (a real stemmer) risks conflating unrelated words
    # and needs a dependency; this covers the most common miss.
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """
    Lowercased word tokens with camelCase/snake_case split apart, so a
    tool named `searchIssues` or `search_issues` matches a question
    containing "search" and "issues". Stopwords removed, plurals folded.
    """
    if not text:
        return []
    text = _CAMEL_RE.sub(" ", text).lower()
    return [_stem(w) for w in _WORD_RE.findall(text) if w not in STOPWORDS and len(w) > 1]


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """
    Okapi BM25 score of each document against `query`. Returns one float
    per document, in the same order. All zeros means no query term
    appears in any document -- see the module docstring for why callers
    treat that as "can't tell", not "irrelevant".
    """
    q_terms = set(tokenize(query))
    docs = [tokenize(d) for d in documents]
    n = len(docs)
    if n == 0 or not q_terms:
        return [0.0] * n

    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter()
    for d in docs:
        df.update(set(d) & q_terms)

    scores = []
    for d in docs:
        tf = Counter(t for t in d if t in q_terms)
        dl = len(d)
        s = 0.0
        for term, f in tf.items():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def matched_terms(query: str, document: str) -> int:
    """
    Number of DISTINCT query terms that appear in `document`. Used as a
    confidence check on top of BM25: a best match that shares only one
    word with the request is usually a coincidence ("book a SLOT" vs. a
    "find free time slots" tool), not a real match -- see
    benchmark_free_features.py, where every wrong pick had exactly one
    shared word and every right pick but one had two or more.
    """
    return len(set(tokenize(query)) & set(tokenize(document)))


def normalize_for_compare(text: str) -> str:
    """Whitespace/case-insensitive key for exact-duplicate detection."""
    return " ".join(text.lower().split())


def shingles(text: str, size: int = 5) -> set:
    words = normalize_for_compare(text).split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
