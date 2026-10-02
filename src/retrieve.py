"""Stage 2 - retrieval.

Embeds the question with the same embedder that built the index, queries the
persistent Chroma collection with a rule-based metadata filter, and returns the
hits whose cosine similarity clears `SIMILARITY_THRESHOLD`.

Two deliberate design points from `architecture.md` §5.3:

- A question that clears no hits is the off-topic path. `NO_RETRIEVAL` is
  rendered by the caller, so an unrelated question is refused before any LLM call
  rather than answered from a loosely related chunk.
- Filtering is one-directional. Naming a plan narrows to `{plan, n_a}` so
  statutory documents survive; naming no plan applies no plan filter at all,
  because defaulting would manufacture cross-plan answers (A-6).
"""

from __future__ import annotations

import argparse
import math
import re
from typing import Any, Sequence

import config
from src.embed import EmbedderProtocol, load_embedder
from src.guardrails import screen_input
from src.ingest import open_collection
from src.models import Hit
from src.state import read_state

SCHEME_ALIASES: dict[str, str] = {
    "large cap": "large_cap",
    "largecap": "large_cap",
    "bluechip": "large_cap",
    "blue chip": "large_cap",
    "flexi cap": "flexi_cap",
    "flexicap": "flexi_cap",
    "elss": "elss",
    "tax saver": "elss",
    "80c": "elss",
    "80 c": "elss",
    "mid cap": "midcap",
    "midcap": "midcap",
    "mid cap fund": "midcap",
}

PLAN_ALIASES: dict[str, str] = {
    "direct": "direct",
    "direct plan": "direct",
    "regular": "regular",
    "regular plan": "regular",
}


def infer_scheme(question: str) -> str | None:
    """Resolve a scheme alias from the question, longest alias first."""
    text = (question or "").lower()
    for alias in sorted(SCHEME_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
            return SCHEME_ALIASES[alias]
    return None


def infer_plan(question: str) -> str | None:
    """Resolve a plan alias from the question, longest alias first."""
    text = (question or "").lower()
    for alias in sorted(PLAN_ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
            return PLAN_ALIASES[alias]
    return None


def infer_metadata_filter(question: str) -> dict[str, Any] | None:
    """Build the Chroma `where` clause for a question.

    Only ever narrows. No alias match means no filter, so a failed guess costs
    precision nothing and recall nothing.
    """
    clauses: list[dict[str, Any]] = []

    scheme = infer_scheme(question)
    if scheme:
        clauses.append({"scheme": scheme})

    plan = infer_plan(question)
    if plan:
        clauses.append({"$or": [{"plan": plan}, {"plan": "n_a"}]})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


STOPWORDS = frozenset(
    """a an the is are was were be been being of on in for to at by with and or as that this these
    those it its from what which who whom whose how many much does do did can could will would
    shall should may might i me my we our you your he she they them there here about into over
    under more most less least than then so such not no nor only own same too very just also""".split()
)


def question_terms(question: str) -> set[str]:
    """Content words of the question, ignoring stopwords and 1-2 char noise."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (question or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def lexical_overlap(question: str, text: str) -> float:
    """Fraction of the question's content words that literally appear in a chunk.

    Cosine similarity alone rewards chunks that repeat a word many times, so
    statutory boilerplate full of "load" out-scored the single chunk that states
    the load value. This rewards the chunk that actually names the thing asked
    about, without disturbing the calibrated embedding threshold.
    """
    terms = question_terms(question)
    if not terms:
        return 0.0
    body = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return len(terms & body) / len(terms)


def lexical_scores(question: str, hits: Sequence[Hit]) -> dict[str, float]:
    """IDF-weighted lexical match of each hit against the question's content words.

    Plain term overlap saturates here: `axis large cap fund regular plan` appear in
    every chunk for this scheme, so nearly every candidate scores 1.0 and the
    ranking barely moves. Weighting each term by inverse document frequency across
    the candidate set lets the one distinctive term carry the signal.
    """
    terms = question_terms(question)
    if not terms or not hits:
        return {hit.chunk_id: 0.0 for hit in hits}

    bodies = [set(re.findall(r"[a-z0-9]+", hit.text.lower())) for hit in hits]
    total = len(hits)
    idf = {
        term: math.log(1 + total / (1 + sum(1 for body in bodies if term in body)))
        for term in terms
    }
    denominator = sum(idf.values()) or 1.0
    return {
        hit.chunk_id: sum(idf[term] for term in terms & body) / denominator
        for hit, body in zip(hits, bodies)
    }


def rerank_hits(question: str, hits: list[Hit]) -> list[Hit]:
    """Blend the lexical bonus into the embedding score, preserving the order tie-break."""
    weight = config.LEXICAL_RERANK_WEIGHT
    lexical = lexical_scores(question, hits)
    return sorted(
        hits,
        key=lambda hit: (-(hit.score + weight * lexical[hit.chunk_id]), -hit.score),
    )


def verify_fingerprint(embedder: EmbedderProtocol) -> None:
    """Refuse to query an index built by a different embedder (NFR-4)."""
    recorded = read_state().get("fingerprint")
    if recorded and recorded != embedder.fingerprint:
        raise RuntimeError(
            "index was built by a different embedder: "
            f"state={recorded!r} current={embedder.fingerprint!r}. "
            "Re-run `python -m src.ingest --force`."
        )


def retrieve(
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
    *,
    collection: Any | None = None,
    embedder: EmbedderProtocol | None = None,
) -> list[Hit]:
    """Return the chunks that can support an answer, best first.

    Empty means off-topic or unsupported, and the caller must render
    `NO_RETRIEVAL` rather than answering anyway.
    """
    where = infer_metadata_filter(question)
    enc = embedder or load_embedder()
    verify_fingerprint(enc)
    col = collection if collection is not None else open_collection()
    qvec = enc.encode([question])[0]

    result = col.query(
        query_embeddings=[qvec],
        n_results=top_k or config.TOP_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    floor = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0] or []
    metas = (result.get("metadatas") or [[]])[0] or []
    dists = (result.get("distances") or [[]])[0] or []

    hits: list[Hit] = []
    for index, chunk_id in enumerate(ids):
        distance = float(dists[index]) if index < len(dists) else 1.0
        score = 1.0 - distance
        if score < floor:
            continue
        hits.append(
            Hit(
                chunk_id=str(chunk_id),
                text=str(docs[index]) if index < len(docs) else "",
                metadata=dict(metas[index]) if index < len(metas) else {},
                score=round(score, 4),
            )
        )
    return rerank_hits(question, hits)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieve",
        description="Retrieve chunks for a question and show the guardrail verdict.",
    )
    parser.add_argument("question", help="the user question")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--full", action="store_true", help="print full chunk text")
    args = parser.parse_args(argv)

    question = args.question.strip()
    if not question:
        parser.error("question must not be empty")

    screen = screen_input(question)
    if not screen.is_factual:
        print(f"verdict    : {screen.verdict} (matched {screen.matched or screen.pii_pattern})")
        print("no embedding, no search, no generation")
        return 0

    where = infer_metadata_filter(question)
    hits = retrieve(question, top_k=args.top_k, threshold=args.threshold)

    print(f"verdict    : {screen.verdict}")
    print(f"filter     : {where or 'none (full corpus)'}")
    print(f"hits       : {len(hits)} at threshold {args.threshold or config.SIMILARITY_THRESHOLD}")
    for rank, hit in enumerate(hits, start=1):
        meta = hit.metadata
        print(
            f"  {rank}. {hit.score:.3f}  {hit.chunk_id}  "
            f"{meta.get('scheme')}/{meta.get('plan')}/{meta.get('doc_type')}"
        )
        if args.full:
            print(f"     {hit.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())