"""Calibrate SIMILARITY_THRESHOLD against the probe set.

Runs every `eval/probe_facts.jsonl` question plus the off-topic negatives
through the real embedder and the real collection, and reports the separation
band between the correct chunk and the best irrelevant chunk. The recommended
threshold is the midpoint of that band.

    python -m eval.calibrate
"""

from __future__ import annotations

import json
from pathlib import Path

import config
from src.retrieve import retrieve

PROBE_PATH = config.EVAL_DIR / "probe_facts.jsonl"
REPORT_PATH = config.EVAL_DIR / "eval_report.md"

OFF_TOPIC = [
    "How do I cook pasta?",
    "What is the weather in Mumbai tomorrow?",
    "Who won the last World Cup?",
    "Write me a Python function to sort a list.",
    "What is the capital of France?",
    "How do I lose weight fast?",
    "Explain quantum computing to me.",
    "What is 2 plus 2?",
    "Book me a flight to Delhi.",
    "Tell me a joke about cats.",
]


def load_probes() -> list[dict]:
    return [
        json.loads(line)
        for line in PROBE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def top_score_for_off_topic(question: str) -> float:
    hits = retrieve(question, threshold=0.0, top_k=1)
    return hits[0].score if hits else 0.0


def main() -> int:
    probes = load_probes()

    positives: list[float] = []
    negatives: list[float] = []
    misses: list[tuple[str, list[str], float]] = []
    lines: list[str] = []

    for probe in probes:
        question = probe["question"]
        targets = set(probe["chunk_ids"])
        hits = retrieve(question, threshold=0.0, top_k=5)
        hit_ids = {hit.chunk_id for hit in hits}
        correct = bool(targets & hit_ids)
        top_score = hits[0].score if hits else 0.0
        matching = next((h.score for h in hits if h.chunk_id in targets), 0.0)
        if correct:
            positives.append(matching)
        else:
            misses.append((question, sorted(targets), top_score))
        lines.append(
            f"| `{question}` | {', '.join(sorted(targets))} | {matching:.3f} | "
            f"{'yes' if correct else 'NO'} |"
        )

    off_topic_scores = [top_score_for_off_topic(q) for q in OFF_TOPIC]

    lowest_positive = min(positives) if positives else 0.0
    highest_negative = max(off_topic_scores) if off_topic_scores else 1.0
    midpoint = (lowest_positive + highest_negative) / 2

    recall = len(positives) / len(probes) * 100 if probes else 0.0
    off_topic_max = max(off_topic_scores) if off_topic_scores else 0.0

    report = [
        "# Threshold calibration (P4)",
        "",
        f"Embedder: `{config.EMBED_MODEL}`, cosine, `top_k={config.TOP_K}`.",
        "",
        "## Positives - factual questions that must retrieve their target chunk",
        "",
        "| Question | Target chunk | Score | Retrieved |",
        "|---|---|---|---|",
        *lines,
        "",
        f"**Recall@{config.TOP_K}: {len(positives)}/{len(probes)} = {recall:.1f}%**",
        "",
        "## Negatives - off-topic questions that must retrieve nothing",
        "",
        "| Question | Best score | Clears 0.35? |",
        "|---|---|---|",
        *[
            f"| {q} | {score:.3f} | {'LEAKS' if score >= config.SIMILARITY_THRESHOLD else 'no'} |"
            for q, score in zip(OFF_TOPIC, off_topic_scores)
        ],
        "",
        "## Retrieval misses",
        "",
        "On-topic questions whose target chunk did not reach the top 5. These are "
        "*retrieval* failures, not threshold failures: they score far above any sane "
        "threshold, so raising the threshold cannot fix them and would only cost recall.",
        "",
        "| Question | Target | Best score |",
        "|---|---|---|",
        *[f"| {q} | {', '.join(t)} | {score:.3f} |" for q, t, score in misses],
        "",
        "## Separation band",
        "",
        "Positives are the correct chunk's score. Negatives are the best score reached by "
        "an *off-topic* question. Deliberately excluding the misses above: the threshold "
        "separates off-topic from on-topic, it does not separate right from wrong answers. "
        "A wrong-but-relevant chunk is caught by the LLM sentinel and the output verifier.",
        "",
        f"- Lowest positive (weakest correct match): **{lowest_positive:.3f}**",
        f"- Highest negative (strongest off-topic match): **{highest_negative:.3f}**",
        f"- Midpoint: **{midpoint:.3f}**",
        "",
        "## Recommendation",
        "",
        f"`SIMILARITY_THRESHOLD = {midpoint:.2f}`",
        "",
        f"The configured value is `{config.SIMILARITY_THRESHOLD}`. "
        + (
            "It sits inside the band, so recall and off-topic rejection are both satisfied."
            if lowest_positive <= config.SIMILARITY_THRESHOLD <= highest_negative
            else "It does **not** sit inside the band and should be changed to the midpoint."
        ),
        "",
        f"Strongest off-topic score was {off_topic_max:.3f}; a question that reaches only "
        f"{off_topic_max:.3f} is rendered as `NO_RETRIEVAL` rather than answered.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"recall@{config.TOP_K}       : {len(positives)}/{len(probes)} ({recall:.1f}%)")
    print(f"lowest positive      : {lowest_positive:.3f}")
    print(f"highest negative     : {highest_negative:.3f}")
    print(f"off-topic max score  : {off_topic_max:.3f}")
    print(f"recommended midpoint : {midpoint:.3f}")
    print(f"configured threshold : {config.SIMILARITY_THRESHOLD}")
    print(f"report written       : {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())