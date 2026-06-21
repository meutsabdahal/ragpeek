from __future__ import annotations
from ragpeek.session import RetrievalSpan
from ragpeek.config import TracerConfig


def analyze_retrieval(span: RetrievalSpan, config: TracerConfig) -> None:
    """
    Examines similarity score distribution and populates span.diagnosis.
    All heuristics are rule-based — no ML, fast, predictable.
    """
    scores = span.scores
    if not scores:
        span.diagnosis.append("No scores available — cannot analyze retrieval.")
        return

    low_scores = [s for s in scores if s < config.min_score_threshold]
    low_ratio = len(low_scores) / len(scores)

    # Heuristic 1: too many low-relevance chunks
    if low_ratio > config.low_relevance_ratio:
        span.diagnosis.append(
            f"{len(low_scores)} of {len(scores)} chunks scored below "
            f"{config.min_score_threshold} — low-relevance padding detected. "
            f"Consider reducing k or raising your retriever's min_score threshold."
        )

    # Heuristic 2: large gap between top-2 scores
    # means the top result dominates and the rest is noise
    if len(scores) >= 2:
        gap = scores[0] - scores[1]
        if gap > config.score_gap_threshold:
            span.diagnosis.append(
                f"Large score gap between rank-1 ({scores[0]:.2f}) and "
                f"rank-2 ({scores[1]:.2f}) — gap: {gap:.2f}. "
                f"Only one chunk is clearly relevant; the rest may be noise."
            )

    # Heuristic 3: all scores are very high
    # might mean chunks are too broad / query too vague
    if scores and min(scores) > 0.90:
        span.diagnosis.append(
            "All scores are very high (>0.90) — chunks may be too broad "
            "or the query is too vague. Consider more specific queries or "
            "finer-grained chunking."
        )

    # Heuristic 4: k_returned < k_requested
    if span.k_returned < span.k_requested:
        span.diagnosis.append(
            f"Retriever returned {span.k_returned} chunks but {span.k_requested} "
            f"were requested. Corpus may be too small or minimum score threshold "
            f"is filtering too aggressively."
        )

    # no issues found
    if not span.diagnosis:
        span.diagnosis.append(
            f"Retrieval looks healthy — top scores: "
            f"{', '.join(f'{s:.2f}' for s in scores[:3])}"
        )
