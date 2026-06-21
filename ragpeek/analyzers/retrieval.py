from __future__ import annotations
from ragpeek.session import RetrievalSpan
from ragpeek.config import TracerConfig


def analyze_retrieval(span: RetrievalSpan, config: TracerConfig) -> None:
    """
    Examines the similarity-score distribution and populates span.diagnosis.

    Scores are read *relative to the result set* (within-set normalization)
    rather than against absolute cutoffs: raw score scales are embedder- and
    metric-specific, so a cosine 0.5 and an inner-product 0.5 say nothing about
    each other. Every line below is therefore a *signal* to calibrate against
    your own embedder — not a verdict.
    """
    scores = span.scores
    if not scores:
        span.diagnosis.append("No scores available — cannot analyze retrieval.")
        return

    lo, hi = min(scores), max(scores)
    spread = hi - lo

    # Signal 1: within-set padding.
    # Normalize each chunk to its position between the worst (0) and best (1)
    # result in THIS set, then flag when most chunks trail in the lower half.
    # Scale-free: it says nothing about absolute quality, only that the set is
    # top-heavy relative to its own best match.
    if spread > 0:
        normalized = [(s - lo) / spread for s in scores]
        trailing = [n for n in normalized if n < 0.5]
        if len(trailing) / len(scores) > config.low_relevance_ratio:
            span.diagnosis.append(
                f"{len(trailing)} of {len(scores)} chunks sit in the lower half of "
                f"this result's score range (top {hi:.2f}, bottom {lo:.2f}) — "
                f"possible low-relevance padding. Signal — calibrate to your embedder."
            )

    # Signal 2: sharp rank-1 separation reads as PRECISION, not noise.
    # A large gap between rank-1 and rank-2 means the retriever cleanly
    # separated the best match from the field — a confidence/precision signal,
    # the opposite of a problem.
    if len(scores) >= 2:
        gap = scores[0] - scores[1]
        if gap > config.score_gap_threshold:
            span.diagnosis.append(
                f"Sharp rank-1 separation: a {gap:.2f} gap between rank-1 "
                f"({scores[0]:.2f}) and rank-2 ({scores[1]:.2f}). The retriever "
                f"cleanly separates the top match — a precision signal. "
                f"Signal — calibrate to your embedder."
            )

    # Signal 3: flat distribution reads as LOW DISCRIMINATION.
    # If the whole set sits within ~2% of the top score, the retriever didn't
    # separate the chunks at all — the query may be too vague or the chunks too
    # broad. We compare spread to the top *magnitude* (a ratio), so this stays
    # scale-free rather than assuming an absolute score band.
    if len(scores) >= 2 and hi > 0 and spread / hi < 0.02:
        span.diagnosis.append(
            f"Flat score distribution: all chunks score within {spread:.2f} of "
            f"each other (top {hi:.2f}). The retriever barely separates them — "
            f"low discrimination. Signal — calibrate to your embedder."
        )

    # Signal 4 (opt-in): absolute floor.
    # Only fires when you've set a cutoff calibrated to your own embedder.
    if config.min_score_threshold is not None:
        below = [s for s in scores if s < config.min_score_threshold]
        if below:
            span.diagnosis.append(
                f"{len(below)} of {len(scores)} chunks fall below your configured "
                f"absolute floor of {config.min_score_threshold:.2f} — an absolute "
                f"cutoff, only meaningful for the embedder you calibrated it for."
            )

    # Signal 5: structural — fewer chunks returned than requested.
    if span.k_returned < span.k_requested:
        span.diagnosis.append(
            f"Retriever returned {span.k_returned} chunks but {span.k_requested} "
            f"were requested. The corpus may be too small, or a retriever-side "
            f"score filter is dropping results."
        )

    # No signals raised.
    if not span.diagnosis:
        span.diagnosis.append(
            f"No retrieval signals — top scores within set: "
            f"{', '.join(f'{s:.2f}' for s in scores[:3])}."
        )
