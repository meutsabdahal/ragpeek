from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TracerConfig:
    # retrieval analyzer thresholds.
    # Signals are computed *within* each result set, so they don't depend on
    # your embedder's absolute score scale. min_score_threshold is an opt-in
    # absolute floor — leave it None unless you've calibrated a cutoff for the
    # specific embedder/metric you're using.
    min_score_threshold: float | None = None
    score_gap_threshold: float = 0.3  # rank-1→rank-2 gap that reads as precision
    low_relevance_ratio: float = 0.5  # flag if >this share of chunks trail in-set

    # context analyzer
    semantic: bool = True  # enables embedding-based analysis
    rank_disagreement_position: int = 1  # flag if the answer's best chunk ranks beyond this

    # rendering
    show_prompt: bool = True
    show_chunks: bool = True
    max_chunk_preview: int = 80  # chars to show per chunk in terminal
