from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TracerConfig:
    # retrieval analyzer thresholds
    min_score_threshold: float = 0.5
    score_gap_threshold: float = 0.3
    low_relevance_ratio: float = 0.5  # flag if >50% chunks below min_score

    # context analyzer
    semantic: bool = True  # enables embedding-based analysis
    lost_in_middle_position: int = 1  # flag if best chunk is beyond this index

    # rendering
    show_prompt: bool = True
    show_chunks: bool = True
    max_chunk_preview: int = 80  # chars to show per chunk in terminal
