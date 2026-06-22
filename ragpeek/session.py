from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass
class RetrievalSpan:
    query: str
    chunks: list[str]
    scores: list[float]
    k_requested: int
    k_returned: int
    latency_ms: float = 0.0
    diagnosis: list[str] = field(default_factory=list)
    analysis_notes: list[dict[str, Any]] = field(default_factory=list)
    event_index: int = -1
    linked_generation_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        # normalise scores — some vector stores return distances
        # (lower = better), others return similarities (higher = better).
        # ragpeek assumes higher = better. Users are responsible for
        # converting distances to similarities before calling log_retrieval.
        if len(self.scores) != len(self.chunks):
            raise ValueError(
                f"scores length ({len(self.scores)}) must match "
                f"chunks length ({len(self.chunks)})"
            )


@dataclass
class GenerationSpan:
    prompt: str
    response: str
    model: str
    prompt_tokens: int = 0
    response_tokens: int = 0
    latency_ms: float = 0.0
    diagnosis: list[str] = field(default_factory=list)
    analysis_notes: list[dict[str, Any]] = field(default_factory=list)
    event_index: int = -1
    linked_retrieval_indices: list[int] = field(default_factory=list)

    def __post_init__(self):
        # estimate token counts if not provided
        # rough approximation: 1 token ≈ 4 chars
        if self.prompt_tokens == 0:
            self.prompt_tokens = len(self.prompt) // 4
        if self.response_tokens == 0:
            self.response_tokens = len(self.response) // 4


@dataclass
class TraceSession:
    query: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    retrieval_spans: list[RetrievalSpan] = field(default_factory=list)
    generation_spans: list[GenerationSpan] = field(default_factory=list)
    analysis_report: dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    _start_time: float = field(default_factory=lambda: time.perf_counter(), repr=False)
    _last_event_time: float = field(init=False, repr=False)
    _event_counter: int = field(default=0, init=False, repr=False)
    _pending_retrieval_indices: list[int] = field(
        default_factory=list, init=False, repr=False
    )
    _retrieval_spans_by_event_index: dict[int, RetrievalSpan] = field(
        default_factory=dict, init=False, repr=False
    )
    _generation_spans_by_event_index: dict[int, GenerationSpan] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        self._last_event_time = self._start_time

    def add_retrieval_span(self, span: RetrievalSpan) -> None:
        span.event_index = self._event_counter
        self._event_counter += 1
        self.retrieval_spans.append(span)
        self._retrieval_spans_by_event_index[span.event_index] = span
        self._pending_retrieval_indices.append(span.event_index)

    def add_generation_span(self, span: GenerationSpan) -> None:
        span.event_index = self._event_counter
        self._event_counter += 1
        span.linked_retrieval_indices = list(self._pending_retrieval_indices)
        for retrieval_index in span.linked_retrieval_indices:
            retrieval_span = self._retrieval_spans_by_event_index.get(retrieval_index)
            if retrieval_span is not None:
                retrieval_span.linked_generation_indices.append(span.event_index)
        self._pending_retrieval_indices.clear()
        self.generation_spans.append(span)
        self._generation_spans_by_event_index[span.event_index] = span

    def link_retrieval_to_generation(
        self, retrieval_event_index: int, generation_event_index: int
    ) -> None:
        retrieval_span = self._retrieval_spans_by_event_index.get(retrieval_event_index)
        if retrieval_span is None:
            raise ValueError(f"Unknown retrieval event index: {retrieval_event_index}")

        generation_span = self._generation_spans_by_event_index.get(
            generation_event_index
        )
        if generation_span is None:
            raise ValueError(
                f"Unknown generation event index: {generation_event_index}"
            )

        if retrieval_event_index not in generation_span.linked_retrieval_indices:
            generation_span.linked_retrieval_indices.append(retrieval_event_index)
        if generation_event_index not in retrieval_span.linked_generation_indices:
            retrieval_span.linked_generation_indices.append(generation_event_index)

    def record_span_latency(self) -> float:
        now = time.perf_counter()
        latency_ms = (now - self._last_event_time) * 1000
        self._last_event_time = now
        return latency_ms

    def finalise(self) -> None:
        self.total_latency_ms = (time.perf_counter() - self._start_time) * 1000

    @property
    def has_retrieval(self) -> bool:
        return len(self.retrieval_spans) > 0

    @property
    def has_generation(self) -> bool:
        return len(self.generation_spans) > 0
