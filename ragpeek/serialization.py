from __future__ import annotations

import json
from typing import Any

from ragpeek.session import GenerationSpan, RetrievalSpan, TraceSession


def _retrieval_span_to_dict(span: RetrievalSpan) -> dict[str, Any]:
    return {
        "event_index": span.event_index,
        "linked_generation_indices": list(span.linked_generation_indices),
        "query": span.query,
        "chunks": span.chunks,
        "scores": [round(score, 4) for score in span.scores],
        "k_requested": span.k_requested,
        "k_returned": span.k_returned,
        "latency_ms": round(span.latency_ms, 1),
        "diagnosis": span.diagnosis,
        "analysis_notes": span.analysis_notes,
    }


def _generation_span_to_dict(span: GenerationSpan) -> dict[str, Any]:
    return {
        "event_index": span.event_index,
        "linked_retrieval_indices": list(span.linked_retrieval_indices),
        "prompt": span.prompt,
        "response": span.response,
        "model": span.model,
        "prompt_tokens": span.prompt_tokens,
        "response_tokens": span.response_tokens,
        "latency_ms": round(span.latency_ms, 1),
        "diagnosis": span.diagnosis,
        "analysis_notes": span.analysis_notes,
    }


def trace_to_dict(session: TraceSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "query": session.query,
        "total_latency_ms": round(session.total_latency_ms, 1),
        "analysis_report": session.analysis_report,
        "retrieval_spans": [
            _retrieval_span_to_dict(span) for span in session.retrieval_spans
        ],
        "generation_spans": [
            _generation_span_to_dict(span) for span in session.generation_spans
        ],
    }


def serialize_trace(session: TraceSession, *, indent: int = 2) -> str:
    return json.dumps(trace_to_dict(session), indent=indent)
