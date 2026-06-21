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


def _retrieval_span_from_dict(data: dict[str, Any]) -> RetrievalSpan:
    return RetrievalSpan(
        query=data["query"],
        chunks=list(data["chunks"]),
        scores=list(data["scores"]),
        k_requested=data["k_requested"],
        k_returned=data["k_returned"],
        latency_ms=data.get("latency_ms", 0.0),
        diagnosis=list(data.get("diagnosis", [])),
        analysis_notes=list(data.get("analysis_notes", [])),
        event_index=data.get("event_index", -1),
        linked_generation_indices=list(data.get("linked_generation_indices", [])),
    )


def _generation_span_from_dict(data: dict[str, Any]) -> GenerationSpan:
    return GenerationSpan(
        prompt=data["prompt"],
        response=data["response"],
        model=data["model"],
        prompt_tokens=data.get("prompt_tokens", 0),
        response_tokens=data.get("response_tokens", 0),
        latency_ms=data.get("latency_ms", 0.0),
        diagnosis=list(data.get("diagnosis", [])),
        analysis_notes=list(data.get("analysis_notes", [])),
        event_index=data.get("event_index", -1),
        linked_retrieval_indices=list(data.get("linked_retrieval_indices", [])),
    )


def trace_from_dict(data: dict[str, Any]) -> TraceSession:
    """Rebuild a TraceSession from a trace_to_dict() payload (the inverse of
    trace_to_dict). Spans keep their stored diagnoses — analyzers are not re-run.
    """
    session = TraceSession(query=data.get("query", ""))
    if data.get("session_id"):
        session.session_id = data["session_id"]
    session.total_latency_ms = data.get("total_latency_ms", 0.0)
    session.analysis_report = data.get("analysis_report", {}) or {}
    session.retrieval_spans = [
        _retrieval_span_from_dict(span) for span in data.get("retrieval_spans", [])
    ]
    session.generation_spans = [
        _generation_span_from_dict(span) for span in data.get("generation_spans", [])
    ]
    return session


def deserialize_trace(payload: str) -> TraceSession:
    return trace_from_dict(json.loads(payload))
