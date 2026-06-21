# ragpeek/analyzers/__init__.py
from __future__ import annotations
from typing import Any
from ragpeek.session import TraceSession
from ragpeek.config import TracerConfig
from ragpeek.analyzers.retrieval import analyze_retrieval
from ragpeek.analyzers.context import analyze_context
from ragpeek.analyzers.generation import analyze_generation


def run_all_analyzers(session: TraceSession, config: TracerConfig) -> dict[str, Any]:
    report: dict[str, Any] = {
        "session_id": session.session_id,
        "trace_mode": "semantic" if config.semantic else "non-semantic",
        "retrieval_spans": [],
        "generation_spans": [],
        "context_reports": [],
    }

    for span in session.retrieval_spans:
        analyze_retrieval(span, config=config)
        report["retrieval_spans"].append(
            {
                "event_index": span.event_index,
                "linked_generation_indices": list(span.linked_generation_indices),
                "diagnosis": list(span.diagnosis),
                "analysis_notes": list(span.analysis_notes),
            }
        )

    for span in session.generation_spans:
        analyze_generation(span, config=config)
        linked_retrievals = [
            retrieval_span
            for retrieval_span in session.retrieval_spans
            if retrieval_span.event_index in span.linked_retrieval_indices
        ]
        if not linked_retrievals and session.retrieval_spans:
            linked_retrievals = [session.retrieval_spans[-1]]
        context_report = analyze_context(
            retrieval_spans=linked_retrievals,
            generation_span=span,
            config=config,
        )
        report["context_reports"].append(context_report)
        report["generation_spans"].append(
            {
                "event_index": span.event_index,
                "linked_retrieval_indices": list(span.linked_retrieval_indices),
                "diagnosis": list(span.diagnosis),
                "analysis_notes": list(span.analysis_notes),
            }
        )

    session.analysis_report = report
    return report
