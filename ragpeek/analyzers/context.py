from __future__ import annotations
from functools import lru_cache
from collections.abc import Sequence
from typing import Any
from ragpeek.session import RetrievalSpan, GenerationSpan
from ragpeek.config import TracerConfig


@lru_cache(maxsize=1)
def _get_semantic_resources():
    """Lazily import semantic dependencies and cache heavy runtime objects."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    except ImportError as exc:
        raise RuntimeError(
            "Semantic analysis requires optional dependencies. "
            "Install with: pip install ragpeek[semantic]"
        ) from exc

    from rich.console import Console

    Console().print("[dim]Loading embedding model (first run)...[/dim]")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, sk_cosine


def _cosine_similarity(a, b) -> float:
    _, sk_cosine = _get_semantic_resources()

    score = float(sk_cosine([a], [b])[0][0])
    return max(0.0, min(1.0, score))


def _record_note(
    span,
    *,
    code: str,
    severity: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    note = {
        "code": code,
        "severity": severity,
        "message": message,
        "details": details,
    }
    span.analysis_notes.append(note)
    span.diagnosis.append(message)
    return note


def analyze_context(
    retrieval_spans: RetrievalSpan | Sequence[RetrievalSpan],
    generation_span: GenerationSpan,
    config: TracerConfig,
) -> dict[str, Any]:
    """
    Checks how well the generated response used the retrieved context.
    Requires semantic=True in config (uses sentence-transformers).
    """
    if isinstance(retrieval_spans, RetrievalSpan):
        retrieval_spans = [retrieval_spans]

    report: dict[str, Any] = {
        "generation_event_index": generation_span.event_index,
        "retrieval_event_indices": [span.event_index for span in retrieval_spans],
        "multi_retrieval": len(retrieval_spans) > 1,
        "pairings": [],
        "findings": [],
    }

    if len(retrieval_spans) > 1:
        note = _record_note(
            generation_span,
            code="multi_retrieval",
            severity="info",
            message=(
                f"Multi-hop retrieval signal: {len(retrieval_spans)} retrieval steps "
                f"were linked to this answer."
            ),
            retrieval_count=len(retrieval_spans),
            retrieval_event_indices=report["retrieval_event_indices"],
        )
        report["findings"].append(note)

    if not config.semantic:
        return report

    try:
        model, _ = _get_semantic_resources()
    except RuntimeError:
        note = _record_note(
            generation_span,
            code="semantic_dependencies_missing",
            severity="info",
            message=(
                "Semantic analysis is enabled but optional semantic dependencies "
                "are not installed. Falling back to non-semantic analysis."
            ),
            install_hint="pip install ragpeek[semantic]",
        )
        report["findings"].append(note)
        return report

    response = generation_span.response
    if not response:
        return report

    context_word_count = 0
    best_overall: dict[str, Any] | None = None

    for retrieval_span in retrieval_spans:
        chunks = retrieval_span.chunks
        pair_report: dict[str, Any] = {
            "retrieval_event_index": retrieval_span.event_index,
            "chunk_count": len(chunks),
            "findings": [],
        }

        if not chunks:
            report["pairings"].append(pair_report)
            continue

        all_texts = [response] + chunks
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        response_emb = embeddings[0]
        chunk_embs = embeddings[1:]

        similarities = [
            _cosine_similarity(response_emb, chunk_emb) for chunk_emb in chunk_embs
        ]

        best_chunk_idx = similarities.index(max(similarities))
        best_score = similarities[best_chunk_idx]
        context_word_count += sum(len(chunk.split()) for chunk in chunks)

        pair_report["best_chunk_index"] = best_chunk_idx
        pair_report["best_similarity"] = round(best_score, 4)

        if best_overall is None or best_score > best_overall["best_similarity"]:
            best_overall = {
                "retrieval_span": retrieval_span,
                "best_chunk_index": best_chunk_idx,
                "best_similarity": best_score,
            }

        if best_chunk_idx > config.rank_disagreement_position:
            original_score = retrieval_span.scores[best_chunk_idx]
            top_score = retrieval_span.scores[0]
            message = (
                f"Rank-disagreement signal: the response aligns most with chunk "
                f"#{best_chunk_idx + 1} (retrieval rank {best_chunk_idx + 1}, "
                f"score {original_score:.2f}), not the retriever's top-ranked "
                f"chunk #1 (score {top_score:.2f}). The retriever's ordering "
                f"disagrees with the chunk the answer actually used — a reranking "
                f"signal. Signal — calibrate to your embedder."
            )
            note = _record_note(
                generation_span,
                code="rank_disagreement",
                severity="warn",
                message=message,
                retrieval_event_index=retrieval_span.event_index,
                retrieval_chunk_index=best_chunk_idx,
                response_similarity=round(best_score, 4),
                retrieval_score=round(original_score, 4),
                top_score=round(top_score, 4),
            )
            retrieval_span.analysis_notes.append(note)
            retrieval_span.diagnosis.append(message)
            pair_report["findings"].append(note)
            report["findings"].append(note)

        report["pairings"].append(pair_report)

    if best_overall is not None and best_overall["best_similarity"] < 0.4:
        note = _record_note(
            generation_span,
            code="low_context_utilisation",
            severity="warn",
            message=(
                f"Low context utilisation — response has low semantic similarity "
                f"to the retrieved chunks (best: {best_overall['best_similarity']:.2f}). "
                f"The LLM may be answering from its training weights rather than "
                f"the provided context. Signal — this similarity floor is absolute; "
                f"calibrate it to your embedder."
            ),
            best_similarity=round(best_overall["best_similarity"], 4),
            retrieval_event_index=best_overall["retrieval_span"].event_index,
            retrieval_chunk_index=best_overall["best_chunk_index"],
        )
        report["findings"].append(note)

    response_word_count = len(response.split())
    if response_word_count < 20 and context_word_count > 200:
        note = _record_note(
            generation_span,
            code="short_response_over_context",
            severity="warn",
            message=(
                f"Response ({response_word_count} words) is very short relative "
                f"to context ({context_word_count} words). "
                f"Consider whether the model has enough information or if the "
                f"prompt is over-constraining response length."
            ),
            response_word_count=response_word_count,
            context_word_count=context_word_count,
        )
        report["findings"].append(note)

    return report
