import pytest
from ragpeek.session import RetrievalSpan, GenerationSpan, TraceSession
import json
from pathlib import Path


def test_retrieval_span_score_length_mismatch():
    with pytest.raises(ValueError):
        RetrievalSpan(
            query="test",
            chunks=["a", "b"],
            scores=[0.9],  # length mismatch
            k_requested=2,
            k_returned=2,
        )


def test_generation_span_token_estimation():
    span = GenerationSpan(
        prompt="a" * 400,
        response="b" * 100,
        model="llama3.2",
    )
    # estimated from char count: 400//4=100, 100//4=25
    assert span.prompt_tokens == 100
    assert span.response_tokens == 25


def test_trace_session_has_retrieval():
    session = TraceSession(query="test")
    assert not session.has_retrieval

    session.add_retrieval_span(
        RetrievalSpan(
            query="test",
            chunks=["chunk"],
            scores=[0.8],
            k_requested=1,
            k_returned=1,
        )
    )
    assert session.has_retrieval


def test_trace_session_links_retrievals_to_generation_events():
    session = TraceSession(query="test")

    first_retrieval = RetrievalSpan(
        query="test",
        chunks=["chunk 1"],
        scores=[0.8],
        k_requested=1,
        k_returned=1,
    )
    second_retrieval = RetrievalSpan(
        query="test",
        chunks=["chunk 2"],
        scores=[0.7],
        k_requested=1,
        k_returned=1,
    )

    session.add_retrieval_span(first_retrieval)
    session.add_generation_span(
        GenerationSpan(prompt="prompt 1", response="response 1", model="test")
    )
    session.add_retrieval_span(second_retrieval)
    session.add_generation_span(
        GenerationSpan(prompt="prompt 2", response="response 2", model="test")
    )

    assert first_retrieval.event_index == 0
    assert first_retrieval.linked_generation_indices == [1]
    assert second_retrieval.event_index == 2
    assert second_retrieval.linked_generation_indices == [3]
    assert session.generation_spans[0].linked_retrieval_indices == [0]
    assert session.generation_spans[1].linked_retrieval_indices == [2]


def test_trace_session_explicit_linking_is_idempotent():
    session = TraceSession(query="test")

    retrieval = RetrievalSpan(
        query="test",
        chunks=["chunk"],
        scores=[0.8],
        k_requested=1,
        k_returned=1,
    )
    session.add_retrieval_span(retrieval)
    generation = GenerationSpan(prompt="prompt", response="response", model="test")
    session.add_generation_span(generation)

    session.link_retrieval_to_generation(retrieval.event_index, generation.event_index)
    session.link_retrieval_to_generation(retrieval.event_index, generation.event_index)

    assert retrieval.linked_generation_indices == [1]
    assert generation.linked_retrieval_indices == [0]


def test_fixture_loads_correctly():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/sample_session.json").read_text()
    )
    assert fixture["session_id"] == "abc12345"
    assert len(fixture["retrieval_spans"][0]["chunks"]) == 5
    assert len(fixture["retrieval_spans"][0]["scores"]) == 5
