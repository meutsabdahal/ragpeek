import pytest
from ragpeek.session import RetrievalSpan, GenerationSpan
from ragpeek.config import TracerConfig
from ragpeek.analyzers.retrieval import analyze_retrieval
from ragpeek.analyzers.generation import analyze_generation
from ragpeek.analyzers.context import analyze_context


def make_retrieval_span(scores):
    return RetrievalSpan(
        query="test",
        chunks=[f"chunk {i}" for i in range(len(scores))],
        scores=scores,
        k_requested=len(scores),
        k_returned=len(scores),
    )


def test_retrieval_within_set_padding_flagged():
    # top-heavy set: most chunks trail the best match within this result set
    span = make_retrieval_span([0.9, 0.85, 0.2, 0.15, 0.1])
    analyze_retrieval(span, config=TracerConfig())
    assert any("padding" in d for d in span.diagnosis)


def test_retrieval_flat_distribution_reads_as_low_discrimination():
    # near-identical scores carry no within-set signal except low discrimination
    span = make_retrieval_span([0.3, 0.3, 0.3, 0.3, 0.3])
    analyze_retrieval(span, config=TracerConfig())
    assert any("discrimination" in d.lower() for d in span.diagnosis)


def test_retrieval_clean_spread_raises_no_signal():
    span = make_retrieval_span([0.85, 0.82, 0.80, 0.78, 0.75])
    analyze_retrieval(span, config=TracerConfig())
    assert any("No retrieval signals" in d for d in span.diagnosis)


def test_retrieval_sharp_gap_reads_as_precision_not_noise():
    span = make_retrieval_span([0.92, 0.40, 0.38, 0.35, 0.33])
    analyze_retrieval(span, config=TracerConfig())
    assert any("precision" in d for d in span.diagnosis)
    assert not any("noise" in d.lower() for d in span.diagnosis)


def test_retrieval_absolute_floor_is_opt_in():
    # default config has no absolute floor — no absolute-cutoff signal fires
    default_span = make_retrieval_span([0.3, 0.3, 0.3, 0.3, 0.3])
    analyze_retrieval(default_span, config=TracerConfig())
    assert not any("absolute floor" in d for d in default_span.diagnosis)

    # setting min_score_threshold opts into the calibrated absolute check
    floor_span = make_retrieval_span([0.3, 0.3, 0.3, 0.3, 0.3])
    analyze_retrieval(floor_span, config=TracerConfig(min_score_threshold=0.5))
    assert any("absolute floor" in d for d in floor_span.diagnosis)


def test_retrieval_k_mismatch_flagged():
    span = RetrievalSpan(
        query="test",
        chunks=["a", "b"],
        scores=[0.8, 0.7],
        k_requested=5,
        k_returned=2,
    )
    analyze_retrieval(span, config=TracerConfig())
    assert any("returned" in d for d in span.diagnosis)


def test_generation_hedging_flagged():
    span = GenerationSpan(
        prompt="prompt",
        response="I believe this is generally true and typically works this way.",
        model="test",
    )
    analyze_generation(span, config=TracerConfig())
    assert any("hedging" in d.lower() for d in span.diagnosis)


def test_generation_healthy_response():
    span = GenerationSpan(
        prompt="prompt",
        response="The NEPSE index reached 3,198 points on August 12, 2023.",
        model="test",
    )
    analyze_generation(span, config=TracerConfig())
    assert any("healthy" in d for d in span.diagnosis)


def test_context_analysis_flags_multi_hop_chains_without_semantic_model():
    retrieval_one = make_retrieval_span([0.9, 0.8])
    retrieval_two = make_retrieval_span([0.85, 0.75])
    retrieval_one.event_index = 0
    retrieval_two.event_index = 1

    generation = GenerationSpan(
        prompt="prompt",
        response="The answer depends on both passes.",
        model="test",
    )
    generation.event_index = 2

    report = analyze_context(
        [retrieval_one, retrieval_two],
        generation,
        config=TracerConfig(semantic=False),
    )

    assert report["multi_retrieval"] is True
    assert report["retrieval_event_indices"] == [0, 1]
    assert any("Multi-hop retrieval signal" in d for d in generation.diagnosis)
    assert generation.analysis_notes[0]["code"] == "multi_retrieval"


def test_context_analysis_semantic_missing_dependencies_falls_back(monkeypatch):
    retrieval = make_retrieval_span([0.9, 0.8])
    retrieval.event_index = 0

    generation = GenerationSpan(
        prompt="prompt",
        response="Answer from context.",
        model="test",
    )
    generation.event_index = 1

    def _raise_missing():
        raise RuntimeError("missing semantic extras")

    monkeypatch.setattr(
        "ragpeek.analyzers.context._get_semantic_resources", _raise_missing
    )

    report = analyze_context(
        retrieval,
        generation,
        config=TracerConfig(semantic=True),
    )

    assert any(
        note["code"] == "semantic_dependencies_missing" for note in report["findings"]
    )
    assert any(
        note["code"] == "semantic_dependencies_missing"
        for note in generation.analysis_notes
    )
