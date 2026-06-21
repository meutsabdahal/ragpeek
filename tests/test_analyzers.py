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


def test_retrieval_no_scores_handled():
    span = RetrievalSpan(
        query="test", chunks=[], scores=[], k_requested=0, k_returned=0
    )
    analyze_retrieval(span, config=TracerConfig())
    assert any("No scores available" in d for d in span.diagnosis)


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


# --- semantic context analysis -------------------------------------------------
# These exercise the embedding-based body of analyze_context without downloading
# a real model: we replace _get_semantic_resources with a fake that yields exactly
# the response/chunk cosine similarities each test needs.


def _make_retrieval_span_with_chunks(chunks, scores):
    return RetrievalSpan(
        query="test",
        chunks=chunks,
        scores=scores,
        k_requested=len(chunks),
        k_returned=len(chunks),
    )


def _fake_semantic_resources(similarities):
    """Return a (_get_semantic_resources) replacement producing controllable
    response-vs-chunk cosine similarities (in chunk order), no model download.
    """
    response_vec = [1.0, 0.0]

    def _vec_for(sim):
        # 2-D unit vector whose cosine with response_vec equals `sim`
        return [sim, max(0.0, 1.0 - sim * sim) ** 0.5]

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            # texts[0] is the response; texts[1:] are the chunks, in order
            return [response_vec] + [_vec_for(s) for s in similarities]

    def _fake_cosine(a, b):
        va, vb = a[0], b[0]
        dot = sum(x * y for x, y in zip(va, vb))
        na = sum(x * x for x in va) ** 0.5
        nb = sum(x * x for x in vb) ** 0.5
        return [[dot / (na * nb if na and nb else 1.0)]]

    return lambda: (_FakeModel(), _fake_cosine)


def _patch_semantic(monkeypatch, similarities):
    monkeypatch.setattr(
        "ragpeek.analyzers.context._get_semantic_resources",
        _fake_semantic_resources(similarities),
    )


def test_semantic_context_records_best_chunk_without_findings(monkeypatch):
    _patch_semantic(monkeypatch, [0.9, 0.7, 0.6])
    retrieval = make_retrieval_span([0.9, 0.8, 0.7])
    retrieval.event_index = 0
    generation = GenerationSpan(
        prompt="prompt",
        response="The NEPSE index reached 3198 points, grounded in the first chunk.",
        model="test",
    )
    generation.event_index = 1

    report = analyze_context(retrieval, generation, config=TracerConfig(semantic=True))

    assert report["pairings"][0]["best_chunk_index"] == 0
    assert report["pairings"][0]["best_similarity"] == pytest.approx(0.9, abs=1e-3)
    assert report["findings"] == []
    assert generation.analysis_notes == []


def test_semantic_context_flags_rank_disagreement(monkeypatch):
    # response aligns most with chunk index 2 — past rank_disagreement_position (1)
    _patch_semantic(monkeypatch, [0.50, 0.55, 0.85])
    retrieval = make_retrieval_span([0.90, 0.80, 0.70])
    retrieval.event_index = 0
    generation = GenerationSpan(
        prompt="prompt",
        response="An answer grounded in the third retrieved chunk, not the first.",
        model="test",
    )
    generation.event_index = 1

    report = analyze_context(retrieval, generation, config=TracerConfig(semantic=True))

    codes = [n["code"] for n in report["findings"]]
    assert "rank_disagreement" in codes
    assert any("Rank-disagreement" in d for d in generation.diagnosis)
    assert any("Rank-disagreement" in d for d in retrieval.diagnosis)
    note = next(n for n in report["findings"] if n["code"] == "rank_disagreement")
    assert note["details"]["retrieval_chunk_index"] == 2


def test_semantic_context_flags_low_context_utilisation(monkeypatch):
    # every chunk is dissimilar to the response (best below the 0.4 floor)
    _patch_semantic(monkeypatch, [0.30, 0.25, 0.20])
    retrieval = make_retrieval_span([0.90, 0.80, 0.70])
    retrieval.event_index = 0
    generation = GenerationSpan(
        prompt="prompt",
        response="A confident answer that ignores the retrieved context entirely.",
        model="test",
    )
    generation.event_index = 1

    report = analyze_context(retrieval, generation, config=TracerConfig(semantic=True))

    codes = [n["code"] for n in report["findings"]]
    assert "low_context_utilisation" in codes
    assert "rank_disagreement" not in codes  # best chunk is index 0
    note = next(n for n in report["findings"] if n["code"] == "low_context_utilisation")
    assert note["details"]["best_similarity"] == pytest.approx(0.30, abs=1e-3)


def test_semantic_context_flags_short_response_over_context(monkeypatch):
    _patch_semantic(monkeypatch, [0.60, 0.50, 0.45])
    long_chunk = " ".join(["context"] * 90)
    retrieval = _make_retrieval_span_with_chunks(
        [long_chunk, long_chunk + " x", long_chunk + " y"],
        [0.90, 0.80, 0.70],
    )
    retrieval.event_index = 0
    generation = GenerationSpan(prompt="prompt", response="Too short.", model="test")
    generation.event_index = 1

    report = analyze_context(retrieval, generation, config=TracerConfig(semantic=True))

    codes = [n["code"] for n in report["findings"]]
    assert "short_response_over_context" in codes
    note = next(
        n for n in report["findings"] if n["code"] == "short_response_over_context"
    )
    assert note["details"]["response_word_count"] == 2
    assert note["details"]["context_word_count"] > 200


def test_semantic_context_empty_response_returns_before_embedding(monkeypatch):
    encoded = {"called": False}

    def _resources():
        class _M:
            def encode(self, *a, **k):
                encoded["called"] = True
                return []

        return _M(), (lambda a, b: [[0.0]])

    monkeypatch.setattr(
        "ragpeek.analyzers.context._get_semantic_resources", _resources
    )
    retrieval = make_retrieval_span([0.9, 0.8])
    retrieval.event_index = 0
    generation = GenerationSpan(prompt="prompt", response="", model="test")
    generation.event_index = 1

    report = analyze_context(retrieval, generation, config=TracerConfig(semantic=True))

    assert report["findings"] == []
    assert report["pairings"] == []
    assert encoded["called"] is False  # bailed out before any embedding work


def test_semantic_context_handles_empty_chunks(monkeypatch):
    _patch_semantic(monkeypatch, [])
    retrieval = RetrievalSpan(
        query="q", chunks=[], scores=[], k_requested=0, k_returned=0
    )
    retrieval.event_index = 0
    generation = GenerationSpan(
        prompt="prompt",
        response="A reasonably long answer that should not trip any length heuristic.",
        model="test",
    )
    generation.event_index = 1

    report = analyze_context(retrieval, generation, config=TracerConfig(semantic=True))

    assert report["pairings"][0]["chunk_count"] == 0
    assert report["findings"] == []
