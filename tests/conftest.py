"""Shared, domain-neutral test data and span builders.

All sample content lives here in one place so the tests aren't tied to a
specific domain. Swapping themes later is a one-constant edit. The corpus is
deliberately free of the generation analyzer's hedging trigger words
(see ragpeek/analyzers/generation.py) so confident text reads as "healthy".
"""

from __future__ import annotations

import pytest

from ragpeek.session import GenerationSpan, RetrievalSpan

SAMPLE_QUERY = "Which is the largest planet in the Solar System?"

SAMPLE_CORPUS = [
    "Jupiter is the largest planet in the Solar System, more massive than all the others combined.",
    "Saturn is the second-largest planet and is best known for its prominent ring system.",
    "Mars, the red planet, hosts Olympus Mons, the tallest volcano in the Solar System.",
    "Venus is the hottest planet, with surface temperatures around 465 degrees Celsius.",
    "Mercury is the smallest planet and the closest to the Sun.",
]

# A confident, non-hedging answer (the generation analyzer should read it healthy).
CONFIDENT_RESPONSE = "Jupiter is the largest planet in the Solar System."


@pytest.fixture
def sample_corpus():
    return list(SAMPLE_CORPUS)


@pytest.fixture
def make_retrieval_span():
    """Builder for RetrievalSpan. Chunks default to the neutral corpus."""

    def _make(scores, chunks=None, *, query=SAMPLE_QUERY, k_requested=None):
        if chunks is None:
            chunks = [SAMPLE_CORPUS[i % len(SAMPLE_CORPUS)] for i in range(len(scores))]
        return RetrievalSpan(
            query=query,
            chunks=chunks,
            scores=list(scores),
            k_requested=k_requested if k_requested is not None else len(scores),
            k_returned=len(chunks),
        )

    return _make


@pytest.fixture
def make_generation_span():
    """Builder for GenerationSpan. Defaults to a confident, non-hedging answer."""

    def _make(response=CONFIDENT_RESPONSE, *, prompt="prompt", model="test"):
        return GenerationSpan(prompt=prompt, response=response, model=model)

    return _make
