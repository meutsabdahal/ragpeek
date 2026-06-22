import io
import json
import re
from pathlib import Path

import pytest
from rich.console import Console

from ragpeek.cli import main
from ragpeek.renderers import terminal as terminal_renderer
from ragpeek.serialization import deserialize_trace, trace_to_dict
from ragpeek.session import GenerationSpan, RetrievalSpan, TraceSession


@pytest.fixture
def captured_console(monkeypatch):
    """Route the terminal renderer's console into a buffer we can assert on."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        terminal_renderer,
        "console",
        Console(file=buffer, force_terminal=False, width=90, color_system=None),
    )
    return buffer


def _keyword_resources():
    """A fake _get_semantic_resources: deterministic bag-of-words embeddings so
    retrieval ranks by exact word overlap — no model download, no network, and no
    reliance on hash() (which is salted per process and would make this flaky)."""
    vocab: dict[str, int] = {}

    def _vec(text):
        counts: dict[int, float] = {}
        for word in re.findall(r"[a-z]+", text.lower()):
            idx = vocab.setdefault(word, len(vocab))
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return counts

    class _Model:
        def encode(self, texts, normalize_embeddings=True):
            return [_vec(t) for t in texts]

    def _cosine(a, b):
        va, vb = a[0], b[0]
        dot = sum(weight * vb[i] for i, weight in va.items() if i in vb)
        na = sum(w * w for w in va.values()) ** 0.5
        nb = sum(w * w for w in vb.values()) ** 0.5
        return [[dot / (na * nb) if na and nb else 0.0]]

    return lambda: (_Model(), _cosine)


@pytest.fixture
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(
        "ragpeek.analyzers.context._get_semantic_resources", _keyword_resources()
    )


def _stub_llm(monkeypatch, response):
    monkeypatch.setattr(
        "ragpeek.cli._ollama_generate",
        lambda prompt, *, model, timeout=60.0: response,
    )


def test_demo_retrieval_ranks_relevant_doc(fake_embeddings):
    from ragpeek.cli import _retrieve

    chunks, scores = _retrieve("How hot is Venus?", 3)
    assert len(chunks) == 3
    assert "Venus" in chunks[0]  # the question's most relevant built-in doc


def test_cli_demo_renders_a_trace(captured_console, fake_embeddings, monkeypatch):
    _stub_llm(monkeypatch, "Venus is the hottest planet in the Solar System.")

    rc = main(["demo", "How hot is Venus?"])
    output = captured_console.getvalue()

    assert rc == 0
    assert "Retrieval" in output
    assert "Generation" in output
    assert "Venus" in output  # retrieved the relevant doc and answered about it


def test_cli_demo_without_llm_notes_retrieval_only(
    captured_console, fake_embeddings, monkeypatch, capsys
):
    _stub_llm(monkeypatch, None)  # no Ollama reachable

    rc = main(["demo", "How hot is Venus?"])

    assert rc == 0
    assert "Retrieval" in captured_console.getvalue()
    assert "no LLM reached" in capsys.readouterr().err


def test_cli_demo_html_export(captured_console, fake_embeddings, monkeypatch, tmp_path):
    _stub_llm(monkeypatch, "An answer.")
    out = tmp_path / "report.html"

    rc = main(["demo", "How hot is Venus?", "--html", str(out)])

    assert rc == 0
    assert out.is_file()
    assert "<!DOCTYPE html>" in out.read_text()


def test_cli_demo_without_semantic_extra_shows_install_hint(monkeypatch, capsys):
    def _raise():
        raise RuntimeError("semantic deps missing")

    monkeypatch.setattr(
        "ragpeek.analyzers.context._get_semantic_resources", _raise
    )
    rc = main(["demo", "How hot is Venus?"])
    assert rc == 2
    assert "semantic extra" in capsys.readouterr().err


def test_cli_views_saved_trace_file(captured_console):
    fixture = Path(__file__).parent / "fixtures" / "sample_session.json"
    rc = main([str(fixture)])  # bare path → view

    assert rc == 0
    output = captured_console.getvalue()
    assert "largest planet" in output  # the query
    assert "padding" in output  # a stored diagnosis


def test_cli_missing_file_reports_error(capsys):
    rc = main(["does-not-exist.json"])
    assert rc == 2
    assert "no such trace file" in capsys.readouterr().err


def test_trace_from_dict_round_trips():
    session = TraceSession(query="q")
    retrieval = RetrievalSpan(
        query="q", chunks=["a", "b"], scores=[0.9, 0.2], k_requested=2, k_returned=2
    )
    session.add_retrieval_span(retrieval)
    session.add_generation_span(
        GenerationSpan(prompt="p", response="r", model="m")
    )
    retrieval.diagnosis.append("some retrieval signal")

    restored = deserialize_trace(json.dumps(trace_to_dict(session)))

    assert restored.query == "q"
    assert restored.session_id == session.session_id
    assert len(restored.retrieval_spans) == 1
    assert len(restored.generation_spans) == 1
    assert restored.retrieval_spans[0].diagnosis == ["some retrieval signal"]
    assert restored.retrieval_spans[0].event_index == 0
    assert restored.generation_spans[0].linked_retrieval_indices == [0]
