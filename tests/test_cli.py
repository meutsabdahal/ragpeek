import io
import json
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


def test_cli_demo_renders_a_trace(captured_console):
    rc = main(["demo"])  # semantic off by default — no model download
    output = captured_console.getvalue()

    assert rc == 0
    assert "Retrieval" in output
    assert "Generation" in output
    # the built-in demo data is top-heavy and the answer hedges
    assert "padding" in output
    assert "hedging" in output.lower()


def test_cli_views_saved_trace_file(captured_console):
    fixture = Path(__file__).parent / "fixtures" / "sample_session.json"
    rc = main([str(fixture)])  # bare path → view
    output = captured_console.getvalue()

    assert rc == 0
    assert "largest planet" in output  # the query
    assert "padding" in output  # a stored diagnosis


def test_cli_missing_file_reports_error(capsys):
    rc = main(["does-not-exist.json"])
    assert rc == 2
    assert "no such trace file" in capsys.readouterr().err


def test_cli_html_export(captured_console, tmp_path):
    out = tmp_path / "report.html"
    rc = main(["demo", "--html", str(out)])
    assert rc == 0
    assert out.is_file()
    assert "<!DOCTYPE html>" in out.read_text()


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
