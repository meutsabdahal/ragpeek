import io

from rich.console import Console

from ragpeek.renderers.html import render_html
from ragpeek.renderers import terminal as terminal_renderer
from ragpeek.config import TracerConfig
from ragpeek.session import GenerationSpan, RetrievalSpan, TraceSession


def test_render_html_escapes_dynamic_content():
    session = TraceSession(query="What happens with <script>alert(1)</script>?")
    session.add_retrieval_span(
        RetrievalSpan(
            query="What happens with <script>alert(1)</script>?",
            chunks=["chunk <img src=x onerror=alert(1)>"],
            scores=[0.9],
            k_requested=1,
            k_returned=1,
        )
    )
    session.add_generation_span(
        GenerationSpan(
            prompt="prompt <b>unsafe</b>",
            response="response <b>unsafe</b>",
            model="test-model",
        )
    )

    html = render_html(session)

    assert "What happens with &lt;script&gt;alert(1)&lt;/script&gt;?" in html
    assert "\\u003cimg src=x onerror=alert(1)>" in html
    assert "\\u003cb>unsafe\\u003c/b>" in html


def test_render_html_includes_trace_mode_and_collapsible_sections():
    session = TraceSession(query="What is the answer?")
    session.analysis_report = {"trace_mode": "non-semantic"}

    for index in range(5):
        session.add_retrieval_span(
            RetrievalSpan(
                query="What is the answer?",
                chunks=[f"chunk {index}"],
                scores=[0.9],
                k_requested=1,
                k_returned=1,
            )
        )

    html = render_html(session)

    assert '"trace_mode": "non-semantic"' in html
    assert "document.createElement('details')" in html
    assert "if (!true) section.open = true;" in html


def test_terminal_render_session_wraps_long_text_and_shows_mode(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(
        terminal_renderer,
        "console",
        Console(file=buffer, force_terminal=False, width=72, color_system=None),
    )

    session = TraceSession(query=" ".join(["long-query"] * 12))
    session.add_retrieval_span(
        RetrievalSpan(
            query=session.query,
            chunks=["chunk text " + "alpha beta " * 10],
            scores=[0.91],
            k_requested=1,
            k_returned=1,
        )
    )
    session.add_generation_span(
        GenerationSpan(
            prompt="prompt text " + "gamma delta " * 10,
            response="response text " + "epsilon zeta " * 10,
            model="test-model",
        )
    )

    terminal_renderer.render_session(session, TracerConfig(semantic=False))
    output = buffer.getvalue()

    assert "Trace mode: non-semantic" in output
    assert "long-query" in output
    assert "chunk text" in output
    assert "response text" in output
