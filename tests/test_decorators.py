import asyncio
import json
import pytest
from ragpeek import trace, log_retrieval, log_generation, link_retrieval_to_generation
from ragpeek.collector import get_collector
from ragpeek.serialization import serialize_trace


def setup_function():
    get_collector().clear()


def test_sync_decorator_returns_value():
    @trace
    def my_func(query: str) -> str:
        return f"answer to: {query}"

    result = my_func("test query")
    assert result == "answer to: test query"


def test_sync_decorator_preserves_name():
    @trace
    def my_pipeline(query: str) -> str:
        return query

    assert my_pipeline.__name__ == "my_pipeline"


@pytest.mark.asyncio
async def test_async_decorator_returns_value():
    @trace
    async def async_pipeline(query: str) -> str:
        return f"async answer: {query}"

    result = await async_pipeline("async query")
    assert result == "async answer: async query"


@pytest.mark.asyncio
async def test_concurrent_async_traces_do_not_cross_spans():
    # The headline feature: each @trace-decorated coroutine gets its own session
    # via ContextVar. We interleave the coroutines with await points *between*
    # logging retrieval and generation, so a shared/global "current session"
    # would route spans into the wrong session and corrupt the assertions below.
    @trace(semantic=False, render=False)
    async def pipeline(query: str) -> object:
        log_retrieval(query=query, chunks=[f"chunk for {query}"], scores=[0.9])
        await asyncio.sleep(0)  # yield — let sibling coroutines run here
        log_generation(
            prompt=f"prompt for {query}",
            response=f"response for {query}",
            model="test",
        )
        await asyncio.sleep(0)
        return get_collector().get_current_session()

    sessions = await asyncio.gather(
        pipeline("alpha"),
        pipeline("beta"),
        pipeline("gamma"),
    )

    by_query = {s.query: s for s in sessions}
    assert set(by_query) == {"alpha", "beta", "gamma"}

    # session ids are all distinct
    assert len({s.session_id for s in sessions}) == 3

    for query, session in by_query.items():
        # exactly its own one retrieval + one generation — nothing leaked in
        assert len(session.retrieval_spans) == 1
        assert len(session.generation_spans) == 1
        assert session.retrieval_spans[0].chunks == [f"chunk for {query}"]
        assert session.generation_spans[0].response == f"response for {query}"
        # and the generation linked to this session's own retrieval, not a sibling's
        assert session.generation_spans[0].linked_retrieval_indices == [
            session.retrieval_spans[0].event_index
        ]


def test_nested_traces_restore_outer_session():
    observed: dict[str, str | None] = {}

    @trace
    def inner() -> str:
        session = get_collector().get_current_session()
        assert session is not None
        observed["inner_session_id"] = session.session_id
        return session.session_id

    @trace
    def outer() -> str:
        session = get_collector().get_current_session()
        assert session is not None
        observed["outer_session_id"] = session.session_id
        inner_session_id = inner()
        current_session = get_collector().get_current_session()
        observed["after_inner_session_id"] = (
            None if current_session is None else current_session.session_id
        )
        return inner_session_id

    outer()

    assert observed["inner_session_id"] is not None
    assert observed["outer_session_id"] is not None
    assert observed["inner_session_id"] != observed["outer_session_id"]
    assert observed["after_inner_session_id"] == observed["outer_session_id"]


def test_log_retrieval_outside_trace_is_noop():
    # calling log_retrieval without @trace should not raise
    log_retrieval(query="test", chunks=["chunk"], scores=[0.8])


def test_trace_with_output_arg():
    import os

    @trace(output="/tmp/test_ragpeek_report.html")
    def pipeline(query: str) -> str:
        log_retrieval(query=query, chunks=["chunk one"], scores=[0.75])
        log_generation(prompt="prompt", response="response", model="test")
        return "done"

    pipeline("test query")
    assert os.path.exists("/tmp/test_ragpeek_report.html")
    os.remove("/tmp/test_ragpeek_report.html")


def test_trace_can_disable_rendering(monkeypatch):
    render_calls = []
    analyzer_calls = []

    monkeypatch.setattr(
        "ragpeek.renderers.terminal.render_session",
        lambda *args, **kwargs: render_calls.append((args, kwargs)),
    )

    def fake_run_all_analyzers(session, config):
        analyzer_calls.append((session.session_id, config.semantic))
        session.analysis_report = {"trace_mode": "non-semantic"}
        return session.analysis_report

    monkeypatch.setattr(
        "ragpeek.analyzers.run_all_analyzers",
        fake_run_all_analyzers,
    )

    @trace(render=False, semantic=False)
    def pipeline(query: str) -> str:
        log_retrieval(query=query, chunks=["chunk"], scores=[0.9])
        log_generation(prompt="prompt", response="response", model="test")
        return "done"

    assert pipeline("test query") == "done"
    assert analyzer_calls
    assert render_calls == []


def test_explicit_linking_and_serialization():
    @trace(render=False, semantic=False)
    def pipeline(query: str):
        retrieval = log_retrieval(query=query, chunks=["chunk 1"], scores=[0.8])
        generation = log_generation(prompt="prompt", response="response", model="test")
        assert retrieval is not None
        assert generation is not None
        link_retrieval_to_generation(retrieval, generation)
        session = get_collector().get_current_session()
        assert session is not None
        return session

    session = pipeline("test query")
    trace_json = serialize_trace(session)
    payload = json.loads(trace_json)

    assert payload["retrieval_spans"][0]["linked_generation_indices"] == [1]
    assert payload["generation_spans"][0]["linked_retrieval_indices"] == [0]
    assert "analysis_report" in payload


def test_span_latencies_are_recorded(monkeypatch):
    ticks = iter([1.0, 1.25, 3.0, 3.5])

    monkeypatch.setattr("ragpeek.session.time.perf_counter", lambda: next(ticks))

    @trace
    def pipeline() -> object:
        log_retrieval(query="test", chunks=["chunk"], scores=[0.8])
        log_generation(prompt="prompt", response="response", model="test")
        return get_collector().get_current_session()

    session = pipeline()

    assert session is not None
    assert session.retrieval_spans[0].latency_ms == pytest.approx(250.0)
    assert session.generation_spans[0].latency_ms == pytest.approx(1750.0)


def test_multiple_retrieval_logs_link_to_one_generation():
    @trace(semantic=False)
    def pipeline(query: str) -> object:
        log_retrieval(query=query, chunks=["chunk 1"], scores=[0.9])
        log_retrieval(query=query, chunks=["chunk 2"], scores=[0.8])
        log_generation(prompt="prompt", response="response", model="test")
        return get_collector().get_current_session()

    session = pipeline("test query")

    assert session is not None
    assert session.generation_spans[0].linked_retrieval_indices == [0, 1]
    assert session.retrieval_spans[0].linked_generation_indices == [2]
    assert session.retrieval_spans[1].linked_generation_indices == [2]
    assert session.analysis_report["context_reports"][0]["multi_retrieval"] is True
    assert session.analysis_report["trace_mode"] == "non-semantic"
