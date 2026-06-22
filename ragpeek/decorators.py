from __future__ import annotations
import asyncio
import functools
from typing import Optional, Callable

from ragpeek.collector import get_collector
from ragpeek.config import TracerConfig


def trace(
    func: Optional[Callable] = None,
    *,
    config: Optional[TracerConfig] = None,
    output: Optional[str] = None,
    semantic: bool = True,
    render: bool = True,
):
    """
    Decorator that instruments a RAG pipeline function.

    Can be used as:
        @trace
        def answer(query): ...

        @trace(output="report.html")
        def answer(query): ...

        @trace(config=TracerConfig(min_score_threshold=0.6))
        def answer(query): ...
    """
    # handle both @trace and @trace(...) usage
    if func is None:
        # called with arguments: @trace(output="report.html")
        # return a decorator
        def decorator(f):
            return _make_wrapper(
                f, config=config, output=output, semantic=semantic, render=render
            )

        return decorator

    # called without arguments: @trace
    return _make_wrapper(
        func, config=config, output=output, semantic=semantic, render=render
    )


def _make_wrapper(func: Callable, *, config, output, semantic, render) -> Callable:
    """Returns sync or async wrapper depending on the function type."""
    if asyncio.iscoroutinefunction(func):
        return _async_wrapper(
            func, config=config, output=output, semantic=semantic, render=render
        )
    return _sync_wrapper(
        func, config=config, output=output, semantic=semantic, render=render
    )


def _sync_wrapper(func, *, config, output, semantic, render):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # extract query — first positional arg or 'query' kwarg
        query = _extract_query(func, args, kwargs)

        collector = get_collector()
        session = collector.start_session(query=query)

        try:
            result = func(*args, **kwargs)
        finally:
            session = collector.end_session(session.session_id)
            if session:
                _post_process(
                    session,
                    config=config,
                    output=output,
                    semantic=semantic,
                    render=render,
                )

        return result

    return wrapper


def _async_wrapper(func, *, config, output, semantic, render):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        query = _extract_query(func, args, kwargs)

        collector = get_collector()
        session = collector.start_session(query=query)

        try:
            result = await func(*args, **kwargs)
        finally:
            session = collector.end_session(session.session_id)
            if session:
                # run analyzers in thread pool — sentence-transformers is sync
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    functools.partial(
                        _post_process,
                        config=config,
                        output=output,
                        semantic=semantic,
                        render=render,
                    ),
                    session,
                )

        return result

    return wrapper


def _extract_query(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Tries to find the query string from the function's arguments.
    Checks 'query' kwarg first, then first positional arg.
    Falls back to empty string if nothing found.
    """
    if "query" in kwargs:
        return str(kwargs["query"])
    if args:
        return str(args[0])
    return ""


def _post_process(session, *, config, output, semantic, render) -> None:
    """Run analyzers then render. Called after the traced function returns."""
    from ragpeek.analyzers import run_all_analyzers
    from ragpeek.renderers.terminal import render_session
    from ragpeek.renderers.html import render_html

    cfg = config or TracerConfig(semantic=semantic)
    run_all_analyzers(session, config=cfg)

    if render:
        render_session(session, config=cfg)

    if output and render:
        html = render_html(session)
        with open(output, "w") as f:
            f.write(html)
