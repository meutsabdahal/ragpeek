from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ragpeek.session import TraceSession, RetrievalSpan, GenerationSpan
from ragpeek.config import TracerConfig

console = Console()


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _score_signal(score: float) -> str:
    if score >= 0.7:
        return "✓"
    if score >= 0.5:
        return "⚠"
    return "✗"


def _trace_mode_label(config: TracerConfig) -> str:
    return "semantic" if config.semantic else "non-semantic"


def _fold_text(value: str) -> Text:
    return Text(value, overflow="fold", no_wrap=False)


def render_retrieval(span: RetrievalSpan, config: TracerConfig) -> None:
    console.print(
        f" [bold]Retrieval[/bold]  "
        f"[dim]{span.latency_ms:.0f}ms  "
        f"k={span.k_returned}/{span.k_requested}[/dim]"
    )

    if config.show_chunks:
        table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), expand=True)
        table.add_column(
            "Chunk",
            style="dim",
            max_width=config.max_chunk_preview,
            overflow="fold",
        )
        table.add_column("Score", justify="right", width=6)
        table.add_column("", width=4)

        for chunk, score in zip(span.chunks, span.scores):
            color = _score_color(score)
            signal = _score_signal(score)
            table.add_row(
                _fold_text(chunk),
                f"[{color}]{score:.2f}[/{color}]",
                f"[{color}]{signal}[/{color}]",
            )
        console.print(table)

    for diagnosis in span.diagnosis:
        # Concern signals warn (yellow); precision / "no signals" read positive.
        # Note: a sharp rank-1 "gap" is now a precision signal, so it is *not*
        # listed here.
        icon = (
            "⚠"
            if any(
                w in diagnosis.lower()
                for w in ["padding", "flat", "discrimination", "returned", "below"]
            )
            else "✓"
        )
        style = "yellow" if icon == "⚠" else "green"
        console.print(f" [{style}]{icon} {diagnosis}[/{style}]")
    console.print()


def render_generation(span: GenerationSpan, config: TracerConfig) -> None:
    linked_retrieval_count = len(span.linked_retrieval_indices)
    linked_label = (
        f"  multi-hop={linked_retrieval_count} retrievals"
        if linked_retrieval_count > 1
        else "  retrieval-linked" if linked_retrieval_count == 1 else ""
    )
    console.print(
        f" [bold]Generation[/bold]  "
        f"[dim]{span.latency_ms:.0f}ms  "
        f"model={span.model}{linked_label}[/dim]"
    )
    console.print(
        f" [dim]Prompt tokens: {span.prompt_tokens}  "
        f"Response tokens: {span.response_tokens}[/dim]"
    )

    if config.show_prompt:
        console.print(Panel(_fold_text(span.prompt), title="Prompt", expand=False))

    console.print()
    console.print(Panel(_fold_text(span.response), title="Response", expand=False))
    console.print()

    for diagnosis in span.diagnosis:
        icon = (
            "⚠"
            if any(
                w in diagnosis.lower()
                for w in ["hedging", "short", "low", "ignored", "failure"]
            )
            else "✓"
        )
        style = "yellow" if icon == "⚠" else "green"
        console.print(f" [{style}]{icon} {diagnosis}[/{style}]")
    console.print()


def render_session(session: TraceSession, config: TracerConfig) -> None:
    console.print(
        Panel(
            _fold_text(session.query),
            title="Query",
            subtitle=f"Trace mode: {_trace_mode_label(config)}",
            expand=False,
        )
    )
    console.print()

    for span in session.retrieval_spans:
        render_retrieval(span, config=config)

    for span in session.generation_spans:
        render_generation(span, config=config)

    console.print(f" [dim]Total latency: {session.total_latency_ms:.0f}ms[/dim]")
    console.rule(style="dim")
    console.print()
