"""Command-line entry point for ragpeek.

    ragpeek demo            # render a diagnostic trace on built-in sample data
    ragpeek trace.json      # render + diagnose a trace you saved earlier
    ragpeek demo --html out.html

ragpeek instruments a *live* pipeline through the @trace decorator, so the CLI
can't trace your own code without that instrumentation. These commands let you
see the tool work and view captured traces with a single command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragpeek.config import TracerConfig

# A small built-in corpus with a top-heavy score distribution and a hedging
# answer, so `ragpeek demo` surfaces real signals: low-relevance padding, sharp
# rank-1 precision, and hedging language.
_DEMO_QUERY = "Which is the largest planet in the Solar System?"
_DEMO_CORPUS = [
    ("Jupiter is the largest planet in the Solar System, more massive than all the others combined.", 0.89),
    ("Saturn is the second-largest planet and is best known for its prominent ring system.", 0.55),
    ("Mars, the red planet, hosts Olympus Mons, the tallest volcano in the Solar System.", 0.21),
    ("Venus is the hottest planet, with surface temperatures around 465 degrees Celsius.", 0.18),
    ("Mercury is the smallest planet and the closest to the Sun.", 0.12),
]
_DEMO_RESPONSE = (
    "I believe Jupiter is generally the largest planet, though this may vary "
    "depending on how you measure it."
)


def _run_demo(*, semantic: bool, html: str | None) -> int:
    # Use the real decorator pipeline so the demo is authentic: @trace runs the
    # analyzers and renders the trace itself.
    from ragpeek import log_generation, log_retrieval, trace

    @trace(semantic=semantic, output=html)
    def answer(query: str) -> str:
        log_retrieval(
            query=query,
            chunks=[chunk for chunk, _ in _DEMO_CORPUS],
            scores=[score for _, score in _DEMO_CORPUS],
            k_requested=len(_DEMO_CORPUS),
        )
        context = "\n\n".join(chunk for chunk, _ in _DEMO_CORPUS)
        prompt = (
            f"Answer using only the context below.\n\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        log_generation(prompt=prompt, response=_DEMO_RESPONSE, model="demo-llm")
        return _DEMO_RESPONSE

    answer(_DEMO_QUERY)
    if html:
        print(f"HTML report written to {html}")
    return 0


def _view(path: str, *, html: str | None) -> int:
    from ragpeek.renderers.terminal import render_session
    from ragpeek.serialization import deserialize_trace

    file = Path(path)
    if not file.is_file():
        print(f"ragpeek: no such trace file: {path}", file=sys.stderr)
        return 2
    try:
        session = deserialize_trace(file.read_text())
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"ragpeek: could not read trace '{path}': {exc}", file=sys.stderr)
        return 2

    trace_mode = (session.analysis_report or {}).get("trace_mode")
    render_session(session, TracerConfig(semantic=trace_mode != "non-semantic"))

    if html:
        from ragpeek.renderers.html import render_html

        Path(html).write_text(render_html(session))
        print(f"HTML report written to {html}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragpeek",
        description="Inspect and diagnose RAG pipeline traces.",
    )
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser(
        "demo", help="render a diagnostic trace on built-in sample data"
    )
    demo.add_argument(
        "--semantic",
        action="store_true",
        help="enable embedding-based context analysis (downloads a model on first run)",
    )
    demo.add_argument("--html", metavar="PATH", help="also write an HTML report to PATH")

    view = sub.add_parser("view", help="render a saved trace .json file")
    view.add_argument("path", help="path to a trace .json file")
    view.add_argument("--html", metavar="PATH", help="also write an HTML report to PATH")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow the bare form `ragpeek <trace.json>` (no `view` subcommand).
    if argv and argv[0] not in {"demo", "view"} and not argv[0].startswith("-"):
        argv = ["view", *argv]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        return _run_demo(semantic=args.semantic, html=args.html)
    if args.command == "view":
        return _view(args.path, html=args.html)

    parser.print_help()
    return 0
