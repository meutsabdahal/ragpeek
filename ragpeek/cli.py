"""Command-line entry point for ragpeek.

    ragpeek demo                    # ask a question, get a real diagnostic trace
    ragpeek demo "How hot is Venus?"
    ragpeek trace.json              # render + diagnose a trace you saved earlier

`ragpeek demo` runs a small, self-contained RAG pipeline over a built-in corpus:
it embeds your question and the documents (needs the `semantic` extra), retrieves
the most relevant passages, optionally answers with a local LLM (Ollama, if one is
running), and traces the whole thing. ragpeek itself only instruments pipelines —
the demo is just a runnable example of that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragpeek.config import TracerConfig

# A small built-in corpus the demo retrieves over. Real cosine similarity decides
# what gets retrieved, so the trace reflects the actual question.
BUILT_IN_DOCS = [
    "Jupiter is the largest planet in the Solar System, more massive than all the others combined.",
    "Saturn is the second-largest planet and is famous for its bright, extensive ring system.",
    "Mars, the red planet, hosts Olympus Mons, the tallest volcano in the Solar System.",
    "Venus is the hottest planet, with surface temperatures around 465 degrees Celsius.",
    "Mercury is the smallest planet and the closest to the Sun.",
    "Neptune is the most distant planet from the Sun and has the strongest winds in the Solar System.",
    "Earth is the only planet known to support life, with liquid water across most of its surface.",
    "Saturn's moon Titan is the only moon with a thick atmosphere and lakes of liquid methane.",
]
_DEFAULT_QUESTION = "Which is the largest planet in the Solar System?"
_OLLAMA_URL = "http://localhost:11434/api/generate"


def _retrieve(question: str, k: int) -> tuple[list[str], list[float]]:
    """Embed the question + built-in docs and return the top-k by cosine.

    Raises RuntimeError (from _get_semantic_resources) if the semantic extra is
    not installed.
    """
    from ragpeek.analyzers.context import _get_semantic_resources

    model, sk_cosine = _get_semantic_resources()
    embeddings = model.encode([question, *BUILT_IN_DOCS], normalize_embeddings=True)
    query_emb, doc_embs = embeddings[0], embeddings[1:]
    scored = [
        (doc, float(sk_cosine([query_emb], [doc_emb])[0][0]))
        for doc, doc_emb in zip(BUILT_IN_DOCS, doc_embs)
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:k]
    return [doc for doc, _ in top], [score for _, score in top]


def _ollama_generate(prompt: str, *, model: str, timeout: float = 60.0) -> str | None:
    """Best-effort generation via a local Ollama server. Returns None if it's
    unreachable, so the demo degrades to retrieval-only."""
    import urllib.error
    import urllib.request

    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(
        _OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read()).get("response")
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _run_demo(question: str, *, k: int, model: str, html: str | None) -> int:
    from ragpeek import log_generation, log_retrieval, trace

    try:
        chunks, scores = _retrieve(question, k)
    except RuntimeError:
        print(
            'ragpeek demo needs the semantic extra:\n'
            '    pip install "ragpeek[semantic]"',
            file=sys.stderr,
        )
        return 2

    generated = {"ok": False}

    @trace(semantic=True, output=html)
    def answer(query: str) -> str:
        log_retrieval(query=query, chunks=chunks, scores=scores, k_requested=k)
        context = "\n\n".join(chunks)
        prompt = (
            f"Answer using only the context below.\n\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        response = _ollama_generate(prompt, model=model)
        if response is None:
            return ""
        generated["ok"] = True
        log_generation(prompt=prompt, response=response, model=model)
        return response

    answer(question)

    if not generated["ok"]:
        print(
            f"(no LLM reached at {_OLLAMA_URL} — showing retrieval only; "
            f"start Ollama with the '{model}' model to generate an answer)",
            file=sys.stderr,
        )
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


def _resolve_question(arg: str | None) -> str:
    if arg:
        return arg
    if sys.stdin.isatty():
        try:
            entered = input("Question> ").strip()
        except EOFError:
            entered = ""
        if entered:
            return entered
    return _DEFAULT_QUESTION


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragpeek",
        description="Inspect and diagnose RAG pipeline traces.",
    )
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser(
        "demo", help="ask a question and render its diagnostic trace"
    )
    demo.add_argument(
        "question", nargs="?", help="question to ask (prompts if omitted)"
    )
    demo.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=4,
        metavar="N",
        help="number of chunks to retrieve (default: 4)",
    )
    demo.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model used for generation (default: llama3.2)",
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
        question = _resolve_question(args.question)
        return _run_demo(question, k=args.top_k, model=args.model, html=args.html)
    if args.command == "view":
        return _view(args.path, html=args.html)

    parser.print_help()
    return 0
