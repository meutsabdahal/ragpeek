# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-22

Initial release.

### Added

- `@trace` decorator that instruments sync **and** async RAG pipelines, with
  `log_retrieval`, `log_generation`, and `link_retrieval_to_generation`. The active
  session rides a `contextvars.ContextVar`, so concurrent traces stay isolated.
- Retrieval, context, and generation analyzers that produce within-set,
  calibration-aware **signals** (low-relevance padding, sharp rank-1 precision, flat
  distribution, k mismatch, rank disagreement, low context utilisation, hedging
  language) rather than absolute verdicts.
- Terminal and HTML trace renderers; `serialize_trace` / `deserialize_trace`.
- `ragpeek` command line:
  - `ragpeek demo` — ask a question, retrieve over a built-in corpus with real
    embeddings, generate via a local Ollama server if available, and render the
    trace.
  - `ragpeek <trace.json>` — view and diagnose a saved trace.
- `TracerConfig` for tuning thresholds; `py.typed` so downstream type checkers use
  the inline type hints.
- Optional extras: `semantic` (embedding-based context analysis) and `examples`.

[Unreleased]: https://github.com/meutsabdahal/ragpeek/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/meutsabdahal/ragpeek/releases/tag/v0.1.0
