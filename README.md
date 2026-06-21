# ragpeek

[![CI](https://github.com/meutsabdahal/ragpeek/actions/workflows/ci.yaml/badge.svg)](https://github.com/meutsabdahal/ragpeek/actions/workflows/ci.yaml)

**A lightweight debugger for RAG pipelines.**

When a RAG pipeline gives a wrong answer, you have no idea why. `ragpeek` wraps your existing pipeline with one decorator and shows you exactly where it broke retrieval, context ranking, or generation.

> **Score convention:** `ragpeek` assumes higher scores mean more relevant chunks.
> If your vector store returns distances, convert them to similarities before logging.

```
$ python app.py

──────────────────────────────────────────────────────────────────
 Query: What was NEPSE's highest index value in 2023?
──────────────────────────────────────────────────────────────────
 Retrieval                                                   80ms
 ┌──────────────────────────────────────────┬───────┬──────┐
 │ Chunk                                    │ Score │      │
 ├──────────────────────────────────────────┼───────┼──────┤
 │ NEPSE reached its peak of 3,198...       │ 0.89  │  ✓   │
 │ The index saw significant volatility...  │ 0.82  │  ✓   │
 │ Trading volumes in Q3 2023 were...       │ 0.45  │  ⚠   │
 │ Market capitalisation stood at...        │ 0.38  │  ⚠   │
 │ Foreign investment flows into...         │ 0.25  │  ⚠   │
 └──────────────────────────────────────────┴───────┴──────┘
 ⚠ 3 of 5 chunks scored below 0.5 — low-relevance padding.
   Consider reducing k or raising min_score threshold.

 Generation                                                 200ms
 Prompt tokens: 87  │  Response tokens: 92  │  Model: llama3.2

 ✓ Generation looks healthy — no obvious issues detected.

 Total latency: 280ms
──────────────────────────────────────────────────────────────────
```

---

## Why this exists

Most RAG debugging looks like this: print retrieved chunks to stdout, read them manually, guess what went wrong. That's not debugging that's hoping.

`ragpeek` gives you a structured trace of every query: what was retrieved, similarity scores per chunk, the exact prompt sent to the model, and a plain-English diagnosis of where the pipeline is weak.

---

## Install

This project is not yet published to PyPI. Install from source (recommended):

```bash
git clone https://github.com/meutsabdahal/ragpeek
cd ragpeek
# Recommended: use the uv-native workflow (creates environment and installs dev deps)
uv sync --group dev
# Run tests or examples via uv
uv run pytest tests/ -v
```

Alternatively, install into a virtual environment with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.10+.

On first semantic run, `ragpeek` may download a small embedding model (~80MB). This is a one-time download.

---

## Quick start

**1. Add two imports and two log calls to your existing pipeline**

```python
from ragpeek import trace, log_retrieval, log_generation

@trace
def answer_question(query: str) -> str:
    docs, scores = retriever.search(query, k=5)
    log_retrieval(query=query, chunks=docs, scores=scores)

    prompt = build_prompt(docs, query)
    response = llm.generate(prompt)
    log_generation(prompt=prompt, response=response, model="llama3.2")

    return response
```

**2. Call your function exactly as before**

```python
answer_question("What was NEPSE's highest index value in 2023?")
```

The trace prints automatically. Nothing else changes.

---

## Usage

### Sync pipeline

```python
from ragpeek import trace, log_retrieval, log_generation

@trace
def answer(query: str) -> str:
    docs, scores = retriever.search(query, k=5)
    log_retrieval(query=query, chunks=docs, scores=scores)

    response = llm.complete(build_prompt(docs, query))
    log_generation(prompt=build_prompt(docs, query),
                   response=response, model="llama3.2")
    return response
```

### Async pipeline

```python
@trace
async def answer(query: str) -> str:
    docs, scores = await retriever.asearch(query, k=5)
    log_retrieval(query=query, chunks=docs, scores=scores)

    response = await llm.acomplete(build_prompt(docs, query))
    log_generation(prompt=build_prompt(docs, query),
                   response=response, model="llama3.2")
    return response
```

### Save an HTML report

```python
@trace(output="report.html")
def answer(query: str) -> str:
    ...
```

### Configure thresholds

```python
from ragpeek import trace, TracerConfig

config = TracerConfig(
    min_score_threshold=0.6,   # flag chunks below this (default: 0.5)
    score_gap_threshold=0.3,   # flag if top-2 score gap exceeds this
    semantic=True,              # embedding-based context analysis
    show_prompt=False,          # hide full prompt in terminal output
)

@trace(config=config)
def answer(query: str) -> str:
    ...
```

### Skip semantic analysis (faster, no embedding model)

```python
@trace(semantic=False)
def answer(query: str) -> str:
    ...
```

### Disable rendering for downstream tooling

```python
from ragpeek import trace, log_retrieval, log_generation, serialize_trace

@trace(render=False)
def answer(query: str) -> str:
    docs, scores = retriever.search(query, k=5)
    log_retrieval(query=query, chunks=docs, scores=scores)

    response = llm.complete(build_prompt(docs, query))
    log_generation(prompt=build_prompt(docs, query), response=response, model="llama3.2")
    return response
```

The analyzers still run and populate `session.analysis_report`; once you have the finalized session object, use `serialize_trace(...)` to hand it to downstream tools.

---

## Works with any vector store

```python
# ChromaDB
results = collection.query(query_texts=[query], n_results=5)
log_retrieval(query=query,
              chunks=results["documents"][0],
              scores=results["distances"][0])

# FAISS
distances, indices = index.search(query_embedding, k=5)
log_retrieval(query=query,
              chunks=[corpus[i] for i in indices[0]],
              scores=distances[0].tolist())

# Qdrant
results = client.search("docs", query_vector=embedding, limit=5)
log_retrieval(query=query,
              chunks=[r.payload["text"] for r in results],
              scores=[r.score for r in results])
```

> **Note on scores:** `ragpeek` assumes higher score = more relevant.
> If your vector store returns distances (lower = better), convert them
> before calling `log_retrieval`: `score = 1.0 - distance`.

### Explicit retrieval-generation pairing

If your workflow needs a non-default association, keep the returned span objects and pair them explicitly:

```python
from ragpeek import trace, log_retrieval, log_generation, link_retrieval_to_generation

@trace(render=False)
def answer(query: str) -> str:
    retrieval = log_retrieval(query=query, chunks=["chunk"], scores=[0.9])
    response = llm.complete(query)
    generation = log_generation(prompt=query, response=response, model="llama3.2")
    link_retrieval_to_generation(retrieval, generation)
    return response
```

This is useful when a generation should be tied to a specific retrieval step after the fact.

---

## What it detects

| Signal | What it means |
|---|---|
| Low retrieval scores | Retrieved chunks don't match the query well |
| High score gap | Top result dominates rest is noise |
| k mismatch | Retriever returned fewer chunks than requested |
| Lost in the middle | Most relevant chunk is not at position 0 |
| Low context utilisation | LLM response is very short relative to context |
| Hedging language | LLM may be answering from training weights, not context |

---

## How it works

1. `@trace` wraps your function and creates a `TraceSession`
2. Session ID is stored in a `contextvars.ContextVar` propagates correctly through both sync and async code without you passing anything around
3. `log_retrieval()` and `log_generation()` read the `ContextVar` and append spans to the active session
4. After your function returns, three analyzers run on the collected data:
   - **Retrieval analyzer**: score distributions, low-relevance padding, k mismatch
   - **Context analyzer**: chunk-response similarity, lost-in-middle detection
   - **Generation analyzer**: hedging language, response length anomalies
5. Terminal renderer prints the trace; HTML renderer saves a shareable report

The embedding model runs entirely locally your data never leaves your machine.

---

## Limitations

`log_retrieval` and `log_generation` must be called manually `ragpeek` does not monkey-patch framework internals. This means it works with any stack but requires three lines of instrumentation code per pipeline. This is a deliberate tradeoff: explicit over magic.

Similarity score thresholds assume higher = better relevance. Convert distances to similarities before calling `log_retrieval` if your vector store returns distances.

---

## Development setup

```bash
git clone https://github.com/meutsabdahal/ragpeek
cd ragpeek
uv sync --group dev
uv run pytest tests/ -v
```

---
## Contributing

Issues and PRs welcome. If a vector store integration doesn't work or a diagnosis is wrong, open an issue with a minimal reproduction.

---

## License

MIT