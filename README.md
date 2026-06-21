# ragpeek

[![CI](https://github.com/meutsabdahal/ragpeek/actions/workflows/ci.yaml/badge.svg)](https://github.com/meutsabdahal/ragpeek/actions/workflows/ci.yaml)

**A lightweight debugger for RAG pipelines.**

When a RAG pipeline gives a wrong answer, you have no idea why. `ragpeek` wraps your existing pipeline with one decorator and shows you exactly where it broke retrieval, context ranking, or generation.

> **Score convention:** `ragpeek` assumes higher scores mean more relevant chunks.
> If your vector store returns distances, convert them to similarities before logging.

```
$ python app.py

──────────────────────────────────────────────────────────────────
 Query: Which is the largest planet in the Solar System?
──────────────────────────────────────────────────────────────────
 Retrieval                                                   80ms
 ┌──────────────────────────────────────────┬───────┬──────┐
 │ Chunk                                    │ Score │      │
 ├──────────────────────────────────────────┼───────┼──────┤
 │ Jupiter is the largest planet in the...  │ 0.89  │  ✓   │
 │ Saturn is the second-largest planet...   │ 0.82  │  ✓   │
 │ Mars hosts Olympus Mons, the tallest...  │ 0.45  │  ⚠   │
 │ Venus is the hottest planet, around...   │ 0.38  │  ⚠   │
 │ Mercury is the smallest planet, near...  │ 0.25  │  ⚠   │
 └──────────────────────────────────────────┴───────┴──────┘
 ⚠ 3 of 5 chunks sit in the lower half of this result's score
   range (top 0.89, bottom 0.25) — possible low-relevance padding.
   Signal — calibrate to your embedder.

 Generation                                                 200ms
 Prompt tokens: 87  │  Response tokens: 92  │  Model: llama3.2

 ✓ Generation looks healthy — no obvious signals.

 Total latency: 280ms
──────────────────────────────────────────────────────────────────
```

---

## Why this exists

Most RAG debugging looks like this: print retrieved chunks to stdout, read them manually, guess what went wrong. That's not debugging that's hoping.

`ragpeek` gives you a structured trace of every query: what was retrieved, similarity scores per chunk, the exact prompt sent to the model, and a plain-English diagnosis of where the pipeline is weak.

---

## Install

```bash
pip install ragpeek
```

The default install is lightweight (only [`rich`](https://github.com/Textualize/rich) at runtime). For the embedding-based context analyzer, add the `semantic` extra:

```bash
pip install "ragpeek[semantic]"
```

Requires Python 3.10+. On first semantic run, `ragpeek` may download a small embedding model (~80MB) — a one-time download.

### From source

```bash
git clone https://github.com/meutsabdahal/ragpeek
cd ragpeek
uv sync --group dev      # create env + install dev deps
uv run pytest tests/ -v
```

---

## Try it instantly

See the full diagnostic trace on built-in sample data — no code required:

```bash
ragpeek demo                      # render the trace; add --semantic for embedding analysis
ragpeek demo --html report.html   # also save a shareable HTML report
```

Already captured a trace? Render and diagnose it with one command:

```bash
ragpeek trace.json                # view a saved trace (from @trace(output=...) or serialize_trace)
```

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
answer_question("Which is the largest planet in the Solar System?")
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
    score_gap_threshold=0.3,     # rank-1→rank-2 gap that reads as precision
    semantic=True,               # embedding-based context analysis
    show_prompt=False,           # hide full prompt in terminal output
    # min_score_threshold=0.6,   # opt-in absolute floor — only set once you've
    #                            # calibrated a cutoff for your own embedder
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
# ChromaDB (cosine space): distance ∈ [0, 2] → similarity = 1 - distance
results = collection.query(query_texts=[query], n_results=5)
log_retrieval(query=query,
              chunks=results["documents"][0],
              scores=[1.0 - d for d in results["distances"][0]])

# FAISS IndexFlatL2 with normalized vectors: similarity = 1 - d² / 2
distances, indices = index.search(query_embedding, k=5)
log_retrieval(query=query,
              chunks=[corpus[i] for i in indices[0]],
              scores=[1.0 - (d ** 2) / 2 for d in distances[0].tolist()])

# Qdrant (cosine): .score is already a similarity — use it as-is
results = client.search("docs", query_vector=embedding, limit=5)
log_retrieval(query=query,
              chunks=[r.payload["text"] for r in results],
              scores=[r.score for r in results])
```

> **Note on scores:** `ragpeek` assumes higher score = more relevant. There is
> no single distance→similarity formula — convert per metric:
>
> | Store returns | Correct conversion |
> |---|---|
> | Cosine distance (∈ [0, 2]) | `score = 1.0 - distance` (exact) |
> | L2 / Euclidean, normalized vectors | `score = 1.0 - distance ** 2 / 2` (exact) |
> | L2 / Euclidean, un-normalized | `score = 1.0 / (1.0 + distance)` (monotonic squash) |
> | Inner product / dot product | already a similarity — use as-is (negate if returned as a distance) |
>
> `score = 1.0 - distance` is **only** correct for cosine distance; using it on
> raw L2 distances silently produces wrong (often negative) similarities.

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

## What it surfaces

These are **signals to calibrate**, not verdicts. Scores are read within each
result set, so they don't assume an absolute scale — tune thresholds to your
own embedder.

| Signal | What it means |
|---|---|
| Within-set padding | Most chunks fall in the lower half of *this result's* score range (relative, not an absolute cutoff) |
| Sharp rank-1 separation | The retriever cleanly separates the top match — a **precision** signal, not noise |
| Flat distribution | Scores barely differ — the retriever can't discriminate (query too vague / chunks too broad) |
| k mismatch | Retriever returned fewer chunks than requested |
| Rank disagreement | The answer aligns with a chunk the retriever didn't rank first — a reranking signal |
| Low context utilisation | The response is semantically dissimilar to every retrieved chunk |
| Hedging language | Phrase-level signal the model may be answering from training weights, not context |

---

## How it works

1. `@trace` wraps your function and creates a `TraceSession`
2. Session ID is stored in a `contextvars.ContextVar` propagates correctly through both sync and async code without you passing anything around
3. `log_retrieval()` and `log_generation()` read the `ContextVar` and append spans to the active session
4. After your function returns, three analyzers run on the collected data:
   - **Retrieval analyzer**: within-set score distribution, low-relevance padding, rank-1 precision, k mismatch
   - **Context analyzer**: chunk-response similarity, rank-disagreement (reranking) signal
   - **Generation analyzer**: hedging language, response length anomalies
5. Terminal renderer prints the trace; HTML renderer saves a shareable report

The embedding model runs entirely locally your data never leaves your machine.

---

## Limitations

`log_retrieval` and `log_generation` must be called manually `ragpeek` does not monkey-patch framework internals. This means it works with any stack but requires three lines of instrumentation code per pipeline. This is a deliberate tradeoff: explicit over magic.

Retrieval signals are computed *within* each result set and assume higher = better relevance, but they can't know your embedder's absolute scale — treat every diagnosis as a signal to calibrate, not a verdict. Convert distances to similarities per metric (see the table above) before calling `log_retrieval`.

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