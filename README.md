# ragpeek

[![CI](https://github.com/meutsabdahal/ragpeek/actions/workflows/ci.yaml/badge.svg)](https://github.com/meutsabdahal/ragpeek/actions/workflows/ci.yaml)

**A lightweight debugger for RAG pipelines.**

When a RAG pipeline returns a bad answer, the usual move is to print the retrieved
chunks and squint at them. ragpeek replaces the squinting: wrap your pipeline in one
decorator and it shows you, per query, what was retrieved, the score of every chunk,
the exact prompt sent to the model, and a plain-English read on where things went
sideways retrieval, context ranking, or generation.

Ask a question in one command — no code (output depends on your question and the LLM):

```
$ ragpeek demo
Question> How hot is Venus?

Retrieval  k=4/4
  ✓ 0.77  Venus is the hottest planet, with surface temperatures…
  ✗ 0.39  Mercury is the smallest planet and the closest to the Sun.
  ✗ 0.34  Neptune is the most distant planet from the Sun…
  ✗ 0.31  Mars hosts Olympus Mons, the tallest volcano…

  ⚠ 3 of 4 chunks sit in the lower half of this result's score range
    (top 0.77, bottom 0.31) — possible low-relevance padding.
  ✓ Sharp rank-1 separation (0.77 vs 0.39): the retriever cleanly
    separates the top match — a precision signal.

Generation  model=llama3.2
  Venus's average surface temperature is around 465 °C…
  ✓ Generation looks healthy — no obvious signals.
```

> **Score convention:** ragpeek assumes **higher scores mean more relevant** chunks.
> If your vector store returns distances, convert them to similarities first — see
> [Works with any vector store](#works-with-any-vector-store).

---

## Install

```bash
pip install ragpeek
```

The default install is lightweight — only [`rich`](https://github.com/Textualize/rich)
at runtime. For the embedding-based context analyzer (and `ragpeek demo`, which
retrieves with real embeddings), add the `semantic` extra:

```bash
pip install "ragpeek[semantic]"
```

Requires Python 3.10+. On first semantic run, ragpeek downloads a small embedding
model (~80MB) once. `ragpeek demo` also generates an answer if a local
[Ollama](https://ollama.com) server is running; without one it shows retrieval only.

**From source:**

```bash
git clone https://github.com/meutsabdahal/ragpeek
cd ragpeek
uv sync --group dev        # create the env + install dev deps
uv run pytest tests/ -v
```

---

## Command line

Once installed, `ragpeek` is a command:

```bash
ragpeek demo                       # prompts for a question, then retrieves + answers + traces it
ragpeek demo "How hot is Venus?"   # or pass the question directly
ragpeek demo --model mistral       # choose the Ollama model (default: llama3.2)
ragpeek demo --html report.html    # also save a shareable HTML report
ragpeek path/to/trace.json         # view a saved trace (from @trace(output=...) / serialize_trace)
ragpeek                            # help
```

`ragpeek demo` retrieves over a small built-in corpus with real embeddings (needs the
`semantic` extra) and answers via a local Ollama server if one is running. Running
from a source checkout instead of an install? Prefix with `uv run`:

```bash
uv run ragpeek demo "How hot is Venus?"
uv run ragpeek demo --html report.html              # also save an HTML report
uv run ragpeek tests/fixtures/sample_session.json   # view a saved trace
uv run ragpeek                                      # help
```

---

## Instrument your pipeline

Tracing your own pipeline is two imports and two log calls — ragpeek never
monkey-patches your stack, so it works with any retriever and any model.

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

Call the function exactly as before — the trace prints automatically:

```python
answer_question("Which is the largest planet in the Solar System?")
```

Async pipelines work the same way; the active session follows your coroutines
through every `await` (it rides a `contextvars.ContextVar`), so concurrent
queries never cross-contaminate:

```python
@trace
async def answer(query: str) -> str:
    docs, scores = await retriever.asearch(query, k=5)
    log_retrieval(query=query, chunks=docs, scores=scores)

    response = await llm.acomplete(build_prompt(docs, query))
    log_generation(prompt=build_prompt(docs, query), response=response, model="llama3.2")
    return response
```

---

## Configuration

Pass a `TracerConfig` to tune thresholds, or flip decorator flags for common cases:

```python
from ragpeek import trace, TracerConfig

config = TracerConfig(
    score_gap_threshold=0.3,     # rank-1→rank-2 gap that reads as precision
    semantic=True,               # embedding-based context analysis
    show_prompt=False,           # hide the full prompt in terminal output
    # min_score_threshold=0.6,   # opt-in absolute floor — only set once you've
    #                            # calibrated a cutoff for your own embedder
)

@trace(config=config)
def answer(query: str) -> str:
    ...
```

```python
@trace(semantic=False)              # skip the embedding model (faster, no download)
@trace(output="report.html")        # save a shareable HTML report
@trace(render=False)                # don't print — just populate session.analysis_report
```

With `render=False` the analyzers still run; grab the finalized session and hand it
to downstream tooling with `serialize_trace(...)` (and `deserialize_trace(...)` to
read it back, e.g. `ragpeek trace.json`).

---

## Works with any vector store

`log_retrieval` takes similarity **scores** (higher = better). Most stores return
those directly; some return distances you convert first.

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

> **Note on scores:** ragpeek assumes higher score = more relevant. There is
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

Need a non-default retrieval→generation association? Keep the returned span objects
and pair them explicitly:

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

1. `@trace` wraps your function and opens a `TraceSession`.
2. The session id lives in a `contextvars.ContextVar`, so it propagates through both
   sync and async code without you threading anything through your call stack.
3. `log_retrieval()` and `log_generation()` read that `ContextVar` and append spans
   to the active session.
4. When your function returns, three analyzers run over the collected spans:
   - **Retrieval** — within-set score distribution, low-relevance padding, rank-1 precision, k mismatch.
   - **Context** — chunk↔response similarity and the rank-disagreement (reranking) signal.
   - **Generation** — hedging language and response-length anomalies.
5. The terminal renderer prints the trace; the HTML renderer saves a shareable report.

The embedding model runs entirely on your machine — your data never leaves it.

---

## Limitations

- **Explicit, not magic.** You call `log_retrieval` / `log_generation` yourself —
  ragpeek doesn't patch framework internals. That's three lines of instrumentation
  per pipeline, traded for working with any stack.
- **Signals, not truth.** Retrieval signals are computed *within* each result set and
  assume higher = better, but they can't know your embedder's absolute scale. Treat
  every diagnosis as a prompt to calibrate, and convert distances to similarities
  per metric (table above) before calling `log_retrieval`.

---

## Contributing

Issues and PRs welcome. If a vector-store integration doesn't work or a diagnosis
looks wrong, open an issue with a minimal reproduction.

## License

MIT
