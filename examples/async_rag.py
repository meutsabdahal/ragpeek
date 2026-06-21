from __future__ import annotations
import asyncio
import httpx
import chromadb

from ragpeek import trace, log_retrieval, log_generation
from ragpeek.config import TracerConfig

# Corpus setup — same tiny in-memory ChromaDB as simple_rag.py
_client = chromadb.Client()
# Use cosine space so distance ∈ [0, 2] and similarity = 1 - distance is exact.
_collection = _client.create_collection(
    "demo_async", metadata={"hnsw:space": "cosine"}
)

_collection.add(
    documents=[
        "NEPSE reached its all-time high of 3,198.60 points on August 12, 2021.",
        "The Nepal Stock Exchange index saw significant volatility throughout 2023.",
        "Trading volumes on NEPSE increased by 34% in the third quarter of 2023.",
        "Market capitalisation of NEPSE stood at NPR 3.2 trillion in December 2023.",
        "Foreign investment in Nepal's capital markets remains restricted by regulation.",
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
)


# Async retriever
async def async_retrieve(query: str, k: int = 3) -> tuple[list[str], list[float]]:
    """
    Wraps ChromaDB's sync .query() in an executor so it doesn't block
    the event loop. Returns (chunks, scores).
    """
    loop = asyncio.get_event_loop()

    def _query():
        return _collection.query(query_texts=[query], n_results=k)

    results = await loop.run_in_executor(None, _query)

    chunks = results["documents"][0]
    # Cosine distance ∈ [0, 2] → similarity = 1 - distance (exact for this metric).
    distances = results["distances"][0]
    scores = [1.0 - d for d in distances]

    return chunks, scores


# Async LLM call (Ollama)
async def async_generate(prompt: str, model: str = "llama3.2") -> str:
    """
    Non-blocking Ollama call using httpx.AsyncClient.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]


# Instrumented async pipeline
@trace
async def answer_question(query: str) -> str:
    """
    Fully async RAG pipeline. @trace detects the async function at decoration
    time and wraps it correctly — the ContextVar propagates through every
    await point so log_retrieval and log_generation always find the right session.
    """
    # async retrieval
    chunks, scores = await async_retrieve(query, k=3)
    log_retrieval(query=query, chunks=chunks, scores=scores, k_requested=3)

    # build prompt
    context = "\n\n".join(chunks)
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    # async generation
    response = await async_generate(prompt)
    log_generation(prompt=prompt, response=response, model="llama3.2")

    return response


# ---------------------------------------------------------------------------
# Run multiple queries concurrently to show ContextVar isolation
#
# This is the key demonstration: when two @trace-decorated coroutines run
# concurrently via asyncio.gather, each has its own ContextVar copy.
# log_retrieval / log_generation in coroutine A never accidentally write
# to coroutine B's session — they are fully isolated.
# ---------------------------------------------------------------------------


async def run_concurrent_queries():
    queries = [
        "What was NEPSE's highest index value?",
        "How did trading volumes change in 2023?",
    ]

    print("\nRunning two queries concurrently — each trace is isolated:\n")

    # asyncio.gather propagates ContextVar correctly per task
    results = await asyncio.gather(*[answer_question(q) for q in queries])

    for query, result in zip(queries, results):
        print(f"Q: {query}")
        print(f"A: {result[:120]}{'...' if len(result) > 120 else ''}\n")


# Custom config example
config = TracerConfig(
    min_score_threshold=0.6,  # stricter than default 0.5
    semantic=True,
    show_prompt=False,  # hide the full prompt in terminal output
)


@trace(config=config, output="async_report.html")
async def answer_with_config(query: str) -> str:
    """Same pipeline, custom config, saves HTML report to async_report.html."""
    chunks, scores = await async_retrieve(query, k=3)
    log_retrieval(query=query, chunks=chunks, scores=scores, k_requested=3)

    context = "\n\n".join(chunks)
    prompt = (
        f"Answer using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = await async_generate(prompt)
    log_generation(prompt=prompt, response=response, model="llama3.2")

    return response


# Entry point
async def main():
    print("=" * 60)
    print("Example 1: single async query")
    print("=" * 60)
    await answer_question("What was NEPSE's highest index value?")

    print("=" * 60)
    print("Example 2: two concurrent queries (ContextVar isolation)")
    print("=" * 60)
    await run_concurrent_queries()

    print("=" * 60)
    print("Example 3: custom config + HTML report saved to async_report.html")
    print("=" * 60)
    await answer_with_config("How did trading volumes change in 2023?")


if __name__ == "__main__":
    asyncio.run(main())
