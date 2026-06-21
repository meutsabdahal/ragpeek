import httpx
from corpus import build_collection
from ragpeek import trace, log_retrieval, log_generation

# Ingest the corpus: load the multi-paragraph docs in examples/data/, split them
# into overlapping chunks, and index those — see examples/corpus.py.
collection = build_collection("demo")


def _call_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Simple sync Ollama call."""
    resp = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["response"]


@trace
def answer_question(query: str) -> str:
    # retrieval
    results = collection.query(query_texts=[query], n_results=3)
    chunks = results["documents"][0]
    # Cosine distance ∈ [0, 2] → similarity = 1 - distance (exact for this metric).
    distances = results["distances"][0]
    scores = [1.0 - d for d in distances]

    log_retrieval(query=query, chunks=chunks, scores=scores, k_requested=3)

    # generation
    context = "\n\n".join(chunks)
    prompt = (
        f"Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    response = _call_ollama(prompt)
    log_generation(prompt=prompt, response=response, model="llama3.2")

    return response


if __name__ == "__main__":
    answer_question("Which is the largest planet in the Solar System?")
