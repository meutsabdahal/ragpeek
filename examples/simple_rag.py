import chromadb
import httpx
from ragpeek import trace, log_retrieval, log_generation

# --- setup a tiny in-memory ChromaDB corpus ---
client = chromadb.Client()
# Use cosine space so distance ∈ [0, 2] and similarity = 1 - distance is exact.
collection = client.create_collection("demo", metadata={"hnsw:space": "cosine"})

collection.add(
    documents=[
        "NEPSE reached its all-time high of 3,198.60 points on August 12, 2021.",
        "The Nepal Stock Exchange index saw significant volatility throughout 2023.",
        "Trading volumes on NEPSE increased by 34% in the third quarter of 2023.",
        "Market capitalisation of NEPSE stood at NPR 3.2 trillion in December 2023.",
        "Foreign investment in Nepal's capital markets remains restricted by regulation.",
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
)


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
    answer_question("What was NEPSE's highest index value?")
