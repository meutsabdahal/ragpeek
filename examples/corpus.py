"""Shared corpus loading + chunking for the examples.

Real RAG doesn't index one-sentence "documents" — it ingests long documents and
splits them into overlapping passages ("chunks"), embeds those, and retrieves them.
This module loads the multi-paragraph .txt files in examples/data/, chunks them with
overlap, and builds an in-memory ChromaDB collection the example pipelines query.

Chunking and retrieval live here in the examples on purpose: ragpeek instruments a
pipeline (it traces what your retriever and model did) — it is not a RAG framework
and does not do chunking or retrieval itself.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).parent / "data"


def chunk_text(text: str, *, size: int = 60, overlap: int = 15) -> list[str]:
    """Split text into overlapping word windows — a minimal text splitter.

    `size` and `overlap` are measured in words. Real splitters often respect
    sentence or token boundaries, but a fixed window with overlap is the
    canonical illustration of the "chunk size + overlap" tradeoff.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break  # this window already reached the end
    return chunks


def load_chunks() -> tuple[list[str], list[dict], list[str]]:
    """Load every .txt doc in examples/data/, chunk it, and return
    (chunks, metadatas, ids) ready for ChromaDB.
    """
    chunks: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        for index, chunk in enumerate(chunk_text(path.read_text())):
            chunks.append(chunk)
            metadatas.append({"source": path.name, "chunk": index})
            ids.append(f"{path.stem}-{index}")
    return chunks, metadatas, ids


def build_collection(name: str = "demo"):
    """Build an in-memory, cosine-space ChromaDB collection from the corpus.

    Cosine space keeps distance in [0, 2], so similarity = 1 - distance is exact
    when you log scores to ragpeek.
    """
    client = chromadb.Client()
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    chunks, metadatas, ids = load_chunks()
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return collection
