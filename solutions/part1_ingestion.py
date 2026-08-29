"""Part 1 — exported from the notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
import chunking
from models import Chunk


def sliding_window(text, chunk_size=200, overlap=40):
    """Split `text` into overlapping windows of words."""
    # --- validation: a window that never advances is an infinite loop, not a bad result.
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:                       # ✏️ which overlap stops the window moving forward?
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap                     # ✏️ how far the window slides between chunks
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]    # ✏️ where does this window end?
        if not window:
            break
        chunks.append(" ".join(window))             # ✏️ the window as one space-joined string
        if start + chunk_size >= len(words):        # ✏️ this window already reached the last word -> stop,
            break                                   #    otherwise the tail chunk is emitted twice
    return chunks


def ingest_document(self, document):
    """Run load->CHUNK->EMBED->store for one Document. Return the list of stored Chunks."""
    # 1. CHUNK — split the text using this core's configured strategy.
    texts = chunking.chunk_text(
        document.text,
        chunk_size=self.chunk_size,
        overlap=self.overlap,
        strategy=self.strategy,
    )
    if not texts:
        return []

    # 2. WRAP — one Chunk per passage, each carrying the document's metadata.
    chunks = [
        Chunk(document_id=document.id, index=i, text=text,
              metadata=dict(document.metadata))                                     # ✏️ a *copy*, so chunks don't share one dict
        for i, text in enumerate(texts)
    ]

    # 3. EMBED — one batched call for the whole document, not one call per chunk.
    embeddings = self.embedder.embed_batch(texts)                                   # ✏️ what gets embedded, in chunk order
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding                                                 # ✏️ attach each vector to its chunk

    # 4. STORE
    self.db.ensure_collection(self.collection_name, vector_size=len(embeddings[0])) # ✏️ how wide is one vector?
    self.db.insert_document(self.collection_name, document, chunks)
    return chunks


def apply_metadata_filter(chunks, metadata_filter):
    """Keep only chunks whose metadata matches every key/value in the filter."""
    if not metadata_filter:
        return chunks                                                                       # ✏️ no filter: hand back the *same list object* (the BM25 cache
                                                                                            #    recognises the full corpus by identity, so don't copy it)
    return [
        chunk for chunk in chunks
        if all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())  # ✏️ what every pair must satisfy
    ]
