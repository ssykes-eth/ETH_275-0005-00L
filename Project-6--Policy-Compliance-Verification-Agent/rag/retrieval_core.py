import re
from enum import Enum

import numpy as np
from rank_bm25 import BM25Okapi

from rag.clients.embedder import TextEmbedder
from rag.models import Chunk


class SearchType(str, Enum):
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


def _tokenize(text: str) -> list[str]:
    # Lowercase + split on word boundaries. No stopword removal for now;
    # BM25's IDF term already down-weights common words.
    return re.findall(r"\w+", text.lower())


class RetrievalCore:
    def __init__(
        self,
        embedder: TextEmbedder,
        chunks: list[Chunk] | None = None,
        top_k: int = 5,
        rrf_k: int = 60,
    ):
        self.embedder = embedder
        # In-memory chunk index (the "simple vector database"); ingestion populates it.
        self.chunks: list[Chunk] = chunks if chunks is not None else []
        self.top_k = top_k
        self.rrf_k = rrf_k

    def keyword_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        query_tokens = _tokenize(query)
        candidates = self._apply_metadata_filter(self.chunks, metadata_filter)
        if not candidates or not query_tokens:
            return []

        corpus = [_tokenize(chunk.text) for chunk in candidates]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)  # np.ndarray, one score per chunk

        return self._top_k(candidates, scores, top_k)

    def embedding_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        candidates = [
            chunk
            for chunk in self._apply_metadata_filter(self.chunks, metadata_filter)
            if chunk.embedding is not None
        ]
        if not candidates:
            return []

        query_vec = np.asarray(self.embedder.embed(query), dtype=np.float32)
        matrix = np.asarray([chunk.embedding for chunk in candidates], dtype=np.float32)

        scores = self._cosine_similarity(query_vec, matrix)
        return self._top_k(candidates, scores, top_k)

    @staticmethod
    def _top_k(candidates: list[Chunk], scores: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]:
        if len(candidates) == 0:
            return []
        top_k = min(top_k, len(candidates))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(candidates[i], float(scores[i])) for i in top_idx]

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        scores = matrix @ query_vec
        return np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = self.top_k if top_k is None else top_k
        keyword_results = self.keyword_search(query, top_k=top_k, metadata_filter=metadata_filter)
        embedding_results = self.embedding_search(query, top_k=top_k, metadata_filter=metadata_filter)
        return self._reciprocal_rank_fusion(
            [keyword_results, embedding_results], top_k=top_k, k=self.rrf_k
        )

    @staticmethod
    def _apply_metadata_filter(
        chunks: list[Chunk],
        metadata_filter: dict | None,
    ) -> list[Chunk]:
        """Keep only chunks whose metadata matches every key=value in the filter.

        ``None`` (the default) means "no filter — search everything".
        """
        if not metadata_filter:
            return chunks
        return [
            chunk
            for chunk in chunks
            if all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[Chunk, float]]],
        top_k: int,
        k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        # RRF: each list contributes 1 / (k + rank) per chunk (rank is 0-based).
        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        for ranked in ranked_lists:
            for rank, (chunk, _score) in enumerate(ranked):
                fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
                chunks_by_id[chunk.id] = chunk

        ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [(chunks_by_id[chunk_id], score) for chunk_id, score in ordered[:top_k]]
