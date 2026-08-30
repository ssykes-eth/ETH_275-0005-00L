"""Part 2 — exported from the notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
import numpy as np
from rank_bm25 import BM25Okapi
from rag_core import NO_CONTEXT_ANSWER
from retrieval_core import _tokenize


def cosine_similarity(query_vec, matrix):
    """Cosine similarity of `query_vec` against every row of `matrix`."""
    # query_vec: (D,)   matrix: (N, D)   returns: (N,)
    # The denominator ||row|| * ||query||, one value per row of the matrix.
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)              # ✏️ which axis is "per row"?
    # The numerator: the dot product of every row with the query.
    scores = matrix @ query_vec   # ✏️ every row against the query in one operation — no Python loop
    # Divide only where the denominator is non-zero: a zero vector scores 0.0, never NaN.
    return np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)    # ✏️ the safety condition


def keyword_search(self, query, top_k=None, metadata_filter=None):
    """Rank self.chunks by BM25 overlap with `query`. Return list[(Chunk, score)]."""
    top_k = self.top_k if top_k is None else top_k
    query_tokens = _tokenize(query)

    candidates = self._apply_metadata_filter(self.chunks, metadata_filter)  # ✏️ narrow the corpus by metadata *before* scoring
    if not candidates or not query_tokens:
        return []                                                           # BM25 raises on an empty corpus

    bm25 = self._bm25_index(candidates)                                     # cached for the unfiltered corpus
    scores = bm25.get_scores(query_tokens)                                  # ✏️ one BM25 score per candidate
    return self._top_k(candidates, scores, top_k)


def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    """Fuse several ranked [(Chunk, score)] lists into one. Return list[(Chunk, fused_score)]."""
    fused_scores = {}   # chunk id -> summed contribution across every list
    chunks_by_id = {}   # chunk id -> the Chunk itself, so we can return objects
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):   # rank is 0-based
            # Rank, never score: each list a chunk appears in adds 1 / (k + rank).
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1 / (k + rank)   # ✏️ this list's contribution
            chunks_by_id[chunk.id] = chunk

    ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)            # ✏️ sort by fused score, not by id
    return [(chunks_by_id[cid], score) for cid, score in ordered[:top_k]]


def hybrid_search(self, query, top_k=None, metadata_filter=None, overretrieve=None):
    """Run both retrievers wide, fuse their rankings, then cut to top_k."""
    top_k = self.top_k if top_k is None else top_k
    overretrieve = self.overretrieve if overretrieve is None else overretrieve

    # Over-retrieve: fusion can only reorder what you hand it.
    pool = top_k * overretrieve  # ✏️ candidates per branch — never fewer than top_k

    keyword_results = self.keyword_search(query, top_k=pool, metadata_filter=metadata_filter)
    # ✏️ the semantic branch gets exactly the same treatment — same pool, same filter.
    #    Filter one branch and not the other and excluded documents walk back in.
    embedding_results = self.embedding_search(query, top_k=pool, metadata_filter=metadata_filter)

    return self._reciprocal_rank_fusion(
        [keyword_results, embedding_results],
        top_k=top_k,            # ✏️ cut the wide fused pool back down to what the caller asked for
        k=self.rrf_k,
    )


def retrieve_and_answer(self, query, search_type=None, metadata_filter=None):
    """Retrieve, then answer with the LLM. Return (answer_str, list[(Chunk, score)])."""
    results = self._retrieve(query, search_type, metadata_filter)
    if not results:
        return NO_CONTEXT_ANSWER, []                                        # ✏️ abstain — and note the LLM call below is never reached

    context = "\n\n".join(chunk.text for chunk, _score in results)          # ✏️ the retrieved passages, as one context block
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = self.llm.complete(prompt, system_prompt=self.system_prompt)    # ✏️ the pipeline's system prompt
    return answer, results
