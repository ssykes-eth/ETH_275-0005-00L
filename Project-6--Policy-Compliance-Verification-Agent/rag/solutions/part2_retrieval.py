"""Drop your WE6 Part 2 export here (cosine / keyword / RRF / answer).

Until you do, these stubs stay marked ``@todo`` and are skipped — the vendored
reference RAG keeps working. Replace the file wholesale with your WE6 export.
"""

from __future__ import annotations

from rag.solutions import todo


@todo
def cosine_similarity(query_vec, matrix):
    raise NotImplementedError


@todo
def keyword_search(self, query, top_k=None, metadata_filter=None):
    raise NotImplementedError


@todo
def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    raise NotImplementedError


@todo
def retrieve_and_answer(self, query, search_type=None, metadata_filter=None):
    raise NotImplementedError
