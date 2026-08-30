"""Part 3 — Policy retrieval using the WE6 RAG core (RAG as a tool).

This is where WE6 becomes a *tool* inside the agent. ``PolicyRetrievalTool`` wraps a
``RAGCore`` (the vendored WE6 pipeline) and exposes one method the rest of the agent
calls: given a ``VerificationContext``, return the most relevant policy passages as
grounding evidence. We deliberately call ``retrieve`` (not ``retrieve_and_answer``) —
the agent wants the *evidence*, not a free-text answer; reasoning happens later, in the
verifier, so the decision stays auditable and citable.
"""

from __future__ import annotations

from agentic.models import RetrievedChunk, VerificationContext
from rag.rag_core import RAGCore
from rag.retrieval_core import SearchType


class PolicyRetrievalTool:
    """Adapter that turns the WE6 RAG pipeline into a policy-evidence tool."""

    def __init__(self, rag: RAGCore, top_k: int = 5, search_type: SearchType = SearchType.HYBRID):
        self.rag = rag
        self.top_k = top_k
        self.search_type = search_type

    def retrieve_policies(self, context: VerificationContext) -> list[RetrievedChunk]:
        raise NotImplementedError("Implement this in the notebook before use.")