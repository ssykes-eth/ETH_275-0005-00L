"""The verifier pipeline — the orchestrator (provided, the WE7 analogue of RAGCore).

This ties the six steps together into one ``verify(action)`` call. You don't implement
this class; you implement the six steps it calls (Parts 1–6). It is the place to *see*
the whole agentic workflow at a glance:

    validate -> build context -> retrieve policies (RAG tool)
             -> verify (agent) -> propose solutions (subagents) -> display (UI tools)

The pipeline resolves each step through the module/class attribute at call-time, so a
student implementation monkey-patched onto those targets (via ``solutions.apply()``)
flows through transparently — exactly the WE6 pattern.
"""

from __future__ import annotations

from pathlib import Path

from agentic import action_validation, context_builder, display_agent
from agentic.display_agent import DirectiveCollector
from agentic.models import Action, VerificationResult, Verdict
from agentic.policy_tool import PolicyRetrievalTool
from agentic.solution_agent import SolutionAgent
from agentic.verifier_agent import VerifierAgent
from rag.rag_core import RAGCore
from rag.retrieval_core import SearchType

DATA_DIR = Path(__file__).resolve().parent.parent / "rag" / "data"


class ActionValidationError(ValueError):
    """Raised when an action fails the structural checks of Part 1."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class VerifierPipeline:
    """Observe an action, verify it against policy, and decide the UI feedback."""

    def __init__(
        self,
        embedder,
        llm,
        db=None,
        db_path: str | None = None,
        collection_name: str = "documents",
        top_k: int = 5,
        search_type: SearchType = SearchType.HYBRID,
        auto_ingest: bool = True,
    ):
        # The vendored WE5 RAG pipeline is our retrieval *tool*.
        self.rag = RAGCore(
            embedder, llm, db=db, db_path=db_path, collection_name=collection_name
        )
        # Make sure the policy corpus is loaded (hydrate-on-init means a populated
        # store is already searchable; otherwise ingest the bundled policy docs once).
        if auto_ingest and not self.rag.retrieval_core.chunks:
            self.rag.ingest_path(DATA_DIR)

        self.policy_tool = PolicyRetrievalTool(self.rag, top_k=top_k, search_type=search_type)
        self.verifier = VerifierAgent(llm)
        self.solver = SolutionAgent(llm)

    def verify(self, action: Action) -> VerificationResult:
        raise NotImplementedError("Implement this in the notebook before use.")