"""Part 2 — Context construction around the action.

The verifier turns the raw action into a **verification context**: a natural-language
query plus an optional metadata filter. This is the bridge to the RAG tool — the query
is what we search the policy database with, and the filter narrows retrieval to the one
policy document that governs this action type (precision, exactly like WE6's metadata
filtering). Nothing about embeddings/chunks/search is rebuilt here; we only *shape the
question* for the tool.
"""

from __future__ import annotations

from agentic.action_types import get_action_type, policy_source_for
from agentic.models import Action, VerificationContext


def build_context(action: Action) -> VerificationContext:
    raise NotImplementedError("Implement this in the notebook before use.")