"""Structured data models for the agentic verifier.

A reliable agentic application is built on **structured inputs and outputs** — not
free-floating strings. Every step of the pipeline hands the next one a typed object,
so each stage is testable in isolation and the whole flow stays auditable.

These are plain dataclasses (UUID ids, ``to_dict`` for the HTTP layer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


# --------------------------------------------------------------------------- #
# Input: what the user did
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """A single user action on the dashboard, as a structured object.

    Attributes:
        action_type: which kind of action, e.g. ``"expense_report"``. Drives which
            policies are relevant (see ``context_builder``).
        fields: the values the user entered, e.g. ``{"amount": 420, "category": "hotel"}``.
        context: ambient facts about the actor, e.g. ``{"role": "employee", "region": "CH"}``.
        id: a stable id so findings/solutions can refer back to the action.
    """

    action_type: str
    fields: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action_type": self.action_type,
            "fields": self.fields,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        return cls(
            action_type=data.get("action_type", ""),
            fields=dict(data.get("fields", {})),
            context=dict(data.get("context", {})),
            id=data.get("id", str(uuid4())),
        )


# --------------------------------------------------------------------------- #
# Step 2 output: the query we will run against the policy database
# --------------------------------------------------------------------------- #
@dataclass
class VerificationContext:
    """The action reshaped into something the RAG tool can search with.

    Attributes:
        query: the natural-language query handed to retrieval.
        metadata_filter: optional ``{key: value}`` filter to narrow retrieval to the
            relevant policy document (e.g. ``{"source": "expense_reimbursement_policy.md"}``).
        summary: a short human-readable description of the action (used in prompts).
        action_id: back-reference to the originating action.
    """

    query: str
    metadata_filter: dict | None
    summary: str
    action_id: str

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "metadata_filter": self.metadata_filter,
            "summary": self.summary,
            "action_id": self.action_id,
        }


# --------------------------------------------------------------------------- #
# Step 3 output: grounding evidence
# --------------------------------------------------------------------------- #
@dataclass
class RetrievedChunk:
    """One retrieved policy passage — the grounding evidence for a decision.

    A thin DTO over a retrieval result ``(Chunk, score)``, keeping only what the
    agents and the UI need. Keeping the source makes every decision *citable*.
    """

    source: str
    chunk_id: int
    document_title: str
    text: str
    score: float
    

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "document_title": self.document_title,
            "text": self.text,
            "score": self.score,
        }


# --------------------------------------------------------------------------- #
# Step 4 output: the verifier's verdict
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    """One compliance problem the verifier found.

    Attributes:
        field: which action field is problematic (used to highlight the UI).
        policy_source: the source filename that justifies the finding.
        chunk_id: the index of the retrieved chunk in policy_source that justifies the finding.
        explanation: why the action violates the policy.
        severity: ``"low"`` | ``"medium"`` | ``"high"``.
        problem_id: a stable id so a ``Solution`` can refer back to it.
    """

    field: str
    policy_source: str
    chunk_id: int
    explanation: str
    severity: str
    problem_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "policy_source": self.policy_source,
            "chunk_id": self.chunk_id,
            "explanation": self.explanation,
            "severity": self.severity,
            "problem_id": self.problem_id,
        }


@dataclass
class Verdict:
    """The verifier agent's structured decision about an action."""

    status: str  # "valid" | "problematic"
    problems: list[Problem] = field(default_factory=list)
    summary: str = ""

    @property
    def is_valid(self) -> bool:
        return self.status == "valid" and not self.problems

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "summary": self.summary,
            "problems": [p.to_dict() for p in self.problems],
        }


# --------------------------------------------------------------------------- #
# Step 5 output: a proposed fix for one problem
# --------------------------------------------------------------------------- #
@dataclass
class Solution:
    """A concrete fix proposed by the solution subagent for one ``Problem``."""

    problem_id: str
    field: str | None
    proposed_fix: str
    corrected_value: Any | None
    policy_source: str
    chunk_id: int
    explanation: str

    def to_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "field": self.field,
            "proposed_fix": self.proposed_fix,
            "corrected_value": self.corrected_value,
            "policy_source": self.policy_source,
            "chunk_id": self.chunk_id,
            "explanation": self.explanation,
        }


# --------------------------------------------------------------------------- #
# Step 6 output: a controlled action on the UI
# --------------------------------------------------------------------------- #
@dataclass
class UIAction:
    """One controlled, auditable instruction for the user interface.

    The display agent emits these instead of touching the DOM directly, so the set
    of things an agent can do to the UI is explicit and reviewable.

    type:
        ``"highlight"``   — flag a field as problematic.
        ``"warning"``     — show a banner / message.
        ``"citation"``    — attach a policy source (optionally to a field).
        ``"correction"``  — offer a suggested value the user can apply.
        ``"ok"``          — confirm the action is compliant.
    """

    type: str
    message: str = ""
    payload: dict = field(default_factory=dict)
    # NOTE: this attribute is named `field`; defining it rebinds the name inside this
    # class body, so it must come AFTER any `field(default_factory=...)` calls above.
    field: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "message": self.message,
            "field": self.field,
            "payload": self.payload,
        }


# --------------------------------------------------------------------------- #
# The full result returned by the pipeline
# --------------------------------------------------------------------------- #
@dataclass
class VerificationResult:
    """Everything the pipeline produced for one action — the API payload."""

    action: Action
    verdict: Verdict
    solutions: list[Solution] = field(default_factory=list)
    ui_actions: list[UIAction] = field(default_factory=list)
    policies: list[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action.to_dict(),
            "verdict": self.verdict.to_dict(),
            "solutions": [s.to_dict() for s in self.solutions],
            "ui_actions": [u.to_dict() for u in self.ui_actions],
            "policies": [p.to_dict() for p in self.policies],
        }
