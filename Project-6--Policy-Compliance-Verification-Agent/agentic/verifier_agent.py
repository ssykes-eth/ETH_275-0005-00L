"""Part 4 — The agent verifier.

The verifier receives the structured action and the retrieved policies and decides
whether the action is compliant. It is the first place the LLM *reasons* — but its
output is **structured**, not prose: a ``Verdict`` with a list of ``Problem`` objects.
Structured output is what makes the rest of the pipeline (subagents, UI) possible, and
forcing every problem to cite policy sources is what makes the decision auditable.
"""

from __future__ import annotations

from agentic.json_utils import extract_json
from agentic.models import Action, Problem, RetrievedChunk, Verdict
from agentic.prompts import VERIFIER_SYSTEM_PROMPT, build_verifier_prompt


class VerifierAgent:
    """Wraps an LLM client to produce a structured compliance verdict."""

    def __init__(self, llm):
        # `llm` is any client with `.complete(prompt, system_prompt=...) -> str`
        # (the WE6 LLMClient, or a mock in tests).
        self.llm = llm

    def verify_action(self, action: Action, policies: list[RetrievedChunk]) -> Verdict:
        raise NotImplementedError("This method is implemented in the solution notebook.")