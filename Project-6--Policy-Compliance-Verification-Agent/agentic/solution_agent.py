"""Part 5 — The solution subagent.

For each problem the verifier found, the pipeline spawns a focused **subagent** whose
only job is to propose one concrete fix for that one problem. This is the "separation
of detection, reasoning, and correction" the project is about: the verifier *detects*,
the solution subagent *repairs*. Each subagent call is small and scoped, and its output
is grounded — it must cite the policy sources that support the fix.
"""

from __future__ import annotations

from agentic.json_utils import extract_json
from agentic.models import Action, Problem, RetrievedChunk, Solution
from agentic.prompts import SOLUTION_SYSTEM_PROMPT, build_solution_prompt


class SolutionAgent:
    """Wraps an LLM client to propose a concrete fix for a single problem."""

    def __init__(self, llm):
        self.llm = llm

    def propose_solution(
        self, action: Action, problem: Problem, policies: list[RetrievedChunk]
    ) -> Solution:
        raise NotImplementedError("This method is implemented in the solution notebook.")