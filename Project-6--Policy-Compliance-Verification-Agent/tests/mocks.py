"""Lightweight test doubles so tests run offline (no API keys, no network).

``MockEmbedder`` is the WE6 double (a fixed vector, deterministic similarity). The
two LLM doubles return *canned JSON*: the agents in WE7 expect structured output, so
the mocks let tests assert two things deterministically — that the student parses the
output correctly, and that the grounding (action + policy text) reached the prompt.
"""

from __future__ import annotations

import json


class MockEmbedder:
    """Returns a fixed vector so cosine similarity is fully deterministic."""

    def __init__(self, vector: list[float] | None = None):
        self.vector = vector if vector is not None else [1.0, 0.0, 0.0]

    def embed(self, text: str) -> list[float]:
        return list(self.vector)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vector) for _ in texts]


class MockLLM:
    """Returns one canned reply for every call; records what it was sent.

    ``reply`` may be a dict/list (returned as a JSON string) or a raw string.
    Use this to drive a *single* agent in isolation (verifier OR solution).
    """

    def __init__(self, reply):
        self.reply = reply
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        self.prompts: list[str] = []

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.prompts.append(prompt)
        if isinstance(self.reply, (dict, list)):
            return json.dumps(self.reply)
        return self.reply


class ScriptedLLM:
    """Routes by the calling agent (via its system prompt) so one client can drive the
    whole pipeline offline: the verifier gets ``verdict``, the solution subagent gets
    ``solution``. Records every prompt for grounding assertions."""

    def __init__(self, verdict: dict, solution: dict):
        self.verdict = verdict
        self.solution = solution
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None
        self.prompts: list[str] = []

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.prompts.append(prompt)
        # SOLUTION_SYSTEM_PROMPT identifies itself as a "remediation agent".
        is_solution = bool(system_prompt) and "remediation" in system_prompt
        return json.dumps(self.solution if is_solution else self.verdict)
