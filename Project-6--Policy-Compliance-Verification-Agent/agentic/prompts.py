"""Prompt templates + formatters for the agents (provided scaffolding).

You don't write these — you *use* them inside the verifier and solution agents. The
formatters turn structured objects (the action, the retrieved policies) into the text
block the LLM sees, which is exactly how **grounding** happens: the model only ever
reasons over the policy passages the retriever found, and is told to cite them.
"""

from __future__ import annotations

from agentic.models import Action, Problem, RetrievedChunk

# --------------------------------------------------------------------------- #
# System prompts — the "role" each agent plays.
# --------------------------------------------------------------------------- #
VERIFIER_SYSTEM_PROMPT = (
    "You are a compliance verification agent. You are given a user ACTION and a set "
    "of POLICY EXCERPTS retrieved from the company policy database. Decide whether the "
    "action complies with the policies. Reason ONLY from the provided excerpts — do not "
    "invent rules. Report one problem per field at fault with the source policy filename "
    "and the chunk_id of the supporting excerpt. If there are multiple problematic fields, "
    "report each as a separate problem. "
    "Respond with a SINGLE JSON object and nothing else, "
    "using exactly this schema:\n"
    '{\n'
    '  "status": "valid" | "problematic",\n'
    '  "summary": "one sentence overall assessment",\n'
    '  "problems": [\n'
    '    {\n'
    '      "field": "name of the action field at fault",\n'
    '      "policy_source": "policy_filename.md",\n'
    '      "chunk_id": <int: the chunk id of the excerpt in the source document>,\n'
    '      "explanation": "why this violates the policy"\n'
    '      "severity": "low" | "medium" | "high",\n'
    '    }\n'
    '  ]\n'
    '}\n'
    'If the action is fully compliant, set "status":"valid" and "problems":[].'
)

SOLUTION_SYSTEM_PROMPT = (
    "You are a remediation agent. You are given a single compliance PROBLEM with an "
    "action and the POLICY EXCERPTS that may apply. Propose one concrete fix that would make "
    "the action compliant. Ground your fix in ONE excerpt and cite the source filename along "
    "with the chunk id. Respond with a SINGLE JSON object and nothing else, using this schema:\n"
    '{\n'
    '  "proposed_fix": "a short, concrete instruction to the user",\n'
    '  "corrected_value": <the corrected field value, or null if not a single value>,\n'
    '  "policy_source": "source_filename.md",\n'
    '  "chunk_id": <int: the chunk_id of the excerpt in the source document>,\n'
    '  "explanation": "why this fix resolves the problem"\n'
    '}'
)


# --------------------------------------------------------------------------- #
# Formatters — structured objects -> the text the model reads.
# --------------------------------------------------------------------------- #
def format_action(action: Action) -> str:
    """Render an action as a compact, readable block for a prompt."""
    lines = [f"action_type: {action.action_type}"]
    if action.context:
        ctx = ", ".join(f"{k}={v}" for k, v in action.context.items())
        lines.append(f"context: {ctx}")
    lines.append("fields:")
    for key, value in action.fields.items():
        lines.append(f"  - {key}: {value}")
    return "\n".join(lines)


def format_policies(policies: list[RetrievedChunk]) -> str:
    """Render retrieved policy passages as a source-tagged block."""
    if not policies:
        return "(no policy excerpts retrieved)"
    blocks = []
    for p in policies:
        blocks.append(f"**policy_source: {p.source}, chunk_id: {p.chunk_id}**\n{p.text}")
    return "\n\n".join(blocks)


def build_verifier_prompt(action: Action, policies: list[RetrievedChunk]) -> str:
    """The user-message for the verifier agent: the action + its grounding."""
    return (
        f"ACTION:\n{format_action(action)}\n\n"
        f"POLICY EXCERPTS:\n{format_policies(policies)}\n\n"
        "Assess the action against these excerpts and return the JSON verdict."
    )


def build_solution_prompt(
    action: Action, problem: Problem, policies: list[RetrievedChunk]
) -> str:
    """The user-message for a solution subagent: one problem + its grounding."""
    return (
        f"ACTION:\n{format_action(action)}\n\n"
        f"PROBLEM:\n"
        f"  field: {problem.field}\n"
        f"  explanation: {problem.explanation}\n"
        f"POLICY EXCERPTS:\n{format_policies(policies)}\n\n"
        "Propose a concrete fix and return the JSON solution."
    )
