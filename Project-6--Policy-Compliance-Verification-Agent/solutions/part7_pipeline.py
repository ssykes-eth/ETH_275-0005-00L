"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
from agentic import action_validation, context_builder, display_agent
from agentic.display_agent import DirectiveCollector
from agentic.models import Action, VerificationResult, Verdict
from agentic.pipeline import ActionValidationError


def verify(self, action):
    """Run the full agentic workflow for one action.
    Arguments:
        action (Action): The action to verify.
    Returns:
        A VerificationResult object containing the verdict, solutions, and UI actions.
    Exceptions:
        ActionValidationError: If the action does not pass the validation step.
    """

    # 🎯 Part 1 — validate action
    errors = action_validation.validate_action(action)
    if errors:
        raise ActionValidationError(errors)

    # 🎯 Part 2 — build the verification context.
    context = context_builder.build_context(action)

    # 🎯 Part 3 — retrieve grounding policies via the RAG tool.
    policies = self.policy_tool.retrieve_policies(context)

    # 🎯 Part 4 — the verifier agent reasons over action + policies.
    verdict: Verdict = self.verifier.verify_action(action, policies)

    result = VerificationResult(
        action=action, verdict=verdict, solutions=[], ui_actions=[], policies=policies
    )

    # 🎯 Part 5 — one solution subagent per detected problem.
    for problem in verdict.problems:
        result.solutions.append(self.solver.propose_solution(action, problem, policies))

    # Part 6 — the display agent acts on the UI through tools.
    ui = DirectiveCollector()
    display_agent.run_display_agent(result, ui)
    result.ui_actions = ui.actions

    return result
