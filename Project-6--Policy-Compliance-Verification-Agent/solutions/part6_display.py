"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations


def run_display_agent(result, ui):
    """Translate a VerificationResult into controlled UI tool calls (returns None)."""

    verdict = result.verdict

    # 🎯 Case 1: The action is valid and complies with company policy
    if verdict.is_valid:
        ui.mark_ok(verdict.summary or "This action complies with company policy.")
        return

    # 🎯 Case 2: The action is not valid and violates company policy
    # Extract the highest severity level from the problems, defaulting to "medium" if none are present
    # Ranking is done using the `rank` dictionary, where "low" < "medium" < "high"
    rank = {"low": 0, "medium": 1, "high": 2}
    top = max((p.severity for p in verdict.problems), key=lambda s: rank.get(s, 1), default="medium")
    ui.warn(verdict.summary or "This action may violate company policy.", severity=top)

    # 🎯 highlight each bad field with its explanation as message
    for problem in verdict.problems:
        ui.highlight_field(problem.field, problem.explanation)
    # 🎯 suggest a correction for each solution proposal
    for solution in result.solutions:
        ui.suggest_correction(solution.field, solution.corrected_value, solution.proposed_fix)

    # Extract all citations
    citation_problems = {
        (p.field, p.policy_source, p.chunk_id) for p in verdict.problems if p.field
    }
    citation_solutions = {
        (s.field, s.policy_source, s.chunk_id) for s in result.solutions if s.field
    }
    all_citations = citation_problems.union(citation_solutions)
    all_chunks = { (p.source, p.chunk_id): p.text for p in result.policies }

    # 🎯 Attach citations to the UI for each unique (field, source, chunk_id) combination
    for field, source, chunk_id in all_citations:
        snippet = all_chunks.get((source, chunk_id), "")  # get the text of the chunk from chunk number `chunk_id` in the document `source` ("" if not found)
        ui.attach_citation(field, source, snippet)
