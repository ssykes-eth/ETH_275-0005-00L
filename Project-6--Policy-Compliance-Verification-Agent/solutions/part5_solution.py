"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
from agentic.json_utils import extract_json
from agentic.models import Solution
from agentic.prompts import SOLUTION_SYSTEM_PROMPT, build_solution_prompt


def propose_solution(self, action, problem, policies):
    """Propose a fix for one `problem` in an action based on `policies`.
    Arguments:
        action (Action): The action to propose a solution for.
        problem (Problem): The specific problem to address.
        policies (List[RetrievedChunk]): The retrieved policy chunks to ground the solution in.
    Returns:
        A Solution object containing the proposed fix, corrected value, supporting sources, and explanation.
    """

    # 🎯 construct the prompt for the solution subagent
    prompt = build_solution_prompt(action, problem, policies)
    raw = self.llm.complete(prompt, system_prompt=SOLUTION_SYSTEM_PROMPT)
    # 🎯 parse the raw LLM output as JSON
    data = extract_json(raw)
    # Extract the first problematic field (if any) and propose a fix for it
    target_field = problem.field if problem.field else None
    return Solution(
        problem_id=problem.problem_id,
        # 🎯 the field to fix
        field=target_field,            
        # 🎯 the proposed fix text
        proposed_fix=data.get("proposed_fix", ""),     
        # 🎯 the corrected value (None if not applicable)
        corrected_value=data.get("corrected_value", ""),  
        # 🎯 the policy source filename
        policy_source=data.get("policy_source", ""),    
        # 🎯 the chunk id of the supporting excerpt (-1 if not applicable)
        chunk_id=data.get("chunk_id", -1),         
        # 🎯 explanation of the fix ("" if not provided)
        explanation=data.get("explanation", ""),      
    )
