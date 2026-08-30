"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
from agentic.json_utils import extract_json
from agentic.models import Problem, Verdict
from agentic.prompts import VERIFIER_SYSTEM_PROMPT, build_verifier_prompt


def verify_action(self, action, policies):
    """Return a Verdict for `action` grounded in `policies`.
    Arguments:
        action (Action): The action to verify.
        policies (List[RetrievedChunk]): The retrieved chunks to ground the verification in.
    Returns:
        A Verdict object containing the status (str), problems (List[Problem]), and summary (str).
    """

    # 🎯 construct the prompt for the verifier LLM
    prompt = build_verifier_prompt(action, policies)
    raw = self.llm.complete(prompt, system_prompt=VERIFIER_SYSTEM_PROMPT)

    # 🎯 parse the raw LLM output as JSON
    data = extract_json(raw)
    problems = [
       Problem(
           # 🎯 one problematic field (default: "")
           field=item.get("field", ""),
           # 🎯 the source policy filename (default: "")
           policy_source=item.get("policy_source", ""),
           # 🎯 the index of the chunk in the source document (-1 if not applicable)
           chunk_id=int(item.get("chunk_id", -1)),
           # 🎯 explanation of the why it is a violation (default: "") 
           explanation=item.get("explanation", ""),
           # 🎯 severity of the problem (default: "medium") 
           severity=item.get("severity", "medium"),
       )
       for item in data.get("problems", [])
    ]
    status = data.get("status", "problematic" if problems else "valid")
    if problems:
        status = "problematic"
    return Verdict(status=status, problems=problems, summary=data.get("summary", ""))
