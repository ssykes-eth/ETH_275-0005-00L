"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
from agentic.action_types import get_action_type, policy_source_for
from agentic.models import VerificationContext


def build_context(action):
    """Turn an Action into a VerificationContext for retrieval.
    Arguments:
        action (Action): An Action object to build context for.
    Returns:
        A VerificationContext object containing the query, metadata filter, summary, and action ID.
    """

    # 🎯 details about the action type
    spec = get_action_type(action.action_type) 
    # name of the action type
    label = spec["label"] if spec else action.action_type 
    field_str = ", ".join(f"{k}: {v}" for k, v in action.fields.items())
    summary = f"{label} — {field_str}" if field_str else label
    # 🎯 a string that asks if the action named `label` is allowed under company policy, with details `field_str`
    query = f"Is this {label} action allowed under company policy? Details: {field_str}" 
    # 🎯 policy source for the action type
    source = policy_source_for(action.action_type)
    metadata_filter = {"source": source} if source else None
    return VerificationContext(query=query, metadata_filter=metadata_filter,
                               summary=summary, action_id=action.id)
