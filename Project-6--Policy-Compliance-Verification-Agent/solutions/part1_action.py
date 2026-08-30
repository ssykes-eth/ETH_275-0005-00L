"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
from agentic.action_types import ACTION_TYPES, required_fields


def validate_action(action):
    """Return a list of error strings for an action.
    Arguments:
        action (Action): An Action object to validate.
    Returns:
        A list of error strings. An empty list means the action is well-formed.
    """

    errors = []
    # 🎯 Step 1: If action.action_type is empty, raise the error
    #            "action_type is required" and return early
    if not action.action_type:
        errors.append("action_type is required for the input action object")
        return errors
    # 🎯 Step 2. If action.action_type not an allowed type, raise the error
    #            "unknown action_type ... (known: ...)" and return early
    if action.action_type not in ACTION_TYPES:
        known = ", ".join(sorted(ACTION_TYPES))
        errors.append(f"unknown action_type '{action.action_type}' (known: {known})")
        return errors
    # 🎯 Step 3: If action.fields is empty, raise the error
    #           "fields must not be empty" and return early
    if not action.fields:
        errors.append("fields must not be empty")
        return errors
    # 🎯 Step 4: For each required field for this action type, if it is missing or its
    #            value is None / an empty string, add an error "required field ... must not be empty"
    for field_name in required_fields(action.action_type):
        if field_name not in action.fields:
            errors.append(f"missing required field '{field_name}'")
            continue
        value = action.fields[field_name]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            errors.append(f"required field '{field_name}' must not be empty")
    return errors
