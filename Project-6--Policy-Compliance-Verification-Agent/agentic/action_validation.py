"""Part 1 — Action representation & the verification trigger.

Before any LLM is involved, an agentic system should run cheap, deterministic checks
on its structured input. A malformed action should never reach the expensive
reasoning steps. ``validate_action`` is that gate: it returns a list of human-readable
error strings (empty list == the action is well-formed), mirroring the WE6
``metadata_schema.validate_payload`` contract.
"""

from __future__ import annotations

from agentic.action_types import ACTION_TYPES, required_fields
from agentic.models import Action


def validate_action(action: Action) -> list[str]:
    raise NotImplementedError("Implement this in the notebook before use.")