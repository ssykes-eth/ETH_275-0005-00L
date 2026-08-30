"""Tolerant JSON extraction for LLM output (provided scaffolding).

LLMs are asked to return JSON but often wrap it in prose or a ```json fence. This
helper digs the JSON object out of a model response so the agents can parse a clean
dict. You don't implement this — you *use* it inside the verifier / solution agents.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Return the first JSON object found in ``text`` as a dict.

    Strategy, most-to-least forgiving:
      1. Parse the whole string.
      2. Parse the contents of a ```json … ``` fenced block.
      3. Parse the substring from the first ``{`` to the last ``}``.

    Raises ``ValueError`` if no JSON object can be recovered.
    """
    if isinstance(text, dict):  # already structured (e.g. a mock) — pass through
        return text

    text = (text or "").strip()

    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No JSON object found in model output: {text[:200]!r}")


def _candidates(text: str):
    yield text
    fence = _FENCE_RE.search(text)
    if fence:
        yield fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        yield text[start : end + 1]
