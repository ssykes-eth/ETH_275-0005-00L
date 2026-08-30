"""Part 6 — The display agent & UI interaction.

The final agent doesn't generate text for the user to read — it *acts on the UI* by
calling tools. To keep that controlled and auditable, the agent may only use a fixed
``UITools`` interface (highlight a field, warn, attach a citation, suggest a correction,
mark OK); it never touches the DOM directly. The same ``run_display_agent`` runs in
tests (against a ``RecordingUI`` that just remembers the calls) and in the web backend
(against a ``DirectiveCollector`` that turns the calls into ``UIAction`` JSON for the
frontend) — the agent can't tell the difference, which is the point of a tool interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic.models import UIAction, VerificationResult


# --------------------------------------------------------------------------- #
# The tool interface the display agent is allowed to use.
# --------------------------------------------------------------------------- #
class UITools(ABC):
    """The controlled set of actions an agent may perform on the user interface."""

    @abstractmethod
    def highlight_field(self, field: str, message: str) -> None:
        """Flag a form field as problematic, with a short message."""

    @abstractmethod
    def warn(self, message: str, severity: str = "medium") -> None:
        """Show a warning banner for the action as a whole."""

    @abstractmethod
    def attach_citation(self, field: str | None, source: str, snippet: str) -> None:
        """Attach a policy citation (optionally to a field)."""

    @abstractmethod
    def suggest_correction(self, field: str | None, corrected_value: Any, rationale: str) -> None:
        """Offer a suggested value the user can apply with one click."""

    @abstractmethod
    def mark_ok(self, message: str) -> None:
        """Confirm to the user that the action is compliant."""


class RecordingUI(UITools):
    """Test/inspection double: records every tool call as ``(name, kwargs)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def highlight_field(self, field: str, message: str) -> None:
        self.calls.append(("highlight_field", {"field": field, "message": message}))

    def warn(self, message: str, severity: str = "medium") -> None:
        self.calls.append(("warn", {"message": message, "severity": severity}))

    def attach_citation(self, field: str | None, source: str, snippet: str) -> None:
        self.calls.append(("attach_citation", {"field": field, "source": source, "snippet": snippet}))

    def suggest_correction(self, field: str | None, corrected_value: Any, rationale: str) -> None:
        self.calls.append(
            ("suggest_correction", {"field": field, "corrected_value": corrected_value, "rationale": rationale})
        )

    def mark_ok(self, message: str) -> None:
        self.calls.append(("mark_ok", {"message": message}))

    def names(self) -> list[str]:
        """The tool names called, in order (handy for assertions)."""
        return [name for name, _ in self.calls]


class DirectiveCollector(UITools):
    """Backend double: turns tool calls into serialisable ``UIAction``s for the API."""

    def __init__(self) -> None:
        self.actions: list[UIAction] = []

    def highlight_field(self, field: str, message: str) -> None:
        self.actions.append(UIAction(type="highlight", field=field, message=message))

    def warn(self, message: str, severity: str = "medium") -> None:
        self.actions.append(UIAction(type="warning", message=message, payload={"severity": severity}))

    def attach_citation(self, field: str | None, source: str, snippet: str) -> None:
        self.actions.append(
            UIAction(type="citation", field=field, message=source, payload={"source": source, "snippet": snippet})
        )

    def suggest_correction(self, field: str | None, corrected_value: Any, rationale: str) -> None:
        self.actions.append(
            UIAction(
                type="correction",
                field=field,
                message=rationale,
                payload={"corrected_value": corrected_value},
            )
        )

    def mark_ok(self, message: str) -> None:
        self.actions.append(UIAction(type="ok", message=message))


def run_display_agent(result: VerificationResult, ui: UITools) -> None:
    raise NotImplementedError("Implement this in the notebook before use.")