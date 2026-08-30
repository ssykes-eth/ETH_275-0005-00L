"""Student-solution loader — monkey-patches exported notebook functions onto the app.

The WE6 course notebook (``notebook/agentic_verifier_project.ipynb``) has students
implement the six pipeline functions themselves. At the end of each part an *export*
cell writes a file here:

    solutions/part1_action.py      (validate_action)
    solutions/part2_context.py     (build_context)
    solutions/part3_retrieval.py   (retrieve_policies)
    solutions/part4_verifier.py    (verify_action)
    solutions/part5_solution.py    (propose_solution)
    solutions/part6_display.py     (run_display_agent)
    solutions/part7_pipeline.py    (verify)

``apply()`` rebinds each of those onto the real ``agentic.*`` classes/modules, so the
runnable app (``main.py`` / ``server.py``) verifies actions using *the student's* code.

It is deliberately *guarded*: a missing file, a missing function, or an un-edited stub
(still marked ``@todo``) is skipped, so a fresh checkout runs on the reference
implementation. This mirrors the WE5 project ``solutions`` loader exactly.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Callable

VALID_MODES = ("auto", "reference", "student")
DEFAULT_MODE = "auto"

_active_mode: str | None = None


def todo(fn: Callable) -> Callable:
    """Mark a shipped stub as not-yet-implemented so ``apply()`` skips it."""
    fn._is_todo = True  # type: ignore[attr-defined]
    return fn


@dataclass(frozen=True)
class PatchSpec:
    part_attr: str  # function name in the part file
    target_module: str  # module that holds the target
    target_path: str  # dotted path within the module ("name" or "Class.method")
    kind: str  # "function" | "method" | "static"


# Part files in display order: (label, part module, specs).
PARTS: list[tuple[str, str, list[PatchSpec]]] = [
    ("Part 1 · Action", "solutions.part1_action",
     [PatchSpec("validate_action", "agentic.action_validation", "validate_action", "function")]),
    ("Part 2 · Context", "solutions.part2_context",
     [PatchSpec("build_context", "agentic.context_builder", "build_context", "function")]),
    ("Part 3 · Retrieval", "solutions.part3_retrieval",
     [PatchSpec("retrieve_policies", "agentic.policy_tool", "PolicyRetrievalTool.retrieve_policies", "method")]),
    ("Part 4 · Verifier", "solutions.part4_verifier",
     [PatchSpec("verify_action", "agentic.verifier_agent", "VerifierAgent.verify_action", "method")]),
    ("Part 5 · Solution", "solutions.part5_solution",
     [PatchSpec("propose_solution", "agentic.solution_agent", "SolutionAgent.propose_solution", "method")]),
    ("Part 6 · Display", "solutions.part6_display",
     [PatchSpec("run_display_agent", "agentic.display_agent", "run_display_agent", "function")]),
    ("Part 7 · Pipeline", "solutions.part7_pipeline",
     [PatchSpec("verify", "agentic.pipeline", "VerifierPipeline.verify", "method")]),
]


def _bind(spec: PatchSpec, fn: Callable) -> None:
    module = importlib.import_module(spec.target_module)
    parts = spec.target_path.split(".")
    owner = module
    for name in parts[:-1]:  # walk into a class if the path is "Class.method"
        owner = getattr(owner, name)
    value = staticmethod(fn) if spec.kind == "static" else fn
    setattr(owner, parts[-1], value)


def _apply_specs(part_module_name: str, specs: list[PatchSpec]) -> list[str]:
    try:
        part = importlib.import_module(part_module_name)
    except ModuleNotFoundError:
        return []
    patched: list[str] = []
    for spec in specs:
        fn = getattr(part, spec.part_attr, None)
        if fn is None or getattr(fn, "_is_todo", False):
            continue
        _bind(spec, fn)
        patched.append(f"{spec.target_module}.{spec.target_path}")
    return patched


def _resolve_mode(mode: str | None) -> str:
    if mode is None:
        mode = os.environ.get("AGENT_IMPL", DEFAULT_MODE)
    mode = mode.strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"AGENT_IMPL must be one of {VALID_MODES}, got {mode!r}")
    return mode


def all_targets() -> list[str]:
    return [f"{s.target_module}.{s.target_path}" for _, _, specs in PARTS for s in specs]


def _describe(fn: Callable | None) -> str:
    return (fn.__doc__ or "").strip().split("\n", 1)[0] if fn is not None else ""


def status() -> dict:
    """Report which exported functions are implemented vs still stubbed/missing.

    Pure introspection (never patches), so it is safe to call on every request.
    Drives the frontend progress dashboard. ``reason`` is ``ok`` | ``stub`` | ``missing``.
    """
    functions: list[dict] = []
    for label, modname, specs in PARTS:
        try:
            part = importlib.import_module(modname)
        except ModuleNotFoundError:
            part = None
        for spec in specs:
            fn = getattr(part, spec.part_attr, None) if part is not None else None
            if fn is None:
                reason = "missing"
            elif getattr(fn, "_is_todo", False):
                reason = "stub"
            else:
                reason = "ok"
            functions.append({
                "part": label,
                "name": spec.part_attr,
                "target": f"{spec.target_module}.{spec.target_path}",
                "kind": spec.kind,
                "implemented": reason == "ok",
                "reason": reason,
                "description": _describe(fn),
            })
    done = sum(f["implemented"] for f in functions)
    return {
        "mode": _active_mode if _active_mode is not None else _resolve_mode(None),
        "implemented": done,
        "total": len(functions),
        "complete": done == len(functions),
        "functions": functions,
    }


def apply(verbose: bool = True, mode: str | None = None) -> list[str]:
    """Select the implementation the live app runs on. Safe to call once at startup."""
    global _active_mode
    mode = _resolve_mode(mode)
    _active_mode = mode

    if mode == "reference":
        if verbose:
            print("[solutions] mode=reference — using the repo's reference implementation.")
        return []

    patched: list[str] = []
    for _label, modname, specs in PARTS:
        patched += _apply_specs(modname, specs)

    if mode == "student":
        missing = sorted(set(all_targets()) - set(patched))
        if missing:
            raise RuntimeError(
                "AGENT_IMPL=student, but these functions are not provided by your "
                "solutions/ files (still a @todo stub or missing): " + ", ".join(missing)
                + ".\nExport them from the notebook, or use AGENT_IMPL=auto."
            )

    if verbose:
        if patched:
            print(f"[solutions] mode={mode} — applied student implementations: {', '.join(patched)}")
        else:
            print(f"[solutions] mode={mode} — no student implementations found; using reference.")
    return patched
