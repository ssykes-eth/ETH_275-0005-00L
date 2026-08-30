"""Extracting a student's work from their notebook.

The notebook *is* the hand-in. Nothing else is submitted: no weights, no
checkpoint, no agent module. The leaderboard reads the notebook, pulls out
the functions and the calibrated constants, and **retrains from scratch**.

Two consequences are the reason for doing it this way. The score becomes
reproducible from the submitted file alone, and nobody can hand in a lucky
checkpoint whose training run they could not repeat.

Extraction deliberately does not execute the notebook. It parses each code
cell and runs only the *definition* cells -- those whose top-level
statements are imports, assignments, functions or classes. A cell that
calls something at top level (a training run, a plot, a print) is skipped,
so pulling the work out of a notebook costs no more than parsing it.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# What a submitted notebook must define by the time its definition cells
# have run. `Config.for_large()` supplies every other hyper-parameter.
REQUIRED_FUNCTIONS = ("build_observation", "compute_gae", "ppo_loss")
REQUIRED_CONFIG = "FINAL_CONFIG"

# Provided implementations the student is not asked to rewrite; the harness
# injects them so a notebook need not carry a copy.
PROVIDED_FUNCTIONS = ("shaped_reward", "bootstrap_truncated")

# A submission may not buy its way up the leaderboard with compute.
MAX_ENV_STEPS = 5_000_000

# Statement types that make a cell safe to execute for its definitions alone.
_DEFINITION_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


class SubmissionError(Exception):
    """Raised with a message meant to be read by the student."""


@dataclass
class Submission:
    """What a notebook yields once its definition cells have run."""

    impl: SimpleNamespace
    config: Any
    namespace: dict[str, Any]
    source_cells: int


def is_definition_cell(source: str) -> bool:
    """Whether a cell can be executed for its definitions alone.

    Magics (``%%writefile`` and friends) are not Python and are always
    skipped; so is any cell that evaluates something at top level, which is
    what keeps a student's own training runs and plots out of the harness.
    """
    stripped = source.lstrip()
    if stripped.startswith(("%", "!")):
        return False
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return bool(tree.body) and all(isinstance(n, _DEFINITION_NODES) for n in tree.body)


def notebook_code_cells(path: str | Path) -> list[str]:
    nb = json.loads(Path(path).read_text())
    if nb.get("nbformat") != 4:
        raise SubmissionError(f"{path} is not an nbformat 4 notebook")
    return [
        "".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code"
    ]


def base_namespace() -> dict[str, Any]:
    """The names a submitted cell may rely on without importing them."""
    import numpy as np
    import torch

    from . import diagnostics, dp, features, student_impl
    from .const import Const
    from .env import VecFlappy
    from .evaluate import heldout_seeds, run_episodes
    from .networks import ActorCritic
    from .policies import LookaheadPolicy, RandomPolicy, ThresholdPolicy
    from .ppo import Config, NetworkPolicy, train

    from typing import Any as _Any

    return {
        "__name__": "submission",
        # Annotations in the TODO cells are evaluated at definition time.
        "Any": _Any,
        "np": np,
        "torch": torch,
        "Const": Const,
        "VecFlappy": VecFlappy,
        "Config": Config,
        "train": train,
        "NetworkPolicy": NetworkPolicy,
        "run_episodes": run_episodes,
        "heldout_seeds": heldout_seeds,
        "ActorCritic": ActorCritic,
        "LookaheadPolicy": LookaheadPolicy,
        "ThresholdPolicy": ThresholdPolicy,
        "RandomPolicy": RandomPolicy,
        "dp": dp,
        "diagnostics": diagnostics,
        "SMALL": Const.small(),
        "LARGE": Const.large(),
        "_build_observation": features.build_observation,
        **{name: getattr(student_impl, name) for name in PROVIDED_FUNCTIONS},
    }


def _resolve_fn(namespace: dict[str, Any], name: str) -> Any | None:
    """Look up a required function under ``name`` or ``_name``.

    Underscore aliases are accepted so a notebook can keep private-looking
    names (``_build_observation``, …) while ``impl`` still exposes the
    canonical API that :func:`flappy.ppo.train` calls. The harness-injected
    ``features.build_observation`` helper must not count as a submission.
    """
    if callable(namespace.get(name)):
        return namespace[name]
    fn = namespace.get(f"_{name}")
    if not callable(fn):
        return None
    if name == "build_observation":
        from . import features

        if fn is features.build_observation:
            return None
    return fn


def collect(namespace: dict[str, Any], source_cells: int = 0) -> Submission:
    """Validate a namespace and package it for :func:`flappy.ppo.train`.

    Also usable from inside the notebook -- ``collect(globals())`` is the
    self-check students run before handing in.
    """
    resolved = {n: _resolve_fn(namespace, n) for n in REQUIRED_FUNCTIONS}
    missing = [n for n, fn in resolved.items() if fn is None]
    if missing:
        alts = ", ".join(f"_{n}" for n in missing)
        raise SubmissionError(
            "the notebook does not define " + ", ".join(missing)
            + f" (or {alts}). Each must be "
            "a plain top-level function in its own cell; the harness does not "
            "execute cells that call or print anything."
        )
    if REQUIRED_CONFIG not in namespace:
        raise SubmissionError(
            f"the notebook does not define {REQUIRED_CONFIG}. Add a cell that "
            f"assigns it, e.g. `{REQUIRED_CONFIG} = Config.for_large(gamma=GAMMA, ...)`."
        )

    cfg = namespace[REQUIRED_CONFIG]
    from .ppo import Config

    if not isinstance(cfg, Config):
        raise SubmissionError(
            f"{REQUIRED_CONFIG} must be a Config, got {type(cfg).__name__}"
        )
    if cfg.instance != "large":
        raise SubmissionError(
            f"{REQUIRED_CONFIG}.instance must be 'large'; the leaderboard runs "
            f"on the large instance, got {cfg.instance!r}"
        )
    try:
        cfg.validate()
    except ValueError as exc:
        raise SubmissionError(
            f"{REQUIRED_CONFIG} is not trainable: {exc}. A hyper-parameter slot "
            "was probably left unfilled."
        ) from exc
    if cfg.total_steps > MAX_ENV_STEPS:
        raise SubmissionError(
            f"{REQUIRED_CONFIG}.total_steps is {cfg.total_steps:,}, above the "
            f"{MAX_ENV_STEPS:,} budget. The leaderboard compares design, not compute."
        )

    impl = SimpleNamespace(
        **resolved,
        **{n: namespace.get(n, base_namespace()[n]) for n in PROVIDED_FUNCTIONS},
    )
    return Submission(impl=impl, config=cfg, namespace=namespace, source_cells=source_cells)


def load_notebook(path: str | Path) -> Submission:
    """Run a notebook's definition cells and collect the submission."""
    namespace = base_namespace()
    executed = 0
    for i, source in enumerate(notebook_code_cells(path)):
        if not is_definition_cell(source):
            continue
        try:
            exec(compile(source, f"<{Path(path).name} cell {i}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            raise SubmissionError(
                f"cell {i} of {Path(path).name} failed to execute: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        executed += 1
    return collect(namespace, source_cells=executed)
