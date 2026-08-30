"""Turning the ground-truth state into a network input.

The scale constants live here, hardcoded and frozen. There is deliberately
no running-statistics normaliser: it would silently couple every run to its
own history, make checkpoints non-portable, and hand students a class of
bug that teaches nothing about control.
"""

from __future__ import annotations

import numpy as np

from .const import Const


def scales(C: Const) -> dict[str, float]:
    """Fixed normalisation constants, chosen so features land in ``[-1, 1]``."""
    return {
        "y": (C.Y - 1) / 2.0,
        "v": float(C.V_max),
        "dx": float(C.X),
        "dy": (C.Y - 1) / 2.0,
    }


def obs_dim(C: Const, mode: str, n_preview: int) -> int:
    if mode == "minimal":
        return 2
    if mode == "full":
        return 2 + 3 * min(n_preview, C.M)
    raise ValueError(f"unknown observation mode {mode!r}")


def build_observation(
    state: dict[str, np.ndarray],
    C: Const,
    mode: str = "full",
    n_preview: int = 2,
) -> np.ndarray:
    """Low-level observation builder (mode/n_preview kwargs).

    The graded Part 2 entry point is ``student_impl.build_observation(state, C, cfg)``;
    this helper is what the submission harness injects as ``_build_observation``.

    Parameters
    ----------
    mode
        ``"minimal"`` sees only its own velocity and the offset to the next
        gap -- enough to almost work, which is the point: students are meant
        to watch it fail on the far obstacles and then extend it.
        ``"full"`` adds a preview of the next ``n_preview`` obstacles.

    Returns
    -------
    ``(N, obs_dim)`` float32.
    """
    s = scales(C)
    y = (state["y"] - (C.Y - 1) / 2.0) / s["y"]
    v = state["v"] / s["v"]

    if mode == "minimal":
        dy0 = state["dy"][:, 0] / s["dy"]
        return np.stack([v, dy0], axis=1).astype(np.float32)

    if mode != "full":
        raise ValueError(f"unknown observation mode {mode!r}")

    P = min(n_preview, C.M)
    cols = [y, v]
    for i in range(P):
        cols.append(state["dx"][:, i] / s["dx"])
        cols.append(state["dy"][:, i] / s["dy"])
        # The activity flag matters: without it an empty slot's sentinel
        # distance is indistinguishable from a genuinely far obstacle.
        cols.append(state["active"][:, i].astype(np.float64))
    return np.stack(cols, axis=1).astype(np.float32)
