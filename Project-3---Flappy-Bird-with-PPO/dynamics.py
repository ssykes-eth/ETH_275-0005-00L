"""Pure, batched dynamics of the Flappy Bird grid MDP.

Everything here is a *pure function* of the current state ``x``, the input
``u`` and an exogenous disturbance ``w``:

.. math:: x_{k+1} = f(x_k, u_k, w_k)

That is the optimal-control form on purpose. Feeding the disturbance in
explicitly (instead of drawing it inside) buys three things at once:
the transition is testable against a scalar reference, the environment is
exactly reproducible from an episode seed regardless of how many
environments run in parallel, and the enumeration in :mod:`flappy.dp` can
sweep over all disturbance outcomes to build the transition kernel.

Every function is vectorised over a leading batch dimension ``N``.

State
-----
``y``   ``(N,) int``    vertical position, ``0`` at the bottom
``v``   ``(N,) int``    vertical velocity
``D``   ``(N, M) int``  *relative* distances: ``D[:, 0]`` to the first
                        obstacle, ``D[:, i]`` from obstacle ``i-1`` to ``i``.
                        ``D[:, i] == 0`` for ``i >= 1`` marks an empty slot.
``H``   ``(N, M) int``  gap centres of the corresponding obstacles.

Disturbance
-----------
``w_spawn`` ``(N,) float`` uniform in ``[0, 1)``, compared against the spawn
                           probability
``w_h``     ``(N,) int``   index into ``C.S_h`` for a freshly spawned gap
``w_flap``  ``(N,) int``   in ``[-V_dev, V_dev]``, applied only when the
                           input is a strong flap
"""

from __future__ import annotations

import numpy as np

from .const import Const

__all__ = [
    "is_in_gap",
    "is_passing",
    "is_collision",
    "spawn_probability",
    "transition",
    "initial_state",
    "absolute_distances",
]


def is_in_gap(C: Const, y: np.ndarray, h1: np.ndarray) -> np.ndarray:
    """Whether height ``y`` is within the gap centred at ``h1``."""
    return np.abs(y - h1) <= C.gap_half


def is_passing(C: Const, y: np.ndarray, d1: np.ndarray, h1: np.ndarray) -> np.ndarray:
    """Whether the bird is level with the first obstacle and inside its gap."""
    return (d1 == 0) & is_in_gap(C, y, h1)


def is_collision(C: Const, y: np.ndarray, d1: np.ndarray, h1: np.ndarray) -> np.ndarray:
    """Whether the bird is level with the first obstacle and outside its gap."""
    return (d1 == 0) & ~is_in_gap(C, y, h1)


def spawn_probability(C: Const, s: np.ndarray) -> np.ndarray:
    """Distance-dependent spawn probability, given free distance ``s``.

    Reaches ``1`` exactly when ``s == X - 1``, which is what guarantees that
    a new obstacle is always queued before the current one is consumed --
    the invariant that keeps ``D[:, 0]`` from going negative.
    """
    return np.clip((s - C.D_min + 1) / (C.X - C.D_min), 0.0, 1.0)


def initial_state(
    C: Const, n: int, w_h: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initial state: centred bird, at rest, one obstacle at the far right.

    Parameters
    ----------
    n
        Batch size.
    w_h
        ``(n,)`` indices into ``C.S_h`` choosing the first gap centre.
    """
    M = C.M
    y = np.full(n, C.Y // 2, dtype=np.int64)
    v = np.zeros(n, dtype=np.int64)
    D = np.zeros((n, M), dtype=np.int64)
    D[:, 0] = C.initial_distance
    H = np.full((n, M), C.S_h[0], dtype=np.int64)
    H[:, 0] = C.S_h_array[w_h]
    return y, v, D, H


def transition(
    C: Const,
    y: np.ndarray,
    v: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    u: np.ndarray,
    w_spawn: np.ndarray,
    w_h: np.ndarray,
    w_flap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance every environment in the batch by one step.

    The caller must guarantee that no input state is already colliding: a
    colliding state is terminal and has no successor. :class:`flappy.env.VecFlappy`
    enforces this by resetting dead environments before stepping.

    Returns
    -------
    ``(y_next, v_next, D_next, H_next)``

    Notes
    -----
    The integration is explicit Euler, ``y_{k+1} = y_k + v_k`` with the *old*
    velocity. Do not "fix" it into a semi-implicit scheme: it would silently
    change the optimal policy and invalidate every number the DP comparison
    reports.
    """
    N, M = D.shape

    # --- obstacle bookkeeping ------------------------------------------
    passing = is_passing(C, y, D[:, 0], H[:, 0])

    # If the bird passes the first obstacle, drop it and shift the queue left.
    D_shift = np.concatenate([D[:, 1:], np.zeros((N, 1), dtype=D.dtype)], axis=1)
    H_shift = np.concatenate(
        [H[:, 1:], np.full((N, 1), C.S_h[0], dtype=H.dtype)], axis=1
    )
    D_shift[:, 0] -= 1

    # Otherwise the first obstacle simply comes one column closer.
    D_keep = D.copy()
    D_keep[:, 0] -= 1

    hat_D = np.where(passing[:, None], D_shift, D_keep)
    hat_H = np.where(passing[:, None], H_shift, H)

    # --- stochastic spawn ----------------------------------------------
    s = (C.X - 1) - hat_D.sum(axis=1)
    do_spawn = w_spawn <= spawn_probability(C, s)

    # Target slot: the first empty interior slot, else the last slot.
    if M >= 3:
        interior_empty = hat_D[:, 1 : M - 1] == 0
        has_empty = interior_empty.any(axis=1)
        k = np.where(has_empty, np.argmax(interior_empty, axis=1) + 1, M - 1)
    else:
        k = np.full(N, M - 1, dtype=np.int64)

    rows = np.nonzero(do_spawn)[0]
    if rows.size:
        cols = k[rows]
        hat_D[rows, cols] = np.clip(s[rows], C.D_min, C.X - 1)
        hat_H[rows, cols] = C.S_h_array[w_h[rows]]

    # --- vertical dynamics ---------------------------------------------
    # The strong flap is the only input whose effect is uncertain.
    w = np.where(u == C.U_strong, w_flap, 0)
    y_next = np.clip(y + v, 0, C.Y - 1)
    v_next = np.clip(v + u + w - C.g, -C.V_max, C.V_max)

    return y_next, v_next, hat_D, hat_H


def absolute_distances(C: Const, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert relative distances to absolute ones, plus an activity mask.

    Returns
    -------
    dx
        ``(N, M)`` absolute horizontal distance to each obstacle. Empty
        slots are reported as ``C.X`` (i.e. "off screen"), so that a policy
        reading this feature sees a finite, out-of-range sentinel rather
        than a discontinuity.
    active
        ``(N, M)`` boolean mask of occupied slots.
    """
    dx = np.cumsum(D, axis=1)
    # Slot i >= 1 is active iff it and all previous slots are non-empty.
    occupied = D != 0
    occupied[:, 0] = True  # the first obstacle always exists
    active = np.cumprod(occupied, axis=1).astype(bool)
    dx = np.where(active, dx, C.X)
    return dx, active
