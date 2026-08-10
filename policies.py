"""Reference policies: tabular DP, a threshold heuristic, and helpers.

These are the entries always published on the leaderboard. The DP policy is
an upper bound that exists only on the small instance; the heuristic is the
four-line controller that a first PPO attempt is expected to lose to.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .const import Const
from .dp import Kernel, state_index_of


class Policy(Protocol):
    """What the harness requires of a submission."""

    def reset(self) -> None:
        """Clear any internal history. Called at the start of every episode."""

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Map a batch of ground-truth states to action indices ``(N,)``."""


class TabularPolicy:
    """Wraps a DP policy table so it can be run in the simulator."""

    def __init__(self, kernel: Kernel, pi: np.ndarray) -> None:
        self.kernel = kernel
        self.pi = np.asarray(pi, dtype=np.int64)

    def reset(self) -> None:
        pass

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return self.pi[state_index_of(self.kernel, state)]


class ThresholdPolicy:
    """Flap when the next gap centre is above the bird, harder if far below.

    Deliberately naive and deliberately short: seeing four lines of
    geometry beat a first PPO attempt is the most useful row on the
    leaderboard.
    """

    def __init__(self, C: Const, margin: int = 0, strong_gap: int | None = None) -> None:
        self.C = C
        self.margin = margin
        # Reach for the strong flap only when the bird is far below the gap.
        self.strong_gap = C.gap_half if strong_gap is None else strong_gap

    def reset(self) -> None:
        pass

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        dy = state["dy"][:, 0]  # gap centre minus bird height
        # Where the bird will be by the time it can react.
        predicted = dy - state["v"]
        action = np.zeros_like(dy)
        action = np.where(predicted > self.margin, 1, action)
        action = np.where(predicted > self.margin + self.strong_gap, 2, action)
        return action.astype(np.int64)


class LookaheadPolicy:
    """Constant-velocity intercept of the next gap -- pure pursuit, discretised.

    Travelling from ``y`` to the gap centre ``h1`` in the ``d1`` steps before
    the obstacle arrives calls for a mean vertical velocity of
    ``(h1 - y) / d1``. The controller simply picks the input that brings the
    *next* velocity closest to that target, breaking ties towards the
    cheaper input.

    Two properties make this the right classical baseline. When the obstacle
    is far the target velocity is near zero, and the controller falls into a
    natural hover whose duty cycle is set by ``g / U_weak`` -- no altitude
    gain is hard-coded. And it plans against the *mean* strong flap, so it
    has no notion of the risk ``w_flap`` carries, which is precisely the
    margin a learned policy can take from it.

    Do not be tempted to replace this with a full ballistic rollout that
    assumes no further flap: over a long approach that trajectory always
    ends on the floor whatever the first input is, every action ties, and
    the controller stops flapping altogether.
    """

    def __init__(self, C: Const) -> None:
        self.C = C

    def reset(self) -> None:
        pass

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        C = self.C
        y, v = state["y"], state["v"]
        d1, h1 = state["dx"][:, 0], state["h"][:, 0]

        v_target = np.clip((h1 - y) / np.maximum(d1, 1), -C.V_max, C.V_max)

        cost = np.empty((C.L, y.size))
        for a, u in enumerate(C.input_space):
            v_next = np.clip(v + u - C.g, -C.V_max, C.V_max)
            cost[a] = np.abs(v_next - v_target) + 0.01 * C.effort[a]
        return cost.argmin(axis=0).astype(np.int64)


class RandomPolicy:
    """Uniform over inputs. The floor of the leaderboard."""

    def __init__(self, C: Const, seed: int = 0) -> None:
        self.C = C
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        return self.rng.integers(self.C.L, size=state["y"].shape[0])
