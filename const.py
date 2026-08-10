"""Problem constants for the Flappy Bird grid MDP.

The bird lives on a discrete grid. Its state is its height and vertical
velocity plus the positions and gap centres of the obstacles ahead; its
input is one of three upward impulses, the strongest of which is noisy.

Two instances are provided:

``Const.small()``
    A grid coarse enough to enumerate, so the exact optimal value function
    and policy can be computed by value iteration (see :mod:`flappy.dp`).
    Used to validate that PPO recovers the optimum.

``Const.large()``
    The same problem on a finer grid, where enumeration is hopeless
    (|S| ~ 1e17). This is the instance the project is actually about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from math import ceil
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Const:
    """Constants describing one instance of the Flappy Bird MDP.

    Attributes
    ----------
    X, Y
        Grid width (columns) and height (rows). The bird always sits in
        column 0; obstacles travel one column to the left per step.
    V_max
        Vertical velocity bound; ``v`` lives in ``[-V_max, V_max]``.
    g
        Gravity: downward velocity decrement applied every step.
    U_no_flap, U_weak, U_strong
        The three admissible inputs, expressed as upward velocity impulses.
    V_dev
        Half-width of the strong-flap disturbance: applying ``U_strong``
        adds ``w_flap`` drawn uniformly from ``[-V_dev, V_dev]``.
    D_min
        Minimum spacing between two consecutive obstacles.
    G
        Gap height, must be odd.
    S_h
        Admissible gap centres.
    lam_weak, lam_strong
        Effort cost of the two flaps. The per-step reward is
        ``1 - lam_u``: survive, but pay for flapping.
    T_max
        Episode truncation horizon, in steps.
    """

    X: int
    Y: int
    V_max: int
    g: int
    U_no_flap: int
    U_weak: int
    U_strong: int
    V_dev: int
    D_min: int
    G: int
    S_h: tuple[int, ...]
    lam_weak: float
    lam_strong: float
    T_max: int
    name: str = "custom"
    d_init: int | None = None
    """Distance to the first obstacle at reset; ``None`` means ``X - 1``.

    ``X - 1`` -- the obstacle entering from the right edge -- is a rendering
    artefact, not physics, and it hands the agent a stretch of free survival
    reward before its first decision matters. Exact DP does not care; a
    sampled policy gradient very much does, because "never flap" collects
    that whole plateau and becomes a deceptive optimum with a wide basin.
    Starting at a steady-state distance removes the artefact.
    """

    def __post_init__(self) -> None:
        if self.G % 2 == 0:
            raise ValueError(f"gap size G must be odd, got {self.G}")
        if not 0 < self.D_min < self.X:
            raise ValueError(f"need 0 < D_min < X, got D_min={self.D_min}, X={self.X}")
        half = (self.G - 1) // 2
        if any(not 0 <= h <= self.Y - 1 for h in self.S_h):
            raise ValueError("every gap centre must lie inside the grid")
        # A gap that is entirely clipped away would make the level unplayable.
        if any(h + half < 0 or h - half > self.Y - 1 for h in self.S_h):
            raise ValueError("every gap must intersect the grid")
        if self.d_init is not None and not 0 < self.d_init <= self.X - 1:
            raise ValueError(f"need 0 < d_init <= X - 1, got {self.d_init}")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def M(self) -> int:
        """Maximum number of obstacles simultaneously on the grid."""
        return ceil(self.X / self.D_min)

    @property
    def initial_distance(self) -> int:
        """Distance to the first obstacle at reset."""
        return self.X - 1 if self.d_init is None else self.d_init

    @property
    def gap_half(self) -> int:
        """Half-height of the gap: the bird clears obstacle ``i`` iff
        ``|y - h_i| <= gap_half``."""
        return (self.G - 1) // 2

    @property
    def input_space(self) -> tuple[int, int, int]:
        return (self.U_no_flap, self.U_weak, self.U_strong)

    @property
    def L(self) -> int:
        """Number of admissible inputs."""
        return len(self.input_space)

    @property
    def effort(self) -> np.ndarray:
        """Effort cost per action index, shape ``(L,)``."""
        return np.array([0.0, self.lam_weak, self.lam_strong], dtype=np.float64)

    @property
    def reward_per_action(self) -> np.ndarray:
        """Per-step reward per action index, ``1 - lam_u``, shape ``(L,)``."""
        return 1.0 - self.effort

    @property
    def S_h_array(self) -> np.ndarray:
        return np.asarray(self.S_h, dtype=np.int64)

    @property
    def S_y(self) -> list[int]:
        return list(range(self.Y))

    @property
    def S_v(self) -> list[int]:
        return list(range(-self.V_max, self.V_max + 1))

    @property
    def S_d1(self) -> list[int]:
        """Admissible values for the distance to the *first* obstacle."""
        return list(range(self.X))

    @property
    def S_d(self) -> list[int]:
        """Admissible values for ``d_2, ..., d_M``; ``0`` means "slot empty"."""
        return [0] + list(range(self.D_min, self.X))

    # ------------------------------------------------------------------
    # Enumeration (only meaningful for the small instance)
    # ------------------------------------------------------------------
    def is_valid_state(self, x: Sequence[int]) -> bool:
        """Return whether ``x = (y, v, d_1..d_M, h_1..h_M)`` is reachable.

        The two structural invariants that matter downstream are: relative
        distances must fit inside the grid, and empty obstacle slots must be
        trailing.
        """
        M = self.M
        y, v = x[0], x[1]
        D = list(x[2 : 2 + M])
        H = list(x[2 + M :])

        if y not in range(self.Y) or v not in range(-self.V_max, self.V_max + 1):
            return False
        if sum(D) > self.X - 1:
            return False
        if M > 1 and D[1] == 0 and D[0] <= 0:
            return False
        if D[0] not in self.S_d1:
            return False
        if any(d not in self.S_d for d in D[1:]):
            return False
        if any(h not in self.S_h for h in H):
            return False
        # Inactive slots carry the canonical placeholder gap centre.
        if any(d == 0 and h != self.S_h[0] for d, h in zip(D[1:], H[1:])):
            return False
        # Once a slot is empty, every later slot must be empty too.
        zero_seen = False
        for d in D[1:]:
            if zero_seen and d != 0:
                return False
            zero_seen = zero_seen or d == 0
        return True

    @cached_property
    def state_space(self) -> list[tuple[int, ...]]:
        """All reachable states, as tuples. Only call on small instances."""
        iterables = (
            [self.S_y, self.S_v, self.S_d1]
            + [self.S_d] * (self.M - 1)
            + [self.S_h] * self.M
        )
        return [tuple(x) for x in product(*iterables) if self.is_valid_state(x)]

    @cached_property
    def _state_index(self) -> dict[tuple[int, ...], int]:
        return {s: i for i, s in enumerate(self.state_space)}

    def state_to_index(self, x: Sequence[int]) -> int:
        key = tuple(int(v) for v in x)
        try:
            return self._state_index[key]
        except KeyError as exc:
            raise KeyError(f"state {key} is not in the state space") from exc

    @property
    def K(self) -> int:
        """Size of the enumerated state space."""
        return len(self.state_space)

    def state_space_size_estimate(self) -> float:
        """Upper bound on ``|S|`` without enumerating -- safe on any instance.

        This is what makes the curse of dimensionality concrete: it stays
        cheap exactly where enumeration stops being possible.
        """
        return float(
            self.Y
            * (2 * self.V_max + 1)
            * self.X
            * (len(self.S_d) ** (self.M - 1))
            * (len(self.S_h) ** self.M)
        )

    # ------------------------------------------------------------------
    # Instances
    # ------------------------------------------------------------------
    @classmethod
    def small(cls) -> "Const":
        """The coarse instance: enumerable, exactly solvable."""
        return cls(
            X=8,
            Y=14,
            V_max=2,
            g=1,
            U_no_flap=0,
            U_weak=2,
            U_strong=3,
            V_dev=1,
            D_min=4,
            G=3,
            S_h=(5, 9, 13),
            lam_weak=0.5,
            lam_strong=0.7,
            T_max=1000,
            name="small",
            d_init=None,  # X - 1 = 7, i.e. ~1.5 * Delta: no plateau worth removing
        )

    @classmethod
    def large(cls) -> "Const":
        """The project instance: the small one on a finer grid.

        Every ratio of :meth:`small` is preserved and only the
        discretisation is refined -- vertical quantities by 8, horizontal
        ones by 4. The continuum problem is therefore *the same*; what
        changes is that ``|S|`` goes from about 8e3 to about 1e17, so value
        iteration stops being an option. That is the entire argument for
        function approximation, and it is made by construction rather than
        by assertion.

        Note that gravity scales with the vertical refinement. It has to:
        with ``g`` left at 1 the bird would still rise impulsively but could
        only fall at one row per step, and a large share of consecutive gap
        pairs would be physically unreachable -- the level generator would
        be handing out unsurvivable episodes.

        Calibrated by ``scripts/calibrate_instance.py`` against the bands
        the notebook's exercises assume: obstacle spacing ``Delta`` in
        ``[12, 18]`` steps, reference-policy episode length in
        ``[200, 650]``, flap duty cycle in ``[0.10, 0.55]``.
        Measured with :class:`flappy.policies.LookaheadPolicy`:
        ``Delta = 16.8``, length ``239``, duty cycle ``0.50``.
        """
        k, G = 8, 15  # vertical refinement, gap height
        Y = 14 * k
        half = (G - 1) // 2
        return cls(
            X=32,
            Y=Y,
            V_max=2 * k,
            g=k,
            U_no_flap=0,
            U_weak=2 * k,
            U_strong=3 * k,
            V_dev=2,
            D_min=10,
            G=G,
            S_h=tuple(range(half, Y - half)),
            lam_weak=0.5,
            lam_strong=0.7,
            T_max=1000,
            name="large",
        )


SMALL = Const.small()
LARGE = Const.large()
