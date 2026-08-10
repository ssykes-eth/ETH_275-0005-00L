"""Batched Flappy Bird environment.

``VecFlappy`` runs ``N`` independent copies of the MDP in lockstep with
plain NumPy array operations -- there are no subprocesses and no Python
loop over environments. On a laptop CPU this reaches order ``1e6``
environment steps per second, which is what makes a full PPO run finish in
about a minute and, in turn, what makes the guided hyper-parameter
calibration in the notebook affordable.

Reproducibility
---------------
Every episode draws its entire disturbance sequence up front from its own
episode seed (a *noise tape*). Consequently an episode is bit-for-bit
reproducible from its seed alone, independently of ``N``, of which slot it
ran in, and of what the other environments were doing. The held-out
evaluation set is therefore just a list of integers.

Semantics
---------
``step`` follows the Gymnasium convention: it applies the input,
integrates, and *then* reports whether the resulting state is terminal.
Collision is a property of the state, so a terminal state has no successor
and is never stepped from -- auto-reset guarantees this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .const import Const
from .dynamics import (
    absolute_distances,
    initial_state,
    is_collision,
    is_passing,
    transition,
)

StateDict = dict[str, np.ndarray]


class VecFlappy:
    """``num_envs`` synchronous copies of the Flappy Bird grid MDP.

    Parameters
    ----------
    C
        Problem constants.
    num_envs
        Number of environments stepped in parallel.
    seed
        Master seed. Episode seeds are spawned from it deterministically.
    auto_reset
        If ``True`` (training), a finished environment is immediately reset
        and keeps producing data. If ``False`` (evaluation), it freezes and
        is excluded from further stepping; see :attr:`alive`.
    """

    def __init__(
        self,
        C: Const,
        num_envs: int,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        self.C = C
        self.num_envs = int(num_envs)
        self.auto_reset = bool(auto_reset)
        self.action_space_n = C.L
        self._actions = np.asarray(C.input_space, dtype=np.int64)
        self._rewards = C.reward_per_action

        self._seed_seq = np.random.SeedSequence(seed)
        self._seed_pool: list[int] = []

        N, M, T = self.num_envs, C.M, C.T_max
        self.y = np.zeros(N, dtype=np.int64)
        self.v = np.zeros(N, dtype=np.int64)
        self.D = np.zeros((N, M), dtype=np.int64)
        self.H = np.zeros((N, M), dtype=np.int64)
        self.t = np.zeros(N, dtype=np.int64)
        self.pipes = np.zeros(N, dtype=np.int64)
        self.ret = np.zeros(N, dtype=np.float64)
        self.alive = np.ones(N, dtype=bool)
        self.ep_seed = np.zeros(N, dtype=np.int64)

        self._w_spawn = np.zeros((N, T), dtype=np.float64)
        self._w_h = np.zeros((N, T), dtype=np.int64)
        self._w_flap = np.zeros((N, T), dtype=np.int64)

    # ------------------------------------------------------------------
    # Seeds and noise tapes
    # ------------------------------------------------------------------
    def _next_seeds(self, n: int) -> np.ndarray:
        """Draw ``n`` fresh episode seeds from the master sequence."""
        while len(self._seed_pool) < n:
            self._seed_pool.extend(
                int(s) for s in self._seed_seq.generate_state(256, dtype=np.uint32)
            )
        out = np.asarray(self._seed_pool[:n], dtype=np.int64)
        del self._seed_pool[:n]
        return out

    def _write_tapes(self, idx: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        """Fill the noise tapes of environments ``idx`` from ``seeds``.

        Returns the initial gap-centre index drawn for each episode.
        """
        C, T = self.C, self.C.T_max
        h0 = np.empty(idx.size, dtype=np.int64)
        n_h = len(C.S_h)
        for j, (i, s) in enumerate(zip(idx, seeds)):
            rng = np.random.default_rng(int(s))
            h0[j] = rng.integers(n_h)
            self._w_spawn[i] = rng.random(T)
            self._w_h[i] = rng.integers(n_h, size=T)
            self._w_flap[i] = rng.integers(-C.V_dev, C.V_dev + 1, size=T)
        return h0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def state_dict(self) -> StateDict:
        """Ground-truth state of every environment.

        This is deliberately *not* a flat vector: choosing which of these
        quantities a policy should see, and how to scale them, is the
        student's first design decision.

        Keys
        ----
        ``y``, ``v``
            Bird position and velocity.
        ``d``, ``h``
            Raw relative distances and gap centres.
        ``dx``
            Absolute distance to each obstacle; empty slots read ``X``.
        ``dy``
            Signed vertical offset ``h_i - y`` of each gap centre.
        ``active``
            Which obstacle slots are occupied.
        ``t``
            Steps elapsed in the current episode.
        """
        C = self.C
        dx, active = absolute_distances(C, self.D)
        return {
            "y": self.y.copy(),
            "v": self.v.copy(),
            "d": self.D.copy(),
            "h": self.H.copy(),
            "dx": dx,
            "dy": np.where(active, self.H - self.y[:, None], 0),
            "active": active,
            "t": self.t.copy(),
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def reset(self, seeds: np.ndarray | list[int] | None = None) -> StateDict:
        """Reset every environment. ``seeds`` fixes the episodes exactly."""
        idx = np.arange(self.num_envs)
        s = self._next_seeds(self.num_envs) if seeds is None else np.asarray(seeds, dtype=np.int64)
        if s.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} seeds, got {s.shape}")
        self._reset_idx(idx, s)
        return self.state_dict()

    def _reset_idx(self, idx: np.ndarray, seeds: np.ndarray) -> None:
        if idx.size == 0:
            return
        h0 = self._write_tapes(idx, seeds)
        y, v, D, H = initial_state(self.C, idx.size, h0)
        self.y[idx], self.v[idx] = y, v
        self.D[idx], self.H[idx] = D, H
        self.t[idx] = 0
        self.pipes[idx] = 0
        self.ret[idx] = 0.0
        self.alive[idx] = True
        self.ep_seed[idx] = seeds

    def step(
        self, actions: np.ndarray
    ) -> tuple[StateDict, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply one action index per environment.

        Parameters
        ----------
        actions
            ``(N,)`` integers in ``[0, L)`` indexing ``C.input_space``.

        Returns
        -------
        state, reward, terminated, truncated, info
            ``state`` is the state *after* the (possible) auto-reset, i.e.
            the one the policy must act on next. ``info["final_state"]``
            holds the pre-reset state, which is what value bootstrapping on
            truncated episodes needs; ``info["done"]`` masks it.
        """
        C = self.C
        a = np.asarray(actions, dtype=np.int64)
        if a.shape != (self.num_envs,):
            raise ValueError(f"expected actions of shape ({self.num_envs},), got {a.shape}")
        if np.any((a < 0) | (a >= C.L)):
            raise ValueError("action index out of range")

        run = self.alive
        u = self._actions[a]
        reward = np.where(run, self._rewards[a], 0.0)

        # Passing is decided on the state we are leaving, before the shift.
        passed = is_passing(C, self.y, self.D[:, 0], self.H[:, 0]) & run

        step_idx = np.minimum(self.t, C.T_max - 1)
        rows = np.arange(self.num_envs)
        y2, v2, D2, H2 = transition(
            C,
            self.y,
            self.v,
            self.D,
            self.H,
            u,
            self._w_spawn[rows, step_idx],
            self._w_h[rows, step_idx],
            self._w_flap[rows, step_idx],
        )

        # Frozen environments (evaluation mode) must not advance.
        self.y = np.where(run, y2, self.y)
        self.v = np.where(run, v2, self.v)
        self.D = np.where(run[:, None], D2, self.D)
        self.H = np.where(run[:, None], H2, self.H)
        self.t += run
        self.pipes += passed
        self.ret += reward

        terminated = is_collision(C, self.y, self.D[:, 0], self.H[:, 0]) & run
        truncated = (self.t >= C.T_max) & ~terminated & run
        done = terminated | truncated

        info: dict[str, Any] = {
            "done": done,
            "final_state": self.state_dict(),
            "episode_return": self.ret.copy(),
            "episode_length": self.t.copy(),
            "episode_pipes": self.pipes.copy(),
            "episode_seed": self.ep_seed.copy(),
            "passed": passed,
        }

        idx = np.nonzero(done)[0]
        if idx.size:
            if self.auto_reset:
                self._reset_idx(idx, self._next_seeds(idx.size))
            else:
                self.alive[idx] = False

        return self.state_dict(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"VecFlappy(instance={self.C.name!r}, num_envs={self.num_envs}, "
            f"auto_reset={self.auto_reset})"
        )
