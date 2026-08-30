from collections import defaultdict
from dataclasses import dataclass, replace
import numpy as np

from .const import Const
from .dynamics import initial_state, is_collision, is_passing, transition
from .env import VecFlappy


@dataclass(frozen=True)
class DRConfig:
    """Which physics knobs to randomize at episode reset."""

    # Strong-flap noise half-width. Default demo knob: keeps the game playable.
    v_dev_choices: tuple[int, ...] | None = (0, 1, 2)
    # If set, multiply (g, U_weak, U_strong, V_max) by this scale together.
    # Do *not* change g alone — see the markdown above.
    vertical_scales: tuple[int, ...] | None = None

class DomainRandomizedVecFlappy(VecFlappy):
    """VecFlappy with per-environment physics resampled on every reset.

    Uses each env's own ``input_space`` / rewards / ``V_dev`` tape width, so
    consistent vertical scaling (g with thrust) actually changes the controls.
    """

    def __init__(
        self,
        C: Const,
        num_envs: int,
        dr_cfg: DRConfig,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        super().__init__(C, num_envs, seed=seed, auto_reset=auto_reset)
        self.base_C = C
        self.dr_cfg = dr_cfg
        self.C_env: list[Const] = [C] * self.num_envs

    def _write_tapes(self, idx: np.ndarray, seeds: np.ndarray) -> np.ndarray:
        h0 = np.empty(idx.size, dtype=np.int64)
        for j, (i, s) in enumerate(zip(idx, seeds)):
            C = self.C_env[i]
            rng = np.random.default_rng(int(s))
            h0[j] = rng.integers(len(C.S_h))
            self._w_spawn[i] = rng.random(C.T_max)
            self._w_h[i] = rng.integers(len(C.S_h), size=C.T_max)
            self._w_flap[i] = rng.integers(-C.V_dev, C.V_dev + 1, size=C.T_max)
        return h0

    @staticmethod
    def _sample_episode_physics(base: Const, rng: np.random.Generator, dr: DRConfig) -> Const:
        """Sample one episode's physics from the episode-seed RNG."""
        kw: dict = {}
        name = base.name

        if dr.vertical_scales is not None:
            s = int(rng.choice(dr.vertical_scales))
            kw.update(
                g=base.g,
                V_max=base.V_max,
                U_weak=base.U_weak,
                U_strong=base.U_strong,
            )
            name = f"{name}_s{s}"

        if dr.v_dev_choices is not None:
            v_dev = int(rng.choice(dr.v_dev_choices))
            kw["V_dev"] = v_dev
            name = f"{name}_vd{v_dev}"

        if not kw:
            return base
        kw["name"] = name
        return replace(base, **kw)

    def _reset_idx(self, idx: np.ndarray, seeds: np.ndarray) -> None:
        if idx.size == 0:
            return
        for i, s in zip(idx, seeds):
            self.C_env[i] = self._sample_episode_physics(
                self.base_C, np.random.default_rng(int(s)), self.dr_cfg
            )
        h0 = self._write_tapes(idx, seeds)
        for j, i in enumerate(idx):
            Ci = self.C_env[i]
            y, v, D, H = initial_state(Ci, 1, np.array([h0[j]]))
            self.y[i], self.v[i] = y[0], v[0]
            self.D[i], self.H[i] = D[0], H[0]
        self.t[idx] = 0
        self.pipes[idx] = 0
        self.ret[idx] = 0.0
        self.alive[idx] = True
        self.ep_seed[idx] = seeds

    def _groups(self, run: np.ndarray) -> dict[tuple, list[int]]:
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i in np.nonzero(run)[0]:
            Ci = self.C_env[i]
            key = (Ci.g, Ci.V_max, Ci.U_weak, Ci.U_strong, Ci.V_dev)
            groups[key].append(int(i))
        return groups

    def step(self, actions: np.ndarray):
        a = np.asarray(actions, dtype=np.int64)
        if a.shape != (self.num_envs,):
            raise ValueError(f"expected actions of shape ({self.num_envs},), got {a.shape}")
        if np.any((a < 0) | (a >= self.C.L)):
            raise ValueError("action index out of range")

        run = self.alive
        # Per-env thrust and reward — required if U_* is randomized.
        u = np.zeros(self.num_envs, dtype=np.int64)
        reward = np.zeros(self.num_envs, dtype=np.float64)
        for i in np.nonzero(run)[0]:
            Ci = self.C_env[i]
            u[i] = Ci.input_space[int(a[i])]
            reward[i] = Ci.reward_per_action[int(a[i])]

        passed = np.zeros(self.num_envs, dtype=bool)
        for idx in self._groups(run).values():
            ii = np.asarray(idx)
            Ci = self.C_env[ii[0]]
            passed[ii] = is_passing(Ci, self.y[ii], self.D[ii, 0], self.H[ii, 0])

        step_idx = np.minimum(self.t, self.C.T_max - 1)
        y2, v2, D2, H2 = self.y.copy(), self.v.copy(), self.D.copy(), self.H.copy()
        for idx in self._groups(run).values():
            ii = np.asarray(idx)
            Ci = self.C_env[ii[0]]
            si = step_idx[ii]
            yy, vv, DD, HH = transition(
                Ci,
                self.y[ii],
                self.v[ii],
                self.D[ii],
                self.H[ii],
                u[ii],
                self._w_spawn[ii, si],
                self._w_h[ii, si],
                self._w_flap[ii, si],
            )
            y2[ii], v2[ii], D2[ii], H2[ii] = yy, vv, DD, HH

        self.y = np.where(run, y2, self.y)
        self.v = np.where(run, v2, self.v)
        self.D = np.where(run[:, None], D2, self.D)
        self.H = np.where(run[:, None], H2, self.H)
        self.t += run
        self.pipes += passed
        self.ret += reward

        terminated = np.zeros(self.num_envs, dtype=bool)
        for idx in self._groups(run).values():
            ii = np.asarray(idx)
            Ci = self.C_env[ii[0]]
            terminated[ii] = is_collision(Ci, self.y[ii], self.D[ii, 0], self.H[ii, 0])

        truncated = (self.t >= self.C.T_max) & ~terminated & run
        done = terminated | truncated

        info = {
            "done": done,
            "final_state": self.state_dict(),
            "episode_return": self.ret.copy(),
            "episode_length": self.t.copy(),
            "episode_pipes": self.pipes.copy(),
            "episode_seed": self.ep_seed.copy(),
            "passed": passed,
        }

        idx_done = np.nonzero(done)[0]
        if idx_done.size:
            if self.auto_reset:
                self._reset_idx(idx_done, self._next_seeds(idx_done.size))
            else:
                self.alive[idx_done] = False

        return self.state_dict(), reward, terminated, truncated, info