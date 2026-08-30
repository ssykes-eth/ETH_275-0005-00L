"""Running policies in the simulator and measuring them.

Two quantities are reported and they answer different questions:

``discounted_return``
    what the agent is actually optimising, and the only thing comparable to
    the DP value function;
``pipes``
    the task metric, which is what the leaderboard ranks on. Never rank on
    the student's own shaped reward -- that would reward whoever writes the
    most generous reward function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .const import Const
from .env import VecFlappy


@dataclass
class EpisodeStats:
    """Per-episode results, aligned with the seeds that produced them."""

    seeds: np.ndarray
    undiscounted_return: np.ndarray
    discounted_return: np.ndarray
    length: np.ndarray
    pipes: np.ndarray
    terminated: np.ndarray
    flap_rate: np.ndarray

    def summary(self) -> dict[str, float]:
        n = max(self.seeds.size, 1)
        return {
            "episodes": float(self.seeds.size),
            "return": float(self.undiscounted_return.mean()),
            "discounted_return": float(self.discounted_return.mean()),
            "discounted_return_sem": float(
                self.discounted_return.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
            ),
            "length": float(self.length.mean()),
            "pipes": float(self.pipes.mean()),
            "pipes_sem": float(self.pipes.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0),
            "crash_rate": float(self.terminated.mean()),
            "flap_rate": float(self.flap_rate.mean()),
        }


def run_episodes(
    C: Const,
    policy,
    seeds,
    gamma: float = 0.99,
    batch_size: int | None = None,
) -> EpisodeStats:
    """Run one episode per seed and collect statistics.

    Episodes are batched for speed, but because each one is driven by its own
    noise tape the results do not depend on ``batch_size``.
    """
    seeds = np.asarray(seeds, dtype=np.int64)
    n = seeds.size
    bs = n if batch_size is None else min(batch_size, n)

    out = {
        k: np.zeros(n, dtype=t)
        for k, t in [
            ("undiscounted_return", np.float64),
            ("discounted_return", np.float64),
            ("length", np.int64),
            ("pipes", np.int64),
            ("terminated", bool),
            ("flap_rate", np.float64),
        ]
    }

    for start in range(0, n, bs):
        chunk = seeds[start : start + bs]
        m = chunk.size
        env = VecFlappy(C, m, seed=0, auto_reset=False)
        state = env.reset(chunk)
        policy.reset()

        disc = np.zeros(m)
        undisc = np.zeros(m)
        flaps = np.zeros(m)
        gpow = np.ones(m)
        crashed = np.zeros(m, dtype=bool)

        while env.alive.any():
            live = env.alive.copy()
            a = np.asarray(policy.act(state), dtype=np.int64)
            state, r, term, _, _ = env.step(a)
            undisc += r
            disc += gpow * r
            gpow = np.where(live, gpow * gamma, gpow)
            flaps += (a > 0) & live
            crashed |= term

        sl = slice(start, start + m)
        out["undiscounted_return"][sl] = undisc
        out["discounted_return"][sl] = disc
        out["length"][sl] = env.t
        out["pipes"][sl] = env.pipes
        out["terminated"][sl] = crashed
        out["flap_rate"][sl] = flaps / np.maximum(env.t, 1)

    return EpisodeStats(seeds=seeds, **out)


def heldout_seeds(n: int = 50, offset: int = 1_000_000) -> np.ndarray:
    """The evaluation set: a fixed block of integers, disjoint from training.

    Training draws its episode seeds from a master ``SeedSequence``, whose
    outputs are 32-bit and effectively never collide with this block. Keeping
    the held-out set this trivial is deliberate: there is no asset to
    version, hash, or accidentally leak.
    """
    return np.arange(offset, offset + n, dtype=np.int64)
