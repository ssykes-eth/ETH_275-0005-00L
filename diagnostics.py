"""Comparing a learned policy against the exact DP solution.

Only available on the small instance, which is the whole reason it exists.
These three numbers are what turn "the curve went up" into "the agent is at
X% of the optimum, and here is where it disagrees with pi*".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .const import Const
from .dp import Kernel, expected_return, policy_evaluation, state_index_of
from .dynamics import absolute_distances
from .env import VecFlappy


def states_to_dict(kernel: Kernel, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Present enumerated kernel states in the simulator's state format.

    Goes through :func:`flappy.dynamics.absolute_distances` rather than
    recomputing the cumsum and the off-screen sentinel by hand: a second
    copy of that convention would diverge silently the day it changes, and
    every diagnostic below would quietly measure the wrong thing.
    """
    C = kernel.C
    M = C.M
    d = kernel.states[idx, 2 : 2 + M]
    h = kernel.states[idx, 2 + M :]
    y = kernel.states[idx, 0]
    dx, active = absolute_distances(C, d)
    return {
        "y": y,
        "v": kernel.states[idx, 1],
        "d": d,
        "h": h,
        "dx": dx,
        "dy": np.where(active, h - y[:, None], 0),
        "active": active,
        "t": np.zeros(idx.size, dtype=np.int64),
    }


@dataclass
class Comparison:
    optimality_ratio: float
    agreement_visited: float
    agreement_uniform: float
    value_rmse_visited: float
    J_star: float
    J_policy: float
    visited_states: np.ndarray
    visit_counts: np.ndarray


def collect_visited(
    C: Const, policy, seeds, max_steps: int = 1000
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Roll the policy out and return the concatenated visited states."""
    seeds = np.asarray(seeds, dtype=np.int64)
    env = VecFlappy(C, seeds.size, seed=0, auto_reset=False)
    state = env.reset(seeds)
    policy.reset()
    chunks: list[dict[str, np.ndarray]] = []
    actions: list[np.ndarray] = []
    steps = 0
    while env.alive.any() and steps < max_steps:
        live = env.alive.copy()
        a = np.asarray(policy.act(state), dtype=np.int64)
        chunks.append({k: v[live] for k, v in state.items()})
        actions.append(a[live])
        state, *_ = env.step(a)
        steps += 1
    visited = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    return visited, np.concatenate(actions)


def compare_to_optimal(
    kernel: Kernel,
    V_star: np.ndarray,
    pi_star: np.ndarray,
    policy,
    gamma: float,
    seeds,
) -> Comparison:
    """Score a policy against ``pi*`` three different ways.

    The two agreement numbers are deliberately both reported. Agreement over
    the uniform state space counts states the agent never visits and is
    therefore pessimistic; agreement over the visitation distribution is the
    one that explains the return. Their gap is the lesson about which states
    a policy actually has to get right.
    """
    C = kernel.C
    visited, taken = collect_visited(C, policy, seeds)
    idx = state_index_of(kernel, visited)

    counts = np.bincount(idx, minlength=kernel.K)
    agreement_visited = float((taken == pi_star[idx]).mean())

    # Uniform agreement, over live states only: terminal states have no action.
    live = np.nonzero(~kernel.terminal)[0]
    uniform_state = states_to_dict(kernel, live)

    policy.reset()
    a_uniform = np.asarray(policy.act(uniform_state), dtype=np.int64)
    agreement_uniform = float((a_uniform == pi_star[live]).mean())

    # Exact value of the learned policy, via its greedy table on live states.
    pi_learned = np.zeros(kernel.K, dtype=np.int64)
    pi_learned[live] = a_uniform
    V_pi = policy_evaluation(kernel, pi_learned, gamma)

    J_star = expected_return(kernel, V_star)
    J_pi = expected_return(kernel, V_pi)

    w = counts / max(counts.sum(), 1)
    rmse = float(np.sqrt(np.sum(w * (V_pi - V_star) ** 2)))

    return Comparison(
        optimality_ratio=J_pi / J_star if J_star else float("nan"),
        agreement_visited=agreement_visited,
        agreement_uniform=agreement_uniform,
        value_rmse_visited=rmse,
        J_star=J_star,
        J_policy=J_pi,
        visited_states=np.nonzero(counts)[0],
        visit_counts=counts,
    )


def critic_vs_v_star(
    kernel: Kernel,
    V_star: np.ndarray,
    net,
    C: Const,
    cfg,
    states_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Data for the ``V_critic`` vs ``V*`` scatter plot.

    The dispersion is concentrated on rarely visited states, which is
    precisely why an imprecise critic still supports a good policy.
    """
    from .student_impl import build_observation

    live = np.nonzero(~kernel.terminal)[0] if states_idx is None else states_idx
    state = states_to_dict(kernel, live)

    with torch.no_grad():
        v = net.value(torch.as_tensor(build_observation(state, C, cfg))).numpy()
    return V_star[live], v
