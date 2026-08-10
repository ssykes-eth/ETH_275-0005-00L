"""Exact dynamic programming on the small instance.

The whole point of keeping a small, enumerable instance around is that it
gives the project something almost no RL course has: **ground truth**.
Value iteration here returns the exact optimal value function ``V*`` and
policy ``pi*``, against which a trained PPO agent can be scored rather than
merely plotted.

The transition kernel is built by sweeping the batched dynamics over every
disturbance outcome, so the model used by DP and the model used by the
simulator are literally the same function. If the two ever disagreed the
whole comparison would be worthless, so this is not a detail: see
``tests/test_dp.py::test_kernel_matches_simulator``.

Only usable on instances small enough to enumerate -- which is exactly the
point Part 6 of the notebook makes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .const import Const
from .dynamics import is_collision, spawn_probability, transition

# Refuse to enumerate something that would exhaust memory. The refusal
# message is itself the lesson, so make it explicit rather than letting the
# machine swap to death.
MAX_ENUMERABLE = 5_000_000


@dataclass
class Kernel:
    """Tabular model of the MDP.

    Attributes
    ----------
    states
        ``(K, dim)`` enumerated state space, in index order.
    next_index
        ``(K, L, B)`` successor index per (state, action, disturbance).
    prob
        ``(K, L, B)`` probability of each disturbance outcome; rows sum to 1.
    terminal
        ``(K,)`` collision states: absorbing, with zero value.
    reward
        ``(L,)`` per-step reward ``1 - lam_u``.
    """

    C: Const
    states: np.ndarray
    next_index: np.ndarray
    prob: np.ndarray
    terminal: np.ndarray
    reward: np.ndarray
    _lookup: np.ndarray = field(repr=False)

    @property
    def K(self) -> int:
        return self.states.shape[0]

    @property
    def L(self) -> int:
        return self.reward.shape[0]

    @property
    def B(self) -> int:
        return self.prob.shape[2]


def _encoder(C: Const):
    """Mixed-radix encoding of states into a dense integer code.

    A dict lookup per successor would dominate build time; a lookup table
    turns it into a single fancy-index.
    """
    M, n_h = C.M, len(C.S_h)
    h_to_idx = np.full(C.Y, -1, dtype=np.int64)
    for i, h in enumerate(C.S_h):
        h_to_idx[h] = i

    radices = [C.Y, 2 * C.V_max + 1] + [C.X] * M + [n_h] * M
    size = int(np.prod(radices))

    def encode(y, v, D, H) -> np.ndarray:
        code = np.asarray(y, dtype=np.int64)
        code = code * (2 * C.V_max + 1) + (np.asarray(v) + C.V_max)
        for j in range(M):
            code = code * C.X + D[:, j]
        for j in range(M):
            code = code * n_h + h_to_idx[H[:, j]]
        return code

    return encode, size


def build_kernel(C: Const) -> Kernel:
    """Enumerate the state space and the exact transition probabilities."""
    est = C.state_space_size_estimate()
    if est > MAX_ENUMERABLE:
        raise MemoryError(
            f"instance {C.name!r} has up to ~{est:.3g} states, so enumeration is "
            "impossible. This is the curse of dimensionality, and it is exactly "
            "why the project turns to policy-gradient methods."
        )

    states = np.asarray(C.state_space, dtype=np.int64)
    K, M = states.shape[0], C.M
    y, v = states[:, 0], states[:, 1]
    D, H = states[:, 2 : 2 + M].copy(), states[:, 2 + M :].copy()

    encode, code_size = _encoder(C)
    lookup = np.full(code_size, -1, dtype=np.int64)
    lookup[encode(y, v, D, H)] = np.arange(K)

    terminal = is_collision(C, y, D[:, 0], H[:, 0])
    live = ~terminal
    if not live.any():
        raise RuntimeError("every enumerated state is terminal")

    # `transition` is only defined on live states. Substitute an arbitrary
    # live state into the terminal rows, then overwrite their successors
    # below: terminal states are absorbing and carry zero value anyway.
    filler = int(np.argmax(live))
    y_in, v_in = y.copy(), v.copy()
    D_in, H_in = D.copy(), H.copy()
    y_in[terminal], v_in[terminal] = y[filler], v[filler]
    D_in[terminal], H_in[terminal] = D[filler], H[filler]

    # Obstacle bookkeeping does not depend on the action, so the spawn
    # probability can be computed once. Passing w_spawn > 1 suppresses the
    # spawn, leaving exactly the pre-spawn queue.
    zeros = np.zeros(K, dtype=np.int64)
    _, _, pre_spawn_D, _ = transition(
        C, y_in, v_in, D_in, H_in, zeros, np.full(K, 2.0), zeros, zeros
    )
    p_spawn = spawn_probability(C, (C.X - 1) - pre_spawn_D.sum(axis=1))

    n_h = len(C.S_h)
    flap_vals = np.arange(-C.V_dev, C.V_dev + 1)
    n_flap = flap_vals.size

    # Obstacle branch: either no spawn, or a spawn with a uniformly chosen
    # gap centre. Flap branch: always enumerated, even for the deterministic
    # inputs -- there the successors simply coincide and the probabilities
    # merge, which keeps the branch count rectangular.
    obstacle_branches = [(np.full(K, 2.0), 0, 1.0 - p_spawn)]
    obstacle_branches += [(np.zeros(K), j, p_spawn / n_h) for j in range(n_h)]

    B = len(obstacle_branches) * n_flap
    next_index = np.zeros((K, C.L, B), dtype=np.int64)
    prob = np.zeros((K, C.L, B), dtype=np.float64)

    for a, u_val in enumerate(C.input_space):
        u = np.full(K, u_val, dtype=np.int64)
        b = 0
        for w_spawn, h_idx, w_obs in obstacle_branches:
            for w_flap in flap_vals:
                y2, v2, D2, H2 = transition(
                    C, y_in, v_in, D_in, H_in, u,
                    w_spawn,
                    np.full(K, h_idx, dtype=np.int64),
                    np.full(K, w_flap, dtype=np.int64),
                )
                idx = lookup[encode(y2, v2, D2, H2)]
                weight = np.broadcast_to(w_obs / n_flap, (K,))

                # Some branches are structurally impossible rather than
                # merely unlikely -- e.g. "no spawn" when the queue is empty
                # and the spawn probability has already saturated at 1. Those
                # land outside the state space, which is fine precisely
                # because they carry zero probability. Anything else is a bug.
                missing = (idx < 0) & live
                if np.any(missing & (weight > 0)):
                    bad = states[np.argmax(missing & (weight > 0))]
                    raise RuntimeError(
                        "the dynamics produced a reachable successor outside the "
                        f"enumerated state space, starting from {bad.tolist()}. "
                        "Either Const.is_valid_state is too strict or the "
                        "transition is wrong; both are bugs."
                    )
                next_index[:, a, b] = np.where(idx < 0, 0, idx)
                prob[:, a, b] = np.where(idx < 0, 0.0, weight)
                b += 1

    # Terminal states absorb into themselves. Their value is pinned to zero,
    # so this only keeps the arrays well formed.
    t_idx = np.nonzero(terminal)[0]
    next_index[t_idx] = t_idx[:, None, None]
    prob[t_idx] = 0.0
    prob[t_idx, :, 0] = 1.0

    total = prob.sum(axis=2)
    if not np.allclose(total, 1.0):
        raise RuntimeError(
            f"transition probabilities do not sum to 1 "
            f"(max deviation {np.abs(total - 1).max():.2e})"
        )

    return Kernel(
        C=C,
        states=states,
        next_index=next_index,
        prob=prob,
        terminal=terminal,
        reward=C.reward_per_action,
        _lookup=lookup,
    )


def _q_values(kernel: Kernel, V: np.ndarray, gamma: float) -> np.ndarray:
    """``Q(s, a) = r(a) + gamma * E[V(s')]``, terminal successors at zero."""
    Vn = np.where(kernel.terminal, 0.0, V)
    exp_next = np.einsum("kab,kab->ka", kernel.prob, Vn[kernel.next_index])
    return kernel.reward[None, :] + gamma * exp_next


def value_iteration(
    kernel: Kernel,
    gamma: float,
    tol: float = 1e-12,
    max_iter: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for ``V*`` and ``pi*``.

    Returns
    -------
    V
        ``(K,)`` optimal value function; terminal states hold ``0``.
    pi
        ``(K,)`` optimal action index; arbitrary (``0``) on terminal states.
    """
    if not 0.0 < gamma < 1.0:
        # Survival earns +1 per step for potentially ever, so the
        # undiscounted problem has unbounded value. gamma is not a tuning
        # knob here -- it is what makes the problem well posed at all.
        raise ValueError(f"gamma must lie strictly in (0, 1), got {gamma}")

    V = np.zeros(kernel.K, dtype=np.float64)
    for _ in range(max_iter):
        Q = _q_values(kernel, V, gamma)
        V_new = np.where(kernel.terminal, 0.0, Q.max(axis=1))
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            break
    else:  # pragma: no cover - only if gamma is pathologically close to 1
        raise RuntimeError(f"value iteration did not converge in {max_iter} sweeps")

    pi = np.where(kernel.terminal, 0, _q_values(kernel, V, gamma).argmax(axis=1))
    return V, pi.astype(np.int64)


def policy_evaluation(
    kernel: Kernel,
    pi: np.ndarray,
    gamma: float,
    tol: float = 1e-12,
    max_iter: int = 200_000,
) -> np.ndarray:
    """Exact value of an arbitrary deterministic policy."""
    V = np.zeros(kernel.K, dtype=np.float64)
    rows = np.arange(kernel.K)
    pi = np.asarray(pi, dtype=np.int64)
    for _ in range(max_iter):
        Q = _q_values(kernel, V, gamma)
        V_new = np.where(kernel.terminal, 0.0, Q[rows, pi])
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            break
    else:  # pragma: no cover
        raise RuntimeError("policy evaluation did not converge")
    return V


def initial_state_indices(kernel: Kernel) -> np.ndarray:
    """Indices of the possible initial states, one per first gap centre."""
    C = kernel.C
    M = C.M
    out = []
    for h in C.S_h:
        D = [C.X - 1] + [0] * (M - 1)
        H = [h] + [C.S_h[0]] * (M - 1)
        out.append(C.state_to_index((C.Y // 2, 0, *D, *H)))
    return np.asarray(out, dtype=np.int64)


def expected_return(kernel: Kernel, V: np.ndarray) -> float:
    """``J = E_{s0}[V(s0)]`` with a uniform initial gap centre.

    This is the scalar the leaderboard's optimality ratio is measured
    against.
    """
    return float(V[initial_state_indices(kernel)].mean())


def state_index_of(kernel: Kernel, state: dict[str, np.ndarray]) -> np.ndarray:
    """Map a batch of simulator states to kernel indices.

    Lets the notebook compare a learned policy against ``pi*`` on the states
    the agent actually visits -- a different, and far more honest, number
    than agreement over the uniform state space.
    """
    encode, _ = _encoder(kernel.C)
    idx = kernel._lookup[encode(state["y"], state["v"], state["d"], state["h"])]
    if np.any(idx < 0):
        raise KeyError("a visited state is absent from the enumerated space")
    return idx
