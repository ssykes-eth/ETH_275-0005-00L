"""Student-facing hooks used by the PPO loop.

Setup only ships the two *provided* helpers below. The three graded pieces
(``build_observation``, ``compute_gae``, ``ppo_loss``) are implemented in
Part 2 of the notebook and written back into this module by the export cell
there — so you never have to dig through Setup to find the TODOs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .const import Const

__all__ = [
    "build_observation",
    "shaped_reward",
    "bootstrap_truncated",
    "compute_gae",
    "ppo_loss",
]


def shaped_reward(
    state: dict[str, np.ndarray],
    action: np.ndarray,
    reward: np.ndarray,
    next_state: dict[str, np.ndarray],
    terminated: np.ndarray,
    truncated: np.ndarray,
    C: Const,
    cfg: Any,
) -> np.ndarray:
    """Reward actually optimised during training.

    The baseline is the native reward ``1 - lam_u``: survive, but pay for
    flapping. Students add terms on top.

    Two traps are worth walking into deliberately:

    * A large terminal penalty combined with the effort cost can make dying
      early *optimal* under the shaped reward -- the agent stops paying
      ``lam_u`` sooner. Watch the flap rate collapse to zero.
    * Any bonus for hovering near the gap centre competes with actually
      passing through it.

    Whatever is done here, the leaderboard still ranks on pipes passed. The
    shaped reward is a means, never the score.
    """
    r = reward.astype(np.float64).copy()
    if cfg.death_penalty:
        r -= cfg.death_penalty * terminated
    if cfg.align_bonus:
        # Negative distance to the next gap centre, normalised by the grid.
        dy = np.abs(next_state["dy"][:, 0]) / ((C.Y - 1) / 2.0)
        r -= cfg.align_bonus * dy
    return r


def bootstrap_truncated(
    reward: np.ndarray,
    truncated: np.ndarray,
    final_value: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Fold the value of a *truncated* final state back into its reward.

    Termination and truncation are not the same event and must not be
    handled the same way. A crash genuinely ends the future: its value is
    zero. Hitting the step cap does not -- the episode was still worth
    something, it was merely cut short by us. Treating truncation as
    termination teaches the critic that surviving to the horizon is
    worthless, and the agent duly learns to die just before it.

    Adding ``gamma * V(s_final)`` to the reward of the truncated step lets
    the rest of the pipeline mask both cases identically with ``done``.
    """
    return reward + gamma * final_value * truncated


# ======================================================================
# Graded in Part 2 — stubs until that section exports the real bodies
# ======================================================================
def build_observation(state: dict[str, np.ndarray], C: Const, cfg: Any) -> np.ndarray:
    raise NotImplementedError(
        "Implement build_observation in Part 2, then run the export cell."
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "Implement compute_gae in Part 2, then run the export cell."
    )


def ppo_loss(
    old_logprob: torch.Tensor,
    new_logprob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_value: torch.Tensor,
    entropy: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    raise NotImplementedError(
        "Implement ppo_loss in Part 2, then run the export cell."
    )
