"""Actor-critic network: a small MLP with a categorical head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate policy and value trunks, ``2 x hidden`` with ``tanh``.

    Parameters
    ----------
    init_action_probs
        Action distribution the policy starts from. This is not cosmetic.
        With a symmetric initialisation the agent flaps on two thirds of the
        frames, pins itself to the ceiling, dies within a handful of steps
        and produces almost no learning signal for the first tens of
        thousands of steps. Biasing the output layer towards the natural
        duty cycle removes that pathological attractor by construction
        rather than hoping the optimiser escapes it.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 64,
        init_action_probs: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        actor_out = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        if init_action_probs is not None:
            p = np.asarray(init_action_probs, dtype=np.float64)
            if p.shape != (n_actions,) or p.min() <= 0:
                raise ValueError("init_action_probs must be positive, one per action")
            logits = np.log(p / p.sum())
            with torch.no_grad():
                actor_out.bias.copy_(torch.as_tensor(logits - logits.mean(), dtype=torch.float32))
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            actor_out,
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(obs))

    def act(
        self, obs: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, log_prob, entropy, value)``.

        ``deterministic`` takes the argmax, which is what evaluation uses:
        combined with a seed-determined environment it makes every reported
        number reproducible bit for bit.
        """
        dist = self.distribution(obs)
        if action is None:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(obs)
