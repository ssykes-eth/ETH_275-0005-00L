"""PPO training loop.

The loop itself is provided; the pieces students write live in
:mod:`flappy.student_impl` and are called from here, so the notebook, the
tests and the leaderboard all exercise exactly the same code.

Everything is CPU-first. On the ``small`` instance a full run is a matter of
seconds, which is the whole point: the guided calibration exercises in the
notebook are only honest if a student can afford to run three seeds per
hypothesis.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import student_impl
from .const import Const
from .env import VecFlappy
from .features import obs_dim as _obs_dim
from .networks import ActorCritic

# The five functions a submission must provide. The training loop takes them
# as an argument rather than importing them, so a notebook can hand over the
# cells a student just wrote without touching the package on disk.
REQUIRED_IMPL = (
    "build_observation",
    "shaped_reward",
    "bootstrap_truncated",
    "compute_gae",
    "ppo_loss",
)


def resolve_impl(impl: Any | None):
    """Validate and return the implementation bundle used by :func:`train`."""
    impl = student_impl if impl is None else impl
    missing = [n for n in REQUIRED_IMPL if not callable(getattr(impl, n, None))]
    if missing:
        raise AttributeError(
            "the implementation bundle is missing "
            + ", ".join(missing)
            + f". It must provide all of: {', '.join(REQUIRED_IMPL)}"
        )
    return impl


@dataclass
class Config:
    """Every knob in one place; nothing is read from a global."""

    instance: str = "small"

    # --- data collection ------------------------------------------------
    num_envs: int = 128  # C2: gradient noise ~ 1/sqrt(N*T)
    rollout: int = 64  # C2: must exceed 2*Delta = 28
    total_steps: int = 2_000_000

    # --- objective ------------------------------------------------------
    gamma: float = 0.99  # C1: 1/(1-g) = 100 steps, ~7 obstacles ahead
    gae_lambda: float = 0.95  # C3: credit horizon 16.8 vs Delta = 14.1
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01  # C4
    norm_adv: bool = True

    # --- optimiser ------------------------------------------------------
    # lr and update_epochs were calibrated with C5, not copied from a
    # reference implementation: at the usual 3e-4 / 4 the measured
    # approx-KL is ~5e-4, twenty times below the useful band, and the
    # agent stalls at 48% of the DP optimum instead of 94%.
    lr: float = 3e-3
    anneal_lr: bool = True
    update_epochs: int = 10
    minibatch_size: int = 2048
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # --- model ----------------------------------------------------------
    hidden: int = 64
    obs_mode: str = "full"
    n_preview: int = 2
    init_action_probs: tuple[float, ...] = (0.75, 0.15, 0.10)

    # --- reward shaping (TODO 2) ----------------------------------------
    # On `small` the native reward suffices; PPO reaches 94% of the DP
    # optimum with these at zero. On `large` it does not: "never flap"
    # collects 31 steps of free survival before the first obstacle and the
    # native reward collapses into it every time. Entropy alone escapes on
    # roughly half the seeds; an alignment term makes it reliable. This is
    # what makes TODO 2 load-bearing rather than decorative -- see
    # Config.for_large().
    death_penalty: float = 0.0
    align_bonus: float = 0.0

    # --- bookkeeping ----------------------------------------------------
    seed: int = 0
    device: str = "cpu"
    log_every: int = 10

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout

    @property
    def num_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)

    def const(self) -> Const:
        return {"small": Const.small, "large": Const.large}[self.instance]()

    @classmethod
    def for_large(cls, **overrides) -> "Config":
        """Reference configuration for the ``large`` instance.

        Measured over 2 seeds at 3M env steps: ``41.8 +- 7.5`` pipes,
        ``637`` steps, against ``13.4`` pipes for the lookahead controller.
        Dropping ``align_bonus`` takes it to ``12.0 +- 12.0`` -- escape on
        one seed, collapse on the other -- and dropping ``ent_coef`` to the
        default takes it to ``0.01``.
        """
        kw = dict(instance="large", total_steps=3_000_000,
                  ent_coef=0.05, align_bonus=0.3)
        kw.update(overrides)
        return cls(**kw)

    def validate(self) -> "Config":
        """Reject a configuration that cannot train, with a readable reason.

        Mostly this catches a hyper-parameter slot left at ``None`` in a
        notebook: failing here beats failing three layers down inside the
        optimiser with a shape error.
        """
        checks = [
            ("gamma", lambda v: isinstance(v, (int, float)) and 0 < v < 1,
             "must lie strictly in (0, 1); survival pays forever, so the "
             "undiscounted problem has unbounded value"),
            ("gae_lambda", lambda v: isinstance(v, (int, float)) and 0 <= v <= 1,
             "must lie in [0, 1]"),
            ("lr", lambda v: isinstance(v, (int, float)) and v > 0, "must be positive"),
            ("update_epochs", lambda v: isinstance(v, int) and v > 0,
             "must be a positive integer"),
            ("rollout", lambda v: isinstance(v, int) and v > 0,
             "must be a positive integer"),
            ("num_envs", lambda v: isinstance(v, int) and v > 0,
             "must be a positive integer"),
            ("minibatch_size", lambda v: isinstance(v, int) and v > 0,
             "must be a positive integer"),
            ("total_steps", lambda v: isinstance(v, int) and v > 0,
             "must be a positive integer"),
        ]
        for name, ok, why in checks:
            value = getattr(self, name)
            if not ok(value):
                raise ValueError(f"Config.{name} = {value!r}: {why}")
        if self.minibatch_size > self.batch_size:
            raise ValueError(
                f"Config.minibatch_size ({self.minibatch_size}) exceeds the batch "
                f"({self.num_envs} x {self.rollout} = {self.batch_size})"
            )
        return self

    def credit_horizon(self) -> float:
        """``1 / (1 - gamma * lam)``: how far GAE actually propagates credit."""
        return 1.0 / (1.0 - self.gamma * self.gae_lambda)

    def effective_horizon(self) -> float:
        """``1 / (1 - gamma)``: the discount's own horizon."""
        return 1.0 / (1.0 - self.gamma)


@dataclass
class TrainingLog:
    """The honesty artefact submitted alongside a checkpoint."""

    config: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)
    env_steps: int = 0
    wall_time: float = 0.0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def train(
    cfg: Config, impl: Any | None = None, verbose: bool = True
) -> tuple[ActorCritic, TrainingLog]:
    """Run PPO and return the trained network plus its log.

    Parameters
    ----------
    impl
        Any object exposing the five functions in :data:`REQUIRED_IMPL`.
        Defaults to :mod:`flappy.student_impl`; the notebook passes a
        namespace built from the cells the student just wrote.
    """
    impl = resolve_impl(impl)
    cfg.validate()
    C = cfg.const()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    env = VecFlappy(C, cfg.num_envs, seed=cfg.seed, auto_reset=True)
    dim = _obs_dim(C, cfg.obs_mode, cfg.n_preview)
    net = ActorCritic(dim, C.L, cfg.hidden, cfg.init_action_probs).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.rollout, cfg.num_envs
    b_obs = np.zeros((T, N, dim), dtype=np.float32)
    b_act = np.zeros((T, N), dtype=np.int64)
    b_logp = np.zeros((T, N), dtype=np.float64)
    b_rew = np.zeros((T, N), dtype=np.float64)
    b_done = np.zeros((T, N), dtype=np.float64)
    b_val = np.zeros((T, N), dtype=np.float64)

    state = env.reset()
    obs = impl.build_observation(state, C, cfg)
    log = TrainingLog(config=asdict(cfg))
    env_steps = 0
    started = time.time()

    # Episode statistics are collected as episodes finish, not per step.
    recent: dict[str, list[float]] = {"return": [], "length": [], "pipes": []}

    for update in range(1, cfg.num_updates + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1) / cfg.num_updates
            opt.param_groups[0]["lr"] = frac * cfg.lr

        flaps = 0
        for t in range(T):
            b_obs[t] = obs
            with torch.no_grad():
                ot = torch.as_tensor(obs, device=device)
                action, logp, _, value = net.act(ot)
            a = action.cpu().numpy()
            b_act[t], b_logp[t], b_val[t] = a, logp.cpu().numpy(), value.cpu().numpy()

            next_state, reward, terminated, truncated, info = env.step(a)
            env_steps += N
            flaps += int((a > 0).sum())

            r = impl.shaped_reward(
                state, a, reward, info["final_state"], terminated, truncated, C, cfg
            )

            # --- terminal handling (TODO 3) -----------------------------
            # info["final_state"] is the pre-reset state, so a truncated
            # episode can still be bootstrapped from where it really stopped.
            if truncated.any():
                with torch.no_grad():
                    final_obs = impl.build_observation(info["final_state"], C, cfg)
                    final_v = net.value(torch.as_tensor(final_obs, device=device))
                r = impl.bootstrap_truncated(
                    r, truncated.astype(np.float64), final_v.cpu().numpy(), cfg.gamma
                )

            b_rew[t] = r
            b_done[t] = (terminated | truncated).astype(np.float64)

            done = info["done"]
            if done.any():
                recent["return"].append(float(info["episode_return"][done].mean()))
                recent["length"].append(float(info["episode_length"][done].mean()))
                recent["pipes"].append(float(info["episode_pipes"][done].mean()))

            state = next_state
            obs = impl.build_observation(state, C, cfg)

        with torch.no_grad():
            last_value = net.value(torch.as_tensor(obs, device=device)).cpu().numpy()

        advantages, returns = impl.compute_gae(
            b_rew, b_val, b_done, last_value, cfg.gamma, cfg.gae_lambda
        )

        # --- update ------------------------------------------------------
        f_obs = torch.as_tensor(b_obs.reshape(-1, dim), device=device)
        f_act = torch.as_tensor(b_act.reshape(-1), device=device)
        f_logp = torch.as_tensor(b_logp.reshape(-1), dtype=torch.float32, device=device)
        f_adv = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        f_ret = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        f_val = torch.as_tensor(b_val.reshape(-1), dtype=torch.float32, device=device)

        idx = np.arange(cfg.batch_size)
        metrics: dict[str, float] = {}
        stop = False
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                _, new_logp, entropy, new_value = net.act(f_obs[mb], f_act[mb])
                loss, metrics = impl.ppo_loss(
                    f_logp[mb], new_logp, f_adv[mb], f_ret[mb], new_value, entropy, cfg
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
            if cfg.target_kl is not None and metrics["approx_kl"] > cfg.target_kl:
                # Early stopping on KL: the cheapest guard against an update
                # that walks so far off-policy the collected data is stale.
                stop = True
                break

        y_pred, y_true = f_val.cpu().numpy(), f_ret.cpu().numpy()
        var = float(np.var(y_true))
        explained_var = float("nan") if var == 0 else 1 - float(np.var(y_true - y_pred)) / var

        row = {
            "update": float(update),
            "env_steps": float(env_steps),
            "lr": float(opt.param_groups[0]["lr"]),
            "flap_rate": flaps / float(T * N),
            "explained_variance": explained_var,
            "kl_early_stop": float(stop),
            **metrics,
        }
        for k, vals in recent.items():
            row[f"ep_{k}"] = float(np.mean(vals[-50:])) if vals else float("nan")
        log.history.append(row)

        if verbose and (update % cfg.log_every == 0 or update == cfg.num_updates):
            print(
                f"upd {update:4d}/{cfg.num_updates}  steps {env_steps:8d}  "
                f"ret {row['ep_return']:7.2f}  len {row['ep_length']:6.1f}  "
                f"pipes {row['ep_pipes']:6.2f}  flap {row['flap_rate']:.3f}  "
                f"KL {row['approx_kl']:.4f}  ev {explained_var:+.3f}"
            )

    log.env_steps = env_steps
    log.wall_time = time.time() - started
    return net, log


class NetworkPolicy:
    """Adapts a trained network to the evaluation/leaderboard interface."""

    def __init__(
        self,
        net: ActorCritic,
        C: Const,
        cfg: Config,
        deterministic: bool = True,
        impl: Any | None = None,
    ) -> None:
        self.net = net
        self.C = C
        self.cfg = cfg
        self.deterministic = deterministic
        # Must be the same observation builder the network was trained with,
        # or the evaluated policy is simply a different function.
        self.impl = resolve_impl(impl)

    def reset(self) -> None:
        pass

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        obs = self.impl.build_observation(state, self.C, self.cfg)
        with torch.no_grad():
            action, *_ = self.net.act(
                torch.as_tensor(obs), deterministic=self.deterministic
            )
        return action.numpy()
