# Defense Walkthrough — Flappy Bird PPO Project

A talking-points script for a 15-minute defense. Each section lists what to
say, not everything that could be said — use `DOCUMENTATION.md` in this
folder if a question needs more depth than what's here.

Suggested timing: **2 min background/scope → 6 min implementation → 4 min
calibration & tuning → 2 min bonus/results → 1 min buffer for questions**.

---

## 1. Background, scope, goal (~2 min)

**Background.** The project trains an agent to play a discretized Flappy
Bird with Proximal Policy Optimization (PPO), an actor-critic RL algorithm.
The bird has a height `y` and velocity `v`, gravity pulls it down every
step, and it has 3 actions (no flap / weak flap / strong flap-with-noise),
each with a cost subtracted from a `+1` per-step survival reward.

**Two regimes, one reason for PPO.**
- `small` (~7,770 states): small enough to solve **exactly** with value
  iteration on the Bellman equation → gives a ground-truth optimum `(V*,
  π*, J*)` to grade a learned policy against.
- `large` (~10^17 states): the *curse of dimensionality* makes an exact
  table physically impossible (would need many terabytes). This is not
  asserted, it's demonstrated live in the notebook (`MemoryError` when the
  table is actually attempted) — this is the whole justification for using
  a function approximator (a neural net) instead of a lookup table.

**Scope of the work (what was actually mine to build):**
1. Three graded functions inside the PPO loop: `build_observation`,
   `compute_gae`, `ppo_loss`.
2. Three calibration tasks: derive `gamma`, `(T, N)`, and `(gae_lambda, lr,
   update_epochs)` from a *measured* property of the game rather than from
   convention.
3. A 20-run one-factor-at-a-time tuning sweep over `Config.for_large()`.
4. A bonus: domain randomization over the strong-flap noise `V_dev`.
5. A final hand-in config assembled from the calibration + sweep evidence.

**Goal.** Not "get a high score" in isolation — the goal is to show that
every design decision (feature scaling, credit assignment, the training
objective, and every hyperparameter) is *justified from RL theory and from
measurements of this specific environment*, not copied from a default or
picked by feel.

---

## 2. The three implemented functions — what and why (~6 min)

### 2.1 `build_observation` — turning game state into network input

**What:** centers and scales `y` and `v`, and for each tracked obstacle
passes its scaled distance `dx`, scaled gap-offset `dy`, and a binary
`active` flag for whether that obstacle slot is real or a placeholder.

**Theory:**
- Neural nets train far better on inputs that are roughly zero-centered
  and unit-scale — unscaled raw state (`y` in `[0,13]` vs. distances in
  different ranges) makes gradient steps behave inconsistently across
  dimensions.
- **Fixed** scales (not running statistics computed during training) are
  required because adaptive normalization would make the meaning of the
  network's input non-stationary — the same physical situation would map
  to different numbers depending on training progress, and a saved
  checkpoint would become meaningless without also saving that history.
- The `active` flag exists because empty obstacle slots are padded with a
  far-away placeholder distance; without a flag, "no obstacle" and "a real
  obstacle very far away" are indistinguishable to the network.

### 2.2 `compute_gae` — credit assignment

**What:** implements Generalized Advantage Estimation: a backward pass
computing the TD-error `delta_t = r_t + γ·V(s_{t+1})·(1-done_t) − V(s_t)`,
then accumulating it as an exponentially-decayed running sum `A_t = δ_t +
γλ·(1-done_t)·A_{t+1}`, with returns for the critic given by `A + V`.

**Theory:**
- This is the bias/variance trade-off at the center of actor-critic
  methods: pure Monte-Carlo returns (`λ=1`) are unbiased but high-variance;
  bootstrapping only one step (`λ=0`) is low-variance but biased by
  whatever the critic currently believes. `λ` interpolates between the two.
- The `(1 - done_t)` mask is the detail that actually matters in practice:
  without it, credit/blame would leak backward across an episode boundary
  into an unrelated next episode. Verified with a unit test that forces a
  `done` and checks zero leakage across it.
- Correctness is also verified against a known closed form: with `V≡0` and
  `λ=1`, GAE must reduce exactly to the plain discounted Monte-Carlo return.

### 2.3 `ppo_loss` — the training objective

**What:** the clipped surrogate objective from Schulman et al. (2017):
`ratio = exp(logπ_new − logπ_old)`, `L = mean(max(-A·ratio, -A·clip(ratio,
1±ε)))`, plus a value-function MSE term and an entropy bonus, combined as
`loss = pg_loss + vf_coef·v_loss − ent_coef·entropy`. `approx_kl` is
computed with the low-variance, non-negative Schulman estimator `(ratio −
1) − log(ratio)` rather than the naive `−log(ratio)`.

**Theory:**
- PPO reuses the same rollout for several gradient epochs, so the ratio
  between the current and the rollout-time policy drifts away from 1 as
  training proceeds within an update. The clip term is what keeps this
  "proximal" — it caps how much a single update is allowed to trust a
  large ratio, preventing destructively large policy shifts (unlike vanilla
  policy gradient, which has no such safeguard).
- Taking `max` (not `min`) of the two pessimistic terms is correct here
  because both terms already carry the sign that turns "maximize return"
  into "minimize loss" — this is the classic sign trap when porting the
  paper's formula.
- The entropy bonus is not decorative: it is what keeps early exploration
  alive so the agent doesn't collapse into the "never flap" trap (see §3).
- `approx_kl` is the diagnostic instrument used later to tune `lr` and
  `update_epochs` empirically instead of by feel (§3).

---

## 3. Calibration and tuning — from measurement to hyperparameter (~4 min)

**Guiding principle:** every hyperparameter is derived from `Delta`, the
measured average step-spacing between pipes — not copied from a paper.

- **`gamma = 0.99`.** Effective horizon `1/(1-γ)` must exceed `Delta`
  (otherwise the agent can't connect "flap now" to "pipe cleared later"),
  but not explode the value scale the critic has to learn (`1/(1-γ)` also
  bounds the magnitude of `V`). `0.99` gives a 100-step horizon, comfortably
  above `Delta ≈ 14` on `large` without destabilizing the critic.
- **`T=32, N=256`.** `T` (rollout length) must be long enough (`≳2×Delta`)
  that a full "climb → pass pipe" story fits inside one recorded window —
  GAE cannot assign credit across rollout boundaries. `N` (parallel envs)
  controls gradient noise, which scales like `1/√(N·T)`, independent of `T`.
  These are two different levers on two different failure modes, not one
  interchangeable "batch size" knob.
- **`gae_lambda=0.95` (initial), lr/epochs via `approx_kl`.** GAE's own
  *effective* horizon is `1/(1-γλ)`, targeted into `[Delta, 2·Delta]`;
  `λ=0.95` lands there for this game — a nice case where a "default
  everyone uses" turns out to be independently derivable, not just cargo
  culted. `lr` and `update_epochs` are tuned by watching `approx_kl` land in
  `[0.01, 0.02]` (too low → wasted rollout data; too high → stale,
  off-policy updates) rather than by watching the score directly — the KL
  metric flagged an undertrained default (`lr=3e-4, epochs=4` → 48% of
  optimum, KL ~20x too low) before the learning curve made it obvious.

**20-run one-factor-at-a-time sweep.** Rules: change one knob at a time,
predict the direction before running, re-run promising configs on 3 seeds
(report mean ± SE) to avoid mistaking seed luck for a real effect. Headline
finding, not predicted in advance: being *too conservative* (small `lr`,
few epochs, tight clipping, low entropy, low `vf_coef`) hurt more reliably
than being too aggressive — several "aggressive" settings
(`update_epochs=20`, `ent_coef=0.1`) beat baseline outright. This matters
for the defense because it's evidence the tuning wasn't just confirming
what the theory already said — the sweep surfaced a genuine, unpredicted
asymmetry that theory alone did not fully anticipate.

**Final config** = calibrated base (`γ=0.99`, `T=32`, `N=256`, `lr=0.003`)
with each individual knob swapped for its best-performing sweep value:
`gae_lambda=0.90`, `update_epochs=20`, `ent_coef=0.1`, `clip_coef=0.3`. This
is deliberately *not* a full grid search (ruled out by the 20-run budget) —
it's the best evidence-based combination obtainable one axis at a time, and
that trade-off is worth naming explicitly if asked "why not search jointly?"

---

## 4. Bonus: domain randomization (~1 min, if time allows)

Randomizes the strong-flap noise `V_dev ∈ {0,1,2}` per episode (drawn from
each episode's own seeded RNG, so runs stay reproducible) instead of
holding it fixed. Theoretical point: a policy trained on one fixed dynamics
instance can overfit to that instance's specific noise profile; sampling a
distribution over dynamics during training is the standard sim-to-real
technique for producing a policy that is robust across a family of similar
environments, at the cost of some peak performance on any single one of
them.

---

## 5. Anticipated questions

- **Why `max` and not `min` in the clipped objective?** Both terms already
  carry the negation that converts "maximize" into "minimize" — `max` of
  the negated terms is `min` of the original, unnegated ones. Getting this
  backward is the single easiest sign bug in a PPO implementation.
- **Why does `(1-done)` appear twice in GAE?** Once to zero out the
  bootstrapped next-value at a terminal step, once to stop the backward
  running sum from crossing into the previous (unrelated) episode.
- **Why not just maximize `J/J*` directly instead of tuning via `approx_kl`
  or `Delta`?** Because on `large` there is no `J*` to compare against —
  the calibration has to generalize from theory/measurement, and `small`'s
  ground truth was only used to *validate* the approach, not to hand-tune
  `large`.
- **Why did the sweep sometimes contradict the prediction?** Because
  single-seed runs are noisy and some knobs (rollout/batch split,
  `gae_lambda`) have a non-monotonic sweet spot rather than a "more is
  always better/worse" relationship — this is exactly why the protocol
  required 3-seed reruns with standard error before trusting a result.
