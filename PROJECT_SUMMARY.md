# Project Summary

This repository holds the graded projects for ETH course **275-0005-00L, "AI Workshop —
From Data to Solutions."** Each project folder is self-contained and has its own
`DOCUMENTATION.md` (full detail, cell/function by cell/function) and `DEFENSE.md` (a
condensed 15-minute oral-defense script). **This file is the detailed overview** for
someone who has never opened any of the four projects: what each one is for, the theory
it rests on, and what was actually built or changed.

Four projects carry full documentation: **Project 1** (regression/ML pipeline),
**Project 3** (reinforcement learning), **Project 5** (retrieval-augmented generation),
and **Project 6** (an agentic system built on top of Project 5's RAG). `code_exercises/`
holds smaller, ungraded practice notebooks with no formal write-up. A former "Project 2"
folder was intentionally dropped from the repository and is not covered here.

For **Projects 3, 5, and 6** — the three notebook-based projects — this file lists
**every notebook cell that contained a `🎯`/`TODO` exercise blank**, in cell order, with
a one- or two-line summary of what was actually filled in. Project 1 has no notebook
(it's a plain Python pipeline), so its "applied changes" are described at the
pipeline-stage level instead.

Cell numbers below are 0-indexed positions in the `.ipynb` file (as `nbformat`/Jupyter
count them), not the "In [ ]:" execution-order numbers shown in a UI.

---

## Project 1 — Brain Age Prediction (supervised regression)

**Scope & goal.** Predict a person's age from 832 anonymized numeric features per
subject, using a labeled training set and an unlabeled Kaggle test set. The features
carry only a weak, diffuse signal (no single feature correlates with age above 0.44),
so the goal is to combine many weak clues carefully rather than find one strong
predictor — and to prove, with measurements, that every pipeline choice actually helps.

**Theory applied, in plain terms.**
- **Train/validation/test discipline:** statistics used to clean or transform data
  (medians for filling gaps, "what counts as normal" ranges, scalers) must be learned
  only from training data and re-applied — never re-learned — on validation/test data.
  Otherwise information "leaks" backward and the reported score becomes dishonestly
  optimistic.
- **Cross-validation for model selection:** instead of judging a model on one lucky (or
  unlucky) held-out slice, the data is split into 5 folds and every candidate is scored
  as the *average* over all 5 — a far more stable way to compare models than a single
  train/test split.
- **Bias–variance trade-off in feature selection:** a univariate statistical test
  (F-test) that only asks "does this one feature line up with age on its own" beat a
  fancier, interaction-aware method (Boruta), because the final model (SVR) itself only
  rewards that kind of direct, per-feature relationship.
- **Ensembling / stacking:** training several different kinds of models (a straight-line
  model, tree ensembles, a kernel method, a Gaussian process) and blending their outputs
  usually beats any single one, because their mistakes are only partly correlated.
- **The "one-standard-error rule":** when two candidate models are statistically tied,
  prefer the more stable one (the blended team) over the technically-highest-scoring one
  — a classical statistics safeguard against picking a model that only "won" due to
  noise.

**Applied changes, by pipeline stage** (each stage is its own module + its own test file):

| Stage (module) | What it does |
|---|---|
| `split.py` | Holds out 10% of training data as a never-peeked-at validation set, before any other stage runs. |
| `outliers.py: remove_cell_outliers` | Fixes single corrupted *cell* values via per-column IQR bounds (`k=2.5`), applied to train/val/test alike. |
| `impute.py` | Fills missing values with each column's *median*, learned from training data only. |
| `outliers.py: remove_outliers` | Drops whole anomalous *rows* from the training set only (`IsolationForest`, `contamination="auto"`). |
| `feature_selection.py` | Drops near-duplicate features (correlation > 0.9, keeping the more age-relevant twin), then keeps the top 100 by an F-test. |
| `scale.py` | Rescales all selected features with a rank-based `QuantileTransformer`. |
| `regression.py: select_best_model` | Compares ~38 model configs across 6 families (Ridge, RandomForest, HistGradientBoosting, CatBoost, SVR, GPR) via 5-fold cross-validation, blends the best of each family (stacking), applies the one-standard-error rule, and refits on train+validation before producing the final model. |

Several plausible alternatives (KNN-based imputation, Boruta feature selection,
`RobustScaler`, age-stratified folds, XGBoost, ElasticNet, ExtraTrees) were tried and
measured to be worse, and deliberately left out — a documented "what we tried and
rejected" record, not just "what we kept."

---

## Project 3 — Flappy Bird with PPO (reinforcement learning)

**Scope & goal.** Train an agent to play a simplified, discretized Flappy Bird using
**Proximal Policy Optimization (PPO)**, an actor–critic reinforcement-learning
algorithm. A small version of the game (~7,770 states) can be solved *exactly* with
classical dynamic programming, giving a ground-truth optimum to grade the learned agent
against; a large version (~10^17 states) cannot — demonstrated live by actually running
out of memory trying — which is the concrete justification for switching to a neural
network that generalizes across states instead of memorizing a lookup table.

**Theory applied, in plain terms.**
- **Reinforcement learning basics:** an agent takes actions in an environment, receives
  a reward, and must learn a *policy* (what to do in each situation) that maximizes
  total future reward, discounted by a factor `γ` for how much it values the future
  versus the present.
- **Actor–critic methods & credit assignment (GAE):** one network proposes actions (the
  actor), another estimates how good a situation is (the critic). Generalized Advantage
  Estimation blends "trust what actually happened" (high variance, unbiased) with "trust
  the critic's current guess" (low variance, biased) using a tunable knob `λ`.
- **PPO's clipped objective:** naive policy-gradient updates can be pushed arbitrarily
  far by a single batch of experience, destabilizing training. PPO clips how much the
  policy is allowed to change in one update step — "proximal" means "stay close to what
  you started the update with."
- **Deriving hyperparameters from measurement, not convention:** every hyperparameter
  (discount factor, rollout length, GAE's `λ`, learning rate) is calculated from an
  actually-measured property of the game (the average spacing between obstacles) rather
  than copied from a textbook default.
- **Domain randomization:** training against a *distribution* of slightly different game
  dynamics (instead of one fixed version) produces a policy that generalizes better
  across variations, at some cost to peak performance on any single variation — the
  standard technique bridging simulation and reality in robotics.

**Every notebook cell with a 🎯/TODO exercise blank**, with the exact applied code:

#### Cell 26 — Task 1 — `_build_observation`
Re-centers/scales the bird's height and velocity onto fixed, hand-picked scales (not
adaptive statistics, so a saved checkpoint stays meaningful); for each tracked obstacle,
appends its scaled distance, scaled gap-offset, and a binary `active` flag so an empty
obstacle slot is never confused with a real, distant one.

```python
# 🎯 TASK 1 — fill in the five ??? below.
def _build_observation(state: dict[str, np.ndarray], C: Const, cfg: Any) -> np.ndarray:
    """Map the ground-truth state to the network input.

    Parameters
    ----------
    state
        The dict returned by :meth:`flappy.env.VecFlappy.state_dict`:
        ``y (N,)``, ``v (N,)``, ``d (N, M)``, ``h (N, M)``, plus the derived
        ``dx (N, M)`` absolute distances, ``dy (N, M)`` gap offsets
        ``h_i - y``, and ``active (N, M)``. Empty obstacle slots report
        ``dx = X`` and ``dy = 0``, so carry ``active`` if you use them --
        otherwise an empty slot is indistinguishable from a genuinely
        distant obstacle.
    cfg
        Carries ``obs_mode`` and ``n_preview``. Whatever you return here
        must have the width that :func:`flappy.features.obs_dim` reports for
        that config, or the network will be built with the wrong input size.

    Returns
    -------
    ``(N, obs_dim)`` float32.

    Notes
    -----
    Use the fixed constants in :func:`flappy.features.scales` and nothing
    else. A running normaliser would couple every run to its own history and
    make checkpoints unportable -- it is a silent-bug factory that teaches
    nothing about control.

    The ``"minimal"`` baseline sees only ``(v, dy_0)``. It almost works.
    Run it, watch where it fails, then decide how much preview to add: that
    experiment is the exercise, not the code.
    """
    s = scales(C)
    # 🎯 the bird height, re-centred on mid-grid ((C.Y - 1) / 2) then scaled by s["y"]
    y = (state["y"] - (C.Y - 1) / 2) / s["y"]
    # 🎯 the vertical velocity, scaled by s["v"]
    v = state["v"] / s["v"]

    if cfg.obs_mode == "minimal":
        dy0 = state["dy"][:, 0] / s["dy"]
        return np.stack([v, dy0], axis=1).astype(np.float32)

    if cfg.obs_mode != "full":
        raise ValueError(f"unknown observation mode {cfg.obs_mode!r}")

    P = min(cfg.n_preview, C.M)
    cols = [y, v]
    for i in range(P):
        # 🎯 obstacle i's absolute distance dx, scaled by s["dx"]
        cols.append(state["dx"][:, i] / s["dx"])
        # 🎯 obstacle i's gap offset dy, scaled by s["dy"]
        cols.append(state["dy"][:, i] / s["dy"])
        # 🎯 the activity flag as a float. It matters: without it an empty slot's
        #    sentinel distance is indistinguishable from a genuinely far obstacle.
        cols.append(state["active"][:, i].astype(np.float32))
    return np.stack(cols, axis=1).astype(np.float32)

```

#### Cell 28 — Task 2 — `_compute_gae`
Implements the backward-pass Generalized Advantage Estimation recurrence: the TD-error
`delta_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)`, then the exponentially-decayed
running sum `A_t = δ_t + γλ·(1−done_t)·A_{t+1}`, with the `(1−done)` mask stopping credit
from leaking across an episode boundary; critic targets are `advantages + values`.

```python
# 🎯 TASK 2 — fill in the three ??? below.
def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) over a truncated rollout.

    Parameters
    ----------
    rewards, values, dones
        ``(T, N)``. ``dones[t]`` marks that the episode ended *on* step
        ``t``, so step ``t + 1`` belongs to a new episode.
    last_value
        ``(N,)`` value of the state following the final step.

    Returns
    -------
    advantages, returns
        Both ``(T, N)``.

    Notes
    -----
    ``delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)``
    ``A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}``

    The credit horizon is ``1 / (1 - gamma * lam)`` steps; calibration
    exercise C3 in the notebook asks for that number explicitly and
    compares it against the measured spacing between obstacles.
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float64)
    last_gae = np.zeros(N, dtype=np.float64)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        # 🎯 the TD error: this step's reward plus the discounted
        #                   next_value, masked by non_terminal, minus values[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        # 🎯 the accumulation: delta plus gamma * lam * non_terminal
        #                       times the last_gae carried back from step t + 1
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    # 🎯 the returns the critic regresses on = advantages + values
    return advantages, advantages + values
```

#### Cell 30 — Task 3 — `_ppo_loss`
Computes the probability ratio `exp(logπ_new − logπ_old)`, the clipped surrogate
objective (`max` of the unclipped and clipped pessimistic terms — deliberately `max`,
since both terms already carry the sign flip from "maximize" to "minimize"), the
value-function MSE loss, the combined loss with an entropy bonus, and the low-variance
Schulman `approx_kl` diagnostic used later for tuning.

```python
# 🎯 TASK 3 — fill in the six ??? below.
def _ppo_loss(
    old_logprob: torch.Tensor,
    new_logprob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    new_value: torch.Tensor,
    entropy: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate + value loss + entropy bonus.

    All inputs are flat ``(B,)`` tensors over one minibatch.

    Returns
    -------
    loss, metrics
        ``metrics`` carries the diagnostics the notebook's calibration
        exercises are read off: ``approx_kl`` drives the epoch count in C5,
        ``entropy`` and the flap rate drive the entropy coefficient in C4.
    """
    log_ratio = new_logprob - old_logprob
    # 🎯 the probability ratio rho = exp(log pi_new - log pi_old)
    ratio = torch.exp(log_ratio)

    if cfg.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_1 = -advantages * ratio
    # 🎯 the same product, but with ratio clamped to
    #    [1 - cfg.clip_coef, 1 + cfg.clip_coef]
    pg_2 = -advantages * torch.clamp(
        ratio,
        1 - cfg.clip_coef,
        1 + cfg.clip_coef,
    )

    # 🎯 the pessimistic branch of the two, averaged over the batch
    pg_loss = torch.max(pg_1, pg_2).mean()
    # 🎯 half the mean squared error between new_value and returns
    v_loss = 0.5 * ((new_value - returns) ** 2).mean()
    ent = entropy.mean()
    # 🎯 policy loss + cfg.vf_coef * value loss - cfg.ent_coef * entropy
    loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

    with torch.no_grad():
        # 🎯 Schulman's low-variance KL estimator, mean of (rho - 1) - log rho.
        #    Always non-negative, unlike the naive -log_ratio.mean(). Call .item()
        #    on it so the metrics dict holds plain floats.
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()

    return loss, {
        "policy_loss": pg_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": ent.item(),
        "approx_kl": approx_kl,
        "clip_fraction": clip_frac,
    }
```

#### Cell 35 — Calibration C1 — discount factor
Chosen so the effective horizon `1/(1−γ)` (≈100 steps) comfortably exceeds the measured
average spacing between obstacles (`Delta ≈ 14`), without over-inflating the value scale
the critic must learn.

```python
GAMMA = 0.99  # 🎯 C1 — your discount factor; justify it in the next cell
```

#### Cell 39 — Calibration C2 — rollout length & parallel envs
Rollout length `T` set to roughly 2× the measured obstacle spacing so a full "climb →
clear a pipe" episode fits inside one recorded window (GAE can't assign credit across a
rollout boundary); parallel-env count `N` set independently, since it controls gradient
noise (`∝ 1/√(N·T)`), a different lever from `T`.

```python
ROLLOUT, NUM_ENVS = 32, 256  # 🎯 C2 — your choice of (T, N)
```

#### Cell 45 — Calibration C3 — GAE λ, learning rate, epoch count
`λ` chosen so GAE's own effective horizon `1/(1−γλ)` lands in `[Delta, 2·Delta]`;
learning rate and epoch count tuned empirically by watching the `approx_kl` diagnostic
from cell 30 land in a healthy `[0.01, 0.02]` band rather than by watching the score
directly.

```python
GAE_LAMBDA, LR, EPOCHS = 0.95, 0.003, 10  # 🎯 C3 — your choices
```

#### Cell 120 — Bonus — `sample_episode_physics` (domain randomization)
Samples one episode's game physics (a joint multiplier on gravity/velocity constants,
and the strong-flap noise level `V_dev`) from that episode's own seeded RNG, so training
sees a distribution of slightly different dynamics instead of one fixed instance, while
staying fully reproducible run-to-run.

```python
from flappy.domain_randomization import DRConfig, DomainRandomizedVecFlappy
import flappy.ppo as _ppo_mod

def sample_episode_physics(
    base: Const, rng: np.random.Generator, dr: DRConfig
) -> Const:
    """Sample one episode's physics from the episode-seed RNG.

    Parameters
    ----------
    base
        Nominal ``Const`` for this run (e.g. ``LARGE``). Randomization is
        applied on top of these values.
    rng
        NumPy generator seeded by the episode seed, so the same episode
        always draws the same physics.
    dr
        Which knobs to randomize: ``vertical_scales`` (joint multiplier on
        ``g``, ``V_max``, ``U_weak``, ``U_strong``).

    Returns
    -------
    Const
        A copy of ``base`` with the sampled knobs filled in, or ``base``
        itself if nothing was randomized. The ``name`` field is tagged with
        the drawn scale / ``V_dev`` for logging.
    """
    kw: dict = {}
    name = base.name

    if dr.vertical_scales is not None:
        s = int(rng.choice(dr.vertical_scales))
        kw.update(
            # 🎯 Update the base variables with the generated multiplier value
            g= int(base.g * s),
            V_max= int(base.V_max * s),
            U_weak= int(base.U_weak * s),
            U_strong= int(base.U_strong * s)
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

flappy.domain_randomization.DomainRandomizedVecFlappy._sample_episode_physics = staticmethod(
    sample_episode_physics
)

```

#### Cell 126 — `FINAL_CONFIG`
Assembles the final hand-in configuration from the three calibrated base values (`γ`,
`T`/`N`, initial `λ`/`lr`/`epochs`) plus the specific knobs the 20-run tuning sweep found
best (`gae_lambda=0.90`, `update_epochs=20`, `ent_coef=0.1`, `clip_coef=0.3`).

```python
# 🎯🎯 Declare your final config here by replacing the hyperparameters below
FINAL_CONFIG = Config.for_large(gamma=GAMMA, gae_lambda=GAE_LAMBDA, lr=LR,
                               update_epochs=EPOCHS, rollout=ROLLOUT,
                               num_envs=NUM_ENVS, ent_coef=ENT_COEF,
                               clip_coef=CLIP_COEF)  # everything else comes from Config.for_large()
```

Beyond these graded blanks, a disciplined 20-run, one-factor-at-a-time tuning sweep was
run over `Config.for_large()` (each promising change re-run across 3 random seeds to
rule out noise) to select the knobs assembled in cell 126.

---

## Project 5 — Compliance Q&A (Retrieval-Augmented Generation)

**Scope & goal.** Build a **RAG (Retrieval-Augmented Generation)** system that answers
questions about company compliance policy documents using only the provided documents —
never the language model's own memory — so answers are grounded and citable rather than
hallucinated.

**Theory applied, in plain terms.**
- **Why retrieval before generation:** an LLM asked to recall a policy from memory can
  produce a fluent, confident, but wrong answer. RAG turns the task into an "open-book
  exam" — first find the actually relevant passages, then force the model to answer only
  from them.
- **Chunking with overlap:** splitting documents into small, overlapping pieces means
  each idea gets its own embedding (a whole document embedded as one vector blurs every
  topic together), while the overlap guarantees a sentence that straddles a chunk
  boundary still appears whole in at least one chunk.
- **Lexical vs. semantic search, and fusing them:** BM25 (keyword/lexical search) finds
  exact word matches but misses synonyms ("time off" won't match "annual leave");
  embedding-based (semantic/cosine) search captures meaning but can miss exact technical
  terms. Because their scores live on incompatible numeric scales, they're combined via
  **Reciprocal Rank Fusion** — which only compares each method's *rank order*, never
  the raw scores directly.
- **Over-retrieval before fusion:** fusion can only re-rank what it's given, so each
  search method must be asked for more candidates than the final answer needs, or a
  genuinely relevant result ranked 8th by one method never gets the chance to surface.
- **Abstention:** retrieval always returns *some* nearest passages, even when nothing in
  the corpus is truly relevant. The system checks a minimum relevance score and returns
  a fixed "I don't know" rather than letting the LLM improvise a plausible-sounding wrong
  answer from weak context — treated as the single highest-stakes design decision in a
  compliance setting.

**Every notebook cell with a `TODO` exercise blank**, with the exact applied code:

#### Cell 24 — `sliding_window` (Part 1)
Splits text into overlapping word windows (`step = chunk_size − overlap`), guarding
against `overlap ≥ chunk_size` since that would make the window never advance (an
infinite loop).

```python
def sliding_window(text, chunk_size=200, overlap=40):
    """Split `text` into overlapping windows of words."""
    # --- validation: a window that never advances is an infinite loop, not a bad result.
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:                       # ✏️ which overlap stops the window moving forward?
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap                     # ✏️ how far the window slides between chunks
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]    # ✏️ where does this window end?
        if not window:
            break
        chunks.append(" ".join(window))             # ✏️ the window as one space-joined string
        if start + chunk_size >= len(words):        # ✏️ this window already reached the last word -> stop,
            break                                   #    otherwise the tail chunk is emitted twice
    return chunks
```

#### Cell 31 — `ingest_document` (Part 1)
Runs chunk → wrap each chunk in a `Chunk` object carrying a **copy** of the document's
metadata (not a shared reference) → batch-embeds all chunks in one call (not one call
per chunk) → sizes and populates the Qdrant collection from the embedding width.

```python
def ingest_document(self, document):
    """Run load->CHUNK->EMBED->store for one Document. Return the list of stored Chunks."""
    # 1. CHUNK — split the text using this core's configured strategy.
    texts = chunking.chunk_text(
        document.text,
        chunk_size=self.chunk_size,
        overlap=self.overlap,
        strategy=self.strategy,
    )
    if not texts:
        return []

    # 2. WRAP — one Chunk per passage, each carrying the document's metadata.
    chunks = [
        Chunk(document_id=document.id, index=i, text=text,
              metadata=dict(document.metadata))                                     # ✏️ a *copy*, so chunks don't share one dict
        for i, text in enumerate(texts)
    ]

    # 3. EMBED — one batched call for the whole document, not one call per chunk.
    embeddings = self.embedder.embed_batch(texts)                                   # ✏️ what gets embedded, in chunk order
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding                                                 # ✏️ attach each vector to its chunk

    # 4. STORE
    self.db.ensure_collection(self.collection_name, vector_size=len(embeddings[0])) # ✏️ how wide is one vector?
    self.db.insert_document(self.collection_name, document, chunks)
    return chunks
```

#### Cell 38 — `apply_metadata_filter` (Part 1)
AND-matches every key/value in the filter dict against each chunk's metadata; returns
the exact same (unfiltered) list object when there's no filter, so the BM25 index cache
— keyed on that list's identity — isn't silently defeated.

```python
def apply_metadata_filter(chunks, metadata_filter):
    """Keep only chunks whose metadata matches every key/value in the filter."""
    if not metadata_filter:
        return chunks                                                                       # ✏️ no filter: hand back the *same list object* (the BM25 cache
                                                                                            #    recognises the full corpus by identity, so don't copy it)
    return [
        chunk for chunk in chunks
        if all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())  # ✏️ what every pair must satisfy
    ]
```

#### Cell 56 — `cosine_similarity` (Part 2)
Vectorized cosine similarity of one query vector against every row of a matrix in a
single NumPy operation (dot products via `@`, per-row norms via `np.linalg.norm`),
dividing safely so a zero vector scores `0.0` instead of `NaN`.

```python
def cosine_similarity(query_vec, matrix):
    """Cosine similarity of `query_vec` against every row of `matrix`."""
    # query_vec: (D,)   matrix: (N, D)   returns: (N,)
    # The denominator ||row|| * ||query||, one value per row of the matrix.
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)              # ✏️ which axis is "per row"?
    # The numerator: the dot product of every row with the query.
    scores = matrix @ query_vec   # ✏️ every row against the query in one operation — no Python loop
    # Divide only where the denominator is non-zero: a zero vector scores 0.0, never NaN.
    return np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)    # ✏️ the safety condition
```

#### Cell 61 — `keyword_search` (Part 2)
Filters candidates by metadata first, then scores the remainder with a cached BM25
index and returns the top-k — lexical, exact-word-overlap ranking.

```python
def keyword_search(self, query, top_k=None, metadata_filter=None):
    """Rank self.chunks by BM25 overlap with `query`. Return list[(Chunk, score)]."""
    top_k = self.top_k if top_k is None else top_k
    query_tokens = _tokenize(query)

    candidates = self._apply_metadata_filter(self.chunks, metadata_filter)  # ✏️ narrow the corpus by metadata *before* scoring
    if not candidates or not query_tokens:
        return []                                                           # BM25 raises on an empty corpus

    bm25 = self._bm25_index(candidates)                                     # cached for the unfiltered corpus
    scores = bm25.get_scores(query_tokens)                                  # ✏️ one BM25 score per candidate
    return self._top_k(candidates, scores, top_k)
```

#### Cell 68 — `reciprocal_rank_fusion` (Part 2)
Sums `1 / (k + rank)` per chunk across every input ranked list (rank-based, so BM25's
and cosine's incompatible score scales never need to be reconciled), sorts by the fused
score, and truncates to `top_k`.

```python
def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    """Fuse several ranked [(Chunk, score)] lists into one. Return list[(Chunk, fused_score)]."""
    fused_scores = {}   # chunk id -> summed contribution across every list
    chunks_by_id = {}   # chunk id -> the Chunk itself, so we can return objects
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):   # rank is 0-based
            # Rank, never score: each list a chunk appears in adds 1 / (k + rank).
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1 / (k + rank)   # ✏️ this list's contribution
            chunks_by_id[chunk.id] = chunk

    ordered = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)            # ✏️ sort by fused score, not by id
    return [(chunks_by_id[cid], score) for cid, score in ordered[:top_k]]
```

#### Cell 73 — `hybrid_search` (Part 2)
Runs keyword search and embedding search each with the same, wider pool
(`top_k × overretrieve`) and the same metadata filter, then fuses the two ranked lists
and cuts back down to `top_k`.

```python
def hybrid_search(self, query, top_k=None, metadata_filter=None, overretrieve=None):
    """Run both retrievers wide, fuse their rankings, then cut to top_k."""
    top_k = self.top_k if top_k is None else top_k
    overretrieve = self.overretrieve if overretrieve is None else overretrieve

    # Over-retrieve: fusion can only reorder what you hand it.
    pool = top_k * overretrieve  # ✏️ candidates per branch — never fewer than top_k

    keyword_results = self.keyword_search(query, top_k=pool, metadata_filter=metadata_filter)
    # ✏️ the semantic branch gets exactly the same treatment — same pool, same filter.
    #    Filter one branch and not the other and excluded documents walk back in.
    embedding_results = self.embedding_search(query, top_k=pool, metadata_filter=metadata_filter)

    return self._reciprocal_rank_fusion(
        [keyword_results, embedding_results],
        top_k=top_k,            # ✏️ cut the wide fused pool back down to what the caller asked for
        k=self.rrf_k,
    )
```

#### Cell 101 — `retrieve_and_answer` (Part 2)
Retrieves; if nothing survives the relevance threshold, returns a fixed "no context"
answer **without calling the LLM** (abstention); otherwise concatenates the retrieved
chunk texts into a context block and prompts the LLM to answer only from it, returning
the answer plus its source chunks for citation.

```python
def retrieve_and_answer(self, query, search_type=None, metadata_filter=None):
    """Retrieve, then answer with the LLM. Return (answer_str, list[(Chunk, score)])."""
    results = self._retrieve(query, search_type, metadata_filter)
    if not results:
        return NO_CONTEXT_ANSWER, []                                        # ✏️ abstain — and note the LLM call below is never reached

    context = "\n\n".join(chunk.text for chunk, _score in results)          # ✏️ the retrieved passages, as one context block
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = self.llm.complete(prompt, system_prompt=self.system_prompt)    # ✏️ the pipeline's system prompt
    return answer, results
```

Correctness was verified two ways: a unit-test cell directly below each exercise, and a
12-question gold-answer evaluation (§2.5 of the notebook) measuring recall *and*
"context cost" together — which surfaced a real, non-obvious tradeoff: an "un-chunked"
whole-document baseline scores highest on recall but at roughly 10× the context cost of
the chunked approach.

---

## Project 6 — Policy-Compliance Verification Agent (agentic system built on RAG)

**Scope & goal.** Reuse Project 5's RAG pipeline as one *tool* inside a larger **agentic
workflow**: watch an employee submit a dashboard request (expense report, procurement
request, access change, or time-off request), check it against company policy, and
drive the UI to explain any violation and suggest a fix — citing the exact policy text
behind every decision.

**Theory applied, in plain terms.**
- **Agentic workflows as small, typed steps — not one big prompt:** a single free-text
  LLM answer can't be unit-tested, can't be forced to cite sources, and any parsing
  failure silently corrupts everything downstream. Instead, every step passes a typed
  Python object to the next (a request → a search query → retrieved evidence → a
  verdict → a proposed fix → a UI instruction), so each step is independently testable.
- **Grounding & auditability:** the reasoning steps (the "verifier" and the "fix
  proposer") are explicitly instructed to reason *only* from retrieved policy text, and
  every finding must cite the exact source document and passage that justifies it —
  never invent a rule.
- **Separation of detection and repair:** one agent's only job is to *detect* problems;
  a separate, narrowly-scoped subagent is spawned per problem to *propose a fix* — a
  smaller, more reliable task than doing both at once.
- **Controlled tool use for anything touching the UI:** the step that updates the
  dashboard is not allowed to freely generate HTML/text — it can only call from a fixed
  set of five actions (mark compliant, warn, highlight a field, attach a citation,
  suggest a correction), which is what makes its behavior fully predictable and
  reviewable, and it deliberately involves **no LLM call at all**.

**Every notebook cell with a `🎯` exercise blank**, with the exact applied code:

#### Cell 15 — `validate_action` (Part 1)
Deterministic, non-AI gatekeeping: rejects an action with no `action_type`, an
unrecognized `action_type`, empty `fields`, or a missing/blank required field —
returning a list of error strings (empty = valid).

```python
def validate_action(action):
    """Return a list of error strings for an action.
    Arguments:
        action (Action): An Action object to validate.
    Returns:
        A list of error strings. An empty list means the action is well-formed.
    """

    errors = []
    # 🎯 Step 1: If action.action_type is empty, raise the error
    #            "action_type is required" and return early
    if not action.action_type:
        errors.append("action_type is required for the input action object")
        return errors
    # 🎯 Step 2. If action.action_type not an allowed type, raise the error
    #            "unknown action_type ... (known: ...)" and return early
    if action.action_type not in ACTION_TYPES:
        known = ", ".join(sorted(ACTION_TYPES))
        errors.append(f"unknown action_type '{action.action_type}' (known: {known})")
        return errors
    # 🎯 Step 3: If action.fields is empty, raise the error
    #           "fields must not be empty" and return early
    if not action.fields:
        errors.append("fields must not be empty")
        return errors
    # 🎯 Step 4: For each required field for this action type, if it is missing or its
    #            value is None / an empty string, add an error "required field ... must not be empty"
    for field_name in required_fields(action.action_type):
        if field_name not in action.fields:
            errors.append(f"missing required field '{field_name}'")
            continue
        value = action.fields[field_name]
        if value is None or (isinstance(value, str) and value.strip() == ""):
            errors.append(f"required field '{field_name}' must not be empty")
    return errors
```

#### Cell 20 — `build_context` (Part 2)
Turns a validated action into a natural-language RAG query plus a `metadata_filter`
restricted to the one policy document that governs this action type.

```python
def build_context(action):
    """Turn an Action into a VerificationContext for retrieval.
    Arguments:
        action (Action): An Action object to build context for.
    Returns:
        A VerificationContext object containing the query, metadata filter, summary, and action ID.
    """

    # 🎯 details about the action type
    spec = get_action_type(action.action_type) 
    # name of the action type
    label = spec["label"] if spec else action.action_type 
    field_str = ", ".join(f"{k}: {v}" for k, v in action.fields.items())
    summary = f"{label} — {field_str}" if field_str else label
    # 🎯 a string that asks if the action named `label` is allowed under company policy, with details `field_str`
    query = f"Is this {label} action allowed under company policy? Details: {field_str}" 
    # 🎯 policy source for the action type
    source = policy_source_for(action.action_type)
    metadata_filter = {"source": source} if source else None
    return VerificationContext(query=query, metadata_filter=metadata_filter,
                               summary=summary, action_id=action.id)

```

#### Cell 25 — `retrieve_policies` (Part 3)
Wraps the RAG tool's `.retrieve(query, search_type, metadata_filter)` call and reshapes
each `(Chunk, score)` pair into a `RetrievedChunk` DTO exposing only `source`,
`chunk_id`, `document_title`, `text`, `score`.

```python
def retrieve_policies(self, context):
    """Retrieve policy passages for a VerificationContext.
    Arguments:
        context (VerificationContext): The context to retrieve policies for.
    Returns:
        A list of RetrievedChunk objects.
    """

    results = self.rag.retrieve(
        # 🎯 the query string from the context
        context.query, 
        # hybrid by default
        search_type=self.search_type, 
        # 🎯 the metadata filter from the context 
        metadata_filter=context.metadata_filter, 
    )
    return [
        RetrievedChunk(
            # 🎯 the source of the chunk (or "unknown" if not present)
            source=chunk.metadata.get("source", "unknown"),
            # 🎯 the index of the chunk in the source document
            chunk_id=chunk.index, 
            # 🎯 the document title of the chunk; or its source if no title is present; or "unknown" if neither is present
            document_title=chunk.metadata.get("title", chunk.metadata.get("source", "unknown")), 
            # 🎯 the text of the chunk
            text=chunk.text, 
            # 🎯 the closeness of the chunk to the query
            score=score, 
        )
        for chunk, score in results
    ]
```

#### Cell 31 — `verify_action` (Part 4)
Builds the verifier's prompt from the action + retrieved policies, calls the LLM under
a system prompt that fixes a strict JSON schema, and parses the reply into a typed
`Verdict` with a list of `Problem`s (each field read defensively with a safe default).

```python
def verify_action(self, action, policies):
    """Return a Verdict for `action` grounded in `policies`.
    Arguments:
        action (Action): The action to verify.
        policies (List[RetrievedChunk]): The retrieved chunks to ground the verification in.
    Returns:
        A Verdict object containing the status (str), problems (List[Problem]), and summary (str).
    """

    # 🎯 construct the prompt for the verifier LLM
    prompt = build_verifier_prompt(action, policies)
    raw = self.llm.complete(prompt, system_prompt=VERIFIER_SYSTEM_PROMPT)

    # 🎯 parse the raw LLM output as JSON
    data = extract_json(raw)
    problems = [
       Problem(
           # 🎯 one problematic field (default: "")
           field=item.get("field", ""),
           # 🎯 the source policy filename (default: "")
           policy_source=item.get("policy_source", ""),
           # 🎯 the index of the chunk in the source document (-1 if not applicable)
           chunk_id=int(item.get("chunk_id", -1)),
           # 🎯 explanation of the why it is a violation (default: "") 
           explanation=item.get("explanation", ""),
           # 🎯 severity of the problem (default: "medium") 
           severity=item.get("severity", "medium"),
       )
       for item in data.get("problems", [])
    ]
    status = data.get("status", "problematic" if problems else "valid")
    if problems:
        status = "problematic"
    return Verdict(status=status, problems=problems, summary=data.get("summary", ""))
```

#### Cell 39 — `propose_solution` (Part 5)
For one detected problem, builds a narrowly-scoped prompt, calls the LLM, and parses
its JSON reply into a `Solution` (proposed fix text, corrected value, and the policy
citation supporting the fix).

```python
def propose_solution(self, action, problem, policies):
    """Propose a fix for one `problem` in an action based on `policies`.
    Arguments:
        action (Action): The action to propose a solution for.
        problem (Problem): The specific problem to address.
        policies (List[RetrievedChunk]): The retrieved policy chunks to ground the solution in.
    Returns:
        A Solution object containing the proposed fix, corrected value, supporting sources, and explanation.
    """

    # 🎯 construct the prompt for the solution subagent
    prompt = build_solution_prompt(action, problem, policies)
    raw = self.llm.complete(prompt, system_prompt=SOLUTION_SYSTEM_PROMPT)
    # 🎯 parse the raw LLM output as JSON
    data = extract_json(raw)
    # Extract the first problematic field (if any) and propose a fix for it
    target_field = problem.field if problem.field else None
    return Solution(
        problem_id=problem.problem_id,
        # 🎯 the field to fix
        field=target_field,            
        # 🎯 the proposed fix text
        proposed_fix=data.get("proposed_fix", ""),     
        # 🎯 the corrected value (None if not applicable)
        corrected_value=data.get("corrected_value", ""),  
        # 🎯 the policy source filename
        policy_source=data.get("policy_source", ""),    
        # 🎯 the chunk id of the supporting excerpt (-1 if not applicable)
        chunk_id=data.get("chunk_id", -1),         
        # 🎯 explanation of the fix ("" if not provided)
        explanation=data.get("explanation", ""),      
    )
```

#### Cell 43 — `run_display_agent` (Part 6)
The one step with **no LLM call**: a deterministic translator from the verdict/solutions
into a fixed set of UI tool calls (`mark_ok`, `warn`, `highlight_field`,
`suggest_correction`, `attach_citation`).

```python
def run_display_agent(result, ui):
    """Translate a VerificationResult into controlled UI tool calls (returns None)."""

    verdict = result.verdict

    # 🎯 Case 1: The action is valid and complies with company policy
    if verdict.is_valid:
        ui.mark_ok(verdict.summary or "This action complies with company policy.")
        return

    # 🎯 Case 2: The action is not valid and violates company policy
    # Extract the highest severity level from the problems, defaulting to "medium" if none are present
    # Ranking is done using the `rank` dictionary, where "low" < "medium" < "high"
    rank = {"low": 0, "medium": 1, "high": 2}
    top = max((p.severity for p in verdict.problems), key=lambda s: rank.get(s, 1), default="medium")
    ui.warn(verdict.summary or "This action may violate company policy.", severity=top)

    # 🎯 highlight each bad field with its explanation as message
    for problem in verdict.problems:
        ui.highlight_field(problem.field, problem.explanation)
    # 🎯 suggest a correction for each solution proposal
    for solution in result.solutions:
        ui.suggest_correction(solution.field, solution.corrected_value, solution.proposed_fix)

    # Extract all citations
    citation_problems = {
        (p.field, p.policy_source, p.chunk_id) for p in verdict.problems if p.field
    }
    citation_solutions = {
        (s.field, s.policy_source, s.chunk_id) for s in result.solutions if s.field
    }
    all_citations = citation_problems.union(citation_solutions)
    all_chunks = { (p.source, p.chunk_id): p.text for p in result.policies }

    # 🎯 Attach citations to the UI for each unique (field, source, chunk_id) combination
    for field, source, chunk_id in all_citations:
        snippet = all_chunks.get((source, chunk_id), "")  # get the text of the chunk from chunk number `chunk_id` in the document `source` ("" if not found)
        ui.attach_citation(field, source, snippet)
```

#### Cell 49 — `VerifierPipeline.verify` (Part 7)
The orchestrator: calls Parts 1–6 in order, short-circuiting with a validation error if
Part 1 fails, and spawning one Part-5 call per problem Part 4 found.

```python
def verify(self, action):
    """Run the full agentic workflow for one action.
    Arguments:
        action (Action): The action to verify.
    Returns:
        A VerificationResult object containing the verdict, solutions, and UI actions.
    Exceptions:
        ActionValidationError: If the action does not pass the validation step.
    """

    # 🎯 Part 1 — validate action
    errors = action_validation.validate_action(action)
    if errors:
        raise ActionValidationError(errors)

    # 🎯 Part 2 — build the verification context.
    context = context_builder.build_context(action)

    # 🎯 Part 3 — retrieve grounding policies via the RAG tool.
    policies = self.policy_tool.retrieve_policies(context)

    # 🎯 Part 4 — the verifier agent reasons over action + policies.
    verdict: Verdict = self.verifier.verify_action(action, policies)

    result = VerificationResult(
        action=action, verdict=verdict, solutions=[], ui_actions=[], policies=policies
    )

    # 🎯 Part 5 — one solution subagent per detected problem.
    for problem in verdict.problems:
        result.solutions.append(self.solver.propose_solution(action, problem, policies))

    # Part 6 — the display agent acts on the UI through tools.
    ui = DirectiveCollector()
    display_agent.run_display_agent(result, ui)
    result.ui_actions = ui.actions

    return result
```

Each function was tested offline first, against scripted/mock AI responses, before an
optional live run against a real OpenRouter model. Separate from the graded exercises,
this session also did the local-environment plumbing to run the notebook and its
companion FastAPI/React dashboard outside of Google Colab (pulling in the project's
supporting code, wiring up a local `.env` API-key file, and fixing a couple of
Windows-specific path/encoding bugs and a local Qdrant file-lock conflict along the way).

---

## Common thread across all four projects

Every project follows the same underlying discipline, just applied to a different
domain: **isolate each step of the pipeline, verify it independently (tests or measured
comparisons), and justify every non-default choice with either a specific measurement or
a specific piece of theory** — never "it seemed to work." That discipline is also why
each project ships its own `DOCUMENTATION.md` (the full trail of what was tried and
measured) and `DEFENSE.md` (the condensed version meant to be spoken, not read).
