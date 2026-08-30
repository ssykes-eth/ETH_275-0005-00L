# Repository Summary

This repository holds the graded projects for ETH course **275-0005-00L, "AI Workshop —
From Data to Solutions."** Each project folder is self-contained and has its own
`DOCUMENTATION.md` (full detail, cell/function by cell/function) and `DEFENSE.md` (a
condensed 15-minute oral-defense script). **This file is the short, top-level overview**
for someone who has never opened any of the four projects: what each one is for, the
theory it rests on, and what was actually built or changed.

Four projects carry full documentation: **Project 1** (regression/ML pipeline),
**Project 3** (reinforcement learning), **Project 5** (retrieval-augmented generation),
and **Project 6** (an agentic system built on top of Project 5's RAG). `code_exercises/`
holds smaller, ungraded practice notebooks with no formal write-up. A former "Project 2"
folder was intentionally dropped from the repository and is not covered here.

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

**Applied changes.** Built a 6-stage pipeline (`split → fix corrupted single values →
fill missing values → remove anomalous rows → select/de-duplicate features → scale →
model selection with stacking`), each stage in its own tested module. Roughly 38 model
configurations across 6 model families were compared honestly via cross-validation;
several plausible alternatives (KNN-based imputation, Boruta feature selection,
`RobustScaler`, age-stratified folds, XGBoost) were tried and measured to be worse, and
deliberately left out — a documented "what we tried and rejected" record, not just "what
we kept."

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

**Applied changes.** Implemented three core functions inside the PPO training loop
(`build_observation`, `compute_gae`, `ppo_loss`), each independently unit-tested against
known closed-form behavior. Derived every hyperparameter from a measured game property
rather than a default, then ran a disciplined 20-run, one-factor-at-a-time tuning sweep
(each promising change re-run across 3 random seeds to rule out noise) to arrive at a
final configuration — plus a bonus domain-randomization variant.

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

**Applied changes.** Implemented 8 functions across the write side (chunking a document,
ingesting/embedding/storing it, applying metadata filters) and the read side (cosine
similarity, BM25 keyword search, reciprocal rank fusion, hybrid search, and the final
retrieve-then-answer/abstain logic). Correctness was verified two ways: unit tests for
each function, and a 12-question gold-answer evaluation measuring recall and "context
cost" together — which surfaced a real, non-obvious tradeoff (an "un-chunked" baseline
scores highest on recall but at roughly 10× the context cost of the chunked approach).

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

**Applied changes.** Implemented the seven pipeline steps (validate → build a search
context → retrieve grounding policy text → verify compliance via LLM → propose a fix
per violation via LLM → translate the result into UI actions → orchestrate all of the
above into one call), each tested offline first against scripted/mock AI responses
before an optional live run against a real model. Also did the local-environment
plumbing to run the notebook and its companion FastAPI/React dashboard outside of Google
Colab (pulling in the project's supporting code, wiring up a local API key file, and
fixing a couple of Windows-specific path/encoding bugs along the way).

---

## Common thread across all four projects

Every project follows the same underlying discipline, just applied to a different
domain: **isolate each step of the pipeline, verify it independently (tests or measured
comparisons), and justify every non-default choice with either a specific measurement or
a specific piece of theory** — never "it seemed to work." That discipline is also why
each project ships its own `DOCUMENTATION.md` (the full trail of what was tried and
measured) and `DEFENSE.md` (the condensed version meant to be spoken, not read).
