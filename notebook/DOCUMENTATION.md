# Documentation: `flappy_bird_project_student.ipynb` 

This document explains the whole notebook in plain language — as if explaining
it to a curious kid who has never seen reinforcement learning before — and
then explains, in detail, every place marked with a 🎯 ("your job") that has
been filled in, why the chosen answer is right, and what would go wrong with
a different answer.

---

## 1. The big picture — what is this notebook actually doing?

Imagine a video game version of Flappy Bird, but instead of *you* pressing
the buttons, you are going to **teach a computer program to play it by
itself**. The computer starts out knowing nothing — it just tries random
button presses. Every time it survives a moment it gets a tiny reward
(like a gold coin), and every time it crashes into a pipe, the game ends.
Over many, many practice games, the computer notices which button presses
tend to lead to more coins, and slowly gets better. This process is called
**Reinforcement Learning (RL)**, and the specific technique used here is
called **PPO** (Proximal Policy Optimization) — one of the most popular and
reliable RL algorithms used in real robots, game AIs, and even chatbots.

### The game, in kid terms

- The bird lives on a grid of squares. It has a **height** (`y`, how high up
  it is) and a **vertical velocity** (`v`, whether it's currently moving up
  or down, and how fast).
- Every tick of time, **gravity** pulls the bird down a little bit,
  automatically, whether you want it to or not.
- The bird has **3 possible moves**:
  1. **Do nothing** — free, but gravity still pulls you down.
  2. **Weak flap** — costs a little energy, gives a small, always-the-same
     push upward.
  3. **Strong flap** — costs more energy, gives a bigger push upward, but the
     push is a little bit random (sometimes a bit stronger, sometimes a bit
     weaker) — like flapping with slightly wet wings.
- Pipes (obstacles) march toward the bird one column at a time, each with a
  **gap** at some height. If the bird is not lined up with the gap when the
  pipe reaches it, the bird crashes and the game is over.
- Every step the bird survives, it earns `reward = 1 - cost_of_move`. So
  doing nothing is "free" reward, but flapping costs some of that reward
  (like paying for fuel). The bird has to learn: *flap only when it actually
  needs to*, not for fun, because every flap has a price.

### Why is this interesting mathematically? ("small" vs "large")

This notebook studies the game on **two different grid sizes**:

- **`small`** — a coarse (chunky, low-resolution) grid with only about 7,770
  possible situations ("states") the bird could ever be in. Because there
  are so few states, a computer can write down, *in a giant table*, the
  perfect answer for every single state — "if you're exactly here, this is
  the best move." This perfect table is computed using an exact mathematical
  method called **value iteration**, solving something called the **Bellman
  equation**. This gives the *true best possible score* (`V*`, `pi*`), like
  an answer key.
- **`large`** — the same game on a much finer grid, with roughly
  `10^17` (100 quadrillion) possible states. That is far too many to fit in
  any table (it would need many terabytes of memory). So here, the exact
  method is *impossible*, and the only realistic option is to use a **neural
  network** to *guess* good moves from experience — that guessing process is
  PPO.

The cool part of the notebook: on `small`, you can compare your PPO agent's
score directly against the mathematically perfect answer key, to see just
*how close to perfect* your learned agent gets.

---

## 2. Walking through the notebook, section by section

### Part 0 — Setup (Cells 0–6)

- **Cell 0 (markdown):** Introduction and grading overview — explains the
  three coding tasks, three calibration tasks, and the tuning table you must
  submit.
- **Cell 1 (markdown):** Explains the physics of the game in a short table
  (no flap / weak flap / strong flap, their costs and effects) before you
  touch any code.
- **Cell 2 (code):** Draws a picture (a "schematic") of the state variables
  `(y, v, d1, h1)` and the three actions, purely for intuition — no game
  code runs here, it's just matplotlib drawing shapes and labels.
- **Cell 3 (code):** An **interactive playground** — buttons you can click to
  manually fly the bird yourself on the small grid, to build intuition before
  ever writing a line of RL code.
- **Cells 5–6 (code):** Downloads/loads the provided course code (the
  simulator, the PPO training loop, etc.) and imports it. This is the
  "kitchen" that's already built for you — you don't need to modify it, only
  use it.
  `SMALL` and `LARGE` are created here: these are `Const` objects holding all
  the numbers that define each grid (width, height, gravity, gap size,
  etc.).

### Discretization explorer (Cells 7–10)

- **Cell 7 (markdown)** and **Cell 8 (code):** An interactive slider lets you
  watch what happens to the *size* of the state space as you make the grid
  finer and finer. At first (level 1, a tiny `4x6` grid) the state count is
  small enough to solve exactly. As you slide up, the state count multiplies
  quickly (this is called the **"curse of dimensionality"**) until it blows
  past a cutoff (5,000,000 states) where an exact table is no longer
  possible. This visually *proves* why we need PPO for the `large` case,
  rather than just asserting it.
- **Cell 10 (code):** Prints the actual numbers for `SMALL` and `LARGE` —
  how many states, what the physics constants are.

### Measuring "Delta" (Cells 11–12)

- **Cell 11 (markdown):** Introduces **Delta** = the average number of steps
  between two pipes. This single number will later be used to sanity-check
  *every* important RL hyperparameter (like a ruler you keep re-using).
- **Cell 12 (code):** `measure_delta` runs many simulated environments for a
  while and measures the real average gap-spacing directly from the game's
  geometry (not from trial runs, since a bad policy that dies immediately
  would give a misleading measurement).

### Watching an example policy fly (Cells 13–16)

- **Cell 13 (markdown)** and **Cell 14 (code):** Introduces
  `LookaheadPolicy` — a simple hand-written (not learned) "pilot" that
  calculates, in a straight-line way, what velocity it needs to reach the
  center of the next gap, and picks the action that gets closest to that.
  This is used throughout the notebook as a reference/baseline (like a
  "reasonably good human pilot" to compare the AI against). The cell then
  draws a graph of height and velocity over time as this pilot flies.
- **Cells 15–16 (markdown/code):** A frame-by-frame interactive "scrubber"
  widget - like a video player with a slider - so you can rewind and inspect
  every single step of an episode, including which action was taken at each
  moment.

### Part 1 — The exact optimum (Cells 17–23)

- **Cell 17 (markdown):** Explains that on `small`, since the state space is
  small enough, we can compute the mathematically perfect value function
  `V*` and perfect policy `pi*` using **value iteration** — an algorithm
  that repeatedly applies the Bellman equation until the values stop
  changing. It stresses that `gamma` (the discount factor, how much future
  reward matters compared to reward right now) *must* be less than 1,
  otherwise the total future reward could be infinite (survive forever = infinite
  reward), which breaks the math.
- **Cell 18 (code):** Actually runs value iteration and prints `J*`, the
  best possible average score.
- **Cells 19–20 (markdown/code):** Shows that changing `gamma` doesn't just
  change *how good* the answer is — it can literally change *what the best
  policy even is* (i.e., picking `gamma` is like picking a different
  problem to solve, not just a different "difficulty setting").
- **Cells 21–23 (markdown/code):** Compares several simple strategies (never
  flap, always weak flap, random, threshold, lookahead, and the true DP
  optimum) side by side. The surprising finding: **"never flap" beats
  "random" and "always flap."** This is because flapping costs reward
  *immediately*, while the benefit of flapping (surviving past a pipe) pays
  off *later*. A learning agent starting from random behavior will initially
  feel like it should stop flapping altogether — a trap called a **"deceptive
  local optimum."** The notebook deliberately keeps this trap in the game so
  students understand it's not a bug if their agent temporarily "gives up".

### Part 2 — Implementing PPO (Cells 24–32) — **the first three 🎯 coding tasks**

This is where the student had to write actual code. See Section 3 below for
a full explanation of each.

### Part 3 — Calibration (Cells 33–47) — **three 🎯 "compute, don't guess" tasks**

Rather than picking hyperparameters (settings that control how the AI
learns) by copying them from somewhere else, this section forces you to
*calculate* good values from measured facts about *this specific game*.
Full breakdown in Section 4.

### Part 4 — Train and compare to the optimum (Cells 48–51)

- **Cell 49 (code):** Trains the PPO agent from scratch, 3 different times
  (3 different random seeds), on `small`. After each training run, it
  compares the learned policy's performance (`J`) against the mathematically
  perfect performance (`J*`) computed back in Part 1, and reports the ratio
  `J/J*` (how close to perfect the AI got) as well as how often the AI's
  chosen actions *agree* with the perfect policy's actions.
- **Cell 51 (code):** Plots the AI's learned value estimates against the
  true optimal values, color-coded by how often each state was actually
  visited. This shows visually that **the critic (the AI's internal "how
  good is this situation" estimator) is most wrong exactly in states it
  rarely sees** — which makes sense: you can't learn from experience you
  never had.

### Part 5 — Now the exact method is impossible (Cells 52–55)

- **Cell 53 (code):** Tries to build the exact lookup table for `LARGE` and
  it fails with a `MemoryError` — a real, live demonstration (not just a
  claim) that the "curse of dimensionality" makes exact solving impossible
  here — you'd need many terabytes of memory just to store the numbers.
- **Cell 54 (markdown):** Explains that on `LARGE`, "never flap" is an even
  more tempting trap (31 free steps of survival before the first obstacle
  even appears), and that raising the "entropy coefficient" (encouraging the
  AI to try more varied actions instead of settling into one habit) helps
  escape the trap only about half the time — you also need a **reward
  shaping bonus** (`align_bonus`, already provided in `shaped_reward`) that
  rewards the bird for lining up with the gap early, to make escaping the
  trap reliable.
- **Cell 55 (code):** Trains PPO on `LARGE` using the pre-built
  `Config.for_large()` settings and compares it against the `LookaheadPolicy`
  reference pilot.

### Part 6 — The tuning table (Cells 56–118) — **20 experiments**

The student had to run 20 experiments, changing **one hyperparameter at a
time**, writing a **prediction before running each experiment**, then
recording what actually happened and whether the prediction was right. This
mirrors how real scientists and engineers work: hypothesize first, then test,
then honestly report whether you were right or wrong (being wrong is just as
valuable a result). Full summary table and analysis in Section 5.

### Bonus — Domain Randomization (Cells 119–123) — **optional 🎯**

Explains that up to this point, the *randomness* in the game only came from
the specific starting seed of each episode (which gap heights, which noise
values) — but the actual *physics* of the world (gravity, flap noise
strength) never changed between episodes. **Domain randomization** means
deliberately varying the physics itself between episodes (e.g., sometimes a
calm "windless" world, sometimes a bumpy "windy" world), so that the trained
AI has to become a policy that works well across a whole *family* of similar
worlds, rather than perfectly memorizing one specific world. This is exactly
the trick real robotics teams use to train robots in simulation that then
work in the unpredictable real world ("sim-to-real transfer"). See Section 6.

### Part 7 — Hand-in (Cells 124–133)

- **Cells 125–126 (code):** The student declares their final chosen
  hyperparameters into `FINAL_CONFIG`. This is the *only* thing that gets
  graded/retrained by the course's automatic grading system, along with the
  three functions from Part 2.
- **Cell 128 (code):** Runs the exact same validation check the grading
  system will run, on the student's own machine, so mistakes are caught
  before submission.
- **Cell 130 (code):** A "dry run" — trains once with the final config and
  compares it to the reference pilot.
- **Cell 132 (code):** Builds a fun, shareable, animated HTML replay of one
  episode flown by the trained AI (a little game you can watch play itself,
  saved as `flappy_trained.html`).

---

## 3. The three 🎯 coding tasks in Part 2 — explained in detail

### 🎯 Task 1 — `build_observation` (Cell 26)

**What problem is this solving?**
A neural network can't "see" the game the way we do — it only accepts
numbers. `build_observation` is the function that converts the raw game
state (bird height, velocity, distances/heights of upcoming pipes) into a
clean list of numbers (a "feature vector") that the network reads as its
input, every single time step.

**The five blanks filled in:**

```python
y = (state["y"] - (C.Y - 1) / 2) / s["y"]     # 1: bird height, centered & scaled
v = state["v"] / s["v"]                        # 2: bird velocity, scaled
...
cols.append(state["dx"][:, i] / s["dx"])       # 3: obstacle i's distance, scaled
cols.append(state["dy"][:, i] / s["dy"])       # 4: obstacle i's gap offset, scaled
cols.append(state["active"][:, i].astype(np.float32))  # 5: is this obstacle real?
```

- **Blank 1 — bird height, re-centered and scaled.** Raw height `y` might be
  a number like `0` to `13`. Neural networks learn much better when their
  inputs are roughly centered around `0` and roughly of size `1` (imagine
  trying to compare "3 apples" against "3000 meters" — the network would
  struggle to weigh them properly without rescaling). So we subtract the
  middle of the grid (`(C.Y - 1) / 2`) to center it around zero, then divide
  by a fixed scale (`s["y"]`) to shrink it to roughly `[-1, 1]`.
- **Blank 2 — velocity, scaled.** Same idea: divide the raw velocity by a
  fixed scale so it's a small, well-behaved number.
- **Blank 3 — obstacle distance (`dx`), scaled.** How far away (in columns)
  the `i`-th upcoming pipe is, scaled the same centering-free way (distance
  is naturally already ≥ 0, so no re-centering needed, just scaling).
- **Blank 4 — obstacle gap offset (`dy`), scaled.** This is `(gap height -
  bird height)` for that obstacle — i.e., "how far above or below the gap
  center am I, relative to this pipe." This is one of the single most
  important numbers for deciding whether to flap.
- **Blank 5 — the "active" flag.** Because the game always reports a fixed
  number of "obstacle slots" even if there aren't that many real obstacles
  nearby yet, empty slots are filled in with a fake placeholder distance
  (`dx = X`, the far edge of the grid). Without telling the network which
  slots are *real* vs *empty placeholders*, the network cannot tell the
  difference between "a real obstacle that happens to be very far away" and
  "there is no obstacle here at all" — both would otherwise look identical.
  Passing along this yes/no flag (as a `0.0` or `1.0` number) removes that
  ambiguity.

**Why fixed scales, not "running/adaptive" statistics?** The instructions
explicitly forbid normalizing using statistics computed *during* training
(like "the average height seen so far"). If you did that, the same physical
situation would be represented by *different* numbers depending on how far
along training you are — the network's understanding of the world would keep
shifting under its feet, and a saved model wouldn't even be usable again
later, since the meaning of its inputs would depend on training history that
isn't saved. Using fixed constants (`scales(C)`) keeps every checkpoint
meaningful and portable.

### 🎯 Task 2 — `compute_gae` (Cell 28)

**What problem is this solving?**
When the bird gets a reward, RL needs to figure out **which past actions
deserve credit** for that reward — this is called the **credit assignment
problem**. `compute_gae` implements **Generalized Advantage Estimation
(GAE)**, a formula that spreads credit backward through time in a smart,
tunable way, producing an **advantage** number for every single time step:
"was this action, at this state, better or worse than what the AI's current
`value estimate` expected?"

**The three blanks filled in:**

```python
delta = rewards[t] + gamma * next_value * non_terminal - values[t]          # 1
last_gae = delta + gamma * lam * non_terminal * last_gae                    # 2
return advantages, advantages + values                                     # 3
```

Think of it like a relay race passing a baton **backward** through time,
from the last step of the rollout to the first:

- **Blank 1 — the TD (temporal-difference) error, `delta`.** In plain
  words: "reward I actually got, plus my prediction of how good the *next*
  situation is (discounted a bit by `gamma` because the future is worth
  slightly less than the present) — minus what I *originally guessed* this
  situation was worth." If `delta` is positive, this step turned out better
  than expected; if negative, worse than expected.
- **Blank 2 — the backward accumulation, `last_gae`.** This is the heart of
  GAE: it doesn't just use this one step's surprise (`delta`), it also
  carries forward a *fading echo* of all the surprises from later steps
  (`last_gae` from step `t+1`), shrunk by `gamma * lam` each step further
  back in time. `lam` (lambda) controls how far that echo travels: `lam=0`
  means "only trust the very next step's surprise" (short memory, but noisy
  less), `lam=1` means "trust the entire rest of the episode equally" (long
  memory, more accurate but noisier).
- **The `(1 - dones[t])` mask, used *twice*.** This is the single most
  important detail in the whole function, called out explicitly in the
  instructions. `dones[t]` is `1` if the episode ended (crash) *on* step
  `t`. If you forget the mask even once, credit or blame from a *brand new,
  unrelated episode* (which starts right after a crash) could accidentally
  leak backward into the episode that just ended — like accidentally
  blaming yesterday's mistake for something that happened today. The test
  cell (Cell 31) specifically checks this: it forces a `done` at step 2 and
  a big reward at step 3, then asserts that **none** of that reward's credit
  leaked back across the done boundary (`adv[:3, 0]` must all be `0`).
- **Blank 3 — the returns the critic learns from, `advantages + values`.**
  The "critic" part of the network is trained to predict, for any state,
  "how much total future reward should I expect from here?" That target
  number is exactly the advantage (how much better than expected) plus what
  was already expected — added back together, you get the *actual* total
  expected future value, which is what the critic tries to match.

**Sanity checks that prove it's right (Cell 31):**
1. Credit does not leak across a `done` (explained above).
2. With `values = 0` everywhere (no critic help at all) and `lam = 1`, the
   advantage exactly equals the plain **Monte Carlo return** — the true,
   textbook-simplest way to add up all future discounted rewards. This is a
   known mathematical special case of GAE, and matching it proves the
   formula was implemented correctly.

### 🎯 Task 3 — `ppo_loss` (Cell 30)

**What problem is this solving?**
This is the actual "learning signal" — the single number the network tries
to make smaller and smaller during training (the "loss"). Making it smaller
nudges the network's weights in the direction of "take actions that led to
better-than-expected outcomes; predict values more accurately; keep
exploring a bit."

**The six blanks filled in:**

```python
ratio = torch.exp(log_ratio)                                                    # 1
pg_2 = -advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)    # 2
pg_loss = torch.max(pg_1, pg_2).mean()                                          # 3
v_loss = 0.5 * ((new_value - returns) ** 2).mean()                              # 4
loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent                       # 5
approx_kl = ((ratio - 1.0) - log_ratio).mean().item()                           # 6
```

- **Blank 1 — the probability ratio, `ratio = exp(log π_new − log π_old)`.**
  PPO reuses the same batch of collected experience for several rounds of
  learning (`update_epochs`). Each round, the policy shifts slightly, so
  "how likely was this action *before* I started this round of updates"
  versus "how likely is it *now*" can differ. `ratio` measures exactly that
  shift: `ratio = 1` means the policy hasn't changed its mind about this
  action at all; `ratio > 1` means it now likes the action more; `ratio < 1`
  means it likes it less.
- **Blank 2 — the clipped version of the surrogate, `pg_2`.** This is PPO's
  signature trick: `pg_1` (already given) is `-advantage * ratio` — a plain,
  unclipped estimate of "how much better/worse things got." But if `ratio`
  swings wildly (the policy changed its mind a lot), trusting it fully can
  send training off a cliff. So `pg_2` computes the *same* quantity but with
  `ratio` clamped ("clipped") to a narrow trust region
  `[1 - clip_coef, 1 + clip_coef]` (e.g. `[0.8, 1.2]` for `clip_coef = 0.2`)
  — it refuses to reward the network for moving the policy *too far* in one
  update.
- **Blank 3 — the pessimistic combination, `torch.max(pg_1, pg_2).mean()`.**
  Because both `pg_1` and `pg_2` already carry the *minus* sign (turning
  "maximize reward" into "minimize loss"), taking the `max` of the two
  (rather than `min`) picks the more pessimistic (worse-for-the-network,
  i.e., more conservative) of the two options. This is the subtle sign trap
  the notebook's hint calls out explicitly: in the original PPO paper the
  formula is written as `min` of the *un-negated* quantities, which becomes
  `max` once you flip the sign to turn it into something to minimize.
- **Blank 4 — the critic's own loss, `v_loss`.** Simple mean-squared-error
  between the network's predicted value and the actual computed return
  (from Task 2): the standard way to train something to predict a number.
- **Blank 5 — combining everything into one loss, `loss`.** Add the policy
  loss (make good actions more likely), add a fraction (`vf_coef`) of the
  value loss (make the critic more accurate), and *subtract* a fraction
  (`ent_coef`) of the **entropy** (how "spread out"/unpredictable the
  action probabilities are). Subtracting entropy in the loss means the
  network is rewarded for *not* becoming too certain too quickly — it keeps
  a bit of healthy randomness/exploration alive, so it doesn't prematurely
  lock itself into a bad habit (like the "never flap" trap from Part 1).
- **Blank 6 — a well-behaved way to measure how much the policy moved,
  `approx_kl`.** KL divergence measures "how different are the new and old
  action-probabilities." The naive way to estimate it, `-log_ratio.mean()`,
  can accidentally come out *negative* (which makes no sense for something
  meant to represent "distance" between two distributions) and is noisier.
  The formula used here, `(ratio - 1) - log(ratio)`, mathematically
  guarantees a non-negative result and has much lower statistical noise —
  it's the estimator recommended by John Schulman (one of PPO's inventors).
  This number becomes the single most important dashboard reading in Part 3
  (Calibration Task C3): if it's too high, the policy is changing too fast
  per update (dangerous); if it's too low, you're wasting your collected
  data by barely using it.

**Sanity check (Cell 31):** if the old and new policies are identical
(`ratio = 1` everywhere), the clipping should do nothing, the KL should be
essentially zero, and — with entropy and value-loss coefficients turned off
— the loss should also be essentially zero. This confirms the formula
collapses correctly to "no update needed" when nothing has changed.

---

## 4. The three 🎯 calibration tasks in Part 3

The philosophy here: **never guess a hyperparameter — compute a range for it
from a measured fact about the game, then verify with a short experiment.**
The measured fact used throughout is **Delta** (average number of steps
between two pipes), computed in Cell 12.

### 🎯 C1 — Discount factor `gamma` (Cells 33–36)

**The idea, for a kid:** `gamma` controls how much the AI cares about
rewards that happen *later* versus *right now*. A `gamma` close to `1` means
"I care almost as much about tomorrow's treat as today's"; a `gamma` close
to `0` means "give me the treat now, I don't care about tomorrow."

**The two competing constraints:**
- *From below:* the number `1 / (1 - gamma)` roughly measures "how many
  steps into the future does the AI actually pay attention to" (its
  **effective horizon**). If this horizon is *shorter* than `Delta` (the
  spacing between pipes), the AI literally cannot "see" far enough ahead to
  connect "I should flap now" with "I passed the pipe later" — it will never
  learn to plan for the pipe at all.
- *From above:* since every surviving step gives `+1` reward, the total
  value the critic must learn to predict is roughly `1 / (1 - gamma)` as
  well. If `gamma` is too close to `1` (like `0.999`), that number becomes
  huge (`1000`), and the critic — which starts out predicting values near
  `0` — has a very hard, very noisy job trying to reach `1000`. This
  dominates and destabilizes training.

**Chosen value:** `GAMMA = 0.99` (Cell 35), giving a horizon of `100` steps
— comfortably above `Delta` (~14 on `large`) while staying well below the
point where the value scale explodes. The written conclusion (Cell 36)
explicitly reasons about this trade-off: farsighted enough to plan for
obstacles, but not so farsighted that training becomes noisy.

### 🎯 C2 — Rollout length `T` and number of parallel environments `N` (Cells 37–40)

**The idea, for a kid:** Training happens in "batches" of `N × T` numbers:
`N` is how many bird-games you run *at the same time* (in parallel, like
having many copies of the game running side-by-side), and `T` is how many
steps you record from each one before pausing to learn. It is tempting to
think only the *product* `N × T` (the total batch size) matters, but that's
wrong:

- **`T` controls whether a "decide to climb → pass the pipe" story can even
  fit inside a single recorded rollout.** GAE (Task 2) can't assign credit
  across events that happened in *different* recorded batches — its memory
  is truncated at `T` steps (beyond that, it can only use the bootstrapped
  value estimate, which is itself still being learned and imperfect). If
  `T < Delta`, no single recording window ever contains a complete
  "obstacle" story.
- **`N` controls how noisy the gradient estimate is** (more parallel games
  running = averaging over more experience = less noisy learning signal),
  scaling like `1/sqrt(N*T)`.

**Chosen values:** `ROLLOUT (T) = 32`, `NUM_ENVS (N) = 256` (Cell 39). Since
`Delta ≈ 14` on `large`, the rule of thumb "`T ≳ 2 × Delta`" suggests `T`
should be at least `~28`; `32` satisfies that comfortably. `N = 256` was then
picked as a reasonable batch size for controlling gradient noise, verified
against alternative `(N, T)` pairs at the same total batch size.

### 🎯 C3 — GAE lambda, learning rate, and epoch count (Cells 41–46)

Two more numbers, **read off measurements rather than chosen by taste**:

**GAE `lambda`.** The *effective* credit horizon of GAE (not the raw
rollout length `T`, but how far GAE's backward-echo actually reaches) is
`1 / (1 - gamma * lambda)`. The notebook computes this for several candidate
`lambda` values and checks which ones land inside the useful target band
`[Delta, 2*Delta]`. The pleasant surprise highlighted in Cell 43: `lambda =
0.95` — the value almost every tutorial and paper uses as a "default" — is
*exactly* the value that lands inside this target band for this specific
game. In other words, a number that looks like an arbitrary convention
turns out to be *derivable* from first principles about the task's timing.
**Chosen value:** `GAE_LAMBDA = 0.95`.

**Learning rate (`lr`) and number of epochs (`update_epochs`).** These
control how big a step the network takes with each update, and how many
times it re-reads the same batch of data before throwing it away. Instead of
tuning these by just watching whether the score goes up, the notebook tunes
them by watching the **approximate KL divergence** metric computed in
`ppo_loss` (Task 3, blank 6). The target band is `0.01`–`0.02`:
- Measured KL too high (> `0.05`) → the policy is moving too far each
  update; the collected data becomes "stale"/off-policy. Fix: reduce epochs
  *before* touching the learning rate.
- Measured KL too low (< `0.003`) → the update is barely nudging the policy
  at all; you're wasting the batch you worked hard to collect. Fix: increase
  epochs or the learning rate.

Cell 44 runs two concrete configurations and measures the actual KL: the
"textbook default" (`lr=3e-4, epochs=4`) gives a KL of about `0.0005` —
twenty times *below* the target band — and the agent stalls, reaching only
about 48% of the optimum. A much larger setting (`lr=3e-3, epochs=10`)
reaches 94% of the optimum. This is presented as proof that **the KL metric
told the true story before the score curve did** — a lesson in using the
right diagnostic, not just "try things and watch the score."

**Chosen values:** `LR = 0.003`, `EPOCHS = 10` (Cell 45). The written
conclusion (Cell 46) explains that although the *measured* KL at these
settings (`~0.00187`) is technically still below the textbook target band,
empirically this configuration is the one known to reach 94% of the optimum
— i.e., in this instance the practical outcome was weighted over a strict
reading of the diagnostic band.

---

## 5. Part 6 — The tuning table: 20 one-factor-at-a-time experiments

The rules the student followed:
1. Change **one** hyperparameter at a time (never two at once — otherwise
   you can't tell which change caused what).
2. Write the **prediction before running** the experiment (this prevents
   "hindsight bias" — quietly deciding what you "expected" only after
   seeing the answer).
3. A promising configuration gets **re-run on 3 seeds** and reported as
   mean ± standard error, so a lucky/unlucky single run isn't mistaken for a
   real effect.
4. **A hypothesis that turns out wrong is just as valuable a result** as one
   that turns out right — both are recorded honestly, with a discussion.

The base configuration being perturbed here is `Config.for_large()`, whose
default pipe count baseline is roughly **~53.8 pipes**.

| # | Hyperparameter changed | Predicted direction | Measured pipes (mean ± SE) | Confirmed? |
|---|---|---|---|---|
| 1 | `gamma=0.95` | somewhat fewer pipes | 35.17 ± 0.91 | direction yes, magnitude bigger than expected |
| 2 | `gamma=0.90` | clearly fewer than run 1 | 34.81 ± 0.59 | no — plateaued, didn't drop further |
| 3 | `gamma=0.999` | below base, high variance | 30.71 ± 3.98 | yes — lowest so far *and* largest error bar |
| 4 | `rollout=8` | sharp drop | *(failed — config error; re-run got 55.54 ± 2.68)* | inconclusive first try |
| 5 | `rollout=16` | better than T=8, still below base | 48.95 ± 9.38 | partially — high variance too |
| 6 | `rollout=128, num_envs=64` (batch fixed) | comparable to base | 37.54 ± 6.02 | no — clearly worse |
| 7 | `num_envs=64` (T fixed) | somewhat fewer pipes | 55.59 ± 3.89 | no — about equal to/above base |
| 8 | `num_envs=512` | comparable/better | 28.81 ± 2.34 | no — much worse (fewer gradient updates in the fixed step budget) |
| 9 | `gae_lambda=0.90` | fewer pipes (below Delta horizon) | 57.77 ± 2.68 | no — actually the **best** of runs 1–10 |
| 10 | `gae_lambda=0.99` | similar/below base (too-long horizon) | 28.59 ± 3.24 | yes — clearly hurt |
| 11 | `lr=3e-4` | well below base, stalled | 17.30 ± 1.15 | yes — lowest/tightest of the whole table |
| 12 | `lr=3e-2` | well below base, unstable | 41.12 ± 7.87 | partially — down, but not "well below"; high variance as predicted |
| 13 | `update_epochs=2` | below base (underfitting) | 25.41 ± 2.11 | yes |
| 14 | `update_epochs=20` | below base (overfitting) | 58.43 ± 3.53 | no — **best result across all 20 runs** |
| 15 | `ent_coef=0.001` | sharp drop toward "never flap" | 41.00 ± 0.60 | partially — down but not collapsed |
| 16 | `ent_coef=0.1` | comparable/below base | 61.61 ± 1.88 | no — **best single result of the whole table** |
| 17 | `vf_coef=0.1` | below base | 51.08 ± 4.06 | roughly — small/within-noise effect |
| 18 | `vf_coef=2.0` | below base (critic dominates) | 28.87 ± 4.24 | yes — clear drop |
| 19 | `clip_coef=0.1` | below base (too cautious) | 29.24 ± 2.08 | yes |
| 20 | `clip_coef=0.3` | below base (too loose/unstable) | 56.46 ± 2.70 | no — at/above base |

**The overall lesson drawn across all 20 runs (Cell 118):** the "textbook"
direction (too high = unstable, too low = underfit) held at the *extreme*
ends (e.g. Run 11's very low `lr` stalling badly, Run 18's high `vf_coef`
collapsing), but there was a consistent, *not originally predicted*
**asymmetry**: being *too cautious/conservative* (low `lr`, few epochs, tight
clipping, low entropy, low value-loss weight) hurt performance more
reliably than being *too aggressive* in the opposite direction — and some
"aggressive" settings (`update_epochs=20`, `ent_coef=0.1`) actually beat the
baseline outright. The two clear exceptions to that overall pattern were the
rollout/batch-size runs (6–8) and the `gae_lambda` runs (9–10), where the
*direction* the base setting should move in wasn't simply "more aggressive
is better" — instead, an intermediate value found genuinely better than
either extreme.

---

## 6. Bonus: 🎯 Domain Randomization (Cells 119–123)

**The idea, for a kid:** Imagine you only ever practiced riding a bike on a
perfectly flat, windless day. You'd get very good at *that specific day*,
but the first time it's windy, you might wobble and fall — because you never
practiced with wind. **Domain randomization** is like *deliberately*
practicing on windy days, rainy days, and bumpy roads too, so you become a
rider who's *robust* to whatever conditions show up, instead of a rider
who's only good at one exact condition.

In this game, the "wind" is the noise on the strong flap, `V_dev`. Up to
this point in the notebook, `V_dev` was fixed at `1` for the entire training
run. The bonus task asks: what if, instead, every new episode randomly drew
its own `V_dev` from `{0, 1, 2}` (calm / normal / windy), and the agent had
to learn a policy that works reasonably well across *all three*, rather than
perfectly optimizing for just one?

**The 🎯 implementation (Cell 120), `sample_episode_physics`:**

```python
kw.update(
    g=int(base.g * s),
    V_max=int(base.V_max * s),
    U_weak=int(base.U_weak * s),
    U_strong=int(base.U_strong * s),
)
...
kw["V_dev"] = v_dev
```

This function is called once at the start of every episode. It:
1. Optionally draws a random overall "vertical scale" `s` from
   `dr.vertical_scales` and applies it consistently to gravity (`g`), the
   maximum velocity cap (`V_max`), and both flap strengths (`U_weak`,
   `U_strong`) — since these all need to scale *together* to keep the
   physics internally consistent (e.g., doubling gravity without doubling
   the flap strength would make the game unfairly harder).
2. Optionally draws a random `V_dev` (the flap-noise half-width) from
   `dr.v_dev_choices` — this is the actual "windiness" knob demonstrated in
   the notebook's brittleness demo.
3. Uses the episode's own seeded random generator (`rng`) for the draw, so
   the *same* episode seed always reproduces the *same* physics — keeping
   experiments reproducible even though physics now varies episode to
   episode.
4. Returns an unmodified `base` if nothing was configured to randomize
   (so normal, non-DR training is completely unaffected).

**The demo (Cell 123):** trains one agent normally (fixed `V_dev = 1`) and
one agent with domain randomization (`V_dev` drawn from `{0, 1, 2}` every
episode), then evaluates *both* agents on all three `V_dev` settings. The
expected finding: the nominally-trained agent does great at `V_dev=1` (what
it was trained on) but may do noticeably worse at `V_dev=0` or `V_dev=2`
(conditions it never saw), while the DR-trained agent is more evenly
competent across all three — a smaller peak performance, perhaps, but far
more *robust*. This is exactly the trade-off real-world robotics and
self-driving-car teams navigate: a policy tuned to one simulator is often
"brittle" once the real world doesn't match the simulator exactly, and
domain randomization is one of the standard fixes.

---

## 7. Part 7 — Final hand-in configuration (Cells 124–133)

The chosen final hyperparameters (Cells 125–126), assembled from the earlier
calibration (Part 3) and the tuning-table findings (Part 6):

| Hyperparameter | Value | Where it came from |
|---|---|---|
| `gamma` | `0.99` | Initial calibration (C1) — kept, since Runs 1–3 in the tuning table only ever made it worse |
| `gae_lambda` | `0.90` | Run 9 — the single best-performing tuning-table result for this knob |
| `lr` | `0.003` | Base calibrated value (C3) |
| `update_epochs` | `20` | Run 14 — best-performing tuning-table result for this knob |
| `rollout (T)` | `32` | Base calibrated value (C2) |
| `num_envs (N)` | `256` | Base calibrated value (C2) |
| `ent_coef` | `0.1` | Run 16 — best single result across the *entire* 20-run table |
| `clip_coef` | `0.3` | Run 20 — best-performing tuning-table result for this knob |

This is essentially "take the calibrated base config, then swap in every
individual knob's best-performing value found during the 20-run sweep" —
a reasonable, evidence-based way to assemble a final configuration when a
full grid search (trying every *combination*) was explicitly ruled out by
the 20-run budget.

Cells 128–130 then double-check that this configuration is valid (passes
the same checks the grading harness will run) and give it one real dry-run
training pass, comparing the result against the `LookaheadPolicy` reference
pilot. Cell 132 renders the resulting trained agent as a fun, shareable,
animated replay (`flappy_trained.html`) that can be opened and watched like
a mini video game, independent of the notebook.

---

## 8. Glossary — every parameter, explained like you're explaining it to a
   ten-year-old

For each entry: **what it is**, **a real-life analogy**, **what happens if
it's too big or too small**, and **the actual number(s) used in this
notebook**, so it's not just abstract.

### 8.1 The game itself (the "world" the bird lives in)

- **State** — a snapshot of *everything about the game right now*: how high
  up the bird is, how fast it's moving up or down, and where the next few
  pipes are. Think of it as one single freeze-frame photo of the game that
  contains every fact you'd need to decide what to do next. Nothing about
  the *past* matters once you have the state — it's a complete "right now."

- **Action** — the one thing the bird is allowed to *do* on a given turn.
  There are exactly 3 choices, like 3 buttons on a controller:
  `0` = do nothing, `1` = weak flap, `2` = strong flap. The bird must press
  exactly one button every single turn (even "do nothing" counts as a
  choice).

- **`y` (bird height)** — literally which row of the grid the bird is
  sitting in, counted from the floor. `y = 0` is the floor, and it goes up
  from there. On the `small` grid, `y` ranges from `0` to `13` (14 rows
  total); imagine 14 shelves stacked up, and the bird sits on one shelf at
  a time.

- **`v` (vertical velocity)** — how fast, and in which direction, the bird
  is currently moving. Positive `v` means "going up," negative `v` means
  "going down," and `v = 0` means "momentarily still." It's exactly like the
  speedometer needle on a car, except it can point backward (negative) too.
  It is capped at `V_max` in each direction so the bird can never move
  impossibly fast.

- **`V_max`** — the *speed limit* on the bird's velocity, in both
  directions. However hard you flap or however long you fall, `v` is never
  allowed to go above `+V_max` or below `-V_max`. On `small`, `V_max = 2`.
  Think of it as a bird that physically cannot flap harder than "this fast"
  no matter what, and cannot fall faster than "this fast" either — like a
  parachute that stops you from ever free-falling infinitely quickly.

- **`g` (gravity)** — how much the bird's velocity decreases *every single
  turn*, automatically, whether you flap or not — exactly like real gravity
  constantly pulling you toward the ground. On `small`, `g = 1`, meaning
  every turn, 1 unit gets subtracted from `v` before anything else happens.
  It's the "tax" the bird always has to pay just for existing.

- **The three actions and their impulses (`U_weak`, `U_strong`) and costs
  (`λ`, lambda-cost)** —
  - *No flap*: pushes velocity by `0` (no help at all), costs `0` reward.
    Free, but useless against gravity.
  - *Weak flap* (`U_weak = +2` on `small`): a small, **always exactly the
    same size**, reliable upward push — like a gentle, practiced hop you've
    done a thousand times and always land the same way. Costs `0.5` reward
    (paid immediately, like spending a coin to make the hop).
  - *Strong flap* (`U_strong = +3` on `small`, **plus random noise**): a
    bigger push, but **not perfectly predictable** — like jumping on a wet,
    slightly slippery trampoline: usually you go about as high as you
    expect, but sometimes a little higher, sometimes a little lower. Costs
    `0.7` reward — the most expensive move, because it's the most powerful.

- **`V_dev` (strong-flap noise / "how windy is it")** — how *unpredictable*
  the strong flap's push is. The actual push isn't always exactly
  `U_strong`; it's `U_strong + noise`, where `noise` is a random whole
  number between `-V_dev` and `+V_dev` (inclusive), picked fresh every time
  you strong-flap. With `V_dev = 1` (the normal setting used through most of
  the notebook), the real push on any given strong flap is `U_strong - 1`,
  `U_strong`, or `U_strong + 1`, each roughly equally likely — like a dice
  roll added onto your jump. In the domain-randomization bonus, `V_dev` is
  itself randomized episode-to-episode between `0` (perfectly calm, no
  randomness at all — a "windless" day), `1` (normal), and `2` (extra gusty
  — a "windy" day where your jumps can be thrown off by up to 2 units).

- **Reward** — the "points" earned each turn: `reward = 1 - cost of the
  action you picked`. So doing nothing earns a full `+1`; a weak flap earns
  `+0.5`; a strong flap earns only `+0.3`. It's like a video game that gives
  you a coin every second you're alive, but flapping costs some of that
  coin as "fuel" — so the smart strategy is to flap only when you truly
  need to, not for fun.

- **Crash / episode end** — if, right when a pipe reaches the bird's column,
  the bird's height `y` is *not* close enough to the gap's center `h`
  (specifically, more than `gap_half` rows away), the bird crashes and the
  game (the "episode") ends immediately, with no more reward from that
  point on.

- **Pipe / obstacle, and its two numbers `d` (distance) and `h` (gap
  height)** — every pipe the bird will encounter has a distance `d` (how
  many columns away it currently is — like "3 steps ahead of you") and a
  gap height `h` (which row the safe hole in the pipe is centered on).
  Every single turn, every pipe's distance `d` shrinks by exactly `1`
  (pipes always march toward the bird at a constant, predictable speed —
  the *only* unpredictable things in this whole game are the strong-flap
  noise and where new gaps appear).

- **`gap_half` (half-width of the safe gap)** — how many rows above *and*
  below the gap's center `h` still count as "safely through the gap." If
  `gap_half = 1`, then being at `h-1`, `h`, or `h+1` all count as safe; any
  further away is a crash. Bigger `gap_half` = a more forgiving, easier
  game (a wider doorway to fly through); smaller `gap_half` = a stricter,
  harder game (a narrower doorway).

- **`X` (grid width) and `Y` (grid height)** — simply how many columns
  (`X`) and rows (`Y`) the whole discrete grid has. `small` uses `X=8,
  Y=14`; `large` uses a much finer grid (`4×` wider, `8×` taller), which is
  exactly why it has so vastly many more possible states.

- **`M` (how many pipes are tracked at once)** — the game doesn't just care
  about the very next pipe; it keeps track of the next `M` pipes coming up
  (their distances `d_1...d_M` and gap heights `h_1...h_M`), so a smart
  pilot can plan a little further ahead than just "the very next one."

- **`D_min` (minimum spacing between pipes)** — the smallest allowed gap (in
  columns) between one pipe and the next. This guarantees pipes don't spawn
  unfairly close together, back-to-back, with no time to react.

- **`S_h` (the list of allowed gap heights)** — instead of a gap being able
  to appear at *any* height at all (infinitely many possibilities), each new
  pipe's gap-center `h` is drawn from a short fixed list of allowed heights
  (e.g. `(5, 9, 13)` on `small`). This is one of the tricks that keeps the
  `small` grid's total number of states low enough to solve exactly.

- **`|S|` (size of the state space)** — the *total count* of every possible
  distinct situation the game could ever be in (every combination of `y`,
  `v`, and all the pipes' distances/heights). On `small`, `|S| ≈ 7,770` —
  small enough to write down in one big table. On `large`, `|S| ≈ 10^17`
  (a hundred million billion!) — far too many to ever list, which is the
  entire reason this notebook needs a *learning* algorithm (PPO) instead of
  just looking up the perfect answer in a table.

- **Seed** — a single starting number fed into the random-number generator.
  Using the exact same seed always produces the exact same sequence of
  "random" events (same noise values, same gap heights) — like re-watching
  the *identical* recording of a dice-rolling machine instead of rolling
  real dice again. This is essential for fair comparisons: if you want to
  know whether *your change* made things better or the game just happened
  to be easier that one time, you need the "randomness" to be identical
  across the comparison.

### 8.2 Big-picture reinforcement learning concepts

- **Policy** — the AI's strategy/rulebook: "given what I currently see (the
  state), what should I do (the action)?" In this notebook the policy is a
  neural network — a big adjustable math formula whose numbers ("weights")
  get nudged bit by bit during training until the formula reliably outputs
  good decisions.

- **Value function (`V`, "the critic")** — a *second* thing the network
  learns to predict: "starting from this exact situation, roughly how much
  total future reward should I expect to collect from here on, if I keep
  playing sensibly?" It's like a fortune-teller that, just by looking at the
  current situation, guesses your total future score — useful because it
  lets the AI judge "did that last move make my situation better or worse
  than I expected?" without having to wait until the entire game is over.

- **Episode** — one complete playthrough, from the very start of the game
  until the bird crashes (or a maximum time limit is reached). Training
  happens across *thousands* of episodes.

- **Rollout** — a *recording* of some number of consecutive turns played by
  the current policy (states, actions, rewards, all logged), which is then
  used as the "homework" the network studies to improve itself.

- **Bellman equation** — a mathematical rule stating: "the true value of
  being in a certain situation equals the reward you get right now, plus a
  slightly-discounted version of the true value of wherever you end up
  next — assuming you always pick the best possible action." It's a bit
  like saying "how good is my current spot in a maze equals one step's
  worth of progress, plus how good the next spot is." If you know this rule
  holds *exactly* for every single situation, you can solve for every
  situation's true value at once.

- **Value iteration** — the exact, table-based method used on `small` to
  solve the Bellman equation: start by guessing every situation is worth
  `0`, then repeatedly apply the Bellman rule to update every entry in the
  table, over and over, until the numbers stop changing. Because `small`
  has "only" ~7,770 situations, a computer can do this in about 2 seconds
  and get the mathematically *exact*, perfect answer — no guessing involved.

- **Curse of dimensionality** — the (very real, not exaggerated) fact that
  as you make a simulated world more detailed (finer grid, more pipes
  tracked, more velocity levels), the *number of possible situations*
  doesn't grow gently — it multiplies together and explodes, going from
  thousands to quintillions extremely quickly. This is why `large` needs
  ~100,000,000,000,000,000 table entries — literally too many atoms of
  memory to ever store, let alone compute.

- **PPO (Proximal Policy Optimization)** — the specific recipe used to
  *approximate* good behavior with a neural network, for situations (like
  `large`) where the exact table-based method is impossible. "Proximal"
  (meaning "nearby/close") refers to PPO's core safety trick: never let one
  single training update change the policy's behavior *too drastically* —
  always keep the new policy "close to" the old one, so learning stays
  smooth and doesn't suddenly go haywire.

- **Advantage** — a single number, computed for every past moment, that
  answers: "was that particular action, in that particular situation,
  better or worse than what I already expected from my value function?" A
  positive advantage means "do more of that"; a negative advantage means
  "do less of that." This is literally the signal that tells the network
  which past decisions to reinforce and which to discourage.

- **GAE (Generalized Advantage Estimation)** — the specific formula (Task 2
  in this notebook) used to compute the advantage above, in a way that
  smartly balances "trust what actually happened" against "trust the
  value-function's predictions," using the `gae_lambda` knob (see below) to
  control that balance.

- **Domain Randomization** — deliberately varying the *rules of the world
  itself* (like gravity or flap-noise strength) from one practice episode to
  the next, so the trained policy becomes good at *many* slightly different
  versions of the game instead of perfectly memorizing just one exact
  version — like practicing basketball with slightly different, randomly
  under- or over-inflated balls, so a real, normally-inflated ball on game
  day doesn't throw you off.

### 8.3 The training "dials" (hyperparameters) — one at a time

Each of these is a single number you can turn up or down before training
starts. Getting them right is most of what Parts 3 and 6 of the notebook
are about.

- **`gamma` (γ, the discount factor) — "how much do I care about tomorrow?"**
  A number strictly between `0` and `1` controlling how much a reward
  *later* is worth compared to the *same* reward *right now*. Picture two
  friends offering you a cookie: one gives it to you now, the other
  promises the same cookie in an hour. `gamma` close to `1` (like `0.99`,
  used here) means "I'll happily wait, a cookie later is worth almost as
  much as a cookie now" — a *patient*, farsighted bird. `gamma` close to `0`
  means "I only care about right now, forget the future" — a very
  *impatient* bird that would rather grab tiny rewards immediately and
  never plan ahead for a pipe. **Too low:** the bird can't connect "flap
  now" with "pipe passed several turns later" — it never learns to plan.
  **Too high** (like `0.999`): the total future score to predict becomes
  enormous (`1000`+), which makes learning slow and wobbly. *Chosen value
  in this notebook: `0.99`.*

- **`Delta` — "how far apart are the pipes, on average?"** Not a dial you
  set, but a *measured fact* about the game (how many turns typically pass
  between one pipe and the next) that is used as a ruler to sanity-check
  nearly every other dial below. If a dial's "memory span" is shorter than
  `Delta`, the bird literally cannot connect actions to their pipe-passing
  consequences.

- **`rollout` / `T` — "how long a movie clip do I record before stopping to
  think?"** How many consecutive turns get recorded together in one
  practice "clip" before the network pauses to learn from them. If `T` is
  shorter than `Delta`, no single recorded clip ever contains a complete
  "flap to climb → pass the pipe" story — like trying to learn to catch a
  ball from video clips that are cut off before the ball ever lands. *Rule
  of thumb used here: `T` should be at least about `2 × Delta`. Chosen
  value: `32`.*

- **`num_envs` / `N` — "how many copies of the game am I practicing on at
  once?"** Instead of playing one game at a time, the computer runs many
  identical copies of the Flappy Bird game *simultaneously* (like `N`
  identical kids all practicing free-throws at the same time on `N`
  side-by-side courts), collecting experience from all of them in parallel.
  More copies = a smoother, less noisy learning signal (because you're
  averaging lessons from many independent tries at once, rather than
  learning too much from one possibly-lucky or possibly-unlucky game).
  **Too few:** noisy, jittery learning. **Too many:** each "batch" of
  learning takes longer to collect and you get fewer separate learning
  updates for the same total practice time. *Chosen value: `256`.*

- **Batch size (`N × T`)** — the total number of individual turns collected
  together before one round of learning happens — simply the number of
  parallel games (`N`) multiplied by how long each recorded clip is (`T`).
  On the final config, `256 × 32 = 8,192` recorded turns per learning round.

- **`gae_lambda` (λ, GAE's own separate memory dial) — "how far back should
  credit for a good outcome travel?"** Different from `gamma`. While
  `gamma` asks "how much do later rewards matter," `gae_lambda` asks "when
  estimating whether a *past* action was good, how much should I trust my
  own value-function's guess (short memory, `lambda` near `0`) versus the
  actual, fully-played-out outcome (long memory, `lambda` near `1`)?" A
  useful way to picture it: `lambda` near `0` is like judging a chess move
  only by "did the very next move go well?" — quick but easily fooled;
  `lambda` near `1` is like judging it only by "did I win the *entire* game
  many moves later?" — accurate in principle, but very noisy because so
  much else happened in between. GAE blends the two smoothly. The combined
  reach `1 / (1 - gamma × lambda)` should land between `Delta` and `2 ×
  Delta`. *Interesting finding in this notebook: the common default,
  `lambda = 0.95`, happens to satisfy exactly this rule for this specific
  game — a "one-size-fits-all default" that turns out to be mathematically
  justified here, not just copied blindly.* Final chosen value: `0.90`
  (found to perform best in the 20-run sweep).

- **`lr` (learning rate) — "how big a step do I take each time I learn
  something?"** Every time the network updates itself based on what it just
  learned, `lr` controls *how much* to change its internal numbers. Imagine
  adjusting a shower's temperature knob: a *tiny* `lr` is like turning the
  knob a millimeter at a time — very safe, but it might take forever to
  reach a comfortable temperature (the network "stalls," barely improving).
  A *huge* `lr` is like yanking the knob wildly — you might swing straight
  past "just right" into "too hot," overshooting and destabilizing what
  you'd already learned. This notebook doesn't guess `lr` by feel; it reads
  a special dashboard number (`approx_kl`, "how much did the policy's
  behavior actually shift after this update?") and tunes `lr` and the
  epoch count until that number lands in a healthy target range
  (`0.01`–`0.02`). *Chosen value: `0.003`.*

- **`update_epochs` — "how many times do I re-study the same homework
  before throwing it away?"** After recording one batch of experience,
  the network doesn't learn from it just once — it re-reads that *same*
  batch of data `update_epochs` times, refining its understanding a little
  more each pass, like re-reading the same chapter of a textbook several
  times before moving on to a new chapter, instead of skimming it once.
  **Too few passes:** you waste most of the value in the data you worked
  hard to collect (under-studying). **Too many passes:** the network starts
  overfitting to *that one specific batch* of experience, drifting away
  from genuinely new, fresh data (over-studying the same practice test
  until you've basically memorized its specific answers, not the general
  skill). *Chosen value: `20`* (found, somewhat surprisingly, to help more
  than hurt in this game's tuning-table sweep).

- **`ent_coef` (entropy coefficient) — "how much do I reward myself for
  staying curious/unpredictable?"** The network could, in theory, become
  extremely confident and always pick the exact same action in a given
  situation ("I always know best!") — but early on, that confidence is
  often *wrong*, and it can get permanently stuck in a bad habit (like the
  "never flap" trap). `ent_coef` adds a small bonus to the training signal
  for keeping some healthy randomness/curiosity alive in its choices — like
  gently rewarding a kid for still occasionally trying a different route to
  school instead of always taking the exact same one, just in case a better
  route exists. **Too low:** the AI can lock into a bad habit too early and
  never escape it. **Too high:** the AI stays too random/scattered for too
  long and struggles to settle into a good, confident policy at all.
  *Chosen final value: `0.1`* (this game's obstacle-avoiding-the-never-flap-
  trap nature specifically benefits from extra encouraged exploration).

- **`vf_coef` (value-function loss coefficient) — "how much attention do I
  split between learning to *act well* versus learning to *predict well*?"**
  The network is being trained on *two* jobs at once inside one combined
  loss number: getting better at choosing actions (the policy part) and
  getting better at predicting future reward (the critic/value part).
  `vf_coef` is a "how much do these two jobs compete for attention" knob:
  a bigger `vf_coef` puts more weight on getting the value predictions
  right; a smaller one puts more relative weight on the action-choosing job.
  **Too low:** the critic (the "how good is this situation" predictor)
  stays inaccurate, which in turn makes the advantage signal (which relies
  on the critic) unreliable. **Too high:** the value-learning job can
  "crowd out" the policy-learning job, starving the part that actually
  decides what to do of its fair share of learning capacity. *Chosen value
  used through most of this notebook: `0.5` (the built-in default), shown
  in the tuning table to sit closer to the "too high" side of its useful
  range (raising it further to `2.0` clearly hurt performance).*

- **`clip_coef` (the PPO clipping range) — "how far am I allowed to trust a
  single update before pumping the brakes?"** PPO's signature safety
  feature. It defines a narrow "trust window" (e.g. `±0.2`, i.e.
  `clip_coef = 0.2`) around "no change at all." If an update would push the
  probability of an action *further* than that window in one go, PPO
  refuses to give it extra credit beyond the window's edge — like a car's
  speed limiter that lets you accelerate, but caps how fast you can go no
  matter how hard you press the pedal, so you can't suddenly lose control.
  **Too tight (small `clip_coef`):** the AI is overly cautious and learns
  more slowly than it safely could. **Too loose (large `clip_coef`):** the
  AI risks taking a single update that's too big, potentially destabilizing
  what it already learned. *Chosen final value: `0.3`* (looser than the
  default `0.2`, found in the tuning table to help slightly rather than
  hurt in this specific game).

- **Probability ratio (`ratio`) — "how much has my opinion of this action
  changed since I collected this data?"** A live measurement, computed
  during each training update, of how much more (or less) likely the
  *current, slightly-updated* policy is to pick a given action compared to
  the *original* policy that actually played the game and collected the
  data. `ratio = 1` means "no change of opinion at all"; this is exactly
  what `clip_coef` keeps a leash on.

- **`approx_kl` (approximate KL divergence) — "the dashboard needle showing
  how much the policy actually moved this update."** A single, always-
  non-negative number summarizing how much the network's behavior shifted
  during the last learning update. This is the *diagnostic instrument*
  (like a car's speedometer) used to tune `lr` and `update_epochs`
  correctly — rather than guessing those two dials by feel, the notebook
  watches this needle and keeps it inside a healthy target band
  (`0.01`–`0.02`): too low means "barely moved, wasted the practice data";
  too high means "moved too fast, the practice data is already stale/no
  longer representative of the current policy."

- **`clip_fraction`** — a companion diagnostic to `approx_kl`: literally
  *what fraction* of all the individual actions in the current batch got
  their update capped by the `clip_coef` safety limiter. A very high
  fraction is another warning sign that updates are too large/aggressive.

- **`total_steps`** — simply how many individual game-turns of practice, in
  total, the AI gets across the *entire* training run, summed over every
  parallel environment and every rollout. This is the overall "training
  budget" — like the total number of practice throws a basketball player
  gets before the season starts, regardless of how those throws are split
  into individual practice sessions.

- **`minibatch_size`** — after collecting one big batch of `N × T` turns,
  the network doesn't necessarily process *all* of them in one single
  mathematical step; it can chop the batch into smaller "minibatches" and
  update on each piece in turn (a common, purely computational/statistical
  trick for how neural networks are actually trained). It must be small
  enough to fit inside the batch you actually collected — one of the
  tuning-table runs (Run 4) failed at first purely because this number
  was set larger than the batch size it was supposed to slice.

- **`align_bonus`** — an extra, optional *shaping* reward (on top of the
  game's normal `1 - cost` reward) that gives the bird a small bonus for
  steering itself toward being lined up with the upcoming gap *early*,
  before the pipe even arrives — like a coach giving small "good
  positioning!" praise during practice, not just cheering at the final
  score. This exists because, on the harder `large` grid, relying on
  entropy (curiosity) alone only escapes the "never flap" trap about half
  the time; adding this gentler, more immediate hint about *good
  positioning* makes learning to actually pass pipes much more reliable.
  Interestingly, the notebook found that a *medium* amount of this bonus
  (`0.3`) works better than a *large* amount (`1.0`) — too much
  gap-hugging reward can start competing with the real goal of actually
  flying through the gap.

- **Standard error (the "± number" next to every measured score)** — a
  measure of how much a *measured average* (like "pipes cleared, averaged
  over 3 training seeds") might wander around just due to random luck,
  rather than reflecting a real, dependable effect. A *small* ± means the
  three seeds landed close together (a reliable, repeatable result); a
  *large* ± means the seeds landed far apart (an unstable result you
  shouldn't trust too much from just 3 tries). This is exactly why the
  tuning-table rules required re-running any promising change on 3 separate
  seeds rather than trusting a single lucky (or unlucky) run.
