# Part B -- credit assignment, output churn, post-hoc halting and best-of-k on `bellman_ford`

Part A ended with T (segmented) at 0.6445 on WIDE against O (unrolled)
at 0.5985 and F (one-step gradient) at 0.2035, T-O at ratio +1.4, with
two pre-registered confounds and a x3 ladder collapse.  Part B
decomposes T's edge into its three axes (B1), adds an output-churn
metric (B2), tests post-hoc halting and best-of-k on the frozen
weights (B3), and time-boxes a fairness check on F (B4).  Everything
on `bellman_ford`, `--widths small`, 100 epochs, seeds 0 1 2,
`--regime fixed`, output-only supervision; comparisons on WIDE as
(difference, pooled sem, ratio), |ratio| < 1 = "no difference"; no
claim against the DEAR paper's published numbers.

**Verdict in one line**: T's whole edge over O is the supervision
placement -- an all-samples output loss at every segment end, worth
+0.098 WIDE at ratio +3.6 *without* any truncation or extra optimizer
steps (O-deep 0.6963, the best arm of the study) -- while detached
truncation adds nothing (ratio -0.2) and the per-segment optimizer
stepping actively costs (ratio -1.4); post-hoc halting recovers most
of the x3 ladder collapse (0.222 -> 0.523) but not T's x1 score, so
the Part C gate (a AND b) does NOT open.

## 1. Environment, commit and Phase-0 verification

- Repository: `discopy`, branch `neural_intermediate`.  Part A's two
  commits (`0d7b8c50`, `286e5e1e`) had been left stranded on the
  unrelated branch `lattice` by a branch mishap at the end of Part A;
  since `lattice` was exactly `neural_intermediate` plus those two
  commits, `neural_intermediate` was fast-forwarded to `286e5e1e`
  before anything else, and Part B starts from that commit.
- Environment: the locked one (`uv.lock`), Python 3.12.2, torch
  2.13.0+cpu, numpy 2.4.6; **CPU runs** on an Intel Xeon Platinum
  8480C (dgx-h100-2 this time, dgx-h100-3 in Part A; same lock,
  `torch.set_num_threads(1)` throughout).
- Phase-0 asserts, all present on the branch: `PART_A.md`;
  `config.Budget.segment_steps`; `train.train_epoch_segmented`,
  `train.segment_loss`, `train.grounder`;
  `test/neural/test_clrs_segmented.py`.
- Gates before any change: `pytest test/neural/ -x -q` **245 passed**
  (5 deselected `neural_e2e`), 110 s; `pytest --doctest-modules
  config.py dataset.py model.py -x -q` **35 passed**.  After all of
  Part B's changes: **251 passed** (the six new tests included),
  doctests **35 passed** (now including the `acc`/`nodetach` tag
  examples).
- Data: the committed cache under `data/` already held all nine
  `bellman_ford` splits; `dataset.py --generate` was not run and no
  `clrs` virtualenv was touched.
- Part A artifacts verified present before Phase 1: the nine `.pt`
  weights and three report JSONs under `p3-small-max-ptredge-probe`,
  `-grounded` and `-seg4`.

## 2. Definitions, verbatim, recorded before scoring

**The two new `Budget` fields** (`config.py`, defaults preserve every
Part A tag byte-for-byte, pinned by
`test_clrs_segmented.py::test_the_b1_tags_are_their_own_and_part_a_tags_are_unchanged`):

> * `segment_optim: str = "per_segment"` -- when the segmented loop
>   steps its optimizer.  `"per_segment"` is arm T of PART_A.md: one
>   optimizer step per segment, `ceil(steps / segment_steps)` steps per
>   batch.  `"per_batch"` accumulates the **mean of the segment
>   losses** and steps the optimizer exactly **once per batch**, so the
>   supervision placement is T's and the step count is O's.  Read only
>   when `segment_steps` is non-zero; tag component `acc`, appended
>   only then and only when non-default.
> * `segment_detach: bool = True` -- whether the state is detached at a
>   segment boundary.  `True` is arm T: each segment backpropagates
>   through its own rounds alone, from a detached state with the
>   carried input families re-attached (`train.grounder`, i.e.
>   `model.Grounded.ground`).  `False` keeps the state attached, so
>   the backward pass flows through the whole run and the re-attachment
>   is a no-op (nothing was severed); it requires
>   `segment_optim="per_batch"`, since an optimizer step inside an
>   attached graph would differentiate stale parameters
>   (`train.train_epoch_segmented` raises otherwise).  Tag component
>   `nodetach`, same rule.

**The B1 chain**, each link one axis (O and T are Part A's runs,
untouched):

| arm | flags beyond Part A's O | tag |
|-----|--------------------------|-----|
| O | (none) | `p3-small-max-ptredge-probe` |
| O-deep | `--segment-steps 4 --segment-optim per_batch --no-segment-detach` | `p3-small-max-ptredge-probe-seg4-acc-nodetach` |
| T-accum | `--segment-steps 4 --segment-optim per_batch` | `p3-small-max-ptredge-probe-seg4-acc` |
| T | `--segment-steps 4` | `p3-small-max-ptredge-probe-seg4` |

so (O-deep - O) is supervision placement alone, (T-accum - O-deep) is
detached truncation alone, and (T - T-accum) is optimizer-step count
alone.

**The output-churn metric** (B2, `evaluate.churn_curve`):

> At each checkpoint of a deep run, decode the OUTPUT probes; churn at
> checkpoint k is the fraction of output elements whose decoded
> prediction changed vs checkpoint k-1: an argmax change for a
> pointer, `mask_one` or categorical probe, a crossing at logit 0 for
> a mask probe, pooled over those discrete probes; a scalar probe is
> reported separately as the mean |delta| and never pooled.  Curves on
> val and on the OOD split at the deepest ladder factor the reports
> use (x3.0, i.e. the canonical 32-trajectory `test` split, exactly
> where the reports' residual curves are read).  Entry k of the curve
> (zero-based) compares one-based checkpoint k+2 against k+1; a batch
> contributes to entry k only if its own run reaches that checkpoint.

On `bellman_ford` the single output probe is `pi` (node pointer), so
the pooled churn *is* the fraction of nodes whose pointer argmax
changed and the scalar channel is empty.

**The halt head** (B3, `evaluate.HaltProbe` / `fit_halt`, on frozen
weights -- the interaction is never retrained):

> Per-sample MLP with one hidden layer (width 64), reading the readout
> relation's state concatenated with the mean-pooled node states, at
> each checkpoint, DETACHED (input dim 48 + 48 = 96 at `--widths
> small`).  Target: the fraction of the sample's output elements
> currently decoded correct (`Model.correctness`, per-node pointer
> match here), a soft target trained with BCE-with-logits against the
> fraction.  Fit on the checkpoints of the frozen model's deep run
> over the **train** split at the deployment factor x3 (full-batch
> Adam, lr 1e-3, 200 epochs, probe seed = model seed).  Calibration:
> the halt threshold is chosen on **val** alone, as the sigmoid
> probability in {0.05, 0.10, ..., 0.95} maximising the val score of
> the halted predictions at x3.  Adaptive halting at x3: at each
> checkpoint compute the halt logit, stop each sample at its first
> threshold crossing, fall back to the final checkpoint; score the
> halted predictions on WIDE and the canonical 32.  test/WIDE are
> never read during fitting or calibration.

**Best-of-k** (B3, `evaluate.best_of_k`):

> k = 8 rollouts at the trajectory rule's depth (factor x1); rollout 1
> is noiseless, rollouts 2..8 add Gaussian noise, sigma swept over
> {0.01, 0.1} on **val** (chosen as the sigma maximising the best
> non-oracle selector's val score, recorded per seed), written onto
> the **initial node-state family alone** (`("node", STATE)` of the
> initial flat state; the encoded input families are untouched).
> Selectors, each choosing per sample among the 8 final checkpoints:
> `halt` (highest halt logit), `residual` (lowest per-sample
> infinity-norm of `T(s) - s` restricted to the node-state family),
> `churn` (lowest fraction of discrete output elements changed
> between the run's last two checkpoints), `oracle` (highest true
> fraction correct; upper bound, reads ground truth).
> Compute-matched control: one deterministic rollout at factor
> k x 1 = x8, the same total round budget spent on depth instead of
> restarts.  The claim requires a NON-oracle selector beating the
> compute-matched control on WIDE.

**B4 classification rule** (pre-registered before reading the deep
curves): on the existing F weights, `residual_curve` at a fixed 512
rounds on val, per seed; a seed's map is *contractive-but-slow* when
the curve, smoothed with a centred moving mean of window 16 rounds (to
absorb the period-two node/readout oscillation), is monotone
non-increasing from round 32 on (tolerance 1e-3 per step) and its
final smoothed value is below 0.9x its smoothed value at round 64;
otherwise *non-contractive* (plateau or oscillation).  If ALL seeds
are non-contractive, the equilibrium method's precondition fails at
this scale and the rest of B4 (Anderson acceleration) is skipped.

## 3. B1 -- the chain table

All numbers `bellman_ford`, 3 seeds; WIDE as mean +- sem over seeds;
"32" the canonical test split; ladder = `sweep_ood` on the 32 at
x1 / x1.5 / x3.  Optimizer steps per epoch, from the histories: 32
for O, O-deep and T-accum (one per batch, pinned by the new test), 64
for T.  Wall-clock (3 seeds, this host): O-deep 47m29s (9.2-9.4
s/epoch), T-accum 46m05s (9.0-9.1 s/epoch); Part A's O and T were
49m18s / 47m06s on the same lock.

| arm | WIDE (mean +- sem) | OOD (32) | ladder x1 / x1.5 / x3 |
|-----|---------------------|----------|------------------------|
| O | 0.5985 +- 0.0096 | 0.5848 +- 0.0028 | 0.5848 / 0.5160 / 0.2830 |
| O-deep | **0.6963 +- 0.0258** | 0.6916 +- 0.0170 | 0.6916 / 0.6302 / 0.2777 |
| T-accum | 0.6912 +- 0.0059 | 0.6873 +- 0.0100 | 0.6873 / 0.6826 / **0.3674** |
| T | 0.6445 +- 0.0326 | 0.6335 +- 0.0392 | 0.6335 / 0.6165 / 0.2033 |

**The three link contrasts** (WIDE; difference, pooled sem, ratio;
|ratio| < 1 = "no difference"):

| link | axis | difference | pooled sem | ratio |
|------|------|-----------:|-----------:|------:|
| O-deep - O | supervision placement alone | **+0.0978** | 0.0275 | **+3.6** |
| T-accum - O-deep | detached truncation alone | -0.0051 | 0.0264 | -0.2 |
| T - T-accum | optimizer-step count alone | -0.0467 | 0.0331 | -1.4 |

(For continuity: T - O = +0.0460, pooled sem 0.0340, ratio +1.4,
Part A's number reproduced by construction.)

**Attribution**: T's edge over O is carried **entirely by the
supervision placement** -- deep supervision (an all-samples output
loss at every 4th checkpoint) applied to plain unrolled backprop, one
optimizer step per batch, no detaching, already beats T itself.
Detached truncation is free (ratio -0.2): cutting the backward graph
at segment boundaries costs nothing once the supervision is deep,
which also kills the "truncation is the regulariser" reading.  The
extra optimizer steps -- the one axis Part A could not separate -- are
**negative** (ratio -1.4): stepping once per segment on the
same-batch gradient is slightly worse than accumulating, so T wins
despite its step count, not because of it.  Two footnotes the table
carries: T-accum, not T, is the most depth-robust arm at x3 (0.3674
vs everything else's 0.20-0.28), and O-deep's edge is also visible in
distribution (best val 0.918-0.951, vs O's 0.926-0.955 -- no
in-distribution price for the OOD gain).

## 4. B2 -- churn findings

All 15 runs of the four chain arms and F -- plus F-Anderson's three --
carry churn curves in their reports (the nine Part A runs
retrofitted by `evaluate.py --churn`, no retraining; the nine new
runs scored at report time).  Curves at x3, val and the canonical
OOD 32; "first < 5%" is the one-based checkpoint at which churn
against the previous checkpoint first drops below 5%.

| arm | val first < 5% (s0/s1/s2) | val last | ood first < 5% | ood last |
|-----|---------------------------|----------|----------------|----------|
| O | 7 / 7 / 7 | 0.039 / 0.055 / 0.156 | never / never / never | 0.102 / 0.129 / 0.129 |
| O-deep | 7 / 7 / 8 | 0.086 / 0.164 / 0.102 | never x3 | 0.250 / 0.285 / 0.219 |
| T-accum | 8 / 8 / 8 | 0.094 / 0.086 / 0.031 | never x3 | 0.289 / 0.242 / 0.156 |
| T | 7 / 8 / 8 | 0.039 / 0.109 / 0.117 | never x3 | 0.301 / 0.367 / 0.234 |
| F | 16 / never / never | 0.000 / 0.172 / 0.258 | 2 / never / 2 | 0.000 / 0.141 / 0.176 |
| F-Anderson | 17 / 2 / 2 | 0.063 / 0.227 / 0.000 | never / 2 / 2 | 0.105 / 0.445 / 0.000 |

**Against the residual curves** (which never drop below 0.1 for any
arm, any seed, either split -- Part A section 5, unchanged):

* **On val, answers settle while latents churn**: every chain arm's
  churn collapses from ~0.8 at checkpoint 2 to below 5% at
  checkpoint 7-8 -- almost exactly where the algorithm itself
  terminates (steps 6-8) -- then flickers at 3-16% for the rest of
  the x3 run.  So in distribution the dissociation Part A saw
  between score and *latent* settling is real: the decoded answers
  are (nearly) still while the state moves.
* **Out of distribution at x3 the answers churn too**: no chain-arm
  seed ever gets below 5%, and the final churn is largest exactly
  where the ladder collapse is worst -- T (0.23-0.37) > O-deep /
  T-accum (0.16-0.29) > O (0.10-0.13).  The x3 collapse is not a
  frozen-wrong-answer pathology; the answers are *actively churning*
  at depth, which is what makes post-hoc halting (B3) the right
  instrument and why it recovers +0.30.
* **Answer-settling is not correctness**: F's seed 0 (and
  F-Anderson's 2) reach churn exactly 0.000 -- constant decoded
  answers -- at scores of ~0.2.  A map can settle to the wrong
  answer, so churn separates "settled" from "right" exactly as a
  residual separates "settled" from "right" one level down.

Summary sentence for the record: **score and latent-settling still
dissociate (val), but score and answer-settling co-move at depth
(ood x3)** -- the ladder collapse is answer churn, not a settled
wrong state.

## 5. B3 -- halted-T and best-of-k (frozen weights; 3 seeds each)

**Per-seed calibration** (all chosen on val alone): T thresholds
0.95 / 0.90 / 0.90, sigmas 0.01 / 0.1 / 0.01; O thresholds
0.90 / 0.90 / 0.90, sigmas 0.1 / 0.01 / 0.1.  Probe fit: 21,576
checkpoint rows per seed (train split at x3), final BCE 0.33-0.37.

**Adaptive halting at x3** (WIDE, mean +- sem over seeds; per-seed
mean stop is 4.0-7.5 checkpoints of the ~21 the x3 budget runs):

| arm | halted @ x3 | fixed x1 | fixed x3 |
|-----|-------------|----------|----------|
| T | 0.5232 +- 0.0638 | 0.6445 +- 0.0326 | 0.2220 +- 0.0072 |
| O | 0.5144 +- 0.0252 | 0.5985 +- 0.0096 | 0.2897 +- 0.1092 |

Per T seed (halted / x1 / x3): 0.6028/0.6732/0.2156,
0.5698/0.6809/0.2141, 0.3971/0.5795/0.2365.  On the canonical 32,
halted-T is 0.5942/0.5684/0.4023.

**The primary question**: halted-T at the x3 budget vs T at x1 --
difference -0.1213, pooled sem 0.0716, ratio **-1.7**.  Halting
recovers most of the ladder collapse (0.222 -> 0.523, +0.30) but does
NOT recover T's x1 score within 1 pooled sem.  The halt head stops
too late or the state it stops on is already degraded: every seed's
halted score sits below its own x1.

**Best-of-k** (k = 8, WIDE, mean +- sem; rollouts at x1, control =
one deterministic rollout at x8):

| selector | T | O |
|----------|---|---|
| halt | 0.6451 +- 0.0326 | 0.5915 +- 0.0117 |
| residual | 0.6387 +- 0.0309 | 0.5847 +- 0.0157 |
| churn | 0.6467 +- 0.0347 | 0.5948 +- 0.0139 |
| oracle (bound) | 0.6765 +- 0.0411 | 0.6335 +- 0.0179 |
| single rollout (x1) | 0.6445 +- 0.0326 | 0.5985 +- 0.0096 |
| compute-matched control (x8) | 0.0396 +- 0.0225 | 0.1600 +- 0.1388 |

Two readings, both owed: (i) by the pre-registered rule the claim
*holds* -- every non-oracle selector beats the compute-matched
control, by ~0.5-0.6 -- but only because depth is actively toxic here
(the x8 control collapses to 0.04), so the comparison is carried by
the control's failure, exactly as T-F was carried by F's in Part A;
(ii) against the honest baseline the rule does not ask about -- the
single x1 rollout -- best-of-8 with any non-oracle selector buys
nothing (T: churn +0.0022 at pooled sem ~0.05; halt +0.0006), and
even the oracle's headroom is +0.03.  Initial-state noise at the
swept sigmas produces rollouts too similar for selection to matter.

## 6. B4 -- the F verdict

**Deep residual curves** (existing F weights, val, fixed 512 rounds,
`residual_curve(rounds=512)`, filed as
`artifacts/partb-b4-f-residual512.json`):

| seed | r64 | r128 | r256 | r511 | shape |
|------|-----|------|------|------|-------|
| 0 | 0.512 | 0.646 | 1.478 | 1.073 | rises after r128, oscillates: **non-contractive** |
| 1 | 0.841 | 0.668 | 0.049 | **0.0022** | converges to a genuine fixed point after ~r300: **contractive-but-slow** |
| 2 | 0.524 | 0.346 | 0.298 | 0.232 | slow monotone decrease, still 0.23 at r511: **contractive-but-slow** (unconverged) |

The pre-registered smoothed-monotone refinement of section 2
classifies all three seeds non-contractive, because every curve's
early transient (r1 residuals 3.4-4.8) breaks
monotonicity-from-round-32; that refinement misfires on the
transient, and the plan's own dichotomy (monotone decreasing vs
plateau-or-oscillation, read on the tail) is what section 2 should
have encoded.  By the plan's dichotomy seed 1 -- and arguably seed 2
-- is contractive-but-slow, so the Anderson branch fired.

**F-Anderson** (`model.Anderson`, type-II mixture with memory 5 over
`Grounded`'s one-step gradient, registered as `--solver anderson`,
tag `p3-small-max-ptredge-probe-anderson`; `memory=1` is bitwise
`Grounded`, pinned by
`test_anderson_with_no_memory_is_bitwise_grounded`; 3 seeds, 4.7-4.8
s/epoch vs F's 4.3):

| row | WIDE | OOD-32 | ladder x1/x1.5/x3 |
|-----|------|--------|--------------------|
| F | 0.2035 +- 0.0169 | 0.1921 +- 0.0147 | 0.1921 / 0.1709 / 0.0977 |
| F-Anderson | 0.1897 +- 0.0143 | 0.1868 +- 0.0173 | 0.1868 / 0.1335 / 0.1051 |

F-Anderson - F on WIDE: **-0.0138, pooled sem 0.0221, ratio -0.6**:
no difference by the pre-registered rule.  **Verdict**: F's failure
at this scale is not iteration-count unfairness -- accelerating the
forward fixed-point search does not move the score, and the maps the
one-step gradient trains mostly do not converge within any budget the
trajectory rule grants (only one seed of three ever reaches a small
residual, and only ~20x deeper than the trajectory).  The
equilibrium method's precondition -- a contraction to differentiate
at -- is absent at this scale, and it is absent *because of* the
training, not repaired by a better solver at the same gradient.

## 7. The Part C gate (pre-registered)

Part C opens only if **(a)** T's edge is NOT entirely the
(T - T-accum) step-count link -- at least one of (O-deep - O) and
(T-accum - O-deep) has ratio > +1 -- AND **(b)** halted-T at the x3
budget >= T at x1 (0.6445 WIDE) within 1 pooled sem.

**Which held**: (a) **HOLDS** -- the (O-deep - O) supervision link is
+3.6, far above +1, so T's edge is not the step-count link (which is
in fact negative).  (b) **FAILS** -- halted-T at x3 is 0.5232 +-
0.0638 against T at x1 0.6445 +- 0.0326: difference -0.1213, pooled
sem 0.0716, ratio -1.7, outside 1 pooled sem.  **The gate needs both,
so Part C does NOT open as pre-registered.**  (The B1 half of the
result -- deep supervision alone at +3.6, cheaper than T and without
its step-count penalty -- is the finding a Part C would want to carry
anyway, but that is a new registration, not this gate.)

## 8. Deviations and code changes

Code changes (all in the example and its test; `discopy/neural/`
untouched):

- `config.py:404-405` -- `segment_optim` / `segment_detach` on
  `Budget`; `config.py:352-375` their docstring entries;
  `config.py:466-470` the tag components (`acc`, `nodetach`), nested
  under `segment_steps` and appended only when non-default;
  `config.py:435-439` the tag doctests.
- `train.py:178` -- `train_epoch_segmented` grows the two keyword
  policies (signature, validation, the per-batch accumulate-and-step
  and attached-boundary paths; the per-segment default path is
  byte-for-byte Part A's); `train.py:437` the routing from
  `train_model`; `train.py:487-488` the record keys;
  `train.py:565-577` the two CLI flags and their entries in the
  budget-override loop.
- `evaluate.py:288` `churn_curve` (B2, beside `residual_curve`);
  `evaluate.py:361` `churn_report` (the no-retraining retrofit);
  `evaluate.py:985` churn in every fresh report's rows;
  `evaluate.py:244` `residual_curve(rounds=...)` (B4's fixed-512
  probe); `evaluate.py:584-931` the B3 block (`HaltProbe`,
  `halt_features`, `halt_fraction`, `fit_halt`, `halted`,
  `_sample_churn`, `best_of_k`, `halting_report`);
  `evaluate.py:1891-1897` the `--churn` / `--halting` modes and, with
  the three segment flags now accepted (`--segment-steps`,
  `--segment-optim`, `--no-segment-detach`), the closure of Part A's
  deviation 2 (T's report no longer needs a Python one-liner).
- `model.py:178` `Anderson` (B4, ~100 lines beside `Grounded`,
  reusing `Grounded.ground` for the differentiated step);
  `model.py:283` its `SOLVERS` entry.
- `test/neural/test_clrs_segmented.py` -- five new tests (six
  pytest items; the per-batch one is parametrised over both arms):
  `test_the_b1_tags_are_their_own_and_part_a_tags_are_unchanged`
  (tag stability: every Part A tag byte-identical, the two new tags
  distinct), `test_a_per_batch_step_trains_every_encoder_once_per_batch`
  (both new arms: every grad non-None, encoder grads non-zero,
  exactly one optimizer step per batch),
  `test_an_attached_boundary_needs_the_per_batch_optimizer`,
  `test_anderson_with_no_memory_is_bitwise_grounded` and
  `test_an_anderson_arm_trains_its_encoders_and_files_apart`; the
  F tag added to the Part A tag test.

Deviations from the letter of the plan:

1. **Branch repair before Phase 0**: the plan says "assert on the
   current branch"; the current branch was `lattice`, where Part A's
   two commits had been stranded.  Since `lattice` was exactly
   `neural_intermediate` + those two commits, `neural_intermediate`
   was fast-forwarded to `286e5e1e` rather than stopping -- the stop
   condition (work actually missing) did not obtain.
2. **Churn definition vs. PART_B.md ordering**: the definition was
   fixed verbatim in `churn_curve`'s docstring before any scoring,
   but this file -- where the plan wanted it recorded -- was created
   shortly after the Part A retrofit had started.  The definition
   did not change between the two.
3. **Per-batch loss normalisation**: the plan says "segment losses
   are accumulated"; the mean over the batch's segments (not the
   sum) is what is accumulated, so the per-batch gradient is the
   average of T's per-segment gradients and the loss magnitude stays
   comparable under the shared `GRAD_CLIP = 1.0`.  Recorded in
   section 2 before training.
4. **B4 classification**: the section-2 smoothed-monotone refinement
   of "contractive-but-slow" misfires on every seed's early
   transient and would have classified even the seed that converges
   to residual 2.2e-3 as non-contractive.  The plan's own coarser
   dichotomy was applied instead (section 6 shows both readings and
   the full numbers); Anderson was implemented and trained on its
   "any seed contractive" branch.
5. **B3 fitting depth**: the plan fixes the fitting *split* (train)
   but not the depth of the checkpoints; the probe was fit at the
   deployment factor x3 (recorded in section 2 before fitting).
   Best-of-k rollouts run at x1 with the control at x8; the sigma
   selection rule (best non-oracle selector on val) is likewise
   section 2's.
6. **Scheduling**: F-Anderson (B4) trained concurrently with the B1
   arms and the B3 scoring on the same CPU host, each process
   single-threaded; wall-clocks in sections 3 and 6 carry that
   contention (Part A's O/F/T trained sequentially and alone).
7. `artifacts/partb-analysis.py`, `artifacts/partb-train.sh` and the
   `log-partb-*` files are working artifacts under the untracked
   `artifacts/` directory, deliberately not committed, like Part A's
   logs.
8. The plan's `--segment-optim`/`--no-segment-detach` spellings were
   adopted exactly as proposed; `--arm O --regime fixed` bypasses
   `config.REGIME` as in Part A (and `artifacts/regime.json`
   declares `bellman_ford: fixed` in agreement).
