# Part A -- TRM-vs-DEAR pilot on `bellman_ford` (O / F / T, small widths)

The project's single question: does TRM-style segmented training beat
deep-equilibrium-style training on neural algorithmic execution?  This
pilot puts both sides of that inequality on one algorithm,
`bellman_ford` -- literally a fixed-point iteration, the equilibrium
method's best case -- at `--widths small`, 100 epochs, 3 seeds,
output-only supervision throughout.  No claim is made against the DEAR
paper's published table: different data scale and processor (Part C).

**Verdict in one line**: T beats F by +0.44 micro-F1 on the 128
-trajectory out-of-distribution split, 12 pooled standard errors -- but
so does plain unrolled backprop (O-F ratio +20), so the gap is mostly
"the one-step gradient fails here" rather than "segments win"; T-O is
+0.046 at ratio +1.4.  Go: proceed to Part B as planned.

## 0. Phase-0 confirmations (read before touching anything)

**Tags at `--widths small`, all mutually distinct and distinct from
everything under `artifacts/`** (checked against `ls artifacts/` before
any run; the closest existing tags are the mpnn-width
`p3-max-ptredge-probe` and `p3-max-ptredge-probe-grounded`):

| arm | budget | tag |
|-----|--------|-----|
| O | `replace(config.H2_ARMS["O"], widths="small", epochs=100)` | `p3-small-max-ptredge-probe` |
| F | `replace(config.H2_ARMS["F"], widths="small", epochs=100)` | `p3-small-max-ptredge-probe-grounded` |
| T | O's budget with `segment_steps=4` | `p3-small-max-ptredge-probe-seg4` |

(`epochs` is not part of `Budget.tag`; the seg component is appended by
the new `segment_steps` field only when non-default, so no existing tag
changes -- pinned by `config.Budget.tag`'s doctest and
`test/neural/test_clrs_segmented.py::test_the_segment_tag_is_its_own_and_default_tags_are_unchanged`.
T's tag was verified distinct from O's, with no artifact under it,
immediately before its training command ran.)

**`--regime fixed` bypasses the `config.REGIME` refusal**: in
`train.main`, `one = replace(budget, mixed=arguments.regime == "mixed")
if arguments.regime else regime_of(algorithm, budget) if arguments.arm
else budget` -- with `--regime` given, `regime_of` (which raises on an
undeclared row) is never consulted; `--regime fixed` sets
`mixed=False`, the `Budget` default, so the tag is unchanged.  (As it
happens `artifacts/regime.json` declares `bellman_ford: fixed` anyway,
so the bypass and the declaration agree.)

**What "output-only" means in this harness** (`probe=True`, kept for
all three arms): the hint loss is decoded from a **detached** state
(`Model.loss`, `found.detach() if self.probe`), so it reaches the hint
decoders and never the interaction -- the interaction (processor +
encoders + output decoders) is trained on the output loss alone, while
the hint heads are still fit, as linear probes of the state.  It is not
`hint_weight=0`: with no hint term the hint heads would be untrained
and every hint curve read from them meaningless.  Their numbers carry
the linear-probe caveat, in this file's table too.

## 1. Environment and commit

- Repository: `discopy`, branch `neural_intermediate`, commit `ae5005a6`
  ("discopy.neural clean commit before merge").  The working tree
  additionally carried the *uncommitted* floor-gap-campaign changes to
  the five example files (T-C `dense`, T-D `feedback`, `trm`,
  `selection`, `eval_every`-in-tag); this pilot's own changes sit on
  top of those and are listed in section 7.
- Environment: the locked one, `uv sync --locked --group all`
  (`uv.lock` pins the `pytorch-cpu` index on Linux) -- Python 3.12.2,
  torch 2.13.0+cpu, numpy 2.4.6.  Install wall-clock: ~20 s (warm uv
  cache).  **CPU runs** (Intel Xeon Platinum 8480C, dgx-h100-3,
  `torch.set_num_threads(1)` per the example's own
  `train.single_threaded`); the H100 on the host is invisible to a CPU
  build, and the prior H2 campaign's CUDA numbers predate the lock.
- Phase-1 gates before the T change: `test_clrs_smoke.py` 120 passed in
  27.3 s; `--doctest-modules config.py dataset.py model.py` 35 passed
  in 2.8 s.  After the T change: 122 passed (the two new tests
  included) in 24.4 s; doctests 35 passed (now including the
  `seg4`-tag example) in 2.6 s.
- Data: the committed cache under `data/` already held all nine
  `bellman_ford` splits (`train`, `train8..16`, `val`, `test`, `wide`;
  238 MB for the whole eight-algorithm cache), so no generation was
  needed and the separate `clrs`-bearing virtualenv was not touched.
- Smoke: `train.py --quick --algorithms bellman_ford` 8.7 s;
  `evaluate.py --quick --algorithms bellman_ford` 73.7 s.
- Wall-clock, training (per seed, `seconds_per_epoch x 100`; per arm,
  `time` around the three-seed command): O 961 / 967 / 978 s, arm total
  49 m 18 s; F 428 / 435 / 1005 s, arm total 32 m 12 s (seed 2's
  10.05 s/epoch against its siblings' 4.3 is shared-node contention);
  T 919 / 913 / 940 s, arm total 47 m 06 s.  Evaluation
  (`evaluate.report`, i.e. scores + sweep + residual curves + hint
  curves for 3 seeds): ~19 min per arm, the three arms in parallel.

## 2. Arms, regime, and the T definition

**Regime declaration**: all runs `--regime fixed` (train on the 1000
CLRS-30 trajectories at `n = 16` alone), on the command line, bypassing
`config.REGIME` as confirmed above.

**Arm O** (control): `config.H2_ARMS["O"]` = `PART3` + `probe=True`.
Unrolled backpropagation (`Iterate`) through the whole run at the
trajectory rule's depth; output loss on every checkpoint from the
sample's own termination onwards; one optimizer step per batch.

**Arm F** (the DEAR-style baseline): `config.H2_ARMS["F"]` = `PART3` +
`probe=True, solver="grounded"`.  Same rounds, same supervision, same
parameters (96,044 in every arm); the Jacobian-free one-step gradient
(`model.Grounded`, which re-attaches the carried input families so the
encoders train -- the library's `FixedPoint(backward="last")` cannot;
forward pass bitwise the library's).

**Arm T** (TRM-style segmented recursion), definition recorded verbatim
before training:

> * Total depth matched to F and O: the trajectory rule's rounds for the
>   batch (model.rounds_of / Model.rounds_for), unchanged.
> * The run is cut into segments of 4 algorithm steps (4 * model.HOPS
>   rounds; final segment may be shorter). Per batch, iterate segments in
>   order; each segment starts from the PREVIOUS segment's final state,
>   detached, with the carried input families re-attached exactly as
>   Grounded.ground does (("node", FEAT) and, when present, ("edge",
>   WEIGHT)) -- without this the encoders receive no gradient after the
>   first segment; add an assertion-based test that every encoder
>   parameter has a non-None grad after a T training step on a real batch.
> * Each segment is differentiated in full (ordinary backprop through its
>   own rounds only). At each segment's final state, decode the OUTPUT
>   probes and take the output loss against the ground-truth outputs for
>   ALL samples (TRM deep supervision: every segment end predicts the
>   final answer; this deliberately differs from O's from-termination-on
>   rule -- record it as a designed difference). One optimizer step per
>   segment (so ~steps/4 optimizer steps per batch where O and F take
>   one; record the optimizer-step counts).

Designed difference, restated: O and F supervise the output only from a
sample's own termination onwards; T supervises the output at **every
segment end for all samples**, which is TRM's deep supervision.  T also
keeps the probe regime: the hint loss inside a segment is decoded from
a detached state, so the interaction still sees the output alone.

**Optimizer-step counts** (from the run histories' `opt_steps`): O and
F take 32 optimizer steps per epoch (one per batch; 3,200 over the
run).  T takes **64** per epoch (6,400 over the run): every training
batch's longest trajectory is 6-8 algorithm steps (rounds 12-16), so
`ceil(steps / 4) = 2` segments per batch.

## 3. Results

OOD micro-F1, `bellman_ford`, mean +- sem over seeds 0/1/2.  "32" is
the canonical CLRS-30 test split (`n = 64`), "WIDE" the 128-trajectory
split; the head split and the ladder are on WIDE and 32 respectively,
as `evaluate.head_split` / `sweep_ood` define them.  Head-split hint
components are linear probes (section 0).  The WIDE column's own
trajectory-level 95% half-width is ~0.008-0.013 for every arm, so the
seed sem is the binding uncertainty.

| arm | OOD (32) | WIDE (mean +- sem) | order-free id / ood / drop | order-dep id / ood / drop | ladder x1 / x1.5 / x3 |
|-----|----------|--------------------|----------------------------|---------------------------|------------------------|
| O | 0.5848 +- 0.0028 | 0.5985 +- 0.0096 | 0.858 / 0.878 / -0.021 | 0.823 / 0.499 / +0.323 | 0.5848 / 0.5160 / 0.2830 |
| F | 0.1921 +- 0.0147 | 0.2035 +- 0.0169 | 0.849 / 0.871 / -0.021 | 0.365 / 0.192 / +0.174 | 0.1921 / 0.1709 / 0.0977 |
| T | 0.6335 +- 0.0392 | 0.6445 +- 0.0326 | 0.858 / 0.872 / -0.014 | 0.815 / 0.505 / +0.310 | 0.6335 / 0.6165 / 0.2033 |

Readings.  The order-free class (the `msk` hint probe here) is
saturated and exact out of distribution for **all three arms** -- even
F's processor carries enough for a linear probe of reachability -- so
the whole contrast lives in the order-dependent class (`pi` output,
`pi_h` probe), as Part 3's protocol predicted.  F's order-dependent
heads are bad already **in distribution** (0.365): the one-step
gradient does not learn the task, it does not merely fail to
generalise.  On the ladder, T is the flattest arm to x1.5 (0.633 ->
0.617, where O gives back 7 points) but collapses hardest by x3
(0.203); no arm is depth-robust.

## 4. The two contrasts (WIDE, 3 seeds; rule: |ratio| < 1 is "no difference")

| contrast | difference | pooled sem | ratio |
|----------|-----------:|-----------:|------:|
| **T - F** | **+0.4411** | 0.0367 | **+12.0** |
| O - F | +0.3951 | 0.0194 | +20.3 |

Both differences are enormous by the pre-registered rule.  Context the
rule does not ask for but the question does: T - O is +0.0460 at
pooled sem 0.0340, ratio +1.4 -- segmented training edges out unrolled
backprop, but the bulk of T - F is F's failure, not T's superiority.

## 5. Residual curves (val, ood; `residual_curve` at factor 3.0, 36 rounds)

"First round below 0.1": **never**, for any arm, any seed, either
split -- no learned map here approaches a fixed point by that
criterion.  Final-round and minimum values, per seed:

| arm | val final (s0/s1/s2) | ood final | ood min over rounds |
|-----|----------------------|-----------|---------------------|
| O | 0.363 / 0.657 / 0.398 | 0.284 / 0.421 / 0.464 | 0.251 / 0.375 / 0.389 |
| F | 0.644 / 0.873 / 1.028 | 0.528 / 1.298 / 0.553 | 0.253 / 0.736 / 0.373 |
| T | 0.467 / 0.705 / 0.935 | 0.732 / 0.844 / 1.312 | 0.530 / 0.780 / 0.866 |

So: **T's map does not settle** -- trained toward no fixed point, it
reaches none, and its residuals are in fact the *highest* of the three
arms out of distribution while its scores are the best.  Notably F,
the arm whose gradient assumes an equilibrium, does not settle either
(its training criterion is the one-step gradient at the trajectory's
end, not contractivity).  The algorithm itself settles at median round
10 of 12-16 (ood, `settles`), so a residual read at 3x depth is well
past termination for every trajectory.  Score and settling dissociate
across all three arms, which is worth carrying into Part B: the
segmented winner wins *without* a basin, so ACT/best-of-k (which read
states, not limits) are the right test-time half to lead with if the
go rule had failed -- it did not, but the dissociation stands.

## 6. Go / no-go

T - F ratio = **+12.0 > +1** -> **GO: proceed to Part B as planned.**
(Honest footnote: the margin is carried by F's collapse; the
T-vs-O margin is +1.4 ratio.  Part B's test-time half remains worth
running regardless, given section 5's no-basin finding.)

## 7. Deviations and code changes

Code changes (all in the example and its test; `discopy/neural/`
untouched):

- `config.py:381` -- `segment_steps: int = 0` on `Budget`;
  `config.py:336-355` its docstring entry; `config.py:436-437` the tag
  component (`seg4`), appended only when non-default;
  `config.py:409-410` the tag doctest.
- `train.py:122` `grounder` (the carried-family re-attachment, reusing
  `model.Grounded.ground` with the same carried tuple `model.build`
  computes); `train.py:136` `segment_loss` (deep-supervision loss of
  one segment); `train.py:178` `train_epoch_segmented` (the segmented
  loop, beside `train_epoch`); `train.py:388` the routing in
  `train_model` when `budget.segment_steps`; `train.py:438` the record
  key; `train.py:508` the `--segment-steps` flag and `train.py:538` its
  entry in the budget-override loop.
- `test/neural/test_clrs_segmented.py` (new, 90 lines): the mandated
  encoder-gradient assertion on a real multi-segment `bellman_ford`
  batch (every parameter's grad non-`None` after the last segment's
  optimizer step, encoder grads non-zero), plus the tag-stability test.

Deviations from the letter of the plan:

1. The working tree already carried uncommitted floor-gap-campaign
   changes to all five example files, and the pilot's hunks interleave
   with them (the `segment_steps` field sits beside the uncommitted
   `trm`; `train.py`'s pre-existing code calls `dataset.densify` and
   `build(trm=...)`, which live in the uncommitted `model.py` /
   `dataset.py`).  Committing `train.py` and `config.py` alone would
   therefore have produced a tree that cannot run, and committing only
   my hunks was not textually possible.  Resolution: the pre-existing
   state of the five files was checked in first as its own
   clearly-labelled commit (not this pilot's work), so that the
   mandated commit contains **exactly** the T implementation, its
   test, this file and `.gitignore` -- at the cost of "exactly one new
   commit" becoming two, of which the second is the pilot's.
2. `evaluate.py` has no `--segment-steps` flag and was left untouched
   (it is not in the commit list): arm T's report was produced by
   calling `evaluate.report("bellman_ford", replace(H2_ARMS["O"],
   widths="small", epochs=100, segment_steps=4))` directly -- the same
   function the CLI dispatches to, writing the same
   `artifacts/p3-small-max-ptredge-probe-seg4-bellman_ford-report.json`.
   O and F used the CLI (`evaluate.py --arm O|F --widths small
   --epochs 100 --algorithms bellman_ford`).
3. Arm O's training command was killed and relaunched once, ~12 min in,
   to add `python -u` (block-buffered stdout had made progress
   invisible); no artifact had been written, and the rerun started from
   scratch with the same seeds.
4. The agent session hosting the three evaluation subprocesses was
   restarted while they ran; all three completed and wrote their
   reports regardless (verified via the `time` footers of
   `artifacts/log-parta-eval-{O,F,T}.txt` and the JSON timestamps).
5. Loss-magnitude caveat, by design of the one-step-per-segment rule:
   T's history `loss`/`output`/`hint` are per-segment means (output
   decoded once per segment on all samples; hint averaged over the
   segment's checkpoints), so they are not unit-comparable with O's
   per-batch, per-run-normalised values.  Scores are unaffected.
6. F's seed 2 trained at 10.05 s/epoch against its siblings' ~4.3
   (shared node); wall-clock only, the run is otherwise ordinary.
7. `dataset.py --generate` was never run: the cache was already
   present and complete for `bellman_ford` (section 1), so the
   separate `clrs` virtualenv step did not arise.
