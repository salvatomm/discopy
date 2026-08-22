# Part D -- length-free sized-depth arms: D vs F-sized vs T-accum on `bellman_ford`

Every arm of Parts A and B ran at the TRAJECTORY RULE's depth: rounds
read off `batch.lengths`, the ground-truth per-sample step counts --
information the deep-equilibrium literature's models are sold as not
needing.  Part D builds the length-free arms and runs the first
information-fair comparison: **D** (deep output supervision in segments
of 4, detached, accumulated to one optimizer step per batch, at a depth
fixed by a SIZE RULE), **F-sized** (the one-step gradient at the same
sized depth, output loss at the final checkpoint only), and **T-accum**
(Part B's bridge row, trajectory-trained, not retrained).  Everything on
`bellman_ford`, `--widths small`, 100 epochs, seeds 0 1 2, `--regime
fixed`, output-only supervision; comparisons on WIDE as (difference,
pooled sem, ratio); no claim against published numbers.

**Verdict in one line**: both pre-registered gates hold at |ratio| ~ 19
-- deep output supervision needs no step-count information at all (D
WIDE **0.7791 +- 0.0327**, the best arm of the whole study, flat on its
own depth ladder), the information-fair equilibrium baseline fails at
the same depth from the same information (F-sized 0.1463 +- 0.0044),
and the step-count information is worth nothing at test (T-accum
re-scored at the sized depth collapses to 0.0815 +- 0.0172) -- **the arm
scales** (next part: more algorithms, then the full protocol).

## 1. Environment, commit and Phase-0 verification

- Repository: `discopy`, branch `neural_intermediate`, starting from
  Part B's commit `e02d145d`.
- Environment: the locked one (`uv.lock`), Python 3.12.2, torch
  2.13.0+cpu, numpy 2.4.6; **CPU runs** on dgx-h100-3,
  `torch.set_num_threads(1)` throughout.  A GPU parity check was
  measured before training (deviation 6): the H100 buys nothing on this
  launch-bound workload, so the lock's CPU build was kept.
- Phase-0 asserts, all present on the branch before any change:
  `PART_A.md`, `PART_B.md`; `config.Budget.segment_steps`,
  `.segment_optim`, `.segment_detach`; `train.train_epoch_segmented`;
  `evaluate.churn_curve` / `churn_report` (Part B's machinery);
  T-accum's artifacts under `p3-small-max-ptredge-probe-seg4-acc`
  (three seeds' weights and the report JSON, all three seed rows
  readable, WIDE 0.6912 +- 0.0059 as `PART_B.md` records).
- Data: the committed cache under `data/` held all nine `bellman_ford`
  splits; `dataset.py --generate` was not run, no `clrs` venv touched.
- Gates before any change: `pytest test/neural/ -x -q` **251 passed**
  (5 deselected `neural_e2e`); `pytest --doctest-modules config.py
  dataset.py model.py -x -q` **35 passed**.  After all of Part D's
  changes: **263 passed** (the twelve new tests included), doctests
  **35 passed** (now including the two `szd` tag examples).

## 2. Definitions, verbatim, recorded before scoring

**The size rule** (`model.SIZED = 2`, `config.Budget.depth_rule`):

> A *sized* run is `SIZED * n` algorithm steps -- `2n`, i.e.
> `2n * model.HOPS` rounds -- where `n` is the batch's node count, an
> input every model reads, at training and at evaluation alike:
> 2 * 16 = 32 steps (64 rounds) trained at `n = 16`, 2 * 64 = 128 steps
> (256 rounds) evaluated at `n = 64`.  A worst-case bound in the size is
> algorithm knowledge; `batch.lengths` is per-sample leakage, and a
> sized run never reads it -- not for its depth, not for its loss, not
> for its evaluation.  `Budget.depth_rule` is `"trajectory"` (default,
> every earlier tag unchanged) or `"sized"` (tag component `szd`); CLI
> `--depth-rule`.

**D's loss** (the sized segmented loop, `train.segment_loss` /
`train.train_epoch_segmented`):

> The run is the sized depth, cut into segments of 4 algorithm steps,
> each backpropagated in full from the previous segment's **detached**
> final state with the carried input families re-attached
> (`train.grounder`, i.e. `model.Grounded.ground`).  At **every**
> segment end the output probes are decoded and scored against the
> ground-truth outputs for **all** samples (deep supervision, Part A's
> segment loss, verified length-free: it reads `batch.outputs` alone).
> The mean of the segment losses is accumulated to exactly **one**
> optimizer step per batch (`segment_optim="per_batch"`, Part B's
> T-accum stepping).  The hint loss is indexed by the trajectory clock,
> so under the sized rule it is **dropped** (`train.py:170`); the arms
> are output-only (`probe`) regardless, so the interaction never saw a
> hint gradient in any part.

**F-sized's loss** (`model.Model.sized_loss`, routed from
`Model.loss` when `depth_rule == "sized"`, `model.py:2051`):

> One run at the sized depth under `solver="grounded"` -- the
> Jacobian-free one-step gradient with the carried input families
> re-attached, bitwise the library's forward -- and the output loss on
> **all** samples at the **final** checkpoint only.  No hint term, no
> read of `batch.lengths` or `batch.hints`.  The trajectory loss path
> is untouched.

**The poisoned-lengths test** (`test/neural/test_clrs_sized.py`):

> For one real batch, `batch.lengths` is overwritten with garbage (all
> ones, then all `10n`) and the tests assert **bitwise** equality
> against the true lengths of: (i) a D training step's loss, its parts
> and every updated parameter (the full segmented epoch, one optimizer
> step), and an F-sized training step's loss and parts; (ii) the
> evaluation forward pass of both arms -- every decoded output
> prediction, at factor 1.0 and 1.5 of the sized depth; (iii) the tag
> test asserts every Part A and Part B tag byte-identical (O, F, T,
> O-deep, T-accum, F-Anderson) and the two new tags distinct from all
> of them and from each other.  The encoder-gradient tests assert every
> parameter but the hint heads holds a gradient after one D step and
> one F-sized step (the hint heads owe none under the sized rule and
> the tests pin that too), every encoder a non-zero one under D.

**Tags** (all under `artifacts/`, epochs not part of a tag):

| arm | flags | tag |
|-----|-------|-----|
| D | `--arm O --widths small --epochs 100 --regime fixed --segment-steps 4 --segment-optim per_batch --depth-rule sized` | `p3-small-max-ptredge-probe-seg4-acc-szd` |
| F-sized | `--arm F --widths small --epochs 100 --regime fixed --depth-rule sized` | `p3-small-max-ptredge-probe-szd-grounded` |
| T-accum | Part B's, not retrained | `p3-small-max-ptredge-probe-seg4-acc` |

**The sized evaluation ladder**: factors {0.5, 1.0, 1.5} **on the sized
depth** (`config.SIZED_SWEEP`; 64 / 128 / 192 steps at `n = 64`), never
on `batch.steps`.  T-accum's frozen weights are additionally re-scored
at the same absolute depths (`artifacts/partd_taccum_sized.py`,
evaluation only), because its own report evaluates under the trajectory
rule and the D vs T-accum contrast is owed a same-depth read.

## 3. The main table

All numbers `bellman_ford`, 3 seeds, WIDE as mean +- sem over seeds;
"32" the canonical test split; sized ladder = factors on the sized
depth (T-accum's trajectory-rule report has no sized ladder, so its row
carries the re-scoring; its own trajectory ladder is x1 0.6873 /
x1.5 0.6826 / x3 0.3674 for continuity).  Wall-clock per seed:
D 57.4 / 56.6 / 62.2 min (34.0-37.3 s/epoch), F-sized 21.2 / 21.1 /
23.2 min (12.6-13.9 s/epoch); best val D 0.9609 / 0.9805 / 0.9707,
F-sized 0.3750 / 0.5000 / 0.4648.

| arm | depth info | WIDE (mean +- sem) | 32 | sized ladder x0.5 / x1 / x1.5 |
|-----|-----------|---------------------|------|-------------------------------|
| **D** | none (size rule) | **0.7791 +- 0.0327** | 0.7902 | 0.7980 / 0.7902 / 0.7925 |
| F-sized | none (size rule) | 0.1463 +- 0.0044 | 0.1338 | 0.1536 / 0.1338 / 0.1331 |
| T-accum, trajectory eval | train + test | 0.6912 +- 0.0059 | 0.6873 | (trajectory ladder above) |
| T-accum, sized eval | train only | 0.0815 +- 0.0172 | 0.0762 | 0.1125 / 0.0815 / 0.0765 |
| O (reference, Part A) | train + test | 0.5985 +- 0.0096 | 0.5848 | (trajectory: 0.5848 / 0.5160 / 0.2830) |

Per-seed WIDE: D 0.7137 / 0.8108 / 0.8126; F-sized 0.1418 / 0.1418 /
0.1552; T-accum sized eval 0.0835 / 0.0507 / 0.1102.

D is the best arm of the entire study -- above O-deep's 0.6963 +-
0.0258 and T-accum's 0.6912 +- 0.0059, both of which read the
step-count at train **and** test -- and it is the first arm whose depth
ladder is flat: 0.798 / 0.790 / 0.793, where every trajectory arm loses
half its score or more by x3.

## 4. The two pre-registered contrasts (WIDE)

Fixed in Phase 5 of the plan before any result existed.

| contrast | difference | pooled sem | ratio | gate | held? |
|----------|-----------:|-----------:|------:|------|-------|
| (a) D - F-sized | +0.6328 | 0.0330 | **+19.2** | > +1 | **HOLDS** |
| (b) T-accum(sized eval) - D | -0.6976 | 0.0369 | **-18.9** | <= +1 | **HOLDS** |

Both hold, so by the pre-registered rule **the arm scales** (next part:
more algorithms, then the full protocol).  Reading them: (a) is the
project's claim in miniature and information-fair -- at the same
length-free depth, from the same inputs, deep output supervision beats
the one-step equilibrium gradient by nineteen standard errors; (b)
says the step-count information costs nothing at test -- the
trajectory-trained bridge row, put at the depth D actually runs,
collapses, so D gives up nothing for being length-free.  One
unregistered observation the table forces: D also beats T-accum *under
T-accum's own trajectory evaluation* (+0.0879, pooled sem 0.0332,
ratio +2.6), so the sized regime is not merely "as good without the
leak" -- training deeper than the trajectory with deep supervision is
itself worth about one T-accum sem times 2.6.

## 5. Residual and churn at the sized depth (D, F-sized)

Curves at x1.5 of the sized depth (192 steps at `n = 64`), val and
WIDE, in the reports.  **F-sized settles and is wrong**: on WIDE its
answer churn at the sized checkpoint is exactly 0.0000 for seeds 0 and
1 (0.0292 for seed 2) and its latent residual falls to 0.003-0.130 --
a genuine near-fixed-point, scoring 0.146; settled is not right, now
demonstrated on the equilibrium arm itself.  **D's answers stay bounded
at depth**: WIDE churn at the sized checkpoint is 0.0045 / 0.0189 /
0.0453 per seed (last checkpoint 0.0016 / 0.0201 / 0.0327) against
0.16-0.37 for every trajectory arm in Part B -- the first arm whose
out-of-distribution churn drops below 5% at all -- while its latent
residual stays 0.10-0.66, so the Part B dissociation (answers still,
latents moving) now holds out of distribution too, which is exactly
what the flat ladder looks like one level down.

## 6. The fallback

Not triggered: the trigger was D underperforming T-accum (sized
re-scoring) at ratio < -1 on WIDE, and D is *above* it at ratio +18.9.
No ramp arm was trained.

## 7. Deviations and code changes

Code changes (all in the example and its test; `discopy/neural/`
untouched):

- `config.py:140-141` `DEPTH_RULES`; `config.py:143-149` `SIZED_SWEEP`;
  `config.py:435` the `Budget.depth_rule` field, `config.py:385-402`
  its docstring entry, `config.py:506-507` the `szd` tag component,
  `config.py:469-474` the two tag doctests.
- `model.py:110-116` `SIZED`; `model.py:1568-1581` `Model` takes and
  validates `depth_rule` (docstring `model.py:1552-1559`);
  `model.py:1591-1598` the sized branch of `Model.steps_of` (the one
  place a depth is decided from); `model.py:1960-1994`
  `Model.sized_loss`; `model.py:2051-2052` the routing from
  `Model.loss`; `model.py:2188` and `model.py:2330-2334` `build`
  passes it through.
- `train.py:145-151, 170` `segment_loss` skips the trajectory-clock
  hint loop under the sized rule; `train.py:316-317` `depth_policy`
  records `"sized"` (so a reused artefact refuses a depth-rule
  mismatch); `train.py:401` the `build` call; `train.py:500` the
  record key; `train.py:590-597, 621` the CLI flag and its override
  entry.
- `evaluate.py:997-1000` a sized report skips the hint curves (they
  would score untrained heads against the forbidden trajectory clock);
  `evaluate.py:1003-1007` it adds the WIDE residual and churn curves;
  `evaluate.py:1920-1923, 1940` the CLI flag; `evaluate.py:1948-1953`
  a sized budget evaluates at `SIZED_SWEEP` (sweep is not part of the
  tag).
- `test/neural/test_clrs_sized.py` -- twelve pytest items: the tag
  test over all eight arms, the depth test, the poisoned-lengths
  bitwise tests (D train, F-sized train, both arms' evaluation, each
  under two poisons), and the two encoder-gradient tests.

Deviations from the letter of the plan:

1. **Hint loss dropped under the sized rule**, for both arms.  The
   plan specifies the two output losses and the poisoned-lengths gate;
   the hint loss is indexed by `hints[step]` on the trajectory clock,
   so any hint term at all would fail the gate.  Consequence: the
   sized arms' hint decoders are untrained, and their reports skip the
   hint curves (`evaluate.py:997`).
2. **Scheduling**: T-accum's sized re-scoring ran concurrently with D
   seed 0's training on the same host (each process single-threaded,
   separate cores; Part B precedent), and the two arms' reports ran as
   two concurrent processes.  Wall-clocks carry that contention.
3. **A session interruption** killed the training chain after D seed 1
   finished; D seed 2 and F-sized were relaunched detached and trained
   after a gap.  Seeding is per-run (`train.seed_everything`), so the
   gap has no protocol effect; the logs under `artifacts/log-partd-*`
   record it.
4. **A GPU parity check** not in the plan: the identical QUICK sized
   run measured 21.0 s on the shared H100 (torch 2.13.0+cu130) against
   20.9 s on one CPU core -- the workload is launch-bound -- so all six
   runs stayed on the locked environment's CPU build, keeping Part D's
   wall-clocks and arithmetic comparable with Parts A and B.
5. `artifacts/partd_taccum_sized.py`, `artifacts/partd_analysis.py`,
   `artifacts/partd-taccum-sized.json` and the `log-partd-*` files are
   working artifacts under the untracked `artifacts/` directory,
   deliberately not committed, like Parts A's and B's.
6. The T-accum bridge caveat the plan itself flags: T-accum was
   *evaluated* under the trajectory rule in Part B; both its numbers
   are in section 3 and the same-depth contrast (b) uses the sized
   re-scoring.
