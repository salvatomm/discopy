# DECISIONS — Maze-Hard port of the lattice-deduction solver

Running log: every choice, measurement and rejected alternative, with reasons.
Phases follow the task brief. Reference code: `lattice-deduction-transformers`
(local clone, read-only), consulted throughout.

## Phase 1 — dataset (GATE PASSED 2026-08-24)

- Loader: `dataset.py::download/encode/load`. CSV parsed with `csv.DictReader`,
  **no whitespace stripping** (free cells are literal spaces — the reference
  loader's documented bug). Unknown-char check via a 255-sentinel lookup table,
  as the reference does. Channel order (wall, free, start, goal, path) verbatim.
- All Phase-1 assertions pass on both splits (1000 puzzles each, `verify()`):
  exactly one S and one G; 900 cells; only expected chars; GT path a single
  connected S→G component with no islands; path-cell count == BFS shortest
  distance − 1 (GT minimal); free cells of `x` exactly {free, path}; givens
  of `x` match `y`; `y` one-hot; path only on question-free cells.
- **Measured, train / test**: wall fraction 0.277–0.450 (mean 0.371) /
  0.293–0.448 (0.373); path cells 108–145 (mean 113.1) / 108–152 (112.5);
  free cells per puzzle 493–649 (mean 564) / 495–634 (563).
- **Definition finding**: the "min path ≥ 110" cutoff counts the whole route
  *including S and G*: min path-cell ('o') count is exactly 108 on both
  splits, i.e. 110 route cells. Assertion adjusted to `n_path + 2 >= 110`
  after measuring the distribution (298/1000 train puzzles fall in 108–109
  path cells, so a ≥110-path-cell reading would be plainly wrong, not a
  boundary accident).
- Branching space vs Sudoku: ~564 binary (free-vs-path) cells per puzzle vs
  Sudoku's ~55 blanks × 9 digits. Comparable bit count, but each maze decision
  is binary and the solution set is astronomically degenerate: exact
  shortest-path counts (bigint) on sampled puzzles span **4–10 decimal
  digits** (10^4–10^10 minimal paths per maze). K=64 covers a vanishing
  fraction — expect the α operator to do real work and CORRECT_ALT to
  dominate CORRECT_GT at eval.
- K-sampler: `sample_k_solutions` ported (BFS d_S, d_G; on-DAG mask
  d_S+d_G==D; suffix-count-weighted uniform walk; exact Python-int weights —
  path counts overflow int64 at 30×30, as the reference warns).
  `solutions[0]` is always the A* GT. Verified: every sampled path on
  25 puzzles × K=8 × both splits passes validity+minimality.
- Also ported into `dataset.py`: `maze_classify` (as `classify_one`: valid /
  minimal / exact), the synthetic generator (`generate_maze`, Searchformer
  recipe + HRM `hard_min_path_len` = 12.2% of cells), and the
  `straight_line` diagnostic.

## Phase 2 — architecture decisions

### (1) One diagram or one per datapoint? → ONE fixed 30×30 grid diagram.

Walls are given singletons in the lattice, exactly like LDT's transformer and
the Sudoku pipeline. Rejected per-puzzle wall-excluding graphs because:
compilation is superlinear in boxes (docs/neural/NOTES.md) and nothing would
intern across puzzles (every maze has different walls), so we would compile
per sample; a heterogeneous pool breaks the fixed-shape carried state the
ACTTrainer-style pool relies on; and walls cost only inference FLOPs — the
modules are shared, so parameters are independent of the 900 cells.

### (2) Wiring → prototype (a): pairwise 4-neighbor grid + one global readout
unit; (b) = (a) + 30 row + 30 col units held ready behind a flag, promoted if
T2 fails; (c) more rounds is a hyperparameter, not a wiring change.

Facts read off `discopy.neural` before deciding:
- `from_relation` (pairwise) moves node→node in 1 round; `from_incidence`
  (units) needs 2 rounds/hop — confirmed in CLRS_small's NOTES ("node→node via
  an edge and node→readout→node are both two rounds"). With shortest paths
  ≥110 cells and grid diameter 58, per-hop cost dominates: pairwise wins.
- Variable degree (2 corners / 3 borders / 4 interior) is supported: a `Cell`
  reads its arity off the width it is handed, at the cost of one batched call
  per distinct degree (3 here). Pool "mean" keeps input scale degree-independent.
- A `Cell` has exactly ONE variable-arity orbit, so the readout wire must ride
  the same MESSAGE orbit as the neighbor wires (mean-pooled together, the
  shared-module price). Neither `from_relation` nor `from_incidence` builds
  this hybrid, but `from_wiring` is exported: we hand-build the wiring
  (cell boxes with arity deg+1, one 900-member "readout" Relation), reusing
  the Sudoku `cell()` / `unit()` signatures unchanged.
- Rounds: the Recursion's one-differentiated-cycle trick caps activation
  memory, so deep propagation (≈128 rounds via cycles×rounds) is affordable
  where LDT's full-BPTT transformer could not go. Exact shape chosen in
  Phase 3/T3/T8 by measurement.
- No positional encoding and none needed — the wiring is the geometry. BUT,
  recorded risk: the model is equivariant to every *graph automorphism* (D4
  about the grid centre) and carries no coordinates; position must emerge
  from boundary-degree structure (corners/edges are degree-2/3). T2
  (straight line) is precisely the test of whether that suffices; the
  fallback is (b), whose row/col units add strong positional structure and
  a 2-hop any-cell-to-any-cell channel. One ill-posedness check done up
  front: inputs whose automorphism stabiliser (fixing the S and G labels
  separately) is nontrivial are rare, and where they occur the Bresenham GT
  is itself symmetric, so T2 is not systematically unlearnable for an
  equivariant model.

### (3) What crosses the solve-step boundary? → V2 (carry z and y and x),
V1 kept as an ablation flag.

At diameter ~58, V1 (reinit latent each step) makes every step re-derive the
distance field from scratch; V2 lets it persist so post-decision steps repair
locally — the TRM-style latent LDT lists as future work. The machinery already
supports it: `Lattice.step` accepts a carried state, the Site is resumable,
and the encoder rewrite before the differentiated cycle refreshes the clue
loop with the projected lattice. Trainer changes: the pool carries
`state.detach()` between iterations; refill AND conflict reset restore
`state ← initial(x0)` — a chain reset that kept z would carry the poisoned
belief across the backtrack, defeating it.

### (4) Roles and heads → identical to the Sudoku port with C=5.

Encoder `Linear(5, dim)` written onto the clue loop immediately before the
differentiated cycle (the encoder-gradient fix — with detached prefix cycles
the encoder otherwise never trains); heads `Linear(y_dim, 5)` for bce and
softmax; conflict per-cell `Linear(y_dim, 1)` aggregated with logsumexp
(noisy-OR). **Departure from the Sudoku default**: the per-cell conflict BCE
(`cell_conf`) is ON for maze. It was off for Sudoku because its target is
ill-posed there (of two duplicate cells only one is labelled inconsistent);
on maze the target ¬(x∩α)_c.any() is exact — a wrong path pin conflicts at
precisely the pinned cell — and it is per-cell supervision LDT's single CLS
logit could not have. If it destabilises the shared bias we revisit with the
measurement in hand.

### (5) Augmentation → drop all D4 (dataset-level and per-step); keep only
the S↔G channel swap at dataset level.

The grid GNN with permutation-symmetric orbits is exactly equivariant to D4
(to be verified numerically in T4 at float64, ~1e-15), so both of LDT's
dihedral augs are no-ops here. The one symmetry the model lacks is S↔G
relabelling (mazes are bidirectional): kept, applied on pool insertion.
Also dropped: digit permutation (channels carry distinct semantics — wall is
not exchangeable with path). Chain diversity at inference comes from decide
sampling (temp 1.5) and latent noise on y (`perturb_answer`), not from aug
or dropout (LDT set eval dropout to 0 at 30×30 anyway).

### Reuse map (how maze stays an instantiation, not a fork)

- `sudoku/lattice.py`: `Heads`, `board_logit`, `project`, `weighted_bce`,
  `conflicts`, the `Lattice` recursion — all already generic over
  `(B, cells, C)`; imported with `n=5`.
- `sudoku/model.py`: the `cell()` / `unit()` signatures, `_site`, `_relation`,
  role types, `Widths` — imported; only the wiring function is maze's.
- `sudoku/lattice_solve.py::solve` — the streaming chain solver — reused via
  module-path resolution: maze's own `lattice.py` exports the same names with
  maze semantics (`valid_board` → valid+minimal path check computable from
  the lattice alone; `peer_incomplete` → a maze analogue or a zero stub,
  diagnostic only). Maze scoring adds the five buckets post hoc (`rescore`).
- `sudoku/lattice_train.py`: the pool trainer structure is copied and
  adapted (as it itself was from `train.py`) because the carried object
  changes (V2 state, α recomputed per step, raw starts only, S/G-swap aug);
  the optimizer/schedule/EMA/compile plumbing is imported from `train.py`
  unchanged.

## Phase 3 — pre-flight (running)

- Model instantiation works: `lattice.Net` = sudoku site/relation/refresh/
  `Lattice(n=5)` on the hand-built grid diagram. Params: **202,715** base
  (pairwise+readout), **245,051** with row/col lines — in the 200–300k
  budget; identical at 30×30 (modules shared). 30×30 compile ~1s (one
  compile), eager no-grad step (rounds 2 × cycles 10, B=64) 61 ms.
- **T1 PASSED** (10×10 synthetic, 100 puzzles, K_chains=8): three oracle
  variants — full deduction (4 calls), decide-only margin 16 (241 calls),
  decide-only margin 6 (241 calls, 9 resets exercised). All 100% valid-
  minimal, 0 unsound, 0 timeouts. Oracle targets the canonical
  (lexicographically-first) shortest path recomputed from each row's
  immutable givens, and the same canonical paths are passed to `solve` as
  GT, so the unsound counter is exact. Two bugs found and fixed en route:
  the canonical-path walk marked the wrong cell, and the oracle cache
  fingerprint originally included the chain's own pins (keyed now on
  walls/S/G only). 30×30 T1 run queued.
- `SolveResult` gained a `board` field (additive change to
  `sudoku/lattice_solve.py`): accepted boards as channel indices, -1 for
  timeouts — what maze bucket-rescoring needs and sudoku never recorded.
- **T5 PASSED** after one finding: `y0` (initial answer) receives no
  gradient — inherited from the sudoku design (`initial` is no-grad on
  purpose; only the encoder's path is restored inside `step`). Exempted
  explicitly, everything else has finite gradients; encoder trains.
- **T4 PASSED**: end-to-end D4 head residual 2.1e-15 at float64 →
  dihedral augmentation is a no-op, dropped as decided.
- **T3**: perturbation at S reaches 1 hop at round 2 and the whole grid
  (Manhattan reach 34) at round 3 — the readout relation is a working
  diameter-2 broadcast; fine-grained detail still travels 1 hop/round on
  the pairwise wires. Lines don't change this coarse metric.
- **T2 — the decisive test**: 15×15 straight-line, 1500 steps, batch 128.
  Without lines: stalls at ~98.7% cell accuracy (slow, noisy climb from
  the ~97.3% trivial plateau). With row/col lines: **99.76%** and clean
  learning from step ~800. GATE (≥99%) passes only with lines →
  **design (b) adopted** (pairwise + readout + 30 row + 30 col units,
  245k params), exactly the promotion rule set in Phase 2.
- **T1 at 30×30 PASSED** (full-deduction oracle, 100 test puzzles: 100%
  valid-minimal, 0 unsound, 4 calls). The decide-only variants pin one
  cell per round (~570 rounds at 30×30 on a CPU oracle — hours), so they
  are exercised at side 10 only; `T1(variants=...)` records the rule.
- Trainer built (`lattice_train.py`): pool = 2×batch with an alternating
  batch window, V2 state carry (`--no-carry` = V1 ablation), α recomputed
  per step with last-non-empty fallback, S/G-swap-only insertion aug, raw
  starts, discard verification = valid+minimal (`valid_board`), in-train
  mini-solve rescored to the five buckets with best-checkpoint saving.
  V2 at eval runs through `lattice.Carried`, which detects solver-side
  resets/refills as any externally rewritten lattice row and reinitialises
  those rows' states — so the sudoku chain solver runs V2 unmodified.
- **T7 PASSED (with a finding)**. 10×10, 100 synthetic mazes, batch 64.
  2000 steps: 73% lenient on the train pool. 6000 steps: best checkpoint
  (step 4500) reaches **97% lenient / 71% strict, 0 timeouts** at budget
  300 (and the same 97% at budget 1000, K=64) — the model solves the pool;
  in-train eval on held-out 10×10 at budget 5 peaked at 0.52.
- **T6 finding — conflict-head starvation in the overfit regime**: past
  step ~4700 every pool entry one-shot solves (solved = batch/step,
  sat 1.00, zero conflicts in the pool), so the head trains on 1500 steps
  of pure negatives and drifts silent; held-out eval then collapses
  (WRONG_INVALID 55/100 by step 6000 — wrong complete boards accepted).
  Root cause understood, not masked: the head's positives come from the
  model's own wrong pins, and a memorized pool stops producing them.
  At 30×30/1000 puzzles memorization of this kind is not expected, but
  the mitigation is on the shelf if starvation reappears: corrupted
  starts on insertion (the sudoku ERROR-hint kind — pin one wrong path
  bit), guaranteeing head positives at any skill level. Best-checkpoint
  selection by in-train solve rate (already implemented) is the second
  guard. Earlier in training (P/R 1.00/0.67 at 4700) the head was
  healthy and fast to learn, as the brief predicted for K=1.
- Maze `unsound_rate` (measured vs the single GT y) OVERCOUNTS on maze:
  killing a y-path bit while another optimal path survives is a sound
  deduction the counter cannot see. Read it as an upper bound; the exact
  soundness signal is `tp/fp_conf` vs α in the trainer log.
- **T8 (cost, 30×30 with lines, H100)** — measured before Phase 5:
  | shape (rounds×cycles) | diff rounds | batch | train s/step | peak MiB | 20k proj |
  | 2×10 all-diff deep | 20 | 48 | 0.179 | 20,252 | 1.0 h |
  | 2×10 all-diff deep | 20 | 96 | 0.305 | 40,440 | 1.7 h |
  | 8×4 detached=3 | 8 | 192 | 0.374 | 31,847 | 2.1 h |
  | 16×4 detached=3 | 16 | 192 | 0.734 | 60,961 | 4.1 h |
  | 32×4 detached=3 | 32 | 192 | OOM | — | — |
  The sudoku recipe (20 all-differentiated rounds) does NOT fit at
  batch 192 on 900 cells (OOM at 80 GB) — memory scales ~linearly in
  batch × differentiated rounds. **Finding contrary to the brief's
  expectation**: `compile_rounds("reduce-overhead")` gives NO speedup
  (0.179→0.189 s/step) — at 900 cells the per-round kernels are compute-
  bound, not launch-bound as at 81. Phase 5 runs eager. Shape chosen:
  **2×10 all-diff deep at batch 96–128** — T7-validated, densest
  supervision (10 readouts/step, one per 2-round hop, LDT's per-loop
  pattern), 1.7 h per 20k steps; V2 carry makes cross-step depth
  effectively unbounded, so 20 rounds/step suffices for diameter 58.
- Cold-start observation (smoke run): at init the conflict head's
  logsumexp fires on everything (~sigmoid(bias+log S) ≈ 1) and the BCE
  head kills ~half the bits (σ(0)≈θ=0.5), so the first ~100 steps discard
  heavily via true conflicts (empty cells). Settles once BCE learns
  "keep alive"; not a bug, matches the reference's early dynamics.

## Phase 3/4 — 15×15 campaign

- 6000 steps at 15×15/1000 puzzles is NOT enough: zero solves at any
  eval (both θ), 175/200 WRONG_INVALID at budget 300 — the run ends
  mid-ignition (deduction and conflict head still improving when the
  cosine floors). Extended to the reference's 20k budget; that fixed it.
- The budget-5 in-train mini-solve is uninformative for this model
  class at 15×15 (a 245k model deduces less per step than LDT's 1.8M —
  the search-compensation regime the brief predicted), so
  `--eval-rounds` is now a flag and Phase 5 selects checkpoints at
  budget ~20.
- **θ_elim sweep (20k steps, 200 held-out, budget 300, K_chains=16)**:
  θ=0.5 → 68.0% lenient / 24.0% strict (11 WRONG_INVALID, 42 TIMEOUT);
  θ=0.1 → **77.5% / 27.5%** (5 WI, 17 TO). **θ=0.1 picked once**, the
  same reversion LDT's 15×15 K-sweep made. θ=0.5 kills faster but
  wrongly (6.7% vs 3.7% unsound-vs-y) — speed at budget 5, losses at 300.
- **Hint starts help**: θ=0.5+hints → 78.5% / 29.5% with **zero
  WRONG_INVALID** — corrupted starts keep the conflict head supplied
  with positives and it learns to veto broken boards. Adopted for the
  K sweep and Phase 5 (`--hints`). This is a deliberate departure from
  the reference's raw-only maze pool, justified by the measured
  WRONG_INVALID collapse without it and by being exactly the sudoku
  stream's mechanism from the same paper.
- CORRECT_ALT ≈ 2× CORRECT_GT everywhere, as the 10^4–10^10
  paths-per-maze count predicted: strict alone is the wrong lens on
  maze; both are always reported.
- **K sweep (θ=0.1 + hints, 20k steps, budget 300, 200 held-out)**:
  K=1 → 71.5% lenient / 22.0% strict; **K=8 → 83.5% / 7.5%**;
  K=64 → 74.0% / 6.5%. The reference's Figure-4b direction reproduced:
  gains concentrate at small K (1→8 = +12pp; 8→64 = −9.5pp). Strict
  falls with K as expected — α-supervision stops privileging the
  canonical path. Single-seed noise is visible (K=1+hints < θ0.1 raw),
  so sweep readings are directional, not point estimates.
- **Phase 3 gate: closed.** T1–T8 pass; 15×15 solve rate demonstrably
  improves with training (0 → 77–84% lenient across 20k steps).

## Phase 4 — final recipe (transplanted values + the two sweeps)

θ_elim 0.1 (swept); θ_cls train 0.5, eval calibrated over
{0.5, 0.53, 0.55, 0.6} on held-out train before the final report;
λ_cls 0.1 + per-cell conflict BCE 0.1 (exact targets on maze); decide
temp 1.5; weighted BCE 4.0/0.5; λ_ce 0.2 masked to
{non-given ∧ SAT ∧ α-singleton}; AdamW lr 3e-3, wd 0.1, betas
(0.9, 0.95), cosine + 10% warmup, clip 1.0; pool 2×batch, max_age 100;
hint starts ON; V2 carry; lines wiring; widths (24, 88, 172, 48) =
245,051 params; recursion 2×10 all-differentiated, deep supervision;
batch 128 at 30×30 (memory-bounded — LDT's 192 OOMs the all-diff
shape); EMA off (curves not noisy); inference K_chains 64, budget
1000, reset x AND latent on conflict (`Carried`), latent noise
available via --sigma.

## Phase 5 — 30×30 runs

- Run 1 (K=1) launched: HF train split, 245,051 params (reported at
  startup), 20k steps, eval-every 250 at budget 20 / 16 chains / 100
  test puzzles, --save-every 2500 + best-by-lenient checkpointing.
- **OOM incident, diagnosed before retrying**: at batch 128 the first
  in-train eval (step 250) pushed past 80 GiB (training high-water
  ~55 GiB + the eval's no-grad forwards and `Carried` state clones) and
  the process wedged with the GPU at 96%. Killed; relaunched at the
  T8-verified **batch 96** (steady 42 GiB including eval) with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Phase-4 note
  amended: batch 96, not 128 — a smaller-batch deviation from LDT's 192
  forced by the all-differentiated shape, recorded honestly.
- Session restart killed the batch-96 K=1 run at step 6500 (no resume in
  the pool trainer — the pool state is the training state). Relaunched
  fresh on GPU 0 (now the only GPU; shared at launch with four sudoku
  jobs, so **batch 64**, 28.3 GiB — the batch deviation is now 64, noted;
  eval cadence relaxed to every 500).
- **K=1 at 30×30: FAILS to solve raw puzzles** (final and milestone
  checkpoints; all θ_cls timeouts on 100 train puzzles at budget 300).
  Diagnosis, not a workaround: chain traces show mass deduction resolves
  ~75–80% of cells in one step, then fill PLATEAUS — the corridor region
  (~110–180 cells) advances ~1 decide/round with steady conflict churn,
  so no chain ever completes ~150 sequential decides. Test-time knobs
  measured and insufficient: cycles 10→30 moves the plateau 0.80→0.82;
  eval θ 0.1→0.5 moves it 0.77→0.87 but triples resets, 0 accepts either
  way. Two causes identified: (1) puzzle-step budget 1.28M vs LDT's
  3.84M (batch 64 forced by the shared GPU); (2) **contradictory K=1
  supervision on the corridor** — the BCE asks the model to kill 'path'
  on cells that lie on equally-optimal alternative routes, a
  near-impossible discrimination at 30×30's 10^4–10^10 optimal paths.
  (2) is exactly what α at K>1 removes, so the K effect should be larger
  at 30×30 than the +12pp at 15×15. Plan: K=8 (running, no in-train
  eval per user directive) is the primary result; K=1 reported honestly
  as ~0 at this budget; a longer K=8 (40k steps, batch 96 = LDT's
  puzzle-step parity) queued after it.
- **K=8 at 20k×64 shows the same plateau** (fill ~0.78, 0 accepts in 80
  rounds on train puzzles): at this dose the 30×30 failure is
  K-independent.
- **Perfect-deduction plateau measured** (30 train puzzles): 551 free
  cells, 147 on-DAG, ≥81 mandatory (single-cell BFS levels — a lower
  bound), ~66 genuinely ambiguous → a perfect α-deducer plateaus at
  fill ≥0.926 and needs only the branch decides. Our model plateaus at
  0.78 (θ 0.1) / 0.87 (θ 0.5): it under-deduces the corridor by 2–3×.
  The quantitative target for the LDT-parity run (m30-k8b: 40k steps ×
  batch 96 = 3.84M puzzle-steps, running): close the 0.78 → 0.93 gap.
  If it does not, the honest conclusion is a deduction-capacity limit of
  the 245k GNN at diameter-58, to be reported with these measurements.
- **LDT-parity run (m30-k8b 15k + warm-start m30-k8c 25k, batch 96)
  FAILED — and diagnosed as a learning-rate problem, not a dose
  problem.** Every k8b/k8c milestone plateaus at fill 0.51 (≈ givens +
  10%): these runs never learned raw-board deduction at all, while the
  batch-64 20k runs reach 0.72–0.75 at round 0. Matched-step logs show
  k8b training worse from the start (loss 0.22–0.27 vs 0.11–0.15,
  solved 1–4 vs 40–79 per 20 steps, conflict FP high). What differs is
  the schedule: a 40k cosine keeps lr near the 3e-3 peak ~2× longer, and
  the warm restart re-applied the peak. Consistent with every earlier
  observation that learning only ignites late in the cosine (15×15 eval
  signal appeared at ~16k/20k, lr ≈ 10% of peak; 30×30 pool solves rise
  only after ~10k). The transplanted lr 3e-3 is LDT's transformer value;
  the sudoku search for this very backbone found 9e-4. Two arms launched
  (K=8, batch 64, 20k): **lr 1e-3 and lr 5e-4** — detached, no in-train
  eval, milestones every 5000.
- Session-restart lesson: background jobs are killed with the session;
  long runs now launch under `setsid nohup ... & disown`. The trainer
  gained `--init-from` (warm start, fresh optimizer/schedule/pool).
