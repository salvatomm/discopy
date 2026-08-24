# -*- coding: utf-8 -*-

"""
The pool trainer for the maze lattice model.

    python lattice_train.py --steps 20000 --name maze-pilot   # disc env

``sudoku/lattice_train.py``'s structure with the maze deltas and no
others: the pool carries the K sampled solutions and the running alpha
target beside the lattice (recomputed every step, with the
last-non-empty fallback); the carried object is optionally the *state*
too (V2, the default -- ``--no-carry`` for the V1 ablation); starts are
raw puzzles only (the model's own decides supply the conflict
positives, which at K = 1 fire on the first wrong pin); the insertion
augmentation is the S/G channel swap, the one symmetry the model lacks
(D4 is exact, measured at 1e-15); and a solved entry is verified as a
*valid minimal path* rather than by cell equality, since any optimal
route is correct.  The optimizer, schedule, EMA and ``compile_rounds``
are ``sudoku/train.py``'s, verbatim.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from dataclasses import dataclass

import numpy as np
import torch

import dataset as maze_data
import lattice
import evaluate as maze_eval
import train as training                       # sudoku's, via sys.path
from config import ARTIFACTS, GRAD_CLIP, Widths


# --- the refill stream ------------------------------------------------------

class Stream:
    """
    The refill stream over a finite pool of puzzles with pre-sampled K
    solutions: per draw, one puzzle under a fresh coin-flip S/G channel
    swap (applied consistently to the lattice and all K solutions).

    Parameters:
        x : The lattices, ``(n, S, 5)`` float32.
        solutions : The K solutions, ``(n, K, S, 5)``.
        rng : The numpy generator behind every draw.
        device : Where :meth:`take` lands its batches.
        augment : Whether to apply the S/G swap.
    """
    #: The sudoku stream's start kinds and probabilities, available
    #: behind ``hints`` (off by default -- the reference's maze pool
    #: starts raw): a partial correct fill gives the pool deep states
    #: early, a corrupted fill guarantees conflict-head positives at
    #: any skill level.
    PROBS = (0.20, 0.55, 0.25)

    def __init__(self, x, solutions, rng, device=None, augment=True,
                 hints=False):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.solutions = np.ascontiguousarray(solutions, dtype=np.float32)
        self.rng, self.device, self.augment = rng, device, augment
        self.hints = hints
        self.order, self.cursor = rng.permutation(len(x)), 0
        self.count = 0

    def hint(self, x, y):
        """ One ``raw``/``correct``/``error`` start, sudoku's rule on
        the binary free/path alphabet, in place on ``x``. """
        kind = self.rng.choice(3, p=self.PROBS)
        blanks = np.where(x.sum(1) > 1.5)[0]
        if kind == 0 or not len(blanks):
            return x
        low = 0.0 if kind == 1 else 0.1
        n_fill = int(self.rng.uniform(low, 1.0) * len(blanks))
        n_fill = max(n_fill, 1) if kind == 2 else n_fill
        if not n_fill:
            return x
        fill = self.rng.choice(blanks, size=n_fill, replace=False)
        x[fill] = y[fill]
        if kind == 2:
            n_bad = max(1, int(self.rng.uniform(0.01, 0.30) * n_fill))
            for cell in self.rng.choice(fill, size=n_bad, replace=False):
                x[cell, [maze_data.CH_FREE, maze_data.CH_PATH]] = \
                    x[cell, [maze_data.CH_PATH, maze_data.CH_FREE]]
        return x

    def sample(self):
        if self.cursor >= len(self.order):
            self.order = self.rng.permutation(len(self.x))
            self.cursor = 0
        index = self.order[self.cursor]
        self.cursor += 1
        self.count += 1
        x = self.x[index].copy()
        sols = self.solutions[index].copy()
        if self.augment and self.rng.integers(2):
            swap = [maze_data.CH_START, maze_data.CH_GOAL]
            x[:, swap] = x[:, swap[::-1]]
            sols[:, :, swap] = sols[:, :, swap[::-1]]
        if self.hints:
            x = self.hint(x, sols[0])
        return x, sols

    def take(self, n: int):
        """ ``n`` fresh starts, ``(n, S, 5)`` and ``(n, K, S, 5)``. """
        pairs = [self.sample() for _ in range(n)]
        x = torch.from_numpy(np.stack([x for x, _ in pairs]))
        sols = torch.from_numpy(np.stack([s for _, s in pairs]))
        return x.to(self.device), sols.to(self.device)


def k_pool(split_or_side, n_puzzles: int, k: int, seed: int = 0):
    """
    The canonical pool with its K solutions: the HF split when given
    ``"train"``/``"test"``, a synthetic pool at side ``int`` otherwise.
    """
    import random
    if isinstance(split_or_side, str):
        x, y = maze_data.load(split_or_side)
        x, y = x[:n_puzzles], y[:n_puzzles]
        side = maze_data.GRID
    else:
        side = split_or_side
        x, y = maze_data.synthetic_pool(n_puzzles, side, seed=seed)
    rng = random.Random(seed + 13)
    sols = np.stack([
        maze_data.sample_k_solutions(
            x[i].reshape(side, side, 5), y[i].reshape(side, side, 5),
            k, rng)
        for i in range(len(x))])
    return x, sols


# --- the pool ---------------------------------------------------------------

@dataclass
class Config:
    """ The training protocol: the reference's maze defaults. """
    steps: int = 1000
    batch_size: int = 192
    pool_mult: int = 2                  # pool = pool_mult * batch (LDT 30x30)
    lr: float = 3e-3
    weight_decay: float = 0.1
    warmup_frac: float = 0.10
    betas: tuple = (0.9, 0.95)
    max_age: int = 100
    ema_decay: float = 0.0
    grad_clip: float = GRAD_CLIP
    rounds: int = 2
    cycles: int = 10
    detached: int = 0
    deep: bool = True
    theta: float = lattice.THETA_MAZE
    cell_conf: bool = True
    carry: bool = True                  # V2; False = V1 ablation
    lines: bool = True
    k: int = 1
    seed: int = 0
    save_every: int = 0
    log_every: int = 20
    eval_every: int = 200
    eval_puzzles: int = 100
    eval_rounds: int = 5
    eval_chains: int = 16


class Trainer:
    """
    The persistent pool: per entry the current lattice ``x``, the start
    ``x0`` (its singletons are the protected givens), the K solutions,
    the running alpha, an age -- and, under V2, the carried flat state.
    One forward per iteration feeds loss and projection alike.
    """
    def __init__(self, net: lattice.Net, stream: Stream, cfg: Config):
        self.net, self.stream, self.cfg = net, stream, cfg
        pool = cfg.pool_mult * cfg.batch_size
        x0, sols = stream.take(pool)
        self.x, self.x0, self.sols = x0.clone(), x0, sols
        self.alpha = sols[:, 0].clone()
        self.age = torch.zeros(pool, dtype=torch.long, device=x0.device)
        self.state = None
        self.cursor = 0

    def rows(self):
        """ The batch-sized window of the pool this iteration steps. """
        pool = len(self.x)
        batch = self.cfg.batch_size
        start = self.cursor
        self.cursor = (self.cursor + batch) % pool
        index = torch.arange(start, start + batch,
                             device=self.x.device) % pool
        return index

    def iterate(self, optimizer=None, scheduler=None, ema=None,
                generator=None) -> dict:
        cfg = self.cfg
        rows = self.rows()
        x, x0 = self.x[rows], self.x0[rows]
        sols = self.sols[rows]
        given = x0.sum(-1) == 1
        alpha = lattice.alpha_surviving(x, sols, self.alpha[rows])
        self.alpha[rows] = alpha
        sat = ~lattice.conflicts(x, alpha)[1]
        if self.state is None and cfg.carry:
            with torch.no_grad():
                self.state = self.net.initial(self.x)
        torch.compiler.cudagraph_mark_step_begin()
        state_in = self.state[rows].clone() if cfg.carry else None
        step = self.net.step(x, given, state=state_in, grad=True,
                             generator=generator)
        if step.every:
            found = [lattice.losses(heads, x, alpha, given, cfg.cell_conf)
                     for heads in step.every]
            loss = sum(one for one, _ in found) / len(found)
            parts = {key: sum(p[key] for _, p in found if key in p)
                     / len(found) for key in found[-1][1]}
        else:
            loss, parts = lattice.losses(step.heads, x, alpha, given,
                                         cfg.cell_conf)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(self.net)
        stats = self.metrics(step, rows, x, alpha, sat)
        stats.update(self.apply(step, rows, alpha))
        stats["loss"] = float(loss.detach())
        stats.update({key: float(value) for key, value in parts.items()})
        return stats

    @torch.no_grad()
    def metrics(self, step, rows, x, alpha, sat) -> dict:
        """ The reference's log block: SAT fraction, conflict P/R, ... """
        _, target = lattice.conflicts(x, alpha)
        fires = torch.sigmoid(
            lattice.board_logit(step.heads.conf.detach())) \
            > self.net.solver.theta_cls
        tp = int((fires & target).sum())
        fp = int((fires & ~target).sum())
        fn = int((~fires & target).sum())
        alpha_alive = (x > 0.5) & (alpha > 0.5)
        deduced = int(step.kill.sum())
        return {
            "sat": float(sat.float().mean()),
            "P": tp / max(tp + fp, 1), "R": tp / max(tp + fn, 1),
            "n_conf": int(target.sum()),
            "deduced": deduced,
            "decided": int(step.decided.sum()),
            "unsound": int((step.kill & alpha_alive).sum())
            / max(deduced, 1),
            "fill": float((x.sum(-1) == 1).float().mean())}

    @torch.no_grad()
    def apply(self, step, rows, alpha) -> dict:
        """ Age, verify, discard, refill -- and the V2 state carry. """
        cfg = self.cfg
        self.age[rows] += 1
        solved_correct = step.solved.clone()
        if bool(step.solved.any()):
            solved_correct[step.solved] = lattice.valid_board(
                step.x[step.solved])
        conflict_post = lattice.conflicts(step.x, alpha)[1]
        tp_conflict = step.conflict & conflict_post
        aged = self.age[rows] >= cfg.max_age
        discard = solved_correct | tp_conflict | aged
        fresh = step.x.detach().clone()
        if cfg.carry:
            self.state = self.state.clone()
            self.state[rows] = step.state.detach()
        n = int(discard.sum())
        if n:
            x_new, sols_new = self.stream.take(n)
            fresh[discard] = x_new
            picked = rows[discard]
            self.x0 = self.x0.clone()
            self.x0[picked] = x_new
            self.sols = self.sols.clone()
            self.sols[picked] = sols_new
            self.alpha[picked] = sols_new[:, 0]
            self.age[picked] = 0
            if cfg.carry:
                self.state[picked] = self.net.initial(x_new)
        self.x = self.x.clone()
        self.x[rows] = fresh
        return {"discarded": n,
                "solved_correct": int(solved_correct.sum()),
                "tp_conflict": int(tp_conflict.sum()),
                "fp_conflict": int((step.conflict & ~conflict_post).sum()),
                "aged": int(aged.sum())}


def run(net: lattice.Net, stream: Stream, cfg: Config, eval_data=None,
        log=print, save=None) -> list:
    """ The training loop, ``sudoku/lattice_train.run`` with the maze
    mini-solve (rescored to the five buckets) as the milestone eval. """
    device = next(net.parameters()).device
    optimizer = training.adamw(net, cfg.lr, cfg.weight_decay)
    for group in optimizer.param_groups:
        group["betas"] = tuple(cfg.betas)
    scheduler = training.cosine_schedule(
        optimizer, int(cfg.warmup_frac * cfg.steps), cfg.steps)
    ema = training.EMA(net, cfg.ema_decay) if cfg.ema_decay else None
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    trainer = Trainer(net, stream, cfg)
    counters = ("solved_correct", "tp_conflict", "aged", "discarded")
    history, tick = [], time.perf_counter()
    totals = dict.fromkeys(counters, 0)
    best = {"lenient": -1.0, "step": 0}
    for step_index in range(1, cfg.steps + 1):
        net.train()
        stats = trainer.iterate(optimizer, scheduler, ema, generator)
        for key in counters:
            totals[key] += stats[key]
        if step_index % cfg.log_every == 0:
            record = {"step": step_index, **stats, **totals,
                      "seconds": time.perf_counter() - tick}
            totals = dict.fromkeys(counters, 0)
            history.append(record)
            log(f"step {step_index:5d}/{cfg.steps}"
                f"  loss {record['loss']:.4f}"
                f"  sat {record['sat']:.2f}"
                f"  fill {record['fill']:.2f}"
                f"  solved {record['solved_correct']:3d}"
                f"  tp/fp_conf {record['tp_conflict']:3d}"
                f"/{record['fp_conflict']:3d}"
                f"  aged {record['aged']:3d}"
                f"  deduce {record['deduced']:5d}"
                f"  decide {record['decided']:3d}"
                f"  unsound {record['unsound']:.3%}"
                f"  cls P/R {record['P']:.2f}/{record['R']:.2f}"
                f" (n={record['n_conf']})"
                f"  ({record['seconds']:.1f}s)")
            tick = time.perf_counter()
        if eval_data is not None and cfg.eval_every \
                and step_index % cfg.eval_every == 0:
            import lattice_solve
            net.eval()
            with (ema.averaged(net) if ema is not None
                  else contextlib.nullcontext()):
                model = lattice.Carried(net) if cfg.carry else net
                result = lattice_solve.solve(
                    model, *eval_data, lattice_solve.SolveConfig(
                        max_rounds=cfg.eval_rounds,
                        n_chains=cfg.eval_chains,
                        batch_size=cfg.batch_size, seed=cfg.seed),
                    verbose=False)
            scored = maze_eval.buckets(result, eval_data[1])
            log(f"  [eval step {step_index}, budget {cfg.eval_rounds}] "
                f"lenient {scored['lenient']:.3f} "
                f"strict {scored['strict']:.3f}  "
                + " ".join(f"{name} {scored[name]}"
                           for name in maze_eval.BUCKETS))
            if save is not None and scored["lenient"] > best["lenient"]:
                best.update(lenient=scored["lenient"], step=step_index)
                save("best", history)
            tick = time.perf_counter()
        if save is not None and cfg.save_every \
                and step_index % cfg.save_every == 0 \
                and step_index < cfg.steps:
            with (ema.averaged(net) if ema is not None
                  else contextlib.nullcontext()):
                save(step_index, history)
    if ema is not None:
        net.load_state_dict(ema.shadow, strict=False)
    log(f"best in-train lenient {best['lenient']:.3f} "
        f"at step {best['step']}")
    return history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--side", default="train",
                        help="'train' for the HF 30x30 split, an int "
                             "for a synthetic pool at that side")
    parser.add_argument("--n-puzzles", type=int, default=1000)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-rounds", type=int, default=None,
                        help="in-train mini-solve budget; the reference's "
                             "5 assumes a deduction-heavy model, a "
                             "search-heavy one needs more to show signal")
    parser.add_argument("--eval-chains", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-carry", action="store_true")
    parser.add_argument("--no-lines", action="store_true")
    parser.add_argument("--no-cell-conf", action="store_true")
    parser.add_argument("--hints", action="store_true",
                        help="sudoku-style correct/error starts")
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--name", default="maze")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--detached", type=int, default=None)
    parser.add_argument("--no-deep", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--widths", default=None,
                        help="dim,state_dim,hidden,y_dim")
    parser.add_argument("--init-from", default=None,
                        help="checkpoint whose weights warm-start the "
                             "run (fresh optimizer, schedule and pool)")
    arguments = parser.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    device = training.default_device()
    cfg = Config(steps=arguments.steps, batch_size=arguments.batch_size,
                 seed=arguments.seed, ema_decay=arguments.ema_decay,
                 eval_every=arguments.eval_every, k=arguments.k,
                 carry=not arguments.no_carry,
                 lines=not arguments.no_lines,
                 cell_conf=not arguments.no_cell_conf,
                 save_every=arguments.save_every)
    for key in ("rounds", "cycles", "detached", "lr", "theta",
                "eval_rounds", "eval_chains"):
        if getattr(arguments, key) is not None:
            setattr(cfg, key, getattr(arguments, key))
    if arguments.no_deep:
        cfg.deep = False
    training.seed_everything(cfg.seed)
    widths = Widths(*map(int, arguments.widths.split(","))) \
        if arguments.widths else None
    side = arguments.side if arguments.side == "train" \
        else int(arguments.side)
    grid = maze_data.GRID if side == "train" else side
    net = lattice.build(widths, side=grid, lines=cfg.lines,
                        rounds=cfg.rounds, cycles=cfg.cycles,
                        theta=cfg.theta, detached=cfg.detached,
                        deep=cfg.deep).to(device)
    if arguments.init_from:
        stored = torch.load(arguments.init_from, map_location="cpu",
                            weights_only=False)
        net.load_state_dict(stored["state_dict"])
        print(f"warm-started from {arguments.init_from}", flush=True)
    print(f"parameters: {lattice.count_parameters(net):,}  "
          f"device: {device}  steps: {cfg.steps}  "
          f"batch: {cfg.batch_size}  K: {cfg.k}  carry: {cfg.carry}  "
          f"lines: {cfg.lines}  theta: {net.solver.theta}", flush=True)
    if not arguments.no_compile and device.type == "cuda":
        net.map.compile_rounds(mode="reduce-overhead")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    x, sols = k_pool(side, arguments.n_puzzles, cfg.k, cfg.seed)
    stream = Stream(x, sols, np.random.default_rng(cfg.seed),
                    device=device, hints=arguments.hints)
    if side == "train":
        test_x, test_y = maze_data.load("test")
    else:
        test_x, test_y = maze_data.synthetic_pool(
            cfg.eval_puzzles, side, seed=cfg.seed + 999)
    eval_data = (
        torch.from_numpy(test_x[:cfg.eval_puzzles]).to(device),
        torch.from_numpy(test_y[:cfg.eval_puzzles]).to(device))

    def payload(history):
        return {
            "state_dict": net.state_dict(),
            "shape": {"widths": net.widths.asdict(),
                      "side": net.side, "lines": net.lines,
                      "rounds": net.solver.rounds,
                      "cycles": net.solver.cycles,
                      "theta": net.solver.theta,
                      "detached": net.solver.detached,
                      "deep": net.solver.deep},
            "config": vars(arguments), "history": list(history)}

    def milestone(step_index, history):
        path = ARTIFACTS / (f"{arguments.name}-{cfg.steps}s"
                            f"-seed{cfg.seed}-{step_index}.pt")
        torch.save(payload(history), path)
        print(f"saved {path}", flush=True)

    start = time.perf_counter()
    history = run(net, stream, cfg, eval_data=eval_data,
                  log=lambda line: print(line, flush=True),
                  save=milestone)
    seconds = time.perf_counter() - start
    peak = (torch.cuda.max_memory_allocated(device) / 2 ** 20
            if device.type == "cuda" else float("nan"))
    print(f"wall-clock {seconds:.0f}s "
          f"({seconds / cfg.steps:.2f}s/step incl. eval), "
          f"peak memory {peak:.0f} MiB", flush=True)
    path = ARTIFACTS / f"{arguments.name}-{cfg.steps}s-seed{cfg.seed}.pt"
    torch.save({**payload(history), "seconds": seconds,
                "peak_memory_mb": peak}, path)
    print(f"saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
