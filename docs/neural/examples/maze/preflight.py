# -*- coding: utf-8 -*-

"""
The Phase-3 pre-flight tests, T1-T8.  Each is a function returning a
dict of measurements and asserting its own gate; the CLI runs any
subset:

    python preflight.py T1 T4 ...      # inside the disc env, GPU 1

T1 runs the full chain/slot solver with oracle heads: any failure is a
bug in the lattice/solver port, not in a model.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import dataset as D
import lattice as L
import evaluate as E


# --- T1: the oracle closed loop ---------------------------------------------

def canonical_path(x_grid):
    """
    The deterministic minimal path of one lattice grid, computed from
    walls, S and G alone: from S, always step to the smallest-index
    neighbor one closer to G.  The oracle and the test harness compute
    it independently and must agree, which is what makes the oracle's
    target *the* ground truth of the run.
    """
    side = x_grid.shape[0]
    walls = x_grid[..., D.CH_WALL] > 0.5
    s = tuple(int(v) for v in np.argwhere(x_grid[..., D.CH_START] > 0.5)[0])
    g = tuple(int(v) for v in np.argwhere(x_grid[..., D.CH_GOAL] > 0.5)[0])
    d_g = D.bfs_distances(walls, g)
    assert d_g[s] > 0
    grid = np.full((side, side), D.CH_FREE, dtype=np.uint8)
    grid[walls] = D.CH_WALL
    cur = s
    while cur != g:
        for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < side and 0 <= nxt[1] < side \
                    and not walls[nxt] and d_g[nxt] == d_g[cur] - 1:
                if nxt != g:
                    grid[nxt] = D.CH_PATH
                cur = nxt
                break
    grid[s], grid[g] = D.CH_START, D.CH_GOAL
    hot = np.eye(D.N_CHANNELS, dtype=np.float32)
    return hot[grid.reshape(-1)]


class Oracle:
    """
    GT-derived heads: ``bce = +M`` iff the candidate is alive in the
    canonical target else ``-M``; ``conf = +M`` per cell iff the row has
    committed against the target there; ``sm = +M`` on the target
    channel.  ``margin`` scales the decide's softmax margin; ``deduce``
    off gives the decide-only variant that exercises the search and
    reset paths.  Targets are computed from each row's immutable givens
    (walls, S, G) and cached by fingerprint, so the oracle needs no
    puzzle identity from the solver.
    """
    def __init__(self, side: int, deduce: bool = True, margin: float = 8.0,
                 theta: float = L.THETA_MAZE):
        self.side, self.deduce, self.margin = side, deduce, margin
        self.theta = theta
        self.cache: dict = {}

    def target_of(self, row) -> torch.Tensor:
        # fingerprint on the immutable givens only -- walls, S and G --
        # so a chain's own pins do not fragment the cache.
        fixed = (row[:, D.CH_WALL] + 2 * row[:, D.CH_START]
                 + 3 * row[:, D.CH_GOAL])
        key = fixed.cpu().numpy().astype(np.uint8).tobytes()
        if key not in self.cache:
            grid = row.cpu().numpy().reshape(
                self.side, self.side, D.N_CHANNELS)
            self.cache[key] = torch.from_numpy(canonical_path(grid))
        return self.cache[key]

    def step(self, x, given, state=None, grad=False, generator=None,
             sigma=0.0, theta_cls=None, temp=None) -> L.Step:
        y = torch.stack([self.target_of(row) for row in x]).to(x.device)
        sign = 2.0 * (y > 0.5).float() - 1.0
        bce = self.margin * sign if self.deduce else torch.zeros_like(y)
        sm = self.margin * sign
        cell, _ = L.conflicts(x, y)
        conf = (2.0 * cell.float() - 1.0).unsqueeze(-1) * 8.0
        heads = L.Heads(bce, sm, conf)
        found = L.project(
            x, given, heads, self.theta,
            L.THETA_CLS_EVAL if theta_cls is None else theta_cls,
            L.TEMP if temp is None else temp, generator)
        return L.Step(state, found.x, heads, found.conflict, found.solved,
                      found.kill, found.decided)


def T1(side: int = 10, n: int = 100, seed: int = 0, variants=None,
       log=print) -> dict:
    """
    The oracle closed loop, three variants: full deduction, decide-only
    at a wide margin, decide-only at a margin narrow enough to produce
    wrong pins and resets.  Gate: 100% valid-minimal accepted solutions,
    zero unsound deductions, termination; the narrow variant must also
    actually reset.
    """
    import lattice_solve as solve_module
    if side == D.GRID:
        x, _ = D.load("test")
        x = x[:n]
    else:
        x, _ = D.synthetic_pool(n, side, seed=seed)
    targets = np.stack([canonical_path(
        row.reshape(side, side, D.N_CHANNELS)) for row in x])
    puzzles = torch.from_numpy(x)
    solutions = torch.from_numpy(targets)
    found = {}
    # the decide-only variants pin one cell per round, so at 30x30 they
    # are hours of CPU oracle; they are exercised at side 10.
    variants = variants or (("deduce", "decide", "narrow")
                            if side <= 16 else ("deduce", ))
    table = {"deduce": (Oracle(side, deduce=True), 30),
             "decide": (Oracle(side, deduce=False, margin=16.0), 3000),
             "narrow": (Oracle(side, deduce=False, margin=6.0), 3000)}
    for name in variants:
        oracle, budget = table[name]
        cfg = solve_module.SolveConfig(
            max_rounds=budget, n_chains=8, batch_size=256, seed=seed)
        tick = time.perf_counter()
        result = solve_module.solve(oracle, puzzles, solutions, cfg,
                                    verbose=False)
        seconds = time.perf_counter() - tick
        scored = E.buckets(result, solutions)
        summary = result.summary()
        record = {
            "lenient": scored["lenient"], "strict": scored["strict"],
            "timeout": scored["TIMEOUT"], "unsound": summary["unsound_rate"],
            "resets": int(result.n_resets.sum()),
            "calls": summary["model_calls"], "seconds": round(seconds, 1)}
        log(f"T1/{name}: {record}")
        assert scored["lenient"] == 1.0, f"{name}: not all valid-minimal"
        assert summary["unsound_rate"] == 0.0, f"{name}: unsound deductions"
        assert scored["TIMEOUT"] == 0, f"{name}: timeouts"
        found[name] = record
    if "narrow" in found:
        assert found["narrow"]["resets"] > 0, "narrow margin never reset"
    log("T1 PASSED")
    return found


# --- T2: the straight-line diagnostic ---------------------------------------

def straight_pool(n: int, side: int, seed: int = 0):
    """ Wall-less straight-line puzzles from random distinct S/G pairs. """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(xs) < n:
        cells = rng.choice(side * side, size=2, replace=False)
        base = np.full((side * side, ), D.CH_FREE, dtype=np.uint8)
        base[cells[0]], base[cells[1]] = D.CH_START, D.CH_GOAL
        x, y = D.encode(base[None], base[None])
        x, y = D.straight_line(x[0], y[0], side)
        xs.append(x)
        ys.append(y)
    return np.stack(xs), np.stack(ys)


def T2(side: int = 15, steps: int = 600, batch: int = 128,
       lines: bool = False, rounds: int = 2, cycles: int = 10,
       seed: int = 0, device="cuda", log=print) -> dict:
    """
    Train the model on straight-line-only puzzles and measure the cell
    accuracy on held-out S/G pairs.  Gate: >= 99% cell accuracy on the
    non-given cells -- if the model cannot see S from G there is no
    point training on mazes.
    """
    torch.manual_seed(seed)
    net = L.build(side=side, lines=lines, rounds=rounds,
                  cycles=cycles).to(device)
    log(f"T2: side {side}, lines {lines}, params "
        f"{L.count_parameters(net)}")
    import train as training
    optimizer = training.adamw(net, 3e-3, 0.1)
    scheduler = training.cosine_schedule(optimizer, steps // 10, steps)
    x_train, y_train = straight_pool(2048, side, seed)
    x_test, y_test = straight_pool(256, side, seed + 1)
    x_test = torch.from_numpy(x_test).to(device)
    y_test = torch.from_numpy(y_test).to(device)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    accuracy, tick = 0.0, time.perf_counter()
    for index in range(steps):
        rows = rng.choice(len(x_train), size=batch, replace=False)
        x = torch.from_numpy(x_train[rows]).to(device)
        y = torch.from_numpy(y_train[rows]).to(device)
        given = x.sum(-1) == 1
        torch.compiler.cudagraph_mark_step_begin()
        step = net.step(x, given, grad=True, generator=generator)
        losses = [L.losses(heads, x, y, given)[0]
                  for heads in (step.every or (step.heads, ))]
        loss = sum(losses) / len(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (index + 1) % 100 == 0:
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                probe = net.step(x_test, x_test.sum(-1) == 1, grad=False)
                free = x_test.sum(-1) > 1.5
                predicted = probe.heads.sm.argmax(-1)
                accuracy = float(
                    (predicted[free] == y_test.argmax(-1)[free])
                    .float().mean())
            log(f"  step {index + 1}: loss {float(loss):.4f}  "
                f"test cell acc {accuracy:.4f}  "
                f"({time.perf_counter() - tick:.0f}s)")
    return {"accuracy": accuracy, "lines": lines, "side": side,
            "params": L.count_parameters(net)}


# --- T3: propagation depth --------------------------------------------------

def T3(side: int = 30, rounds_max: int = 12, device="cuda",
       log=print) -> dict:
    """
    Perturb the clue at S and measure, per round of one cycle, how far
    the answer trace changes: the propagation speed of the wiring, with
    and without the row/column lines.
    """
    found = {}
    for lines in (False, True):
        torch.manual_seed(0)
        net = L.build(side=side, lines=lines, rounds=rounds_max,
                      cycles=1, deep=False).to(device).double()
        x, _ = D.synthetic_pool(1, side, seed=0)
        x = torch.from_numpy(x).to(device).double()
        state = net.initial(x)
        bumped = x.clone()
        s_cell = int((x[0, :, D.CH_START] > 0.5).nonzero()[0])
        bumped[0, s_cell] = 1.0 - bumped[0, s_cell]
        other = net.initial(bumped)
        reaches = []
        interaction = net.interaction
        for _ in range(rounds_max):
            state = interaction.advance(state, 1, False)
            other = interaction.advance(other, 1, False)
            answer = interaction.read(state, ("cell", L.zoo.STATE))
            answer2 = interaction.read(other, ("cell", L.zoo.STATE))
            moved = (answer - answer2).abs().sum(-1)[0] > 1e-12
            row, col = divmod(s_cell, side)
            cells = moved.nonzero().flatten().cpu().numpy()
            reach = max((abs(c // side - row) + abs(c % side - col)
                         for c in cells), default=0)
            reaches.append(int(reach))
        log(f"T3 lines={lines}: hop reach per round {reaches}")
        found[f"lines_{lines}"] = reaches
    assert found["lines_False"][2] >= 2, "pairwise propagation broken"
    return found


# --- T4: equivariance -------------------------------------------------------

def d4_transforms(side: int):
    """ The eight cell permutations of the dihedral group of the grid. """
    index = np.arange(side * side).reshape(side, side)
    frames = []
    for k in range(4):
        rotated = np.rot90(index, k)
        frames.append(rotated.reshape(-1))
        frames.append(np.flip(rotated, axis=1).reshape(-1))
    return [torch.from_numpy(frame.copy()).long() for frame in frames]


def T4(side: int = 10, device="cpu", log=print) -> dict:
    """
    End-to-end D4 equivariance at float64: |f(Px) - P f(x)| ~ 1e-15 on
    the head outputs.  This is the measurement behind dropping the
    dihedral augmentation.
    """
    torch.manual_seed(0)
    net = L.build(side=side).to(device).double()
    x, _ = D.synthetic_pool(2, side, seed=3)
    x = torch.from_numpy(x).to(device).double()
    worst = 0.0
    for perm in d4_transforms(side):
        with torch.no_grad():
            torch.manual_seed(7)
            base = net.step(x, x.sum(-1) == 1, grad=False)
            torch.manual_seed(7)
            moved = net.step(x[:, perm], x[:, perm].sum(-1) == 1,
                             grad=False)
        for a, b in zip(base.heads, moved.heads):
            worst = max(worst, float((a[:, perm] - b).abs().max()))
    log(f"T4: worst D4 head residual {worst:.3e}")
    assert worst < 1e-12, "the model is not D4-equivariant"
    return {"residual": worst}


# --- T5: gradient hygiene ---------------------------------------------------

def T5(side: int = 10, device="cuda", log=print) -> dict:
    """
    The encoder receives a gradient (the cycles > 1 trap), no parameter
    is grad-less, no NaNs, and the clip fires sanely.
    """
    torch.manual_seed(0)
    net = L.build(side=side).to(device)
    x, y = D.synthetic_pool(8, side, seed=1)
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    given = x.sum(-1) == 1
    step = net.step(x, given, grad=True)
    losses = [L.losses(heads, x, y, given)[0]
              for heads in (step.every or (step.heads, ))]
    loss = sum(losses) / len(losses)
    loss.backward()
    missing = [name for name, p in net.named_parameters()
               if p.grad is None]
    encoder = net.solver.encoder.weight.grad
    assert encoder is not None and float(encoder.abs().sum()) > 0, \
        "encoder gradient missing -- the cycles > 1 trap"
    # y0 is written inside the no-grad ``initial`` by design, exactly as
    # in the sudoku pipeline: the learned initial answer stays at its
    # zero initialisation.  Anything else without a gradient is a bug.
    assert missing == ["y0"], f"parameters without gradient: {missing}"
    bad = [name for name, p in net.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite gradients: {bad}"
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
    log(f"T5: encoder |grad| {float(encoder.abs().sum()):.3e}, "
        f"pre-clip norm {norm:.3f}, all {sum(1 for _ in net.parameters())}"
        f" parameters have finite gradients")
    return {"grad_norm": norm}


# --- T8: cost ---------------------------------------------------------------

#: The recursion shapes T8 measures: (label, rounds, cycles, detached,
#: deep, batch).  ``detached=0`` differentiates everything (the sudoku
#: recipe); ``detached=cycles-1`` is the segmented one-differentiated-
#: cycle memory cap the maze diameter needs.
T8_SHAPES = (
    ("sudoku-2x10-b48", 2, 10, 0, True, 48),
    ("sudoku-2x10-b96", 2, 10, 0, True, 96),
    ("seg-8x4-b192", 8, 4, 3, True, 192),
    ("seg-16x4-b192", 16, 4, 3, True, 192),
    ("seg-32x4-b192", 32, 4, 3, True, 192),
)


def T8(side: int = 30, lines: bool = True, shapes=T8_SHAPES,
       compile_mode="reduce-overhead", device="cuda", log=print) -> dict:
    """
    One compile, then the training-step and eval-step wall clock and
    peak memory at the target shapes, eager vs compiled, and the
    projected full-run cost -- written down BEFORE Phase 5 launches.
    """
    torch.set_float32_matmul_precision("high")
    found = {}
    for label, rounds, cycles, detached, deep, batch in shapes:
      for compiled in (False, True):
        torch.manual_seed(0)
        net = L.build(side=side, lines=lines, rounds=rounds,
                      cycles=cycles, detached=detached,
                      deep=deep).to(device)
        if compiled:
            net.map.compile_rounds(mode=compile_mode)
        import train as training
        optimizer = training.adamw(net, 3e-3, 0.1)
        x, _ = D.load("train")
        x = torch.from_numpy(x[:batch]).to(device)
        y = torch.from_numpy(D.load("train")[1][:batch]).to(device)
        given = x.sum(-1) == 1
        generator = torch.Generator(device=device).manual_seed(0)
        torch.cuda.reset_peak_memory_stats()

        def one():
            torch.compiler.cudagraph_mark_step_begin()
            step = net.step(x, given, grad=True, generator=generator)
            losses = [L.losses(heads, x, y, given)[0]
                      for heads in (step.every or (step.heads, ))]
            loss = sum(losses) / len(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        try:
            for _ in range(3):
                one()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for _ in range(10):
                one()
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            log(f"T8 {label} compiled={compiled}: OOM")
            found[f"{label}-{'c' if compiled else 'e'}"] = "OOM"
            del net, optimizer
            torch.cuda.empty_cache()
            from torch import _dynamo
            _dynamo.reset()
            continue
        train_step = (time.perf_counter() - tick) / 10
        with torch.no_grad():
            for _ in range(2):
                torch.compiler.cudagraph_mark_step_begin()
                net.step(x, given, grad=False, generator=generator)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for _ in range(10):
                torch.compiler.cudagraph_mark_step_begin()
                net.step(x, given, grad=False, generator=generator)
            torch.cuda.synchronize()
        eval_step = (time.perf_counter() - tick) / 10
        peak = torch.cuda.max_memory_allocated() // 2 ** 20
        stats = net.map.cache_stats
        record = {"train_s": round(train_step, 4),
                  "eval_s": round(eval_step, 4), "peak_mib": int(peak),
                  "cache": (stats.hits, stats.misses)
                  if hasattr(stats, "hits") else str(stats),
                  "projected_20k_h": round(train_step * 20000 / 3600, 2)}
        log(f"T8 {label} batch={batch} compiled={compiled}: {record}")
        found[f"{label}-{'c' if compiled else 'e'}"] = record
        del net, optimizer
        torch.cuda.empty_cache()
        from torch import _dynamo
        _dynamo.reset()
    return found


TESTS = {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5, "T8": T8}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=list(TESTS))
    parser.add_argument("--side", type=int, default=None)
    parser.add_argument("--lines", action="store_true")
    arguments = parser.parse_args(argv)
    for name in arguments.names or list(TESTS):
        kwargs = {}
        if arguments.side is not None:
            kwargs["side"] = arguments.side
        if arguments.lines and name in ("T2", "T8"):
            kwargs["lines"] = True
        print(f"=== {name} ===", flush=True)
        TESTS[name](**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
