# -*- coding: utf-8 -*-

"""
Evaluate a maze checkpoint with the streaming chain solver and the
five-bucket rescoring.

    python solve_cli.py CKPT.pt --split test --n 1000 --budget 1000
    python solve_cli.py CKPT.pt --side 10 --train-pool --n 100

Prints a machine-parsable RESULT line.  ``--train-pool`` rebuilds the
training pool (same seed) and evaluates on it -- the overfit check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import dataset as maze_data
import lattice
import evaluate as maze_eval


def load(path, device=None):
    """ Rebuild a maze :class:`lattice.Net` from a checkpoint. """
    stored = torch.load(path, map_location="cpu", weights_only=False)
    shape = stored["shape"]
    from config import Widths
    net = lattice.build(
        Widths(**shape["widths"]), side=shape["side"],
        lines=shape["lines"], rounds=shape["rounds"],
        cycles=shape["cycles"], theta=shape.get("theta", 0.5),
        detached=shape["detached"], deep=shape["deep"])
    net.load_state_dict(stored["state_dict"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device).eval(), stored


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--side", type=int, default=None,
                        help="synthetic side; default = the HF split")
    parser.add_argument("--train-pool", action="store_true")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--n-chains", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--theta-cls", type=float,
                        default=lattice.THETA_CLS_EVAL)
    parser.add_argument("--temp", type=float, default=lattice.TEMP)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--sync-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-carry", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--oracle-accept", action="store_true")
    parser.add_argument("--calibrate", default=None,
                        help="comma-separated theta_cls values to sweep "
                             "(the run is repeated per value)")
    arguments = parser.parse_args(argv)

    torch.set_float32_matmul_precision("high")
    net, stored = load(arguments.checkpoint)
    device = next(net.parameters()).device
    if not arguments.no_compile and device.type == "cuda":
        net.map.compile_rounds(mode="reduce-overhead")

    side = arguments.side or net.side
    if arguments.train_pool:
        seed = stored["config"].get("seed", 0)
        if side == maze_data.GRID and stored["config"].get(
                "side") == "train":
            x, y = maze_data.load("train")
            x, y = x[:arguments.n], y[:arguments.n]
        else:
            x, y = maze_data.synthetic_pool(
                stored["config"].get("n_puzzles", arguments.n),
                side, seed=seed)
            x, y = x[:arguments.n], y[:arguments.n]
    elif side == maze_data.GRID:
        x, y = maze_data.load(arguments.split)
        x, y = x[:arguments.n], y[:arguments.n]
    else:
        x, y = maze_data.synthetic_pool(
            arguments.n, side, seed=arguments.seed + 999)
    puzzles = torch.from_numpy(x).to(device)
    solutions = torch.from_numpy(y).to(device)

    import lattice_solve
    if arguments.calibrate:
        for value in map(float, arguments.calibrate.split(",")):
            cfg = lattice_solve.SolveConfig(
                max_rounds=arguments.budget, n_chains=arguments.n_chains,
                batch_size=arguments.batch_size, theta_cls=value,
                temp=arguments.temp, sigma=arguments.sigma,
                sync_every=arguments.sync_every, seed=arguments.seed)
            model = net if arguments.no_carry else lattice.Carried(net)
            result = lattice_solve.solve(model, puzzles, solutions, cfg,
                                         verbose=False)
            scored = maze_eval.buckets(result, solutions)
            print(f"CALIB theta_cls {value}: "
                  f"lenient {scored['lenient']:.4f} "
                  f"strict {scored['strict']:.4f}  "
                  + " ".join(f"{k} {scored[k]}"
                             for k in maze_eval.BUCKETS), flush=True)
        return 0
    cfg = lattice_solve.SolveConfig(
        max_rounds=arguments.budget, n_chains=arguments.n_chains,
        batch_size=arguments.batch_size, theta_cls=arguments.theta_cls,
        temp=arguments.temp, sigma=arguments.sigma,
        sync_every=arguments.sync_every, seed=arguments.seed,
        oracle_accept=arguments.oracle_accept)
    model = net if arguments.no_carry else lattice.Carried(net)
    print(f"{arguments.checkpoint.name}: {len(puzzles)} puzzles, "
          f"K={cfg.n_chains}, budget {cfg.max_rounds}, "
          f"theta_cls {cfg.theta_cls}, sigma {cfg.sigma}, "
          f"carry {not arguments.no_carry}", flush=True)
    tick = time.perf_counter()
    result = lattice_solve.solve(model, puzzles, solutions, cfg,
                                 verbose=False)
    seconds = time.perf_counter() - tick
    found = maze_eval.report(result, solutions)
    found.pop("labels")
    print("RESULT " + json.dumps(
        {**{key: found[key] for key in
            ("n", "strict", "lenient", *maze_eval.BUCKETS)},
         "model_calls": found["summary"]["model_calls"],
         "unsound_rate": found["summary"]["unsound_rate"],
         "seconds": round(seconds, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
