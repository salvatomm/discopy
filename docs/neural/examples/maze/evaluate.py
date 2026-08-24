# -*- coding: utf-8 -*-

"""
Maze evaluation: the five buckets over a chain-solver result.

Correctness on a maze is "any valid minimal path", not cell equality, so
the sudoku solver's strict ``correct`` is rescored post hoc from the
recorded accepted boards:

    CORRECT_GT      solved, valid, minimal, cellwise equal to the GT
    CORRECT_ALT     solved, valid, minimal, a different optimal path
    WRONG_VALID     solved, a legal S-to-G path, longer than optimal
    WRONG_INVALID   solved, broken/disconnected/multi-S-or-G
    TIMEOUT         never accepted

The strict metric is CORRECT_GT alone; the lenient one is
CORRECT_GT + CORRECT_ALT.  The reference repo documents a ~7pp gap
between the two -- never report one as the other.

This module shadows ``sudoku/evaluate.py`` when running from the maze
directory, so it re-exports ``perturb_answer`` (which the sudoku
``Lattice.step`` imports lazily for the sigma knob).
"""

from __future__ import annotations

import numpy as np
import torch

import dataset as maze_data
from lattice import SUDOKU, N_CHANNELS
import importlib.util
import sys


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sudoku_evaluate = _load("sudoku_evaluate", SUDOKU / "evaluate.py")
perturb_answer = _sudoku_evaluate.perturb_answer

BUCKETS = ("CORRECT_GT", "CORRECT_ALT", "WRONG_VALID", "WRONG_INVALID",
           "TIMEOUT")


def buckets(result, gt) -> dict:
    """
    The five-bucket classification of a ``SolveResult`` whose ``board``
    holds accepted channel-index boards (-1 rows for timeouts).

    Parameters:
        result : The solver's ``SolveResult``.
        gt : The ground-truth one-hot solutions, ``(P, S, 5)``.

    Returns:
        Bucket counts plus the strict and lenient rates and the
        per-puzzle labels.
    """
    gt_index = gt.argmax(-1).cpu().numpy()
    side = int(round(gt.shape[1] ** 0.5))
    boards = result.board.cpu().numpy()
    labels = []
    for p in range(len(boards)):
        if (boards[p] < 0).any():
            labels.append("TIMEOUT")
            continue
        valid, minimal, exact = maze_data.classify_one(
            boards[p].reshape(side, side),
            gt_index[p].reshape(side, side))
        labels.append(
            "CORRECT_GT" if exact else "CORRECT_ALT" if minimal
            else "WRONG_VALID" if valid else "WRONG_INVALID")
    counts = {name: labels.count(name) for name in BUCKETS}
    n = len(labels)
    return {
        **counts, "n": n, "labels": labels,
        "strict": counts["CORRECT_GT"] / n,
        "lenient": (counts["CORRECT_GT"] + counts["CORRECT_ALT"]) / n}


def report(result, gt, log=print) -> dict:
    """ The buckets beside the solver's own summary, logged. """
    found = buckets(result, gt)
    summary = result.summary()
    log("buckets: " + "  ".join(
        f"{name} {found[name]}" for name in BUCKETS))
    log(f"strict (exact-match) {found['strict']:.4f}   "
        f"lenient (any-valid-optimal) {found['lenient']:.4f}")
    log(f"solver: calls {summary['model_calls']}  "
        f"unsound {summary['unsound_rate']:.4%}  "
        f"conflict tp/fp/fn {summary['conflict_tp']}"
        f"/{summary['conflict_fp']}/{summary['conflict_fn']}  "
        + "  ".join(f"{key} {summary[key]:.0f}" for key in summary
                    if key.startswith("calls_p")))
    return {**found, "summary": summary}
