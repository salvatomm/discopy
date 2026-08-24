# -*- coding: utf-8 -*-

"""
The lattice-deduction solver instantiated on Maze-Hard.

Everything generic is ``examples/sudoku``'s, imported: the ``Lattice``
recursion with its encoder-gradient fix, the three heads, the
projection, the weighted BCE, and (via module-path resolution, see
``lattice_solve`` below) the streaming chain solver.  What is maze's is
in this file and nowhere else:

1. the **wiring**: one fixed 30x30 grid diagram -- pairwise 4-neighbor
   wires (one round per hop) plus one global ``readout`` relation wired
   to every cell (a diameter-2 broadcast channel), and optionally 30
   row + 30 column ``line`` relations (any cell to any cell in two unit
   hops).  Hand-built with :func:`discopy.neural.from_wiring`, reusing
   the sudoku ``cell``/``unit`` signatures unchanged;
2. the **alpha operator** of the reference trainer
   (``_alpha_surviving``): the union of the K sampled solutions the
   lattice has not committed against, with the last-non-empty fallback;
3. the alpha-aware **losses** -- CE masked to alpha-singleton cells,
   per-cell conflict BCE (well-posed here, unlike sudoku);
4. maze **validity**: an all-singleton lattice is a valid board when its
   path cells are one connected S-to-G component of minimal length --
   computable from the lattice alone, so it can be the oracle-accept
   rule of the solver.

This module deliberately exports the full name surface
``sudoku/lattice_solve.py`` reads off ``import lattice``, so the chain
solver runs on maze unmodified: run from this directory, ``import
lattice`` resolves here while ``import lattice_solve`` falls through to
the sudoku file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SUDOKU = HERE.parent / "sudoku"
if str(SUDOKU) not in sys.path:
    sys.path.append(str(SUDOKU))

import model as zoo                          # noqa: E402  (sudoku's)
from config import Widths                    # noqa: E402  (sudoku's)

from discopy.neural import MapNN, Refresh, from_wiring  # noqa: E402
from discopy import frobenius                # noqa: E402

import dataset as maze_data                  # noqa: E402  (maze's)
from dataset import (                        # noqa: E402
    CH_FREE, CH_GOAL, CH_PATH, CH_START, CH_WALL, GRID, N_CHANNELS)


def _load_sudoku_lattice():
    if "sudoku_lattice" in sys.modules:
        return sys.modules["sudoku_lattice"]
    spec = importlib.util.spec_from_file_location(
        "sudoku_lattice", SUDOKU / "lattice.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sudoku_lattice"] = module
    spec.loader.exec_module(module)
    return module


SL = _load_sudoku_lattice()

# --- the generic machinery, re-exported for lattice_solve -------------------

THETA = SL.THETA
THETA_CLS = SL.THETA_CLS
THETA_CLS_EVAL = SL.THETA_CLS_EVAL
TEMP = SL.TEMP
W_POS, W_NEG, W_SM, W_CONF, W_CELL = (
    SL.W_POS, SL.W_NEG, SL.W_SM, SL.W_CONF, SL.W_CELL)
Heads = SL.Heads
Projection = SL.Projection
Step = SL.Step
project = SL.project
board_logit = SL.board_logit
weighted_bce = SL.weighted_bce
conflicts = SL.conflicts
board_of = SL.board_of
digit_perm = SL.digit_perm
permute = SL.permute
unpermute = SL.unpermute
Lattice = SL.Lattice

#: The default elimination threshold of the reference's maze_hard run
#: (0.5, against sudoku's 0.1); Phase 4 sweeps {0.1, 0.5} at 15x15.
THETA_MAZE = 0.5

#: The maze widths: sudoku's ``EXTREME_TRM``-shaped 205k recipe rescaled
#: to land in the 200-300k budget with the extra shared relations.
MAZE_WIDTHS = Widths(dim=24, state_dim=88, hidden=172, y_dim=48)


# --- the wiring -------------------------------------------------------------

def neighbors_of(side: int):
    """ The 4-neighborhood of each cell of a ``side x side`` grid. """
    found = []
    for cell in range(side * side):
        row, col = divmod(cell, side)
        found.append(tuple(
            r * side + c
            for r, c in ((row - 1, col), (row + 1, col),
                         (row, col - 1), (row, col + 1))
            if 0 <= r < side and 0 <= c < side))
    return tuple(found)


def grid_graph(side: int = GRID, lines: bool = False):
    """
    The maze diagram: one ``cell`` box per grid cell whose message orbit
    holds its 4-neighbor wires plus one wire to the global ``readout``
    relation (and, under ``lines``, one to its row and one to its column
    ``line`` relation); the sudoku signatures, resized per degree.

    Pairwise neighbor wires move information one hop per round; the
    units cost two rounds per hop but connect everything.

    Example
    -------
    >>> shape = grid_graph(3)
    >>> len(shape.boxes), shape.n_ports // 2
    (10, 66)
    >>> sorted(set(box.name for box in grid_graph(3, lines=True).boxes))
    ['cell', 'line', 'readout']
    """
    neighborhoods = neighbors_of(side)
    n_cells = side * side
    extra = 3 if lines else 1
    cells = [zoo.cell(len(hood) + extra) for hood in neighborhoods]
    units = [zoo.unit(n_cells)]
    if lines:
        units += [zoo.unit(side) for _ in range(2 * side)]

    wires: list = []
    for cell, hood in enumerate(neighborhoods):
        for position, other in enumerate(hood):
            if cell < other:
                wires.append(
                    ((cell, position),
                     (other, neighborhoods[other].index(cell))))
        degree = len(hood)
        row, col = divmod(cell, side)
        wires.append(((cell, degree), (n_cells, cell)))
        if lines:
            wires.append(((cell, degree + 1), (n_cells + 1 + row, col)))
            wires.append(
                ((cell, degree + 2), (n_cells + 1 + side + col, row)))
        _wire_loops(wires, cell, cells[cell])

    boxes = tuple(sig.box("cell", frobenius) for sig in cells)
    boxes += (units[0].box("readout", frobenius), )
    boxes += tuple(sig.box("line", frobenius) for sig in units[1:])
    return from_wiring(frobenius.CMap, boxes, wires)


def _wire_loops(wires, index, signature):
    wires += [((index, source), (index, target))
              for source, target in signature.loops()]


# --- maze validity ----------------------------------------------------------

def valid_board(x):
    """
    Whether an all-singleton lattice is a valid *minimal* maze solution:
    exactly one S and one G, the traversable cells one connected
    component containing both with no islands, and the path-cell count
    equal to the BFS shortest distance minus one -- all computable from
    the lattice alone, no ground truth.  The oracle-accept rule of the
    chain solver, and the trainer's discard verification.

    Parameters:
        x : The lattice, ``(rows, cells, 5)``.

    Returns:
        ``(rows, )`` bool.
    """
    side = int(round(x.shape[1] ** 0.5))
    boards = x.argmax(-1).cpu().numpy().reshape(-1, side, side)
    singleton = (x.sum(-1) == 1).all(-1).cpu().numpy()
    out = np.zeros(len(boards), dtype=bool)
    for index, board in enumerate(boards):
        if not singleton[index]:
            continue
        ends = maze_data.endpoints(board)
        if ends is None:
            continue
        s, g = ends
        walls = board == CH_WALL
        dist = maze_data.bfs_distances(walls, s)
        if dist[g] <= 0:
            continue
        valid, _, _ = maze_data.classify_one(board, board)
        out[index] = valid and int((board == CH_PATH).sum()) \
            == int(dist[g]) - 1
    return torch.from_numpy(out).to(x.device)


def peer_incomplete(x):
    """
    The sudoku diagnostic has no maze analogue -- no local elimination
    rule makes a complete wrong board impossible -- so the solver's
    running counter is fed zeros and wrong accepts are diagnosed by the
    five evaluation buckets instead.
    """
    rows = torch.zeros(len(x), dtype=torch.long, device=x.device)
    return rows, rows


def lattice_of(puzzles):
    """ Maze lattices are built by ``dataset.encode``, not from digits. """
    raise NotImplementedError("use dataset.load / dataset.encode")


# --- the alpha operator -----------------------------------------------------

def alpha_surviving(x, solutions, last_alpha=None):
    """
    The reference's ``_alpha_surviving``: per row, the union (OR over
    solutions, per cell and channel) of the K solutions the lattice has
    not committed against -- a solution survives when every cell keeps
    at least one channel alive in it.  When the lattice has committed
    against all K, fall back to ``last_alpha``, the previous step's
    alpha (well-defined: an entry only stays in the pool while not in
    detected conflict), or to ``solutions[:, 0]`` -- the canonical
    ground truth -- when there is none.

    Parameters:
        x : The lattice, ``(B, S, C)``.
        solutions : The K sampled solutions, ``(B, K, S, C)``.
        last_alpha : The pre-step alpha, ``(B, S, C)`` or None.

    Example
    -------
    >>> a = torch.tensor([[[1., 0.], [0., 1.]]])   # two solutions ...
    >>> b = torch.tensor([[[1., 0.], [1., 0.]]])
    >>> sols = torch.stack([a, b], 1)
    >>> x = torch.ones(1, 2, 2)                    # ... both alive
    >>> alpha_surviving(x, sols)
    tensor([[[1., 0.],
             [1., 1.]]])
    >>> x = torch.tensor([[[1., 0.], [0., 1.]]])   # committed to a
    >>> alpha_surviving(x, sols)
    tensor([[[1., 0.],
             [0., 1.]]])
    """
    alive = (x > 0.5).unsqueeze(1)
    hot = solutions > 0.5
    consistent = (alive & hot).any(-1)              # (B, K, S)
    surviving = consistent.all(-1)                  # (B, K)
    any_surviving = surviving.any(-1)               # (B, )
    alpha = (hot & surviving.unsqueeze(-1).unsqueeze(-1)).any(1)
    fallback = solutions[:, 0] if last_alpha is None else last_alpha
    return torch.where(any_surviving.view(-1, 1, 1),
                       alpha.to(solutions.dtype), fallback)


# --- the loss ---------------------------------------------------------------

def losses(heads: Heads, x, alpha, given, cell_conf: bool = True):
    """
    The per-step loss on the pre-projection lattice against the alpha
    target (one-hot at K=1): the weighted BCE against ``x * alpha``, the
    softmax CE on cells that are neither given nor alpha-multi-alive
    (with K > 1 many cells legitimately hold both a path and a free
    solution) nor on a conflicting board, the board-level conflict BCE
    on the logsumexp aggregate, and -- on by default here, unlike
    sudoku, because the target is exact on maze -- the per-cell
    conflict BCE against ``not (x & alpha).any(-1)``.
    """
    cell, board = conflicts(x, alpha)
    total = weighted_bce(heads.bce, x * alpha)
    parts = {"bce": total.detach()}
    multi = alpha.sum(-1) > 1.5
    mask = ~given & ~multi & ~board.unsqueeze(-1)
    if mask.any():
        digits = torch.nn.functional.cross_entropy(
            heads.sm[mask], alpha.argmax(-1)[mask])
        total = total + W_SM * digits
        parts["sm"] = digits.detach()
    conf = torch.nn.functional.binary_cross_entropy_with_logits(
        board_logit(heads.conf), board.to(heads.conf.dtype))
    total = total + W_CONF * conf
    parts["conf"] = conf.detach()
    if cell_conf:
        extra = torch.nn.functional.binary_cross_entropy_with_logits(
            heads.conf.squeeze(-1), cell.to(heads.conf.dtype))
        total = total + W_CELL * extra
        parts["cell_conf"] = extra.detach()
    return total, parts


# --- the model --------------------------------------------------------------

class Net(torch.nn.Module):
    """
    The maze lattice model: the sudoku site and refresh on the grid
    diagram, one or three extra shared relations, and the sudoku
    ``Lattice`` solver at ``n = 5``.  Mirrors ``sudoku/lattice.Net``;
    the submodule construction order (site, readout, line, refresh,
    solver, y0) is load-bearing under a seed.

    Parameters:
        widths : The widths, :data:`MAZE_WIDTHS` by default.
        rounds, cycles : The recursion shape per supervision step.
        side : The grid side, 30 for the benchmark.
        lines : Whether to add the row/column relations (design (b)).
        theta : The elimination threshold, the reference's maze 0.5.
        detached, deep, conf_init : Passed to the sudoku ``Lattice``.
    """
    def __init__(self, widths: Widths = None, rounds: int = 2,
                 cycles: int = 10, side: int = GRID, lines: bool = False,
                 theta: float = THETA_MAZE, detached: int = 0,
                 deep: bool = True, conf_init: str = "default"):
        super().__init__()
        widths = widths or MAZE_WIDTHS
        site = zoo._site(widths, widths.y_dim, resumable=True)
        readout = zoo._relation(widths)
        line = zoo._relation(widths) if lines else None
        refresh = Refresh(
            torch.nn.GRUCell(widths.state_dim, widths.y_dim),
            torch.nn.LayerNorm(widths.y_dim),
            source=("cell", zoo.STATE), target=("cell", zoo.ANSWER))
        solver = Lattice(rounds, cycles, refresh=refresh, dim=widths.dim,
                         y_dim=widths.y_dim, n=N_CHANNELS, theta=theta,
                         detached=detached, deep=deep, conf_init=conf_init)
        self.diagram = grid_graph(side, lines)
        ar = {"cell": site, "readout": readout}
        if lines:
            ar["line"] = line
        self.map = MapNN(zoo.factor_ob(widths, widths.y_dim), ar,
                         solver=solver)
        self.y0 = torch.nn.Parameter(torch.zeros(widths.y_dim))
        self.widths, self.side, self.lines = widths, side, lines
        self.n = N_CHANNELS
        self.answer = ("cell", zoo.ANSWER)
        self.n_cells = self.map.sites(self.diagram, ("cell", zoo.CLUE))

    @property
    def interaction(self):
        return self.map.compile(self.diagram)

    @property
    def solver(self) -> Lattice:
        return self.map.solver

    @torch.no_grad()
    def initial(self, x):
        """ The initial flat state of a lattice, as sudoku's. """
        values = {self.solver.clue: self.solver.encoder(x),
                  self.answer: self.y0.expand(
                      len(x), self.n_cells, len(self.y0))}
        return self.map.initial(self.diagram, values)

    def step(self, x, given, state=None, grad: bool = True, generator=None,
             sigma: float = 0.0, theta_cls: float = None,
             temp: float = None) -> Step:
        """
        One supervision step -- from ``initial(x)`` when no state is
        carried (V1), or from the caller's carried state (V2, the
        hybrid latent).  ``sigma`` adds Gaussian noise on the answer
        trace, the evaluation-time diversity knob.
        """
        if state is None:
            state = self.initial(x)
        if sigma:
            import evaluate as evaluations
            state = evaluations.perturb_answer(self, state, sigma,
                                               generator)
        return self.solver.step(self.interaction, state, x, given,
                                grad=grad, generator=generator,
                                theta_cls=theta_cls, temp=temp)


class Carried:
    """
    The V2 evaluation adapter: a stateful view of a :class:`Net` that
    carries the latent across solve steps, so the chain solver -- which
    only ever hands the lattice back -- runs the hybrid model without
    modification.  A reset (or a slot refill) is detected as the
    incoming lattice differing from the one this wrapper last returned:
    the projection is monotone and the solver passes our own output back
    for every untouched chain, so any mismatch means the solver rewrote
    the row, and its carried state is reinitialised -- which is exactly
    the "reset z and y along with x" rule.
    """
    def __init__(self, net: Net):
        self.net = net
        self.state = None
        self.last_x = None

    def step(self, x, given, state=None, grad: bool = False,
             generator=None, sigma: float = 0.0, theta_cls: float = None,
             temp: float = None) -> "Step":
        fresh = self.state is None
        if fresh:
            reset = torch.ones(len(x), dtype=torch.bool, device=x.device)
        else:
            reset = (x != self.last_x).any(-1).any(-1)
        if bool(reset.any()):
            initial = self.net.initial(x[reset])
            if fresh:
                self.state = initial
            else:
                self.state = self.state.clone()
                self.state[reset] = initial
        step = self.net.step(x, given, state=self.state, grad=grad,
                             generator=generator, sigma=sigma,
                             theta_cls=theta_cls, temp=temp)
        self.state = step.state.detach().clone()
        self.last_x = step.x.detach().clone()
        return step


def build(widths: Widths = None, **kwargs) -> Net:
    return Net(widths, **kwargs)


def count_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
