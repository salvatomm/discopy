# -*- coding: utf-8 -*-

"""
The lattice deduction mechanism on model C, pinned before any GPU time.

``lattice.py`` carries the mechanism -- the multi-hot candidate lattice,
the three heads, the DPLL-style projection, the gradient-restoring clue
rewrite -- and this file pins each of its promises on the CPU, in float64
where a promise is numerical: the projection is monotone and protects the
givens; the status booleans mean what they say; every module trains; the
rewrite is a forward no-op, bitwise; the model is exactly D4-equivariant
(so the trainer is right to skip that augmentation); the chain solver's
plumbing works with a symbolic oracle in place of any network; the pool's
discard rule is the paper's, verification included; the loss is the
specified one; and a tiny training run moves.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

NEURAL = Path(__file__).resolve().parents[2] / "docs" / "neural"
SUDOKU = NEURAL / "examples" / "sudoku"
if str(SUDOKU) not in sys.path:
    sys.path.insert(0, str(SUDOKU))

import lattice                                          # noqa: E402
import lattice_solve                                    # noqa: E402
import lattice_train                                    # noqa: E402
import model as zoo                                     # noqa: E402
from config import Widths                               # noqa: E402

torch.set_num_threads(1)

TINY = Widths(dim=8, state_dim=16, hidden=32, y_dim=8)
ROUNDS, CYCLES = 2, 2


def tiny(seed: int = 0, dtype=torch.float64) -> lattice.Net:
    torch.manual_seed(seed)
    return lattice.build(TINY, rounds=ROUNDS, cycles=CYCLES).to(dtype)


# --- puzzle fixtures --------------------------------------------------------

def solved_grid() -> np.ndarray:
    """ A valid solved board, as 81 digits. """
    row, col = np.divmod(np.arange(81), 9)
    return ((3 * (row % 3) + row // 3 + col) % 9 + 1).astype(np.int64)


PEERS = zoo.peers_of(9)


def count_solutions(board: np.ndarray, limit: int = 2) -> int:
    """ Backtracking solution count, stopping at ``limit``. """
    board = board.copy()
    blanks = np.where(board == 0)[0]

    def recurse(index: int) -> int:
        if index == len(blanks):
            return 1
        cell, found = blanks[index], 0
        taken = {board[peer] for peer in PEERS[cell]}
        for digit in range(1, 10):
            if digit not in taken:
                board[cell] = digit
                found += recurse(index + 1)
                board[cell] = 0
                if found >= limit:
                    break
        return found

    return recurse(0)


def naked_singles(board: np.ndarray):
    """
    Iterated naked-single filling -- exactly what the oracle's kills
    implement.  Returns the filled board and whether it stalled.
    """
    board = board.copy()
    while True:
        progress = False
        for cell in np.where(board == 0)[0]:
            taken = {board[peer] for peer in PEERS[cell]} - {0}
            options = [d for d in range(1, 10) if d not in taken]
            if len(options) == 1:
                board[cell], progress = options[0], True
        if not progress:
            return board, bool((board == 0).any())


def random_puzzle(rng, n_blanks: int) -> tuple:
    """ A digit-relabeled solved grid with ``n_blanks`` cells removed. """
    digits = np.concatenate([[0], 1 + rng.permutation(9)])
    solution = digits[solved_grid()]
    puzzle = solution.copy()
    puzzle[rng.choice(81, size=n_blanks, replace=False)] = 0
    return puzzle, solution


def easy_puzzles(rng, count: int, max_blanks: int = 5) -> list:
    """ Unique puzzles that iterated naked singles solve outright. """
    found = []
    while len(found) < count:
        puzzle, solution = random_puzzle(
            rng, int(rng.integers(1, max_blanks + 1)))
        filled, stalled = naked_singles(puzzle)
        if not stalled and (filled == solution).all() \
                and count_solutions(puzzle) == 1:
            found.append((puzzle, solution))
    return found


def stall_puzzle(rng, tries: int = 200_000) -> tuple:
    """
    A puzzle with a unique solution on which naked singles stall, i.e.
    the oracle solver is forced to guess -- and a wrong guess, having a
    unique solution to contradict, must propagate to an empty cell.
    """
    for _ in range(tries):
        puzzle, solution = random_puzzle(rng, int(rng.integers(30, 40)))
        _, stalled = naked_singles(puzzle)
        if stalled and count_solutions(puzzle) == 1:
            return puzzle, solution
    raise AssertionError("no stall puzzle found")


def one_hot(solutions: np.ndarray) -> torch.Tensor:
    return lattice.lattice_of(np.atleast_2d(solutions))


# --- 1. the projection ------------------------------------------------------

def test_projection_is_monotone_and_protects_givens():
    rng = np.random.default_rng(1)
    puzzles = np.stack([random_puzzle(rng, 40)[0] for _ in range(16)])
    x = lattice.lattice_of(puzzles).double()
    given = x.sum(-1) == 1
    torch.manual_seed(1)
    heads = lattice.Heads(torch.randn(16, 81, 9, dtype=torch.float64),
                          torch.randn(16, 81, 9, dtype=torch.float64),
                          torch.randn(16, 81, 1, dtype=torch.float64) - 3)
    generator = torch.Generator().manual_seed(0)
    found = lattice.project(x, given, heads, generator=generator)

    assert bool((found.x <= x).all())
    assert torch.equal(found.x[given], x[given])

    deduced = x.masked_fill(found.kill, 0.0)
    touched = (found.x != deduced).any(-1)
    for row in range(16):
        if found.decided[row]:
            assert int(touched[row].sum()) == 1
            cell = int(touched[row].nonzero()[0, 0])
            assert float(found.x[row, cell].sum()) == 1.0
            digit = int(found.x[row, cell].argmax())
            assert float(deduced[row, cell, digit]) == 1.0
            assert float(deduced[row, cell].sum()) > 1.5
        else:
            assert found.solved[row] or found.conflict[row]
            assert not bool(touched[row].any())


# --- 2. the status ----------------------------------------------------------

def test_status():
    quiet = lattice.Heads(torch.full((1, 81, 9), 10.0),
                          torch.zeros(1, 81, 9),
                          torch.full((1, 81, 1), -10.0))
    rng = np.random.default_rng(2)
    puzzle, solution = random_puzzle(rng, 30)

    empty = one_hot(solution).clone()
    empty[0, 3] = 0.0
    found = lattice.project(empty, empty.sum(-1) == 1, quiet)
    assert bool(found.conflict[0]) and not bool(found.solved[0])

    full = one_hot(solution)
    found = lattice.project(full, full.sum(-1) == 1, quiet)
    assert bool(found.solved[0]) and not bool(found.conflict[0])

    loud = lattice.Heads(quiet.bce, quiet.sm, -quiet.conf)
    found = lattice.project(full, full.sum(-1) == 1, loud)
    assert bool(found.conflict[0]) and not bool(found.solved[0])

    x, y = lattice.lattice_of(puzzle[None]), one_hot(solution)
    cell, board = lattice.conflicts(x, y)
    assert not bool(board[0])
    assert torch.equal(cell.any(-1), board)

    hint = lattice_train.make_sample(
        x[0].numpy().copy(), y[0].numpy(), rng, "error")
    cell, board = lattice.conflicts(
        torch.from_numpy(hint)[None], y)
    assert bool(board[0])
    assert torch.equal(cell.any(-1), board)


# --- 3, 4. the solver step --------------------------------------------------

def test_encoder_and_heads_train():
    net = tiny()
    rng = np.random.default_rng(3)
    pairs = [random_puzzle(rng, 30) for _ in range(4)]
    puzzles = np.stack([puzzle for puzzle, _ in pairs])
    solutions = np.stack([solution for _, solution in pairs])
    x = lattice.lattice_of(puzzles).double()
    y = one_hot(solutions).double()
    given = x.sum(-1) == 1

    step = net.step(x, given, grad=True,
                    generator=torch.Generator().manual_seed(0))
    loss, _ = lattice.losses(step.heads, x, y, given)
    loss.backward()

    solver = net.solver
    for name, module in (("encoder", solver.encoder), ("bce", solver.bce),
                         ("sm", solver.sm), ("conf", solver.conf),
                         ("cell", net.map.ar["cell"]),
                         ("unit", net.map.ar["unit"])):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads), f"{name} has no grad"
        assert any(bool((g != 0).any()) for g in grads), f"{name} grad zero"
    assert step.state.grad_fn is not None
    assert step.state.detach().grad_fn is None


def test_reattach_is_the_same_forward():
    from discopy.neural import Recursion
    net = tiny()
    rng = np.random.default_rng(4)
    puzzles = np.stack([random_puzzle(rng, 30)[0] for _ in range(3)])
    x = lattice.lattice_of(puzzles).double()
    given = x.sum(-1) == 1

    step = net.step(x, given, grad=False,
                    generator=torch.Generator().manual_seed(0))

    state = net.initial(x)
    with torch.no_grad():
        state = Recursion.step(
            net.solver, net.interaction, state, grad=False)
        answer = net.interaction.read(state, net.answer)
        heads = net.solver.heads(answer)
    assert torch.equal(step.state, state)
    for ours, plain in zip(step.heads, heads):
        assert torch.equal(ours, plain)


def test_deep_supervision_reads_every_hop():
    torch.manual_seed(0)
    net = lattice.build(TINY, rounds=1, cycles=3, detached=0,
                        deep=True).to(torch.float64)
    rng = np.random.default_rng(3)
    pairs = [random_puzzle(rng, 30) for _ in range(2)]
    x = lattice.lattice_of(np.stack([p for p, _ in pairs])).double()
    y = one_hot(np.stack([s for _, s in pairs])).double()
    given = x.sum(-1) == 1

    step = net.step(x, given, grad=True,
                    generator=torch.Generator().manual_seed(0))
    assert len(step.every) == 3
    assert all(torch.equal(a, b)
               for a, b in zip(step.every[-1], step.heads))
    loss = sum(lattice.losses(h, x, y, given)[0]
               for h in step.every) / 3
    loss.backward()
    assert net.solver.encoder.weight.grad is not None
    assert bool((net.solver.encoder.weight.grad != 0).any())


# --- 5. the symmetries ------------------------------------------------------

def _cell_perms() -> dict:
    grid = np.arange(81).reshape(9, 9)
    return {"transpose": grid.T.reshape(-1),
            "rot90": np.rot90(grid).reshape(-1),
            "fliplr": np.fliplr(grid).reshape(-1)}


def test_d4_equivariance_and_digit_perm_roundtrip():
    net = tiny()
    rng = np.random.default_rng(5)
    puzzles = np.stack([random_puzzle(rng, 40)[0] for _ in range(2)])
    x = lattice.lattice_of(puzzles).double()
    given = x.sum(-1) == 1
    heads = net.step(x, given, grad=False,
                     generator=torch.Generator().manual_seed(0)).heads

    for name, perm in _cell_perms().items():
        index = torch.as_tensor(perm.copy())
        moved = net.step(x[:, index], given[:, index], grad=False,
                         generator=torch.Generator().manual_seed(0)).heads
        for ours, theirs in ((heads.bce, moved.bce), (heads.sm, moved.sm)):
            error = (ours[:, index] - theirs).abs().max()
            assert float(error) < 1e-12, f"{name}: {float(error)}"

    perm = lattice.digit_perm(2, generator=torch.Generator().manual_seed(1))
    y = one_hot(np.stack([solved_grid(), solved_grid()])).double()
    for tensor in (x, y, heads.bce, heads.sm):
        assert torch.equal(
            lattice.unpermute(lattice.permute(tensor, perm), perm), tensor)


# --- 6. the chain solver, with a symbolic model -----------------------------

class Oracle:
    """
    Naked-single elimination as head logits: kill exactly the digits of
    singleton peers, uniform digit logits, a conflict head that never
    fires -- so the search plumbing (reset, accept, refill, budgets) is
    tested independently of any network.
    """
    def __init__(self):
        matrix = torch.zeros(81, 81)
        for cell, others in enumerate(PEERS):
            matrix[cell, list(others)] = 1.0
        self.peers = matrix

    def step(self, x, given, grad=False, generator=None, sigma=0.0,
             theta_cls=None, temp=None) -> lattice.Step:
        single = (x.sum(-1) == 1).unsqueeze(-1)
        committed = torch.where(single, x, torch.zeros_like(x))
        present = torch.einsum(
            "pq,bqd->bpd", self.peers.to(x.dtype), committed) > 0.5
        heads = lattice.Heads(
            torch.where(present, -10.0, 10.0), torch.zeros_like(x),
            torch.full_like(x[..., :1], -10.0))
        found = lattice.project(x, given, heads, generator=generator)
        return lattice.Step(x, found.x, heads, found.conflict,
                            found.solved, found.kill, found.decided)


def test_oracle_solver():
    rng = np.random.default_rng(6)
    pairs = easy_puzzles(rng, 19) + [stall_puzzle(rng)]
    puzzles = np.stack([puzzle for puzzle, _ in pairs])
    solutions = np.stack([solution for _, solution in pairs])
    x = lattice.lattice_of(puzzles)
    y = one_hot(solutions)

    result = lattice_solve.solve(
        Oracle(), x, y, lattice_solve.SolveConfig(
            max_rounds=400, n_chains=64, batch_size=128, seed=0),
        verbose=False)
    summary = result.summary()
    assert summary["correct"] == 20
    assert summary["wrong"] == 0
    assert summary["timeout"] == 0
    assert int(result.n_resets[:19].sum()) == 0, \
        "an easy puzzle should never guess, let alone reset"
    assert int(result.n_resets[19]) >= 1, \
        "the forced-guess puzzle should reset at least one chain"
    assert result.deduced > 0 and result.active_chain_rounds > 0


# --- 7. the pool ------------------------------------------------------------

def test_pool_discard_and_refill():
    rng = np.random.default_rng(7)
    pairs = [random_puzzle(rng, 30) for _ in range(64)]
    puzzles = np.stack([puzzle for puzzle, _ in pairs])
    solutions = np.stack([solution for _, solution in pairs])
    stream = lattice_train.Stream(
        puzzles, solutions, np.random.default_rng(0))

    net = tiny(dtype=torch.float32)
    cfg = lattice_train.Config(steps=1, batch_size=6, max_age=100)
    trainer = lattice_train.Trainer(net, stream, cfg)
    y = trainer.y

    x_new = trainer.x.clone()
    conflict = torch.zeros(6, dtype=torch.bool)
    solved = torch.zeros(6, dtype=torch.bool)
    x_new[0] = y[0]                       # solved and correct
    solved[0] = True
    x_new[1] = y[1].roll(1, dims=-1)      # all singleton, every cell wrong
    solved[1] = True
    x_new[2, 0] = 1.0 - y[2, 0]           # the true bit is dead
    conflict[2] = True                    # ... and detected: true positive
    conflict[3] = True                    # detected on a consistent row
    trainer.age[4] = 99                   # hits the cap this iteration
    step = lattice.Step(None, x_new, None, conflict, solved,
                        torch.zeros(6, 81, 9, dtype=torch.bool),
                        torch.zeros(6, dtype=torch.bool))
    before_x0 = trainer.x0.clone()
    stats = trainer.apply(step)

    assert stats["discarded"] == 3
    assert stats["solved_correct"] == 1
    assert stats["tp_conflict"] == 1
    assert stats["aged"] == 1
    for row in (0, 2, 4):
        assert torch.equal(trainer.x[row], trainer.x0[row]), \
            "a refilled row restarts from its fresh x0"
        assert int(trainer.age[row]) == 0
        assert not torch.equal(trainer.x0[row], before_x0[row])
    for row in (1, 3, 5):
        assert torch.equal(trainer.x[row], x_new[row])
        assert torch.equal(trainer.x0[row], before_x0[row])
        assert int(trainer.age[row]) == 1

    counting = lattice_train.Stream(
        puzzles, solutions, np.random.default_rng(1))
    counting.take(2000)
    for kind, expected in zip(lattice_train.KINDS, lattice_train.PROBS):
        assert abs(counting.counts[kind] / 2000 - expected) < 0.05

    puzzle, solution = pairs[0]
    raw = lattice.lattice_of(puzzle[None])[0].numpy()
    hint = lattice_train.make_sample(
        raw.copy(), one_hot(solution)[0].numpy(),
        np.random.default_rng(2), "error")
    given = hint.sum(-1) == 1
    assert given.sum() > (puzzle > 0).sum(), \
        "the given mask includes the hint fills"
    corrupted = (hint.sum(-1) == 1) & ~(
        (hint > 0.5) & (one_hot(solution)[0].numpy() > 0.5)).any(-1)
    assert corrupted.any() and given[corrupted].all(), \
        "corrupted fills are givens too"


# --- 8. the loss ------------------------------------------------------------

def test_losses():
    rng = np.random.default_rng(8)
    puzzle, solution = random_puzzle(rng, 30)
    x = lattice.lattice_of(np.stack([puzzle, puzzle])).double()
    y = one_hot(np.stack([solution, solution])).double()
    x[1, 2] = 1.0 - y[1, 2]               # row 1 is UNSAT
    given = x.sum(-1) == 1
    torch.manual_seed(8)
    heads = lattice.Heads(torch.randn(2, 81, 9, dtype=torch.float64),
                          torch.randn(2, 81, 9, dtype=torch.float64),
                          torch.randn(2, 81, 1, dtype=torch.float64))

    total, parts = lattice.losses(heads, x, y, given)

    logsig = torch.nn.functional.logsigmoid
    target = x * y
    bce = -(4.0 * target * logsig(heads.bce)
            + 0.5 * (1 - target) * logsig(-heads.bce)).mean()
    assert torch.allclose(parts["bce"], bce)

    cell, board = lattice.conflicts(x, y)
    assert bool(board[1]) and not bool(board[0])
    mask = ~given & ~board.unsqueeze(-1)
    assert not bool(mask[1].any()), "an UNSAT board is excluded from CE"
    assert not bool(mask[0][given[0]].any()), "givens are excluded from CE"
    expected = bce + 0.2 * torch.nn.functional.cross_entropy(
        heads.sm[mask], y.argmax(-1)[mask])
    expected = expected \
        + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
            lattice.board_logit(heads.conf), board.double())
    assert torch.allclose(total, expected), \
        "the default loss is board-level conflict BCE only"
    with_cell, _ = lattice.losses(heads, x, y, given, cell_conf=True)
    assert torch.allclose(
        with_cell, expected
        + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(
            heads.conf.squeeze(-1), cell.double()))


# --- 9. a tiny training run -------------------------------------------------

def test_smoke_train_decreases_loss():
    rng = np.random.default_rng(9)
    pairs = [random_puzzle(rng, int(rng.integers(1, 4)))
             for _ in range(60)]
    puzzles = np.stack([puzzle for puzzle, _ in pairs])
    solutions = np.stack([solution for _, solution in pairs])
    stream = lattice_train.Stream(
        puzzles, solutions, np.random.default_rng(0))

    # the "halt" init, so the conflict head does not spend the whole of
    # this tiny run in the default init's always-fire transient (the
    # logsumexp aggregation starts the board logit at about +4.4).
    torch.manual_seed(0)
    net = lattice.build(TINY, rounds=ROUNDS, cycles=CYCLES,
                        conf_init="halt").to(torch.float32)
    cfg = lattice_train.Config(steps=60, batch_size=32, log_every=5,
                               seed=0)
    history = lattice_train.run(net, stream, cfg, eval_data=None,
                                log=lambda *args: None)

    losses = [record["loss"] for record in history]
    assert all(np.isfinite(losses))
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    assert sum(record["solved_correct"] for record in history) > 0
    assert all(0.0 <= record["unsound"] <= 1.0 for record in history)
