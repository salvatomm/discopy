# -*- coding: utf-8 -*-

"""
The fused Triton round of :mod:`discopy.neural.fused` against the reference
step of ``CMap._step_body`` on the sudoku lattice model: the forward and
every gradient, the exactness of the hand-written backward in float64, the
fallback on a map the kernels do not serve, and a training step of the
lattice trainer with the fused path on and off.

Everything here needs a GPU and triton.  Float32 comparisons run at
``"highest"`` matmul precision on both sides, so that only rounding
differs, and on inputs at which no ReLU of the round sits within ``3e-6``
of its kink: a kink that the two roundings straddle is an O(1) change of
one term of a weight gradient that no tolerance on rounding covers, and
at batch 512 -- twenty-five million ReLU evaluations -- some always do, so
that comparison runs in float64.  A float32 weight gradient is a sum of
thousands of terms that largely cancel, whose rounding alone reaches
``1e-4`` of its largest entry, so the float32 gradients are held to that
rather than entry by entry; the float64 ones are held entry by entry.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused round runs on a GPU")

SUDOKU = Path(__file__).resolve().parents[2] / "docs" / "neural" \
    / "examples" / "sudoku"
if str(SUDOKU) not in sys.path:
    sys.path.insert(0, str(SUDOKU))

import lattice                                          # noqa: E402
import lattice_train                                    # noqa: E402
import model as zoo                                     # noqa: E402
from config import Widths                               # noqa: E402
from discopy.neural import fused                        # noqa: E402

HALF = Widths(dim=13, state_dim=34, hidden=68, y_dim=17)
TINY = Widths(dim=8, state_dim=16, hidden=32, y_dim=8)
DEVICE = torch.device("cuda")
CLOSE = dict(atol=1e-5, rtol=1e-4)


@pytest.fixture(autouse=True)
def highest():
    """ Full-precision float32 dots on both sides. """
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(precision)


def build(widths=HALF, dtype=torch.float32, seed=0):
    """ The lattice model, its closed map and the map's device routing. """
    torch.manual_seed(seed)
    net = lattice.build(widths, rounds=2, cycles=2).to(DEVICE).to(dtype)
    cmap = net.interaction.cmap
    return net, cmap, cmap._device_routing(DEVICE)


def round_of(step, x, init, params, seed=1):
    """
    One round forward and backward under a random cotangent: the output,
    the gradients of ``x``, ``init`` and the parameters, and the group
    outputs.
    """
    x = x.clone().requires_grad_(True)
    init = None if init is None else init.clone().requires_grad_(True)
    for param in params:
        param.grad = None
    out, groups = step(x, None, init)
    torch.manual_seed(seed)
    (out * torch.randn_like(out)).sum().backward()
    grads = [x.grad, None if init is None else init.grad] \
        + [param.grad for param in params]
    return out.detach(), grads, [group.detach() for group in groups]


def clear_of_kinks(cmap, rows, dtype, margin=3e-6):
    """
    A random input at which every ReLU of the reference round is at least
    ``margin`` from its kink, reseeding until one is.
    """
    geo, closest = fused.geometry(cmap), []
    site, relation = (meta[0] for meta in cmap._fused_routing["metas"])
    hooks = [layer.register_forward_hook(
        lambda layer, args, out: closest.append(out.abs().min().item()))
        for layer in (site.encode[0], relation.phi[0], relation.rho[0])]
    for seed in range(rows, rows + 100):
        torch.manual_seed(seed)
        x = torch.randn(rows, geo.total, device=DEVICE, dtype=dtype)
        closest.clear()
        with torch.no_grad():
            cmap._step_body(DEVICE)(x, None, None)
        if min(closest) > margin:
            break
    for hook in hooks:
        hook.remove()
    return x


def both(rows, dtype, init=False, widths=HALF):
    """ The reference round and the fused round on the same random input. """
    net, cmap, routing = build(widths, dtype)
    assert fused.geometry(cmap) is not None
    x = clear_of_kinks(cmap, rows, dtype) if dtype == torch.float32 \
        else torch.randn(rows, fused.geometry(cmap).total, device=DEVICE,
                         dtype=dtype)
    init = torch.randn_like(x) if init else None
    params = list(net.map.ar.parameters())
    reference = round_of(cmap._step_body(DEVICE), x, init, params)
    found = round_of(fused.step_of(cmap, routing), x, init, params)
    return reference, found


@pytest.mark.parametrize("rows", [1, 7, 512])
def test_forward_matches_reference(rows):
    (out, _, groups), (found, _, found_groups) = both(
        rows, torch.float32, init=rows == 7)
    torch.testing.assert_close(found, out, **CLOSE)
    assert len(found_groups) == len(groups) == 2
    for group, found_group in zip(groups, found_groups):
        torch.testing.assert_close(found_group, group, **CLOSE)


@pytest.mark.parametrize("rows, dtype", [
    (1, torch.float32), (7, torch.float32), (512, torch.float64)])
def test_gradients_match_reference(rows, dtype):
    (_, grads, _), (_, found, _) = both(rows, dtype, init=rows == 7)
    assert len(found) == len(grads) == 2 + 18
    for grad, found_grad in zip(grads, found):
        if grad is None:
            assert found_grad is None
        elif dtype == torch.float64:
            torch.testing.assert_close(found_grad, grad, **CLOSE)
        else:
            torch.testing.assert_close(
                found_grad, grad, atol=1e-4 * grad.abs().max().item(),
                rtol=1e-4)


def test_backward_is_exact():
    """ The hand-written backward passes ``gradcheck`` in float64. """
    _, cmap, routing = build(TINY, torch.float64)
    geo = fused.geometry(cmap)
    params = fused.parameters(*(meta[0] for meta in routing["metas"]))
    x = torch.randn(2, geo.total, device=DEVICE, dtype=torch.float64,
                    requires_grad=True)

    def once(x, *params):
        return fused.FusedRound.apply(
            x, None, routing["perm_inverse"], geo, "ieee", *params)
    assert torch.autograd.gradcheck(
        once, (x, *params), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_no_grad_keeps_nothing():
    """ Under ``no_grad`` the round allocates its output and nothing else. """
    _, cmap, routing = build()
    step = fused.step_of(cmap, routing)
    x = torch.randn(3, fused.geometry(cmap).total, device=DEVICE)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        out, _ = step(x, None, None)
    assert not out.requires_grad
    assert torch.cuda.memory_allocated() - before <= 2 * out.nbytes


def test_other_maps_keep_the_reference_step():
    """
    Model A erases the answer role, so its site reads one given role
    rather than two: no geometry, and :meth:`CMap.compile_fused` leaves
    the reference step in place.  Model C is the lattice's map and is
    served.
    """
    torch.manual_seed(0)
    cmap = zoo.goi(HALF).to(DEVICE).interaction.cmap
    assert fused.geometry(cmap) is None
    assert fused.step_of(cmap, cmap._device_routing(DEVICE)) is None
    step = cmap.compile_fused()._step_body(DEVICE)
    x = torch.randn(2, cmap._routing["total"], device=DEVICE)
    out, groups = step(x, None, None)
    assert out.shape == x.shape and len(groups) == 2
    assert fused.geometry(zoo.trm(HALF).interaction.cmap) is not None


def puzzles(rng, count: int) -> tuple:
    """ Digit-relabeled copies of one solved grid with cells blanked. """
    row, col = np.divmod(np.arange(81), 9)
    grid = (3 * (row % 3) + row // 3 + col) % 9 + 1
    pairs = []
    for _ in range(count):
        solution = np.concatenate([[0], 1 + rng.permutation(9)])[grid]
        puzzle = solution.copy()
        puzzle[rng.choice(81, size=int(rng.integers(20, 50)),
                          replace=False)] = 0
        pairs.append((puzzle, solution))
    return tuple(np.stack(side) for side in zip(*pairs))


def losses(fused_round: bool, steps: int = 5) -> list:
    """ The loss of each of ``steps`` trainer iterations from seed 0. """
    torch.manual_seed(0)
    net = lattice.build(HALF, rounds=2, cycles=2).to(DEVICE)
    stream = lattice_train.Stream(
        *puzzles(np.random.default_rng(0), 96),
        np.random.default_rng(0), device=DEVICE)
    cfg = lattice_train.Config(
        steps=steps, batch_size=32, log_every=1, eval_every=0, seed=0,
        fast=True, fused=fused_round)
    history = lattice_train.run(
        net, stream, cfg, eval_data=None, log=lambda line: None)
    return [record["loss"] for record in history]


def test_training_step_agrees():
    """ Five trainer iterations, the fused round off and on. """
    reference, found = losses(False), losses(True)
    assert len(reference) == len(found) == 5
    for loss, found_loss in zip(reference, found):
        assert abs(found_loss - loss) <= 1e-4 * abs(loss), (loss, found_loss)
