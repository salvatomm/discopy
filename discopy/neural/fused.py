# -*- coding: utf-8 -*-

"""
One round of message passing over a site-and-relation map as two fused
Triton kernels, with their hand-written backward.

The reference round of :meth:`~discopy.neural.CMap._step_body` runs the
shared :class:`~discopy.neural.Site` on every cell, the shared
:class:`~discopy.neural.Relation` on every unit and routes the outputs
along the wires -- some forty small kernels forward and twice as many
backward, every activation of every cell taking a trip to memory in
between.  Here the whole cell -- encoder, pooling, ``GRUCell``,
``LayerNorm``, emission and the echoes of its traced roles -- is one
kernel, the whole unit -- ``phi``, pooling, ``rho`` -- is another, both
read their weights once per program and keep a cell's activations in
registers, and both write their outputs straight to the ports the wires
carry them to, so the routing permutation costs nothing either: one round
is two launches forward.  The backward is four kernels -- the cell's in
two halves, gates and encoder, the unit's data and weight gradients --
and the cell's weight gradients as a handful of GEMMs over every cell of
the batch on what the kernels kept; the forward saves only what the
backward cannot cheaply recompute, the gates and the normalisation
statistics of the cell and nothing at all of the unit.

A ``[rows, hidden]`` tile of width 68 is held as a ``[rows, 64]`` tile
and a ``[rows, 16]`` tile, since a Triton dot wants powers of two: padding
68 to 128 would double the arithmetic where splitting it costs a fifth.
Each dot's float32 precision follows ``torch.get_float32_matmul_precision``
like the reference GEMMs do, tensor cores under ``"high"`` and
``"medium"``, IEEE under ``"highest"``; float64 is exact, which is what
:func:`torch.autograd.gradcheck` wants.

Nothing here is on by default: :meth:`~discopy.neural.CMap.compile_fused`
opts a map in, and only a map whose :func:`geometry` the kernels serve.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:

    geometry
    fused_step
    step_of
"""

from __future__ import annotations

from typing import NamedTuple

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover
    from types import SimpleNamespace
    triton = SimpleNamespace(jit=lambda function: function)
    tl = libdevice = None

from discopy.neural.cells import POOL, Relation, Site
from discopy.neural.core import _perm_gather

#: Cells or units one program handles, and the warps it runs on, per kernel.
BLOCK = {"cell_fwd": 64, "cell_bwd_gate": 32, "cell_bwd_encode": 32,
         "unit_fwd": 64, "unit_bwd_leg": 64, "unit_bwd_weight": 64}
WARPS = 4

#: The widest tile a kernel holds: a role padded past it needs more shared
#: memory than an SM has.
TILE = 128


class Geometry(NamedTuple):
    """
    The shape of a site-and-relation map, as the kernels read it: the
    flat state is ``cells`` blocks of width ``cell`` -- ``legs`` message
    legs of width ``d``, then two copies each of a state of width ``s``,
    a clue of width ``c`` and an answer of width ``y`` -- followed at
    ``unit_offset`` by ``units`` blocks of ``members`` legs.
    """
    total: int
    cells: int
    cell: int
    units: int
    unit: int
    unit_offset: int
    legs: int
    members: int
    d: int
    s: int
    h: int
    hu: int
    c: int
    y: int
    echo: tuple
    eps: float


def _pow2(n: int) -> int:
    """ The least power of two that is at least ``n`` and at least 16. """
    return max(16, 1 << (n - 1).bit_length())


def _split(h: int) -> tuple[int, int]:
    """ The hidden width as a big and a small power-of-two tile. """
    big = 1 << (h.bit_length() - 1)
    return big, _pow2(h - big) if h > big else 16


def _pads(geo: Geometry) -> dict:
    """ The padded tile widths of a geometry, as kernel constants. """
    ha, hb = _split(geo.h)
    ua, ub = _split(geo.hu)
    return dict(DP=_pow2(geo.d), SP=_pow2(geo.s), HA=ha, HB=hb,
                UA=ua, UB=ub, CP=_pow2(geo.c), YP=_pow2(geo.y))


# --- the tiles -----------------------------------------------------------

@triton.jit
def _rows(ptr, base, off, n, valid, W: tl.constexpr):
    """ ``ptr[base + off + w]`` as a ``[BM, W]`` tile, zero at ``w >= n``. """
    w = tl.arange(0, W)
    return tl.load(ptr + base[:, None] + off + w[None, :],
                   mask=valid[:, None] & (w[None, :] < n), other=0.0)


@triton.jit
def _put(ptr, base, off, n, valid, tile, W: tl.constexpr):
    """ The store of :func:`_rows`. """
    w = tl.arange(0, W)
    tl.store(ptr + base[:, None] + off + w[None, :], tile,
             mask=valid[:, None] & (w[None, :] < n))


@triton.jit
def _emit(ptr, init_ptr, base, off, n, valid, tile,
          W: tl.constexpr, HAS_INIT: tl.constexpr):
    """ :func:`_put`, adding the injected messages when there are any. """
    if HAS_INIT:
        tile = tile + _rows(init_ptr, base, off, n, valid, W)
    _put(ptr, base, off, n, valid, tile, W)


@triton.jit
def _vec(ptr, off, n, W: tl.constexpr):
    """ ``ptr[off + w]`` as a ``[W]`` vector, zero at ``w >= n``. """
    w = tl.arange(0, W)
    return tl.load(ptr + off + w, mask=w < n, other=0.0)


@triton.jit
def _lin(x, ptr, stride, k0, n0, k_hi, n_hi,
         K: tl.constexpr, N: tl.constexpr, PREC: tl.constexpr):
    """
    ``x @ W[n0:n_hi, k0:k_hi].T`` for a row-major ``W``: a linear layer
    read on the tile ``x`` of its input columns ``k0`` onwards.
    """
    k = k0 + tl.arange(0, K)
    n = n0 + tl.arange(0, N)
    w = tl.load(ptr + n[None, :] * stride + k[:, None],
                mask=(k[:, None] < k_hi) & (n[None, :] < n_hi), other=0.0)
    return tl.dot(x, w, input_precision=PREC)


@triton.jit
def _back(g, ptr, stride, r0, c0, r_hi, c_hi,
          R: tl.constexpr, C: tl.constexpr, PREC: tl.constexpr):
    """
    ``g @ W[r0:r_hi, c0:c_hi]``: the gradient of a linear layer's input
    columns ``c0`` onwards, from the gradient ``g`` of its outputs ``r0``
    onwards.
    """
    r = r0 + tl.arange(0, R)
    c = c0 + tl.arange(0, C)
    w = tl.load(ptr + r[:, None] * stride + c[None, :],
                mask=(r[:, None] < r_hi) & (c[None, :] < c_hi), other=0.0)
    return tl.dot(g, w, input_precision=PREC)


@triton.jit
def _outer(g, x, PREC: tl.constexpr):
    """ ``g.T @ x``, the weight gradient of a linear layer on a block. """
    return tl.dot(tl.trans(g), x, input_precision=PREC)


@triton.jit
def _acc(ptr, stride, r0, c0, r_hi, c_hi, tile,
         R: tl.constexpr, C: tl.constexpr):
    """ Add a tile into a program's partial weight gradient. """
    r = r0 + tl.arange(0, R)
    c = c0 + tl.arange(0, C)
    p = ptr + r[:, None] * stride + c[None, :]
    mask = (r[:, None] < r_hi) & (c[None, :] < c_hi)
    tl.store(p, tl.load(p, mask=mask, other=0.0) + tile, mask=mask)


@triton.jit
def _acc_vec(ptr, off, n, vec, W: tl.constexpr):
    """ Add a vector into a program's partial bias gradient. """
    w = tl.arange(0, W)
    p = ptr + off + w
    tl.store(p, tl.load(p, mask=w < n, other=0.0) + vec, mask=w < n)


@triton.jit
def _sigmoid(x):
    return 1 / (1 + tl.exp(-x))


@triton.jit
def _tanh(x):
    return libdevice.tanh(x)


# --- the cell ------------------------------------------------------------

@triton.jit
def _gate(pool_a, pool_b, c, a, s, wih, whh, bih, bhh, n0,
          S: tl.constexpr, H: tl.constexpr, C: tl.constexpr, Y: tl.constexpr,
          SP: tl.constexpr, HA: tl.constexpr, HB: tl.constexpr,
          CP: tl.constexpr, YP: tl.constexpr, PREC: tl.constexpr):
    """ The input and hidden halves of one GRU gate, rows ``n0`` on. """
    X: tl.constexpr = H + C + Y
    n_hi = n0 + S
    gi = _lin(pool_a, wih, X, 0, n0, H, n_hi, HA, SP, PREC) \
        + _lin(pool_b, wih, X, HA, n0, H, n_hi, HB, SP, PREC) \
        + _lin(c, wih, X, H, n0, H + C, n_hi, CP, SP, PREC) \
        + _lin(a, wih, X, H + C, n0, X, n_hi, YP, SP, PREC) \
        + _vec(bih, n0, S, SP)[None, :]
    gh = _lin(s, whh, S, 0, n0, S, n_hi, SP, SP, PREC) \
        + _vec(bhh, n0, S, SP)[None, :]
    return gi, gh


@triton.jit
def _gate_grads(part_ih, part_hh, part_bih, part_bhh, gi, gh, n0,
                pool_a, pool_b, c, a, s,
                S: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                Y: tl.constexpr, SP: tl.constexpr, HA: tl.constexpr,
                HB: tl.constexpr, CP: tl.constexpr, YP: tl.constexpr,
                PREC: tl.constexpr):
    """ The weight gradients of one GRU gate from a block. """
    X: tl.constexpr = H + C + Y
    n_hi = n0 + S
    _acc(part_ih, X, n0, 0, n_hi, H, _outer(gi, pool_a, PREC), SP, HA)
    _acc(part_ih, X, n0, HA, n_hi, H, _outer(gi, pool_b, PREC), SP, HB)
    _acc(part_ih, X, n0, H, n_hi, H + C, _outer(gi, c, PREC), SP, CP)
    _acc(part_ih, X, n0, H + C, n_hi, X, _outer(gi, a, PREC), SP, YP)
    _acc_vec(part_bih, n0, S, tl.sum(gi, 0), SP)
    _acc(part_hh, S, n0, 0, n_hi, S, _outer(gh, s, PREC), SP, SP)
    _acc_vec(part_bhh, n0, S, tl.sum(gh, 0), SP)


@triton.jit
def _cell_fwd(x_ptr, out_ptr, init_ptr, save_ptr, pinv_ptr, n_items, eps,
              w1, b1, w2, b2, wih, whh, bih, bhh, gamma, beta, we, be,
              TOTAL: tl.constexpr, NC: tl.constexpr, CW: tl.constexpr,
              LEGS: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
              H: tl.constexpr, C: tl.constexpr, Y: tl.constexpr,
              ECHO_C: tl.constexpr, ECHO_A: tl.constexpr,
              DP: tl.constexpr, SP: tl.constexpr, HA: tl.constexpr,
              HB: tl.constexpr, CP: tl.constexpr, YP: tl.constexpr,
              BM: tl.constexpr, HAS_INIT: tl.constexpr, SAVE: tl.constexpr,
              PREC: tl.constexpr):
    """ :meth:`Site.forward` on a block of cells, routed on the way out. """
    O_SI: tl.constexpr = LEGS * D
    O_SO: tl.constexpr = O_SI + S
    O_CI: tl.constexpr = O_SO + S
    O_CO: tl.constexpr = O_CI + C
    O_AI: tl.constexpr = O_CO + C
    O_AO: tl.constexpr = O_AI + Y
    SV: tl.constexpr = 4 * S + H + 2
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = m < n_items
    row = (m // NC).to(tl.int64)
    cell = m % NC
    base = row * TOTAL + cell * CW
    s = _rows(x_ptr, base, O_SI, S, valid, SP)
    c = _rows(x_ptr, base, O_CI, C, valid, CP)
    a = _rows(x_ptr, base, O_AI, Y, valid, YP)

    zs_a = _lin(s, w1, S + D, 0, 0, S, H, SP, HA, PREC)
    zs_b = _lin(s, w1, S + D, 0, HA, S, H, SP, HB, PREC)
    b1_a = _vec(b1, 0, H, HA)[None, :]
    b1_b = _vec(b1, HA, H - HA, HB)[None, :]
    pool_a = tl.zeros([BM, HA], dtype=s.dtype)
    pool_b = tl.zeros([BM, HB], dtype=s.dtype)
    for i in range(LEGS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        h1_a = tl.maximum(
            zs_a + _lin(leg, w1, S + D, S, 0, S + D, H, DP, HA, PREC) + b1_a,
            0.0)
        h1_b = tl.maximum(
            zs_b + _lin(leg, w1, S + D, S, HA, S + D, H, DP, HB, PREC) + b1_b,
            0.0)
        pool_a += _lin(h1_a, w2, H, 0, 0, H, H, HA, HA, PREC) \
            + _lin(h1_b, w2, H, HA, 0, H, H, HB, HA, PREC)
        pool_b += _lin(h1_a, w2, H, 0, HA, H, H, HA, HB, PREC) \
            + _lin(h1_b, w2, H, HA, HA, H, H, HB, HB, PREC)
    pool_a = pool_a / LEGS + _vec(b2, 0, H, HA)[None, :]
    pool_b = pool_b / LEGS + _vec(b2, HA, H - HA, HB)[None, :]

    ir, hr = _gate(pool_a, pool_b, c, a, s, wih, whh, bih, bhh, 0,
                   S, H, C, Y, SP, HA, HB, CP, YP, PREC)
    iz, hz = _gate(pool_a, pool_b, c, a, s, wih, whh, bih, bhh, S,
                   S, H, C, Y, SP, HA, HB, CP, YP, PREC)
    inn, hn = _gate(pool_a, pool_b, c, a, s, wih, whh, bih, bhh, 2 * S,
                    S, H, C, Y, SP, HA, HB, CP, YP, PREC)
    r = _sigmoid(ir + hr)
    z = _sigmoid(iz + hz)
    n = _tanh(inn + r * hn)
    h = (1 - z) * n + z * s

    smask = (tl.arange(0, SP) < S)[None, :]
    mu = tl.sum(h, 1) / S
    xc = tl.where(smask, h - mu[:, None], 0.0)
    rstd = 1 / tl.sqrt(tl.sum(xc * xc, 1) / S + eps)
    y = xc * rstd[:, None] * _vec(gamma, 0, S, SP)[None, :] \
        + _vec(beta, 0, S, SP)[None, :]
    belief = _lin(y, we, S, 0, 0, S, D, SP, DP, PREC) \
        + _vec(be, 0, D, DP)[None, :]

    for i in range(LEGS):
        dest = row * TOTAL + tl.load(
            pinv_ptr + cell * CW + i * D, mask=valid, other=0)
        _emit(out_ptr, init_ptr, dest, 0, D, valid, belief, DP, HAS_INIT)
    _emit(out_ptr, init_ptr, base, O_SI, S, valid, y, SP, HAS_INIT)
    _emit(out_ptr, init_ptr, base, O_SO, S, valid, y, SP, HAS_INIT)
    if not ECHO_C:
        c = tl.zeros_like(c)
    if not ECHO_A:
        a = tl.zeros_like(a)
    _emit(out_ptr, init_ptr, base, O_CI, C, valid, c, CP, HAS_INIT)
    _emit(out_ptr, init_ptr, base, O_CO, C, valid, c, CP, HAS_INIT)
    _emit(out_ptr, init_ptr, base, O_AI, Y, valid, a, YP, HAS_INIT)
    _emit(out_ptr, init_ptr, base, O_AO, Y, valid, a, YP, HAS_INIT)
    if SAVE:
        sv = m.to(tl.int64) * SV
        _put(save_ptr, sv, 0, S, valid, r, SP)
        _put(save_ptr, sv, S, S, valid, z, SP)
        _put(save_ptr, sv, 2 * S, S, valid, n, SP)
        _put(save_ptr, sv, 3 * S, S, valid, hn, SP)
        _put(save_ptr, sv, 4 * S, HA, valid, pool_a, HA)
        _put(save_ptr, sv, 4 * S + HA, H - HA, valid, pool_b, HB)
        tl.store(save_ptr + sv + 4 * S + H, mu, mask=valid)
        tl.store(save_ptr + sv + 4 * S + H + 1, rstd, mask=valid)


@triton.jit
def _cell_bwd_gate(x_ptr, g_ptr, gin_ptr, save_ptr, pool_ptr, keep_ptr,
                   pinv_ptr, n_items, wih, whh, gamma, beta, we,
                   TOTAL: tl.constexpr, NC: tl.constexpr, CW: tl.constexpr,
                   LEGS: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                   H: tl.constexpr, C: tl.constexpr, Y: tl.constexpr,
                   ECHO_C: tl.constexpr, ECHO_A: tl.constexpr,
                   DP: tl.constexpr, SP: tl.constexpr, HA: tl.constexpr,
                   HB: tl.constexpr, CP: tl.constexpr, YP: tl.constexpr,
                   BM: tl.constexpr, PREC: tl.constexpr):
    """
    The first half of the backward of :func:`_cell_fwd`, from the output
    ports down to the pooled encoding: the gradients of the state, clue
    and answer ports, the pooled gradient handed to
    :func:`_cell_bwd_encode`, and, kept for the weight gradients of
    :func:`_cell_grads`, the gate gradients, the GRU's input and the
    state each with a column of ones, the normalised state with one, the
    belief's gradient, and the state's gradient beside its product with
    the normalised state.
    """
    O_SI: tl.constexpr = LEGS * D
    O_SO: tl.constexpr = O_SI + S
    O_CI: tl.constexpr = O_SO + S
    O_CO: tl.constexpr = O_CI + C
    O_AI: tl.constexpr = O_CO + C
    O_AO: tl.constexpr = O_AI + Y
    X: tl.constexpr = H + C + Y
    SV: tl.constexpr = 4 * S + H + 2
    K_X: tl.constexpr = 4 * S
    K_S: tl.constexpr = K_X + X + 1
    K_Y: tl.constexpr = K_S + S + 1
    K_B: tl.constexpr = K_Y + S + 1
    K_G: tl.constexpr = K_B + D
    KW: tl.constexpr = K_G + 2 * S
    smask = (tl.arange(0, SP) < S)[None, :]
    gam = _vec(gamma, 0, S, SP)[None, :]
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = m < n_items
    row = (m // NC).to(tl.int64)
    cell = m % NC
    base = row * TOTAL + cell * CW
    sv = m.to(tl.int64) * SV
    keep = m.to(tl.int64) * KW
    s = _rows(x_ptr, base, O_SI, S, valid, SP)
    ones = tl.full([BM], 1.0, dtype=s.dtype)
    _put(keep_ptr, keep, K_X, HA, valid,
         _rows(save_ptr, sv, 4 * S, HA, valid, HA), HA)
    _put(keep_ptr, keep, K_X + HA, H - HA, valid,
         _rows(save_ptr, sv, 4 * S + HA, H - HA, valid, HB), HB)
    _put(keep_ptr, keep, K_X + H, C, valid,
         _rows(x_ptr, base, O_CI, C, valid, CP), CP)
    _put(keep_ptr, keep, K_X + H + C, Y, valid,
         _rows(x_ptr, base, O_AI, Y, valid, YP), YP)
    tl.store(keep_ptr + keep + K_X + X, ones, mask=valid)
    _put(keep_ptr, keep, K_S, S, valid, s, SP)
    tl.store(keep_ptr + keep + K_S + S, ones, mask=valid)
    z = _rows(save_ptr, sv, S, S, valid, SP)
    n = _rows(save_ptr, sv, 2 * S, S, valid, SP)
    mu = tl.load(save_ptr + sv + 4 * S + H, mask=valid, other=0.0)
    rstd = tl.load(save_ptr + sv + 4 * S + H + 1, mask=valid, other=0.0)
    h = (1 - z) * n + z * s
    xhat = tl.where(smask, (h - mu[:, None]) * rstd[:, None], 0.0)
    _put(keep_ptr, keep, K_Y, S, valid,
         xhat * gam + _vec(beta, 0, S, SP)[None, :], SP)
    tl.store(keep_ptr + keep + K_Y + S, ones, mask=valid)

    g_belief = tl.zeros([BM, DP], dtype=s.dtype)
    for i in range(LEGS):
        dest = row * TOTAL + tl.load(
            pinv_ptr + cell * CW + i * D, mask=valid, other=0)
        g_belief += _rows(g_ptr, dest, 0, D, valid, DP)
    _put(keep_ptr, keep, K_B, D, valid, g_belief, DP)
    g_y = _rows(g_ptr, base, O_SI, S, valid, SP) \
        + _rows(g_ptr, base, O_SO, S, valid, SP) \
        + _back(g_belief, we, S, 0, 0, D, S, DP, SP, PREC)
    _put(keep_ptr, keep, K_G, S, valid, g_y * xhat, SP)
    _put(keep_ptr, keep, K_G + S, S, valid, g_y, SP)
    gx = g_y * gam
    mgx = tl.sum(gx, 1) / S
    mgxx = tl.sum(gx * xhat, 1) / S
    g_h = tl.where(smask, rstd[:, None] * (
        gx - mgx[:, None] - xhat * mgxx[:, None]), 0.0)

    g_s = g_h * z
    g_az = g_h * (s - n) * z * (1 - z)
    g_an = g_h * (1 - z) * (1 - n * n)
    r = _rows(save_ptr, sv, 0, S, valid, SP)
    g_hn = g_an * r
    g_ar = g_an * _rows(save_ptr, sv, 3 * S, S, valid, SP) * r * (1 - r)
    _put(keep_ptr, keep, 0, S, valid, g_ar, SP)
    _put(keep_ptr, keep, S, S, valid, g_az, SP)
    _put(keep_ptr, keep, 2 * S, S, valid, g_hn, SP)
    _put(keep_ptr, keep, 3 * S, S, valid, g_an, SP)
    g_s += _back(g_ar, whh, S, 0, 0, S, S, SP, SP, PREC) \
        + _back(g_az, whh, S, S, 0, 2 * S, S, SP, SP, PREC) \
        + _back(g_hn, whh, S, 2 * S, 0, 3 * S, S, SP, SP, PREC)
    _put(gin_ptr, base, O_SI, S, valid, g_s, SP)
    _put(gin_ptr, base, O_SO, S, valid, tl.zeros_like(g_s), SP)

    pm = m.to(tl.int64) * H
    _put(pool_ptr, pm, 0, HA, valid,
         _back(g_ar, wih, X, 0, 0, S, H, SP, HA, PREC)
         + _back(g_az, wih, X, S, 0, 2 * S, H, SP, HA, PREC)
         + _back(g_an, wih, X, 2 * S, 0, 3 * S, H, SP, HA, PREC), HA)
    _put(pool_ptr, pm, HA, H - HA, valid,
         _back(g_ar, wih, X, 0, HA, S, H, SP, HB, PREC)
         + _back(g_az, wih, X, S, HA, 2 * S, H, SP, HB, PREC)
         + _back(g_an, wih, X, 2 * S, HA, 3 * S, H, SP, HB, PREC), HB)
    g_c = _back(g_ar, wih, X, 0, H, S, H + C, SP, CP, PREC) \
        + _back(g_az, wih, X, S, H, 2 * S, H + C, SP, CP, PREC) \
        + _back(g_an, wih, X, 2 * S, H, 3 * S, H + C, SP, CP, PREC)
    if ECHO_C:
        g_c += _rows(g_ptr, base, O_CI, C, valid, CP) \
            + _rows(g_ptr, base, O_CO, C, valid, CP)
    _put(gin_ptr, base, O_CI, C, valid, g_c, CP)
    _put(gin_ptr, base, O_CO, C, valid, tl.zeros_like(g_c), CP)
    g_a = _back(g_ar, wih, X, 0, H + C, S, X, SP, YP, PREC) \
        + _back(g_az, wih, X, S, H + C, 2 * S, X, SP, YP, PREC) \
        + _back(g_an, wih, X, 2 * S, H + C, 3 * S, X, SP, YP, PREC)
    if ECHO_A:
        g_a += _rows(g_ptr, base, O_AI, Y, valid, YP) \
            + _rows(g_ptr, base, O_AO, Y, valid, YP)
    _put(gin_ptr, base, O_AI, Y, valid, g_a, YP)
    _put(gin_ptr, base, O_AO, Y, valid, tl.zeros_like(g_a), YP)


@triton.jit
def _cell_bwd_encode(x_ptr, gin_ptr, pool_ptr, keep_ptr, n_items, w1, b1, w2,
                     TOTAL: tl.constexpr, NC: tl.constexpr, CW: tl.constexpr,
                     LEGS: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
                     H: tl.constexpr, DP: tl.constexpr, SP: tl.constexpr,
                     HA: tl.constexpr, HB: tl.constexpr, BM: tl.constexpr,
                     PREC: tl.constexpr):
    """
    The second half of the backward of :func:`_cell_fwd`, the encoder:
    from the pooled gradient of :func:`_cell_bwd_gate`, recomputing the
    first layer from the incoming messages, the gradients of the message
    legs and of the state, and, kept for the weight gradients of
    :func:`_cell_grads`, the gradient of every leg's first-layer
    pre-activation, the legs read, the summed gradient and the summed
    activations with a column of ``LEGS``.
    """
    O_SI: tl.constexpr = LEGS * D
    K_L: tl.constexpr = LEGS * H
    K_P: tl.constexpr = K_L + LEGS * D
    K_H: tl.constexpr = K_P + H
    KW: tl.constexpr = K_H + H + 1
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = m < n_items
    row = (m // NC).to(tl.int64)
    cell = m % NC
    base = row * TOTAL + cell * CW
    pm = m.to(tl.int64) * H
    keep = m.to(tl.int64) * KW
    s = _rows(x_ptr, base, O_SI, S, valid, SP)
    g_h2_a = _rows(pool_ptr, pm, 0, HA, valid, HA) / LEGS
    g_h2_b = _rows(pool_ptr, pm, HA, H - HA, valid, HB) / LEGS
    g_h1_a = _back(g_h2_a, w2, H, 0, 0, H, H, HA, HA, PREC) \
        + _back(g_h2_b, w2, H, HA, 0, H, H, HB, HA, PREC)
    g_h1_b = _back(g_h2_a, w2, H, 0, HA, H, H, HA, HB, PREC) \
        + _back(g_h2_b, w2, H, HA, HA, H, H, HB, HB, PREC)
    zs_a = _lin(s, w1, S + D, 0, 0, S, H, SP, HA, PREC) \
        + _vec(b1, 0, H, HA)[None, :]
    zs_b = _lin(s, w1, S + D, 0, HA, S, H, SP, HB, PREC) \
        + _vec(b1, HA, H - HA, HB)[None, :]
    h1s_a = tl.zeros([BM, HA], dtype=s.dtype)
    h1s_b = tl.zeros([BM, HB], dtype=s.dtype)
    gp_a = tl.zeros([BM, HA], dtype=s.dtype)
    gp_b = tl.zeros([BM, HB], dtype=s.dtype)
    for i in range(LEGS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        _put(keep_ptr, keep, K_L + i * D, D, valid, leg, DP)
        pre_a = zs_a + _lin(leg, w1, S + D, S, 0, S + D, H, DP, HA, PREC)
        pre_b = zs_b + _lin(leg, w1, S + D, S, HA, S + D, H, DP, HB, PREC)
        h1s_a += tl.maximum(pre_a, 0.0)
        h1s_b += tl.maximum(pre_b, 0.0)
        gpre_a = tl.where(pre_a > 0, g_h1_a, 0.0)
        gpre_b = tl.where(pre_b > 0, g_h1_b, 0.0)
        gp_a += gpre_a
        gp_b += gpre_b
        _put(keep_ptr, keep, i * H, HA, valid, gpre_a, HA)
        _put(keep_ptr, keep, i * H + HA, H - HA, valid, gpre_b, HB)
        _put(gin_ptr, base, i * D, D, valid,
             _back(gpre_a, w1, S + D, 0, S, H, S + D, HA, DP, PREC)
             + _back(gpre_b, w1, S + D, HA, S, H, S + D, HB, DP, PREC), DP)
    _put(keep_ptr, keep, K_P, HA, valid, gp_a, HA)
    _put(keep_ptr, keep, K_P + HA, H - HA, valid, gp_b, HB)
    _put(keep_ptr, keep, K_H, HA, valid, h1s_a, HA)
    _put(keep_ptr, keep, K_H + HA, H - HA, valid, h1s_b, HB)
    tl.store(keep_ptr + keep + K_H + H, tl.full([BM], LEGS, dtype=s.dtype),
             mask=valid)
    _put(gin_ptr, base, O_SI, S, valid,
         _rows(gin_ptr, base, O_SI, S, valid, SP)
         + _back(gp_a, w1, S + D, 0, 0, H, S, HA, SP, PREC)
         + _back(gp_b, w1, S + D, HA, 0, H, S, HB, SP, PREC), SP)


# --- the unit ------------------------------------------------------------

@triton.jit
def _unit_fwd(x_ptr, out_ptr, init_ptr, pinv_ptr, n_items,
              wphi, bphi, wr1, br1, wr2, br2,
              TOTAL: tl.constexpr, UOFF: tl.constexpr, NU: tl.constexpr,
              UW: tl.constexpr, MEMBERS: tl.constexpr, D: tl.constexpr,
              H: tl.constexpr, DP: tl.constexpr, HA: tl.constexpr,
              HB: tl.constexpr, BM: tl.constexpr, HAS_INIT: tl.constexpr,
              PREC: tl.constexpr):
    """ :meth:`Relation.forward` on a block of units, routed on exit. """
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = m < n_items
    row = (m // NU).to(tl.int64)
    unit = m % NU
    base = row * TOTAL + UOFF + unit * UW
    bphi_a = _vec(bphi, 0, H, HA)[None, :]
    bphi_b = _vec(bphi, HA, H - HA, HB)[None, :]
    pool_a = tl.zeros([BM, HA], dtype=bphi_a.dtype)
    pool_b = tl.zeros([BM, HB], dtype=bphi_a.dtype)
    for i in range(MEMBERS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        pool_a += tl.maximum(
            _lin(leg, wphi, D, 0, 0, D, H, DP, HA, PREC) + bphi_a, 0.0)
        pool_b += tl.maximum(
            _lin(leg, wphi, D, 0, HA, D, H, DP, HB, PREC) + bphi_b, 0.0)
    zp_a = _lin(pool_a, wr1, D + H, D, 0, D + H, H, HA, HA, PREC) \
        + _lin(pool_b, wr1, D + H, D + HA, 0, D + H, H, HB, HA, PREC) \
        + _vec(br1, 0, H, HA)[None, :]
    zp_b = _lin(pool_a, wr1, D + H, D, HA, D + H, H, HA, HB, PREC) \
        + _lin(pool_b, wr1, D + H, D + HA, HA, D + H, H, HB, HB, PREC) \
        + _vec(br1, HA, H - HA, HB)[None, :]
    br2_v = _vec(br2, 0, D, DP)[None, :]
    for i in range(MEMBERS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        rho_a = tl.maximum(
            zp_a + _lin(leg, wr1, D + H, 0, 0, D, H, DP, HA, PREC), 0.0)
        rho_b = tl.maximum(
            zp_b + _lin(leg, wr1, D + H, 0, HA, D, H, DP, HB, PREC), 0.0)
        out = _lin(rho_a, wr2, H, 0, 0, H, D, HA, DP, PREC) \
            + _lin(rho_b, wr2, H, HA, 0, H, D, HB, DP, PREC) + br2_v
        dest = row * TOTAL + tl.load(
            pinv_ptr + UOFF + unit * UW + i * D, mask=valid, other=0)
        _emit(out_ptr, init_ptr, dest, 0, D, valid, out, DP, HAS_INIT)


@triton.jit
def _unit_bwd_leg(x_ptr, g_ptr, gin_ptr, pool_ptr, pinv_ptr, n_items,
                  wphi, bphi, wr1, br1, wr2,
                  TOTAL: tl.constexpr, UOFF: tl.constexpr, NU: tl.constexpr,
                  UW: tl.constexpr, MEMBERS: tl.constexpr, D: tl.constexpr,
                  H: tl.constexpr, DP: tl.constexpr, HA: tl.constexpr,
                  HB: tl.constexpr, BM: tl.constexpr, PREC: tl.constexpr):
    """
    The gradient of every leg of a block of units, recomputing the
    forward from the incoming messages, and the gradient of the pooled
    embedding handed to :func:`_unit_bwd_weight`.
    """
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = m < n_items
    row = (m // NU).to(tl.int64)
    unit = m % NU
    base = row * TOTAL + UOFF + unit * UW
    bphi_a = _vec(bphi, 0, H, HA)[None, :]
    bphi_b = _vec(bphi, HA, H - HA, HB)[None, :]
    pool_a = tl.zeros([BM, HA], dtype=bphi_a.dtype)
    pool_b = tl.zeros([BM, HB], dtype=bphi_a.dtype)
    for i in range(MEMBERS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        pool_a += tl.maximum(
            _lin(leg, wphi, D, 0, 0, D, H, DP, HA, PREC) + bphi_a, 0.0)
        pool_b += tl.maximum(
            _lin(leg, wphi, D, 0, HA, D, H, DP, HB, PREC) + bphi_b, 0.0)
    zp_a = _lin(pool_a, wr1, D + H, D, 0, D + H, H, HA, HA, PREC) \
        + _lin(pool_b, wr1, D + H, D + HA, 0, D + H, H, HB, HA, PREC) \
        + _vec(br1, 0, H, HA)[None, :]
    zp_b = _lin(pool_a, wr1, D + H, D, HA, D + H, H, HA, HB, PREC) \
        + _lin(pool_b, wr1, D + H, D + HA, HA, D + H, H, HB, HB, PREC) \
        + _vec(br1, HA, H - HA, HB)[None, :]
    gs_a = tl.zeros([BM, HA], dtype=pool_a.dtype)
    gs_b = tl.zeros([BM, HB], dtype=pool_a.dtype)
    for i in range(MEMBERS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        pre_a = zp_a + _lin(leg, wr1, D + H, 0, 0, D, H, DP, HA, PREC)
        pre_b = zp_b + _lin(leg, wr1, D + H, 0, HA, D, H, DP, HB, PREC)
        dest = row * TOTAL + tl.load(
            pinv_ptr + UOFF + unit * UW + i * D, mask=valid, other=0)
        g_out = _rows(g_ptr, dest, 0, D, valid, DP)
        gr_a = tl.where(
            pre_a > 0, _back(g_out, wr2, H, 0, 0, D, H, DP, HA, PREC), 0.0)
        gr_b = tl.where(
            pre_b > 0, _back(g_out, wr2, H, 0, HA, D, H, DP, HB, PREC), 0.0)
        gs_a += gr_a
        gs_b += gr_b
        _put(gin_ptr, base, i * D, D, valid,
             _back(gr_a, wr1, D + H, 0, 0, H, D, HA, DP, PREC)
             + _back(gr_b, wr1, D + H, HA, 0, H, D, HB, DP, PREC), DP)
    g_pool_a = _back(gs_a, wr1, D + H, 0, D, H, D + H, HA, HA, PREC) \
        + _back(gs_b, wr1, D + H, HA, D, H, D + H, HB, HA, PREC)
    g_pool_b = _back(gs_a, wr1, D + H, 0, D + HA, H, D + H, HA, HB, PREC) \
        + _back(gs_b, wr1, D + H, HA, D + HA, H, D + H, HB, HB, PREC)
    pm = m.to(tl.int64) * H
    _put(pool_ptr, pm, 0, HA, valid, g_pool_a, HA)
    _put(pool_ptr, pm, HA, H - HA, valid, g_pool_b, HB)
    for i in range(MEMBERS):
        leg = _rows(x_ptr, base, i * D, D, valid, DP)
        gphi_a = tl.where(
            _lin(leg, wphi, D, 0, 0, D, H, DP, HA, PREC) + bphi_a > 0,
            g_pool_a, 0.0)
        gphi_b = tl.where(
            _lin(leg, wphi, D, 0, HA, D, H, DP, HB, PREC) + bphi_b > 0,
            g_pool_b, 0.0)
        _put(gin_ptr, base, i * D, D, valid,
             _rows(gin_ptr, base, i * D, D, valid, DP)
             + _back(gphi_a, wphi, D, 0, 0, H, D, HA, DP, PREC)
             + _back(gphi_b, wphi, D, HA, 0, H, D, HB, DP, PREC), DP)


@triton.jit
def _unit_bwd_weight(x_ptr, g_ptr, pool_ptr, pinv_ptr, part_ptr, n_items,
                     n_blocks, n_part, wphi, bphi, wr1, br1, wr2,
                     TOTAL: tl.constexpr, UOFF: tl.constexpr,
                     NU: tl.constexpr, UW: tl.constexpr,
                     MEMBERS: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
                     DP: tl.constexpr, HA: tl.constexpr, HB: tl.constexpr,
                     BM: tl.constexpr, PREC: tl.constexpr):
    """
    This program's share of the weight gradients of the unit, recomputing
    the forward from the incoming messages and the pooled gradient of
    :func:`_unit_bwd_leg`.
    """
    P_BPHI: tl.constexpr = H * D
    P_WR1: tl.constexpr = P_BPHI + H
    P_BR1: tl.constexpr = P_WR1 + H * (D + H)
    P_WR2: tl.constexpr = P_BR1 + H
    P_BR2: tl.constexpr = P_WR2 + D * H
    part = part_ptr + tl.program_id(0).to(tl.int64) * n_part
    bphi_a = _vec(bphi, 0, H, HA)[None, :]
    bphi_b = _vec(bphi, HA, H - HA, HB)[None, :]
    br1_a = _vec(br1, 0, H, HA)[None, :]
    br1_b = _vec(br1, HA, H - HA, HB)[None, :]

    for blk in range(tl.program_id(0), n_blocks, tl.num_programs(0)):
        m = blk * BM + tl.arange(0, BM)
        valid = m < n_items
        row = (m // NU).to(tl.int64)
        unit = m % NU
        base = row * TOTAL + UOFF + unit * UW
        pm = m.to(tl.int64) * H
        g_pool_a = _rows(pool_ptr, pm, 0, HA, valid, HA)
        g_pool_b = _rows(pool_ptr, pm, HA, H - HA, valid, HB)
        pool_a = tl.zeros([BM, HA], dtype=bphi_a.dtype)
        pool_b = tl.zeros([BM, HB], dtype=bphi_a.dtype)
        gwp_a = tl.zeros([HA, DP], dtype=bphi_a.dtype)
        gwp_b = tl.zeros([HB, DP], dtype=bphi_a.dtype)
        gbp_a = tl.zeros([HA], dtype=bphi_a.dtype)
        gbp_b = tl.zeros([HB], dtype=bphi_a.dtype)
        for i in range(MEMBERS):
            leg = _rows(x_ptr, base, i * D, D, valid, DP)
            pre_a = _lin(leg, wphi, D, 0, 0, D, H, DP, HA, PREC) + bphi_a
            pre_b = _lin(leg, wphi, D, 0, HA, D, H, DP, HB, PREC) + bphi_b
            pool_a += tl.maximum(pre_a, 0.0)
            pool_b += tl.maximum(pre_b, 0.0)
            gphi_a = tl.where(pre_a > 0, g_pool_a, 0.0)
            gphi_b = tl.where(pre_b > 0, g_pool_b, 0.0)
            gwp_a += _outer(gphi_a, leg, PREC)
            gwp_b += _outer(gphi_b, leg, PREC)
            gbp_a += tl.sum(gphi_a, 0)
            gbp_b += tl.sum(gphi_b, 0)
        _acc(part, D, 0, 0, H, D, gwp_a, HA, DP)
        _acc(part, D, HA, 0, H, D, gwp_b, HB, DP)
        _acc_vec(part + P_BPHI, 0, H, gbp_a, HA)
        _acc_vec(part + P_BPHI, HA, H - HA, gbp_b, HB)

        zp_a = _lin(pool_a, wr1, D + H, D, 0, D + H, H, HA, HA, PREC) \
            + _lin(pool_b, wr1, D + H, D + HA, 0, D + H, H, HB, HA, PREC) \
            + br1_a
        zp_b = _lin(pool_a, wr1, D + H, D, HA, D + H, H, HA, HB, PREC) \
            + _lin(pool_b, wr1, D + H, D + HA, HA, D + H, H, HB, HB, PREC) \
            + br1_b
        gs_a = tl.zeros([BM, HA], dtype=pool_a.dtype)
        gs_b = tl.zeros([BM, HB], dtype=pool_a.dtype)
        gw1_a = tl.zeros([HA, DP], dtype=pool_a.dtype)
        gw1_b = tl.zeros([HB, DP], dtype=pool_a.dtype)
        gw2_a = tl.zeros([DP, HA], dtype=pool_a.dtype)
        gw2_b = tl.zeros([DP, HB], dtype=pool_a.dtype)
        gb2 = tl.zeros([DP], dtype=pool_a.dtype)
        for i in range(MEMBERS):
            leg = _rows(x_ptr, base, i * D, D, valid, DP)
            pre_a = zp_a + _lin(leg, wr1, D + H, 0, 0, D, H, DP, HA, PREC)
            pre_b = zp_b + _lin(leg, wr1, D + H, 0, HA, D, H, DP, HB, PREC)
            dest = row * TOTAL + tl.load(
                pinv_ptr + UOFF + unit * UW + i * D, mask=valid, other=0)
            g_out = _rows(g_ptr, dest, 0, D, valid, DP)
            gw2_a += _outer(g_out, tl.maximum(pre_a, 0.0), PREC)
            gw2_b += _outer(g_out, tl.maximum(pre_b, 0.0), PREC)
            gb2 += tl.sum(g_out, 0)
            gr_a = tl.where(
                pre_a > 0, _back(g_out, wr2, H, 0, 0, D, H, DP, HA, PREC), 0.0)
            gr_b = tl.where(
                pre_b > 0, _back(g_out, wr2, H, 0, HA, D, H, DP, HB, PREC),
                0.0)
            gs_a += gr_a
            gs_b += gr_b
            gw1_a += _outer(gr_a, leg, PREC)
            gw1_b += _outer(gr_b, leg, PREC)
        _acc(part + P_WR2, H, 0, 0, D, H, gw2_a, DP, HA)
        _acc(part + P_WR2, H, 0, HA, D, H, gw2_b, DP, HB)
        _acc_vec(part + P_BR2, 0, D, gb2, DP)
        _acc(part + P_WR1, D + H, 0, 0, H, D, gw1_a, HA, DP)
        _acc(part + P_WR1, D + H, HA, 0, H, D, gw1_b, HB, DP)
        _acc(part + P_WR1, D + H, 0, D, H, D + H, _outer(gs_a, pool_a, PREC),
             HA, HA)
        _acc(part + P_WR1, D + H, 0, D + HA, H, D + H,
             _outer(gs_a, pool_b, PREC), HA, HB)
        _acc(part + P_WR1, D + H, HA, D, H, D + H, _outer(gs_b, pool_a, PREC),
             HB, HA)
        _acc(part + P_WR1, D + H, HA, D + HA, H, D + H,
             _outer(gs_b, pool_b, PREC), HB, HB)
        _acc_vec(part + P_BR1, 0, H, tl.sum(gs_a, 0), HA)
        _acc_vec(part + P_BR1, HA, H - HA, tl.sum(gs_b, 0), HB)


# --- the round -----------------------------------------------------------

def geometry(cmap) -> Geometry | None:
    """
    The :class:`Geometry` of a closed map the kernels serve, ``None`` for
    any other: exactly one :class:`~discopy.neural.Site` group and one
    :class:`~discopy.neural.Relation` group, the site a depth-2 encoder,
    mean pooling, a ``GRUCell`` on one state role and two given roles, a
    ``LayerNorm`` and a broadcast emission, the relation sum-pooled, every
    role at most :data:`TILE` wide once padded to a power of two, and a
    wiring that moves every message leg whole and closes every traced role
    on its own box.

    Parameters:
        cmap : The closed :class:`~discopy.neural.CMap`.
    """
    metas = cmap._fused_routing["metas"]
    if len(metas) != 2:
        return None
    (site, _, cells, cell), (relation, _, units, unit) = metas
    if not isinstance(site, Site) or not isinstance(relation, Relation) \
            or len(site.encode) != 3 or len(relation.phi) != 2 \
            or len(relation.rho) != 3 \
            or site.pooling is not POOL["mean"] \
            or relation.pooling is not POOL["sum"] \
            or not isinstance(site.update, torch.nn.GRUCell) \
            or not isinstance(site.norm, torch.nn.LayerNorm) \
            or site.norm.bias is None or site.update.bias_ih is None \
            or site.emit is None or site.per_leg \
            or site.state_index != (0,) or site.input_index != (1, 2):
        return None
    layout = site.layout(cell)
    d, s, h = site.leg, site.state_width, site.encode[0].out_features
    c, y = (site.widths[role] for role in site.inputs)
    hu = relation.phi[0].out_features
    legs, members = layout.arity, relation.layout(unit).arity
    if min(d, s, c, y) < 1 or min(h, hu) < 16 or relation.leg != d \
            or unit != members * d or layout.traced != (
                slice(legs * d, legs * d + s),
                slice(legs * d + 2 * s, legs * d + 2 * s + c),
                slice(legs * d + 2 * (s + c), legs * d + 2 * (s + c) + y)) \
            or cell != legs * d + 2 * (s + c + y):
        return None
    geo = Geometry(cmap._fused_routing["perm"].numel(), cells, cell, units,
                   unit, cells * cell, legs, members, d, s, h, hu, c, y,
                   tuple(site.echo), site.norm.eps)
    if max(_pads(geo).values()) > TILE \
            or not _routes(cmap._fused_routing["perm"], geo):
        return None
    return geo


def _routes(perm, geo: Geometry) -> bool:
    """
    Whether a round's permutation moves each leg whole between a cell
    and a unit and swaps the two copies of every traced role of a cell.
    """
    box = torch.arange(geo.cells)[:, None] * geo.cell
    legs = (box + torch.arange(geo.legs)[None, :] * geo.d).reshape(-1)
    units = geo.unit_offset + torch.arange(geo.units)[:, None] * geo.unit
    members = (units + torch.arange(geo.members)[None, :] * geo.d).reshape(-1)
    starts = torch.cat([legs, members])
    span = torch.arange(geo.d)
    blocks = perm[starts[:, None] + span]
    if not torch.equal(blocks, perm[starts][:, None] + span):
        return False
    inside = (perm[legs] >= geo.unit_offset).all() \
        and (perm[members] < geo.unit_offset).all()
    cursor, traced = geo.legs * geo.d, []
    for width in (geo.s, geo.c, geo.y):
        first = box + cursor + torch.arange(width)
        traced.append((first, first + width))
        cursor += 2 * width
    return bool(inside) and all(
        torch.equal(perm[first], second) and torch.equal(perm[second], first)
        for first, second in traced)


def parameters(site: Site, relation: Relation) -> tuple:
    """ The tensors of the two modules, in the order the kernels take. """
    return (site.encode[0].weight, site.encode[0].bias,
            site.encode[2].weight, site.encode[2].bias,
            site.update.weight_ih, site.update.weight_hh,
            site.update.bias_ih, site.update.bias_hh,
            site.norm.weight, site.norm.bias,
            site.emit.weight, site.emit.bias,
            relation.phi[0].weight, relation.phi[0].bias,
            relation.rho[0].weight, relation.rho[0].bias,
            relation.rho[2].weight, relation.rho[2].bias)


def precision(tensor) -> str:
    """ The dot precision of a tensor's dtype, as the reference GEMMs use. """
    if tensor.dtype != torch.float32 \
            or torch.get_float32_matmul_precision() == "highest":
        return "ieee"
    return "tf32"


def _grid(n_items: int, kernel: str) -> int:
    """ The programs a kernel needs for that many cells or units. """
    return triton.cdiv(n_items, BLOCK[kernel])


def _programs(n_blocks: int, device) -> int:
    """ The persistent grid of a backward kernel: two programs per SM. """
    return min(n_blocks, 2 * torch.cuda.get_device_properties(
        device).multi_processor_count)


class FusedRound(torch.autograd.Function):
    """
    One round, ``incoming -> incoming[:, perm] (+ init)``, as the two
    forward kernels and, when anything needs a gradient, the two backward
    ones; under ``no_grad`` nothing is saved.
    """
    @staticmethod
    def forward(ctx, incoming, init, pinv, geo, prec, *params):
        rows = incoming.shape[0]
        incoming = incoming.contiguous()
        init = None if init is None else init.contiguous()
        out = torch.empty_like(incoming)
        save = any(ctx.needs_input_grad)
        saved = incoming.new_empty(
            rows * geo.cells, 4 * geo.s + geo.h + 2) if save else incoming
        cell, unit = params[:12], params[12:]
        pads = _pads(geo)
        n_cells, n_units = rows * geo.cells, rows * geo.units
        _cell_fwd[(_grid(n_cells, "cell_fwd"),)](
            incoming, out, incoming if init is None else init, saved, pinv,
            n_cells, geo.eps, *cell,
            TOTAL=geo.total, NC=geo.cells, CW=geo.cell, LEGS=geo.legs,
            D=geo.d, S=geo.s, H=geo.h, C=geo.c, Y=geo.y,
            ECHO_C=geo.echo[0], ECHO_A=geo.echo[1],
            DP=pads["DP"], SP=pads["SP"], HA=pads["HA"], HB=pads["HB"],
            CP=pads["CP"], YP=pads["YP"], BM=BLOCK["cell_fwd"],
            HAS_INIT=init is not None, SAVE=save, PREC=prec, num_warps=WARPS)
        _unit_fwd[(_grid(n_units, "unit_fwd"),)](
            incoming, out, incoming if init is None else init, pinv,
            n_units, *unit,
            TOTAL=geo.total, UOFF=geo.unit_offset, NU=geo.units,
            UW=geo.unit, MEMBERS=geo.members, D=geo.d, H=geo.hu,
            DP=pads["DP"], HA=pads["UA"], HB=pads["UB"], BM=BLOCK["unit_fwd"],
            HAS_INIT=init is not None, PREC=prec, num_warps=WARPS)
        if save:
            ctx.save_for_backward(incoming, saved, pinv, *params)
            ctx.geo, ctx.prec, ctx.has_init = geo, prec, init is not None
        return out

    @staticmethod
    def backward(ctx, grad):
        incoming, saved, pinv, *params = ctx.saved_tensors
        geo, prec, pads = ctx.geo, ctx.prec, _pads(ctx.geo)
        grad = grad.contiguous()
        rows = incoming.shape[0]
        cell, unit = params[:12], params[12:]
        grad_in = torch.empty_like(incoming)
        n_cells, n_units = rows * geo.cells, rows * geo.units
        pool_c = incoming.new_empty(n_cells, geo.h)
        pool_u = incoming.new_empty(n_units, geo.hu)
        gate = incoming.new_empty(
            n_cells, 8 * geo.s + geo.h + geo.c + geo.y + geo.d + 3)
        encode = incoming.new_empty(
            n_cells, geo.legs * (geo.h + geo.d) + 2 * geo.h + 1)
        blocks_u = triton.cdiv(n_units, BLOCK["unit_bwd_weight"])
        progs_u = _programs(blocks_u, incoming.device)
        part_u = incoming.new_zeros(progs_u, sum(p.numel() for p in unit))
        shape_c = dict(
            TOTAL=geo.total, NC=geo.cells, CW=geo.cell, LEGS=geo.legs,
            D=geo.d, S=geo.s, H=geo.h, DP=pads["DP"], SP=pads["SP"],
            HA=pads["HA"], HB=pads["HB"], PREC=prec)
        shape_u = dict(
            TOTAL=geo.total, UOFF=geo.unit_offset, NU=geo.units, UW=geo.unit,
            MEMBERS=geo.members, D=geo.d, H=geo.hu, DP=pads["DP"],
            HA=pads["UA"], HB=pads["UB"], PREC=prec)
        _cell_bwd_gate[(_grid(n_cells, "cell_bwd_gate"),)](
            incoming, grad, grad_in, saved, pool_c, gate, pinv, n_cells,
            cell[4], cell[5], cell[8], cell[9], cell[10],
            C=geo.c, Y=geo.y, ECHO_C=geo.echo[0], ECHO_A=geo.echo[1],
            CP=pads["CP"], YP=pads["YP"], BM=BLOCK["cell_bwd_gate"],
            **shape_c, num_warps=WARPS, num_stages=1)
        _cell_bwd_encode[(_grid(n_cells, "cell_bwd_encode"),)](
            incoming, grad_in, pool_c, encode, n_cells,
            cell[0], cell[1], cell[2], BM=BLOCK["cell_bwd_encode"],
            **shape_c, num_warps=WARPS, num_stages=1)
        _unit_bwd_leg[(_grid(n_units, "unit_bwd_leg"),)](
            incoming, grad, grad_in, pool_u, pinv, n_units, *unit[:5],
            BM=BLOCK["unit_bwd_leg"], **shape_u, num_warps=WARPS,
            num_stages=1)
        _unit_bwd_weight[(progs_u,)](
            incoming, grad, pool_u, pinv, part_u, n_units, blocks_u,
            part_u.shape[1], *unit[:5], BM=BLOCK["unit_bwd_weight"],
            **shape_u, num_warps=WARPS, num_stages=1)
        grads = _cell_grads(pool_c, gate, encode, geo)
        grads += [
            piece.view_as(param) for param, piece in zip(unit, torch.split(
                part_u.sum(0), [p.numel() for p in unit]))]
        return (grad_in, grad if ctx.has_init else None, None, None, None,
                *grads)


def _cell_grads(pool, gate, encode, geo) -> list:
    """
    The weight gradients of the cell as GEMMs over every cell of the
    batch, from what :func:`_cell_bwd_gate` and :func:`_cell_bwd_encode`
    kept: ``G.T @ [x | 1]`` for a linear layer with output gradient ``G``
    on input ``x``, whose last column is the bias gradient.
    """
    s, d, h, x, legs = geo.s, geo.d, geo.h, geo.h + geo.c + geo.y, geo.legs
    k_x, k_s, k_y = 4 * s, 4 * s + x + 1, 4 * s + x + s + 2
    k_b, k_g = k_y + s + 1, k_y + s + 1 + d
    hidden, input_gates = gate[:, :3 * s], gate[:, 3 * s:4 * s]
    ones = gate[:, k_x:k_s]
    state, normed = gate[:, k_s:k_y], gate[:, k_y:k_b]
    belief, g_normed = gate[:, k_b:k_g], gate[:, k_g:].sum(0)
    k_l, k_p = legs * h, legs * (h + d)
    pre = encode[:, :k_l].reshape(-1, h)
    read = encode[:, k_l:k_p].reshape(-1, d)
    first, pooled = encode[:, k_p:k_p + h], encode[:, k_p + h:]
    w1, w2 = first.t() @ state, pool.t() @ pooled / legs
    wih = torch.cat([hidden[:, :2 * s].t() @ ones, input_gates.t() @ ones])
    whh, we = hidden.t() @ state, belief.t() @ normed
    return [torch.cat([w1[:, :s], pre.t() @ read], 1), w1[:, s],
            w2[:, :h], w2[:, h], wih[:, :x], whh[:, :s], wih[:, x], whh[:, s],
            g_normed[:s], g_normed[s:], we[:, :s], we[:, s]]


def fused_step(incoming, init, params: tuple, geo: Geometry, perm_inverse,
               backend: str = "triton"):
    """
    One fused round: the next round's incoming messages, in the box-order
    layout of :attr:`~discopy.neural.CMap._fused_routing`.

    Parameters:
        incoming : The incoming messages, ``(rows, total)``.
        init : The injected messages to add after routing, or ``None``.
        params : The tensors of :func:`parameters`.
        geo : The :func:`geometry` of the map.
        perm_inverse : The inverse of the round's permutation, on device.
        backend : ``"triton"`` for the kernels here, ``"cuda"`` for those
                  of :mod:`discopy.neural.fused_cuda`.
    """
    if backend == "cuda":
        from discopy.neural import fused_cuda
        return fused_cuda.fused_step(incoming, init, params, geo,
                                     perm_inverse)
    if backend != "triton":
        raise ValueError(f"unknown fused backend {backend!r}")
    return FusedRound.apply(incoming, init, perm_inverse, geo,
                            precision(incoming), *params)


class Outputs:
    """
    The per-group box outputs of a fused round, ``(rows, n_boxes, width)``
    each, read off the routed state only if a caller iterates them: the
    kernels write the next round's incoming messages directly, so the box
    outputs are their inverse routing, which the flat-state path never
    needs.

    Parameters:
        incoming : The routed messages the round returned.
        init : What was injected after routing, or ``None``.
        routing : The device routing of the map.
    """
    def __init__(self, incoming, init, routing: dict):
        self.incoming, self.init, self.routing = incoming, init, routing

    def __len__(self):
        return len(self.routing["metas"])

    def __iter__(self):
        out = self.incoming if self.init is None \
            else self.incoming - self.init
        out = _perm_gather(
            out, self.routing["perm_inverse"], self.routing["perm"])
        offset = 0
        for _, _, n_boxes, width in self.routing["metas"]:
            block = n_boxes * width
            yield out[:, offset:offset + block].reshape(-1, n_boxes, width)
            offset += block


def step_of(cmap, routing: dict, backend: str = "triton"):
    """
    The fused round step of a closed map, ``(incoming, source, init) ->
    (incoming, outputs)`` as :meth:`~discopy.neural.CMap._step_body`
    returns it, or ``None`` when the map has no :func:`geometry`.

    Parameters:
        cmap : The closed :class:`~discopy.neural.CMap`.
        routing : Its device routing.
        backend : The kernels of :func:`fused_step`.
    """
    geo = geometry(cmap)
    if geo is None:
        return None
    if backend == "triton" and tl is None:
        raise ImportError("the fused round needs triton")
    site, relation = routing["metas"][0][0], routing["metas"][1][0]
    perm_inverse = routing["perm_inverse"]

    def step(incoming, source, init):
        incoming = fused_step(incoming, init, parameters(site, relation),
                              geo, perm_inverse, backend)
        return incoming, Outputs(incoming, init, routing)
    return step
