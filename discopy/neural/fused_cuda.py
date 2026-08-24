# -*- coding: utf-8 -*-

"""
The fused round of :mod:`discopy.neural.fused` on hand-written CUDA
kernels, compiled at runtime.

The Triton kernels of :mod:`discopy.neural.fused` are register-bound: on
Hopper a Triton dot is a 64-row warpgroup tile whose float32 accumulators
alone cost 32 registers per thread, and it wants powers of two, so a
hidden width of 68 pads to 128.  The kernels of ``cuda/round.cu`` run the
same round on ``mma.sync.m16n8k8`` tiles instead -- 16 rows per warp, four
accumulator registers per tile, widths padded to multiples of 8 -- with
the weights staged once per block in shared memory, a persistent grid of
two blocks per SM looping over the (rows, cell) items, and the
activations of a cell never leaving registers.  They are compiled here
with the NVRTC that ships with torch, once per geometry, device and
precision, the cubin cached on disk under ``~/.cache/discopy`` (or
``$DISCOPY_CACHE``) keyed on the source and the geometry, and launched
through the driver API on torch's current stream, so the round captures
into a CUDA graph like any other kernel.

Every kernel is templated on a precision: ``"tf32"`` under
``torch.set_float32_matmul_precision("high")`` or ``"medium"`` runs the
tensor cores as cuBLAS does, ``"fp32"`` under ``"highest"`` and
``"fp64"`` on double tensors run plain fused multiply-adds through the
same fragments, slowly, as the proof of the layouts.  The buffers the
cell's backward keeps are those of the Triton round, so its weight
gradients are the GEMMs of :func:`~discopy.neural.fused._cell_grads`;
the unit's backward keeps the same kind of buffer and :func:`_unit_grads`
does the same for it.

Summary
-------

.. autosummary::
    :template: function.rst
    :nosignatures:

    precision
    kernels
    fused_step
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from importlib import resources
from pathlib import Path

import torch

try:
    from cuda.bindings import driver, nvrtc
except ImportError:  # pragma: no cover
    driver = nvrtc = None

from discopy.neural.fused import Geometry, _cell_grads

#: Threads per block, a warp per 16-row tile of the (rows, cell) items.
THREADS, WARPS = 256, 8

#: The kernels of ``round.cu`` and the precisions they are instantiated at.
KERNELS = ("cell_fwd", "unit_fwd", "cell_bwd_gate", "cell_bwd_encode",
           "unit_bwd")
PRECISION = {"tf32": 0, "fp32": 1, "fp64": 2}

CACHE = Path(os.environ.get(
    "DISCOPY_CACHE", Path.home() / ".cache" / "discopy")) / "nvrtc"


def precision(tensor) -> str:
    """ The precision the kernels run a tensor at. """
    if tensor.dtype == torch.float64:
        return "fp64"
    if torch.get_float32_matmul_precision() == "highest":
        return "fp32"
    return "tf32"


def source() -> str:
    """ The CUDA source of the kernels. """
    return resources.files("discopy.neural").joinpath(
        "cuda/round.cu").read_text()


def options(geo: Geometry, arch: str) -> tuple:
    """ The NVRTC options compiling the source for a geometry. """
    macros = dict(
        D=geo.d, S=geo.s, H=geo.h, HU=geo.hu, C=geo.c, Y=geo.y,
        LEGS=geo.legs, MEMBERS=geo.members, TOTAL=geo.total, NC=geo.cells,
        CW=geo.cell, NU=geo.units, UW=geo.unit, UOFF=geo.unit_offset,
        ECHO_C=int(geo.echo[0]), ECHO_A=int(geo.echo[1]))
    return (f"--gpu-architecture={arch}", "-std=c++17",
            *(f"-D{key}={value}" for key, value in macros.items()))


def _check(result):
    """ The values of a driver or NVRTC call, its status checked. """
    status, *values = result if isinstance(result, tuple) else (result,)
    if int(status):
        raise RuntimeError(f"{status}")
    return values[0] if len(values) == 1 else values


def cubin(src: str, opts: tuple, names: tuple) -> tuple[bytes, dict, bool]:
    """
    The cubin of a source and the lowered name of each kernel, compiled or
    read from the cache, and whether the cache had it.
    """
    err, major, minor = nvrtc.nvrtcVersion()
    key = hashlib.sha256("\0".join(
        [src, *opts, *names, f"{major}.{minor}"]).encode()).hexdigest()
    path, index = CACHE / f"{key}.cubin", CACHE / f"{key}.json"
    if path.exists() and index.exists():
        return path.read_bytes(), json.loads(index.read_text()), True
    prog = _check(nvrtc.nvrtcCreateProgram(
        src.encode(), b"round.cu", 0, [], []))
    for name in names:
        _check(nvrtc.nvrtcAddNameExpression(prog, name.encode()))
    status, = nvrtc.nvrtcCompileProgram(
        prog, len(opts), [opt.encode() for opt in opts])
    log = bytes(_check(nvrtc.nvrtcGetProgramLogSize(prog)))
    nvrtc.nvrtcGetProgramLog(prog, log)
    if int(status):
        raise RuntimeError(log.decode())
    lowered = {
        name: _check(nvrtc.nvrtcGetLoweredName(prog, name.encode())).decode()
        for name in names}
    image = bytes(_check(nvrtc.nvrtcGetCUBINSize(prog)))
    _check(nvrtc.nvrtcGetCUBIN(prog, image))
    nvrtc.nvrtcDestroyProgram(prog)
    CACHE.mkdir(parents=True, exist_ok=True)
    for target, data in ((path, image), (index, json.dumps(lowered))):
        temp = target.with_suffix(f".{os.getpid()}.tmp")
        temp.write_bytes(data if isinstance(data, bytes) else data.encode())
        os.replace(temp, target)
    return image, lowered, False


def tiles(width: int) -> int:
    """ The 8-wide tiles a width takes. """
    return -(-width // 8)


def pad(width: int) -> int:
    """ A width padded to a multiple of 8 with room for the bias column. """
    return 8 * tiles(width + 1)


def stride(width: int) -> int:
    """ The row stride of a staged weight matrix, bank-conflict free. """
    return width if width % 32 in (8, 24) else width + 8


def saved_width(geo: Geometry) -> int:
    """ The row width of the forward's save buffer, ``round.cu``'s ``SV``. """
    return 4 * pad(geo.s) + pad(geo.h) + 8


def unit_width(geo: Geometry) -> int:
    """ The row width of the unit backward's keep buffer, ``KW_UNIT``. """
    m, d, h = geo.members, geo.d, geo.hu
    return m * h + m * (d + 1) + m * h + m * (h + 1) + m * d + 2 * h


def items(rows: int, boxes: int) -> int:
    """ The 16-row tiles of that many rows of that many cells or units. """
    return -(-rows // 16) * boxes


def grid(items: int, device) -> int:
    """ The persistent grid over that many items: two blocks per SM. """
    return min(-(-items // WARPS), 2 * torch.cuda.get_device_properties(
        device).multi_processor_count)


def smem(geo: Geometry, kernel: str) -> int:
    """
    The shared-memory elements a kernel stages, as ``round.cu`` lays them
    out: its weights, and for the backward kernels a scratch tile per warp.
    """
    d, s, h, u, x = (
        pad(w) for w in (geo.d, geo.s, geo.h, geo.hu, geo.c + geo.y))
    scratch = WARPS * 32 * 4
    if kernel == "cell_fwd":
        return h * stride(s + d) + h * stride(h) + 3 * s * stride(h + x) \
            + 3 * s * stride(s) + d * stride(s)
    if kernel == "unit_fwd":
        return u * stride(d) + u * stride(d + u) + d * stride(u)
    if kernel == "cell_bwd_gate":
        return s * stride(3 * s) + (h + x) * stride(3 * s) + s * stride(d)
    if kernel == "cell_bwd_encode":
        return h * stride(s + d) + (s + d) * stride(h) + h * stride(h) \
            + (h // 8) * scratch
    if kernel == "unit_bwd":
        return u * stride(d) + u * stride(d + u) + u * stride(d) \
            + (d + u) * stride(u) + d * stride(u) + (u // 8) * scratch
    raise KeyError(kernel)


ATTRIBUTES = dict(regs="CU_FUNC_ATTRIBUTE_NUM_REGS",
                  local="CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES",
                  shared="CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES")


class Kernels:
    """
    The kernels of one geometry on one device: a module per precision,
    built the first time that precision is launched, on torch's current
    stream.

    Parameters:
        geo : The geometry the source is compiled for.
        device : The device the modules are loaded on.
    """
    def __init__(self, geo: Geometry, device: torch.device):
        if driver is None:
            raise ImportError("the CUDA round needs cuda-bindings and NVRTC")
        self.geo, self.device = geo, torch.device(device)
        self.index = torch.cuda.current_device() \
            if self.device.index is None else self.device.index
        major, minor = torch.cuda.get_device_capability(self.index)
        self.arch = f"sm_{major}{minor}"
        self.modules, self.functions, self.cached = {}, {}, {}
        self.configured = set()

    def build(self, prec: str):
        """ Compile or read from the cache, and load, one precision. """
        names = tuple(f"{kernel}<{PRECISION[prec]}>" for kernel in KERNELS)
        image, lowered, self.cached[prec] = cubin(
            source(), options(self.geo, self.arch), names)
        with torch.cuda.device(self.index):
            torch.empty(0, device=self.index)
            self.modules[prec] = _check(driver.cuModuleLoadData(image))
            for kernel, name in zip(KERNELS, names):
                self.functions[kernel, prec] = _check(
                    driver.cuModuleGetFunction(
                        self.modules[prec], lowered[name].encode()))

    def function(self, kernel: str, prec: str):
        """ The handle of a kernel at a precision. """
        if prec not in self.modules:
            self.build(prec)
        return self.functions[kernel, prec]

    def attributes(self, kernel: str, prec: str) -> dict:
        """ Registers, local memory bytes and static shared memory bytes. """
        function = self.function(kernel, prec)
        return {label: _check(driver.cuFuncGetAttribute(
            getattr(driver.CUfunction_attribute, name), function))
            for label, name in ATTRIBUTES.items()}

    def bytes(self, kernel: str, prec: str) -> int:
        """ The dynamic shared memory of a launch. """
        return smem(self.geo, kernel) * (8 if prec == "fp64" else 4)

    def launch(self, kernel: str, prec: str, items: int, *args):
        """
        Launch a kernel on torch's current stream, on the persistent grid
        over ``items``: tensors pass as pointers, booleans and integers as
        ``int``, floats as ``double``.
        """
        function, size = self.function(kernel, prec), self.bytes(kernel, prec)
        if size > 48 * 1024 and (kernel, prec) not in self.configured:
            _check(driver.cuFuncSetAttribute(
                function, driver.CUfunction_attribute
                .CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, size))
            self.configured.add((kernel, prec))
        values, types = [], []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                values.append(arg.data_ptr())
                types.append(ctypes.c_void_p)
            elif isinstance(arg, float):
                values.append(arg)
                types.append(ctypes.c_double)
            else:
                values.append(int(arg))
                types.append(ctypes.c_int)
        stream = torch.cuda.current_stream(self.index).cuda_stream
        _check(driver.cuLaunchKernel(
            function, grid(items, self.index), 1, 1, THREADS, 1, 1, size,
            driver.CUstream(stream), (tuple(values), tuple(types)), 0))


_KERNELS: dict = {}


def kernels(geo: Geometry, device) -> Kernels:
    """ The :class:`Kernels` of a geometry on a device, built once. """
    device = torch.device(device)
    index = torch.cuda.current_device() if device.index is None \
        else device.index
    if (geo, index) not in _KERNELS:
        _KERNELS[geo, index] = Kernels(geo, torch.device("cuda", index))
    return _KERNELS[geo, index]


class FusedRound(torch.autograd.Function):
    """
    :class:`discopy.neural.fused.FusedRound` on the CUDA kernels: the
    same signature, the same buffers kept for the cell's weight gradients.
    """
    @staticmethod
    def forward(ctx, incoming, init, pinv, geo, prec, *params):
        rows = incoming.shape[0]
        incoming = incoming.contiguous()
        init = None if init is None else init.contiguous()
        out = torch.empty_like(incoming)
        save = any(ctx.needs_input_grad)
        saved = incoming.new_empty(
            rows * geo.cells, saved_width(geo)) if save else incoming
        params = tuple(param.contiguous() for param in params)
        cell, unit = params[:12], params[12:]
        run = kernels(geo, incoming.device)
        injected = incoming if init is None else init
        run.launch("cell_fwd", prec, items(rows, geo.cells), incoming, out,
                   injected, saved, pinv, rows, float(geo.eps),
                   init is not None, save, *cell)
        run.launch("unit_fwd", prec, items(rows, geo.units), incoming, out,
                   injected, pinv, rows, init is not None, *unit)
        if save:
            ctx.save_for_backward(incoming, saved, pinv, *params)
            ctx.geo, ctx.prec, ctx.has_init = geo, prec, init is not None
        return out

    @staticmethod
    def backward(ctx, grad):
        incoming, saved, pinv, *params = ctx.saved_tensors
        geo, prec = ctx.geo, ctx.prec
        grad = grad.contiguous()
        rows = incoming.shape[0]
        cell, unit = params[:12], params[12:]
        grad_in = torch.empty_like(incoming)
        n_cells, n_units = rows * geo.cells, rows * geo.units
        pool = incoming.new_empty(n_cells, geo.h)
        gate = incoming.new_empty(
            n_cells, 8 * geo.s + geo.h + geo.c + geo.y + geo.d + 3)
        encode = incoming.new_empty(
            n_cells, geo.legs * (geo.h + geo.d) + 2 * geo.h + 1)
        keep = incoming.new_empty(n_units, unit_width(geo))
        run = kernels(geo, incoming.device)
        cells, units = items(rows, geo.cells), items(rows, geo.units)
        run.launch("cell_bwd_gate", prec, cells, incoming, grad, grad_in,
                   saved, pool, gate, pinv, rows,
                   cell[4], cell[5], cell[8], cell[9], cell[10])
        run.launch("cell_bwd_encode", prec, cells, incoming, grad_in, pool,
                   encode, rows, cell[0], cell[1], cell[2])
        run.launch("unit_bwd", prec, units, incoming, grad, grad_in, keep,
                   pinv, rows, *unit[:5])
        grads = _cell_grads(pool, gate, encode, geo) + _unit_grads(keep, geo)
        return (grad_in, grad if ctx.has_init else None, None, None, None,
                *grads)


def _unit_grads(keep, geo: Geometry) -> list:
    """
    The weight gradients of the unit as GEMMs over every unit of the
    batch on what ``unit_bwd`` kept, ``G.T @ [x | 1]`` per linear layer
    like :func:`~discopy.neural.fused._cell_grads`.
    """
    m, d, h = geo.members, geo.d, geo.hu
    u_l, u_r = m * h, m * h + m * (d + 1)
    u_h, u_o = u_r + m * h, u_r + m * h + m * (h + 1)
    u_p = u_o + m * d
    gphi = keep[:, :u_l].reshape(-1, h)
    legs = keep[:, u_l:u_r].reshape(-1, d + 1)
    g_rho = keep[:, u_r:u_h].reshape(-1, h)
    rho = keep[:, u_h:u_o].reshape(-1, h + 1)
    g_out = keep[:, u_o:u_p].reshape(-1, d)
    pooled, g_sum = keep[:, u_p:u_p + h], keep[:, u_p + h:]
    wphi, wr1, wr2 = gphi.t() @ legs, g_rho.t() @ legs, g_out.t() @ rho
    return [wphi[:, :d], wphi[:, d],
            torch.cat([wr1[:, :d], g_sum.t() @ pooled], 1), wr1[:, d],
            wr2[:, :h], wr2[:, h]]


def fused_step(incoming, init, params: tuple, geo: Geometry, perm_inverse):
    """
    One round on the CUDA kernels, see
    :func:`discopy.neural.fused.fused_step`.
    """
    return FusedRound.apply(incoming, init, perm_inverse, geo,
                            precision(incoming), *params)
