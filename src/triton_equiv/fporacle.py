"""Floating-point tier of the oracle: labels mutants that are exact over the reals but round differently.

Labels (most severe first)
--------------------------
``oob``, ``inequivalent``  from the bit-vector tier (``oracle``), unchanged.
``numerical``     **benign over the reals but numerically different**. Z3 proves the mutant
                  computes the same real-valued function as the reference at the oracle sizes,
                  *and* a concrete input (the witness) makes the two kernels' rounded outputs
                  differ bit for bit. ``precision_loss`` says whether the mutant's running error
                  bound (max over the inputs tried) is more than ``PRECISION_LOSS_RATIO`` times the
                  reference's (accumulator downgrade, a narrower cast) or about the same (a
                  legal-looking reordering).
``benign``        real-equivalent and no witness. ``fp_proof="uf"`` means bitwise identity is
                  *proved* (see check 2). ``fp_proof=None`` means only that the witness search
                  found nothing: that is evidence, not a proof.

Approach chosen: fp64 reference plus bounded error analysis (not Z3 floating point)
-----------------------------------------------------------------------------------
Three checks, run on every mutant the bit-vector tier calls benign:

1. **Real equivalence (Z3, proof at each size).** The kernel is traced with values as Z3
   ``Real`` terms: ``+``/``*`` are exact and every cast is the identity. After substituting each
   program id, the output terms are polynomials in the input cells. Z3 decides ``ref != mut``.
2. **Bitwise identity (Z3, proof at each size).** The kernel is traced again with every rounded
   operation as an *uninterpreted function* keyed by its dtype and shape: ``fadd_float32``,
   ``fsum_float16_8``, ``fdot_float32_4``, ``fround_bfloat16``, and so on. Casts that widen exactly
   (fp16 -> fp32, for example) are the identity, and ``tl.store`` rounds to the buffer dtype.
   If the reference and mutant output terms are syntactically identical, the outputs are
   bitwise equal under *any* semantics in which each primitive is a deterministic function of
   its operands. This holds in the Triton interpreter, and on a GPU if the same shape gets the
   same reduction tree. The check is purely structural: no commutativity, no ``x*0 = 0``.
   When it fails on a kernel with ``tl.dot``, a second pass (``fp_proof="uf-lane"``) adds the
   lane-product assumption and the ``-inf`` padding rules of ``_LaneFolder``, which also lean
   on the no-overflow scope for fp32/fp64 sums and products.
3. **Witness search and error bounds (numpy).** A concrete emulator runs the kernel source
   with the *same numpy calls as the Triton interpreter* (``tl.sum`` is ``np.sum``, ``tl.dot`` is
   ``np.matmul(a, b, dtype=acc.dtype) + acc``, fp32 -> bf16 truncates). It is run in the declared
   dtypes for reference and mutant, and all in fp64 as the reference value. Inputs come from
   seeded ``randn``, ``rand`` and a wide-exponent "mixed" generator; the first input whose
   outputs differ bitwise is the witness. Alongside every value the emulator carries a
   **running error bound** (Higham, *Accuracy and Stability*, ch. 3-4): ``u|r|`` per rounded
   elementwise op, ``gamma_{n-1} sum|x|`` for an n-term ``tl.sum`` and ``gamma_K sum|a||b|`` for a
   K-term ``tl.dot``, with ``gamma_k = k u / (1 - k u)``. Those reduction bounds hold for *any*
   summation order and with or without FMA, so they do not depend on the interpreter's order.
   ``u`` is 2^-11 for fp16, 2^-24 for fp32, and 2^-7 for bf16 (the interpreter truncates, so we
   take the truncation bound, which also covers round-to-nearest).

exp, sqrt, max, division and ``tl.where`` (softmax, layernorm, attention)
------------------------------------------------------------------------
* Real tier: ``/`` is Z3 real division, ``max`` is ``If(a >= b, a, b)``, ``tl.where`` is ``If``,
  and ``exp``/``sqrt`` are **uninterpreted** (Z3 reals have no exp). ``-inf`` and other
  non-finite constants are named uninterpreted reals. "Equal" is still a proof (congruence);
  ``sat`` or ``unknown`` is not a real counterexample, so for these kernels it is confirmed by
  the fp64 emulator before a mutant is called ``inequivalent`` (``confirm_fp64``). An unconfirmed
  real difference is recorded in ``notes`` and the mutant goes on to the witness search with
  ``real_proof=False``.
* Uninterpreted-function tier: ``fdiv_<dt>``, ``fmax_<dt>_<n>``, ``fexp_<dt>``, ``fsqrt_<dt>``.
* Emulator and error bounds: ``max`` is 1-Lipschitz, so its bound is the max of the operand
  bounds; ``exp(x)`` gets ``|r| (expm1(e) + 2u) / (1 - 2u)`` (numpy's exp is not correctly rounded;
  2u allows one ulp); ``sqrt`` gets ``e / sqrt(x) + u|r|``; ``a / b`` gets
  ``(|a| e_b + |b| e_a) / (|b| (|b| - e_b)) + u|r|`` (infinite if ``|b| <= e_b``). An infinite
  result of an operation on an infinite operand (``-inf`` padding) is carried with bound 0.
* Dtypes follow the interpreter, checked by a probe: a 0-d tensor or scalar operand takes the
  tensor operand's dtype (fp16 tile + fp32 0-d mean stays fp16), and ``/`` on fp16/bf16
  returns fp32.

Why not Z3 floating-point theory at a small format: (a) bit-blasted fp add/mul makes even a
4x4x4 fp16 matmul slow, and the audit needs reduction sizes in the thousands; (b) a proof or
witness in a toy format (say 4-bit exponent, 4-bit mantissa) says nothing about fp32 at
n = 4096, which is where tolerance checks start to matter; (c) its witnesses could not be
replayed in the Triton interpreter, while ours can (``tests/test_interpreter.py`` does).
The cost is that a ``benign`` label without the structural proof is not a proof.

What this proves, and what it does not
--------------------------------------
Proves (at the listed sizes only): real equivalence (check 1); bitwise identity for
``fp_proof="uf"`` (check 2), relative to the deterministic-primitive assumption above; and, per
evaluated input, a rigorous bound on |computed - exact| (up to evaluating the bound itself in
fp64).
Does not prove: that a ``benign`` label with ``fp_proof=None`` has no fp divergence; that a
witness reproduces bitwise on a GPU (other reduction trees, FMA, round-to-nearest bf16); that a
tolerance check passes or fails on inputs we did not try. It also does not model overflow,
NaN/inf or subnormals: generators keep values in range, and a non-finite output is reported
rather than bounded. Symbolic sizes are out of scope here too.

Run ``python -m triton_equiv.fporacle --out results/labels.json`` to label every mutant.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import z3

from . import oracle
from .families import family
from .kernels import KERNELS, KernelSpec, Launch
from .oracle import IDX_BITS, Ptr, T, Trace, Unsupported, _bcast, _obj

FP_SEVERITY = {"benign": 0, "numerical": 1, "inequivalent": 2, "oob": 3}

# Unit roundoff per dtype. bf16 uses the truncation bound (the interpreter truncates fp32 -> bf16).
UNIT_ROUNDOFF = {"float16": 2.0**-11, "bfloat16": 2.0**-7, "float32": 2.0**-24, "float64": 2.0**-53}
_WIDTH = {"float16": 16, "bfloat16": 16, "float32": 32, "float64": 64}
_EXACT_WIDEN = {("float16", "float32"), ("float16", "float64"), ("bfloat16", "float32"),
                ("bfloat16", "float64"), ("float32", "float64")}
PRECISION_LOSS_RATIO = 4.0  # mutant bound / reference bound above this -> precision_loss


def promote(a: str, b: str) -> str:
    """Triton's fp promotion: same -> same; fp16 with bf16 -> fp32; otherwise the wider one."""
    if a == b:
        return a
    if {a, b} == {"float16", "bfloat16"}:
        return "float32"
    return a if _WIDTH[a] > _WIDTH[b] else b


def result_dtype(a_dt: str, a_scalar: bool, b_dt: str, b_scalar: bool, op: str) -> str:
    """Interpreter dtype of a binary fp op (checked with a probe under TRITON_INTERPRET=1).

    A 0-d operand (scalar argument, Python constant or reduced tile) next to a tile does not
    take part in promotion; ``/`` on fp16 or bf16 yields fp32.
    """
    if a_scalar != b_scalar:
        dt = b_dt if a_scalar else a_dt
    else:
        dt = promote(a_dt, b_dt)
    if op == "div" and dt in ("float16", "bfloat16"):
        dt = "float32"
    return dt


def gamma(k: int, u: float) -> float:
    return math.inf if k * u >= 1 else k * u / (1 - k * u)


class _DT:
    float16, bfloat16, float32, float64 = "float16", "bfloat16", "float32", "float64"


def _dtype_name(d) -> str:
    if isinstance(d, str) and d in _WIDTH:
        return d
    raise Unsupported(f"dtype {d!r}")


# ===================================================================== symbolic (real | uf)
def _const_real(c):
    if math.isfinite(c):
        return z3.RealVal(c)
    return z3.Real(f"fconst_{c!r}")  # -inf etc.: a named uninterpreted constant


class _RealDom:
    """Exact real arithmetic: casts are the identity; exp and sqrt are uninterpreted."""

    name = "real"

    def __init__(self):
        self._fns: dict = {}
        self.abstract = False

    def const(self, c):
        if not math.isfinite(c):
            self.abstract = True
        return _const_real(c)

    def _f(self, name):
        self.abstract = True
        if name not in self._fns:
            self._fns[name] = z3.Function(name, z3.RealSort(), z3.RealSort())
        return self._fns[name]

    def div(self, a, b, dt): return a / b

    def exp(self, a, dt): return self._f("exp")(a)

    def sqrt(self, a, dt): return self._f("sqrt")(a)

    def max(self, xs, dt):
        acc = xs[0]
        for v in xs[1:]:
            acc = z3.If(acc >= v, acc, v)
        return acc

    def cvt(self, x, src, dst):
        return x

    def add(self, a, b, dt): return a + b
    def sub(self, a, b, dt): return a - b
    def mul(self, a, b, dt): return a * b

    def sum(self, xs, dt):
        return z3.Sum(list(xs)) if len(xs) > 1 else xs[0]

    def dot(self, row, col, dt):
        return z3.Sum([a * b for a, b in zip(row, col)])


class _UFDom:
    """Every rounded op is an uninterpreted function keyed by dtype (and length for reductions)."""

    name = "uf"

    def __init__(self, lanes: bool = False):
        self._fns: dict = {}
        self.lanes = lanes

    def _f(self, name, arity):
        key = (name, arity)
        if key not in self._fns:
            self._fns[key] = z3.Function(name, *([z3.RealSort()] * (arity + 1)))
        return self._fns[key]

    def const(self, c):
        if c == 0 and math.copysign(1.0, c) < 0:
            return z3.Real(_NEG_ZERO)  # bitwise identity must tell -0 from the numeral 0 (= +0)
        return _const_real(c)

    def div(self, a, b, dt): return self._f(f"fdiv_{dt}", 2)(a, b)
    def exp(self, a, dt): return self._f(f"fexp_{dt}", 1)(a)
    def sqrt(self, a, dt): return self._f(f"fsqrt_{dt}", 1)(a)

    def max(self, xs, dt):
        return xs[0] if len(xs) == 1 else self._f(f"fmax_{dt}_{len(xs)}", len(xs))(*xs)

    def cvt(self, x, src, dst):
        if src == dst or (src, dst) in _EXACT_WIDEN:
            return x
        return self._f(f"fround_{dst}", 1)(x)

    def add(self, a, b, dt): return self._f(f"fadd_{dt}", 2)(a, b)
    def sub(self, a, b, dt): return self._f(f"fsub_{dt}", 2)(a, b)
    def mul(self, a, b, dt): return self._f(f"fmul_{dt}", 2)(a, b)

    def sum(self, xs, dt):
        return xs[0] if len(xs) == 1 else self._f(f"fsum_{dt}_{len(xs)}", len(xs))(*xs)

    def dot(self, row, col, dt):
        k = len(row)
        if self.lanes:
            prods = [self.lane_product(a, b, dt) for a, b in zip(row, col)]
            return self._f(f"fdotp_{dt}_{k}", k)(*prods)
        return self._f(f"fdot_{dt}_{k}", 2 * k)(*row, *col)

    def lane_product(self, a, b, dt):
        return self._f(f"{_LANE_PROD}{dt}", 2)(a, b)


# Lane-product model (fp_proof="uf-lane"). Assumption, stated in the README: ``tl.dot`` is a
# deterministic function of the sequence of exact lane products a_i*b_i, taken after its inputs
# are rounded to the dot's input precision (fp16, bf16 or tf32). FMA chains and exact-product
# tensor-core accumulation both have this form. Under it, a lane product of two constants can be
# replaced by its value when both are exact at every such input precision and the product is not
# -0 (a Z3 numeral cannot carry the sign of zero).
_LANE_PROD = "flaneprod_"


def _narrow_exact(c: float) -> bool:
    """True if ``c`` is exactly representable in fp16 and bf16 (hence also in tf32)."""
    if c == 0:
        return True
    if not math.isfinite(c) or not (2.0 ** -14 <= abs(c) <= 65504.0):
        return False
    m, _ = math.frexp(abs(c))
    return (m * 2 ** 8).is_integer()


def _numeral(t):
    if z3.is_rational_value(t):
        return float(t.as_fraction())
    return None


# -inf padding rules, applied in the same pass. Two IEEE facts: exp(-inf) = +0, and
# -inf - x = -inf for every finite x. In the uf domain -0.0 is its own named constant, so the
# numeral 0 always means +0 there. Finiteness is decided conservatively and leans on the stated
# scope (finite inputs, no overflow), restricted to fp32/fp64 where overflow needs values beyond
# 3.4e38: loads are finite; a numeral is finite if it is within fp32 range; an fp32/fp64 add, sub,
# mul, sum, dot or lane product is finite if its operands are; an fp32/fp64 max is finite if one
# operand is finite and the rest are finite or -inf. Everything else (exp, sqrt, division,
# rounding, fp16 and bf16 ops, other constants) is treated as possibly non-finite.
_NEG_INF = repr(-math.inf)
_NEG_ZERO = "fconst_-0.0"
_FP32_MAX = float(np.finfo(np.float32).max)
_TOTAL_OPS = ("fadd_", "fsub_", "fmul_", "fsum_", "fdot_", "fdotp_", _LANE_PROD)
_WIDE = ("float32", "float64")


def _is_neg_inf(t) -> bool:
    return z3.is_const(t) and t.decl().name() == f"fconst_{_NEG_INF}"


def _op_dtype(name: str) -> str:
    return name.split("_")[1] if "_" in name else ""


class _LaneFolder:
    """One bottom-up rewrite for the lane tier: the -inf rules, then constant lane products.

    The caches live as long as the folder, so the subterms that every output cell of a trace
    shares (a row's max and exp terms, say) are rewritten once.
    """

    def __init__(self):
        self._out: dict = {}
        self._finite: dict = {}

    # Both caches keep the term next to its result: an AST id is only unique while the term is
    # alive, and the simplified terms of earlier output cells would otherwise be freed and their
    # ids reused for different terms.
    def finite(self, t) -> bool:
        key = t.get_id()
        if key not in self._finite:
            self._finite[key] = (t, self._decide_finite(t))
        return self._finite[key][1]

    def _decide_finite(self, t) -> bool:
        if z3.is_rational_value(t):
            return abs(float(t.as_fraction())) <= _FP32_MAX
        if z3.is_select(t):
            return z3.is_const(t.arg(0))  # a load from input memory, not through a Store chain
        if not (z3.is_app(t) and t.num_args() and t.decl().kind() == z3.Z3_OP_UNINTERPRETED):
            return False
        name, kids = t.decl().name(), t.children()
        if _op_dtype(name) not in _WIDE:
            return False
        if name.startswith("fmax_"):
            return all(self.finite(k) or _is_neg_inf(k) for k in kids) and any(self.finite(k) for k in kids)
        return name.startswith(_TOTAL_OPS) and all(self.finite(k) for k in kids)

    def __call__(self, t):
        key = t.get_id()
        if key not in self._out:
            self._out[key] = (t, self._rewrite(t))
        return self._out[key][1]

    def _rewrite(self, t):
        if not (z3.is_app(t) and t.num_args()):
            return t
        kids = [self(c) for c in t.children()]
        decl = t.decl()
        if decl.kind() != z3.Z3_OP_UNINTERPRETED:
            changed = [(c, k) for c, k in zip(t.children(), kids) if not z3.eq(c, k)]
            return z3.substitute(t, *changed) if changed else t
        name = decl.name()
        if name.startswith("fsub_") and _is_neg_inf(kids[0]) and self.finite(kids[1]):
            return kids[0]
        if name.startswith("fexp_") and _is_neg_inf(kids[0]):
            return z3.RealVal(0)
        if name.startswith(_LANE_PROD):
            folded = _fold_product(kids[0], kids[1])
            if folded is not None:
                return folded
        return decl(*kids)


def _fold_product(x, y):
    """The value of a lane product of two narrow-exact numerals, or None (also when it is -0)."""
    a, b = _numeral(x), _numeral(y)
    if a is None or b is None or not (_narrow_exact(a) and _narrow_exact(b)):
        return None
    p = a * b
    return None if p == 0 and math.copysign(1.0, p) < 0 else z3.RealVal(p)


def _fold_lane(term):
    """Apply the lane-tier rewrites to one term (a fresh folder; ``uf_identical`` shares one)."""
    return _LaneFolder()(term)


class FT:
    """Symbolic float tile: object array of Z3 Real terms, a dtype, and the value domain."""

    def __init__(self, arr, dtype: str, dom):
        self.arr, self.dtype, self.dom = _obj(arr), dtype, dom

    @property
    def shape(self):
        return self.arr.shape

    def _lift(self, o) -> "FT":
        if isinstance(o, FT):
            return o
        if isinstance(o, (int, float, np.integer, np.floating)) and not isinstance(o, bool):
            return FT(self.dom.const(float(o)), self.dtype, self.dom)
        raise Unsupported(f"fp operand {o!r}")

    def _conv(self, dt):
        if dt == self.dtype:
            return self.arr
        return np.frompyfunc(lambda x: self.dom.cvt(x, self.dtype, dt), 1, 1)(self.arr)

    def _bin(self, o, op, rev=False):
        o = self._lift(o)
        dt = result_dtype(self.dtype, self.arr.ndim == 0, o.dtype, o.arr.ndim == 0, op)
        a, b = self._conv(dt), o._conv(dt)
        if rev:
            a, b = b, a
        f = getattr(self.dom, op)
        return FT(np.frompyfunc(lambda x, y: f(x, y, dt), 2, 1)(a, b), dt, self.dom)

    def __add__(self, o): return self._bin(o, "add")
    def __radd__(self, o): return self._bin(o, "add", rev=True)
    def __sub__(self, o): return self._bin(o, "sub")
    def __rsub__(self, o): return self._bin(o, "sub", rev=True)
    def __mul__(self, o): return self._bin(o, "mul")
    def __rmul__(self, o): return self._bin(o, "mul", rev=True)
    def __truediv__(self, o): return self._bin(o, "div")
    def __rtruediv__(self, o): return self._bin(o, "div", rev=True)

    def _unary(self, op):
        f = getattr(self.dom, op)
        return FT(np.frompyfunc(lambda x: f(x, self.dtype), 1, 1)(self.arr), self.dtype, self.dom)

    def __getitem__(self, key):
        return FT(self.arr[key], self.dtype, self.dom)

    def __bool__(self):
        raise Unsupported("data-dependent Python control flow on a tile")

    def to(self, dtype, **_kw):
        dt = _dtype_name(dtype)
        return FT(self._conv(dt), dt, self.dom)


class _FpSymTL(oracle._SymTL):
    float16, bfloat16, float32, float64 = "float16", "bfloat16", "float32", "float64"

    def __init__(self, ctx, mem, tag, dom, launch: Launch):
        super().__init__(ctx, mem, tag)
        self._dom, self._launch = dom, launch

    def zeros(self, shape, dtype=None):
        arr = np.empty(tuple(shape), dtype=object)
        arr.fill(self._dom.const(0))
        return FT(arr, _dtype_name(dtype or "float32"), self._dom)

    def _undef(self):
        self._n += 1
        return z3.Real(f"undef_{self._tag}_{self._n}")

    def load(self, ptr, mask=None, other=None, **_kw):
        if not isinstance(ptr, Ptr):
            raise Unsupported("tl.load of a non-pointer")
        shape = ptr.off.shape
        m = _bcast(mask, shape)
        arr = np.empty(shape, dtype=object)
        for i in np.ndindex(shape):
            sel = z3.Select(self._mem[ptr.buf], ptr.off.arr[i])
            if m is None:
                arr[i] = sel
            else:
                oth = self._dom.const(float(other)) if other is not None else self._undef()
                arr[i] = z3.If(m[i], sel, oth)
        self._ctx.accesses.append(oracle.Access("load", ptr.buf, np.asarray(ptr.off.arr), m))
        return FT(arr, self._launch.dtype(ptr.buf), self._dom)

    def store(self, ptr, value, mask=None, **_kw):
        if not isinstance(ptr, Ptr):
            raise Unsupported("tl.store to a non-pointer")
        shape = ptr.off.shape
        if not isinstance(value, FT):
            raise Unsupported("store of a non-float value")
        value = value.to(self._launch.dtype(ptr.buf))  # tl.store casts to the pointee dtype
        self._ctx.accesses.append(oracle.Access("store", ptr.buf, np.asarray(ptr.off.arr),
                                                _bcast(mask, shape), _bcast(T(value.arr, "val"), shape)))

    def sum(self, x, axis=None, **_kw):
        if axis is None:
            xs = list(x.arr.reshape(-1))
            return FT(self._dom.sum(xs, x.dtype), x.dtype, self._dom)
        moved = np.moveaxis(x.arr, axis, -1)
        out = np.empty(moved.shape[:-1], dtype=object)
        for i in np.ndindex(out.shape):
            out[i] = self._dom.sum(list(moved[i]), x.dtype)
        return FT(out, x.dtype, self._dom)

    def dot(self, a, b, acc=None, out_dtype=None, **_kw):
        dt = acc.dtype if acc is not None else _dtype_name(out_dtype or "float32")
        (m, _), (_, n) = a.shape, b.shape
        out = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                out[i, j] = self._dom.dot(list(a.arr[i, :]), list(b.arr[:, j]), dt)
        r = FT(out, dt, self._dom)
        return r if acc is None else r + acc

    @staticmethod
    def reshape(x, shape, *_a, **_kw):
        return FT(np.reshape(x.arr, tuple(shape)), x.dtype, x.dom)

    @staticmethod
    def trans(x, *_a, **_kw):
        return FT(x.arr.T, x.dtype, x.dom)

    @staticmethod
    def exp(x):
        return x._unary("exp")

    @staticmethod
    def sqrt(x):
        return x._unary("sqrt")

    def max(self, x, axis=None, **_kw):
        if axis is None:
            return FT(self._dom.max(list(x.arr.reshape(-1)), x.dtype), x.dtype, self._dom)
        moved = np.moveaxis(x.arr, axis, -1)
        out = np.empty(moved.shape[:-1], dtype=object)
        for i in np.ndindex(out.shape):
            out[i] = self._dom.max(list(moved[i]), x.dtype)
        return FT(out, x.dtype, self._dom)

    def where(self, c, x, y):
        ref = x if isinstance(x, FT) else y
        x, y = ref._lift(x), ref._lift(y)
        dt = result_dtype(x.dtype, x.arr.ndim == 0, y.dtype, y.arr.ndim == 0, "where")
        shape = np.broadcast_shapes(c.shape, x.shape, y.shape)
        cond = np.broadcast_to(c.arr, shape)
        xa, ya = np.broadcast_to(x._conv(dt), shape), np.broadcast_to(y._conv(dt), shape)
        return FT(np.frompyfunc(z3.If, 3, 1)(cond, xa, ya), dt, self._dom)


def _real_memory(launch: Launch) -> dict:
    return {b: z3.Array(f"mem_{b}", z3.BitVecSort(IDX_BITS), z3.RealSort()) for b in launch.buffers}


def trace_fp(source: str, fn_name: str, launch: Launch, mem: dict, tag: str, dom) -> Trace:
    grid = tuple(launch.grid) + (1,) * (3 - len(launch.grid))
    tr = Trace(pids=[z3.BitVec(f"pid{a}", IDX_BITS) for a in range(3)], grid=grid)
    tl = _FpSymTL(tr, mem, tag, dom, launch)
    ns = {"tl": tl, "triton": oracle._SymTriton}
    exec(compile(source, f"<{fn_name}:{tag}:{dom.name}>", "exec"), ns)
    args = {k: (Ptr(v, T(z3.BitVecVal(0, IDX_BITS), "idx")) if isinstance(v, str) else v)
            for k, v in launch.args.items()}
    if hasattr(dom, "abstract"):
        dom.abstract = False  # per trace: a domain object may be reused for the next trace
    ns[fn_name](**args)
    tr.abstract = getattr(dom, "abstract", False)
    return tr


def _final_values(tr: Trace, launch: Launch) -> dict:
    vals, _ = oracle._final_memory(tr, launch)
    return vals


def real_counterexample(spec: KernelSpec, mut_src: str, size: dict) -> dict | None:
    """None if ref and mutant compute the same real-valued outputs at ``size`` (a proof)."""
    launch = spec.make_launch(**size)
    mem = _real_memory(launch)
    dom = _RealDom()  # one domain, so both traces share the same exp/sqrt functions
    ref_tr = trace_fp(spec.source, spec.fn_name, launch, mem, "ref", dom)
    mut_tr = trace_fp(mut_src, spec.fn_name, launch, mem, "mut", dom)
    abstract = ref_tr.abstract or mut_tr.abstract
    ref, mut = _final_values(ref_tr, launch), _final_values(mut_tr, launch)
    for buf in launch.outputs:
        for i in range(launch.buffers[buf]):
            init = z3.Select(mem[buf], z3.BitVecVal(i, IDX_BITS))
            r, u = ref.get((buf, i), init), mut.get((buf, i), init)
            if z3.eq(r, u) or z3.is_true(z3.simplify(r == u)):
                continue
            if abstract and z3.eq(z3.simplify(r, som=True, arith_lhs=True),
                                      z3.simplify(u, som=True, arith_lhs=True)):
                continue
            if z3.simplify(r - u, som=True).eq(z3.RealVal(0)):
                continue
            s = z3.Solver()
            s.set("timeout", 20000)
            s.add(r != u)
            res = s.check()
            if res == z3.unsat:
                continue
            w = {"buffer": buf, "index": i, "solver": str(res)}
            if abstract:
                w["abstract"] = True
            if res == z3.sat:
                mdl = s.model()
                w.update(ref_value=str(mdl.eval(r, model_completion=True)),
                         mut_value=str(mdl.eval(u, model_completion=True)))
            return w
    return None


def uf_identical(spec: KernelSpec, mut_src: str, size: dict, lanes: bool = False) -> bool:
    """True if every output term is syntactically identical with rounded ops uninterpreted.

    ``lanes=True`` models ``tl.dot`` through its lane products and rewrites with ``_LaneFolder``:
    the ``-inf`` padding rules and constant lane products. That adds the lane-product assumption
    and, for the ``-inf`` rules, the no-overflow scope on fp32/fp64 sums and products.
    """
    launch = spec.make_launch(**size)
    mem = _real_memory(launch)
    dom = _UFDom(lanes=lanes)
    ref = _final_values(trace_fp(spec.source, spec.fn_name, launch, mem, "ref", dom), launch)
    mut = _final_values(trace_fp(mut_src, spec.fn_name, launch, mem, "mut", dom), launch)
    if lanes:
        fold = _LaneFolder()
        norm = lambda t: z3.simplify(fold(z3.simplify(t)))  # noqa: E731
    else:
        norm = z3.simplify
    for buf in launch.outputs:
        for i in range(launch.buffers[buf]):
            init = z3.Select(mem[buf], z3.BitVecVal(i, IDX_BITS))
            if not z3.eq(norm(ref.get((buf, i), init)), norm(mut.get((buf, i), init))):
                return False
    return True


# ===================================================================== numeric emulator
_NP = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def _np_dtype(dt: str):
    if dt == "bfloat16":
        return np.float32  # bf16 values are held in fp32 containers, already truncated
    return _NP[dt]


def _to_bf16(x: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 by truncation (what the Triton interpreter's cast does), held in fp32."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return b.view(np.float32)


class NV:
    """Concrete float tile: value array, Triton dtype, running error bound (fp64)."""

    def __init__(self, arr, dtype: str, err, emu: "_NumTL"):
        self.arr, self.dtype, self.emu = np.asarray(arr), dtype, emu
        self.err = np.broadcast_to(np.asarray(err, dtype=np.float64), self.arr.shape)

    @property
    def shape(self):
        return self.arr.shape

    def _round(self, r: np.ndarray, dt: str, exact_err) -> "NV":
        """Wrap result ``r`` (already in dt's container), adding u|r| rounding error."""
        u = 0.0 if self.emu.exact else UNIT_ROUNDOFF[dt]
        err = exact_err + (u / (1 - u)) * np.abs(r.astype(np.float64))
        return NV(r, dt, err, self.emu)

    def _cast_arr(self, dt):
        if self.emu.exact:
            return self.arr.astype(np.float64)
        if dt == "bfloat16":
            return _to_bf16(self.arr.astype(np.float32))
        return self.arr.astype(_np_dtype(dt))

    def to(self, dtype, **_kw):
        dt = _dtype_name(dtype)
        r = self._cast_arr(dt)
        if dt == self.dtype or (self.dtype, dt) in _EXACT_WIDEN or self.emu.exact:
            return NV(r, dt, self.err, self.emu)
        return self._round(r, dt, self.err)

    def _lift(self, o) -> "NV":
        if isinstance(o, NV):
            return o
        if isinstance(o, (int, float, np.integer, np.floating)) and not isinstance(o, bool):
            return NV(np.asarray(o, dtype=np.float64 if self.emu.exact else _np_dtype(self.dtype)),
                      self.dtype, 0.0, self.emu)
        raise Unsupported(f"fp operand {o!r}")

    def _bin(self, o, op, rev=False):
        o = self._lift(o)
        dt = result_dtype(self.dtype, self.arr.ndim == 0, o.dtype, o.arr.ndim == 0, op)
        if dt == "bfloat16" and not self.emu.exact:
            raise Unsupported("bf16 arithmetic (the interpreter stores bf16 as uint16)")
        a, b = self.to(dt), o.to(dt)
        if rev:
            a, b = b, a
        ea, eb = a.err, b.err
        fa, fb = np.abs(a.arr.astype(np.float64)), np.abs(b.arr.astype(np.float64))
        if op == "add":
            r, e = a.arr + b.arr, ea + eb
        elif op == "sub":
            r, e = a.arr - b.arr, ea + eb
        elif op == "mul":
            r = a.arr * b.arr
            e = fa * eb + fb * ea + ea * eb
        else:  # div
            r = a.arr / b.arr
            den = fb * (fb - eb)
            e = np.where(fb > eb, (fa * eb + fb * ea) / np.where(den > 0, den, 1.0), np.inf)
        out = self._round(np.asarray(r), dt, e)
        # -inf padding: inf (op) finite is exactly inf in both the computed and the exact run.
        inf_in = np.isinf(np.broadcast_to(fa, out.shape)) | np.isinf(np.broadcast_to(fb, out.shape))
        out.err = np.where(np.isinf(out.arr) & inf_in, 0.0, out.err)
        return out

    def __add__(self, o): return self._bin(o, "add")
    def __radd__(self, o): return self._bin(o, "add", rev=True)
    def __sub__(self, o): return self._bin(o, "sub")
    def __rsub__(self, o): return self._bin(o, "sub", rev=True)
    def __mul__(self, o): return self._bin(o, "mul")
    def __rmul__(self, o): return self._bin(o, "mul", rev=True)
    def __truediv__(self, o): return self._bin(o, "div")
    def __rtruediv__(self, o): return self._bin(o, "div", rev=True)

    def __getitem__(self, key):
        return NV(self.arr[key], self.dtype, self.err[key], self.emu)

    def __bool__(self):
        raise Unsupported("data-dependent Python control flow on a tile")


class NPtr:
    def __init__(self, buf: str, off):
        self.buf, self.off = buf, np.asarray(off)

    def __add__(self, o):
        return NPtr(self.buf, self.off + np.asarray(o))

    __radd__ = __add__


class _NumTL:
    """Concrete ``tl`` shim mirroring the Triton interpreter's numpy calls."""

    float16, bfloat16, float32, float64 = "float16", "bfloat16", "float32", "float64"
    constexpr = object

    def __init__(self, launch: Launch, mem: dict, err: dict, exact: bool):
        self.launch, self.mem, self.errmem, self.exact = launch, mem, err, exact
        self.pid = (0, 0, 0)

    def program_id(self, axis):
        return self.pid[axis]

    def num_programs(self, axis):
        return (tuple(self.launch.grid) + (1, 1))[axis]

    @staticmethod
    def cdiv(a, b):
        return -(-a // b)

    @staticmethod
    def arange(start, end):
        return np.arange(start, end, dtype=np.int64)

    def zeros(self, shape, dtype=None):
        dt = _dtype_name(dtype or "float32")
        return NV(np.zeros(tuple(shape), dtype=np.float64 if self.exact else _np_dtype(dt)), dt, 0.0, self)

    def load(self, ptr, mask=None, other=None, **_kw):
        buf, off = ptr.buf, ptr.off
        data, n = self.mem[buf], self.launch.buffers[buf]
        m = np.ones(off.shape, bool) if mask is None else np.broadcast_to(np.asarray(mask), off.shape)
        if np.any(m & ((off < 0) | (off >= n))):
            raise Unsupported(f"out-of-bounds load from {buf} in the numeric emulator")
        safe = np.where(m, off, 0)
        fill = 0.0 if other is None else float(other)
        val = np.where(m, data[safe], np.asarray(fill, dtype=data.dtype))
        err = np.where(m, self.errmem[buf][safe], 0.0)
        return NV(val.astype(data.dtype), self.launch.dtype(buf), err, self)

    def store(self, ptr, value, mask=None, **_kw):
        buf, off = ptr.buf, ptr.off
        if not isinstance(value, NV):
            raise Unsupported("store of a non-float value")
        v = value.to(self.launch.dtype(buf))
        vals = np.broadcast_to(v.arr, off.shape)
        errs = np.broadcast_to(v.err, off.shape)
        m = np.ones(off.shape, bool) if mask is None else np.broadcast_to(np.asarray(mask), off.shape)
        if np.any(m & ((off < 0) | (off >= self.launch.buffers[buf]))):
            raise Unsupported(f"out-of-bounds store to {buf} in the numeric emulator")
        self.mem[buf][off[m]] = vals[m]
        self.errmem[buf][off[m]] = errs[m]

    def sum(self, x, axis=None, **_kw):
        r = np.asarray(np.sum(x.arr, axis=axis))  # interpreter: np.sum, dtype preserved
        n = x.arr.size if axis is None else x.arr.shape[axis]
        u = 0.0 if self.exact else UNIT_ROUNDOFF[x.dtype]
        e = np.sum(x.err, axis=axis) + gamma(n - 1, u) * np.sum(np.abs(x.arr.astype(np.float64)), axis=axis)
        return NV(r, x.dtype, e, self)

    def dot(self, a, b, acc=None, out_dtype=None, **_kw):
        dt = acc.dtype if acc is not None else _dtype_name(out_dtype or "float32")
        if acc is None:
            acc = self.zeros((a.shape[0], b.shape[1]), dt)
        npdt = np.float64 if self.exact else _np_dtype(dt)
        prod = np.matmul(a.arr, b.arr, dtype=npdt)  # interpreter: np.matmul(..., dtype=d) + d
        r = np.asarray(prod + acc.arr)
        u = 0.0 if self.exact else UNIT_ROUNDOFF[dt]
        fa, fb = np.abs(a.arr.astype(np.float64)), np.abs(b.arr.astype(np.float64))
        k = a.shape[1]
        e_dot = gamma(k, u) * (fa @ fb) + fa @ b.err + a.err @ fb + a.err @ b.err
        e = e_dot + acc.err + (u / (1 - u)) * np.abs(r.astype(np.float64))
        return NV(r, dt, e, self)

    @staticmethod
    def reshape(x, shape, *_a, **_kw):
        return NV(np.reshape(x.arr, tuple(shape)), x.dtype, np.reshape(x.err, tuple(shape)), x.emu)

    @staticmethod
    def trans(x, *_a, **_kw):
        return NV(x.arr.T, x.dtype, x.err.T, x.emu)

    def _check_fp32(self, x, name):
        if x.dtype in ("float16", "bfloat16"):  # the interpreter raises the same way
            raise Unsupported(f"tl.{name} of {x.dtype} (Triton accepts fp32/fp64 only)")

    def exp(self, x):
        self._check_fp32(x, "exp")
        r = np.asarray(np.exp(x.arr))
        u = 0.0 if self.exact else 2 * UNIT_ROUNDOFF[x.dtype]
        fr = np.abs(r.astype(np.float64))
        with np.errstate(invalid="ignore", over="ignore"):
            e = np.where(fr == 0, 0.0, fr * (np.expm1(x.err) + u) / (1 - u))
        e = np.where(np.isinf(x.arr) & (x.err == 0), 0.0, e)
        return NV(r, x.dtype, e, self)

    def sqrt(self, x):
        self._check_fp32(x, "sqrt")
        r = np.asarray(np.sqrt(x.arr))
        u = 0.0 if self.exact else UNIT_ROUNDOFF[x.dtype]
        fx = x.arr.astype(np.float64)
        root = np.sqrt(np.maximum(fx, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(root > 0, x.err / np.where(root > 0, root, 1.0), np.inf)
        e = np.where(x.err == 0, 0.0, e) + (u / (1 - u)) * np.abs(r.astype(np.float64))
        return NV(r, x.dtype, e, self)

    def max(self, x, axis=None, **_kw):
        # 1-Lipschitz in the max norm: |max(xhat) - max(x)| <= max |xhat_i - x_i|. No rounding.
        return NV(np.asarray(np.max(x.arr, axis=axis)), x.dtype, np.max(x.err, axis=axis), self)

    def where(self, c, x, y):
        ref = x if isinstance(x, NV) else y
        x, y = ref._lift(x), ref._lift(y)
        dt = result_dtype(x.dtype, x.arr.ndim == 0, y.dtype, y.arr.ndim == 0, "where")
        x, y = x.to(dt), y.to(dt)
        c = np.asarray(c)
        return NV(np.where(c, x.arr, y.arr), dt, np.where(c, x.err, y.err), self)


class _NumTriton:
    @staticmethod
    def jit(fn=None, **_kw):
        return fn if fn is not None else (lambda f: f)


def emulate(source: str, fn_name: str, launch: Launch, inputs: dict, exact: bool = False):
    """Run every program in order. Returns (memory, error bounds) per buffer.

    ``inputs`` maps buffer name -> numpy array already in the buffer's dtype. Output buffers
    start at zero. With ``exact=True`` everything is computed in fp64 (the reference value).
    """
    mem, err = {}, {}
    for b, n in launch.buffers.items():
        dt = np.float64 if exact else _np_dtype(launch.dtype(b))
        src = inputs.get(b)
        mem[b] = np.zeros(n, dtype=dt) if src is None else np.asarray(src).astype(dt).copy()
        err[b] = np.zeros(n)
    tl = _NumTL(launch, mem, err, exact)
    ns = {"tl": tl, "triton": _NumTriton}
    exec(compile(source, f"<{fn_name}:num>", "exec"), ns)
    fn = ns[fn_name]
    grid = tuple(launch.grid) + (1,) * (3 - len(launch.grid))
    with np.errstate(over="ignore", invalid="ignore"):
        for prog in itertools.product(*(range(g) for g in grid)):
            tl.pid = prog
            fn(**{k: (NPtr(v, 0) if isinstance(v, str) else v) for k, v in launch.args.items()})
    return mem, err


# ===================================================================== inputs
GENERATORS = ("randn", "rand", "mixed")


def make_inputs(launch: Launch, kind: str, seed: int) -> dict:
    """Seeded inputs rounded to each input buffer's dtype (fp16 / fp32)."""
    rng = np.random.default_rng(seed)
    out = {}
    for b, n in launch.buffers.items():
        if b in launch.outputs:
            continue
        if kind == "randn":
            x = rng.standard_normal(n)
        elif kind == "rand":
            x = rng.random(n)
        elif kind == "mixed":
            # Wide exponent spread (most of fp16's normal range) so that partial sums lose
            # low bits: fp32 sums of a few fp16 values are otherwise usually exact.
            lo, hi = (-14, 14) if launch.dtype(b) == "float16" else (-30, 30)
            x = rng.choice([-1.0, 1.0], n) * np.exp2(rng.integers(lo, hi, n)) * (1 + rng.random(n))
        else:
            raise ValueError(kind)
        out[b] = x.astype(_np_dtype(launch.dtype(b)))
    return out


# ===================================================================== labelling
@dataclass
class FpLabel:
    id: str
    label: str
    bv_label: str
    sizes: list
    fp_proof: str | None = None  # "uf" when bitwise identity is proved structurally
    real_proof: bool = True  # False: real equivalence not proved (uninterpreted exp/sqrt), only not refuted
    precision_loss: bool | None = None
    bound_ratio: float | None = None  # max over trials of mutant bound / reference bound
    witness: dict | None = None
    trials: int = 0
    seconds: float = 0.0
    notes: list = field(default_factory=list)


def _bitwise_diff(a: np.ndarray, b: np.ndarray):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    same = (a64 == b64) | (np.isnan(a64) & np.isnan(b64))
    return np.flatnonzero(~same)


def search_witness(spec: KernelSpec, mut_src: str, size: dict, trials: int = 10,
                   generators=GENERATORS, first_seed: int = 0, stop_early: bool = False):
    """Return (witness or None, max bound ratio, number of inputs tried)."""
    launch = spec.make_launch(**size)
    ratio, tried, witness = 0.0, 0, None
    for kind in generators:
        for seed in range(first_seed, first_seed + trials):
            if stop_early and witness is not None:
                return witness, ratio, tried
            inputs = make_inputs(launch, kind, seed)
            rm, re_ = emulate(spec.source, spec.fn_name, launch, inputs)
            mm, me = emulate(mut_src, spec.fn_name, launch, inputs)
            xm, _ = emulate(spec.source, spec.fn_name, launch, inputs, exact=True)
            tried += 1
            for buf in launch.outputs:
                rb, mb = float(np.max(re_[buf])), float(np.max(me[buf]))
                if rb > 0:
                    ratio = max(ratio, mb / rb)
                elif mb > 0:
                    ratio = math.inf
                diff = _bitwise_diff(rm[buf], mm[buf])
                if witness is None and diff.size:
                    i = int(diff[0])
                    witness = {
                        "size": size, "generator": kind, "seed": seed,
                        "inputs": {b: [float(v) for v in x] for b, x in inputs.items()},
                        "buffer": buf, "index": i,
                        "ref_value": float(rm[buf][i]), "mut_value": float(mm[buf][i]),
                        "fp64_value": float(xm[buf][i]),
                        "ref_err_bound": float(re_[buf][i]), "mut_err_bound": float(me[buf][i]),
                    }
    return witness, ratio, tried


WIDE_ROWS, WIDE_SEEDS = 256, 400  # row-widened search, only for row-wise mutants still unwitnessed


def search_witness_wide(spec: KernelSpec, mut_src: str, size: dict, rows: int = WIDE_ROWS,
                        seeds: int = WIDE_SEEDS, generators=GENERATORS):
    """Witness for a rare row-local difference, at ``size``; returns (witness or None, inputs tried).

    Each program of a row-wise kernel reads and writes only its own row, so ``rows`` rows per input
    give ``rows`` independent chances. The first differing row is copied into row 0 of an input at
    ``size`` and the difference is re-checked there, so the witness is at a listed size.
    """
    if "n_rows" not in size:
        return None, 0
    wide, launch = spec.make_launch(**{**size, "n_rows": rows}), spec.make_launch(**size)
    tried = 0
    for seed in range(seeds):
        for kind in generators:
            winp = make_inputs(wide, kind, seed)
            rm, _ = emulate(spec.source, spec.fn_name, wide, winp)
            mm, _ = emulate(mut_src, spec.fn_name, wide, winp)
            tried += 1
            for buf in wide.outputs:
                diff = _bitwise_diff(rm[buf], mm[buf])
                if not diff.size:
                    continue
                row = int(diff[0]) // (wide.buffers[buf] // rows)
                inputs = make_inputs(launch, kind, seed)
                for b, x in winp.items():
                    stride = wide.buffers[b] // rows
                    inputs[b] = inputs[b].copy()
                    inputs[b][:stride] = x[row * stride:(row + 1) * stride]
                w, _, _ = _witness_from(spec, mut_src, size, launch, inputs, kind, seed)
                if w is not None:
                    return {**w, "widened_from_rows": rows, "widened_row": row}, tried
    return None, tried


def _witness_from(spec, mut_src, size, launch, inputs, kind, seed):
    rm, re_ = emulate(spec.source, spec.fn_name, launch, inputs)
    mm, me = emulate(mut_src, spec.fn_name, launch, inputs)
    xm, _ = emulate(spec.source, spec.fn_name, launch, inputs, exact=True)
    for buf in launch.outputs:
        diff = _bitwise_diff(rm[buf], mm[buf])
        if diff.size:
            i = int(diff[0])
            return {
                "size": size, "generator": kind, "seed": seed,
                "inputs": {b: [float(v) for v in x] for b, x in inputs.items()},
                "buffer": buf, "index": i,
                "ref_value": float(rm[buf][i]), "mut_value": float(mm[buf][i]),
                "fp64_value": float(xm[buf][i]),
                "ref_err_bound": float(re_[buf][i]), "mut_err_bound": float(me[buf][i]),
            }, None, None
    return None, None, None


CONFIRM_RTOL = 1e-6  # fp64 outputs further apart than this are a confirmed real difference


def confirm_fp64(spec: KernelSpec, mut_src: str, sizes, trials: int = 5) -> dict | None:
    """Concrete fp64 witness that ``mut`` computes a different real function, or None.

    Used when a symbolic tier reported a difference only under uninterpreted exp/sqrt/max/div or
    a named constant. fp64 rounding (relative ~1e-16) cannot reach ``CONFIRM_RTOL``.
    """
    for size in sizes:
        launch = spec.make_launch(**size)
        for kind in ("randn", "rand"):
            for seed in range(trials):
                inputs = make_inputs(launch, kind, seed)
                try:
                    ref, _ = emulate(spec.source, spec.fn_name, launch, inputs, exact=True)
                    mut, _ = emulate(mut_src, spec.fn_name, launch, inputs, exact=True)
                except Unsupported:
                    continue
                for buf in launch.outputs:
                    r, m = ref[buf], mut[buf]
                    bad = ~(np.isclose(r, m, rtol=CONFIRM_RTOL, atol=CONFIRM_RTOL) | (np.isnan(r) & np.isnan(m)))
                    if bad.any():
                        i = int(np.flatnonzero(bad)[0])
                        return {"size": size, "theory": "fp64", "generator": kind, "seed": seed,
                                "buffer": buf, "index": i, "ref_value": float(r[i]), "mut_value": float(m[i])}
    return None


EXTENDED_TRIALS = 300  # extra "mixed" seeds, only for mutants neither proved nor witnessed


def label(spec: KernelSpec, mutant_source: str, mutant_id: str = "", trials: int = 10) -> FpLabel:
    t0 = time.time()
    bv = oracle.label(spec, mutant_source)
    sizes = list(spec.oracle_sizes) + list(spec.fp_sizes)
    out = FpLabel(mutant_id, bv.label, bv.label, sizes)
    w = bv.witness
    if bv.label != "benign" and not (bv.label == "inequivalent" and w.witness.get("abstract")):
        out.witness = {"size": w.size, **w.witness} if w else None
        out.seconds = round(time.time() - t0, 3)
        return out
    if bv.label == "inequivalent":  # difference only under uninterpreted functions: confirm
        conf = confirm_fp64(spec, mutant_source, [w.size] + [s for s in sizes if s != w.size])
        if conf is not None:
            out.witness = {"size": w.size, **w.witness, "fp64": conf}
            out.notes.append("bit-vector difference under uninterpreted functions, confirmed in fp64")
            out.seconds = round(time.time() - t0, 3)
            return out
        out.notes.append("bit-vector difference only under uninterpreted functions; not confirmed in fp64")
    for size in sizes:
        if (cx := real_counterexample(spec, mutant_source, size)) is not None:
            if cx.get("abstract"):
                conf = confirm_fp64(spec, mutant_source, [size])
                if conf is None:
                    out.real_proof = False
                    out.notes.append(f"real tier not proved at {size} (uninterpreted exp/sqrt); "
                                     "no fp64 difference found")
                    continue
                cx = {**cx, "fp64": conf}
            out.label, out.witness = "inequivalent", {"size": size, "theory": "real", **cx}
            out.notes.append("bit-vector benign but real-inequivalent")
            out.seconds = round(time.time() - t0, 3)
            return out
    if all(uf_identical(spec, mutant_source, s) for s in sizes):
        out.label, out.fp_proof = "benign", "uf"
        out.seconds = round(time.time() - t0, 3)
        return out
    if "tl.dot" in spec.source and all(uf_identical(spec, mutant_source, s, lanes=True) for s in sizes):
        out.label, out.fp_proof = "benign", "uf-lane"
        out.notes.append("proved under the lane-product model of tl.dot")
        out.seconds = round(time.time() - t0, 3)
        return out
    ratio = 0.0
    for size in sizes:
        w, r, n = search_witness(spec, mutant_source, size, trials)
        ratio, out.trials = max(ratio, r), out.trials + n
        if w is not None and out.witness is None:
            out.witness = w
    if out.witness is None and EXTENDED_TRIALS:
        # Rare witnesses (e.g. an fp32 reordering whose difference must survive a final
        # fp16 rounding): a longer search on the largest size, stopping at the first hit.
        big = max(sizes, key=lambda s: sum(spec.make_launch(**s).buffers.values()))
        w, r, n = search_witness(spec, mutant_source, big, EXTENDED_TRIALS, ("mixed", "randn"),
                                 first_seed=trials, stop_early=True)
        ratio, out.trials, out.witness = max(ratio, r), out.trials + n, w
    if out.witness is None and WIDE_SEEDS:
        for size in sizes:
            w, n = search_witness_wide(spec, mutant_source, size)
            out.trials += n
            if w is not None:
                out.witness = w
                out.notes.append(f"witness from the row-widened search ({WIDE_ROWS} rows per input)")
                break
    out.bound_ratio = ratio
    if out.witness is not None:
        out.label = "numerical"
        out.precision_loss = ratio > PRECISION_LOSS_RATIO
    else:
        out.label = "benign"
    out.seconds = round(time.time() - t0, 3)
    return out


def label_all(trials: int = 10, kernels=None) -> list[FpLabel]:
    from .mutate import mutants
    res = []
    for name, spec in KERNELS.items():
        if kernels and name not in kernels:
            continue
        for m in mutants(name, spec.source):
            lab = label(spec, m.source, m.id, trials)
            lab.notes.insert(0, m.description)
            res.append(lab)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/labels.json")
    ap.add_argument("--trials", type=int, default=10, help="seeds per generator per size")
    args = ap.parse_args(argv)
    t0 = time.time()
    labs = label_all(args.trials)
    counts: dict = {}
    for lab in labs:
        key = family(lab)
        counts[key] = counts.get(key, 0) + 1
    doc = {
        "command": "PYTHONPATH=src python -m triton_equiv.fporacle "
                   + " ".join(argv if argv is not None else sys.argv[1:]),
        "python": platform.python_version(), "numpy": np.__version__, "z3": z3.get_version_string(),
        "seconds": round(time.time() - t0, 1), "counts": counts,
        "labels": [asdict(lab) for lab in labs],
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
