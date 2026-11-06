"""Stage-1 labelling oracle: Z3 bit-vectors over index/mask logic at small fixed sizes.

What it decides
---------------
For a reference kernel R, a mutant M and one concrete launch (fixed tile sizes, fixed
problem sizes, including sizes that are **not** tile multiples), it returns one label:

``oob``           some program, at some lane, performs an unmasked-in-effect load or store
                  outside its buffer. The witness gives the program id, the lane and the address.
``inequivalent``  both are in bounds, but some output element can differ (or two programs
                  race on it with different values). The witness gives the buffer, the index
                  and a model.
``benign``        no OOB access and, for **all** input data, every buffer element ends up
                  identical. This is a proof *at that size only*.

Across several sizes the most severe label wins (oob > inequivalent > benign).

How (architecture)
------------------
1. The kernel source is ``exec``'d against a **symbolic shim** of ``triton.language``
   (``_SymTL``). Program ids are Z3 ``BitVec(32)`` variables. ``tl.arange``, offsets and
   masks are Z3 bit-vector/Bool terms. Python-level control flow (``range`` loops) stays
   concrete because all sizes and constexprs are concrete Python ints.
2. Memory: each buffer is a Z3 array ``BV32 -> BV16``. A masked-off lane reads ``other``,
   or a **fresh unconstrained value** when no ``other`` is given (Triton leaves it
   undefined). That value is Alive2-style ``undef``: if it can reach an output, the
   mutant is inequivalent.
3. OOB: one query per load/store site over the **symbolic program id** constrained to the
   grid, asking whether any lane has ``mask and not (0 <= addr < len)``. This covers every
   program in one query, the same encoding as triton-verify.
4. Equivalence: substitute each concrete program id, replay the stores in program
   order to get final memory, then ask Z3 whether ``ref[i] != mut[i]`` is satisfiable
   for every element of every buffer (structurally identical terms are skipped).

What it does NOT model (out of scope for stage 1)
-------------------------------------------------
* **Floating point.** Values are 16-bit bit-vectors with wrap-around ``+``/``*``.
  ``exp``, ``sqrt``, ``max`` and ``/`` (softmax, layernorm, attention) have no exact
  bit-vector meaning, so they are **uninterpreted functions** (``fexp``, ``fsqrt``, ``fmax``,
  ``fdiv``), and a non-integral or infinite float constant (``eps``, ``-inf``) is a named
  uninterpreted constant. Equal terms under uninterpreted functions are equal under any
  interpretation, so ``benign`` stays a proof. A difference is only a difference *for some*
  interpretation, so ``inequivalent`` from such a trace is provisional: the witness carries
  ``"abstract": True`` and ``fporacle`` keeps the label only if an fp64 run confirms it.
  Float rounding, NaN/inf, accumulator dtype (``.to``/``dtype=`` are ignored) and
  **``tl.dot`` numerics** (only the exact sum-of-products structure is kept) are not modelled.
  Consequence: a BV difference implies the integer polynomials differ, and therefore the
  real-valued kernels differ. A BV *equality* of low-degree polynomials is taken as
  equivalence, so reduction-order, accumulator-dtype and cast mutants all look benign here.
  ``fporacle`` re-examines every benign verdict over the reals and in floating point.
* Symbolic problem sizes: sizes are fixed per query, and the label holds only at those sizes.
* Read-after-write through memory inside one kernel, atomics, ``tl.where`` on pointers,
  and data-dependent control flow (the shim raises ``Unsupported``).
* Integer overflow of index arithmetic: indices are 32-bit, but sizes are tiny.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import z3

from .kernels import KernelSpec, Launch

IDX_BITS = 32
VAL_BITS = 16
SEVERITY = {"benign": 0, "inequivalent": 1, "oob": 2}


class Unsupported(Exception):
    """The kernel uses a construct the bit-vector shim does not model."""


# Uninterpreted functions for exp/sqrt/max/div. ``_ABSTRACT[0]`` is set whenever one (or a
# named float constant) is used during a trace, so that inequivalence can be marked provisional.
_UFS: dict = {}
_ABSTRACT = [False]


def _uf(name: str, arity: int):
    _ABSTRACT[0] = True
    key = (name, arity)
    if key not in _UFS:
        _UFS[key] = z3.Function(name, *([z3.BitVecSort(VAL_BITS)] * (arity + 1)))
    return _UFS[key]


# --------------------------------------------------------------------------- tensors
def _obj(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    a = np.empty((), dtype=object)
    a[()] = x
    return a


def _const(x, kind: str):
    if kind == "bool":
        return z3.BoolVal(bool(x))
    if isinstance(x, float):
        if not x.is_integer():  # eps, -inf, ...: a named uninterpreted constant
            if kind != "val":
                raise Unsupported(f"non-integral index constant {x}")
            _ABSTRACT[0] = True
            return z3.BitVec(f"fconst_{x!r}", VAL_BITS)
        x = int(x)
    return z3.BitVecVal(int(x), VAL_BITS if kind == "val" else IDX_BITS)


class T:
    """A tile: numpy object array of Z3 terms plus a kind (idx | val | bool)."""

    def __init__(self, arr, kind: str):
        self.arr = _obj(arr)
        self.kind = kind

    @property
    def shape(self):
        return self.arr.shape

    def _coerce(self, other) -> "T":
        if isinstance(other, T):
            return other
        if isinstance(other, (bool, np.bool_)):
            return T(_const(other, "bool"), "bool")
        if isinstance(other, (int, float, np.integer)):
            return T(_const(other, "val" if self.kind == "val" else "idx"), self.kind)
        raise Unsupported(f"operand {other!r}")

    def _bin(self, other, fn, out_kind=None, rev=False):
        o = self._coerce(other)
        if o.kind != self.kind:
            raise Unsupported(f"mixing {self.kind} and {o.kind}")
        a, b = (o.arr, self.arr) if rev else (self.arr, o.arr)
        return T(np.frompyfunc(fn, 2, 1)(a, b), out_kind or self.kind)

    def __add__(self, o):
        if isinstance(o, Ptr):
            return o + self
        return self._bin(o, lambda a, b: a + b)

    def __radd__(self, o): return self._bin(o, lambda a, b: a + b, rev=True)
    def __sub__(self, o): return self._bin(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._bin(o, lambda a, b: a - b, rev=True)
    def __mul__(self, o): return self._bin(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._bin(o, lambda a, b: a * b, rev=True)
    def __floordiv__(self, o): return self._bin(o, lambda a, b: a / b)  # indices are non-negative

    def __truediv__(self, o):
        if self.kind != "val":
            raise Unsupported("true division of an index tile")
        return self._bin(o, lambda a, b: _uf("fdiv", 2)(a, b))

    def __rtruediv__(self, o):
        if self.kind != "val":
            raise Unsupported("true division of an index tile")
        return self._bin(o, lambda a, b: _uf("fdiv", 2)(a, b), rev=True)
    def __mod__(self, o): return self._bin(o, lambda a, b: z3.SRem(a, b))
    def __neg__(self): return T(np.frompyfunc(lambda a: -a, 1, 1)(self.arr), self.kind)
    def __lt__(self, o): return self._bin(o, lambda a, b: a < b, "bool")
    def __le__(self, o): return self._bin(o, lambda a, b: a <= b, "bool")
    def __gt__(self, o): return self._bin(o, lambda a, b: a > b, "bool")
    def __ge__(self, o): return self._bin(o, lambda a, b: a >= b, "bool")
    def __eq__(self, o): return self._bin(o, lambda a, b: a == b, "bool")  # type: ignore[override]
    def __ne__(self, o): return self._bin(o, lambda a, b: a != b, "bool")  # type: ignore[override]
    def __and__(self, o): return self._bin(o, lambda a, b: z3.And(a, b))
    def __or__(self, o): return self._bin(o, lambda a, b: z3.Or(a, b))
    def __invert__(self): return T(np.frompyfunc(z3.Not, 1, 1)(self.arr), "bool")
    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, key):
        return T(self.arr[key], self.kind)

    def __bool__(self):
        raise Unsupported("data-dependent Python control flow on a tile")

    def to(self, *_a, **_k):
        return self  # dtype casts are not modelled (fp out of scope)


class Ptr:
    """A tile of pointers: buffer name plus an idx-tile of element offsets."""

    def __init__(self, buf: str, off: T):
        self.buf, self.off = buf, off

    def __add__(self, o):
        return Ptr(self.buf, self.off + o)

    __radd__ = __add__


def _bcast(x: T | None, shape) -> np.ndarray | None:
    return None if x is None else np.broadcast_to(x.arr, shape)


# --------------------------------------------------------------------------- shim
@dataclass
class Access:
    kind: str  # load | store
    buf: str
    off: np.ndarray
    mask: np.ndarray | None
    val: np.ndarray | None = None
    lineno: int = 0


@dataclass
class Trace:
    pids: list
    grid: tuple[int, int, int]
    accesses: list[Access] = field(default_factory=list)
    abstract: bool = False  # used an uninterpreted function or named float constant


class _SymTL:
    constexpr = object
    float16 = float32 = float64 = bfloat16 = int32 = int64 = "dtype"

    def __init__(self, ctx: Trace, mem: dict, tag: str):
        self._ctx, self._mem, self._tag, self._n = ctx, mem, tag, 0

    def program_id(self, axis):
        return T(self._ctx.pids[axis], "idx")

    def num_programs(self, axis):
        return self._ctx.grid[axis]

    @staticmethod
    def cdiv(a, b):
        if not (isinstance(a, int) and isinstance(b, int)):
            raise Unsupported("cdiv on symbolic values")
        return -(-a // b)

    @staticmethod
    def arange(start, end):
        return T(np.array([z3.BitVecVal(i, IDX_BITS) for i in range(start, end)], dtype=object), "idx")

    @staticmethod
    def zeros(shape, dtype=None):
        arr = np.empty(tuple(shape), dtype=object)
        arr.fill(z3.BitVecVal(0, VAL_BITS))
        return T(arr, "val")

    def _undef(self):
        self._n += 1
        return z3.BitVec(f"undef_{self._tag}_{self._n}", VAL_BITS)

    def load(self, ptr, mask=None, other=None, **_kw):
        if not isinstance(ptr, Ptr):
            raise Unsupported("tl.load of a non-pointer")
        shape = ptr.off.shape
        m = _bcast(mask, shape)
        oth = None if other is None else _bcast(T(_const(other, "val"), "val")
                                                 if not isinstance(other, T) else other, shape)
        arr = np.empty(shape, dtype=object)
        for i in np.ndindex(shape):
            sel = z3.Select(self._mem[ptr.buf], ptr.off.arr[i])
            if m is None:
                arr[i] = sel
            else:
                arr[i] = z3.If(m[i], sel, oth[i] if oth is not None else self._undef())
        self._ctx.accesses.append(Access("load", ptr.buf, np.asarray(ptr.off.arr), m))
        return T(arr, "val")

    def store(self, ptr, value, mask=None, **_kw):
        if not isinstance(ptr, Ptr):
            raise Unsupported("tl.store to a non-pointer")
        shape = ptr.off.shape
        if not isinstance(value, T):
            value = T(_const(value, "val"), "val")
        self._ctx.accesses.append(Access("store", ptr.buf, np.asarray(ptr.off.arr),
                                         _bcast(mask, shape), _bcast(value, shape)))

    @staticmethod
    def sum(x, axis=None, **_kw):
        return T(np.sum(x.arr, axis=axis), "val")

    @staticmethod
    def reshape(x, shape, *_a, **_kw):
        return T(np.reshape(x.arr, tuple(shape)), x.kind)  # row-major, as Triton without can_reorder

    @staticmethod
    def dot(a, b, acc=None, **_kw):
        r = T(np.dot(a.arr, b.arr), "val")
        return r if acc is None else acc + r

    @staticmethod
    def where(c, x, y):
        x = x if isinstance(x, T) else T(_const(x, "val"), "val")
        y = y if isinstance(y, T) else T(_const(y, "val"), "val")
        return T(np.frompyfunc(z3.If, 3, 1)(c.arr, x.arr, y.arr), x.kind)

    @staticmethod
    def trans(x, *_a, **_kw):
        return T(x.arr.T, x.kind)

    @staticmethod
    def exp(x):
        return T(np.frompyfunc(lambda a: _uf("fexp", 1)(a), 1, 1)(x.arr), "val")

    @staticmethod
    def sqrt(x):
        return T(np.frompyfunc(lambda a: _uf("fsqrt", 1)(a), 1, 1)(x.arr), "val")

    @staticmethod
    def max(x, axis=None, **_kw):
        """Left fold of an uninterpreted binary ``fmax`` in index order."""
        def fold(xs):
            acc = xs[0]
            for v in xs[1:]:
                acc = _uf("fmax", 2)(acc, v)
            return acc
        if axis is None:
            return T(fold(list(x.arr.reshape(-1))), "val")
        moved = np.moveaxis(x.arr, axis, -1)
        out = np.empty(moved.shape[:-1], dtype=object)
        for i in np.ndindex(out.shape):
            out[i] = fold(list(moved[i]))
        return T(out, "val")


class _SymTriton:
    @staticmethod
    def jit(fn=None, **_kw):
        return fn if fn is not None else (lambda f: f)

    cdiv = staticmethod(_SymTL.cdiv)


def _memory(launch: Launch) -> dict:
    return {b: z3.Array(f"mem_{b}", z3.BitVecSort(IDX_BITS), z3.BitVecSort(VAL_BITS))
            for b in launch.buffers}


def trace(source: str, fn_name: str, launch: Launch, mem: dict, tag: str) -> Trace:
    grid = tuple(launch.grid) + (1,) * (3 - len(launch.grid))
    tr = Trace(pids=[z3.BitVec(f"pid{a}", IDX_BITS) for a in range(3)], grid=grid)
    tl = _SymTL(tr, mem, tag)
    ns = {"tl": tl, "triton": _SymTriton}
    exec(compile(source, f"<{fn_name}:{tag}>", "exec"), ns)
    args = {k: (Ptr(v, T(z3.BitVecVal(0, IDX_BITS), "idx")) if isinstance(v, str) else v)
            for k, v in launch.args.items()}
    _ABSTRACT[0] = False
    ns[fn_name](**args)
    tr.abstract = _ABSTRACT[0]
    return tr


# --------------------------------------------------------------------------- queries
@dataclass
class Verdict:
    label: str
    size: dict
    witness: dict = field(default_factory=dict)


def _pid_bounds(tr: Trace):
    return [z3.And(p >= 0, p < g) for p, g in zip(tr.pids, tr.grid)]


def find_oob(tr: Trace, launch: Launch) -> dict | None:
    for acc in tr.accesses:
        n = launch.buffers[acc.buf]
        lanes = list(np.ndindex(acc.off.shape))
        bad = [z3.And(acc.mask[i] if acc.mask is not None else z3.BoolVal(True),
                      z3.Or(acc.off[i] < 0, acc.off[i] >= n)) for i in lanes]
        s = z3.Solver()
        s.add(*_pid_bounds(tr), z3.Or(*bad))
        if s.check() == z3.sat:
            m = s.model()
            lane = next(i for i, b in zip(lanes, bad) if z3.is_true(m.eval(b, model_completion=True)))
            return {
                "access": acc.kind, "buffer": acc.buf, "length": n,
                "pid": tuple(m.eval(p, model_completion=True).as_signed_long() for p in tr.pids[:len(launch.grid)]),
                "lane": lane,
                "address": m.eval(acc.off[lane], model_completion=True).as_signed_long(),
            }
    return None


def _final_memory(tr: Trace, launch: Launch):
    writes: dict = {}
    races = []
    for prog in itertools.product(*(range(g) for g in tr.grid)):
        sub = [(p, z3.BitVecVal(v, IDX_BITS)) for p, v in zip(tr.pids, prog)]
        for acc in tr.accesses:
            if acc.kind != "store":
                continue
            for i in np.ndindex(acc.off.shape):
                if acc.mask is not None:
                    m = z3.simplify(z3.substitute(acc.mask[i], *sub))
                    if z3.is_false(m):
                        continue
                    if not z3.is_true(m):
                        raise Unsupported("store mask depends on data")
                addr = z3.simplify(z3.substitute(acc.off[i], *sub)).as_signed_long()
                val = z3.simplify(z3.substitute(acc.val[i], *sub))
                key = (acc.buf, addr)
                if key in writes and writes[key][0] != prog:
                    races.append((key, writes[key][1], val))
                writes[key] = (prog, val)
    return {k: v for k, (_, v) in writes.items()}, races


def _differ(a, b):
    if z3.eq(a, b):
        return None
    s = z3.Solver()
    s.add(a != b)
    return s.model() if s.check() == z3.sat else None


def compare(ref: Trace, mut: Trace, launch: Launch, mem: dict) -> dict | None:
    """Return a counterexample dict if ``mut`` can leave different memory, else None."""
    ref_mem, ref_races = _final_memory(ref, launch)
    assert not any(_differ(a, b) for _, a, b in ref_races), "reference kernel races"
    mut_mem, mut_races = _final_memory(mut, launch)
    for (buf, addr), a, b in mut_races:
        if (m := _differ(a, b)) is not None:
            return {"reason": "write-write race", "buffer": buf, "index": addr}
    for buf, n in launch.buffers.items():
        for i in range(n):
            init = z3.Select(mem[buf], z3.BitVecVal(i, IDX_BITS))
            r, u = ref_mem.get((buf, i), init), mut_mem.get((buf, i), init)
            if (m := _differ(r, u)) is not None:
                return {
                    "reason": "output differs" if (buf, i) in mut_mem else "output not written",
                    "buffer": buf, "index": i,
                    "ref_value": m.eval(r, model_completion=True).as_long(),
                    "mut_value": m.eval(u, model_completion=True).as_long(),
                }
    return None


def label_at(spec: KernelSpec, mutant_source: str, size: dict) -> Verdict:
    launch = spec.make_launch(**size)
    mem = _memory(launch)
    ref = trace(spec.source, spec.fn_name, launch, mem, "ref")
    assert find_oob(ref, launch) is None, f"reference {spec.name} is OOB at {size}"
    mut = trace(mutant_source, spec.fn_name, launch, mem, "mut")
    if (w := find_oob(mut, launch)) is not None:
        return Verdict("oob", size, w)
    if (w := compare(ref, mut, launch, mem)) is not None:
        if ref.abstract or mut.abstract:
            w["abstract"] = True  # provisional: a difference for *some* interpretation of the UFs
        return Verdict("inequivalent", size, w)
    return Verdict("benign", size)


@dataclass
class Label:
    label: str
    verdicts: list[Verdict]

    @property
    def witness(self) -> Verdict | None:
        return next((v for v in self.verdicts if v.label == self.label and v.label != "benign"), None)


def label(spec: KernelSpec, mutant_source: str, sizes: list[dict] | None = None) -> Label:
    """Label a mutant at every size in ``sizes`` (default: ``spec.oracle_sizes``)."""
    verdicts = [label_at(spec, mutant_source, s) for s in (sizes or spec.oracle_sizes)]
    worst = max(verdicts, key=lambda v: SEVERITY[v.label]).label
    return Label(worst, verdicts)
