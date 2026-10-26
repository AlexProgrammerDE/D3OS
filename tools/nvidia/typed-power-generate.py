#!/usr/bin/env python3
"""Reproduce typed Power fixtures with exact stored inputs and MPFR 4.2.2.

--check is read-only. Writes require --output-root. No cache module, host libm,
production arithmetic, compiler result, or GPU supplies numerical expectations.
"""

import argparse
from collections import Counter
import ctypes as C
import ctypes.util
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import struct

FORMATS = {
    "f64": (11, 52),
    "f32": (8, 23),
    "flex32": (8, 23),
    "f16": (5, 10),
    "bf16": (8, 7),
}
HALVES = ("f16", "bf16")
LIB = C.CDLL(ctypes.util.find_library("mpfr"))


class MP(C.Structure):
    _fields_ = [
        ("precision", C.c_long),
        ("sign", C.c_int),
        ("exponent", C.c_long),
        ("limbs", C.POINTER(C.c_ulong)),
    ]


P = C.POINTER(MP)


def bind(name, args, result=C.c_int):
    fn = getattr(LIB, "mpfr_" + name)
    fn.argtypes, fn.restype = args, result
    return fn


INIT = bind("init2", [P, C.c_long], None)
CLEAR = bind("clear", [P], None)
SET = bind("set_str", [P, C.c_char_p, C.c_int, C.c_int])
SCALE = bind("mul_2si", [P, P, C.c_long, C.c_int])
POW = bind("pow", [P, P, P, C.c_int])
GET = bind(
    "get_str",
    [C.c_void_p, C.POINTER(C.c_long), C.c_int, C.c_size_t, P, C.c_int],
    C.c_void_p,
)
FREE = bind("free_str", [C.c_void_p], None)
GETEXP = bind("get_exp", [P], C.c_long)
ISNAN = bind("nan_p", [P])
ISINF = bind("inf_p", [P])
ISZERO = bind("zero_p", [P])
VERSION = bind("get_version", [], C.c_char_p)
assert VERSION() == b"4.2.2", VERSION()


def pow2(k):
    return Q(1 << max(k, 0), 1 << max(-k, 0))


def info(fmt):
    e, f = FORMATS[fmt]
    return e, f, (1 << (e - 1)) - 1, 1 << (e + f), ((1 << e) - 1) << f


def decode(bits, fmt):
    e, f, bias, sign, inf = info(fmt)
    assert 0 <= bits < sign * 2
    field, frac = (bits >> f) & ((1 << e) - 1), bits & ((1 << f) - 1)
    if field == (1 << e) - 1:
        return ("nan" if frac else "inf"), (-1 if bits & sign else 1)
    n = frac if field == 0 else (1 << f) + frac
    q = n * pow2((1 - bias if field == 0 else field - bias) - f)
    return "finite", (-q if bits & sign else q)


def floor_log2(q):
    q = abs(q)
    k = q.numerator.bit_length() - q.denominator.bit_length()
    return k - int(q < pow2(k))


def rne_integer(q):
    n, r = divmod(q.numerator, q.denominator)
    return n + int(2 * r > q.denominator or (2 * r == q.denominator and n & 1))


def encode(q, fmt, negative_zero=False):
    e, f, bias, sign, inf = info(fmt)
    s = sign if q < 0 or (q == 0 and negative_zero) else 0
    q = abs(q)
    if not q:
        return s
    k = floor_log2(q)
    if k > bias + 1:
        return s | inf
    if k < -bias - f - 2:
        return s
    step = max(k - f, 1 - bias - f)
    n = rne_integer(q / pow2(step))
    if not n:
        return s
    if n >= 1 << (f + 1):
        n >>= 1
        step += 1
    if step + f > bias:
        return s | inf
    if n < 1 << f:
        return s | n
    return s | ((step + f + bias) << f) | (n - (1 << f))


def classify(bits, fmt):
    if bits is None:
        return "nan"
    e, f, bias, sign, inf = info(fmt)
    b = bits & (sign - 1)
    if b > inf:
        return "nan"
    if b == inf:
        return "inf"
    if not b:
        return "zero"
    return "subnormal" if b < 1 << f else "normal"


def rank(bits, fmt):
    sign = info(fmt)[3]
    return (~bits & (2 * sign - 1)) if bits & sign else bits | sign


def unrank(r, fmt):
    sign = info(fmt)[3]
    return r & (sign - 1) if r & sign else ~r & (2 * sign - 1)


def cast(bits, src, dst):
    if bits is None:
        return None
    kind, q = decode(bits, src)
    if kind == "nan":
        return None
    if kind == "inf":
        return info(dst)[4] | (info(dst)[3] if q < 0 else 0)
    return encode(q, dst, bool(bits & info(src)[3]))


def code(bits, fmt):
    return None if bits is None else f"{bits:0{(sum(FORMATS[fmt]) + 1) // 4}x}"


def set_exact(v, q):
    assert q.denominator & (q.denominator - 1) == 0
    n = q.numerator
    trailing = (abs(n) & -abs(n)).bit_length() - 1 if n else 0
    assert (abs(n) >> trailing).bit_length() <= v.precision
    assert SET(C.byref(v), str(n >> trailing).encode(), 10, 0) == 0
    assert (
        SCALE(C.byref(v), C.byref(v), trailing - (q.denominator.bit_length() - 1), 0)
        == 0
    )


def endpoint(v, fmt):
    """Classify MPFR range BEFORE materializing any dyadic Fraction."""
    e, f, bias, sign, inf = info(fmt)
    s = sign if v.sign < 0 else 0
    if ISNAN(C.byref(v)):
        return None
    if ISINF(C.byref(v)):
        return s | inf
    if ISZERO(C.byref(v)):
        return s
    exponent = GETEXP(C.byref(v))
    if exponent > bias + 2:
        return s | inf
    if exponent < 1 - bias - f - 1:
        return s
    out_exp = C.c_long()
    ptr = GET(None, C.byref(out_exp), 2, 0, C.byref(v), 0)
    try:
        digits = C.string_at(ptr).decode()
    finally:
        FREE(ptr)
    negative = digits.startswith("-")
    digits = digits.lstrip("-")
    assert "@" not in digits
    q = int(digits, 2) * pow2(out_exp.value - len(digits))
    return encode(-q if negative else q, fmt)


def directed(x, y, fmt, precision):
    values = [MP() for _ in range(4)]
    for v in values:
        INIT(C.byref(v), precision)
    a, b, lo, hi = values
    try:
        set_exact(a, x)
        set_exact(b, y)
        POW(C.byref(lo), C.byref(a), C.byref(b), 3)  # RNDD
        POW(C.byref(hi), C.byref(a), C.byref(b), 2)  # RNDU
        targets = (fmt, "f32") if fmt in HALVES else (fmt,)
        return tuple((endpoint(lo, target), endpoint(hi, target)) for target in targets)
    finally:
        for v in values:
            CLEAR(C.byref(v))


def special(xbits, ybits, fmt):
    """Return (handled, result). Parity uses exact stored exponents."""
    kx, x = decode(xbits, fmt)
    ky, y = decode(ybits, fmt)
    _, _, _, sign, inf = info(fmt)
    one = encode(Q(1), fmt)
    if ky == "finite" and y == 0:
        return True, one
    if kx == "finite" and x == 1:
        return True, one
    if kx == "finite" and x == -1 and ky == "inf":
        return True, one
    if kx == "nan" or ky == "nan":
        return True, None
    if ky == "inf":
        if kx == "inf":
            return True, (inf if y > 0 else 0)
        ax = abs(x)
        return True, (inf if (ax > 1) == (y > 0) else 0)
    integer = y.denominator == 1
    odd = integer and bool(y.numerator & 1)
    negative = bool(xbits & sign) and odd
    s = sign if negative else 0
    if kx == "inf":
        return True, s | (inf if y > 0 else 0)
    if x == 0:
        return True, s | (0 if y > 0 else inf)
    if x < 0 and not integer:
        return True, None
    if x == -1:
        return True, encode(Q(-1 if odd else 1), fmt)
    if y == 1:
        return True, xbits
    return False, None


def allowed_half(center, fmt, exact=False):
    if center is None:
        return []
    if exact:
        return [cast(center, "f32", fmt)]
    return sorted(
        {
            cast(unrank(r, "f32"), "f32", fmt)
            for r in range(rank(center, "f32") - 4, rank(center, "f32") + 5)
            if 0 <= r < 1 << 32 and classify(unrank(r, "f32"), "f32") != "nan"
        }
    )


def reference(xbits, ybits, fmt, minimum=128):
    handled, ideal = special(xbits, ybits, fmt)
    center, p = None, 0
    if handled:
        if fmt in HALVES:
            center = cast(ideal, fmt, "f32")
    else:
        _, x = decode(xbits, fmt)
        _, y = decode(ybits, fmt)
        p = minimum
        while p <= 8192:
            bounds = directed(x, y, fmt, p)
            if all(a == b for a, b in bounds):
                ideal = bounds[0][0]
                if fmt in HALVES:
                    center = bounds[1][0]
                break
            p *= 2
        else:
            raise RuntimeError(("unresolved", fmt, code(xbits, fmt), code(ybits, fmt)))
    expected = cast(center, "f32", fmt) if fmt in HALVES else ideal
    out = dict(
        ideal=ideal,
        center=center,
        expected=expected,
        precision=p,
        method="special" if handled else "directed_mpfr",
    )
    if fmt in HALVES:
        out["allowed"] = allowed_half(center, fmt, handled)
    return out


def qualify(actual, fmt, ref, budget=None):
    expected = ref["expected"]
    ec, ac = classify(expected, fmt), classify(actual, fmt)
    finite = ("normal", "subnormal")
    failures, diagnostics = [], []
    if ec != ac:
        if ec in finite and ac in finite:
            diagnostics.append("normal_subnormal_transition")
        else:
            failures.append("class_mismatch")
    if (
        ec != "nan"
        and ac != "nan"
        and bool(expected & info(fmt)[3]) != bool(actual & info(fmt)[3])
    ):
        failures.append("sign_mismatch")
    if ref["method"] == "special" and ec != "nan" and actual != expected:
        failures.append("identity_mismatch")
    distance = (
        abs(rank(actual, fmt) - rank(expected, fmt))
        if ec in finite and ac in finite
        else None
    )
    if fmt in HALVES and ec != "nan" and actual not in ref["allowed"]:
        failures.append("half_membership")
    if fmt not in HALVES and distance is not None:
        limit = budget if budget is not None else (2 if fmt == "f64" else 4)
        if distance > limit:
            failures.append("step_budget")
    return dict(failures=failures, diagnostics=diagnostics, distance=distance)


def scalar_effective(variant, value, fmt):
    """value is F64 bits for Float, an integer for Int/UInt/Bool."""
    assert variant in ("Float", "Int", "UInt", "Bool")
    if variant == "Float":
        bits64 = value
    else:
        bits64 = encode(Q(int(value)), "f64")
    return (
        cast(bits64, "f64", fmt)
        if fmt not in HALVES
        else cast(cast(bits64, "f64", "f32"), "f32", fmt)
    )


def scalar_shortcut(variant, value):
    if variant == "Float":
        kind, q = decode(value, "f64")
        if kind != "finite":
            return None
    else:
        q = Q(int(value))
    return int(q) if q in (0, 1, 2, -1, -2) else None


def linux_status(variant, value, api="powf"):
    if variant == "Float":
        kind, q = decode(value, "f64")
        if api == "powf" and kind == "nan":
            return "generic"
        if kind != "finite":
            return "panic"
        if api == "powf" and q.denominator != 1:
            return "generic"
        # The inherited powi API converts nonintegral Float through i64 first.
        n = int(q)
        if not -(1 << 63) <= n < 1 << 63:
            return "panic"
        if api == "powf" or q.denominator == 1:
            return "shortcut" if n in (0, 1, 2, -1, -2) else "generic"
        return "truncated_dispatch"
    n = int(value)
    if variant == "UInt" and n >= 1 << 63:
        return "panic"
    return "shortcut" if n in (0, 1, 2, -1, -2) else "generic"


def primitive_mul_square(xbits, fmt):
    kind, x = decode(xbits, fmt)
    if kind == "nan":
        return None
    if kind == "inf":
        return info(fmt)[4]
    # The Native shortcut uses one RNE storage multiplication. This contract
    # does not borrow the approximation budget of mathematical Power.
    return encode(x * x, fmt)


def primitive_recip(xbits, fmt):
    if xbits is None:
        return None
    kind, x = decode(xbits, fmt)
    s = info(fmt)[3] if xbits & info(fmt)[3] else 0
    if kind == "nan":
        return None
    if kind == "inf":
        return s
    if x == 0:
        return s | info(fmt)[4]
    # Native uses destination division with explicit nearest-even rounding.
    # Linux parity remains subject to its separately qualified division route.
    return encode(1 / x, fmt)


def wrapper_reference(xbits, variant, value, fmt):
    shortcut = scalar_shortcut(variant, value)
    if shortcut is None:
        y = scalar_effective(variant, value, fmt)
        if y is None:
            y = info(fmt)[4] | (1 << (FORMATS[fmt][1] - 1))
        out = reference(xbits, y, fmt)
        return dict(out, shortcut=None, exponent=y)
    square = primitive_mul_square(xbits, fmt) if shortcut in (2, -2) else None
    result = (
        encode(Q(1), fmt)
        if shortcut == 0
        else xbits
        if shortcut == 1
        else square
        if shortcut == 2
        else primitive_recip(xbits if shortcut == -1 else square, fmt)
    )
    return dict(
        expected=result,
        square=square,
        shortcut=shortcut,
        method="stored_primitive_schedule",
    )


def qualify_wrapper(actual, fmt, ref):
    """Native stored RNE schedule. Never apply the Power approximation budget."""
    if ref["method"] != "stored_primitive_schedule":
        return qualify(actual, fmt, ref)
    expected = ref["expected"]
    ec, ac = classify(expected, fmt), classify(actual, fmt)
    failures = []
    if ec == "nan":
        if ac != "nan":
            failures.append("class_mismatch")
    elif actual != expected:
        failures.append("stored_schedule_mismatch")
    return dict(failures=failures, diagnostics=[], distance=None)


def edge_pool(fmt):
    e, f, bias, sign, inf = info(fmt)
    pool = set()
    for s in (0, sign):
        for b in (
            0,
            1,
            2,
            (1 << f) - 1,
            1 << f,
            (1 << f) + 1,
            inf - 2,
            inf - 1,
            inf,
            inf + 1,
            inf + (1 << (f - 1)),
            inf + (1 << f) - 1,
        ):
            pool.add(s | b)
    for q in (
        Q(1, 2),
        Q(1),
        Q(3, 2),
        Q(2),
        Q(3),
        Q(4),
        Q(1, 3),
        Q(10),
        pow2(f),
        pow2(f + 1),
        pow2(f + 2),
    ):
        for s in (-1, 1):
            b = encode(s * q, fmt)
            for d in (-1, 0, 1):
                if 0 <= b + d < 2 * sign:
                    pool.add(b + d)
    return sorted(pool)


def corpus(fmt):
    e, f, bias, sign, inf = info(fmt)
    pairs = set()
    if fmt in HALVES:
        for y in (Q(1, 2), Q(3)):
            ybits = encode(y, fmt)
            pairs.update((x, ybits) for x in range(65536))
    edge = edge_pool(fmt)
    pairs.update((x, y) for x in edge for y in edge)
    # Every finite normal binade, both signs, three significands, three varied
    # exponents. Noninteger negative-base domains remain explicit.
    ys = [Q(-3), Q(-1, 2), Q(1, 2), Q(3), Q(3, 2), Q(11), Q(-11)]
    for field in range(1, (1 << e) - 1):
        for frac in (0, 1, (1 << f) - 1):
            for s in (0, sign):
                x = s | field << f | frac
                for offset in (0, 2, 4):
                    y = ys[(field + frac + offset) % len(ys)]
                    pairs.add((x, encode(y, fmt)))
    # Near +-1, include very large exponents and parity thresholds.
    exponents = [pow2(k) for k in (f - 1, f, f + 1, f + 2, bias // 2, bias)]
    exponents += [pow2(f) - 1, pow2(f + 1) - 1]
    for d in range(-16, 17):
        for s in (0, sign):
            x = s | (bias << f) + d
            for y in exponents:
                for sy in (-1, 1):
                    pairs.add((x, encode(sy * y, fmt)))
    # Exact base-2 exponent neighborhoods at zero, min-normal and overflow.
    # Half stages also include F32 intermediate boundaries.
    for target in (fmt, "f32") if fmt in HALVES else (fmt,):
        _, tf, tbias, _, _ = info(target)
        for cut in (Q(-tbias - tf), Q(1 - tbias - tf), Q(1 - tbias), Q(tbias + 1)):
            for base, exponent in ((Q(2), cut), (Q(1, 2), -cut)):
                y = encode(exponent, fmt)
                for d in range(-8, 9):
                    if 0 <= y + d < 2 * sign:
                        pairs.add((encode(base, fmt), y + d))
    # Independent deterministic pairs. Flex32 deliberately shares F32 values.
    seed = ("f32" if fmt == "flex32" else fmt) + " stored Power v1"
    state = hashlib.sha256(seed.encode()).digest()
    for _ in range(1024):
        state = hashlib.sha256(state).digest()
        pairs.add(
            (
                int.from_bytes(state[:8], "little") % (2 * sign),
                int.from_bytes(state[8:16], "little") % (2 * sign),
            )
        )
    return sorted(pairs)


SCALARS = []


def add_scalar(kind, value):
    SCALARS.append((kind, value))


for n in (
    0,
    1,
    2,
    -1,
    -2,
    3,
    -3,
    (1 << 24) + 1,
    (1 << 53) + 1,
    (1 << 62) + (1 << 38) + 1,
    -(1 << 63),
    (1 << 63) - 1,
):
    add_scalar("Int", n)
for n in (
    0,
    1,
    2,
    3,
    (1 << 24) + 1,
    (1 << 53) + 1,
    (1 << 63) - 1,
    1 << 63,
    (1 << 64) - 1,
):
    add_scalar("UInt", n)
for n in (0, 1):
    add_scalar("Bool", n)
for q in (
    Q(0),
    Q(1),
    Q(2),
    Q(-1),
    Q(-2),
    Q(3),
    Q(-3),
    Q(1, 2),
    Q(-1, 2),
    Q(3, 2),
    Q(-3, 2),
    Q(5, 2),
    Q(-5, 2),
    Q((1 << 24) + 1),
    Q(10**18),
    Q(1 << 63),
    Q(-(1 << 63)),
    Q(1 << 64),
    Q(2) + pow2(-30),
    Q(1) + pow2(-11) + pow2(-30),
    Q(1) + pow2(-8) + pow2(-30),
):
    add_scalar("Float", encode(q, "f64"))
for b in (
    1 << 63,
    info("f64")[4],
    (1 << 63) | info("f64")[4],
    0x7FF8000000000000,
    0xFFF8000000000123,
    0x7FF0000000000123,
    info("f64")[4] - 1,
):
    add_scalar("Float", b)
SCHEDULES = (
    "Mathematical",
    "Ones",
    "Identity",
    "Square",
    "Reciprocal",
    "SquareReciprocal",
)
INTEGER_EXPONENTS = (
    -(1 << 63),
    -((1 << 24) + 1),
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    (1 << 24) + 1,
    (1 << 53) + 1,
    (1 << 62) + (1 << 38) + 1,
    (1 << 63) - 1,
)
FILES = (
    "os/library/ai/src/validation/typed_power_fixtures.rs",
    "os/library/ai/src/validation/typed_power_fixtures.bin",
    "tools/nvidia/typed-power-manifest.json",
)


def canonical_nan(fmt):
    return info(fmt)[4] | (1 << (FORMATS[fmt][1] - 1))


def word(bits, fmt):
    return canonical_nan(fmt) if bits is None else bits


def valid_powi(kind, value):
    if kind != "Float":
        return True
    category, q = decode(value, "f64")
    return category == "finite" and q.denominator == 1


def normalized(kind, value):
    return value if kind == "Float" else encode(Q(value), "f64")


def shortcut_schedule(kind, value):
    n = scalar_shortcut(kind, value)
    return {None: 0, 0: 1, 1: 2, 2: 3, -1: 4, -2: 5}[n]


def row_reference(x, y, fmt, schedule=0):
    ref = reference(x, y, fmt)
    ideal, staged = word(ref["ideal"], fmt), word(ref["expected"], fmt)
    center = word(ref["center"], "f32") if fmt in HALVES else 0
    exact = ref["method"] == "special"
    if schedule:
        exp = {1: 0, 2: 1, 3: 2, 4: -1, 5: -2}[schedule]
        wrapper = wrapper_reference(x, "Int", exp, fmt)
        staged = word(wrapper["expected"], fmt)
        allowed = [staged]
        exact = True
    elif fmt in HALVES:
        allowed = ref["allowed"] or [staged]
    else:
        allowed = [staged]
    allowed = sorted(allowed, key=lambda b: rank(b, fmt))
    # The nine-point image is contiguous in destination representable order.
    if len(allowed) > 1:
        assert rank(allowed[-1], fmt) - rank(allowed[0], fmt) + 1 == len(allowed)
    return (
        x,
        y,
        ideal,
        staged,
        allowed[0],
        allowed[-1],
        center,
        schedule,
        int(exact),
        ref["precision"],
    )


def permitted(row, fmt):
    _, _, _, expected, lo, hi, _, schedule, exact, _ = row
    if exact or classify(expected, fmt) in ("nan", "inf", "zero"):
        return [expected]
    if fmt in HALVES:
        choices = range(rank(lo, fmt), rank(hi, fmt) + 1)
    else:
        n = 2 if fmt == "f64" else 4
        choices = range(rank(expected, fmt) - n, rank(expected, fmt) + n + 1)
    out = []
    for r in choices:
        actual = unrank(r, fmt)
        if classify(actual, fmt) not in ("normal", "subnormal"):
            continue
        if bool(actual & info(fmt)[3]) != bool(expected & info(fmt)[3]):
            continue
        out.append(actual)
    return out


def build():
    payload = bytearray()
    ranges = []
    scalar_inputs = []
    effective = []
    integer_stored = []
    chains = []
    counts = {}
    checks = Counter()
    for fmt in FORMATS:
        table = {}

        def put(x, y, schedule=0):
            key = (schedule, x, y)
            if key not in table:
                table[key] = row_reference(x, y, fmt, schedule)
            return table[key]

        math_pairs = corpus(fmt)
        for x, y in math_pairs:
            put(x, y)
        bases = set(edge_pool(fmt))
        for q in (
            Q(-1),
            Q(2),
            Q(-2),
            1 - pow2(-53),
            pow2(520),
            pow2(65),
            pow2(8),
            pow2(67),
        ):
            bases.add(encode(q, fmt))
        bases = sorted(bases)
        scalar_inputs.append(bases)
        effects = []
        for kind, value in SCALARS:
            y = word(scalar_effective(kind, value, fmt), fmt)
            norm = normalized(kind, value)
            py = word(scalar_effective("Float", norm, fmt), fmt)
            schedule = shortcut_schedule(kind, value)
            ps = shortcut_schedule("Float", norm)
            effects.append((y, py, schedule, ps))
            for x in bases:
                put(x, y)
                put(x, y, schedule)
                put(x, py)
                put(x, py, ps)
        effective.append(effects)
        ints = [encode(Q(n), fmt) for n in INTEGER_EXPONENTS]
        integer_stored.append(ints)
        for y in ints:
            for x in bases:
                put(x, y)
        # Independent closure for materialized chain steps. Mathematical legal
        # outputs all get a following reference. Exact shortcut stages branch once.
        plans = []
        for seed in (Q(2), Q(-2), Q(1, 2)):
            initial = encode(seed, fmt)
            ys = [encode(Q(n), fmt) for n in (3, 2, -1)]
            schedules = [0, 3, 4]
            states = {initial}
            population = []
            for y, schedule in zip(ys, schedules):
                following = set()
                stage_classes = set()
                for x in states:
                    row = put(x, y, schedule)
                    stage_classes.add((classify(row[2], fmt), classify(row[3], fmt)))
                    following.update(permitted(row, fmt))
                assert stage_classes == {("normal", "normal")}, stage_classes
                population.append(len(states))
                states = following
            plans.append((initial, ys, schedules, population))
        chains.append(plans)
        rows = sorted(table.values(), key=lambda r: (r[7], r[0], r[1]))
        offset = len(payload)
        width = (sum(FORMATS[fmt]) + 1) // 8
        classes = Counter()
        schedules = Counter()
        precisions = Counter()
        schedule_classes = {}
        for row in rows:
            x, y, ideal, staged, lo, hi, center, schedule, exact, p = row
            for v in (x, y, ideal, staged, lo, hi):
                payload.extend(v.to_bytes(width, "little"))
            payload.extend(struct.pack("<IBB", center, schedule, exact))
            classes[classify(staged, fmt)] += 1
            schedules[SCHEDULES[schedule]] += 1
            precisions[str(p)] += 1
            schedule_classes.setdefault(SCHEDULES[schedule], Counter())[
                classify(staged, fmt)
            ] += 1
            # Internal center must pass exactly the same classification policy.
            if schedule:
                expected = wrapper_reference(
                    x, "Int", {1: 0, 2: 1, 3: 2, 4: -1, 5: -2}[schedule], fmt
                )
                assert not qualify_wrapper(expected["expected"], fmt, expected)[
                    "failures"
                ]
            else:
                ref = dict(
                    expected=staged, method="special" if exact else "directed_mpfr"
                )
                if fmt in HALVES:
                    ref["allowed"] = [
                        unrank(r, fmt) for r in range(rank(lo, fmt), rank(hi, fmt) + 1)
                    ]
                assert not qualify(staged, fmt, ref)["failures"]
            checks["center_policy"] += 1
        # Higher precision sample includes special/domain rows and both halves.
        for row in rows[:: max(1, len(rows) // 512)]:
            if row[7]:
                continue
            ref = reference(row[0], row[1], fmt, minimum=512)
            assert (
                word(ref["ideal"], fmt) == row[2]
                and word(ref["expected"], fmt) == row[3]
            )
            checks["higher_precision"] += 1
        ranges.append((offset, len(rows), len(math_pairs)))
        counts[fmt] = dict(
            offset_bytes=offset,
            record_bytes=6 * width + 6,
            rows=len(rows),
            core_mathematical_pairs=len(math_pairs),
            classes=dict(classes),
            schedules=dict(schedules),
            schedule_classes={k: dict(v) for k, v in schedule_classes.items()},
            precision=dict(precisions),
            scalar_inputs=len(bases),
            scalar_specs=len(SCALARS),
            chain_closure_populations=[p[3] for p in plans],
        )
        print(fmt, len(rows), flush=True)
    # Exact binary encoders round-trip every finite half encoding.
    for fmt in HALVES:
        for b in range(65536):
            category, q = decode(b, fmt)
            if category == "finite":
                assert encode(q, fmt, bool(b & info(fmt)[3])) == b
                checks["half_roundtrip"] += 1
    # Range-first classification protects against billion-bit Fraction allocation.
    for fmt in FORMATS:
        huge = info(fmt)[4] - 1
        for x, wanted in ((Q(2), info(fmt)[4]), (Q(1, 2), 0)):
            assert reference(encode(x, fmt), huge, fmt)["expected"] == wanted
            checks["range_gate"] += 1
    digest = hashlib.sha256(payload).hexdigest()
    policy = dict(
        f64_steps=2,
        f32_flex32_steps=4,
        half="destination RNE image of +/-4 F32 neighbors",
        classes="strict NaN/domain/sign/zero/nonzero/overflow; normal-subnormal transitions diagnostic within budget",
        wrappers="exact stored RNE schedules; mathematical ideal retained separately",
        nan="output classification except Identity byte alias",
        fast_math=False,
    )
    policy_hash = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    lines = [
        "// Generated by tools/nvidia/typed-power-generate.py. Do not edit.",
        "pub const POLICY_SHA256: &str = " + json.dumps(policy_hash) + ";",
        "pub const FIXTURE_SHA256: &str = " + json.dumps(digest) + ";",
        'const FIXTURE_BYTES: &[u8] = include_bytes!("typed_power_fixtures.bin");',
        "const RANGES: [(usize, usize, usize); 5] = "
        + repr(ranges).replace("[", "[", 1)
        + ";",
        "const SCALAR_SPECS: [ScalarSpec; " + str(len(SCALARS)) + "] = [",
    ]
    kinds = {"Float": 0, "Int": 1, "UInt": 2, "Bool": 3}
    for kind, value in SCALARS:
        norm = normalized(kind, value)
        flags = (
            valid_powi(kind, value),
            linux_status(kind, value) != "panic",
            valid_powi(kind, value) and linux_status(kind, value, "powi") != "panic",
            norm,
            valid_powi("Float", norm),
            linux_status("Float", norm, "powi") != "panic"
            and valid_powi("Float", norm),
            linux_status("Float", norm) != "panic",
        )
        names = (
            "valid_powi",
            "linux_powf",
            "linux_powi",
            "normalized",
            "public_powi_valid",
            "public_linux_powi",
            "public_linux_powf",
        )
        fields = ["kind:" + str(kinds[kind]), "value:" + str(value & ((1 << 64) - 1))]
        fields += [name + ":" + str(v).lower() for name, v in zip(names, flags)]
        lines.append("    ScalarSpec {" + ",".join(fields) + "},")
    lines.append("];")
    for name, data in (
        ("SCALAR_INPUTS", scalar_inputs),
        ("INTEGER_STORED", integer_stored),
    ):
        lines.append(
            "const "
            + name
            + ": [&[u64]; 5] = ["
            + ",".join("&" + repr(x) for x in data)
            + "];"
        )
    lines.append("const EFFECTIVE: [&[(u64,u64,Schedule,Schedule)]; 5] = [")
    for data in effective:
        lines.append(
            "    &["
            + ",".join(
                "("
                + str(a)
                + ","
                + str(b)
                + ",Schedule::"
                + SCHEDULES[c]
                + ",Schedule::"
                + SCHEDULES[d]
                + ")"
                for a, b, c, d in data
            )
            + "],"
        )
    lines.append("];")
    lines.append(
        "const INTEGER_EXPONENTS: [i64; "
        + str(len(INTEGER_EXPONENTS))
        + "] = "
        + repr(list(INTEGER_EXPONENTS))
        + ";"
    )
    lines.append("const CHAINS: [[ChainFixture; 3]; 5] = [")
    for plans in chains:
        lines.append("    [")
        for initial, ys, schedules, pop in plans:
            lines.append(
                "        ChainFixture { initial: "
                + str(initial)
                + ", rhs: "
                + repr(ys)
                + ", schedules: ["
                + ",".join("Schedule::" + SCHEDULES[s] for s in schedules)
                + "] },"
            )
        lines.append("    ],")
    lines.append("];")
    fixture = ("\n".join(lines) + "\n").encode()
    manifest = dict(
        schema=1,
        corpus_version=1,
        policy_sha256=policy_hash,
        fixture_bytes=len(payload),
        operation="Power",
        opcode=29,
        burn="0.22.1",
        cubecl="0.11.3",
        mpfr=VERSION().decode(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=digest,
        encoding="dtype-width little-endian lhs,rhs,ideal,staged,allowed_low,allowed_high; uint32 F32 center; uint8 schedule; uint8 exact-special",
        dtype_order=list(FORMATS),
        schedule_order=list(SCHEDULES),
        formats=counts,
        checks=dict(checks),
        policy=policy,
        provenance=dict(
            nvidia_url="https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#power-functions",
            nvidia_table=49,
            powf_ulp=4,
            pow_ulp=2,
            half_native=False,
            source_routes=[
                "burn-std/src/element/scalar.rs:Scalar::new/try_as_integer",
                "burn-backend/src/backend/ops/tensor.rs:float_powf_scalar/float_powi/float_powi_scalar",
                "burn-tensor/src/tensor/api/numeric.rs:powi_scalar",
                "cubecl-cpp/src/shared/binary.rs:Powf no_half",
            ],
            compiler_report="feasibility only; no numerical authority",
        ),
        numerical_backend_claim=False,
        gpu_executed=False,
        linux_extension_filter="--common-linux explicitly omits inherited backend panic/rejection probes; Native defaults retain them",
    )
    return {
        FILES[0]: fixture,
        FILES[1]: bytes(payload),
        FILES[2]: (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error("writes require --output-root")
    artifacts = build()
    root = args.output_root or Path(__file__).resolve().parents[2]
    for name, data in artifacts.items():
        path = root / name
        if args.check:
            if not path.exists() or path.read_bytes() != data:
                raise SystemExit("not reproducible: " + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(
        "typed Power fixtures reproduced"
        if args.check
        else "typed Power fixtures generated"
    )


if __name__ == "__main__":
    main()
