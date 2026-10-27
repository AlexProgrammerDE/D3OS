"""Selected C3 reference only. Exact integers/Fraction; no floats or prior-model import."""
from fractions import Fraction as Q
from pathlib import Path
from itertools import product
import hashlib, json, os, resource, time

P = Path(__file__).parent
FORMATS = {'F16': (5, 10), 'BF16': (8, 7), 'Flex32': (8, 23), 'F64': (11, 52)}


def pow2(e):
    return Q(1 << e) if e >= 0 else Q(1, 1 << -e)


class Format:
    def __init__(self, eb, fb):
        self.eb, self.fb = eb, fb
        self.sign = 1 << (eb + fb)
        self.inf = ((1 << eb) - 1) << fb
        self.quiet = 1 << (fb - 1)
        self.canonical = self.inf | self.quiet
        self.bias = (1 << (eb - 1)) - 1

    def kind(self, b):
        magnitude = b & (self.sign - 1)
        if magnitude > self.inf:
            return 'nan'
        if magnitude == self.inf:
            return 'inf'
        return 'zero' if magnitude == 0 else 'finite'

    def value(self, b):
        assert self.kind(b) in ('finite', 'zero')
        field, fraction = divmod(b & (self.sign - 1), 1 << self.fb)
        v = Q(fraction + ((1 << self.fb) if field else 0)) * pow2((field or 1) - self.bias - self.fb)
        return -v if b & self.sign else v

    def encode(self, x, zero_sign=0):
        # Model A: search monotonically ordered positive raw encodings.
        # Overflow compares against the exact midpoint to the virtual 2^(emax+1).
        if x == 0:
            return zero_sign * self.sign
        s = self.sign if x < 0 else 0
        x = abs(x)
        lo, hi = 0, self.inf - 1
        while lo < hi:
            middle = (lo + hi + 1) // 2
            if self.value(middle) <= x:
                lo = middle
            else:
                hi = middle - 1
        if lo == self.inf - 1:
            boundary = (self.value(lo) + pow2(self.bias + 1)) / 2
            return s | (self.inf if x >= boundary else lo)
        distance_low = x - self.value(lo)
        distance_high = self.value(lo + 1) - x
        return s | (lo + int(distance_high < distance_low or (distance_high == distance_low and lo & 1)))


F32 = Format(8, 23)


def classify_b(b, f):
    # Independent extraction and class predicates, no Format.kind/value calls.
    sign = b >> (f.eb + f.fb)
    exponent = (b >> f.fb) & ((1 << f.eb) - 1)
    mantissa = b & ((1 << f.fb) - 1)
    if exponent == (1 << f.eb) - 1:
        return ('n' if mantissa else 'i'), sign, 0, 0
    if exponent:
        mantissa += 1 << f.fb
    return ('z' if not mantissa else 'f'), sign, mantissa, (exponent or 1) - ((1 << (f.eb - 1)) - 1) - f.fb


def round_b(m, e, f, zero_sign=0):
    # Model B: integer lattice quantization, with discarded-bit RNE.
    if m == 0:
        return zero_sign << (f.eb + f.fb)
    sign = int(m < 0) << (f.eb + f.fb)
    n = abs(m)
    bias = (1 << (f.eb - 1)) - 1
    unit = max(1 - bias, n.bit_length() - 1 + e) - f.fb
    discard = unit - e
    if discard > 0:
        kept = n >> discard
        remainder = n - (kept << discard)
        half = 1 << (discard - 1)
        kept += int(remainder > half or (remainder == half and kept & 1))
    else:
        kept = n << -discard
    if kept.bit_length() > f.fb + 1:
        kept >>= 1
        unit += 1
    exponent = unit + f.fb
    if exponent > bias:
        return sign | (((1 << f.eb) - 1) << f.fb)
    if kept < 1 << f.fb:
        return sign | kept
    return sign | ((exponent + bias) << f.fb) | (kept - (1 << f.fb))


def convert_a(b, src, dst):
    kind = src.kind(b)
    if kind == 'nan':
        return dst.canonical
    if kind == 'inf':
        return dst.inf | (dst.sign if b & src.sign else 0)
    return dst.encode(src.value(b), int(bool(b & src.sign)))


def convert_b(b, src, dst):
    kind, sign, m, e = classify_b(b, src)
    if kind == 'n':
        return (((1 << dst.eb) - 1) << dst.fb) | (1 << (dst.fb - 1))
    if kind == 'i':
        return (((1 << dst.eb) - 1) << dst.fb) | (sign << (dst.eb + dst.fb))
    return round_b(-m if sign else m, e, dst, sign)


def arithmetic_a(a, b, op, f):
    ka, kb = f.kind(a), f.kind(b)
    sa, sb = int(bool(a & f.sign)), int(bool(b & f.sign))
    if 'nan' in (ka, kb):
        return f.canonical
    if op == 'Add':
        if ka == kb == 'inf' and sa != sb:
            return f.canonical
        if ka == 'inf' or kb == 'inf':
            return a if ka == 'inf' else b
        return f.encode(f.value(a) + f.value(b), int(ka == kb == 'zero' and sa == sb == 1))
    if (ka == 'inf' and kb == 'zero') or (kb == 'inf' and ka == 'zero'):
        return f.canonical
    if 'inf' in (ka, kb):
        return f.inf | ((sa ^ sb) * f.sign)
    return f.encode(f.value(a) * f.value(b), sa ^ sb)


def arithmetic_b(a, b, op, f):
    ka, sa, ma, ea = classify_b(a, f)
    kb, sb, mb, eb = classify_b(b, f)
    nan = (((1 << f.eb) - 1) << f.fb) | (1 << (f.fb - 1))
    if ka == 'n' or kb == 'n':
        return nan
    if op == 'Mul':
        if {ka, kb} == {'z', 'i'}:
            return nan
        if ka == 'i' or kb == 'i':
            return (((1 << f.eb) - 1) << f.fb) | ((sa ^ sb) << (f.eb + f.fb))
        return round_b((-1 if sa ^ sb else 1) * ma * mb, ea + eb, f, sa ^ sb)
    if ka == 'i':
        return nan if kb == 'i' and sa != sb else a
    if kb == 'i':
        return b
    bottom = min(ea, eb)
    exact = ((-ma if sa else ma) << (ea - bottom)) + ((-mb if sb else mb) << (eb - bottom))
    return round_b(exact, bottom, f, sa & sb if ka == kb == 'z' else 0)


def minmax_a(a, b, op, f):
    ka, kb = f.kind(a), f.kind(b)
    if ka == kb == 'nan':
        return a | f.quiet
    if ka == 'nan':
        return b
    if kb == 'nan':
        return a
    if ka == kb == 'zero':
        return (a | b) if op == 'Min' else (a & b)
    if a == b:
        return a
    if ka == 'inf':
        less = bool(a & f.sign)
    elif kb == 'inf':
        less = not bool(b & f.sign)
    else:
        less = f.value(a) < f.value(b)
    return (a if less else b) if op == 'Min' else (b if less else a)


def minmax_b(a, b, op, f):
    ka, _, _, _ = classify_b(a, f)
    kb, _, _, _ = classify_b(b, f)
    if ka == 'n' or kb == 'n':
        return (a | (1 << (f.fb - 1))) if ka == kb == 'n' else (b if ka == 'n' else a)
    # IEEE monotone bit key independently handles signed zero and infinities.
    mask = (1 << (1 + f.eb + f.fb)) - 1
    def key(x):
        return (~x & mask) if x >> (f.eb + f.fb) else (x ^ (1 << (f.eb + f.fb)))
    if key(a) == key(b):
        return a
    return (a if key(a) < key(b) else b) if op == 'Min' else (a if key(a) > key(b) else b)


def step(a, b, op, f, independent=False):
    if op in ('Min', 'Max'):
        return (minmax_b if independent else minmax_a)(a, b, op, f)
    arith = arithmetic_b if independent else arithmetic_a
    cvt = convert_b if independent else convert_a
    compute = F32 if f.fb in (7, 10) else f
    return cvt(arith(cvt(a, f, compute), cvt(b, f, compute), op, compute), compute, f)


def fold(base, updates, op, f, independent=False):
    # Absence of updates deliberately performs no conversion/classification.
    for b in updates:
        base = step(base, b, op, f, independent)
    return base


def final_store(base, updates, op, f):
    compute = F32 if f.fb in (7, 10) else f
    a = convert_a(base, f, compute)
    for b in updates:
        a = arithmetic_a(a, convert_a(b, f, compute), op, compute)
    return convert_a(a, compute, f)


def check_equal(actual, expected):
    assert actual == expected, (hex(actual), hex(expected))

