#!/usr/bin/env python3
"""Generate independent typed remainder fixtures with exact Fraction arithmetic.

Run with --check to verify committed bytes, or --mpfr for optional local residue
cross-checks. Python standard library only. No production arithmetic imports.
"""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import struct

@dataclass(frozen=True)
class Format:
    name: str
    ebits: int
    fbits: int

    @property
    def sign(self): return 1 << (self.ebits + self.fbits)
    @property
    def inf(self): return ((1 << self.ebits) - 1) << self.fbits
    @property
    def nan(self): return self.inf | (1 << (self.fbits - 1))
    @property
    def bias(self): return (1 << (self.ebits - 1)) - 1
    @property
    def minimum(self): return 1 - self.bias

F64 = Format('F64', 11, 52)
F32 = Format('F32', 8, 23)
FLEX = Format('Flex32', 8, 23)
F16 = Format('F16', 5, 10)
BF16 = Format('BF16', 8, 7)
FORMATS = (F64, F32, FLEX, F16, BF16)


def two(e):
    return Q(1 << e) if e >= 0 else Q(1, 1 << -e)

def decode(f, bits):
    """Class, sign, signed rational value. Zero sign is separate."""
    assert 0 <= bits < f.sign * 2
    negative = bool(bits & f.sign)
    exponent, fraction = divmod(bits & (f.sign - 1), 1 << f.fbits)
    if exponent == (1 << f.ebits) - 1:
        return ('nan' if fraction else 'inf', negative, Q(0))
    value = fraction * two(f.minimum - f.fbits)
    if exponent:
        value = ((1 << f.fbits) + fraction) * two(exponent - f.bias - f.fbits)
    return 'finite', negative, -value if negative else value

def encode(f, value, negative_zero=False):
    """Encode an exact rational by scaling integers, with gradual underflow."""
    value = Q(value)
    if not value:
        return f.sign if negative_zero else 0
    sign = f.sign if value < 0 else 0
    n, d = abs(value.numerator), value.denominator
    e = n.bit_length() - d.bit_length()
    if (n < d << e) if e >= 0 else (n << -e < d):
        e -= 1
    shift = max(e, f.minimum) - f.fbits
    numerator, denominator = (n, d << shift) if shift >= 0 else (n << -shift, d)
    rounded, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and rounded & 1):
        rounded += 1
    if e < f.minimum:
        return sign | rounded
    if rounded == 1 << (f.fbits + 1):
        rounded >>= 1
        e += 1
    if e > f.bias:
        return sign | f.inf
    return sign | ((e + f.bias) << f.fbits) | (rounded - (1 << f.fbits))

def convert(src, dst, bits):
    kind, sign, value = decode(src, bits)
    if kind != 'finite':
        return (dst.nan if kind == 'nan' else dst.inf) | (dst.sign if sign else 0)
    return encode(dst, value, sign)

def arithmetic(f, op, a, b):
    ka, sa, x = decode(f, a)
    kb, sb, y = decode(f, b)
    if 'nan' in (ka, kb):
        return f.nan
    if op == 'sub':
        return arithmetic(f, 'add', a, b ^ f.sign)
    if op == 'add':
        if 'inf' in (ka, kb):
            if ka == kb == 'inf' and sa != sb:
                return f.nan
            return f.inf | (f.sign if (sa if ka == 'inf' else sb) else 0)
        return encode(f, x + y, sa and sb)
    sign = f.sign if sa != sb else 0
    if op == 'mul':
        if (ka == 'inf' and kb == 'finite' and y == 0) or (kb == 'inf' and ka == 'finite' and x == 0):
            return f.nan
        return f.inf | sign if 'inf' in (ka, kb) else encode(f, x * y, sa != sb)
    assert op == 'div'
    if ka == kb == 'inf' or (ka == kb == 'finite' and x == y == 0):
        return f.nan
    if ka == 'inf' or (kb == 'finite' and y == 0):
        return f.inf | sign
    return sign if kb == 'inf' else encode(f, x / y, sa != sb)

def trunc(q):
    """Integer quotient toward zero, never a host floating quotient."""
    magnitude = abs(q.numerator) // q.denominator
    return -magnitude if q < 0 else magnitude

def raw_remainder(f, a, b):
    ka, sa, x = decode(f, a)
    kb, _, y = decode(f, b)
    if 'nan' in (ka, kb) or ka == 'inf' or (kb == 'finite' and y == 0):
        return f.nan
    if kb == 'inf':
        return a
    exact = x - trunc(x / y) * y
    bits = encode(f, exact, sa)
    assert decode(f, bits)[2] == exact, 'Same-format fmod residue must be exactly representable'
    return bits

def correction(f, residue, divisor):
    # NaN payload/sign choices are immaterial to the classified NaN result.
    if residue & (f.sign - 1) and bool((residue ^ divisor) & f.sign):
        return arithmetic(f, 'add', residue, divisor)
    return residue

def native(f, a, b, schedule='requested'):
    if f in (F16, BF16):
        x, y = convert(f, F32, a), convert(f, F32, b)
        r = raw_remainder(F32, x, y)
        if schedule == 'requested':
            return convert(F32, f, correction(F32, r, y))
        assert schedule == 'direct'
        return correction(f, convert(F32, f, r), b)
    return correction(f, raw_remainder(f, a, b), b)

def scalar_bits(f, kind, value):
    assert kind in ('Float', 'Int', 'UInt', 'Bool')
    if kind == 'Int': assert -(1 << 63) <= value < (1 << 63)
    if kind == 'UInt': assert 0 <= value < (1 << 64)
    if kind == 'Bool': assert value in (0, 1)
    first = value if kind == 'Float' else encode(F64, Q(value))
    if f == F64:
        return first
    second = convert(F64, F32, first)
    return second if f in (F32, FLEX) else convert(F32, f, second)

def floor_node(f, bits):
    kind, sign, q = decode(f, bits)
    if kind != 'finite' or not q:
        return bits
    return encode(f, Q(q.numerator // q.denominator), sign)

def linux_expression(f, a, b, fused=False):
    """A labeled diagnostic schedule, not a measured Linux GPU result."""
    quotient = arithmetic(f, 'div', a, b)
    k = floor_node(f, quotient)
    if not fused:
        return arithmetic(f, 'sub', a, arithmetic(f, 'mul', b, k))
    # Model final x - y*k with one RN, after rounded division and floor.
    kb, sb, y = decode(f, b)
    kk, sk, q = decode(f, k)
    ka, sa, x = decode(f, a)
    if ka == kb == kk == 'finite':
        return encode(f, x - y * q, sa and (sb == sk))
    return arithmetic(f, 'sub', a, arithmetic(f, 'mul', b, k))

def expectation(f, bits):
    kind, _, _ = decode(f, bits)
    return {'class': 'nan'} if kind == 'nan' else {'class': kind, 'bits': hx(f, bits)}

def hx(f, bits):
    return f'0x{bits:0{(f.ebits + f.fbits + 4) // 4}x}'

def equivalent(f, a, b):
    return a == b or decode(f, a)[0] == decode(f, b)[0] == 'nan'

def fixture_groups(f):
    one, three = encode(f, 1), encode(f, 3)
    normal = 1 << f.fbits
    maximum = f.inf - 1
    sign_pairs = [(encode(f, 5), three), (encode(f, 6), three), (one, three),
                  (one - 1, one), (one + 1, one), (three - 1, three), (three + 1, three)]
    groups = {'sign-and-multiples': sign_pairs}
    tiny = [1, 2, 3, normal - 1, normal, normal + 1]
    groups['subnormal-and-boundary'] = [(x, y) for x, y in (
        [(x, three) for x in tiny] + [(x, normal) for x in tiny] +
        [(normal, normal - 1), (normal + 1, normal - 1), (normal - 1, 1),
         (normal, 3), (normal + 1, 3), (three, 1), (maximum, normal - 1)])]
    groups['huge-quotient'] = [(maximum, 1), (maximum, 3), (maximum, normal - 1),
        (maximum, normal), (maximum, three), (one, 3), (one, normal - 1),
        (encode(f, two(f.bias)), encode(f, 3 * two(f.minimum - f.fbits)))]
    # Correct opposite signs in the sign expansion. r=y/2 and neighbors probe ties.
    midpoint = encode(f, two(-f.fbits - 2))
    groups['correction-rounding'] = [(x, one) for x in [
        midpoint - 1, midpoint + 1,
        encode(f, two(-f.fbits - 2)), encode(f, two(-f.fbits - 1)),
        encode(f, two(-f.fbits)), encode(f, Q(1, 2)) - 1,
        encode(f, Q(1, 2)), encode(f, Q(1, 2)) + 1, one - 1]]
    if f == F64:
        groups['f64-precision-range'] = [
            (one + 1, one), (one + 2, one + 1),
            (encode(f, two(900)) + 1, three),
            (encode(f, two(-900)) + 1, encode(f, 3 * two(-1000))),
            (maximum, encode(f, 3 * two(-1074))),
            (normal + 1, normal), (one, normal + 1)]
    result = []
    for group, pairs in groups.items():
        for a, b in pairs:
            for sa, sb in [(0, 0), (0, f.sign), (f.sign, 0), (f.sign, f.sign)]:
                result.append((group, a | sa, b | sb))
    # Full bounded Cartesian exception grid preserves both zero signs and NaN classes.
    edges = [0, f.sign, one, one | f.sign, 1, f.sign | 1,
             f.inf, f.inf | f.sign, f.nan | 5, f.sign | f.nan | 3,
             f.inf | 1, f.sign | f.inf | 1]
    result += [('ieee-exceptions', a, b) for a in edges for b in edges]
    return result

def scalar_specs(f):
    specs = [('Int', 0), ('Int', 1), ('Int', -3), ('Int', -(1 << 63)),
             ('Int', (1 << 63) - 1), ('UInt', 0), ('UInt', 1),
             ('UInt', (1 << 64) - 1), ('Bool', 0), ('Bool', 1),
             ('Int', (1 << 53) + 1), ('Int', (1 << 62) + (1 << 38) + 1),
             ('UInt', (1 << 63) + (1 << 39) + 1)]
    floats = [0, F64.sign, 1, F64.sign | 1, F64.inf, F64.inf | F64.sign,
              F64.nan | 0x123, F64.sign | F64.nan | 0x321, F64.inf | 1,
              encode(F64, 3), encode(F64, -3), encode(F64, 1) + 1,
              encode(F64, two(900)), encode(F64, two(-900)), F64.inf - 1,
              encode(F64, Q(10**100)), encode(F64, 1 + two(-40))]
    # Destination, F32 underflow/overflow, and normal-boundary midpoint neighbors.
    mids = [1 + two(-f.fbits - 1), two(f.minimum - f.fbits - 1),
            two(f.minimum) - two(f.minimum - f.fbits - 1),
            two(f.bias + 1) - two(f.bias - f.fbits - 1),
            two(-150), two(128) - two(103)]
    for mid in mids:
        bits = encode(F64, mid)
        if 0 < bits < F64.inf:
            floats += [bits - 1, bits, bits + 1]
    specs += [('Float', b) for b in dict.fromkeys(floats)]
    return specs

def self_check():
    count = 0
    for f in FORMATS:
        one = encode(f, 1)
        quantum = two(f.minimum - f.fbits)
        for bits in [0, f.sign, 1, f.sign | 1, (1 << f.fbits) - 1,
                     1 << f.fbits, one - 1, one, one + 1, f.inf - 1]:
            _, s, v = decode(f, bits)
            assert encode(f, v, s) == bits
            count += 1
        assert encode(f, 1 + two(-f.fbits - 1)) == one
        assert encode(f, 1 + 3 * two(-f.fbits - 1)) == one + 2
        assert encode(f, quantum / 2) == 0
        assert encode(f, -quantum / 2) == f.sign
        assert encode(f, 3 * quantum / 2) == 2
        assert encode(f, two(f.minimum) - quantum / 2) == 1 << f.fbits
        threshold = two(f.bias + 1) - two(f.bias - f.fbits - 1)
        assert encode(f, threshold) == f.inf
        assert encode(f, threshold - quantum) == f.inf - 1
        for a, b, r in [(5, 3, 2), (-5, 3, 1), (5, -3, -1), (-5, -3, -2)]:
            assert native(f, encode(f, a), encode(f, b)) == encode(f, r)
        assert native(f, f.sign, f.inf) == f.sign
        assert decode(f, native(f, encode(f, 1), f.inf | f.sign))[0] == 'inf'
        count += 14
    # F64 encoding and typed computation must not pass through F32.
    assert native(F64, encode(F64, 1) + 1, encode(F64, 1)) == encode(F64, two(-52))
    return count + 1

def mpfr_check(pairs):
    """Secondary fmod check with exact binary inputs and a wide MPFR exponent range."""
    library = ctypes.util.find_library('mpfr')
    if not library:
        return {'available': False, 'checked': 0}
    class Number(ctypes.Structure):
        _fields_ = [('precision', ctypes.c_long), ('sign', ctypes.c_int),
                    ('exponent', ctypes.c_long), ('limbs', ctypes.c_void_p)]
    lib = ctypes.CDLL(library)
    ptr = ctypes.POINTER(Number)
    lib.mpfr_init2.argtypes = [ptr, ctypes.c_long]
    lib.mpfr_clear.argtypes = [ptr]
    lib.mpfr_set_str.argtypes = [ptr, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.mpfr_fmod.argtypes = [ptr, ptr, ptr, ctypes.c_int]
    lib.mpfr_get_str.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_size_t, ptr, ctypes.c_int]
    lib.mpfr_get_str.restype = ctypes.c_void_p
    lib.mpfr_free_str.argtypes = [ctypes.c_void_p]
    lib.mpfr_zero_p.argtypes = [ptr]
    lib.mpfr_signbit.argtypes = [ptr]
    count, subnormal, zeros = 0, 0, 0
    for f, entries in pairs:
        x, y, z = Number(), Number(), Number()
        for item in [x, y, z]: lib.mpfr_init2(ctypes.byref(item), f.fbits + 1)
        try:
            for a, b in entries:
                ka, sa, va = decode(f, a)
                kb, sb, vb = decode(f, b)
                if ka != 'finite' or kb != 'finite' or not vb: continue
                for item, value, sign in [(x, va, sa), (y, vb, sb)]:
                    mag = abs(value)
                    text = ('-' if sign else '') + bin(mag.numerator)[2:] + '@' + str(-(mag.denominator.bit_length() - 1))
                    assert lib.mpfr_set_str(ctypes.byref(item), text.encode(), 2, 0) == 0
                # A finite same-format fmod residue is exactly representable, even subnormal.
                assert lib.mpfr_fmod(ctypes.byref(z), ctypes.byref(x), ctypes.byref(y), 0) == 0
                if lib.mpfr_zero_p(ctypes.byref(z)):
                    actual = f.sign if lib.mpfr_signbit(ctypes.byref(z)) else 0
                    zeros += 1
                else:
                    exponent = ctypes.c_long()
                    address = lib.mpfr_get_str(None, ctypes.byref(exponent), 2, 0, ctypes.byref(z), 0)
                    try: digits = ctypes.string_at(address).decode()
                    finally: lib.mpfr_free_str(address)
                    negative = digits.startswith('-')
                    digits = digits.lstrip('-')
                    value = Q(int(digits, 2)) * two(exponent.value - len(digits))
                    actual = encode(f, -value if negative else value)
                    subnormal += 0 < (actual & (f.sign - 1)) < 1 << f.fbits
                assert actual == raw_remainder(f, a, b), (f.name, hx(f, a), hx(f, b))
                count += 1
        finally:
            for item in [x, y, z]: lib.mpfr_clear(ctypes.byref(item))
    return {'available': True, 'library': library, 'checked': count,
            'nonzero_subnormal_results': subnormal, 'signed_zero_results': zeros,
            'scope': 'raw finite fmod only, exact MPFR result required; not a GPU check'}

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def half_traces():
    rows = []
    for f in (F16, BF16):
        one = encode(f, 1)
        for a, b in [(1, encode(f, -3)), (one - 1, one | f.sign),
                     (encode(f, two(-f.fbits - 2)) + 1, one | f.sign),
                     (f.inf - 1, encode(f, 11))]:
            x, y = convert(f, F32, a), convert(f, F32, b)
            r32 = raw_remainder(F32, x, y)
            rh = convert(F32, f, r32)
            rows.append({'format': f.name, 'a': hx(f, a), 'b': hx(f, b),
                'widened_a_f32': hx(F32, x), 'widened_b_f32': hx(F32, y),
                'residue_f32': hx(F32, r32), 'direct_residue_half': hx(f, rh),
                'direct_corrected_half': expectation(f, correction(f, rh, b)),
                'requested_corrected_f32': expectation(F32, correction(F32, r32, y)),
                'requested_final_half': expectation(f, native(f, a, b))})
    return rows

def scalar_normalization_change():
    rows = []
    value = (1 << 62) + (1 << 38) + 1
    for f in (F32, FLEX):
        lhs = encode(f, -3)
        canonical = scalar_bits(f, 'Int', value)
        legacy = encode(f, Q(value))
        correct, previous = native(f, lhs, canonical), native(f, lhs, legacy)
        assert canonical == 0x5e800000 and legacy == 0x5e800001
        assert correct == canonical and previous == legacy
        rows.append({'format': f.name, 'scalar_kind': 'Int', 'scalar_value': str(value),
            'lhs': hx(f, lhs), 'canonical_rhs': hx(f, canonical), 'legacy_rhs': hx(f, legacy),
            'expected': expectation(f, correct), 'legacy_result': expectation(f, previous)})
    return {'status': 'intentional scalar normalization correction, mathematical model only',
        'old_wrapper': 'backend/float.rs float_remainder_scalar -> rhs.elem::<f32>() -> scalar',
        'new_wrapper': 'float_remainder_scalar -> canonical float_scalar -> float_scalar_bits',
        'canonical_schedule': 'Scalar Int/UInt -> F64 -> F32, matching pinned Linux InputScalar',
        'unchanged': 'existing F32 GPU entry and remf plus RN non-FTZ correction arithmetic',
        'cases': rows}

def mutant_witnesses():
    """Concrete counterexamples to plausible but unauthorized schedules."""
    result = []
    def add(f, label, a, b, wrong):
        correct = native(f, a, b)
        assert not equivalent(f, correct, wrong), (f.name, label)
        result.append({'format': f.name, 'wrong_schedule': label, 'a': hx(f, a),
                       'b': hx(f, b), 'expected': expectation(f, correct),
                       'wrong_result': expectation(f, wrong)})
    for f in FORMATS:
        one, three = encode(f, 1), encode(f, 3)
        a, b = 1, three | f.sign
        add(f, 'omit opposite-sign correction', a, b, raw_remainder(f, a, b))
        add(f, 'flush subnormal input to signed zero', a, b, native(f, 0, b))
        a, b = encode(f, -6), three
        add(f, 'normalize exact-zero result to positive zero', a, b, 0)
        a, b = one - 1, one | f.sign
        add(f, 'return raw residue rather than cancellation correction', a, b, raw_remainder(f, a, b))
        a, b = f.inf - 1, encode(f, 11)
        qbits = arithmetic(f, 'div', a, b)
        kq, sq, q = decode(f, qbits)
        if kq == 'finite':
            wrong_raw = encode(f, decode(f, a)[2] - trunc(q) * decode(f, b)[2])
            add(f, 'round quotient before exact-looking reduction', a, b, correction(f, wrong_raw, b))
    for a, b in [(encode(F64, 1) + 1, encode(F64, 1)),
                 (encode(F64, two(900)), encode(F64, 3)),
                 (1, encode(F64, 3))]:
        wrong = convert(F32, F64, native(F32, convert(F64, F32, a), convert(F64, F32, b)))
        add(F64, 'narrow operands and computation to F32', a, b, wrong)
    for f in (F16, BF16):
        scalar = encode(F64, 1 + 3 * two(-f.fbits - 3))
        b = scalar_bits(f, 'Float', scalar)
        a = encode(f, 1)
        wide_b = convert(F64, F32, scalar)
        wide_a = convert(f, F32, a)
        wrong = convert(F32, f, correction(F32, raw_remainder(F32, wide_a, wide_b), wide_b))
        add(f, 'skip half scalar storage before F32 computation', a, b, wrong)
        result[-1]['scalar_f64_bits'] = hx(F64, scalar)
    return result

PINNED_SOURCE_SHA256 = {'burn-cubecl-0.22.1/src/kernel/binary.rs': '4b44ea79b1d24e08c63553c8be48adb6e05580a19356bc1b8b44cf3d9b02a12a', 'burn-cubecl-0.22.1/src/ops/tensor.rs': '01fb0bed638dbae21971b673eea0b9de655d750a1b87b253a8168f3850f8f246', 'burn-std-0.22.1/src/element/scalar.rs': '9d03a0325f47d6930cf858b0020a8683f0ecdd7a941270941d9bf3e364fa89c6', 'cubecl-core-0.11.3/src/frontend/scalar.rs': '26ce53adba57286e209a7b4e5f2dc03a1ea0e6bb6db04df979e10ce49a8c48d0', 'cubecl-cpp-0.11.3/src/shared/binary.rs': 'eadef2109d7bb1d1cf0822afc7848f297aa7e93f2e679bf97b24ff589b7c93b9', 'half-2.7.1/src/num_traits.rs': '4ec1ab5a2ace7afabcce4e1265753a5a309c04c28144888e3ce8f181d149d973', 'num-traits-0.2.19/src/cast.rs': 'd2c9337163c3d594a701bfce21aec56571625fa0117f970f14fd82dcc4504bf6'}
PINNED_SOURCE_SHA256.update({'burn-tensor-0.22.1/src/tensor/api/numeric.rs': 'b6e41a354bf335871e3eef6764bb1d9c2c1be248ada86bd8acafc0d9c8b0f78f', 'burn-tensor-0.22.1/src/bridge/ops/float.rs': 'c6805e2989f6a5d232d259cc61b68f1a8dda3d463c97323eaf6d6173dc52b007'})
REFERENCE_MODEL_SHA256 = '2ad0b9d0c2bba6515bb6d403dbd74202de76a8adaadfe53e2315f5441f83a837'


GROUPS = ('sign-and-multiples', 'subnormal-and-boundary', 'huge-quotient',
          'correction-rounding', 'f64-precision-range', 'ieee-exceptions')


def scalar_conversion_counterexamples():
    rows = []
    for f in FORMATS:
        inputs = [0, f.sign, 1, f.sign | 1, encode(f, 1), encode(f, 1) + 1,
                  encode(f, 3), encode(f, -3), (1 << f.fbits) + 1, f.inf - 1]
        for kind, value in scalar_specs(f):
            if kind == 'Float' and decode(F64, value)[0] != 'finite':
                continue
            exact = decode(F64, value)[2] if kind == 'Float' else Q(value)
            correct = scalar_bits(f, kind, value)
            direct = encode(f, exact, kind == 'Float' and bool(value & F64.sign))
            if correct != direct:
                rows.append({'format': f.name, 'kind': kind,
                    'value': hx(F64, value) if kind == 'Float' else str(value),
                    'correct_rhs': hx(f, correct), 'wrong_rhs': hx(f, direct),
                    'discriminating_lhs': [hx(f, a) for a in inputs
                        if not equivalent(f, native(f, a, correct), native(f, a, direct))],
                    'wrong_schedule': 'Scalar directly to destination'})
    assert len(rows) == 12
    return rows


def chain_rows(f):
    rows = []
    for initial, divisors in [(-5, [3, -3, 3]), (-6, [3, -3, 3]),
                              (Q(1, 2), [-1, 3, -1])]:
        current = encode(f, initial)
        rhs = [encode(f, v) for v in divisors]
        expected = []
        for b in rhs:
            current = native(f, current, b)
            expected.append(current)
        rows.append((encode(f, initial), rhs, expected))
    return rows


def generate():
    blob = bytearray()
    pair_ranges, scalar_ranges, counts, chains = [], [], {}, []
    for f in FORMATS:
        entries = fixture_groups(f)
        pair_ranges.append((len(blob), len(entries)))
        for group, a, b in entries:
            blob.extend(struct.pack('<QQQQ', a, b, native(f, a, b), GROUPS.index(group)))
        specs = scalar_specs(f)
        scalar_ranges.append((len(blob), len(specs)))
        inputs = [0, f.sign, 1, f.sign | 1, encode(f, 1), encode(f, 1) + 1,
                  encode(f, 3), encode(f, -3), (1 << f.fbits) + 1, f.inf - 1]
        for kind, value in specs:
            rhs = scalar_bits(f, kind, value)
            words = [('Float', 'Int', 'UInt', 'Bool').index(kind), value % (1 << 64), rhs]
            words += inputs + [native(f, a, rhs) for a in inputs]
            blob.extend(struct.pack('<' + 'Q' * len(words), *words))
        counts[f.name] = {'tensor_rows': len(entries), 'scalar_specs': len(specs),
                          'scalar_results': len(specs) * len(inputs),
                          'groups': {g: sum(r[0] == g for r in entries) for g in GROUPS},
                          'linux_unfused_diagnostic_differences': sum(
                              not equivalent(f, native(f, a, b), linux_expression(f, a, b))
                              for _, a, b in entries)}
        chains.append(chain_rows(f))
    fixture_hash = hashlib.sha256(blob).hexdigest()
    rust = '// Generated by tools/nvidia/typed-remainder-generate.py. Do not edit.\n'
    rust += f'pub const FIXTURE_SHA256: &str = "{fixture_hash}";\n'
    rust += 'const FIXTURE_BYTES: &[u8] = include_bytes!("typed_remainder_fixtures.bin");\n'
    rust += 'const PAIR_RANGES: [(usize, usize); 5] = '+str(pair_ranges)+';\n'
    rust += 'const SCALAR_RANGES: [(usize, usize); 5] = '+str(scalar_ranges)+';\n'
    rust += 'pub const GROUPS: [&str; 6] = ['+', '.join(json.dumps(g) for g in GROUPS)+'];\n'
    storage = []
    for index, f in [(3, F16), (4, BF16)]:
        source = encode(F64, 1 + 3 * two(-f.fbits - 3))
        rhs = scalar_bits(f, 'Float', source)
        lhs = encode(f, 1)
        storage.append((index, source, rhs, lhs, native(f, lhs, rhs)))
    rust += 'const STORAGE_CASES: [(usize, u64, u64, u64, u64); 2] = '+str(storage)+';\n'
    rust += 'const CHAINS: [[ChainFixture; 3]; 5] = [\n'
    for rows in chains:
        rust += '    [\n'
        for initial, rhs, expected in rows:
            rust += f'        ChainFixture {{ initial: {initial}, rhs: {rhs}, expected: {expected} }},\n'
        rust += '    ],\n'
    rust += '];\n'
    manifest = {'version': 1, 'corpus_version': 1, 'reference': 'independent Fraction truncation and integer IEEE RNE',
        'reference_model_sha256': REFERENCE_MODEL_SHA256,
        'source_dependencies_sha256': PINNED_SOURCE_SHA256,
        'generator_sha256': digest(Path(__file__)),
        'fixture_sha256': fixture_hash,
        'fixture_rust_sha256': hashlib.sha256(rust.encode()).hexdigest(),
        'fixture_bytes': len(blob), 'supplemental_storage_scalar_specs': 2, 'materialized_chain_cases': 15,
        'counts': counts, 'tensor_rows': 1608, 'scalar_specs': 221, 'scalar_results': 2210,
        'self_check_assertions': self_check(),
        'schedules': {'F64': 'F64 remf residue and RN correction',
            'F32/Flex32': 'F32 remf residue and RN correction',
            'F16/BF16': 'stored operands -> F32 remf -> F32 RN correction -> one destination RNE',
            'scalar': 'Scalar Int/UInt/Bool -> F64; Float retains F64; then F32 then half storage'},
        'strict_policy': 'finite, infinity, signed zero exact bits; output NaN classification; preserved sources exact bytes',
        'linux_expression': 'diagnostic only: x-y*floor(x/y), never the numerical oracle',
        'half_stage_traces': half_traces(), 'scalar_normalization_change': scalar_normalization_change(),
        'scalar_conversion_counterexamples': scalar_conversion_counterexamples(),
        'mutant_witnesses': mutant_witnesses()}
    assert sum(r['tensor_rows'] for r in counts.values()) == manifest['tensor_rows']
    assert sum(r['scalar_specs'] for r in counts.values()) == manifest['scalar_specs']
    return bytes(blob), rust.encode(), (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--mpfr', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = root / 'os/library/ai/src/validation'
    targets = [directory / 'typed_remainder_fixtures.bin', directory / 'typed_remainder_fixtures.rs',
               root / 'tools/nvidia/typed-remainder-manifest.json']
    for path, payload in zip(targets, generate()):
        if args.check:
            assert path.read_bytes() == payload, f'stale fixture: {path}'
        else:
            path.write_bytes(payload)
    if args.mpfr:
        print(json.dumps(mpfr_check([(f, [(a, b) for _, a, b in fixture_groups(f)]) for f in FORMATS]), sort_keys=True))
    print('typed remainder fixtures verified' if args.check else 'typed remainder fixtures generated')


if __name__ == '__main__':
    main()
