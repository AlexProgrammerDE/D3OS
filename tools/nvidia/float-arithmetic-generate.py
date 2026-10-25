#!/usr/bin/env python3
"""Generate bounded IEEE fixtures with exact rational arithmetic, without Burn.

Run from any directory. --check verifies checked-in artifacts. --mpfr adds a
local, optional MPFR cross-check through ctypes, without a Python dependency.
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
    exponent: int
    fraction: int

    @property
    def width(self):
        return 1 + self.exponent + self.fraction

    @property
    def sign(self):
        return 1 << (self.width - 1)

    @property
    def inf(self):
        return ((1 << self.exponent) - 1) << self.fraction

    @property
    def bias(self):
        return (1 << (self.exponent - 1)) - 1

    @property
    def nan(self):
        return self.inf | (1 << (self.fraction - 1))


F64, F32, F16, BF16 = FORMATS = (
    Format("F64", 11, 52), Format("F32", 8, 23),
    Format("F16", 5, 10), Format("BF16", 8, 7),
)
OPS = ("Add", "Sub", "Mul", "Div", "Neg", "Abs", "Recip", "Sign")
N = 32


def pow2(exponent):
    return Q(1 << exponent) if exponent >= 0 else Q(1, 1 << -exponent)


def decode(fmt, bits):
    """Return (class, sign, positive exact magnitude). Keep zero sign separate."""
    sign = bool(bits & fmt.sign)
    exponent = (bits & ~fmt.sign) >> fmt.fraction
    fraction = bits & ((1 << fmt.fraction) - 1)
    if exponent == (1 << fmt.exponent) - 1:
        return ("nan" if fraction else "inf", sign, Q(0))
    if exponent:
        value = Q((1 << fmt.fraction) + fraction) * pow2(exponent - fmt.bias - fmt.fraction)
    else:
        value = Q(fraction) * pow2(1 - fmt.bias - fmt.fraction)
    return ("finite", sign, value)


def nearest_even(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = remainder * 2
    return quotient + (twice > value.denominator or (twice == value.denominator and quotient & 1))


def encode(fmt, value, negative_zero=False):
    """Independent IEEE RNE encoder: integer quotient/remainder, no host floats."""
    if value == 0:
        return fmt.sign if negative_zero else 0
    sign = fmt.sign if value < 0 else 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < pow2(exponent):
        exponent -= 1
    minimum = 1 - fmt.bias
    quantum = pow2(max(exponent, minimum) - fmt.fraction)
    significand = nearest_even(value / quantum)
    if significand == 0:
        return sign
    if exponent < minimum:
        return sign | significand  # Carry at the subnormal boundary is intentional.
    if significand == 1 << (fmt.fraction + 1):
        significand >>= 1
        exponent += 1
    if exponent > fmt.bias:
        return sign | fmt.inf
    return sign | ((exponent + fmt.bias) << fmt.fraction) | (significand - (1 << fmt.fraction))


def convert(source, destination, bits):
    kind, sign, magnitude = decode(source, bits)
    if kind == "nan":
        return destination.nan | (destination.sign if sign else 0)
    if kind == "inf":
        return destination.inf | (destination.sign if sign else 0)
    return encode(destination, -magnitude if sign else magnitude, sign)


def arithmetic(fmt, op, a, b=0, intermediate=None):
    ka, sa, ma = decode(fmt, a)
    kb, sb, mb = decode(fmt, b)
    if op == "Neg":
        return a ^ fmt.sign
    if op == "Abs":
        return a & ~fmt.sign
    if op == "Sign":
        return 0 if ka == "nan" or (ka == "finite" and ma == 0) else encode(fmt, Q(-1 if sa else 1))
    if op == "Recip":
        return arithmetic(fmt, "Div", encode(fmt, Q(1)), a, intermediate)
    if ka == "nan" or kb == "nan":
        return fmt.nan
    if op == "Sub":
        sb = not sb
        op = "Add"
    if op == "Add":
        if ka == "inf" or kb == "inf":
            if ka == kb == "inf" and sa != sb:
                return fmt.nan
            return fmt.inf | (fmt.sign if (sa if ka == "inf" else sb) else 0)
        result = (-ma if sa else ma) + (-mb if sb else mb)
        negative_zero = sa and sb
    elif op == "Mul":
        if (ka == "inf" and kb == "finite" and mb == 0) or (kb == "inf" and ka == "finite" and ma == 0):
            return fmt.nan
        if ka == "inf" or kb == "inf":
            return fmt.inf | (fmt.sign if sa != sb else 0)
        result, negative_zero = ma * mb, sa != sb
        if negative_zero:
            result = -result
    elif op == "Div":
        if ka == kb == "inf" or (ka == kb == "finite" and ma == mb == 0):
            return fmt.nan
        sign = fmt.sign if sa != sb else 0
        if ka == "inf" or (kb == "finite" and mb == 0):
            return sign | fmt.inf
        if kb == "inf":
            return sign
        result, negative_zero = ma / mb, sa != sb
        if negative_zero:
            result = -result
    else:
        raise AssertionError(op)
    if intermediate is not None:
        return convert(intermediate, fmt, encode(intermediate, result, negative_zero))
    return encode(fmt, result, negative_zero)


def pool(fmt):
    one = encode(fmt, Q(1))
    k = {F64: 27, F32: 13, F16: 6, BF16: 5}[fmt]
    minimum = 1 << fmt.fraction
    values = [encode(fmt, Q(i), negative_zero=(i == 0 and negative))
              for i, negative in [(0, False), (0, True), (1, False), (-1, True), (2, False), (-2, True), (3, False), (-3, True)]]
    values += [one + 1, one - 1, encode(fmt, pow2(-fmt.fraction - 1)), encode(fmt, -pow2(-fmt.fraction - 1)),
               1, fmt.sign | 1, 2, fmt.sign | 2,
               minimum - 1, fmt.sign | (minimum - 1), minimum, fmt.sign | minimum,
               fmt.inf - 1, fmt.sign | (fmt.inf - 1), fmt.inf, fmt.sign | fmt.inf,
               fmt.nan | 0x15, fmt.sign | fmt.nan | 0x23, fmt.inf | 1, fmt.sign | fmt.inf | 3,
               encode(fmt, 1 + pow2(-k)), encode(fmt, 1 - pow2(-k)), encode(fmt, Q(1, 3)), encode(fmt, Q(-1, 3))]
    assert len(values) == N and len(set(values)) == N
    return values


def scalars(fmt):
    early = 1 + 3 * pow2(-{F16: 13, BF16: 10}.get(fmt, 20))
    values = [("Int", 0), ("Int", 1), ("UInt", 0), ("UInt", 1), ("Bool", 0), ("Bool", 1),
              ("Float", encode(F64, early)), ("Float", F64.sign), ("Float", F64.inf),
              ("Float", F64.sign | F64.inf), ("Float", F64.nan | 0x123), ("Float", encode(F64, Q(1)))]
    if fmt == F64:
        values += [("Float", encode(F64, pow2(-40))), ("Float", encode(F64, 1 + pow2(-40))),
                   ("Float", encode(F64, Q(10**100))), ("Float", 1), ("Int", 2**53 + 1), ("UInt", 2**64 - 1)]
    elif fmt == F32:
        values += [("Int", 2**62 + 2**38 + 1)]
    else:
        midpoint = encode(F64, 1 + pow2(-{F16: 11, BF16: 8}[fmt]))
        values += [("Float", midpoint - 1), ("Float", midpoint + 1)]
    return values


def normalize(fmt, scalar):
    kind, value = scalar
    first = value if kind == "Float" else encode(F64, Q(value))
    if fmt == F64:
        return first
    second = convert(F64, F32, first)
    return second if fmt == F32 else convert(F32, fmt, second)


def chain_fixtures(fmt):
    p = pool(fmt)
    # Each destination is an independently rounded and stored public primitive.
    # The third chain uses an early-rounded scalar before materialization.
    descriptions = [
        (p[28], [("Mul", p[29]), ("Add", p[3])]),
        (p[2], [("Add", p[10]), ("Sub", p[2]), ("Neg", 0)]),
        (p[6], [("Mul", normalize(fmt, scalars(fmt)[6])), ("Div", p[6]), ("Recip", 0)]),
    ]
    result = []
    for initial, steps in descriptions:
        current, entries = initial, []
        for op, rhs in steps:
            current = arithmetic(fmt, op, current, rhs)
            entries.append((op, rhs, current))
        result.append((initial, entries))
    assert result[0][1][0][2] == encode(fmt, Q(1))
    assert result[0][1][1][2] == 0
    return result


def self_test():
    for fmt in FORMATS:
        one, ulp = encode(fmt, Q(1)), pow2(-fmt.fraction)
        assert encode(fmt, 1 + ulp / 2) == one
        assert encode(fmt, 1 + 3 * ulp / 2) == one + 2
        assert encode(fmt, -(1 + ulp / 2)) == one | fmt.sign
        minsub = pow2(1 - fmt.bias - fmt.fraction)
        assert encode(fmt, minsub / 2) == 0
        assert encode(fmt, -minsub / 2) == fmt.sign
        assert encode(fmt, 3 * minsub / 2) == 2
        assert encode(fmt, ((1 << fmt.fraction) - Q(1, 2)) * minsub) == 1 << fmt.fraction
        threshold = pow2(fmt.bias + 1) - pow2(fmt.bias - fmt.fraction - 1)
        assert encode(fmt, threshold) == fmt.inf
        assert encode(fmt, threshold - minsub) == fmt.inf - 1
        assert arithmetic(fmt, "Sub", one, one) == 0
        assert arithmetic(fmt, "Add", fmt.sign, fmt.sign) == fmt.sign
        assert arithmetic(fmt, "Div", one, fmt.sign) == fmt.sign | fmt.inf
        assert arithmetic(fmt, "Div", one, fmt.sign | fmt.inf) == fmt.sign
        assert arithmetic(fmt, "Div", one, encode(fmt, Q(3))) == encode(fmt, Q(1, 3))
        assert arithmetic(fmt, "Sign", fmt.nan) == 0
    wide = ("Int", 2**62 + 2**38 + 1)
    assert normalize(F32, wide) != encode(F32, Q(wide[1]))
    for fmt in [F16, BF16]:
        assert normalize(fmt, scalars(fmt)[6]) == encode(fmt, Q(1))
        assert arithmetic(fmt, "Mul", encode(fmt, Q(3)), normalize(fmt, scalars(fmt)[6])) != encode(fmt, Q(3) * decode(F64, scalars(fmt)[6][1])[2])
        assert normalize(fmt, scalars(fmt)[12]) == normalize(fmt, scalars(fmt)[13])
        assert convert(F64, fmt, scalars(fmt)[13][1]) != normalize(fmt, scalars(fmt)[13])


def mpfr_check():
    """Check all finite binary pool pairs at target precision, exact exponent.

    MPFR computes subnormal results with a second rounding only here, so compare
    normal results directly; subnormal encoding is covered by rational tests.
    MPFR's huge exponent range also makes overflow a separate threshold test.
    """
    class Number(ctypes.Structure):
        _fields_ = [("precision", ctypes.c_long), ("sign", ctypes.c_int), ("exponent", ctypes.c_long), ("limbs", ctypes.c_void_p)]
    library = ctypes.util.find_library("mpfr")
    if not library:
        raise RuntimeError("--mpfr requested but local MPFR is unavailable")
    lib = ctypes.CDLL(library)
    ptr = ctypes.POINTER(Number)
    lib.mpfr_init2.argtypes = [ptr, ctypes.c_long]
    lib.mpfr_clear.argtypes = [ptr]
    lib.mpfr_set_str.argtypes = [ptr, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.mpfr_get_str.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long), ctypes.c_int, ctypes.c_size_t, ptr, ctypes.c_int]
    lib.mpfr_get_str.restype = ctypes.c_void_p
    lib.mpfr_free_str.argtypes = [ctypes.c_void_p]
    for op in OPS[:4]:
        getattr(lib, "mpfr_" + op.lower()).argtypes = [ptr, ptr, ptr, ctypes.c_int]
    checked = 0
    for fmt in FORMATS:
        a, b, out = Number(), Number(), Number()
        for item in [a, b, out]:
            lib.mpfr_init2(ctypes.byref(item), fmt.fraction + 1)
        try:
            for x in pool(fmt):
                for y in pool(fmt):
                    kx, sx, mx = decode(fmt, x)
                    ky, sy, my = decode(fmt, y)
                    if kx != "finite" or ky != "finite" or not mx or not my:
                        continue
                    for item, magnitude, sign in [(a, mx, sx), (b, my, sy)]:
                        # All IEEE inputs are dyadic: MPFR's binary @ exponent syntax.
                        text = ("-" if sign else "") + bin(magnitude.numerator)[2:] + "@" + str(-(magnitude.denominator.bit_length() - 1))
                        assert lib.mpfr_set_str(ctypes.byref(item), text.encode(), 2, 0) == 0
                    for op in OPS[:4]:
                        expected = arithmetic(fmt, op, x, y)
                        kind, _, magnitude = decode(fmt, expected)
                        if kind != "finite" or magnitude < pow2(1 - fmt.bias):
                            continue
                        getattr(lib, "mpfr_" + op.lower())(ctypes.byref(out), ctypes.byref(a), ctypes.byref(b), 0)
                        exponent = ctypes.c_long()
                        address = lib.mpfr_get_str(None, ctypes.byref(exponent), 2, 0, ctypes.byref(out), 0)
                        try:
                            digits = ctypes.string_at(address).decode()
                        finally:
                            lib.mpfr_free_str(address)
                        negative = digits.startswith("-")
                        digits = digits.lstrip("-")
                        exact = int(digits, 2) * pow2(exponent.value - len(digits))
                        assert encode(fmt, -exact if negative else exact) == expected, (fmt, op, x, y)
                        checked += 1
        finally:
            for item in [a, b, out]:
                lib.mpfr_clear(ctypes.byref(item))
    return checked


def artifacts():
    data, descriptors, chains, scalar_specs = bytearray(), [], [], []
    emulation_checks = 0
    for fmt in FORMATS:
        values, specs = pool(fmt), scalars(fmt)
        normalized = [normalize(fmt, s) for s in specs]
        offset = len(data)

        def append(bits):
            data.extend(bits.to_bytes(fmt.width // 8, "little"))

        for x in values:
            append(x)
        for op in OPS[:4]:
            for a in values:
                for b in values:
                    expected = arithmetic(fmt, op, a, b)
                    append(expected)
                    if fmt in [F16, BF16]:
                        emulated = arithmetic(fmt, op, a, b, F32)
                        assert expected == emulated, (fmt, op, a, b, expected, emulated)
                        emulation_checks += 1
        for op in OPS[4:]:
            for a in values:
                append(arithmetic(fmt, op, a))
        for x in normalized:
            append(x)
        for scalar in normalized:
            for op in OPS[:4]:
                for a in values:
                    expected = arithmetic(fmt, op, a, scalar)
                    append(expected)
                    if fmt in [F16, BF16]:
                        assert expected == arithmetic(fmt, op, a, scalar, F32)
                        emulation_checks += 1
        descriptors.append(f"    Fixture {{ offset: {offset}, scalar_count: {len(specs)} }},")
        scalar_specs.append("&[" + ", ".join(
            f"Scalar::Float(f64::from_bits(0x{value:016x}))" if kind == "Float" else
            f"Scalar::Bool({'true' if value else 'false'})" if kind == "Bool" else f"Scalar::{kind}({value})"
            for kind, value in specs) + "]")
        rows = []
        for chain_index, (initial, steps) in enumerate(chain_fixtures(fmt)):
            items = ", ".join(f"ChainStep {{ operation: Operation::{op}, rhs: 0x{rhs:x}, expected: 0x{expected:x} }}" for op, rhs, expected in steps)
            fused = "None"
            if chain_index == 0:
                # Exact a*b+c is deliberately separate from the stored product.
                _, _, a = decode(fmt, initial)
                _, _, b = decode(fmt, steps[0][1])
                fused_bits = encode(fmt, a * b - 1)
                assert fused_bits != steps[1][2]
                fused = f"Some(0x{fused_bits:x})"
            rows.append(f"ChainFixture {{ initial: 0x{initial:x}, fused: {fused}, steps: &[{items}] }}")
        chains.append("&[" + ",\n".join(rows) + "]")
    digest = hashlib.sha256(data).hexdigest()
    source = "// Generated by tools/nvidia/float-arithmetic-generate.py. Do not edit.\n"
    source += f'pub const FIXTURE_SHA256: &str = "{digest}";\n'
    source += 'const FIXTURE_BYTES: &[u8] = include_bytes!("float_arithmetic_fixtures.bin");\n'
    source += "const FIXTURES: [Fixture; 4] = [\n" + "\n".join(descriptors) + "\n];\n"
    source += "const SCALARS: [&[Scalar]; 4] = [\n" + ",\n".join(scalar_specs) + "\n];\n"
    source += "const CHAINS: [&[ChainFixture]; 4] = [\n" + ",\n".join(chains) + "\n];\n"
    manifest = {
        "version": 1, "oracle": "Python exact integers/Fraction; independent IEEE RNE encoder",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": digest, "fixture_bytes": len(data), "pool_values_per_format": N,
        "formats": [fmt.name for fmt in FORMATS], "Flex32": "reuses F32 values, retains separate dtype cases",
        "layout": "per format: pool[32], binary[4][32][32], unary[4][32], scalar_bits[S], scalar[ S ][4][32]; native-width little endian",
        "half_ideal_vs_f32_intermediate_checks": emulation_checks,
        "maximum_logical_cases": 900, "base_cases": 760, "maximum_output_elements": 257,
        "nan_policy": "classification for arithmetic; stored raw source bytes remain exact",
    }
    root = Path(__file__).resolve().parents[2]
    shared = root / "os/library/ai/src/validation"
    return {shared / "float_arithmetic_fixtures.bin": bytes(data),
            shared / "float_arithmetic_fixtures.rs": source.encode(),
            Path(__file__).with_name("float-arithmetic-manifest.json"): (json.dumps(manifest, indent=2) + "\n").encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--mpfr", action="store_true")
    args = parser.parse_args()
    self_test()
    checked = mpfr_check() if args.mpfr else 0
    for path, content in artifacts().items():
        if args.check:
            assert path.read_bytes() == content, f"stale fixture: {path}"
        else:
            path.write_bytes(content)
    print(json.dumps({"oracle_tests": "passed", "mpfr_normal_results_checked": checked, "artifacts": "verified" if args.check else "generated"}))


if __name__ == "__main__":
    main()
