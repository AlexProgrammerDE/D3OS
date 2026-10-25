#!/usr/bin/env python3
"""Generate exact unary float fixtures. --check verifies; --mpfr cross-checks.

Uses the frozen arithmetic encoder through an explicit import, never production
or host floating arithmetic. Sqrt decisions compare exact squared midpoints.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from fractions import Fraction as Q
import hashlib
import importlib.util
import json
from math import isqrt
from pathlib import Path
import sys

ENCODER_PATH = Path(__file__).with_name("float-arithmetic-generate.py")
spec = importlib.util.spec_from_file_location("float_arithmetic_encoder", ENCODER_PATH)
encoder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = encoder
spec.loader.exec_module(encoder)
F64, F32, F16, BF16 = FORMATS = encoder.FORMATS
decode, encode, pow2, convert = encoder.decode, encoder.encode, encoder.pow2, encoder.convert
OPS = ("Sqrt", "Floor", "Ceil", "Round", "Trunc")
N = 64


def rounded_root(ratio):
    """Nearest integer to sqrt(A/B), with ties choosing an even integer."""
    a, b = ratio.numerator, ratio.denominator
    r = isqrt(a // b)
    assert r * r * b <= a < (r + 1) ** 2 * b
    midpoint = b * (2 * r + 1) ** 2
    return r + (4 * a > midpoint or (4 * a == midpoint and r & 1))


def sqrt_rational(fmt, magnitude):
    assert magnitude > 0
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < pow2(exponent):
        exponent -= 1
    quantum_exponent = max(exponent // 2 - fmt.fraction, 1 - fmt.bias - fmt.fraction)
    quantum = pow2(quantum_exponent)
    return encode(fmt, rounded_root(magnitude / (quantum * quantum)) * quantum)


def unary(fmt, op, bits):
    kind, sign, magnitude = decode(fmt, bits)
    if kind == "nan":
        return fmt.nan if op == "Sqrt" else bits
    if op == "Sqrt":
        if kind == "finite" and magnitude == 0:
            return bits
        if sign:
            return fmt.nan
        if kind == "inf":
            return bits
        return sqrt_rational(fmt, magnitude)
    if kind == "inf":
        return bits
    whole, remainder = divmod(magnitude.numerator, magnitude.denominator)
    denominator = magnitude.denominator
    increment = {
        "Floor": sign and remainder != 0,
        "Ceil": not sign and remainder != 0,
        "Trunc": False,
        "Round": 2 * remainder > denominator or (2 * remainder == denominator and whole & 1),
    }[op]
    integer = whole + increment
    return encode(fmt, Q(-integer if sign else integer), negative_zero=sign)


def staged_sqrt(fmt, bits):
    return convert(F32, fmt, unary(F32, "Sqrt", convert(fmt, F32, bits)))


def pool(fmt):
    # Fixed bounded pool: signed IEEE classes, tiny/boundary inputs, both sides
    # of all four integral ties, large integral regimes, and exact sqrt witnesses.
    minimum = 1 << fmt.fraction
    values = [0, fmt.sign, fmt.inf, fmt.sign | fmt.inf,
              fmt.nan | 0x15, fmt.sign | fmt.nan | 0x23,
              fmt.inf | 1, fmt.sign | fmt.inf | 3]
    for bits in [1, minimum - 1, minimum, minimum + 1, fmt.inf - 1]:
        values.extend([bits, fmt.sign | bits])
    for tie in [Q(1, 2), Q(3, 2), Q(5, 2), Q(7, 2)]:
        bits = encode(fmt, tie)
        for neighbor in [bits - 1, bits, bits + 1]:
            values.extend([neighbor, fmt.sign | neighbor])
    for bits in [encode(fmt, pow2(fmt.fraction)) - 1,
                 encode(fmt, pow2(fmt.fraction)),
                 encode(fmt, pow2(fmt.fraction)) + 1]:
        values.extend([bits, fmt.sign | bits])
    values += [encode(fmt, Q(1)), encode(fmt, Q(2)), encode(fmt, Q(4)),
               encode(fmt, Q(25, 4)) + 1,
               encode(fmt, 1 + pow2(1 - fmt.fraction)),
               encode(fmt, pow2(min(200, fmt.bias))),
               encode(fmt, pow2(-min(101, fmt.bias))),
               encode(fmt, pow2(min(64, fmt.bias))),
               encode(fmt, pow2(min(30, fmt.fraction - 2)) + Q(1, 2)),
               encode(fmt, pow2(min(30, fmt.fraction - 2)) + Q(3, 2))]
    values = list(dict.fromkeys(values))
    # Complete the 64 slots with non-square significands in exponent bands.
    for exponent in range(1, (1 << fmt.exponent) - 1):
        bits = (exponent << fmt.fraction) | 3
        if bits not in values:
            values.append(bits)
        if len(values) == N:
            break
    assert len(values) == N and len(set(values)) == N, (fmt, len(values))
    return values


def chains(fmt):
    descriptions = [
        (encode(fmt, Q(25, 4)) + 1, ["Sqrt", "Round"]),
        (fmt.sign | 1, ["Ceil", "Trunc", "Round"]),
        (encode(fmt, Q(7, 2)), ["Floor", "Sqrt", "Ceil"]),
    ]
    result = []
    for initial, ops in descriptions:
        current, steps = initial, []
        for op in ops:
            current = unary(fmt, op, current)
            steps.append((op, current))
        result.append((initial, steps))
    assert result[0][1] == [("Sqrt", encode(fmt, Q(5, 2))), ("Round", encode(fmt, Q(2)))]
    # The real sqrt exceeds 2.5, so a contracted real sqrt->round would give 3.
    assert decode(fmt, result[0][0])[2] > Q(25, 4)
    return result


def self_test():
    encoder.self_test()
    for r in [0, 1, 2, 3, 2**24 - 1, 2**52 - 1]:
        midpoint = Q((2 * r + 1) ** 2, 4)
        delta = Q(1, 16)
        assert rounded_root(midpoint - delta) == r
        assert rounded_root(midpoint) == r + (r & 1)
        assert rounded_root(midpoint + delta) == r + 1
    for fmt in FORMATS:
        for n in [1, 4, 9, 16]:
            assert unary(fmt, "Sqrt", encode(fmt, Q(n))) == encode(fmt, Q(isqrt(n)))
        assert unary(fmt, "Sqrt", fmt.sign) == fmt.sign
        assert unary(fmt, "Sqrt", fmt.inf) == fmt.inf
        assert unary(fmt, "Sqrt", fmt.sign | fmt.inf) == fmt.nan
        for magnitude, integer in [(Q(1, 2), 0), (Q(3, 2), 2), (Q(5, 2), 2), (Q(7, 2), 4)]:
            for sign in [False, True]:
                bits = encode(fmt, -magnitude if sign else magnitude)
                assert unary(fmt, "Round", bits) == encode(fmt, Q(-integer if sign else integer), sign)
        for op in OPS:
            assert unary(fmt, op, 0) == 0
            assert unary(fmt, op, fmt.sign) == fmt.sign
        for bits in pool(fmt):
            kind, sign, value = decode(fmt, bits)
            if kind == "finite" and value > 0 and not sign:
                result = unary(fmt, "Sqrt", bits)
                _, _, y = decode(fmt, result)
                _, _, lo = decode(fmt, result - 1)
                _, _, hi = decode(fmt, result + 1)
                lower, upper = ((lo + y) / 2) ** 2, ((y + hi) / 2) ** 2
                assert lower <= value <= upper
                if value in [lower, upper]:
                    assert result & 1 == 0
        chains(fmt)
    assert unary(F64, "Sqrt", 1) == encode(F64, pow2(-537))
    assert unary(F64, "Sqrt", encode(F64, pow2(200))) == encode(F64, pow2(100))
    assert unary(F64, "Sqrt", encode(F64, 1 + pow2(-51))) == encode(F64, 1) + 1


def mpfr_check():
    """Round directly at target precision, including subnormal source operands."""
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
    for name in ["sqrt", "floor", "ceil", "trunc"]:
        getattr(lib, "mpfr_" + name).argtypes = [ptr, ptr] + ([ctypes.c_int] if name == "sqrt" else [])
    lib.mpfr_rint.argtypes = [ptr, ptr, ctypes.c_int]
    checked = 0
    for fmt in FORMATS:
        a, out = Number(), Number()
        for item in [a, out]:
            lib.mpfr_init2(ctypes.byref(item), fmt.fraction + 1)
        try:
            inputs = range(65536) if fmt in [F16, BF16] else pool(fmt)
            for bits in inputs:
                kind, sign, magnitude = decode(fmt, bits)
                if kind != "finite" or not magnitude:
                    continue
                text = ("-" if sign else "") + bin(magnitude.numerator)[2:] + "@" + str(1 - magnitude.denominator.bit_length())
                assert lib.mpfr_set_str(ctypes.byref(a), text.encode(), 2, 0) == 0
                for op in OPS:
                    if op == "Sqrt" and sign:
                        continue
                    function = getattr(lib, "mpfr_" + {"Round": "rint"}.get(op, op.lower()))
                    function(ctypes.byref(out), ctypes.byref(a), *([0] if op in ["Sqrt", "Round"] else []))
                    exponent = ctypes.c_long()
                    address = lib.mpfr_get_str(None, ctypes.byref(exponent), 2, 0, ctypes.byref(out), 0)
                    try:
                        digits = ctypes.string_at(address).decode()
                    finally:
                        lib.mpfr_free_str(address)
                    negative = digits.startswith("-")
                    digits = digits.lstrip("-")
                    exact = int(digits, 2) * pow2(exponent.value - len(digits))
                    actual = encode(fmt, -exact if negative else exact, negative)
                    assert actual == unary(fmt, op, bits), (fmt.name, op, bits, actual)
                    checked += 1
        finally:
            for item in [a, out]:
                lib.mpfr_clear(ctypes.byref(item))
    return checked


def artifacts():
    data, descriptors, chain_rows, half_rows, comparisons = bytearray(), [], [], [], {}
    def append(fmt, bits):
        data.extend(bits.to_bytes(fmt.width // 8, "little"))
    for fmt in FORMATS:
        offset = len(data)
        values = pool(fmt)
        for bits in values:
            append(fmt, bits)
        for op in OPS:
            for bits in values:
                append(fmt, unary(fmt, op, bits))
        descriptors.append(str(offset))
        rows = []
        for initial, steps in chains(fmt):
            entries = ", ".join(f"ChainStep {{ operation: Operation::{op}, expected: 0x{bits:x} }}" for op, bits in steps)
            rows.append(f"ChainFixture {{ initial: 0x{initial:x}, steps: &[{entries}] }}")
        chain_rows.append("&[" + ",\n".join(rows) + "]")
    for fmt in [F16, BF16]:
        offset = len(data)
        for op in OPS:
            for bits in range(65536):
                append(fmt, unary(fmt, op, bits))
        staged_offset = len(data)
        differences, exact_comparisons = [], 0
        for bits in range(65536):
            staged = staged_sqrt(fmt, bits)
            ideal = unary(fmt, "Sqrt", bits)
            append(fmt, staged)
            if decode(fmt, ideal)[0] != "nan":
                exact_comparisons += 1
                if ideal != staged:
                    differences.append({"input": bits, "ideal": ideal, "staged": staged})
        comparisons[fmt.name] = {"all_codes": 65536, "non_nan_exact_comparisons": exact_comparisons,
                                 "differences": differences, "nan_policy": "classification"}
        half_rows.append(f"({offset}, {staged_offset})")
    digest = hashlib.sha256(data).hexdigest()
    source = "// Generated by tools/nvidia/float-rounding-generate.py. Do not edit.\n"
    source += f'pub const FIXTURE_SHA256: &str = "{digest}";\n'
    source += 'const FIXTURE_BYTES: &[u8] = include_bytes!("float_rounding_fixtures.bin");\n'
    source += "const POOL_OFFSETS: [usize; 4] = [" + ", ".join(descriptors) + "];\n"
    source += "const HALF_OFFSETS: [(usize, usize); 2] = [" + ", ".join(half_rows) + "];\n"
    source += "const CHAINS: [&[ChainFixture]; 4] = [\n" + ",\n".join(chain_rows) + "\n];\n"
    root = Path(__file__).resolve().parents[2]
    shared = root / "os/library/ai/src/validation"
    manifest = {
        "version": 1,
        "oracle": "Python exact Fraction; isqrt plus squared midpoint comparison; imported frozen IEEE RNE encoder",
        "generator_dependencies_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in [Path(__file__).resolve(), ENCODER_PATH.resolve()]},
        "fixture_sha256": digest, "fixture_bytes": len(data), "pool_values_per_format": N,
        "formats": [f.name for f in FORMATS], "Flex32": "F32 encodings; separate dtype cases",
        "operations": OPS,
        "layout": "native-width little endian; per format pool[64],expected[5][64]; then F16/BF16 expected[5][65536],staged_sqrt[65536]; offsets in generated Rust",
        "half_ideal_vs_staged_exact_f32_sqrt": comparisons,
        "public_cases": 465, "public_maximum_values": 257,
        "full_half_expected_values": 655360,
        "nan_policy": "public classification; separate raw integral payload preservation",
        "linux_approximation_policy": "observed differences remain failures; exact/staged equality does not qualify Linux approximation",
        "python_dependencies": "standard library only; optional local libmpfr via ctypes",
    }
    return {shared / "float_rounding_fixtures.bin": bytes(data), shared / "float_rounding_fixtures.rs": source.encode(),
            Path(__file__).with_name("float-rounding-manifest.json"): (json.dumps(manifest, indent=2) + "\n").encode()}


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
    print(json.dumps({"oracle_tests": "passed", "mpfr_results_checked": checked,
                      "artifacts": "verified" if args.check else "generated"}))


if __name__ == "__main__":
    main()
