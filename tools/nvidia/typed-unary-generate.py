#!/usr/bin/env python3
"""Reproduce typed Log1p/Sinh/Cosh/Tan fixtures with MPFR 4.2.2.

Read-only --check reproduces all artifacts. Writes require --output-root.
No production arithmetic, host libm, external reference module, or GPU
supplies expected values. Forward qualification only.
"""
import argparse
import ctypes as C
import ctypes.util
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

FORMATS = {'f64': (11, 52), 'f32': (8, 23), 'flex32': (8, 23),
           'f16': (5, 10), 'bf16': (8, 7)}
OPS = {'log1p': 17, 'sinh': 30, 'cosh': 31, 'tan': 32}
LIMITS = {'f64': dict(log1p=1, sinh=2, cosh=1, tan=2),
          'f32': dict(log1p=1, sinh=3, cosh=2, tan=4)}
LIB = C.CDLL(ctypes.util.find_library('mpfr'))
class MP(C.Structure):
    _fields_ = [('precision', C.c_long), ('sign', C.c_int),
                ('exponent', C.c_long), ('limbs', C.POINTER(C.c_ulong))]
P = C.POINTER(MP)
def bind(name, args, result=C.c_int):
    fn = getattr(LIB, 'mpfr_' + name)
    fn.argtypes, fn.restype = args, result
    return fn
INIT = bind('init2', [P, C.c_long], None)
CLEAR = bind('clear', [P], None)
SET = bind('set_str', [P, C.c_char_p, C.c_int, C.c_int])
MUL2 = bind('mul_2si', [P, P, C.c_long, C.c_int])
GET = bind('get_str', [C.c_void_p, C.POINTER(C.c_long), C.c_int, C.c_size_t, P, C.c_int], C.c_void_p)
FREE = bind('free_str', [C.c_void_p], None)
VERSION = bind('get_version', [], C.c_char_p)
PI = bind('const_pi', [P, C.c_int])
FUN = {op: bind(op, [P, P, C.c_int]) for op in (*OPS, 'asinh', 'acosh', 'atan', 'expm1', 'log')}
assert VERSION() == b'4.2.2', VERSION()

def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':')).encode()
def pow2(k): return Q(1 << max(k, 0), 1 << max(-k, 0))
def info(fmt):
    e, f = FORMATS[fmt]
    return e, f, (1 << (e-1))-1, 1 << (e+f), ((1 << e)-1) << f
def floor_log2(q):
    q = abs(q)
    k = q.numerator.bit_length()-q.denominator.bit_length()
    return k-int(q < pow2(k))
def decode(bits, fmt):
    e, f, bias, sign, inf = info(fmt)
    assert 0 <= bits < sign*2
    field, fraction = (bits >> f) & ((1 << e)-1), bits & ((1 << f)-1)
    if field == (1 << e)-1:
        return ('nan' if fraction else 'inf'), (-1 if bits & sign else 1)
    n = fraction if field == 0 else (1 << f)+fraction
    q = n*pow2((1-bias if field == 0 else field-bias)-f)
    return 'finite', (-q if bits & sign else q)
def rne_integer(q):
    n, r = divmod(q.numerator, q.denominator)
    return n+int(2*r > q.denominator or (2*r == q.denominator and n & 1))
def encode(q, fmt, negative_zero=False):
    e, f, bias, sign, inf = info(fmt)
    s = sign if q < 0 or (not q and negative_zero) else 0
    q = abs(q)
    if not q: return s
    step = max(floor_log2(q)-f, 1-bias-f)
    n = rne_integer(q/pow2(step))
    if not n: return s
    if n >= 1 << (f+1):
        n >>= 1
        step += 1
    if step+f > bias: return s | inf
    if n < 1 << f: return s | n
    return s | ((step+f+bias) << f) | (n-(1 << f))
def classify(bits, fmt):
    if bits is None: return 'nan'
    e, f, bias, sign, inf = info(fmt)
    b = bits & (sign-1)
    if b > inf: return 'nan'
    if b == inf: return 'inf'
    if not b: return 'zero'
    return 'subnormal' if b < 1 << f else 'normal'
def code(bits, fmt): return None if bits is None else format(bits, f'0{(sum(FORMATS[fmt])+1)//4}x')
def rank(bits, fmt):
    sign = info(fmt)[3]
    return (~bits & (2*sign-1)) if bits & sign else bits | sign
def unrank(r, fmt):
    sign = info(fmt)[3]
    return r & (sign-1) if r & sign else ~r & (2*sign-1)
def cast(bits, src, dst):
    kind, q = decode(bits, src)
    if kind == 'nan': return None
    if kind == 'inf': return info(dst)[4] | (info(dst)[3] if q < 0 else 0)
    return encode(q, dst, bool(bits & info(src)[3]))
def rational(x):
    exponent = C.c_long()
    ptr = GET(None, C.byref(exponent), 2, 0, C.byref(x), 0)
    try: digits = C.string_at(ptr).decode()
    finally: FREE(ptr)
    assert '@' not in digits, digits
    negative = digits.startswith('-')
    digits = digits.lstrip('-')
    q = int(digits, 2)*pow2(exponent.value-len(digits))
    return -q if negative else q
def interval(q, op, precision):
    values = [MP(), MP(), MP()]
    for v in values: INIT(C.byref(v), precision)
    x, lo, hi = values
    try:
        assert q.denominator & (q.denominator-1) == 0
        n = q.numerator
        trailing = (abs(n) & -abs(n)).bit_length()-1 if n else 0
        assert SET(C.byref(x), str(n >> trailing).encode(), 10, 0) == 0
        assert MUL2(C.byref(x), C.byref(x), trailing-(q.denominator.bit_length()-1), 0) == 0
        assert rational(x) == q
        FUN[op](C.byref(lo), C.byref(x), 3)
        FUN[op](C.byref(hi), C.byref(x), 2)
        a, b = rational(lo), rational(hi)
        assert a <= b
        return a, b
    finally:
        for v in values: CLEAR(C.byref(v))
def pi_interval(p):
    values = [MP(), MP()]
    for v in values: INIT(C.byref(v), p)
    try:
        PI(C.byref(values[0]), 3)
        PI(C.byref(values[1]), 2)
        return tuple(rational(v) for v in values)
    finally:
        for v in values: CLEAR(C.byref(v))
def qjson(q): return [str(q.numerator), str(q.denominator)]
def reference(bits, fmt, op, minimum=128, keep_bounds=False):
    kind, x = decode(bits, fmt)
    sign, inf = info(fmt)[3:]
    half = fmt in ('f16', 'bf16')
    method, p, bounds = 'exact_special', 0, None
    result, center = None, None
    if kind == 'nan': result = None
    elif kind == 'inf':
        result = (None if x < 0 else inf) if op == 'log1p' else (
            None if op == 'tan' else inf if op == 'cosh' or x > 0 else sign | inf)
    elif op == 'log1p' and x <= -1:
        result = sign | inf if x == -1 else None
    elif not x:
        result = encode(Q(1), fmt) if op == 'cosh' else bits
    elif op in ('sinh', 'cosh') and abs(x) >= 2048:
        # e>2 implies sinh(x)>2^(x-1)-1/2>2^1024 for x>=2048.
        # cosh(x)>sinh(x). The largest destination exponent is 1023.
        result = inf | (sign if op == 'sinh' and x < 0 else 0)
        method = 'analytic_overflow_abs_ge_2048'
    else:
        method = 'directed_mpfr'
        p = max(minimum, 2*max(0, floor_log2(x))+256 if op == 'tan' else 128)
        while p <= 8192:
            lo, hi = interval(x, op, p)
            result = encode(lo, fmt)
            center = encode(lo, 'f32') if half else None
            if result == encode(hi, fmt) and (not half or center == encode(hi, 'f32')):
                bounds = {'lower': qjson(lo), 'upper': qjson(hi)}
                break
            p *= 2
        else: raise RuntimeError(('unresolved', fmt, op, code(bits, fmt)))
    if half and method != 'directed_mpfr': center = None if result is None else cast(result, fmt, 'f32')
    staged = cast(center, 'f32', fmt) if half and center is not None else None
    out = dict(ideal=result, f32=center, staged=staged, method=method, precision=p)
    if half:
        allowed = [] if result is None else [staged]
        if method != 'exact_special':
            allowed = sorted({cast(unrank(r, 'f32'), 'f32', fmt)
                              for r in range(rank(center, 'f32')-LIMITS['f32'][op],
                                             rank(center, 'f32')+LIMITS['f32'][op]+1)
                              if 0 <= r < 1 << 32 and classify(unrank(r, 'f32'), 'f32') != 'nan'})
        out['allowed'] = allowed
    if keep_bounds and bounds is not None: out['bounds'] = bounds
    return out
def qualify(bits, actual, fmt, op, ref=None):
    ref = ref or reference(bits, fmt, op)
    half = fmt in ('f16', 'bf16')
    expected = ref['staged'] if half else ref['ideal']
    ec, ac = classify(expected, fmt), classify(actual, fmt)
    failures, diagnostics = [], []
    finite = ('normal', 'subnormal')
    if ec != ac and not (ec in finite and ac in finite): failures.append('class_mismatch')
    if ec in finite and ac in finite and ec != ac: diagnostics.append('normal_subnormal_transition')
    if ec != 'nan' and ac != 'nan' and bool(expected & info(fmt)[3]) != bool(actual & info(fmt)[3]):
        failures.append('sign_mismatch')
    if ref['method'] == 'exact_special' and ec != 'nan' and actual != expected: failures.append('exact_special_mismatch')
    if op == 'cosh' and ac != 'nan' and (actual & info(fmt)[3] or ac == 'zero' or (ac in finite and decode(actual, fmt)[1] < 1)):
        failures.append('cosh_codomain')
    distance = abs(rank(actual, fmt)-rank(expected, fmt)) if ac in finite and ec in finite else None
    if half:
        if ec != 'nan' and actual not in ref['allowed']: failures.append('half_membership')
    elif distance is not None and distance > LIMITS['f64' if fmt == 'f64' else 'f32'][op]:
        failures.append('step_budget')
    return dict(failures=failures, diagnostics=diagnostics, distance=distance)

def boundary(fmt, label, fn, argument, negate=False):
    for p in (256, 512, 1024, 2048, 4096, 8192):
        lo, hi = interval(argument, fn, p)
        if negate: lo, hi = -hi, -lo
        a, b = encode(lo, fmt), encode(hi, fmt)
        if a == b:
            return dict(dtype=fmt, label=label, function=fn, argument=qjson(argument),
                        negate=negate, center=code(a, fmt), precision=p,
                        bounds=dict(lower=qjson(lo), upper=qjson(hi)))
    raise RuntimeError(('boundary unresolved', fmt, label))
def boundaries(fmt):
    e, f, bias, sign, inf = info(fmt)
    delta = pow2(1-bias-f)
    cuts = [('zero_midpoint', delta/2), ('normal_midpoint', pow2(1-bias)-delta/2)]
    out = []
    for label, y in cuts:
        for negative in (False, True):
            sy = -y if negative else y
            for op, fn in (('log1p', 'expm1'), ('sinh', 'asinh'), ('tan', 'atan')):
                out.append(boundary(fmt, op+'_'+label+('_negative' if negative else '_positive'), fn, sy))
    # Exact RNE overflow midpoint includes the tie, whose upper significand is even.
    overflow = pow2(bias+1)-pow2(bias-f-1)
    for stage, cut in [('ideal', overflow)]:
        for op, inverse in [('sinh', 'asinh'), ('cosh', 'acosh')]:
            out.append(boundary(fmt, op+'_overflow_'+stage, inverse, cut))
    if fmt in ('f16', 'bf16'):
        c = encode(overflow, 'f32')
        cut = (decode(c-1, 'f32')[1]+overflow)/2
        assert not c & 1
        for op, inverse in [('sinh', 'asinh'), ('cosh', 'acosh')]:
            out.append(boundary(fmt, op+'_overflow_staged', inverse, cut))
        # Include the actual F32-operation overflow transition separately.
        e32, f32, b32, _, _ = info('f32')
        cut32 = pow2(b32+1)-pow2(b32-f32-1)
        for op, inverse in [('sinh', 'asinh'), ('cosh', 'acosh')]:
            out.append(boundary(fmt, op+'_overflow_f32_intermediate', inverse, cut32))
    # RNE boundary leaving cosh's flat region at +1.
    out.append(boundary(fmt, 'cosh_one_midpoint_ideal', 'acosh', 1+pow2(-f-1)))
    return out
def corpus(fmt):
    bounds = boundaries(fmt)
    if fmt in ('f16', 'bf16'):
        return list(range(65536)), {'all_storage_encodings': 65536}, bounds
    e, f, bias, sign, inf = info(fmt)
    groups = {}
    def add(b, tag):
        if 0 <= b < 2*sign: groups.setdefault(b, set()).add(tag)
    def around(b, tag, radius=8, mirror=False):
        if classify(b, fmt) == 'nan': return
        for d in range(-radius, radius+1):
            r = rank(b, fmt)+d
            if 0 <= r < 2*sign:
                n = unrank(r, fmt)
                if classify(n, fmt) != 'nan':
                    add(n, tag)
                    if mirror: add(n ^ sign, tag)
    for s in (0, sign):
        for b in (0, 1, 2, 3, 4, (1 << f)-2, (1 << f)-1, 1 << f,
                  (1 << f)+1, inf-2, inf-1, inf, inf+1,
                  inf+(1 << (f-1)), inf+(1 << (f-1))+21, inf+(1 << f)-1):
            add(s | b, 'special_subnormal_extreme')
        for exponent in range(1, (1 << e)-1):
            for fraction in (0, 1, (1 << (f-1)), (1 << f)-2, (1 << f)-1):
                add(s | (exponent << f) | fraction, 'every_exponent_five_significands')
    for n in (Q(-1), Q(1), Q(1,2), Q(1,4), Q(3,4), Q(2), Q(8), Q(16),
              Q(64), Q(88), Q(89), Q(90), Q(710), Q(711), Q(712), Q(1024), Q(2048)):
        around(encode(n, fmt), 'arithmetic_boundary', 16, True)
    for k in (-f-1, -f-2, -f, -(f//2), -(f//2)-1, 1-bias, 1-bias-f):
        around(encode(pow2(k), fmt), 'tiny_absorption_boundary', 16, True)
    for b in bounds: around(int(b['center'],16), b['label'], 8, True)
    # Small odd/even half-pi multiples include poles and roots. These are stored dyadics.
    for k in range(1, 65):
        lo, hi = pi_interval(256)
        lo, hi = lo*Q(k,2), hi*Q(k,2)
        assert encode(lo, fmt) == encode(hi, fmt)
        b = encode(lo, fmt)
        bounds.append(dict(dtype=fmt, label='pi_half_multiple', multiplier=qjson(Q(k,2)),
                           center=code(b, fmt), precision=256,
                           bounds=dict(lower=qjson(lo), upper=qjson(hi))))
        around(b, 'tan_pole_or_root', 8, True)
    # Full-range deterministic large reductions. No search for a worst argument.
    ks = (20, 40, 60, 100, 200, 500, 900, 990, 1000, 1020) if fmt == 'f64' else (20, 40, 60, 100, 120)
    for k in ks:
        p = 2*k+256
        pi_lo, pi_hi = pi_interval(p)
        for label, scale in [('pi_power',pow2(k)), ('pi_large_odd_pole',(pow2(k)+1)/2)]:
            lo, hi = pi_lo*scale, pi_hi*scale
            assert encode(lo, fmt) == encode(hi, fmt)
            b = encode(lo, fmt)
            bounds.append(dict(dtype=fmt, label=label, multiplier=qjson(scale),
                               center=code(b, fmt), precision=p,
                               bounds=dict(lower=qjson(lo), upper=qjson(hi))))
            around(b, 'tan_fullrange_'+label, 8, True)
    for i in range(2048):
        digest = hashlib.sha256(f'next-unary-reference-v1/{fmt}/{i}'.encode()).digest()
        b = int.from_bytes(digest[:(e+f+1)//8], 'little')
        add(b, 'sha256_bitpattern')
    words = sorted(groups)
    cap = 28000 if fmt == 'f64' else 9000
    assert len(words) <= cap, (fmt, len(words))
    return words, dict(Counter(tag for tags in groups.values() for tag in tags)), bounds


POLICY_TEXT = '{\n  "accuracy_accounting": {\n    "contract": "accounting-contract.json; account_actual in class-scan.py",\n    "forbidden": "Never assume ideal_finite_pairs equals finite_pairs or a finite count from an earlier corpus.",\n    "half_scan": "Exhaustively scan ideal/staged class and nonzero-finite status for each of 65536 inputs and four operations.",\n    "maxima": "Each histogram, maximum, and worst-input record uses its own eligible-pair denominator.",\n    "pair_rule": "Derive staged and ideal metric eligibility separately from each actual result row."\n  },\n  "comparator_order": [\n    "structure",\n    "raw classes",\n    "exact specials",\n    "class/sign/domain/codomain gates",\n    "step distance or half-image membership",\n    "boundary diagnostics",\n    "observed metrics"\n  ],\n  "criteria": [\n    {\n      "dtype": "f64",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 1,\n      "mode": "representable_step_distance",\n      "opcode": 17,\n      "operation": "log1p",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f64",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 2,\n      "mode": "representable_step_distance",\n      "opcode": 30,\n      "operation": "sinh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f64",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 1,\n      "mode": "representable_step_distance",\n      "opcode": 31,\n      "operation": "cosh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f64",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 2,\n      "mode": "representable_step_distance",\n      "opcode": 32,\n      "operation": "tan",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 1,\n      "mode": "representable_step_distance",\n      "opcode": 17,\n      "operation": "log1p",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 3,\n      "mode": "representable_step_distance",\n      "opcode": 30,\n      "operation": "sinh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 2,\n      "mode": "representable_step_distance",\n      "opcode": 31,\n      "operation": "cosh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 4,\n      "mode": "representable_step_distance",\n      "opcode": 32,\n      "operation": "tan",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "flex32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 1,\n      "mode": "representable_step_distance",\n      "opcode": 17,\n      "operation": "log1p",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "flex32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 3,\n      "mode": "representable_step_distance",\n      "opcode": 30,\n      "operation": "sinh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "flex32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 2,\n      "mode": "representable_step_distance",\n      "opcode": 31,\n      "operation": "cosh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "flex32",\n      "f32_intermediate_step_limit": null,\n      "finite_step_limit": 4,\n      "mode": "representable_step_distance",\n      "opcode": 32,\n      "operation": "tan",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f16",\n      "f32_intermediate_step_limit": 1,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 17,\n      "operation": "log1p",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f16",\n      "f32_intermediate_step_limit": 3,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 30,\n      "operation": "sinh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f16",\n      "f32_intermediate_step_limit": 2,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 31,\n      "operation": "cosh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "f16",\n      "f32_intermediate_step_limit": 4,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 32,\n      "operation": "tan",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "bf16",\n      "f32_intermediate_step_limit": 1,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 17,\n      "operation": "log1p",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "bf16",\n      "f32_intermediate_step_limit": 3,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 30,\n      "operation": "sinh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "bf16",\n      "f32_intermediate_step_limit": 2,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 31,\n      "operation": "cosh",\n      "project_target": true,\n      "vendor_guaranteed": false\n    },\n    {\n      "dtype": "bf16",\n      "f32_intermediate_step_limit": 4,\n      "finite_step_limit": null,\n      "mode": "f32_neighbor_destination_image",\n      "opcode": 32,\n      "operation": "tan",\n      "project_target": true,\n      "vendor_guaranteed": false\n    }\n  ],\n  "dtypes": [\n    "f64",\n    "f32",\n    "flex32",\n    "f16",\n    "bf16"\n  ],\n  "exact_specials": {\n    "cosh": {\n      "nan": "nan_class",\n      "negative_infinity": "positive_infinity",\n      "negative_zero": "positive_one",\n      "positive_infinity": "positive_infinity",\n      "positive_zero": "positive_one"\n    },\n    "log1p": {\n      "less_than_minus_one": "nan_class",\n      "minus_one": "negative_infinity",\n      "nan": "nan_class",\n      "negative_infinity": "nan_class",\n      "negative_zero": "negative_zero",\n      "positive_infinity": "positive_infinity",\n      "positive_zero": "positive_zero"\n    },\n    "sinh": {\n      "nan": "nan_class",\n      "negative_infinity": "negative_infinity",\n      "negative_zero": "negative_zero",\n      "positive_infinity": "positive_infinity",\n      "positive_zero": "positive_zero"\n    },\n    "tan": {\n      "nan": "nan_class",\n      "negative_infinity": "nan_class",\n      "negative_zero": "negative_zero",\n      "positive_infinity": "nan_class",\n      "positive_zero": "positive_zero"\n    }\n  },\n  "expected_summary": {\n    "exclude": [\n      "max errors",\n      "histograms",\n      "worst inputs",\n      "mismatch counts",\n      "GPU metadata",\n      "pass/fail",\n      "timings"\n    ],\n    "include": [\n      "policy hash",\n      "input manifest hash",\n      "immutable selection",\n      "dtype/operation counts",\n      "materialized structural counts"\n    ],\n    "structural_count": null\n  },\n  "finite_metric": {\n    "distance": "abs(rank(actual)-rank(reference)); finite nonzero pairs only",\n    "meaning": "representable-step metric; not a claim about local-spacing real-error ULP",\n    "rank": "positive b -> b|sign; negative b -> (~b)&(2*sign-1)"\n  },\n  "finite_normal_subnormal": "Diagnostic within the step or half-image allowance, consistent with frozen Exp/Log v2 and trig.",\n  "half_envelope": {\n    "algorithm": [\n      "Certify c=RN32(f(x)) with directed MPFR.",\n      "Enumerate rank(c)-b through rank(c)+b using the operation-specific F32 budget.",\n      "Retain non-NaN F32 codes, including zeros, subnormals, and infinities.",\n      "Convert each code once with exact destination RNE and deduplicate.",\n      "Apply exact-special, staged class, sign, nonzero, and codomain gates independently."\n    ],\n    "exact_specials": "singleton expected bits or NaN class; no widening",\n    "infinite_center": "No finite metric. Staged infinity class remains strict even if finite neighbors enter the image.",\n    "limit": "Final membership cannot establish the hidden intermediate or emitted schedule.",\n    "maximum_candidates": {\n      "cosh": 5,\n      "log1p": 3,\n      "sinh": 7,\n      "tan": 9\n    }\n  },\n  "independent_gates": [\n    "Shape, dtype, output byte count, exact source bytes, aliases, guards, odd packed tails, and no launches for empty tensors.",\n    "NaN versus infinity versus finite versus zero classes match; all expected nonzero outputs remain nonzero.",\n    "Every non-NaN output sign matches the reference, including signed zero and overflow infinity.",\n    "Log1p domain and minus-one singularity remain strict.",\n    "Cosh finite results are positive and at least one; infinity is positive.",\n    "Expected finite output cannot become infinity, even within one rank of maximum finite.",\n    "Expected overflow infinity cannot become maximum finite, even within one rank.",\n    "Exact-special inputs bypass numerical widening. NaN payload/sign/quiet-bit outputs remain observations."\n  ],\n  "integration": {\n    "constraints": "Reuse the current typed ABI and Native dispatch paths; do not create a duplicate family.",\n    "implementation_status": "No repository implementation in this cache-only task.",\n    "opcodes": "Preserve Elementwise discriminants: Log1p17, Sinh30, Cosh31, Tan32.",\n    "prerequisite": "Compiler full-entry qualification, followed by actual shared conformance.",\n    "proposal": "Extend existing FloatArithmetic operation list from 19 to 23."\n  },\n  "log1p": {\n    "cubecl_0_11_3": "Lowered through polyfill (input + one).ln(); source-level mismatch with tiny/signed-zero target.",\n    "native_f32": "Existing compensation log(u)*(x/(u-1)), u=RN32(1+x), returns original bits for u=1 or +Inf.",\n    "options": [\n      "Prefer a qualified genuine typed primitive if the pinned frontend exposes one.",\n      "Otherwise qualify a compensated implementation separately for each arithmetic type.",\n      "Keep strict specials and evaluate primitive rounding, division, contraction, and FTZ behavior."\n    ],\n    "oracle": "MPFR log1p on exact stored x; never log(RN(1+x))",\n    "warning": "Algebra does not prove an error budget. No inherited CUDA Log1p guarantee covers a composed implementation."\n  },\n  "no_ftz": "Any expected nonzero-to-zero output fails independently of numerical distance.",\n  "operations": {\n    "cosh": 31,\n    "log1p": 17,\n    "sinh": 30,\n    "tan": 32\n  },\n  "options": {\n    "comparison": "independent reference for each actual stored input",\n    "half": "all 65536 stored encodings per dtype and operation",\n    "math": "ordinary typed Tan/Sinh/Cosh; no approximate or fast-math substitution",\n    "oracle": "MPFR 4.2.2 RNDD/RNDU; exact integer RNE; fail unresolved above 8192 bits",\n    "runtime": "future official CLI and actual native GPU qualification required",\n    "selection": "all four operations and all five dtypes",\n    "wide": "fixed bounded boundary, exponent-stratified, pi-derived, and SHA256 input lists"\n  },\n  "policy_id": "burn-typed-next-unary-reference-proposed-v1",\n  "references": {\n    "f32": "RN32(f(exact stored binary32 input))",\n    "f64": "RN64(f(exact stored binary64 input)); no public-path F32 narrowing",\n    "flex32": "same values as F32, separate metadata; no TF32 quantization",\n    "half_ideal": "RN_destination(f(exact stored half input))",\n    "half_schedule": "stored half -> exact F32 promotion -> F32 operation -> one destination RNE",\n    "half_staged": "RN_destination(RN32(f(exact stored half input)))"\n  },\n  "schema_version": 1,\n  "source_urls": [\n    "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html",\n    "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html",\n    "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html",\n    "https://www.mpfr.org/mpfr-4.2.2/mpfr.html#Rounding",\n    "https://docs.rs/crate/cubecl-core/0.11.3/source/src/frontend/polyfills.rs",\n    "https://docs.rs/crate/cubecl-cpp/0.11.3/source/src/shared/unary.rs",\n    "https://docs.rs/crate/burn-cubecl/0.22.1/source/src/kernel/unary_float.rs",\n    "https://docs.rs/crate/burn-tensor/0.22.1/source/src/tensor/api/float.rs"\n  ],\n  "status": "proposed project targets; no GPU accuracy or backend parity claim",\n  "vendor_figures": {\n    "double": {\n      "cosh": 1,\n      "log1p": 1,\n      "sinh": 2,\n      "tan": 2\n    },\n    "single": {\n      "cosh": 2,\n      "log1p": 1,\n      "sinh": 3,\n      "tan": 4\n    },\n    "source": "CUDA Programming Guide 5.5.7, tables 48, 50, 51",\n    "status": "empirical nonexhaustive maxima, not guarantees; no transfer to composed Log1p implementation"\n  }\n}\n'
BASELINE = {'f64': {'input_count': 26778, 'input_sha256': 'fff7347fcf632fc0860689dfade25cd094d66ecf42ee182f0f49fbf221eb9337', 'reference_stream_sha256': {'cosh': 'f95c84247625491d231c0b34da4e3f6a4032beb4cc8e3d807620ee727dcf290b', 'log1p': 'b24bb5a545ee39217e9b061b674be82b99f60990958a1a2c520b2eb747c2633e', 'sinh': 'fbe9c6329f8e95c5a9821b8b8bd1c3a4865213748916ab940cdeb414ae6b4512', 'tan': 'f7d4360494ad28f08dc7d021c9cddc0338f1d4480ea9d0aa400602370123bf19'}}, 'f32': {'input_count': 8552, 'input_sha256': '4ffdbe2fed2764e0a0a37487074950bc91b4c7c464f6f666e188f10b2cf0a498', 'reference_stream_sha256': {'cosh': '66b40f50bc9bb72c138daa3c1527561844be1eb3f1fdb27d86924e12a7604a26', 'log1p': 'abc6b99c722a5315f9e3b6fd8197d145d47884bd8bf0883f18db544a7e3e3c25', 'sinh': '57ba8527e018bbabe6d9ace78bada518ac1c6c3f7e67d23ab51b891f31d2d5f8', 'tan': '0385dfe6dd7f0210703a19bc0cad0a067916cb476d4d0ae534910029f5e3d139'}}, 'f16': {'input_count': 65536, 'input_sha256': '68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b', 'reference_stream_sha256': {'cosh': 'f2a8defc0404a15726482f14f02b8cf971e656c9af497d08e57fb876203ba9a0', 'log1p': 'f958a5db907c9ade634f04f84cd8e98326c3c458b2d984f208d1fa58509d581e', 'sinh': 'd4bb338be3e6158b694d85f3cd600b25336b557c1f9bec3312208c807c60428b', 'tan': '24db0c3f75a46648e81fae2f33cf70b84e78064b4429f3bed645188fc2dfcf27'}}, 'bf16': {'input_count': 65536, 'input_sha256': '68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b', 'reference_stream_sha256': {'cosh': '139d2be26972f55d6c749d77ce8393ad6b6efa4c37b4a382189394dc2b47631d', 'log1p': '9d21eb2d41fa3e857f1159a14d0a744da8d9ebe7948b8106f6233e08fedd5347', 'sinh': '190a5f460698c3f54c30f7170ddf21d717901bb83c3f149e11818018028dbcc8', 'tan': '787fb7c7f33c2c63ac883ddc95df092b4eb5f5bfdc38a9776566e242efd76cb5'}}}
PREPARATION_MODEL_SHA256 = '803c6a3ba8940b19924ce54197dfa9fbd069288d967387e6e6903cb0e11a4417'
PREPARATION_MANIFEST_SHA256 = '0670c66c430ce2d5e38e548d012e89c5f5cd5b0069233194fc84d8e21c8b3e20'
PRIMARY_SOURCES = {'cuda-guide.html': {'bytes': 245348, 'sha256': '4277615c7d34b1f24fa9dd7908b7f71e63cc87d516740cc89bf56f05b5be7f77', 'url': 'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html'}, 'cuda-math-double.html': {'bytes': 245449, 'sha256': '33fedd606826502cf91036bc5f054dc3ef266d3228930ca667d30e5fa1fa7855', 'url': 'https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html'}, 'cuda-math-single.html': {'bytes': 243054, 'sha256': 'de1b072026dae5d016001e21574daa1cb566ed21c424dccc29b069bad21b4195', 'url': 'https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html'}, 'cuda-tile-13.4-operations.html': {'bytes': 1177348, 'sha256': 'b05512d7cc3b1571adacb29ab40e209ade6e826fc8d38cb0b760881e2b281773', 'url': 'https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html'}, 'mpfr-4.2.2.html': {'bytes': 562991, 'sha256': 'ee996f3dcc11d1f88393da162bdd90c3fa29191ca658c0fc594ffd38a22638bc', 'url': 'https://www.mpfr.org/mpfr-4.2.2/mpfr.html'}}
RECIPES = [['log1p', 'log1p', 'log1p'], ['sinh', 'sinh', 'sinh'], ['cosh', 'cosh', 'cosh'], ['tan', 'tan', 'tan'], ['sinh', 'log1p', 'tan'], ['cosh', 'log1p', 'sinh'], ['tan', 'cosh', 'log1p'], ['log1p', 'sinh', 'cosh']]

def canonical_nan(fmt):
    return info(fmt)[4] | (1 << (FORMATS[fmt][1]-1))

def fixture_row(bits, fmt, operation, result=None):
    r = reference(bits, fmt, operation) if result is None else result
    ideal = canonical_nan(fmt) if r['ideal'] is None else r['ideal']
    staged = ideal if r['staged'] is None else r['staged']
    allowed = sorted(r.get('allowed') or [staged], key=lambda b: rank(b, fmt))
    assert all(rank(b, fmt) == rank(a, fmt)+1 for a,b in zip(allowed,allowed[1:]))
    intermediate = r['f32']
    if intermediate is None:
        intermediate = canonical_nan('f32') if fmt in ('f16','bf16') and classify(ideal,fmt)=='nan' else 0
    return bits,ideal,staged,allowed[0],allowed[-1],intermediate

def legal_successors(bits,fmt,operation):
    r = reference(bits,fmt,operation)
    expected = r['staged'] if fmt in ('f16','bf16') else r['ideal']
    if expected is None:
        return [canonical_nan(fmt)]
    if r['method']=='exact_special' or classify(expected,fmt) in ('zero','inf'):
        return [expected]
    if fmt in ('f16','bf16'):
        candidates = r['allowed']
    else:
        budget = LIMITS['f64' if fmt=='f64' else 'f32'][operation]
        candidates = [unrank(n,fmt) for n in range(rank(expected,fmt)-budget,rank(expected,fmt)+budget+1)
                      if 0 <= n < 2*info(fmt)[3]]
    return sorted(b for b in candidates if not qualify(bits,b,fmt,operation,r)['failures'])

def chain_closure(fmt):
    sign,inf=info(fmt)[3:]
    seeds=[sign,sign|encode(Q(1),fmt),1,encode(Q(1),fmt),inf-1]
    inputs={op:set() for op in OPS}
    plans=[]
    for recipe in RECIPES:
        for seed in seeds:
            levels=[[seed]]
            finite=True
            for operation in recipe:
                level=levels[-1]
                inputs[operation].update(level)
                for bits in level:
                    r=reference(bits,fmt,operation)
                    finite &= decode(bits,fmt)[0]=='finite' and classify(r['ideal'],fmt) not in ('inf','nan')
                next_level=sorted({b for bits in level for b in legal_successors(bits,fmt,operation)})
                assert len(next_level)<=729
                levels.append(next_level)
            assert sum(map(len,levels))<=820
            plans.append(dict(seed=code(seed,fmt),operations=recipe,finite=finite,
                              levels=[[code(b,fmt) for b in level] for level in levels],
                              total_node_count=sum(map(len,levels))))
    assert len(plans)==40
    return inputs,plans

def self_check():
    count=0
    for fmt in ('f16','bf16'):
        for bits in range(65536):
            kind,q=decode(bits,fmt)
            if kind=='finite':
                assert encode(q,fmt,bool(bits & info(fmt)[3]))==bits
                assert decode(cast(bits,fmt,'f32'),'f32')[1]==q
                count+=1
    for fmt in FORMATS:
        _,f,bias,sign,inf=info(fmt)
        minimum=pow2(1-bias-f)
        assert encode(minimum/2,fmt)==0
        assert encode(-minimum/2,fmt)==sign
        assert encode(3*minimum/2,fmt)==2
        one=encode(Q(1),fmt)
        midpoint=(decode(one,fmt)[1]+decode(one+1,fmt)[1])/2
        assert encode(midpoint,fmt)==one
        assert encode(pow2(bias+1)-pow2(bias-f-1),fmt)==inf
    return dict(finite_half_roundtrips_and_exact_promotions=count,
                output_half_nan_intermediate='canonical F32 quiet NaN 7fc00000; input bytes remain exact',
                no_host_libm=True,no_production_oracle=True,precision_cap=8192)

def verify_boundaries(fmt,bounds):
    for b in bounds:
        precision=b['precision']*2
        if 'multiplier' in b:
            lo,hi=pi_interval(precision)
            scale=Q(int(b['multiplier'][0]),int(b['multiplier'][1]))
            lo,hi=lo*scale,hi*scale
        else:
            argument=Q(int(b['argument'][0]),int(b['argument'][1]))
            lo,hi=interval(argument,b['function'],precision)
            if b['negate']: lo,hi=-hi,-lo
        oldlo,oldhi=(Q(int(v[0]),int(v[1])) for v in (b['bounds']['lower'],b['bounds']['upper']))
        assert oldlo<=lo<=hi<=oldhi
        center=int(b['center'],16)
        assert encode(lo,fmt)==center==encode(hi,fmt)
        b['verified_at_precision']=precision

def generate():
    checks=self_check()
    blob=bytearray()
    ranges=[]
    counts={}
    plans={}
    baseline_total=extra_total=0
    f32_saved=None
    for fmt in FORMATS:
        if fmt=='flex32':
            payload,saved_ranges,saved_counts,saved_plans=f32_saved
            start=len(blob)
            ranges.append([(start+offset,count) for offset,count in saved_ranges])
            blob.extend(payload)
            counts[fmt]=saved_counts
            plans[fmt]=saved_plans
            baseline_total+=sum(c['baseline_count'] for c in saved_counts.values())
            extra_total+=sum(c['additional_closure_count'] for c in saved_counts.values())
            continue
        original,groups,bounds=corpus(fmt)
        verify_boundaries(fmt,bounds)
        width=(sum(FORMATS[fmt])+1)//8
        data=b''.join(bits.to_bytes(width,'little') for bits in original)
        assert len(original)==BASELINE[fmt]['input_count']
        assert hashlib.sha256(data).hexdigest()==BASELINE[fmt]['input_sha256']
        closure,chain_plans=chain_closure(fmt)
        plans[fmt]=chain_plans
        per_op=[]
        fmt_counts={}
        start=len(blob)
        for operation in OPS:
            extras=closure[operation]-set(original)
            inputs=sorted(set(original)|closure[operation])
            per_op.append((len(blob),len(inputs)))
            finite=differences=nonsingle=class_disagreements=nz_disagreements=0
            methods,precisions,classes,transitions=Counter(),Counter(),Counter(),Counter()
            stream=hashlib.sha256()
            for bits in inputs:
                r=reference(bits,fmt,operation)
                row=fixture_row(bits,fmt,operation,r)
                for word in row[:5]: blob.extend(word.to_bytes(width,'little'))
                blob.extend(row[5].to_bytes(4,'little'))
                ic,sc=classify(row[1],fmt),classify(row[2],fmt)
                finite+=decode(bits,fmt)[0]=='finite' and ic not in ('inf','nan')
                differences+=row[1]!=row[2]
                nonsingle+=row[3]!=row[4]
                class_disagreements+=ic!=sc
                nz_disagreements+=(ic in ('normal','subnormal'))!=(sc in ('normal','subnormal'))
                transitions[ic+'->'+sc]+=1
                classes[sc]+=1
                methods[r['method']]+=1
                precisions[str(r['precision'])]+=1
                if bits not in extras: stream.update(canonical([bits,r])+b'\n')
            assert stream.hexdigest()==BASELINE[fmt]['reference_stream_sha256'][operation],(fmt,operation,'baseline stream changed')
            fmt_counts[operation]=dict(all=len(inputs),finite=finite,baseline_count=len(original),
                additional_closure_count=len(extras),additional_input_bits=[code(b,fmt) for b in sorted(extras)],
                baseline_reference_stream_sha256=stream.hexdigest(),direct_staged_differences=differences,
                non_singleton_envelopes=nonsingle,ideal_staged_class_disagreements=class_disagreements,
                ideal_staged_nonzero_finite_disagreements=nz_disagreements,
                ideal_staged_class_transitions=dict(sorted(transitions.items())),classes=dict(sorted(classes.items())),
                method_counts=dict(sorted(methods.items())),precision_counts=dict(sorted(precisions.items())),
                input_groups_overlap=groups,input_boundaries=bounds)
            baseline_total+=len(original)
            extra_total+=len(extras)
            print(fmt,operation,len(inputs),'baseline',len(original),'closure additions',len(extras),flush=True)
        ranges.append(per_op)
        counts[fmt]=fmt_counts
        if fmt=='f32':
            f32_saved=(bytes(blob[start:]),[(offset-start,count) for offset,count in per_op],fmt_counts,chain_plans)
    assert baseline_total==699816
    policy_hash=hashlib.sha256(POLICY_TEXT.encode()).hexdigest()
    fixture_hash=hashlib.sha256(blob).hexdigest()
    rust='// Generated by tools/nvidia/typed-unary-generate.py. Do not edit.\n'
    rust+=f'pub const POLICY_SHA256: &str = "{policy_hash}";\n'
    rust+=f'pub const FIXTURE_SHA256: &str = "{fixture_hash}";\n'
    rust+='const FIXTURE_BYTES: &[u8] = include_bytes!("typed_unary_fixtures.bin");\n'
    rust+='#[rustfmt::skip]\nconst RANGES: [[(usize, usize); 4]; 5] = '+str(ranges)+';\n'
    rust+='#[rustfmt::skip]\nconst CHAIN_PLANS: [[ChainPlan; 40]; 5] = [\n'
    for fmt in FORMATS:
        rust+='[\n'
        for plan in plans[fmt]:
            names=', '.join('Operation::'+{'log1p':'Log1p','sinh':'Sinh','cosh':'Cosh','tan':'Tan'}[op] for op in plan['operations'])
            rust+=f'ChainPlan {{ seed: 0x{plan["seed"]}, operations: [{names}], finite: {str(plan["finite"]).lower()} }},\n'
        rust+='],\n'
    rust+='];\n'
    manifest=dict(version=1,corpus_version=1,schema_version=3,policy_sha256=policy_hash,
        policy=json.loads(POLICY_TEXT),policy_source_bytes=POLICY_TEXT,fixture_sha256=fixture_hash,
        fixture_bytes=len(blob),fixture_rust_sha256=hashlib.sha256(rust.encode()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparation_model_sha256=PREPARATION_MODEL_SHA256,preparation_manifest_sha256=PREPARATION_MANIFEST_SHA256,
        primary_sources=PRIMARY_SOURCES,mpfr_version=VERSION().decode(),
        baseline=BASELINE,baseline_operation_cases=baseline_total,additional_closure_operation_cases=extra_total,
        total_operation_cases=baseline_total+extra_total,counts=counts,chain_plans=plans,
        chain='Each primitive is materialized/read; lookup uses actual stored input. Mixed recipes execute all three operations when selected by any member. No aggregate-chain error bound.',
        selection='--op selects primitive/public operation plus every complete chain containing that operation. --finite retains chains only when every legal closure input and ideal output is finite.',
        binary_encoding='dtype-major f64,f32,flex32,f16,bf16; operation-major log1p,sinh,cosh,tan; sorted unsigned input; five destination-width little-endian words input,ideal,staged,allowed_low,allowed_high then u32 intermediate_f32',
        normalization='Only expected NaNs and semantic chain NaN lookup are canonicalized. Half NaN intermediate is canonical F32 7fc00000. Input/source bytes remain exact.',
        unknown_inputs='Unknown finite stored inputs are observed failures. Arbitrary NaN inputs have exact class-special fallback; no finite approximation fallback.',
        unknown_matrix_accounting='Missing-reference observations occupy the NaN row in both matrices for total bookkeeping only, with separate unreferenced_observed_classes. They always fail and have no finite-pair metric.',
        threshold_proofs='Each counts.<dtype>.<operation>.input_boundaries entry retains directed bounds and a twice-precision recheck. Overflow entries distinguish ideal, half-staged, and F32-intermediate thresholds.',
        accuracy='Independent ideal/staged pair eligibility and transition matrices per actual row. Ideal direct metrics are diagnostic; passing class/special gates use staged contract.',
        reference='Directed MPFR 4.2.2 plus exact rational RNE; no production or host libm expected arithmetic',
        scope='Forward only; no autodiff, GPU accuracy, Linux parity, or performance claim',checks=checks)
    artifacts={
        'os/library/ai/src/validation/typed_unary_fixtures.bin':bytes(blob),
        'os/library/ai/src/validation/typed_unary_fixtures.rs':rust.encode(),
        'tools/nvidia/typed-unary-manifest.json':json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n'}
    return artifacts

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    if args.check == (args.output_root is not None):
        parser.error('Choose read-only --check or an explicit --output-root')
    root=Path(__file__).resolve().parents[2] if args.check else args.output_root
    for relative,data in generate().items():
        target=root/relative
        if args.check:
            assert target.read_bytes()==data,str(target)+' differs'
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or target.read_bytes()!=data:
                target.write_bytes(data)
    print('PASS: deterministic independent reference artifacts',flush=True)
