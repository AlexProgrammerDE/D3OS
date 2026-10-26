#!/usr/bin/env python3
"""Reproduce typed Tanh/Sin/Cos fixtures using Python stdlib and MPFR 4.2.2.

Use --check for read-only reproduction; writes require --output-root.
Inputs are exact stored dyadics. Directed MPFR intervals and rational RNE
produce references independently of production arithmetic and host libm.
"""
import argparse
import ctypes as C
import ctypes.util
from collections import Counter
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import struct

FORMATS = {'f64': (11, 52), 'f32': (8, 23), 'flex32': (8, 23),
           'f16': (5, 10), 'bf16': (8, 7)}
OPS = {'tanh': 18, 'sin': 19, 'cos': 20}
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
FUNCTIONS = {op: bind(op, [P, P, C.c_int]) for op in (*OPS, 'atanh')}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()

def info(fmt):
    e, f = FORMATS[fmt]
    return e, f, (1 << (e-1))-1, 1 << (e+f), ((1 << e)-1) << f

def pow2(k):
    return Q(1 << max(k, 0), 1 << max(-k, 0))
def floor_log2(q):
    q = abs(q)
    k = q.numerator.bit_length() - q.denominator.bit_length()
    return k - int(q < pow2(k))
def decode(bits, fmt):
    e, f, bias, sign, inf = info(fmt)
    assert 0 <= bits < sign * 2
    field, fraction = (bits >> f) & ((1 << e)-1), bits & ((1 << f)-1)
    if field == (1 << e)-1:
        return ('nan' if fraction else 'inf'), (-1 if bits & sign else 1)
    n = fraction if field == 0 else (1 << f) + fraction
    q = n * pow2((1-bias if field == 0 else field-bias)-f)
    return 'finite', (-q if bits & sign else q)
def rne_integer(q):
    n, r = divmod(q.numerator, q.denominator)
    return n + int(2*r > q.denominator or (2*r == q.denominator and n & 1))
def encode(q, fmt, negative_zero=False):
    e, f, bias, sign, inf = info(fmt)
    s = sign if q < 0 or (not q and negative_zero) else 0
    q = abs(q)
    if not q:
        return s
    step = max(floor_log2(q)-f, 1-bias-f)
    n = rne_integer(q / pow2(step))
    if not n:
        return s
    if n >= 1 << (f+1):
        n >>= 1
        step += 1
    if step+f > bias:
        return s | inf
    if n < 1 << f:
        return s | n
    return s | ((step+f+bias) << f) | (n-(1 << f))
def code(bits, fmt):
    return None if bits is None else format(bits, f'0{(sum(FORMATS[fmt])+1)//4}x')
def classify(bits, fmt):
    if bits is None:
        return 'nan'
    e, f, bias, sign, inf = info(fmt)
    magnitude = bits & (sign-1)
    if magnitude > inf:
        return 'nan'
    if magnitude == inf:
        return 'inf'
    if magnitude == 0:
        return 'zero'
    return 'subnormal' if magnitude < (1 << f) else 'normal'
def rank(bits, fmt):
    sign = info(fmt)[3]
    return (~bits & (2*sign-1)) if bits & sign else bits | sign
def unrank(value, fmt):
    sign = info(fmt)[3]
    return value & (sign-1) if value & sign else ~value & (2*sign-1)
def rational(x):
    exponent = C.c_long()
    ptr = GET(None, C.byref(exponent), 2, 0, C.byref(x), 0)
    try:
        digits = C.string_at(ptr).decode()
    finally:
        FREE(ptr)
    negative = digits.startswith('-')
    digits = digits.lstrip('-')
    q = int(digits, 2) * pow2(exponent.value-len(digits))
    return -q if negative else q

def interval(q, operation, precision):
    values = [MP(), MP(), MP()]
    for value in values:
        INIT(C.byref(value), precision)
    x, lo, hi = values
    try:
        assert q.denominator & (q.denominator-1) == 0
        assert SET(C.byref(x), str(q.numerator).encode(), 10, 0) == 0
        assert MUL2(C.byref(x), C.byref(x), -(q.denominator.bit_length()-1), 0) == 0
        assert rational(x) == q
        FUNCTIONS[operation](C.byref(lo), C.byref(x), 3)
        FUNCTIONS[operation](C.byref(hi), C.byref(x), 2)
        return rational(lo), rational(hi)
    finally:
        for value in values:
            CLEAR(C.byref(value))

def pi_interval(precision):
    values = [MP(), MP()]
    for value in values:
        INIT(C.byref(value), precision)
    try:
        PI(C.byref(values[0]), 3)
        PI(C.byref(values[1]), 2)
        return tuple(rational(x) for x in values)
    finally:
        for value in values:
            CLEAR(C.byref(value))

def certified(q, operation, fmt, minimum=128, keep_bounds=False):
    # This conservative input-exponent budget also covers the separate pi corpus
    # construction. MPFR itself supplies correctly rounded full-range functions.
    p = max(minimum, 2*max(0, floor_log2(q))+256 if operation in ('sin', 'cos') and q else minimum)
    while p <= 8192:
        lo, hi = interval(q, operation, p)
        ideal = encode(lo, fmt)
        center = encode(lo, 'f32') if fmt in ('f16', 'bf16') else None
        if ideal == encode(hi, fmt) and (center is None or center == encode(hi, 'f32')):
            staged = encode(decode(center, 'f32')[1], fmt, bool(center & info('f32')[3])) if center is not None else None
            result = {'ideal': ideal, 'staged': staged, 'f32': center,
                      'precision': p, 'method': 'directed_mpfr'}
            if keep_bounds:
                result['bounds'] = {'lower': [str(lo.numerator), str(lo.denominator)],
                                    'upper': [str(hi.numerator), str(hi.denominator)]}
            return result
        p *= 2
    raise RuntimeError(('uncertified interval', fmt, operation, str(q)))

def reference(bits, fmt, operation, minimum=128, keep_bounds=False):
    kind, q = decode(bits, fmt)
    sign = info(fmt)[3]
    half = fmt in ('f16', 'bf16')
    result = None
    method = 'exact_special'
    if kind == 'nan' or (kind == 'inf' and operation != 'tanh'):
        ideal = None
    elif kind == 'inf':
        ideal = encode(Q(q), fmt)
    elif not q:
        ideal = encode(Q(1), fmt) if operation == 'cos' else bits
    elif operation == 'tanh' and abs(q) >= 64:
        # 1-tanh(x)=2/(exp(2x)+1)<2^-127 for x>=64, since e>2.
        # This lies below the midpoint gap 2^-54 at binary64 one.
        ideal = encode(Q(-1 if q < 0 else 1), fmt)
        method = 'analytic_saturation_ge_64'
    else:
        result = certified(q, operation, fmt, minimum, keep_bounds)
    if result is None:
        center = None if ideal is None else encode(decode(ideal, fmt)[1], 'f32', bool(bits & sign)) if half else None
        result = {'ideal': ideal, 'staged': ideal if half else None, 'f32': center,
                  'precision': 0, 'method': method}
    if half:
        if result['method'] == 'exact_special':
            allowed = [] if result['ideal'] is None else [result['ideal']]
        else:
            c = result['f32']
            allowed = sorted({encode(decode(b, 'f32')[1], fmt, bool(b & info('f32')[3]))
                              for r in range(rank(c, 'f32')-2, rank(c, 'f32')+3)
                              for b in [unrank(r, 'f32')] if classify(b, 'f32') in ('zero', 'subnormal', 'normal')})
        result['allowed'] = allowed
    return result

def boundary_center(fmt, kind, parameter):
    # Correctly rounded input centers are independently enclosed, then neighboring
    # stored words are included. Mathematical pi is never a host double constant.
    p = max(256, 2*abs(parameter)+256) if kind == 'pi-power' else 256
    while p <= 8192:
        if kind.startswith('pi'):
            lo, hi = pi_interval(p)
            scale = pow2(parameter) if kind == 'pi-power' else Q(parameter, 2)
            lo, hi = lo*scale, hi*scale
        else:
            # Midpoint before 1: 1-2^(-f-2); atanh gives the ideal
            # destination saturation transition, not the compiler's threshold.
            f = FORMATS[fmt][1] if kind == 'destination-tanh' else 23
            lo, hi = interval(1-pow2(-f-2), 'atanh', p)
        a, b = encode(lo, fmt), encode(hi, fmt)
        if a == b:
            return a, p
        p *= 2
    raise RuntimeError(('uncertified corpus boundary', fmt, kind, parameter))

def corpus(fmt):
    if fmt in ('f16', 'bf16'):
        return range(65536), {'all-storage-encodings': 65536}, []
    e, f, bias, sign, inf = info(fmt)
    groups = {}
    boundaries = []
    def add(bits, tag):
        if 0 <= bits < 2*sign:
            groups.setdefault(bits, set()).add(tag)
    def around(center, tag, radius=4):
        # Sign-magnitude neighborhoods around a positive finite center.
        if center >= inf:
            return
        for d in range(-radius, radius+1):
            if 0 <= center+d < inf:
                for s in (0, sign):
                    add(s | (center+d), tag)
    for s in (0, sign):
        for b in (0, 1, 2, 3, 4, (1 << f)-2, (1 << f)-1, 1 << f,
                  (1 << f)+1, inf-2, inf-1, inf, inf+1, inf+(1 << (f-1)),
                  inf+(1 << (f-1))+21, inf+(1 << f)-1):
            add(s | b, 'special-and-subnormal')
        for exponent in range(1, (1 << e)-1):
            for fraction in (0, (1 << f)-1):
                add(s | (exponent << f) | fraction, 'every-exponent-edge')
    for n in (Q(1, 2), Q(3, 5), 1, 2, 4, 5, 8, 9, 10, 16, 19, 20, 32, 64,
              105615, 2147483648):
        around(encode(Q(n), fmt), 'arithmetic-and-schedule-neighborhood', 8)
    # Exact SASS threshold centers. F64 tanh compares the input high word.
    sass_centers = (0x3fe4f92224dd2f1a, 0x40330fc100000000, 0x40330fc1ffffffff,
                    0x40330fc200000000) if fmt == 'f64' else (0x3f19999a, 0x41102cb4)
    for b in sass_centers:
        around(b, 'observed-sass-boundary', 8)
    for k in range(1, 33):
        b, p = boundary_center(fmt, 'pi-half-multiple', k)
        around(b, 'pi-half-multiple-cancellation', 4)
        boundaries.append({'kind': 'pi-half-multiple', 'k': k, 'center': code(b, fmt), 'precision': p})
    for k in (10, 20, 30, 50, 100, 120, 500, 900, 1000, 1020):
        if k < bias:
            b, p = boundary_center(fmt, 'pi-power', k)
            around(b, 'huge-pi-power-reduction', 4)
            boundaries.append({'kind': 'pi-power', 'k': k, 'center': code(b, fmt), 'precision': p})
    for kind in ('destination-tanh', 'f32-tanh'):
        b, p = boundary_center(fmt, kind, 0)
        around(b, kind, 16)
        boundaries.append({'kind': kind, 'center': code(b, fmt), 'precision': p})
    for k in (100, 1000, 1023) if fmt == 'f64' else (100, 126, 127):
        around(encode(pow2(k), fmt), 'huge-dyadic-neighbors', 8)
    around(inf-1, 'max-finite-neighbors', 16)
    if fmt == 'f64':
        # Frozen relocation-probe input encodings, not host decimal conversions.
        for b in (0x0000000000000000, 0x8000000000000000, 0x3fe0000000000000,
                  0xbfe0000000000000, 0x3ff0000000000000, 0xc000000000000000,
                  0x4045200000000000, 0x4202a05f20000000, 0x54b249ad2594c37d,
                  0x7e37e43c8800759c, 0x400921fb54442d18, 0x3ff921fb54442d18):
            add(b, 'frozen-relocation-probe')
    # Stable bit-pattern sampling includes both signs and all input classes.
    state = hashlib.sha256((('f32' if fmt == 'flex32' else fmt)+' typed-trig-v1').encode()).digest()
    for _ in range(256):
        state = hashlib.sha256(state).digest()
        add(int.from_bytes(state, 'little') % (2*sign), 'sha256-bit-patterns')
    counts = Counter(tag for tags in groups.values() for tag in tags)
    return sorted(groups), dict(sorted(counts.items())), boundaries

def portable(result, fmt):
    return {'ideal_bits': code(result['ideal'], fmt),
            'staged_bits': code(result['staged'], fmt),
            'f32_ideal_bits': code(result['f32'], 'f32'),
            'allowed_destination_bits': [code(b, fmt) for b in result.get('allowed', [])]}

def qualify(bits, actual, fmt, operation):
    """Evaluate independent project gates, including gates stricter than ULP."""
    result = reference(bits, fmt, operation)
    expected = result['staged'] if fmt in ('f16', 'bf16') else result['ideal']
    expected_class, actual_class = classify(expected, fmt), classify(actual, fmt)
    failures = []
    diagnostics = []
    distance = None
    membership = None
    if expected_class != actual_class:
        if {expected_class, actual_class} <= {'normal', 'subnormal'}:
            diagnostics.append('normal_subnormal_transition')
        else:
            failures.append('class_mismatch')
    if expected_class == 'zero' and actual != expected:
        failures.append('signed_zero_mismatch')
    if expected_class in ('normal', 'subnormal') and actual_class in ('normal', 'subnormal'):
        if bool(expected & info(fmt)[3]) != bool(actual & info(fmt)[3]):
            failures.append('finite_sign_mismatch')
        distance = abs(rank(actual, fmt)-rank(expected, fmt))
    if actual_class in ('normal', 'subnormal') and abs(decode(actual, fmt)[1]) > 1:
        failures.append('outside_unit_range')
    if fmt in ('f16', 'bf16') and expected_class != 'nan':
        membership = actual in result['allowed']
        if not membership:
            failures.append('half_envelope_mismatch')
    elif distance is not None:
        budget = 1 if fmt == 'f64' and operation == 'tanh' else 2
        if distance > budget:
            failures.append('finite_step_limit')
    if result['method'] == 'exact_special' and expected_class != 'nan' and actual != expected:
        failures.append('exact_special_mismatch')
    return {'failures': failures, 'diagnostics': diagnostics, 'finite_step_distance': distance,
            'half_envelope_membership': membership}

def self_check():
    count = 0
    for fmt in ('f16', 'bf16'):
        for b in range(65536):
            kind, q = decode(b, fmt)
            if kind == 'finite':
                assert encode(q, fmt, bool(b & info(fmt)[3])) == b
                assert decode(encode(q, 'f32', bool(b & info(fmt)[3])), 'f32')[1] == q
                # Independent format decoding only, never a transcendental oracle.
                host = struct.unpack('<e', b.to_bytes(2, 'little'))[0] if fmt == 'f16' else struct.unpack('<f', (b << 16).to_bytes(4, 'little'))[0]
                assert Q(host) == q
                count += 1
    for fmt in FORMATS:
        e, f, bias, sign, inf = info(fmt)
        m = decode(1, fmt)[1]
        one = encode(Q(1), fmt)
        assert encode(m/2, fmt) == 0
        assert encode(3*m/2, fmt) == 2
        assert encode(-m/2, fmt) == sign
        assert encode((decode(one, fmt)[1]+decode(one+1, fmt)[1])/2, fmt) == one
        maximum = decode(inf-1, fmt)[1]
        assert encode(maximum+(maximum-decode(inf-2, fmt)[1])/2, fmt) == inf
        for b in (0, sign, 1, sign | 1, one, inf, sign | inf, inf+1):
            for op in OPS:
                r = reference(b, fmt, op)
                if fmt in ('f16', 'bf16') and r['ideal'] is not None:
                    assert r['staged'] in r['allowed']
        assert reference(sign, fmt, 'sin')['ideal'] == sign
        assert reference(sign, fmt, 'tanh')['ideal'] == sign
        assert reference(sign, fmt, 'cos')['ideal'] == one
        assert 'class_mismatch' in qualify(1, 0, fmt, 'sin')['failures']
        assert 'signed_zero_mismatch' in qualify(sign, 0, fmt, 'tanh')['failures']
        assert 'outside_unit_range' in qualify(inf, one+1, fmt, 'tanh')['failures']
    return {'finite_half_roundtrips_and_struct_decode': count,
            'ties_gradual_underflow_overflow_signed_zero': True,
            'all_finite_half_promotions_exact': True,
            'independent_class_sign_range_adversarial_checks': True}

POLICY_TEXT = r'''{
  "classification_qualification_target": {
    "classes": "require matching zero/nonzero, infinity and NaN classes; normal and subnormal finite values share numerical qualification",
    "consistency_with_exp_log_v2": "preserve the accepted classification policy; no stricter automatic normal/subnormal boundary failure",
    "normal_subnormal_boundary": "record finite normal/subnormal changes as diagnostics; accept within the finite step or half neighbor envelope",
    "range": "finite outputs must lie in [-1,1]; unexpected infinity always fails",
    "sign": "matching sign for every finite nonzero and zero output; NaN output sign/payload not specified",
    "status": "independent project gates can be stricter than vendor empirical accuracy figures"
  },
  "comparator_order": [
    "check structure and dtype",
    "classify stored raw bits",
    "check exact specials",
    "apply independent exact class/sign/range gates",
    "apply finite step or half destination-image gate",
    "record ideal/staged differences and every boundary transition",
    "record observed metrics separately"
  ],
  "exact_special_policy": {
    "cos": {
      "nan": "nan_class",
      "negative_infinity": "nan_class",
      "negative_zero": "positive_one",
      "positive_infinity": "nan_class",
      "positive_zero": "positive_one"
    },
    "sin": {
      "nan": "nan_class",
      "negative_infinity": "nan_class",
      "negative_zero": "negative_zero",
      "positive_infinity": "nan_class",
      "positive_zero": "positive_zero"
    },
    "tanh": {
      "nan": "nan_class",
      "negative_infinity": "negative_one",
      "negative_zero": "negative_zero",
      "positive_infinity": "positive_one",
      "positive_zero": "positive_zero"
    }
  },
  "expected_summary_excludes": [
    "max_observed_ulp",
    "histograms",
    "worst_input_bits",
    "classification_mismatch_counts",
    "measured_gpu_metadata",
    "pass_fail",
    "runtime_timings",
    "half_membership_failures"
  ],
  "expected_summary_fields": [
    "schema_version",
    "policy_sha256",
    "manifest_sha256",
    "selection_options",
    "selected_scalar_case_count",
    "selected_dtype_operation_counts",
    "planned_structural_case_counts"
  ],
  "expected_summary_rule": "expected equality contains immutable corpus/selection/structure facts only; no mutable maxima or run observations",
  "finite_metric": {
    "boundary": "representable-step metric, explicitly not a real-error/local-spacing ULP definition",
    "distance": "abs(rank(actual)-rank(reference)), finite nonzero pairs only",
    "rank": "positive b -> b|sign; negative b -> (~b)&(2*sign-1)"
  },
  "half_acceptance_set": {
    "baseline_for_class_and_sign": "staged destination reference, not ideal direct destination rounding",
    "center": "c = interval-certified RN32(f(exact stored half input))",
    "construction": "enumerate ranks rank(c)-2 through rank(c)+2, convert each finite F32 value once to destination RNE, deduplicate",
    "ideal_difference": "distance from ideal and staged references is diagnostic; neither changes the acceptance set",
    "independent_checks": "zero/nonzero, Inf/NaN class, sign, signed zero, no FTZ, and abs(actual)<=1 all apply even when membership succeeds",
    "interpretation": "membership is necessary, not proof of the GPU intermediate or its schedule",
    "specials": "exact special inputs use singleton bits or NaN class; no neighbor widening"
  },
  "no_ftz_policy": {
    "cos_limit": "cos(tiny) rounds to one, so those outputs alone cannot detect input FTZ; Sin/Tanh provide sensitivity",
    "evidence": "no public FTZ modifier or visible SASS FTZ is not runtime proof",
    "input_checks": "all half finite subnormal inputs included; wide signed subnormal and normal boundary neighborhoods included",
    "rule": "expected nonzero to zero fails class qualification, including minimum subnormal code 1 to code 0"
  },
  "numerical_criteria": [
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 1,
      "dtype": "f64",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 1,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "tanh",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": 1,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "f64",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "sin",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "f64",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "cos",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "f32",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "tanh",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": 2,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "f32",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "sin",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "f32",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "cos",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "flex32",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "tanh",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": 2,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "flex32",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "sin",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": null,
      "cuda_math_reported_maximum_ulp": 2,
      "dtype": "flex32",
      "f32_intermediate_step_limit": null,
      "finite_step_limit": 2,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "cos",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": 0,
      "cuda_math_reported_maximum_ulp": null,
      "dtype": "f16",
      "f32_intermediate_step_limit": 2,
      "finite_step_limit": null,
      "mode": "f32_neighbor_destination_image",
      "operation": "tanh",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": 0,
      "cuda_math_reported_maximum_ulp": null,
      "dtype": "f16",
      "f32_intermediate_step_limit": 2,
      "finite_step_limit": null,
      "mode": "f32_neighbor_destination_image",
      "operation": "sin",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": 0,
      "cuda_math_reported_maximum_ulp": null,
      "dtype": "f16",
      "f32_intermediate_step_limit": 2,
      "finite_step_limit": null,
      "mode": "f32_neighbor_destination_image",
      "operation": "cos",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": 0,
      "cuda_math_reported_maximum_ulp": null,
      "dtype": "bf16",
      "f32_intermediate_step_limit": 2,
      "finite_step_limit": null,
      "mode": "f32_neighbor_destination_image",
      "operation": "tanh",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": 0,
      "cuda_math_reported_maximum_ulp": null,
      "dtype": "bf16",
      "f32_intermediate_step_limit": 2,
      "finite_step_limit": null,
      "mode": "f32_neighbor_destination_image",
      "operation": "sin",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    },
    {
      "cuda_math_half_api_reported_ulp_not_transferred": 0,
      "cuda_math_reported_maximum_ulp": null,
      "dtype": "bf16",
      "f32_intermediate_step_limit": 2,
      "finite_step_limit": null,
      "mode": "f32_neighbor_destination_image",
      "operation": "cos",
      "project_qualification_target": true,
      "tileir_full_documented_ulp": null,
      "vendor_guaranteed": false
    }
  ],
  "policy_id": "burn-typed-trig-reference-v1-proposed",
  "production_risks": {
    "audit_compatibility": "prepare-kernels.py lists burn_float_arithmetic in a blanket LDL/STL rejection gate; observed F64 Sin/Cos helper ABI stack can conflict with that assumption; next full-entry qualification must distinguish ABI stack and actual spills",
    "f64_sin_cos": "two live mathematical CB4 pointers to the 272-byte global table; observed allocated REL numeric type 2",
    "f64_tanh": "vendor float reduction seed plus retained double residual/polynomial; no frontend narrowing replacement",
    "loader_evidence_limit": "user-provided status, not independently inspected here; 12 input witnesses only; full typed trig entry and full regression qualification remain pending",
    "loader_status": "parent reports real-GPU relocation-probe PASS: 48 comparisons, maximum 1 ULP, guards/tails/source/cache and three guest code hashes match; bundle 248 native CB4 data pointers work on this fixture",
    "release_gate": "reference preparation alone cannot qualify Burn, compiler changes, loader, Linux parity, or GPU arithmetic"
  },
  "provenance": {
    "accuracy_figure_status": "CUDA tables report observed maxima from nonexhaustive testing, not guarantees.",
    "bytecode": "V13_4",
    "compiler_evidence": "typed-trig-compiler-qualification.txt and hashed IR/SASS evidence",
    "cutile_pin": "2f47ad0036a7c18ec11d460f70b297a5824f9788",
    "half_status": "CUDA hsin/hcos/htanh 0-ULP table entries do not qualify this explicit F32 emulation schedule.",
    "qualified_artifact_target": "sm_89",
    "transfer_status": "Sin/Cos CUDA Math figures are explicit project targets for TileIR; end-to-end proof remains pending.",
    "vendor_tools": "13.4.92"
  },
  "references": {
    "bf16": "primary staged RNB16(RN32(f(exactB16))); retain ideal RNB16(f(exactB16)) separately",
    "f16": "primary staged RN16(RN32(f(exact16))); retain ideal RN16(f(exact16)) separately",
    "f32": "RN32(f(exact32))",
    "f64": "RN64(f(exact64)); tanh Full is the vendor F64 function, not RN64(f(RN32(x)))",
    "flex32": "same values as F32; preserve Flex32 metadata; no TF32 input quantization",
    "half_execution_schedule": "exact stored half -> explicit RNE F32 promotion -> F32 Sin/Cos or Tanh Full -> exactly one destination RNE",
    "half_schedule_evidence": "F2FP packing is allowed only when each logical value has one destination rounding.",
    "inputs": "integer stored bit patterns decoded to exact dyadic rationals; no decimal/host-libm input oracle",
    "nan": "null ideal bits represent NaN class; output sign/payload/quiet-bit are observations only"
  },
  "schema_version": 1,
  "scope": {
    "autodiff": "full autodiff is already authorized; this forward-only slice makes no backward claim; derivatives and reductions need separate reference qualification",
    "direction": "forward only",
    "existing_f32_validator": "separate until explicit integration",
    "formats": [
      "f64",
      "f32",
      "flex32",
      "f16",
      "bf16"
    ],
    "operation_ids": {
      "cos": 20,
      "sin": 19,
      "tanh_full": 18
    },
    "priority": "Full Burn before optimization"
  },
  "selection_options": {
    "dtype": "one requested format or all",
    "finite": "finite input and finite ideal output, including zero and subnormals",
    "no_flex32": "omit only Flex32 metadata path",
    "operation": "18/19/20 or tanh/sin/cos; Tanh always Full",
    "skip_rank_zero": "omit rank-0 layout only; retain rank-1 [1]"
  },
  "source_urls": {
    "cuda_guide": "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html",
    "cuda_guide_captured_version": "13.4.2",
    "cuda_math_double": "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html",
    "cuda_math_single": "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html",
    "mpfr": "https://www.mpfr.org/mpfr-current/mpfr.html",
    "tileir": "https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html"
  },
  "status": "cache-only proposed qualification policy; no production arithmetic, compilation, or GPU result"
}
'''
BASELINE = {
  "bf16": {
    "input_count": 65536,
    "input_sha256": "68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b",
    "operations": {
      "cos": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 2,
        "reference_stream_sha256": "37863a1653cd72c08b19330238f343872c08997c49e805bcf1ced809b3dc720f"
      },
      "sin": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 4,
        "reference_stream_sha256": "edf64c44bcff10d4de9acb582bffcc33e6acb83da980891c1d0658128dbb6b56"
      },
      "tanh": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "4c18fe0bc18e95049bd39bb92d5e82b75314706d2318d551f5d74ca512437b46"
      }
    }
  },
  "f16": {
    "input_count": 65536,
    "input_sha256": "68e419472d25e0b85e9917ccf692fd58245c5e95e9a46f07d1df81d2e9da246b",
    "operations": {
      "cos": {
        "ideal_staged_difference_count": 2,
        "non_singleton_envelope_count": 26,
        "reference_stream_sha256": "55e39a377b9283c1a08c04eed629e44e640a0b6ccf11887ba07ad93867eb417b"
      },
      "sin": {
        "ideal_staged_difference_count": 4,
        "non_singleton_envelope_count": 22,
        "reference_stream_sha256": "e88c4e00eb641a8f01da6693d0497030e4576e1a337c1a90466c4def67dbfc08"
      },
      "tanh": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 4,
        "reference_stream_sha256": "3556e47c896bc52bb4f54eae6c843971060e117235f3e035a7c02d50899dc3fd"
      }
    }
  },
  "f32": {
    "input_count": 2678,
    "input_sha256": "60d9286acbbd8c38263dd89f78b2e1b1fcba33ea84834b041cdd5bcfe2ad5ee5",
    "operations": {
      "cos": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "26c6ecb5df1fa0a13c1e10602e75e0d516d6f82c185da51ef30b15b5430f0c92"
      },
      "sin": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "e36de8916a574e2136bec082684e19bd51de748614bf8160e997f29a0e8ad06b"
      },
      "tanh": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "23662998ead37af7860f6312860f6b1c2c1a7fa91715c2e2481674219bc6ff10"
      }
    }
  },
  "f64": {
    "input_count": 10092,
    "input_sha256": "6b029502106ebbf683e5120e9965c23aba25b8ed352a50d6e7b58c90678315b6",
    "operations": {
      "cos": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "18b310828ef2b9c05f44d503553d546daa3a098bbc7183d64d08b26044553715"
      },
      "sin": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "d3b1586400ef2b30788cf3f3069ae1f22ad098b00886d0907d4e6a737bea4b61"
      },
      "tanh": {
        "ideal_staged_difference_count": 0,
        "non_singleton_envelope_count": 0,
        "reference_stream_sha256": "367e93c05032b064dacd31cf9e31257c48e94a01c5ca6f45f754722288fc1197"
      }
    }
  }
}
PRIMARY_SOURCES = {
  "cuda-guide.html": {
    "bytes": 245348,
    "retrieved_utc_date": "2026-10-26",
    "sha256": "4277615c7d34b1f24fa9dd7908b7f71e63cc87d516740cc89bf56f05b5be7f77",
    "url": "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html"
  },
  "cuda-math-double.html": {
    "bytes": 245449,
    "retrieved_utc_date": "2026-10-26",
    "sha256": "33fedd606826502cf91036bc5f054dc3ef266d3228930ca667d30e5fa1fa7855",
    "url": "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html"
  },
  "cuda-math-single.html": {
    "bytes": 243054,
    "retrieved_utc_date": "2026-10-26",
    "sha256": "de1b072026dae5d016001e21574daa1cb566ed21c424dccc29b069bad21b4195",
    "url": "https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html"
  },
  "mpfr-manual.html": {
    "bytes": 562991,
    "retrieved_utc_date": "2026-10-26",
    "sha256": "ee996f3dcc11d1f88393da162bdd90c3fa29191ca658c0fc594ffd38a22638bc",
    "url": "https://www.mpfr.org/mpfr-current/mpfr.html"
  },
  "tileir-operations.html": {
    "bytes": 1177348,
    "retrieved_utc_date": "2026-10-26",
    "sha256": "b05512d7cc3b1571adacb29ab40e209ade6e826fc8d38cb0b760881e2b281773",
    "url": "https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html"
  }
}
PREPARATION_MODEL_SHA256 = 'a42e77a6efa2b586ae26c04c35175e3d0edc5290a313230f27a0f7778aedaffd'
PREPARATION_MANIFEST_SHA256 = '12f2e4d8a7ce0473f178c7c6317791158b94dd827729240b5479cbfc2890dfd1'

def canonical_nan(fmt):
    return info(fmt)[4] | (1 << (FORMATS[fmt][1]-1))

def fixture_row(bits, fmt, op, result=None):
    r = reference(bits, fmt, op) if result is None else result
    ideal = canonical_nan(fmt) if r['ideal'] is None else r['ideal']
    staged = ideal if r['staged'] is None else r['staged']
    allowed = sorted(r.get('allowed') or [staged], key=lambda b: rank(b, fmt))
    # Monotonic destination RNE maps five adjacent F32 codes to consecutive
    # destination codes. Endpoints therefore encode the exact image, not a hull.
    assert all(rank(b, fmt) == rank(a, fmt)+1 for a, b in zip(allowed, allowed[1:]))
    intermediate = r['f32']
    if intermediate is None:
        intermediate = canonical_nan('f32') if fmt in ('f16', 'bf16') and r['ideal'] is None else 0
    return bits, ideal, staged, allowed[0], allowed[-1], intermediate

def legal_successors(bits, fmt, op):
    r = reference(bits, fmt, op)
    expected = r['staged'] if fmt in ('f16', 'bf16') else r['ideal']
    wanted = classify(expected, fmt)
    if wanted == 'nan':
        return []
    if r['method'] == 'exact_special' or wanted == 'zero':
        candidates = [expected]
    elif fmt in ('f16', 'bf16'):
        candidates = r['allowed']
    else:
        budget = 1 if fmt == 'f64' and op == 'tanh' else 2
        candidates = [unrank(n, fmt) for n in range(rank(expected, fmt)-budget, rank(expected, fmt)+budget+1)]
    # Independent class/sign/codomain gates constrain which intermediates may
    # reach another primitive, including at +/-one and normal/subnormal edges.
    return sorted(b for b in candidates
                  if classify(b, fmt) in ('zero', 'normal', 'subnormal')
                  and (classify(b, fmt) == wanted or {classify(b, fmt), wanted} <= {'normal', 'subnormal'})
                  and bool(b & info(fmt)[3]) == bool(expected & info(fmt)[3])
                  and abs(decode(b, fmt)[1]) <= 1)

def chain_closure(fmt, op):
    seeds = [0, encode(Q(1), fmt), encode(Q(64), fmt)]
    levels = [set(seeds)]
    for _ in range(2):
        levels.append({b for seed in levels[-1] for b in legal_successors(seed, fmt, op)})
    inputs = set().union(*levels)
    assert len(inputs) <= 93  # three seeds * (1 + 5 + 25)
    return inputs, {'depth': 3, 'seeds': [code(b, fmt) for b in seeds],
                    'input_levels': [[code(b, fmt) for b in sorted(level)] for level in levels],
                    'unique_input_count': len(inputs), 'maximum_unique_input_count': 93}

def generate():
    assert VERSION().decode() == '4.2.2', 'Pinned reproduction requires MPFR 4.2.2'
    checks = self_check()
    blob, ranges, counts = bytearray(), [], {}
    base_total, extra_total = 0, 0
    f32_saved = None
    for fmt in FORMATS:
        if fmt == 'flex32':
            saved_ranges, saved_counts, payload = f32_saved
            base = len(blob)
            ranges.append([(base+offset, count) for offset, count in saved_ranges])
            blob.extend(payload)
            counts[fmt] = saved_counts
            base_total += sum(c['baseline_count'] for c in saved_counts.values())
            extra_total += sum(c['additional_closure_count'] for c in saved_counts.values())
            continue
        original, groups, boundaries = corpus(fmt)
        original = list(original)
        width = (sum(FORMATS[fmt])+1)//8
        original_bytes = b''.join(b.to_bytes(width, 'little') for b in original)
        assert len(original) == BASELINE[fmt]['input_count']
        assert hashlib.sha256(original_bytes).hexdigest() == BASELINE[fmt]['input_sha256']
        start = len(blob)
        per_op, fmt_counts = [], {}
        for op in OPS:
            closure, closure_info = chain_closure(fmt, op)
            additional = sorted(closure-set(original))
            inputs = sorted(set(original) | closure)
            per_op.append((len(blob), len(inputs)))
            classes, precisions, methods = Counter(), Counter(), Counter()
            finite_count, differences, nonsingle = 0, 0, 0
            stream = hashlib.sha256()
            extra_set = set(additional)
            for bits in inputs:
                r = reference(bits, fmt, op)
                words = fixture_row(bits, fmt, op, r)
                for word in words[:5]:
                    blob.extend(word.to_bytes(width, 'little'))
                blob.extend(words[5].to_bytes(4, 'little'))
                classes[classify(words[2], fmt)] += 1
                precisions[str(r['precision'])] += 1
                methods[r['method']] += 1
                finite_count += decode(bits, fmt)[0] == 'finite' and classify(words[1], fmt) not in ('nan', 'inf')
                differences += words[1] != words[2]
                nonsingle += words[3] != words[4]
                if bits not in extra_set:
                    stream.update(canonical({'input_bits': code(bits, fmt), **portable(r, fmt)})+b'\n')
            baseline = BASELINE[fmt]['operations'][op]
            assert stream.hexdigest() == baseline['reference_stream_sha256'], (fmt, op, 'baseline changed')
            closure_info['additional_input_bits'] = [code(b, fmt) for b in additional]
            fmt_counts[op] = {'all': len(inputs), 'finite': finite_count, 'baseline_count': len(original),
                             'additional_closure_count': len(additional), 'chain_closure': closure_info,
                             'baseline_reference_stream_sha256': stream.hexdigest(),
                             'direct_staged_differences': differences, 'non_singleton_envelopes': nonsingle,
                             'classes': dict(sorted(classes.items())), 'precision_counts': dict(sorted(precisions.items())),
                             'method_counts': dict(sorted(methods.items())), 'input_groups_overlap': groups,
                             'input_boundaries': boundaries}
            base_total += len(original)
            extra_total += len(additional)
            print(fmt, op, len(inputs), 'baseline', len(original), 'closure additions', len(additional), flush=True)
        ranges.append(per_op)
        counts[fmt] = fmt_counts
        if fmt == 'f32':
            f32_saved = ([(offset-start, count) for offset, count in per_op], fmt_counts, bytes(blob[start:]))
    assert base_total == 439560
    policy_hash = hashlib.sha256(POLICY_TEXT.encode()).hexdigest()
    fixture_hash = hashlib.sha256(blob).hexdigest()
    rust = '// Generated by tools/nvidia/typed-trig-generate.py. Do not edit.\n'
    rust += f'pub const POLICY_SHA256: &str = "{policy_hash}";\n'
    rust += f'pub const FIXTURE_SHA256: &str = "{fixture_hash}";\n'
    rust += 'const FIXTURE_BYTES: &[u8] = include_bytes!("typed_trig_fixtures.bin");\n'
    rust += 'const RANGES: [[(usize, usize); 3]; 5] = '+str(ranges)+';\n'
    manifest = {'version': 1, 'corpus_version': 1, 'schema_version': 1,
                'policy_sha256': policy_hash, 'policy': json.loads(POLICY_TEXT), 'policy_source_bytes': POLICY_TEXT,
                'fixture_sha256': fixture_hash, 'fixture_bytes': len(blob),
                'fixture_rust_sha256': hashlib.sha256(rust.encode()).hexdigest(),
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'preparation_model_sha256': PREPARATION_MODEL_SHA256,
                'preparation_manifest_sha256': PREPARATION_MANIFEST_SHA256,
                'primary_sources': PRIMARY_SOURCES, 'mpfr_version': VERSION().decode(),
                'reference': 'adaptive directed MPFR intervals; exact rational RNE; no decimal/libm/production/GPU expected arithmetic',
                'baseline': BASELINE, 'baseline_operation_cases': base_total,
                'additional_closure_operation_cases': extra_total, 'operation_cases': base_total+extra_total,
                'finite_operation_cases': sum(c['finite'] for f in counts.values() for c in f.values()),
                'counts': counts, 'self_checks': checks,
                'binary_encoding': 'dtype-major f64,f32,flex32,f16,bf16; operations tanh,sin,cos; sorted unsigned inputs; five little-endian dtype-width words input,ideal,staged,allowed_low,allowed_high then u32 intermediate_f32; canonical expected NaNs classify only',
                'chain': 'each primitive materialized; references cover ACTUAL stored bits via offline legal-neighbor closure through depth 3; three seeds 0,1,64 per dtype/op; independent class/sign/codomain gates; no total-chain error budget',
                'shared_scope': 'forward primitive/public routes; actual Tensor ranks 0..8; empties/tails/views; exact retained source/alias/poison bytes; physical output guards/padding/launch counters require raw harness',
                'qualification_status': 'project qualification targets, not vendor guarantees; forward only, no autodiff/GPU/performance/full Burn acceptance'}
    return bytes(blob), rust.encode(), (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error('Writes require an explicit --output-root')
    root = args.output_root or Path(__file__).resolve().parents[2]
    directory = root / 'os/library/ai/src/validation'
    targets = [directory / 'typed_trig_fixtures.bin', directory / 'typed_trig_fixtures.rs',
               root / 'tools/nvidia/typed-trig-manifest.json']
    for target, payload in zip(targets, generate()):
        if args.check:
            assert target.read_bytes() == payload, f'stale fixture: {target}'
        else:
            target.write_bytes(payload)
    print('typed trig fixtures verified' if args.check else 'typed trig fixtures generated')

if __name__ == '__main__':
    main()
