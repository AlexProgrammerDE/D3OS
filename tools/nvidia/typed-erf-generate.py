#!/usr/bin/env python3
"""Independent directed MPFR Erf fixture generator.

reference(bits, fmt) returns integer encoding fields, never approximate host
float values. `ideal_bits` is RN_fmt(erf(exact stored input)); `staged_bits`
for halves is RN_half(RN_f32(erf(exact stored input))). NaN uses a class rule,
not an expected payload. No libm, fdlibm, CUDA or candidate coefficients enter
expected values. All dyadic conversions and destination RNE use integers.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import ctypes as C
import ctypes.util
from fractions import Fraction as Q
from functools import lru_cache

FORMATS = {'f64': (11, 52), 'f32': (8, 23), 'flex32': (8, 23),
           'f16': (5, 10), 'bf16': (8, 7)}
M = C.CDLL(ctypes.util.find_library('mpfr'))
class MP(C.Structure):
    _fields_ = [('prec', C.c_long), ('sign', C.c_int), ('exp', C.c_long),
                ('limbs', C.POINTER(C.c_ulong))]
P = C.POINTER(MP)
def bind(name, args, result=C.c_int):
    fn = getattr(M, 'mpfr_' + name)
    fn.argtypes, fn.restype = args, result
    return fn
init = bind('init2', [P, C.c_long], None)
clear = bind('clear', [P], None)
setstr = bind('set_str', [P, C.c_char_p, C.c_int, C.c_int])
mul2 = bind('mul_2si', [P, P, C.c_long, C.c_int])
erf = bind('erf', [P, P, C.c_int])
erfc = bind('erfc', [P, P, C.c_int])
getstr = bind('get_str', [C.c_void_p, C.POINTER(C.c_long), C.c_int,
                         C.c_size_t, P, C.c_int], C.c_void_p)
free = bind('free_str', [C.c_void_p], None)
version = bind('get_version', [], C.c_char_p)
assert version() == b'4.2.2', version()

def info(fmt):
    e, f = FORMATS[fmt]
    return e, f, (1 << (e - 1)) - 1, 1 << (e + f), ((1 << e) - 1) << f

def dyadic(n, power):
    return Q(n << max(power, 0), 1 << max(-power, 0))

def decode(bits, fmt):
    e, f, bias, sign, inf = info(fmt)
    if not isinstance(bits, int) or not 0 <= bits < sign * 2:
        raise ValueError((bits, fmt))
    field, frac = (bits >> f) & ((1 << e) - 1), bits & ((1 << f) - 1)
    if field == (1 << e) - 1:
        return ('nan' if frac else 'inf'), (-1 if bits & sign else 1)
    n = frac if field == 0 else (1 << f) + frac
    q = dyadic(n, (1 - bias if field == 0 else field - bias) - f)
    return 'finite', -q if bits & sign else q

def rne_int(q):
    n, r = divmod(q.numerator, q.denominator)
    return n + int(2 * r > q.denominator or (2 * r == q.denominator and n & 1))

def encode(q, fmt, negative_zero=False):
    e, f, bias, sign, inf = info(fmt)
    s = sign if q < 0 or (not q and negative_zero) else 0
    q = abs(q)
    if not q:
        return s
    k = q.numerator.bit_length() - q.denominator.bit_length()
    if q < dyadic(1, k):
        k -= 1
    step = max(k - f, 1 - bias - f)
    n = rne_int(q / dyadic(1, step))
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

def as_q(x):
    exponent = C.c_long()
    ptr = getstr(None, C.byref(exponent), 2, 0, C.byref(x), 0)
    try:
        digits = C.string_at(ptr).decode()
    finally:
        free(ptr)
    negative = digits.startswith('-')
    digits = digits.lstrip('-')
    value = dyadic(int(digits, 2), exponent.value - len(digits))
    return -value if negative else value

def interval(q, precision=128, operation='erf'):
    """RNDD and RNDU enclose the exact real operation at exact dyadic q."""
    values = [MP(), MP(), MP()]
    for v in values:
        init(C.byref(v), precision)
    x, lo, hi = values
    try:
        assert q.denominator & (q.denominator - 1) == 0
        n = abs(q.numerator)
        significant_bits = n.bit_length() - ((n & -n).bit_length() - 1) if n else 0
        assert significant_bits <= precision
        assert setstr(C.byref(x), str(q.numerator).encode(), 10, 0) == 0
        assert mul2(C.byref(x), C.byref(x), -(q.denominator.bit_length() - 1), 0) == 0
        assert as_q(x) == q
        fn = {'erf': erf, 'erfc': erfc}[operation]
        fn(C.byref(lo), C.byref(x), 3)  # MPFR_RNDD
        fn(C.byref(hi), C.byref(x), 2)  # MPFR_RNDU
        return as_q(lo), as_q(hi)
    finally:
        for v in values:
            clear(C.byref(v))

@lru_cache(maxsize=200000)
def confirmed_positive(q, fmt, start=128):
    p = start
    while p <= 16384:
        lo, hi = interval(q, p)
        a, b = encode(lo, fmt), encode(hi, fmt)
        c, d = encode(lo, 'f32'), encode(hi, 'f32')
        if a == b and (fmt not in ('f16', 'bf16') or c == d):
            staged = encode(decode(c, 'f32')[1], fmt) if fmt in ('f16', 'bf16') else a
            return a, staged, c if fmt in ('f16', 'bf16') else None, p
        p *= 2
    raise ArithmeticError(('unresolved directed interval', fmt, str(q)))

def ordered(bits, fmt):
    sign = info(fmt)[3]
    return (~bits & (2 * sign - 1)) if bits & sign else bits | sign

def distance(a, b, fmt):
    return abs(ordered(a, fmt) - ordered(b, fmt))

def half_neighbor_images(f32_bits, fmt, steps=2):
    """RNE images of finite same-sign nonzero F32 neighbors inside [-1,1]."""
    assert fmt in ('f16', 'bf16')
    sign = f32_bits & (1 << 31)
    mag = f32_bits & 0x7fffffff
    out = set()
    for n in range(max(1, mag - steps), min(0x3f800000, mag + steps) + 1):
        value = decode(sign | n, 'f32')[1]
        out.add(encode(value, fmt))
    return sorted(out)

def reference(bits, fmt, start=128):
    kind, x = decode(bits, fmt)
    sign = info(fmt)[3]
    s = bits & sign
    base = {'input_bits': bits, 'format': fmt, 'input_class': kind}
    if kind == 'nan':
        return dict(base, expected_class='nan', ideal_bits=None, staged_bits=None,
                    f32_bits=None, precision=0, half_neighbor_images_2=None,
                    half_neighbor_images_8_diagnostic=None)
    if kind == 'inf' or x == 0:
        result = (encode(Q(1), fmt) | s) if kind == 'inf' else s
        f32 = ((0x3f800000 if kind == 'inf' else 0) | (0x80000000 if s else 0))
        return dict(base, expected_class='finite', ideal_bits=result, staged_bits=result,
                    f32_bits=f32 if fmt in ('f16', 'bf16') else None, precision=0,
                    half_neighbor_images_2=[result] if fmt in ('f16', 'bf16') else None,
                    half_neighbor_images_8_diagnostic=[result] if fmt in ('f16', 'bf16') else None)
    a, staged, f32, p = confirmed_positive(abs(x), fmt, start)
    a, staged = a | s, staged | s
    if f32 is not None and s:
        f32 |= 0x80000000
    return dict(base, expected_class='finite', ideal_bits=a, staged_bits=staged,
                f32_bits=f32, precision=p,
                half_neighbor_images_2=half_neighbor_images(f32, fmt) if f32 is not None else None,
                half_neighbor_images_8_diagnostic=half_neighbor_images(f32, fmt, 8) if f32 is not None else None)

def saturation(fmt, staged=False):
    """Binary-search positive stored lattice, then prove the adjacent pair.

    For direct rounding, erf(x) rounds to 1 iff erfc(x) <= 2^(-p-1),
    where p is destination significand precision. For staged half rounding,
    the effective edge is the midpoint of the two F32 values straddling the
    half midpoint. The F32 value at that half midpoint is even and accepted.
    """
    f = info(fmt)[1]
    one = encode(Q(1), fmt)
    def is_one(b):
        r = reference(b, fmt)
        return r['staged_bits' if staged else 'ideal_bits'] == one
    low, high = 0, encode(Q(16), fmt)
    while low + 1 < high:
        mid = (low + high) // 2
        if is_one(mid):
            high = mid
        else:
            low = mid
    threshold = dyadic(1, -f - 2)
    if staged and fmt in ('f16', 'bf16'):
        threshold += dyadic(1, -25)
    # Tail comparison avoids cancellation of 1-erf(x).
    proof = []
    for b in [low, high]:
        q = decode(b, fmt)[1]
        for p in [128, 256, 512, 1024, 2048, 4096]:
            lo, hi = interval(q, p, 'erfc')
            if (lo > threshold if b == low else hi < threshold):
                proof.append({'input_bits': b, 'input_hex': float(q).hex(),
                              'erfc_lo': str(lo), 'erfc_hi': str(hi),
                              'precision': p, 'relation': 'above' if b == low else 'below',
                              'reference': reference(b, fmt)})
                break
        else:
            raise ArithmeticError(('unresolved saturation edge', fmt, staged, b))
    return {'format': fmt, 'schedule': 'staged' if staged else 'ideal',
            'first_unit_bits': high, 'last_nonunit_bits': low,
            'erfc_threshold': str(threshold), 'proof': proof,
            'negative_edge': 'XOR input and output sign bits',
            'global_extension': 'Erf is strictly increasing. All later positive finite inputs round to unit.'}


POLICY = {
    'id': 'burn-typed-erf-forward-v1',
    'status': 'project qualification targets, not vendor guarantees',
    'operation': {'erf': 33},
    'f64_f32_flex32_ordered_steps': 2,
    'half_compute_f32_ordered_steps': 2,
    'half_schedule': 'exact stored input -> exact F32 promotion -> F32 Erf -> one destination RNE',
    'ideal': 'RN_destination(erf(exact stored input))',
    'staged': 'RN_half(RN_F32(erf(exact stored half input)))',
    'neighbor': 'exact RNE image of same-sign nonzero F32 neighbors inside [-1,1]',
    'strict': ['NaN class', 'signed zero', 'infinity -> signed unit', 'nonzero/sign preservation', 'codomain [-1,1]', 'paired finite oddness'],
    'normal_subnormal_boundary': 'diagnostic within numerical allowance, zero/nonzero remains strict',
    'legacy_f32': 'existing eight-step policy elsewhere stays unchanged until new execution proof and policy decision',
    'scope': 'forward Erf only, no autodiff, GELU, GPU, full Burn, or performance claim',
    'dependencies': 'Burn Erf backward uses square, negate, exp and scalar arithmetic. Default GELU backward uses tanh approximation. Qualify separately.',
    'oracle': 'MPFR 4.2.2 directed RNDD/RNDU at exact dyadic inputs, exact rational RNE; precision cap 16384',
    'sources': [
        'https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html',
        'https://docs.nvidia.com/cuda/archive/13.2.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html',
        'https://www.mpfr.org/mpfr-4.2.2/mpfr.html'],
    'vendor_context': 'NVIDIA lists observed 2 ULP for erf and erff and explicitly does not guarantee its nonexhaustive bounds.',
}
POLICY_TEXT = json.dumps(POLICY, sort_keys=True, indent=2) + '\n'


def canonical_nan(fmt):
    return info(fmt)[4] | (1 << (FORMATS[fmt][1] - 1))


def classify(bits, fmt):
    sign = info(fmt)[3]
    magnitude = bits & (sign - 1)
    inf = info(fmt)[4]
    if magnitude > inf: return 'nan'
    if magnitude == inf: return 'infinity'
    if not magnitude: return 'zero'
    return 'subnormal' if magnitude < 1 << FORMATS[fmt][1] else 'normal'


def fixture_row(bits, fmt, result=None):
    r = reference(bits, fmt) if result is None else result
    ideal = r['ideal_bits'] if r['ideal_bits'] is not None else canonical_nan(fmt)
    staged = r['staged_bits'] if r['staged_bits'] is not None else ideal
    allowed = sorted(r['half_neighbor_images_2'] or [staged], key=lambda b: ordered(b, fmt))
    assert all(ordered(b, fmt) == ordered(a, fmt) + 1 for a, b in zip(allowed, allowed[1:]))
    intermediate = r['f32_bits']
    if intermediate is None:
        intermediate = canonical_nan('f32') if fmt in ('f16', 'bf16') and classify(ideal, fmt) == 'nan' else 0
    return bits, ideal, staged, allowed[0], allowed[-1], intermediate


def successors(bits, fmt):
    r = reference(bits, fmt)
    expected = r['staged_bits']
    sign = info(fmt)[3]
    if expected is None: return [canonical_nan(fmt)]
    if r['input_class'] != 'finite' or bits & (sign - 1) == 0: return [expected]
    if fmt in ('f16', 'bf16'): return r['half_neighbor_images_2']
    magnitude = expected & (sign - 1)
    return [(expected & sign) | b for b in range(max(1, magnitude - 2), min(encode(Q(1), fmt), magnitude + 2) + 1)]


def chain_closure(fmt):
    sign, inf = info(fmt)[3:]
    witness = 0x1612 if fmt == 'f16' else encode(Q(1, 8), fmt)
    positives = [0, 1, encode(Q(1), fmt), inf - 1, inf, canonical_nan(fmt), witness]
    seeds = sorted({b | s for b in positives for s in (0, sign)})
    assert len(seeds) == 14
    plans, closure = [], set()
    for seed in seeds:
        level = [seed]
        levels = [level]
        finite = True
        for _ in range(3):
            closure.update(level)
            finite &= all(decode(b, fmt)[0] == 'finite' for b in level)
            level = sorted({v for b in level for v in successors(b, fmt)})
            assert len(level) <= 125
            levels.append(level)
        plans.append({'seed': seed, 'finite': finite, 'operations': ['erf'] * 3,
                      'levels': levels, 'total_nodes': sum(map(len, levels))})
    return closure, plans


def self_check():
    for fmt in FORMATS:
        _, f, bias, sign, inf = info(fmt)
        minimum = dyadic(1, 1 - bias - f)
        assert encode(minimum / 2, fmt) == 0
        assert encode(-minimum / 2, fmt) == sign
        assert encode(3 * minimum / 2, fmt) == 2
        assert reference(inf - 1, fmt)['ideal_bits'] == encode(Q(1), fmt)
        for b in (0, sign, inf, sign | inf):
            assert fixture_row(b, fmt)[1] == fixture_row(b, fmt)[2]
    for fmt in ('f16', 'bf16'):
        for bits in range(65536):
            kind, q = decode(bits, fmt)
            if kind == 'finite':
                assert encode(q, fmt, bool(bits & info(fmt)[3])) == bits
                assert decode(encode(q, 'f32'), 'f32')[1] == q
    assert reference(0x1612, 'f16')['ideal_bits'] == 0x16d9
    assert reference(0x1612, 'f16')['staged_bits'] == 0x16da
    assert reference(0x9612, 'f16')['staged_bits'] == 0x96da


def generate():
    self_check()
    proofs = [saturation(fmt, staged) for fmt in FORMATS
              for staged in ([False, True] if fmt in ('f16', 'bf16') else [False])]
    allsets, plan = {}, {}
    for fmt in FORMATS:
        e,f,bias,sign,inf=info(fmt)
        if fmt in ('f16','bf16'):
            cases={b:{'all-stored-encodings'} for b in range(65536)}
        else:
            cases={}
            def add(b,tag):
                if 0<=b<sign*2: cases.setdefault(b,set()).add(tag)
            def both(b,tag):
                for s in [0,sign]: add(b|s,tag)
            def around(b,tag,radius=32):
                for v in range(max(0,b-radius),min(inf,b+radius)+1): both(v,tag)
            for exp in range(1<<e):
                for frac in [0,1,(1<<f)//3,(1<<f)//2,(1<<f)-2,(1<<f)-1]:
                    both((exp<<f)|frac,'exponent-lattice')
            for b in range(2049): both(b,'tiny-subnormal-prefix')
            around(1<<f,'subnormal-normal-seam',256)
            # Locate the output's normal/subnormal transition independently.
            lo, hi = 0, 1 << f
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if reference(mid, fmt)['ideal_bits'] >= 1 << f: hi = mid
                else: lo = mid
            around(hi, 'erf-output-normal-subnormal-seam', 128)
            # Source seams identify inputs only. Expected values remain MPFR.
            centers=[encode(Q(27,32),fmt),encode(Q(5,4),fmt),encode(Q(6),fmt),encode(Q(1),fmt),encode(Q(8),fmt)]
            if fmt=='f64': centers += [0x3e30000000000000,0x4006db6d00000000,0x4006db6e00000000,encode(Q(20,7),fmt)]
            else: centers += [0x31800000,0x4036db6d]
            for b in centers: around(b,'approximation-region-seam',128)
            # First input whose separately rounded square becomes infinity.
            # This is an intermediate-overflow trap, never an Erf infinity result.
            overflow_tie=dyadic(1,bias+1)-dyadic(1,bias-f-1)
            low,high=0,inf-1
            while low+1<high:
                mid=(low+high)//2
                if decode(mid,fmt)[1]**2>=overflow_tie: high=mid
                else: low=mid
            around(high,'intermediate-square-overflow-to-unit',128)
            for p in proofs:
                if p['format']==fmt: around(p['first_unit_bits'],'round-to-unit-boundary',128)
            # No floating random generator and no host-libm expected values.
            state=0x9e3779b97f4a7c15
            for _ in range(4096):
                state=(state*6364136223846793005+1442695040888963407)&((1<<64)-1)
                both(state%inf,'fixed-lcg-full-finite')
            for i in range(2049): both(encode(Q(i,256),fmt),'dense-central-dyadic')
            for frac in [1,2,(1<<(f-1))-1,1<<(f-1),(1<<(f-1))+1,(1<<f)-1]: both(inf|frac,'nan-payload-class')
            both(inf,'infinity')
            # Values distinguishable in F64 but collapsed by a premature F32 cast.
            if fmt=='f64':
                for v in [Q(1,10),Q(1,2),Q(1),Q(5,4),Q(2),Q(4)]:
                    around(encode(v,fmt),'f64-distinct-f32-collision',16)
        allsets[fmt]=cases
        plan[fmt]={'baseline_count':len(cases),'tag_counts':dict(Counter(t for ts in cases.values() for t in ts))}
    blob, ranges, counts, plans = bytearray(), [], {}, {}
    for fmt, cases in allsets.items():
        closure, chain_plans = chain_closure(fmt)
        plans[fmt] = chain_plans
        extras = sorted(closure - cases.keys())
        for bits in closure: cases.setdefault(bits, set()).add('materialized-chain-closure')
        width = (sum(FORMATS[fmt]) + 1) // 8
        ranges.append([(len(blob), len(cases))])
        classes, ideal_classes = Counter(), Counter()
        differences = nonsingle = finite = 0
        stream = hashlib.sha256()
        for bits, tags in sorted(cases.items()):
            r = reference(bits, fmt)
            row = fixture_row(bits, fmt, r)
            for word in row[:5]: blob.extend(word.to_bytes(width, 'little'))
            blob.extend(row[5].to_bytes(4, 'little'))
            finite += decode(bits, fmt)[0] == 'finite'
            classes[classify(row[2], fmt)] += 1
            ideal_classes[classify(row[1], fmt)] += 1
            differences += row[1] != row[2]
            nonsingle += row[3] != row[4]
            stream.update(json.dumps(row, separators=(',', ':')).encode() + b'\n')
        counts[fmt] = dict(plan[fmt], all=len(cases), finite=finite,
                           additional_closure_count=len(extras), additional_input_bits=extras,
                           direct_staged_differences=differences, non_singleton_envelopes=nonsingle,
                           staged_classes=dict(classes), ideal_classes=dict(ideal_classes),
                           row_stream_sha256=stream.hexdigest())
        print(fmt, len(cases), 'rows, closure additions', len(extras), flush=True)
    policy_hash = hashlib.sha256(POLICY_TEXT.encode()).hexdigest()
    fixture_hash = hashlib.sha256(blob).hexdigest()
    rust = '// Generated by tools/nvidia/typed-erf-generate.py. Do not edit.\n'
    rust += f'pub const POLICY_SHA256: &str = "{policy_hash}";\n'
    rust += f'pub const FIXTURE_SHA256: &str = "{fixture_hash}";\n'
    rust += 'const FIXTURE_BYTES: &[u8] = include_bytes!("typed_erf_fixtures.bin");\n'
    rust += '#[rustfmt::skip]\nconst RANGES: [[(usize, usize); 1]; 5] = ' + str(ranges) + ';\n'
    rust += '#[rustfmt::skip]\nconst CHAIN_PLANS: [[ChainPlan; 14]; 5] = [\n'
    for fmt in FORMATS:
        rust += '[\n'
        for p in plans[fmt]:
            rust += f'ChainPlan {{ seed: {p["seed"]:#x}, operations: [Operation::Erf; 3], finite: {str(p["finite"]).lower()} }},\n'
        rust += '],\n'
    rust += '];\n'
    manifest = dict(version=1, corpus_version=1, schema_version=3,
                    policy_sha256=policy_hash, policy=POLICY, policy_source_bytes=POLICY_TEXT,
                    fixture_sha256=fixture_hash, fixture_bytes=len(blob),
                    fixture_rust_sha256=hashlib.sha256(rust.encode()).hexdigest(),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mpfr_version=version().decode(), counts=counts, chain_plans=plans,
                    total_operation_cases=sum(c['all'] for c in counts.values()),
                    saturation_proofs=proofs,
                    binary_encoding='dtype-major f64,f32,flex32,f16,bf16, one Erf operation, ascending unsigned inputs; five destination-width LE words input/ideal/staged/allowed_low/allowed_high, then u32 intermediate_f32',
                    nan='Expected NaN is canonical class sentinel; every input/source NaN byte remains exact. Arbitrary chain NaN inputs use class-only fallback.',
                    closure='Every legally admitted stored intermediate input at each of three materialized Erf steps has an immutable reference. Unknown finite inputs fail.',
                    metrics='Ideal and staged metrics use independent pair eligibility, matrices and worst witnesses. Half acceptance uses exact F32 neighbor image.',
                    scope='forward Erf only; no autodiff, GELU, GPU or full Burn claim')
    return {
        'os/library/ai/src/validation/typed_erf_fixtures.bin': bytes(blob),
        'os/library/ai/src/validation/typed_erf_fixtures.rs': rust.encode(),
        'tools/nvidia/typed-erf-manifest.json': json.dumps(manifest, sort_keys=True, indent=2).encode() + b'\n',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if args.check == (args.output_root is not None):
        parser.error('Choose read-only --check or an explicit --output-root')
    root = Path(__file__).resolve().parents[2] if args.check else args.output_root
    for relative, data in generate().items():
        target = root / relative
        if args.check:
            assert target.read_bytes() == data, str(target) + ' differs'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != data: target.write_bytes(data)
    print('PASS: deterministic independent Erf fixtures', flush=True)
