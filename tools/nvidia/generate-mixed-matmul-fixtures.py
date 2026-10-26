#!/usr/bin/env python3
"""Reproduce the bounded mixed matrix corpus with exact rational/RNE arithmetic.

Adapted with provenance from the independently reviewed D3OS cache reference
(reference-design-v1, 2026-10-26). Original model SHA256:
bec224241888c7259a316ac29aa307bd1dd3cb3b3cae5cd9d4e95eec9558abdb.
Original fixtures SHA256:
ea1a32874f02abfbe3ebffc7c9df6eed5419e500c37b5af093c3b0b45666a08f.
Independent review report SHA256:
2f6f09f37d60da5ef3cd83a410ccaacd29a8c67e78ff94030e427627ce2a73c1.

No Burn/candidate/NumPy/CUDA arithmetic is imported. Generated Rust contains
only data descriptors and immutable source/expected bytes. Project intervals
are conditional empirical targets, never a vendor rounding or order promise.
Run --check for byte-for-byte reproduction; --self-test validates arithmetic.
"""
import argparse
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

"""Independent exact dyadic reference. Cache design evidence, not production code.

No candidate, NumPy, Burn, CUDA, MPFR, or host floating arithmetic in the oracle.
Finite math uses Fraction. Exceptions and zero sign use an explicit IEEE model.
An ideal dot, serial F32 dot, and an all-order exact certificate are distinct APIs.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from math import prod

FORMATS = {"F16": (5, 10), "BF16": (8, 7), "F32": (8, 23), "F64": (11, 52)}


def pow2(e):
    return Q(1 << e) if e >= 0 else Q(1, 1 << -e)


def storage(dtype):
    return "F32" if dtype == "Flex32" else dtype


@dataclass(frozen=True)
class Value:
    kind: str
    q: Q = Q(0)
    sign: int = 0


def finite(q, sign=0):
    q = Q(q)
    return Value("finite", q, int(q < 0) if q else sign)


def decode(bits, fmt):
    eb, fb = FORMATS[fmt]
    sign = bits >> (eb + fb)
    assert 0 <= bits < 1 << (eb + fb + 1)
    exp, frac = (bits >> fb) & ((1 << eb) - 1), bits & ((1 << fb) - 1)
    if exp == (1 << eb) - 1:
        return Value("nan" if frac else "inf", sign=sign)
    bias = (1 << (eb - 1)) - 1
    q = Q(frac) * pow2(1 - bias - fb) if exp == 0 else Q((1 << fb) + frac) * pow2(exp - bias - fb)
    return finite(-q if sign else q, sign)


def floor_log2(q):
    assert q > 0
    e = q.numerator.bit_length() - q.denominator.bit_length()
    return e - int(q < pow2(e))


def round_integer(q, mode, sign):
    n, rem = divmod(q.numerator, q.denominator)
    if mode == "rn":
        return n + int(2 * rem > q.denominator or (2 * rem == q.denominator and n % 2 == 1))
    if mode == "rz":
        return n
    if mode == "rp":
        return n + int(bool(rem) and not sign)
    if mode == "rm":
        return n + int(bool(rem) and sign)
    raise ValueError(mode)


def encode(v, fmt, mode="rn"):
    if not isinstance(v, Value):
        v = finite(v)
    eb, fb = FORMATS[fmt]
    top = ((1 << eb) - 1) << fb
    sign = v.sign << (eb + fb)
    if v.kind == "nan":
        return sign | top | (1 << (fb - 1))
    if v.kind == "inf":
        return sign | top
    q = abs(v.q)
    if not q:
        return sign
    bias = (1 << (eb - 1)) - 1
    emin, emax = 1 - bias, bias
    e = floor_log2(q)
    step = pow2(max(e, emin) - fb)
    n = round_integer(q / step, mode, v.sign)
    if e < emin:
        return sign | n
    if n == 1 << (fb + 1):
        e, n = e + 1, n >> 1
    if e > emax:
        use_inf = mode == "rn" or (mode == "rp" and not v.sign) or (mode == "rm" and v.sign)
        return sign | (top if use_inf else top - 1)
    return sign | ((e + bias) << fb) | (n - (1 << fb))


def rounded(v, fmt, mode="rn"):
    return decode(encode(v, fmt, mode), fmt)


def add(a, b):
    if a.kind == "nan" or b.kind == "nan":
        return Value("nan")
    if a.kind == "inf" and b.kind == "inf" and a.sign != b.sign:
        return Value("nan")
    if a.kind == "inf":
        return a
    if b.kind == "inf":
        return b
    q = a.q + b.q
    return finite(q, a.sign & b.sign if not a.q and not b.q else 0)


def mul(a, b):
    if a.kind == "nan" or b.kind == "nan":
        return Value("nan")
    sign = a.sign ^ b.sign
    if a.kind == "inf" or b.kind == "inf":
        if (a.kind == "finite" and not a.q) or (b.kind == "finite" and not b.q):
            return Value("nan")
        return Value("inf", sign=sign)
    return finite(a.q * b.q, sign)


def operand(v, dtype):
    return rounded(v, "F16") if dtype == "Flex32" else v


def exact_dot(a, b, dtype):
    acc = finite(0)
    for x, y in zip(a, b, strict=True):
        acc = add(acc, mul(operand(x, dtype), operand(y, dtype)))
    return acc


def serial_dot(a, b, dtype, mode="rn", accfmt="F32"):
    """Named diagnostic graph only. This is never the implicit MMA oracle."""
    acc = finite(0)
    for x, y in zip(a, b, strict=True):
        acc = rounded(add(acc, mul(operand(x, dtype), operand(y, dtype))), accfmt, mode)
    return rounded(acc, storage(dtype))


def all_order_certificate(terms, fmt="F32"):
    """Sufficient proof for every subset sum, hence every permutation/tree.

    Normal products lie on lattice 2^e. Sum of absolute integer coefficients
    fits p bits. Every subset sum is exactly representable. Any rounding mode
    that preserves exact representable results gives the same magnitude.
    Nonempty zero sign remains outside the proof. No FTZ assumption is needed
    for nonzero sums because the lattice and operands/products are normal.
    """
    if any(t.kind != "finite" for t in terms):
        return {"proved": False, "reason": "nonfinite term"}
    qs = [t.q for t in terms if t.q]
    if not qs:
        return {"proved": True, "integer_abs_sum": 0, "zero_sign": "unqualified"}
    exps = []
    for q in qs:
        assert q.denominator & (q.denominator - 1) == 0
        n = abs(q.numerator)
        exps.append((n & -n).bit_length() - 1 - (q.denominator.bit_length() - 1))
    e = min(exps)
    lattice = pow2(e)
    s = sum(abs(q / lattice) for q in qs)
    assert s.denominator == 1
    eb, fb = FORMATS[fmt]
    emin, emax = 2 - (1 << (eb - 1)), (1 << (eb - 1)) - 1
    good = e >= emin and s < (1 << (fb + 1)) and s * lattice <= (2 - pow2(-fb)) * pow2(emax)
    return {"proved": bool(good), "quantum_exponent": e, "integer_abs_sum": int(s),
            "precision_bits": fb + 1, "all_permutations_and_binary_trees": bool(good),
            "zero_sign": "unqualified", "assumption": "each exact normal operation is preserved; no extra lossy operand conversion"}


def json_value(v, fmt):
    return {"bits": f"{encode(v, fmt):0{(sum(FORMATS[fmt]) + 1) // 4}x}",
            "class": "zero" if v.kind == "finite" and not v.q else v.kind,
            "rational": str(v.q) if v.kind == "finite" else None, "sign": v.sign}


def coordinates(shape):
    return product(*(range(n) for n in shape))


def flat_index(shape, coord):
    n = 0
    for d, i in zip(shape, coord, strict=True):
        assert 0 <= i < d
        n = n * d + i
    return n


@dataclass
class Tensor:
    shape: tuple
    values: list
    dtype: str

    def __post_init__(self):
        assert len(self.values) == prod(self.shape)

    def at(self, c):
        return self.values[flat_index(self.shape, c)]

    def transpose(self):
        s = self.shape[:-2] + (self.shape[-1], self.shape[-2])
        return Tensor(s, [self.at(c[:-2] + (c[-1], c[-2])) for c in coordinates(s)], self.dtype)

    def export(self):
        f = storage(self.dtype)
        return {"shape": list(self.shape), "dtype": self.dtype,
                "bits": [json_value(v, f)["bits"] for v in self.values]}


def broadcast(a, b):
    n = max(len(a), len(b))
    a, b = (1,) * (n-len(a)) + a, (1,) * (n-len(b)) + b
    out = []
    for x, y in zip(a, b, strict=True):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            raise ValueError("incompatible batch")
    return tuple(out)


def batch_coord(c, shape):
    c = c[len(c)-len(shape):] if shape else ()
    return tuple(0 if s == 1 else i for i, s in zip(c, shape, strict=True))


def matmul(a, b, require_proof=True):
    assert a.dtype == b.dtype and 2 <= len(a.shape) <= 8 and 2 <= len(b.shape) <= 8
    m, k = a.shape[-2:]
    assert k == b.shape[-2]
    n = b.shape[-1]
    bs = broadcast(a.shape[:-2], b.shape[:-2])
    values, proofs, exacts = [], [], []
    for c in coordinates(bs + (m, n)):
        batch, i, j = c[:-2], c[-2], c[-1]
        av = [a.at(batch_coord(batch, a.shape[:-2]) + (i, t)) for t in range(k)]
        bv = [b.at(batch_coord(batch, b.shape[:-2]) + (t, j)) for t in range(k)]
        terms = [mul(operand(x, a.dtype), operand(y, a.dtype)) for x, y in zip(av, bv, strict=True)]
        accfmt = 'F64' if a.dtype=='F64' else 'F32'
        proof = all_order_certificate(terms,accfmt)
        if require_proof:
            assert proof['proved'], proof
        q = exact_dot(av, bv, a.dtype)
        # Only the all-order domain justifies this ideal F32 scratch value.
        v = rounded(rounded(q, accfmt), storage(a.dtype))
        exacts.append(str(q.q))
        proofs.append(proof)
        values.append(v)
    return Tensor(bs + (m, n), values, a.dtype), proofs, exacts


def sum_axis(t, axis):
    """Pinned typed SUM: serial logical F32 RN/noFTZ, then storage RNE."""
    s = t.shape[:axis] + (1,) + t.shape[axis+1:]
    out, proofs = [], []
    for c in coordinates(s):
        terms = [t.at(c[:axis] + (i,) + c[axis+1:]) for i in range(t.shape[axis])]
        proofs.append(all_order_certificate(terms,'F64' if t.dtype=='F64' else 'F32'))
        acc = finite(0)
        for v in terms:
            acc = rounded(add(acc, v), "F64" if t.dtype == "F64" else "F32")
        out.append(rounded(acc, storage(t.dtype)))
    return Tensor(s, out, t.dtype), proofs


def reduce_broadcast(t, target):
    assert len(t.shape) == len(target)
    stages = []
    for axis in range(len(target)):
        if t.shape[axis] != target[axis]:
            assert target[axis] == 1
            t, proof = sum_axis(t, axis)
            stages.append({"axis": axis, "output": t.export(), "all_order_exact": all(x['proved'] for x in proof)})
    assert t.shape == target
    return t, stages


def backward(a, b, g):
    da0, pa, _ = matmul(g, b.transpose())
    db0, pb, _ = matmul(a.transpose(), g)
    da, sa = reduce_broadcast(da0, a.shape)
    db, sb = reduce_broadcast(db0, b.shape)
    return da, db, {"dA_before_sum": da0.export(), "dB_before_sum": db0.export(),
                    "dA_axes": sa, "dB_axes": sb, "product_all_order_exact": all(x['proved'] for x in pa+pb)}


def adjoint_scatter(a, b, g):
    """Independent exact index differentiation, without matrix transposes."""
    da, db = [Q(0)] * len(a.values), [Q(0)] * len(b.values)
    bs = broadcast(a.shape[:-2], b.shape[:-2])
    for batch in coordinates(bs):
        ac, bc = batch_coord(batch, a.shape[:-2]), batch_coord(batch, b.shape[:-2])
        for i, j, k in product(range(a.shape[-2]), range(b.shape[-1]), range(a.shape[-1])):
            ai, bi = ac + (i, k), bc + (k, j)
            gv = g.at(batch + (i, j)).q
            da[flat_index(a.shape, ai)] += gv * operand(b.at(bi), a.dtype).q
            db[flat_index(b.shape, bi)] += operand(a.at(ai), a.dtype).q * gv
    return da, db


def linear(x, w, bias):
    rows = prod(x.shape[:-1])
    x2 = Tensor((rows, x.shape[-1]), x.values, x.dtype)
    y, p, _ = matmul(x2, w)
    out = [rounded(add(v, bias.values[i % w.shape[-1]]), storage(x.dtype))
           for i, v in enumerate(y.values)] if bias else y.values
    return Tensor(x.shape[:-1] + (w.shape[-1],), out, x.dtype), {"matmul_stored": y.export(), "proofs": p}


def linear_backward(x, w, g):
    rows = prod(x.shape[:-1])  # zero-aware, never divides by last width
    x2 = Tensor((rows, x.shape[-1]), x.values, x.dtype)
    g2 = Tensor((rows, w.shape[-1]), g.values, x.dtype)
    dx, px, _ = matmul(g2, w.transpose())
    dw, pw, _ = matmul(x2.transpose(), g2)
    db, ps = sum_axis(g2, 0)
    dx.shape = x.shape
    db.shape = (w.shape[-1],)
    return dx, dw, db, {"all_order_exact": all(p['proved'] for p in px+pw+ps)}


def conditional_dot_bound(a, b, dtype, rounding="rn"):
    """A theorem under a declared elementary-operation model, not an MMA guarantee.

    Products must be exactly F32 normal/zero, with at most K-1 rounded sums
    on every term's path, no overflow/underflow, and faithful normal rounding.
    Induction bounds each term's factor error by gamma_d = d*u/(1-d*u).
    Absolute error <= gamma_d * sum(abs(products)). Directed modes use 2^-23.
    """
    terms = [mul(operand(x, dtype), operand(y, dtype)) for x, y in zip(a, b, strict=True)]
    assert all(t.kind == "finite" for t in terms)
    for t in terms:
        assert rounded(t, "F32").q == t.q
        assert not t.q or abs(t.q) >= pow2(-126)
    s = sum((abs(t.q) for t in terms), Q(0))
    exact = sum((t.q for t in terms), Q(0))
    d, u = max(0, len(terms)-1), pow2(-24 if rounding == "rn" else -23)
    assert d*u < 1 and s < pow2(127)
    error = d*u/(1-d*u) * s
    lo, hi = exact-error, exact+error
    fmt = storage(dtype)
    return {"status": "conditional_not_qualified_mma", "d": d, "u": str(u), "sum_abs_products": str(s),
            "exact_dot": str(exact), "scratch_error_bound": str(error), "scratch_interval": [str(lo),str(hi)],
            "output_interval_bits": [json_value(rounded(lo,fmt),fmt)['bits'],json_value(rounded(hi,fmt),fmt)['bits']],
            "requires": ["exact F32 products", "no overflow or subnormal intermediate", "no FTZ", "at most d faithful roundings per term path", "actual lowering establishes model"],
            "rounding_model": rounding}


def finite_metrics(observed_bits, fmt, exact_ideal):
    """Report errors without selecting an acceptance tolerance."""
    v = decode(observed_bits, fmt)
    if v.kind != 'finite' or exact_ideal.kind != 'finite':
        return {'observed_class':v.kind,'ideal_class':exact_ideal.kind,
                'class_equal':v.kind==exact_ideal.kind,'finite_errors':None}
    error = abs(v.q-exact_ideal.q)
    width = sum(FORMATS[fmt])+1
    signbit = 1 << (width-1)
    mask = (1 << width)-1
    def ordered(b):
        return (~b & mask) if b & signbit else b | signbit
    target = encode(exact_ideal,fmt)
    return {'absolute_error':str(error),
            'relative_error':str(error/abs(exact_ideal.q)) if exact_ideal.q else None,
            'destination_ULP_distance_to_RNE_ideal':abs(ordered(observed_bits)-ordered(target)),
            'signed_zero_equal':v.sign==exact_ideal.sign if not v.q and not exact_ideal.q else None,
            'acceptance':'requires case policy; metrics alone do not approve'}


def stored_sum_interval(intervals, dtype):
    """Propagate finite interval endpoints through the pinned serial typed SUM.

    This needs valid input intervals first. It does not supply an HMMA bound.
    Monotonic rounding/addition permits endpoint propagation even when input
    errors are correlated. Signed-zero and nonfinite cases need separate rules.
    """
    fmt = 'F64' if dtype=='F64' else 'F32'
    lo = hi = finite(0)
    for left,right in intervals:
        assert left <= right
        lo = rounded(add(lo,finite(left)),fmt)
        hi = rounded(add(hi,finite(right)),fmt)
        assert lo.kind == hi.kind == 'finite', 'overflow interval requires classification policy'
    lo,hi = rounded(lo,storage(dtype)),rounded(hi,storage(dtype))
    assert lo.kind == hi.kind == 'finite'
    return lo.q,hi.q


def typed_bias_interval(interval, bias, dtype):
    assert bias.kind == 'finite'
    lo,hi = [rounded(finite(q+bias.q),storage(dtype)) for q in interval]
    assert lo.kind == hi.kind == 'finite'
    return lo.q,hi.q


def small_tensor(shape, dtype, seed):
    vals = []
    for c in coordinates(tuple(shape)):
        # Coordinate-dependent signed, nonsymmetric values. All are normal/zero.
        n = seed + sum((i+1) * (x+1)**2 + (i+3)*x for i,x in enumerate(c))
        vals.append(rounded(Q(n % 3 - 1, 16), storage(dtype)))
    return Tensor(tuple(shape), vals, dtype)


def proof_summary(ps):
    return {'cells': len(ps), 'all_order_exact': all(p['proved'] for p in ps),
            'max_integer_abs_sum': max((p.get('integer_abs_sum',0) for p in ps),default=0),
            'min_quantum_exponent': min((p.get('quantum_exponent',0) for p in ps),default=0),
            'nonempty_zero_sign': 'diagnostic_until_pin'}


def layout(t, mode='dense', odd=False):
    """Concrete source bytes plus logical-to-physical affine layout, holes poisoned."""
    s = t.shape
    if mode == 'transpose':
        order = list(range(len(s)-2)) + [len(s)-1,len(s)-2]
        strides = [0] * len(s)
        step = 1
        for axis in reversed(order):
            strides[axis] = step
            step *= max(s[axis],1)
    else:
        strides = [prod(s[i+1:]) for i in range(len(s))]
        if mode == 'signed':
            strides = [v * 2 for v in strides]
            strides[-2] = -strides[-2]
            if len(s) > 2:
                strides[0] = -strides[0]
    low = sum(min(0,(n-1)*v) for n,v in zip(s,strides) if n)
    high = sum(max(0,(n-1)*v) for n,v in zip(s,strides) if n)
    base = -low + (1 if odd else 2)
    if odd and base % 2 == 0:
        base += 1
    count = base + high + 3 if t.values else 4
    fmt = storage(t.dtype)
    poison = encode(Value('nan'), fmt)
    bits = [poison] * count
    addresses = []
    for c, v in zip(coordinates(s),t.values,strict=True):
        addr = base + sum(i*v for i,v in zip(c,strides,strict=True))
        assert 0 <= addr < count
        bits[addr] = encode(v,fmt)
        addresses.append(addr)
    for c, v in zip(coordinates(s),t.values,strict=True):
        addr = base + sum(i*v for i,v in zip(c,strides,strict=True))
        assert decode(bits[addr],fmt) == v
    packed = []
    if fmt in ('F16','BF16'):
        packed = [bits[i] | ((bits[i+1] if i+1<len(bits) else 0) << 16) for i in range(0,len(bits),2)]
    batch_records = []
    for c in coordinates(s[:-2]):
        off = base + sum(i*v for i,v in zip(c,strides[:-2],strict=True))
        matrix_n = prod(s[-2:])
        batch_records.append({'coordinate':list(c),'element_base':off,'aligned_word_base':off//2,
                              'half_slot':off%2,'dense_matrix_words':(off%2+matrix_n+1)//2,
                              'note':'dense extent applies after matrix canonicalization; signed/stepped spans use layout bounds'})
    return {'shape':list(s),'strides_elements':strides,'base_elements':base,'storage_elements':count,
            'storage_bits':[f'{b:0{(sum(FORMATS[fmt])+1)//4}x}' for b in bits],
            'packed_u32_le':[f'{b:08x}' for b in packed], 'source_sha256': hashlib.sha256(b''.join(b.to_bytes((sum(FORMATS[fmt])+1)//8,'little') for b in bits)).hexdigest(),
            'batch_addresses':batch_records,'guard':'all unaddressed elements are quiet NaN poison; source byte identity required after consumers/backward'}


def expected_tensor(t, zero_k=False):
    d = t.export()
    d['compare'] = ['exact_bits' if v.q or zero_k else 'zero_class_sign_diagnostic' for v in t.values]
    if storage(t.dtype) in ('F16','BF16'):
        bits = [encode(v,storage(t.dtype)) for v in t.values]
        d['packed_output_u32_le'] = [f'{bits[i] | ((bits[i+1] if i+1<len(bits) else 0)<<16):08x}' for i in range(0,len(bits),2)]
        d['unused_final_half'] = 'zero' if len(bits)%2 else 'none'
    return d


def geometry(row):
    a, b = row['lhs_shape'],row['rhs_shape']
    suffix = row['key'].split('_',1)[1]
    # Fixed replacements cover ranks 5,6,7 without new rows or dtype cross products.
    target = {'attention_av':5, 'broadcast_both':6, 'broadcast_rhs':7}.get(suffix)
    if target:
        a,b = [1]*(target-len(a))+a, [1]*(target-len(b))+b
    dtype = row['dtype']
    raw_unequal = dtype=='Flex32' and suffix=='broadcast_rhs'
    if raw_unequal:
        b = b[-2:]
    a,b = small_tensor(a,dtype,2),small_tensor(b,dtype,7)
    y,p,_ = matmul(a,b)
    mode = 'signed' if suffix=='signed_views' else 'transpose' if suffix in ('transpose_views','attention_qk') else 'dense'
    return {'key':row['key'],'group':row['group'],'dtype':dtype,'status':'cpu_reference_proved_gpu_pending',
            'lhs':a.export(),'rhs':b.export(),'lhs_physical':layout(a,mode,True),
            'rhs_physical':layout(b,mode,True),'expected':expected_tensor(y,a.shape[-1]==0),
            'proof':proof_summary(p),'refinement':'raw Native unequal-rank shared RHS; public uses same bits with leading singleton shape; no unequal-rank Autodiff claim' if raw_unequal else 'rank coverage via leading singleton axes' if target else None,
            'public_vector_fold':'must compare actual public transform with these same logical values' if suffix=='public_vector_fold' else None,
            'runtime_assertions':['padding excluded from logical dot','source bytes unchanged','last-drop consumer result before and after saved-source backward','one whole-output packed cast']}


def grad(row):
    suffix = row['key'].split('_',1)[1]
    dtype = row['dtype']
    linear_case = suffix.startswith('linear')
    if linear_case:
        xs,ws = ([0,3,5],[5,7]) if suffix=='linear_empty_batch_nonempty_dW_db' else ([2,3,5],[5,0]) if suffix=='linear_zero_output_width' else ([2,3,5],[5,7])
        a,b = small_tensor(xs,dtype,2),small_tensor(ws,dtype,7)
        bias = small_tensor([ws[-1]],dtype,4)
        y,forward = linear(a,b,bias)
        g = small_tensor(y.shape,dtype,11)
        dx,dw,db,proof = linear_backward(a,b,g)
        return {'key':row['key'],'group':row['group'],'dtype':dtype,'status':'cpu_reference_proved_gpu_pending',
                'input':a.export(),'weight':b.export(),'bias':bias.export(),'upstream':g.export(),
                'forward':expected_tensor(y),'stored_forward':forward['matmul_stored'],
                'dx':expected_tensor(dx,ws[-1]==0),'dW':expected_tensor(dw,prod(xs[:-1])==0),
                'db':expected_tensor(db,prod(xs[:-1])==0),'proof':proof,
                'graph':'flatten leading dimensions with explicit zero-aware product; dW flattened matmul; db one typed axis-0 sum',
                'runtime_assertions':['drop original input/weight handles before backward; saved clones survive','source bytes unchanged','consume dx/dW/db after last temporary drops']}
    if suffix=='both_multi_axis_batch_broadcast':
        ash,bsh = [2,1,3,1,3,5],[1,4,1,5,5,7]
    elif suffix=='zero_K_matmul':
        ash,bsh = [3,0],[0,7]
    elif suffix=='shared_source_both_operands':
        ash,bsh = [3,3],[3,3]
    elif suffix=='transpose_signed_view_adjoints':
        ash,bsh = [2,3,5],[2,5,7]
    else:
        ash,bsh = [3,5],[5,7]
    a,b = small_tensor(ash,dtype,2),small_tensor(bsh,dtype,7)
    shared = suffix=='shared_source_both_operands'
    if shared:
        b = a
    y,p,_ = matmul(a,b)
    g = small_tensor(y.shape,dtype,11)
    da,db,stages = backward(a,b,g)
    sa,sb = adjoint_scatter(a,b,g)
    assert [v.q for v in da.values] == sa
    assert [v.q for v in db.values] == sb
    d = {'key':row['key'],'group':row['group'],'dtype':dtype,'status':'cpu_reference_proved_gpu_pending',
         'lhs':a.export(),'rhs':b.export(),'upstream':g.export(),'forward':expected_tensor(y,ash[-1]==0),
         'dA':expected_tensor(da),'dB':expected_tensor(db),'stages':stages,'proof':proof_summary(p),
         'independent_index_adjoint':'matched exact scatter derivative','tracked': 'lhs_only' if suffix=='lhs_tracked_only' else 'rhs_only' if suffix=='rhs_tracked_only' else 'both',
         'runtime_assertions':['saved opposite operand survives last external drop','untracked path also preserves dtype','source/sibling bytes unchanged','typed consumers of forward and gradients survive scratch drops']}
    if shared:
        combined = [rounded(add(x,y),storage(dtype)) for x,y in zip(da.values,db.values,strict=True)]
        d['shared_source_gradient'] = expected_tensor(Tensor(a.shape,combined,dtype))
        d['grad_join'] = 'typed add of both registered parents; traversal/order still runtime qualification'
    if suffix=='transpose_signed_view_adjoints':
        d['lhs_physical'],d['rhs_physical'] = layout(a,'signed',True),layout(b,'transpose',True)
        # One-to-one view scatter to backing storage, zero for unaddressed locations.
        for name,t in [('lhs',da),('rhs',db)]:
            lay = d[name+'_physical']
            backing = [finite(0)] * lay['storage_elements']
            for c,v in zip(coordinates(t.shape),t.values,strict=True):
                addr = lay['base_elements'] + sum(i*s for i,s in zip(c,lay['strides_elements'],strict=True))
                backing[addr] = v
            d[name+'_backing_adjoint'] = Tensor((len(backing),),backing,dtype).export()
    return d


def cell(name, a, b, dtype, role='diagnostic', bits=False):
    fmt = storage(dtype)
    av = [decode(v,fmt) if bits else rounded(v,fmt) for v in a]
    bv = [decode(v,fmt) if bits else rounded(v,fmt) for v in b]
    ideal = exact_dot(av,bv,dtype)
    terms = [mul(operand(x,dtype),operand(y,dtype)) for x,y in zip(av,bv,strict=True)]
    cert = all_order_certificate(terms)
    serial = serial_dot(av,bv,dtype)
    d = {'name':name,'lhs_source_bits':[json_value(v,fmt)['bits'] for v in av],
         'rhs_source_bits':[json_value(v,fmt)['bits'] for v in bv],
         'lhs_operand_bits':[json_value(operand(v,dtype),'F16' if dtype=='Flex32' else fmt)['bits'] for v in av],
         'rhs_operand_bits':[json_value(operand(v,dtype),'F16' if dtype=='Flex32' else fmt)['bits'] for v in bv],
         'exact_operand_dot':json_value(ideal,fmt), 'ideal_one_F32_scratch_then_storage':json_value(rounded(rounded(ideal,'F32'),fmt),fmt),
         'serial_F32_diagnostic':json_value(serial,fmt),'all_order_certificate':cert,'role':role}
    operand_fmt = 'F16' if dtype=='Flex32' else fmt
    eb, _ = FORMATS[operand_fmt]
    normal_floor = pow2(2-(1<<(eb-1)))
    normal_operands = all(v.kind=='finite' and (not v.q or abs(v.q)>=normal_floor) for v in [operand(x,dtype) for x in av+bv])
    d['normal_or_zero_operands'] = normal_operands
    if role=='class_requirement':
        d['acceptance']='proposed_class_requirement_blocked_until_lowering_and_GPU_qualification'
    elif cert['proved'] and normal_operands and ideal.kind=='finite' and ideal.q:
        d['acceptance']='exact_bits_under_exact_normal_MMA_semantics'
    else:
        d['acceptance']='diagnostic_no_tolerance_approved'
    if all(t.kind=='finite' and (not t.q or abs(t.q)>=pow2(-126)) and rounded(t,'F32').q==t.q for t in terms):
        d['conditional_bounds'] = [conditional_dot_bound(av,bv,dtype,r) for r in ('rn','any_directed')]
    return d


def numeric(row):
    dt = row['dtype']; suffix=row['key'].split('_',1)[1]
    cells=[]; staging={}
    def addc(name,a,b,role='diagnostic',bits=False):
        cells.append(cell(name,a,b,dt,role,bits))
    if suffix=='exact_finite_dyadic':
        addc('nonsymmetric_signed_normal',[Q(1,16),Q(-1,16),Q(1,16)],[Q(1,16),Q(1,16),Q(1,16)])
        q = pow2(-7 if dt=='BF16' else -10)
        addc('positive_well_conditioned_nonexact',[1+q,pow2(-8),pow2(-12)],[1+q,pow2(-8),pow2(-12)])
    elif suffix=='destination_ties_both_parities_and_sides':
        if dt!='Flex32':
            fb=FORMATS[dt][1]; h=pow2(-fb-1); eps=pow2(-fb-4)
            for parity in (0,1):
                for side in (-1,0,1):
                    addc(f'output_RNE_parity{parity}_side{side}',[1+parity*pow2(-fb),h,side*eps],[1,1,1])
        else:
            for parity in (0,1):
                for side in (-1,0,1):
                    addc(f'F16_operand_RNE_parity{parity}_side{side}',[1+parity*pow2(-10)+pow2(-11)+side*pow2(-23)],[1])
            addc('F32_result_tie_even',[1,pow2(-12)],[1,pow2(-12)])
    elif suffix in ('cancellation_order','K32_boundary_cancellation'):
        k=3 if suffix=='cancellation_order' else 65
        av,bv=[Q(0)]*k,[Q(0)]*k
        where=[0,1,2] if k==3 else [31,32,64]
        for i,a,b in zip(where,[4096,1,-4096],[4096,1,4096],strict=True):
            av[i],bv[i]=a,b
        addc('2pow24_plus_one_minus_2pow24',av,bv)
        v=cells[-1]
        v['alternative_exact_regrouped_result']=json_value(rounded(1,storage(dt)),storage(dt))
        assert v['serial_F32_diagnostic']['rational']=='0'
        assert v['exact_operand_dot']['rational']=='1'
    elif suffix=='minimum_subnormal_input':
        tiny=pow2(-24 if dt=='F16' else -133 if dt=='BF16' else -149)
        addc('minimum_source_subnormal_times_one',[tiny],[1],'class_requirement')
        if dt=='Flex32':
            addc('minimum_F16_operand_subnormal_survival',[pow2(-24)],[1],'class_requirement')
            cells[0]['proposed_requirement']='F32 source bits retained, RNE F16 conversion is +0; F32 result +0 magnitude, nonempty sign pending'
    elif suffix=='accumulator_or_output_underflow':
        if dt=='F16':
            addc('normal_operands_final_half_subnormal',[pow2(-12)],[pow2(-12)],'class_requirement')
            addc('half_output_zero_tie_even',[pow2(-12)],[pow2(-13)],'class_requirement')
        elif dt=='BF16':
            addc('normal_operands_F32_subnormal',[pow2(-126)],[pow2(-7)],'class_requirement')
            addc('normal_operands_F32_half_min_subnormal',[pow2(-126)],[pow2(-24)],'class_requirement')
        else:
            addc('F16_subnormal_operands_product_normal_F32',[pow2(-24)],[pow2(-24)],'class_requirement')
    elif suffix=='live_nan_inf_and_masked_padding':
        f=storage(dt); eb,fb=FORMATS[f]; inf=((1<<eb)-1)<<fb; one=encode(1,f); zero=0; nan=inf|(1<<(fb-1)); neginf=inf|(1<<(eb+fb))
        for name,a,b in [('live_nan',[nan],[one]),('infinity_times_zero',[inf],[zero]),('single_infinity',[inf],[one]),('opposite_infinities',[inf,neginf],[one,one]),('signed_zero_only',[1<<(eb+fb)],[one])]:
            addc(name,a,b,'class_requirement',True)
        addc('logical_finite_mask_control',[1,2],[3,4])
        addc('exact_finite_cancellation_zero',[1,-1],[1,1])
    elif suffix=='operand_range_and_output_overflow':
        if dt=='F16':
            addc('finite_operands_half_output_overflow',[65504],[2],'class_requirement')
        elif dt=='BF16':
            addc('finite_F32_intermediate_overflow_then_cancellation',[pow2(127),-pow2(127)],[2,2])
        else:
            addc('finite_F32_to_F16_overflow_times_zero',[65536],[0],'class_requirement')
            addc('largest_F16_finite_source',[65504],[1])
    else:
        raise AssertionError(suffix)
    if suffix=='destination_ties_both_parities_and_sides' and dt!='Flex32':
        fb=FORMATS[dt][1];h=pow2(-fb-1)
        product=rounded(1+h,'F32')
        staged=rounded(rounded(add(rounded(product,dt),finite(h)),'F32'),dt)
        fused=rounded(add(product,finite(h)),dt)
        assert staged.q != fused.q
        # Broadcast gradient: three independent matmul cells, each 1+h.
        each=rounded(product,dt)
        right=rounded(rounded(add(add(each,each),each),'F32'),dt)
        wrong=rounded(add(add(product,product),product),dt)
        assert right.q != wrong.q
        ga = Tensor((1,1,1),[rounded(1,dt)],dt)
        gb = Tensor((3,1,2),[rounded(1,dt),rounded(h,dt)]*3,dt)
        gg = Tensor((3,1,2),[rounded(1,dt)]*6,dt)
        gy, _, _ = matmul(ga,gb)
        assert gy.shape == gg.shape
        da0, pp, _ = matmul(gg,gb.transpose())
        da, axes = reduce_broadcast(da0,ga.shape)
        assert da.values[0].q == right.q
        staging={'actual_broadcast_dA_fixture':{'A':ga.export(),'B':gb.export(),'forward':gy.export(),'dY':gg.export(),'dA_before_sum':da0.export(),'axis_stages':axes,'dA':da.export(),'matmul_proof':proof_summary(pp)},'linear':{'product_exact':str(1+h),'bias':str(h),'stored_then_typed_bias':json_value(staged,dt),'fused_epilogue_wrong_graph':json_value(fused,dt)},
                 'broadcast_gradient':{'matmul_cells_exact':[str(1+h)]*3,'stored_cells':[json_value(each,dt)]*3,'typed_axis_sum':json_value(right,dt),'sum_scratch_then_cast_wrong_graph':json_value(wrong,dt)}}
    max_k = max(len(c['lhs_source_bits']) for c in cells)
    fmt = storage(dt)
    z = json_value(finite(0),fmt)['bits']
    av = [decode(int(v,16),fmt) for c in cells for v in c['lhs_source_bits']+[z]*(max_k-len(c['lhs_source_bits']))]
    bv = [decode(int(v,16),fmt) for c in cells for v in c['rhs_source_bits']+[z]*(max_k-len(c['rhs_source_bits']))]
    a = Tensor((len(cells),1,max_k),av,dt)
    b = Tensor((len(cells),max_k,1),bv,dt)
    return {'key':row['key'],'group':'numeric','dtype':dt,'batched_lhs':a.export(),'batched_rhs':b.export(),'status':'mixed_exact_proofs_and_unqualified_diagnostics',
            'cells':cells,'staging':staging,
            'launch_recipe':'one batched dot fixture, pad all cells to max K with logical zeros; poison only outside logical K',
            'mask_requirements':'poison unaddressed storage including odd sibling half, masked M/N/K and K32 padding with NaN/Inf; live results unchanged; guard bytes untouched'}


def f64(row):
    key=row['key']
    pairs={'low_mantissa_1_plus_2pow_minus40':([1+pow2(-40)],[1]),
           'range_2pow600_times_2pow_minus600':([pow2(600)],[pow2(-600)]),
           'min_subnormal_2pow_minus1074':([pow2(-1074)],[1]),
           'cancellation_near_2pow53':([pow2(53),1,-pow2(53)],[1,1,1])}
    d={'key':key,'group':'f64_separate_audit','status':'real_double_diagnostic_only','arithmetic':'explicit F64 FMA graph; never F32/F16 narrowed fallback'}
    if key in pairs:
        av,bv=pairs[key]; a=[rounded(v,'F64') for v in av];b=[rounded(v,'F64') for v in bv]
        d.update(lhs=Tensor((1,len(a)),a,'F64').export(),rhs=Tensor((len(b),1),b,'F64').export(),
                 exact_dot=json_value(exact_dot(a,b,'F64'),'F64'),serial_F64_FMA=json_value(serial_dot(a,b,'F64',accfmt='F64'),'F64'))
    elif key=='batched_broadcast_signed_views':
        a=small_tensor([2,3,5],'F64',2);b=small_tensor([1,5,7],'F64',7)
        y,p,_=matmul(a,b)
        d.update(lhs=a.export(),rhs=b.export(),lhs_physical=layout(a,'signed'),rhs_physical=layout(b,'transpose'),expected=expected_tensor(y),proof=proof_summary(p))
    elif key=='zero_K_and_empty_metadata':
        cases=[]
        for ash,bsh in [([3,0],[0,7]),([0,5],[5,7])]:
            a=small_tensor(ash,'F64',2);b=small_tensor(bsh,'F64',7);y,_,_=matmul(a,b)
            cases.append({'lhs':a.export(),'rhs':b.export(),'expected':expected_tensor(y,True)})
        d['zero_or_empty_cases']=cases
    elif key=='both_gradients_and_broadcast_reduction':
        a=small_tensor([2,1,3,5],'F64',2);b=small_tensor([1,4,5,7],'F64',7)
        y,_,_=matmul(a,b);g=small_tensor(y.shape,'F64',11)
        da,db,stages=backward(a,b,g);sa,sb=adjoint_scatter(a,b,g)
        assert [v.q for v in da.values]==sa and [v.q for v in db.values]==sb
        d.update(lhs=a.export(),rhs=b.export(),upstream=g.export(),dA=expected_tensor(da),dB=expected_tensor(db),stages=stages,independent_index_adjoint='matched genuine-double exact domain')
    else:
        d['request']={'dtype':'F64','lhs_shape':[1,1],'rhs_shape':[1,1]}
        d['expected_policy']='Reject unsupported public dtype without launching a narrower arithmetic route. Do not count as mixed acceptance.'
        d['runtime_pending']=True
    return d


def planned(row):
    d=dict(row)
    d['status']='runtime_gate_not_cpu_qualified'
    d['oracle']='existing route/source identity and regression suite; no mixed numerical oracle substitution'
    if row['group']=='errors_and_lifecycle':
        d['oracle']='validate metadata before empty shortcut; source/guard exact bits; no arithmetic tolerance'
    if row['key']=='packed_sibling_alias_preservation':
        d['scenario']={'source_half_slot':1,'dense_matrix_elements':15,'batch_elements':15,'batches':4,
                       'checks':['even and odd sibling source bits unchanged','whole output word ownership','saved clone readable after both derivatives']}
    if row['key']=='queued_temporaries_failure_recovery':
        d['sequence']=['record forward and typed consumer','drop user handles and scratch owners','enqueue both backward branches and typed gradient consumers','drop intermediates','apply allocation pressure','read explicit outputs and guards','submit independent valid op after rejected invalid shape']
        d['qualification']='requires resident driver execution; CPU model proves no fence/lifetime behavior'
    return d



BUDGET = {'schema_version': 1, 'unit': 'selected fixture row; both derivatives and source/tail assertions belong to the row; no implicit Cartesian expansion', 'selected_rows': 121, 'row_counts': {'errors_and_lifecycle': 12, 'f64_separate_audit': 8, 'geometry': 39, 'gradient': 30, 'numeric': 24, 'preservation_regressions': 8}, 'reserved_rows': 3, 'hard_cap_rows': 124, 'runtime_ids_allocated': [], 'fixtures': [{'key': 'f16_dense_odd', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [3, 5], 'rhs_shape': [5, 7], 'purpose': 'ordinary dense; odd packed output', 'status': 'planned'}, {'key': 'f16_attention_qk', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [2, 3, 5, 7], 'rhs_shape': [2, 3, 7, 11], 'purpose': 'attention scores; rhs created by transpose view', 'status': 'planned'}, {'key': 'f16_attention_av', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [2, 3, 5, 11], 'rhs_shape': [2, 3, 11, 7], 'purpose': 'attention value product', 'status': 'planned'}, {'key': 'f16_broadcast_both', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [2, 1, 3, 5], 'rhs_shape': [1, 4, 5, 7], 'purpose': 'two independent batch broadcast axes', 'status': 'planned'}, {'key': 'f16_broadcast_rhs', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [2, 3, 5, 7], 'rhs_shape': [1, 1, 7, 11], 'purpose': 'shared dense weights, no physical batch expansion', 'status': 'planned'}, {'key': 'f16_signed_views', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [2, 3, 5], 'rhs_shape': [2, 5, 7], 'purpose': 'negative batch stride; reversed K; sliced odd base; stepped N', 'status': 'planned'}, {'key': 'f16_transpose_views', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [3, 5], 'rhs_shape': [5, 7], 'purpose': 'both matrices materialized from nonsymmetric transpose views', 'status': 'planned'}, {'key': 'f16_rank_eight', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [1, 2, 1, 1, 1, 1, 3, 5], 'rhs_shape': [1, 1, 1, 1, 1, 1, 5, 7], 'purpose': 'layout maximum rank with batch broadcast', 'status': 'planned'}, {'key': 'f16_public_vector_fold', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [2, 3, 1, 5], 'rhs_shape': [2, 1, 5, 7], 'purpose': 'public Tensor matmul vector/batch swap transform', 'status': 'planned'}, {'key': 'f16_zero_batch', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [0, 3, 5], 'rhs_shape': [1, 5, 7], 'purpose': 'empty batch broadcast with singleton', 'status': 'planned'}, {'key': 'f16_zero_M', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [0, 5], 'rhs_shape': [5, 7], 'purpose': 'empty output rows', 'status': 'planned'}, {'key': 'f16_zero_N', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [3, 5], 'rhs_shape': [5, 0], 'purpose': 'empty output columns', 'status': 'planned'}, {'key': 'f16_zero_K', 'group': 'geometry', 'dtype': 'F16', 'lhs_shape': [3, 0], 'rhs_shape': [0, 7], 'purpose': 'nonempty typed positive zero output', 'status': 'planned'}, {'key': 'f16_exact_finite_dyadic', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_destination_ties_both_parities_and_sides', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_cancellation_order', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_K32_boundary_cancellation', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_minimum_subnormal_input', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_accumulator_or_output_underflow', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_live_nan_inf_and_masked_padding', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_operand_range_and_output_overflow', 'group': 'numeric', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_both_operands_dense', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_lhs_tracked_only', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_rhs_tracked_only', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_both_multi_axis_batch_broadcast', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_transpose_signed_view_adjoints', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_shared_source_both_operands', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_linear_with_bias', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_linear_empty_batch_nonempty_dW_db', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_zero_K_matmul', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'f16_linear_zero_output_width', 'group': 'gradient', 'dtype': 'F16', 'status': 'planned'}, {'key': 'bf16_dense_odd', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [3, 5], 'rhs_shape': [5, 7], 'purpose': 'ordinary dense; odd packed output', 'status': 'planned'}, {'key': 'bf16_attention_qk', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [2, 3, 5, 7], 'rhs_shape': [2, 3, 7, 11], 'purpose': 'attention scores; rhs created by transpose view', 'status': 'planned'}, {'key': 'bf16_attention_av', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [2, 3, 5, 11], 'rhs_shape': [2, 3, 11, 7], 'purpose': 'attention value product', 'status': 'planned'}, {'key': 'bf16_broadcast_both', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [2, 1, 3, 5], 'rhs_shape': [1, 4, 5, 7], 'purpose': 'two independent batch broadcast axes', 'status': 'planned'}, {'key': 'bf16_broadcast_rhs', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [2, 3, 5, 7], 'rhs_shape': [1, 1, 7, 11], 'purpose': 'shared dense weights, no physical batch expansion', 'status': 'planned'}, {'key': 'bf16_signed_views', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [2, 3, 5], 'rhs_shape': [2, 5, 7], 'purpose': 'negative batch stride; reversed K; sliced odd base; stepped N', 'status': 'planned'}, {'key': 'bf16_transpose_views', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [3, 5], 'rhs_shape': [5, 7], 'purpose': 'both matrices materialized from nonsymmetric transpose views', 'status': 'planned'}, {'key': 'bf16_rank_eight', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [1, 2, 1, 1, 1, 1, 3, 5], 'rhs_shape': [1, 1, 1, 1, 1, 1, 5, 7], 'purpose': 'layout maximum rank with batch broadcast', 'status': 'planned'}, {'key': 'bf16_public_vector_fold', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [2, 3, 1, 5], 'rhs_shape': [2, 1, 5, 7], 'purpose': 'public Tensor matmul vector/batch swap transform', 'status': 'planned'}, {'key': 'bf16_zero_batch', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [0, 3, 5], 'rhs_shape': [1, 5, 7], 'purpose': 'empty batch broadcast with singleton', 'status': 'planned'}, {'key': 'bf16_zero_M', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [0, 5], 'rhs_shape': [5, 7], 'purpose': 'empty output rows', 'status': 'planned'}, {'key': 'bf16_zero_N', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [3, 5], 'rhs_shape': [5, 0], 'purpose': 'empty output columns', 'status': 'planned'}, {'key': 'bf16_zero_K', 'group': 'geometry', 'dtype': 'BF16', 'lhs_shape': [3, 0], 'rhs_shape': [0, 7], 'purpose': 'nonempty typed positive zero output', 'status': 'planned'}, {'key': 'bf16_exact_finite_dyadic', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_destination_ties_both_parities_and_sides', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_cancellation_order', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_K32_boundary_cancellation', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_minimum_subnormal_input', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_accumulator_or_output_underflow', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_live_nan_inf_and_masked_padding', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_operand_range_and_output_overflow', 'group': 'numeric', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_both_operands_dense', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_lhs_tracked_only', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_rhs_tracked_only', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_both_multi_axis_batch_broadcast', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_transpose_signed_view_adjoints', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_shared_source_both_operands', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_linear_with_bias', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_linear_empty_batch_nonempty_dW_db', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_zero_K_matmul', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'bf16_linear_zero_output_width', 'group': 'gradient', 'dtype': 'BF16', 'status': 'planned'}, {'key': 'flex32_dense_odd', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [3, 5], 'rhs_shape': [5, 7], 'purpose': 'ordinary dense; odd packed output', 'status': 'planned'}, {'key': 'flex32_attention_qk', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [2, 3, 5, 7], 'rhs_shape': [2, 3, 7, 11], 'purpose': 'attention scores; rhs created by transpose view', 'status': 'planned'}, {'key': 'flex32_attention_av', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [2, 3, 5, 11], 'rhs_shape': [2, 3, 11, 7], 'purpose': 'attention value product', 'status': 'planned'}, {'key': 'flex32_broadcast_both', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [2, 1, 3, 5], 'rhs_shape': [1, 4, 5, 7], 'purpose': 'two independent batch broadcast axes', 'status': 'planned'}, {'key': 'flex32_broadcast_rhs', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [2, 3, 5, 7], 'rhs_shape': [1, 1, 7, 11], 'purpose': 'shared dense weights, no physical batch expansion', 'status': 'planned'}, {'key': 'flex32_signed_views', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [2, 3, 5], 'rhs_shape': [2, 5, 7], 'purpose': 'negative batch stride; reversed K; sliced odd base; stepped N', 'status': 'planned'}, {'key': 'flex32_transpose_views', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [3, 5], 'rhs_shape': [5, 7], 'purpose': 'both matrices materialized from nonsymmetric transpose views', 'status': 'planned'}, {'key': 'flex32_rank_eight', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [1, 2, 1, 1, 1, 1, 3, 5], 'rhs_shape': [1, 1, 1, 1, 1, 1, 5, 7], 'purpose': 'layout maximum rank with batch broadcast', 'status': 'planned'}, {'key': 'flex32_public_vector_fold', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [2, 3, 1, 5], 'rhs_shape': [2, 1, 5, 7], 'purpose': 'public Tensor matmul vector/batch swap transform', 'status': 'planned'}, {'key': 'flex32_zero_batch', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [0, 3, 5], 'rhs_shape': [1, 5, 7], 'purpose': 'empty batch broadcast with singleton', 'status': 'planned'}, {'key': 'flex32_zero_M', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [0, 5], 'rhs_shape': [5, 7], 'purpose': 'empty output rows', 'status': 'planned'}, {'key': 'flex32_zero_N', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [3, 5], 'rhs_shape': [5, 0], 'purpose': 'empty output columns', 'status': 'planned'}, {'key': 'flex32_zero_K', 'group': 'geometry', 'dtype': 'Flex32', 'lhs_shape': [3, 0], 'rhs_shape': [0, 7], 'purpose': 'nonempty typed positive zero output', 'status': 'planned'}, {'key': 'flex32_exact_finite_dyadic', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_destination_ties_both_parities_and_sides', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_cancellation_order', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_K32_boundary_cancellation', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_minimum_subnormal_input', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_accumulator_or_output_underflow', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_live_nan_inf_and_masked_padding', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_operand_range_and_output_overflow', 'group': 'numeric', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_both_operands_dense', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_lhs_tracked_only', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_rhs_tracked_only', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_both_multi_axis_batch_broadcast', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_transpose_signed_view_adjoints', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_shared_source_both_operands', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_linear_with_bias', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_linear_empty_batch_nonempty_dW_db', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_zero_K_matmul', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'flex32_linear_zero_output_width', 'group': 'gradient', 'dtype': 'Flex32', 'status': 'planned'}, {'key': 'mixed_dtype_before_empty', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'rank_zero', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'rank_one', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'rank_nine', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'K_mismatch', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'zero_batch_with_two_rejects', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'dimension_over_limit_even_if_empty', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'nonempty_count_overflow', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'padded_launch_or_word_extent_overflow', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'out_of_bounds_signed_source', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'packed_sibling_alias_preservation', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'queued_temporaries_failure_recovery', 'group': 'errors_and_lifecycle', 'status': 'planned'}, {'key': 'F32_K8_and_linear_Project_BiasRow', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'F32_K16_thresholds', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'F32_splitK_and_merge', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'I32_full_range_wrapping', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'Qwen_existing_packed_extension', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'AlexNet_existing', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'MNIST_existing', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'DOOM_existing', 'group': 'preservation_regressions', 'status': 'planned'}, {'key': 'low_mantissa_1_plus_2pow_minus40', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'range_2pow600_times_2pow_minus600', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'min_subnormal_2pow_minus1074', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'cancellation_near_2pow53', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'batched_broadcast_signed_views', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'zero_K_and_empty_metadata', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'both_gradients_and_broadcast_reduction', 'group': 'f64_separate_audit', 'status': 'planned'}, {'key': 'explicit_rejection_without_narrowing_until_implemented', 'group': 'f64_separate_audit', 'status': 'planned'}], 'compiler_probe_program_budget': 6, 'compiler_probes': [{'key': 'existing_half_TC_bridge', 'variants': ['F16', 'BF16'], 'check': 'existing F32 pointer loops, RNE operands, F32 accumulators, finite and subnormal round trips'}, {'key': 'direct_packed_input', 'variants': ['F16', 'BF16'], 'check': 'U16 load/bitcast, odd logical base, masked K tails, no source numerical conversion'}, {'key': 'flex32_precision', 'variants': ['F16_operand_F32_storage'], 'check': 'RNE F16 operands and F32 accumulator/output; no silent compensated TF32 substitution'}, {'key': 'packed_output_composition', 'variants': ['odd_rows_columns_batch'], 'check': 'dense F32 scratch plus one whole-tensor cast; full word ownership and zero tail'}, {'key': 'double_scalar_FMA', 'variants': ['F64_all_stages'], 'check': 'explicit RNE F64 FMA; two-word storage; no F32 conversion or accidental contraction change'}, {'key': 'double_MMA_feasibility', 'variants': ['F64_m8n8k4_or_legal_TileIR_shape'], 'check': 'SM89 acceptance or explicit diagnostic; DMMA versus DFMA/emulation; double type preservation'}], 'compiler_variants_note': 'Static dtype variants inside six bounded probe programs; compile-only future work, no production registry mutation in design', 'linux_selected_comparison_budget': 18, 'linux_selected': [{'dtype': 'F16', 'fixture': 'dense_odd', 'status': 'linux_selected_pending'}, {'dtype': 'F16', 'fixture': 'broadcast_both', 'status': 'linux_selected_pending'}, {'dtype': 'F16', 'fixture': 'destination_ties', 'status': 'linux_selected_pending'}, {'dtype': 'F16', 'fixture': 'linear_with_bias_gradients', 'status': 'linux_selected_pending'}, {'dtype': 'F16', 'fixture': 'minimum_subnormal', 'status': 'linux_selected_pending'}, {'dtype': 'BF16', 'fixture': 'dense_odd', 'status': 'linux_selected_pending'}, {'dtype': 'BF16', 'fixture': 'broadcast_both', 'status': 'linux_selected_pending'}, {'dtype': 'BF16', 'fixture': 'destination_ties', 'status': 'linux_selected_pending'}, {'dtype': 'BF16', 'fixture': 'linear_with_bias_gradients', 'status': 'linux_selected_pending'}, {'dtype': 'BF16', 'fixture': 'minimum_subnormal', 'status': 'linux_selected_pending'}, {'dtype': 'Flex32', 'fixture': 'dense_odd', 'status': 'linux_selected_pending'}, {'dtype': 'Flex32', 'fixture': 'broadcast_both', 'status': 'linux_selected_pending'}, {'dtype': 'Flex32', 'fixture': 'destination_ties', 'status': 'linux_selected_pending'}, {'dtype': 'Flex32', 'fixture': 'linear_with_bias_gradients', 'status': 'linux_selected_pending'}, {'dtype': 'Flex32', 'fixture': 'minimum_subnormal', 'status': 'linux_selected_pending'}, {'dtype': 'F64', 'fixture': 'low_mantissa', 'status': 'linux_selected_pending'}, {'dtype': 'F64', 'fixture': 'wide_range', 'status': 'linux_selected_pending'}, {'dtype': 'F64', 'fixture': 'both_gradients', 'status': 'linux_selected_pending'}], 'linux_note': 'Comparison subset plus three F64 diagnostics, not 18 additional Native acceptance cases. Keep errors/panics and selected actual dtypes in results.', 'execution': {'native_gpu': False, 'linux_gpu': False, 'compiler': False, 'regressions': False, 'cpu_reference': 'witness-model.py only, one taskset CPU20,22 process'}}
ORIGINAL_POLICY = {'API_recommendation': 'implement the chosen three packed/Flex precision entries internally with typed storage boundaries; qualify exact-normal geometry/gradients first. Do not advertise full general numerical acceptance until blockers pass. No silent F32 legacy, host, or narrowed F64 fallback.', 'Linux': {'F64': 'active from_globals preserves double; no narrowing inference from deprecated mapping; any narrowing result is separate diagnostic', 'actual_selected_observations': [], 'labels': ['linux_source_derived', 'linux_selected_pending', 'linux_selected_observed', 'linux_unsupported_or_panicked'], 'source_derived': 'Burn0.22.1 output lhs dtype, Cubek0.3.1 half/BF16 F32 accumulators, accelerated Flex32 F16 operand stages; strategy-dependent'}, 'acceptance': {'all_order_certificate': 'products are integer multiples of 2^e; sum of absolute coefficients <2^24, lattice >=2^-126, all normal intermediate magnitudes within range; every exact normal subset sum fits F32', 'cancellation': 'named diagnostic with ideal1, serialF32=0; no scalar-FMA equality or global relative tolerance', 'conditional_bound': 'E=gamma_d(u)*sum_abs_products, gamma_d=d*u/(1-d*u); d<=K-1 for exact normal products and a faithful elementary addition tree; RN u=2^-24, unspecified directed mode u=2^-23', 'conditional_bound_status': 'not justified merely by HMMA opcode/type; depends on actual internal arithmetic/error contract, no FTZ/overflow/underflow, and path rounding count. Pending', 'destination_conversion': 'RNE of ACTUAL F32 scratch, not direct ideal-real-to-half rounding; compare exact output interval only after scratch model is established', 'exact_normal_nonzero': 'exact bits only after all-order certificate and operand-domain checks; no tolerance', 'gradient_staging': 'matmul-storage-before-sum and storage-after-each-axis are mandatory; numeric dependency pinned separately', 'linear_staging': 'matmul-storage-before-bias is mandatory, including zero K plus bias', 'nonexact_finite': 'no general approved tolerance; report exact rational dot, sum_abs_products, abs error, relative error only for nonzero ideal, storage ULP distance and explicit case interval', 'source_copy': 'all source bytes, masks, sibling halves, and guards exact bit identity; NaN payload bits in untouched storage also exact'}, 'authorization': 'cache design only. Shared oracle or production implementation requires later parent authorization.', 'budget': {'Linux_comparisons': 'separate at most18 selected graphs; none run', 'compiler_programs': 'separate six-program budget; no expansion', 'reserved_rows': 3, 'selected_rows': 121, 'total_cap': 124}, 'compiler_observations': {'accuracy_status': 'all numeric HMMA schedule/FTZ/exceptional questions remain unqualified; index F2I.FTZ is not a payload FTZ observation', 'entries': [{'FTZ_instruction_sites': [{'offset': 400, 'owners': ['packed_f16'], 'sass': '/*0190*/                   F2I.FTZ.U32.TRUNC.NTZ R3, R2 ;', 'section': '.text.packed_f16'}], 'SASS_sha256': '1a913807c3749d18c0093b9814b6748a51b5617a1973e568c23158371f878356', 'all_pointer_offsets_zero_extended': True, 'allocated_relocations': [{'addend': 0, 'offset': 0, 'symbol': '___NV_TILE_LAUNCH_META_DATA___', 'symbol_defined': True, 'target': '.nv.constant4', 'type': 2}], 'attempt': 'attempts/003-packed_f16', 'calls': 0, 'code_bytes': 8320, 'conversion_IR': [], 'cubin_sha256': 'd7f3e027d4788f7e8cd5445caf7061ae4a2d3ae74e520e9f94191bb5ccd42686', 'entry': 'packed_f16', 'floating_FMA_IR': [], 'floating_MMA_IR': [{'attributes': '[]', 'opcode': 'MmaF', 'operand_types': ['Tile(TileType { shape: [16, 32], element_type: Scalar(F16) })', 'Tile(TileType { shape: [32, 32], element_type: Scalar(F16) })', 'Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })]'}], 'frame_bytes': {'packed_f16': 0}, 'local_instructions': 0, 'local_sections': [], 'lowering': 'half_BF16_HMMA_F32_accumulator', 'minimum_stack_bytes': {'packed_f16': 0}, 'production_direct_constant_qualifier': 'PASS', 'registers': 80, 'required_threads': [[128, 1, 1]], 'selected_machine_operations': {'F2I.FTZ.U32.TRUNC.NTZ': 1, 'HMMA.16816.F32': 4, 'I2F.U32.RP': 1, 'MUFU.RCP': 1}, 'source_sha256': 'f9966a469ac211fffcdb5462ddbe35ed508bd70e7a76ceb1c0cf1b86ddf17d10', 'static_shared_bytes': 9216, 'tilebc_sha256': 'aec00f6e1494e4eb1977570e3eceff08db2bac2ead126f7545ab7e830a7a6f20', 'undefined_symbols': 0}, {'FTZ_instruction_sites': [{'offset': 400, 'owners': ['packed_bf16'], 'sass': '/*0190*/                   F2I.FTZ.U32.TRUNC.NTZ R3, R2 ;', 'section': '.text.packed_bf16'}], 'SASS_sha256': '47891d66138da05491b0b89c715028eb249df42a3860156a6df72912b45ea6f8', 'all_pointer_offsets_zero_extended': True, 'allocated_relocations': [{'addend': 0, 'offset': 0, 'symbol': '___NV_TILE_LAUNCH_META_DATA___', 'symbol_defined': True, 'target': '.nv.constant4', 'type': 2}], 'attempt': 'attempts/004-packed_bf16', 'calls': 0, 'code_bytes': 8320, 'conversion_IR': [], 'cubin_sha256': 'a16295a07f39c1b7aba78f7d0dbb54143313ee2074c0eda8431f4af89ebb2b77', 'entry': 'packed_bf16', 'floating_FMA_IR': [], 'floating_MMA_IR': [{'attributes': '[]', 'opcode': 'MmaF', 'operand_types': ['Tile(TileType { shape: [16, 32], element_type: Scalar(BF16) })', 'Tile(TileType { shape: [32, 32], element_type: Scalar(BF16) })', 'Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })]'}], 'frame_bytes': {'packed_bf16': 0}, 'local_instructions': 0, 'local_sections': [], 'lowering': 'half_BF16_HMMA_F32_accumulator', 'minimum_stack_bytes': {'packed_bf16': 0}, 'production_direct_constant_qualifier': 'PASS', 'registers': 80, 'required_threads': [[128, 1, 1]], 'selected_machine_operations': {'F2I.FTZ.U32.TRUNC.NTZ': 1, 'HMMA.16816.F32.BF16': 4, 'I2F.U32.RP': 1, 'MUFU.RCP': 1}, 'source_sha256': 'f9966a469ac211fffcdb5462ddbe35ed508bd70e7a76ceb1c0cf1b86ddf17d10', 'static_shared_bytes': 9216, 'tilebc_sha256': 'c809060e8ee8eec5711a52e45b851c092b4613f9a5fe345ff48fff42b4688aed', 'undefined_symbols': 0}, {'FTZ_instruction_sites': [{'offset': 384, 'owners': ['flex32_f16_operand'], 'sass': '/*0180*/                   F2I.FTZ.U32.TRUNC.NTZ R3, R2 ;', 'section': '.text.flex32_f16_operand'}], 'SASS_sha256': 'f82718646bb3e42764b2c1be191beeb58208b426a0163a5cb41ed34e4546cb50', 'all_pointer_offsets_zero_extended': True, 'allocated_relocations': [{'addend': 0, 'offset': 0, 'symbol': '___NV_TILE_LAUNCH_META_DATA___', 'symbol_defined': True, 'target': '.nv.constant4', 'type': 2}], 'attempt': 'attempts/005-flex32_f16_operand', 'calls': 0, 'code_bytes': 6912, 'conversion_IR': [{'attributes': '[("rounding_mode", Integer(0, Scalar(I32)))]', 'opcode': 'FToF', 'operand_types': ['Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [16, 32], element_type: Scalar(F16) })]'}, {'attributes': '[("rounding_mode", Integer(0, Scalar(I32)))]', 'opcode': 'FToF', 'operand_types': ['Tile(TileType { shape: [32, 32], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [32, 32], element_type: Scalar(F16) })]'}], 'cubin_sha256': 'b63ec18d2441451989909b7d3c17e795c014bc1ae644f92de0a3f931f4f34712', 'entry': 'flex32_f16_operand', 'floating_FMA_IR': [], 'floating_MMA_IR': [{'attributes': '[]', 'opcode': 'MmaF', 'operand_types': ['Tile(TileType { shape: [16, 32], element_type: Scalar(F16) })', 'Tile(TileType { shape: [32, 32], element_type: Scalar(F16) })', 'Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [16, 32], element_type: Scalar(F32) })]'}], 'frame_bytes': {'flex32_f16_operand': 0}, 'local_instructions': 0, 'local_sections': [], 'lowering': 'half_BF16_HMMA_F32_accumulator', 'minimum_stack_bytes': {'flex32_f16_operand': 0}, 'production_direct_constant_qualifier': 'PASS', 'registers': 62, 'required_threads': [[128, 1, 1]], 'selected_machine_operations': {'F2FP.F16.F32.PACK_AB': 12, 'F2I.FTZ.U32.TRUNC.NTZ': 1, 'HMMA.16816.F32': 4, 'I2F.U32.RP': 1, 'MUFU.RCP': 1}, 'source_sha256': 'f9966a469ac211fffcdb5462ddbe35ed508bd70e7a76ceb1c0cf1b86ddf17d10', 'static_shared_bytes': 9216, 'tilebc_sha256': '09f4baad0dbce465fc5c4b6c753a380564977e0a6f0bc1ef81fbecbb8a55997b', 'undefined_symbols': 0}, {'FTZ_instruction_sites': [], 'SASS_sha256': '1a7afe6028cf0f506c9abd6ca1b60c8537b41332762ccf2746dca7469e777377', 'all_pointer_offsets_zero_extended': True, 'allocated_relocations': [{'addend': 0, 'offset': 0, 'symbol': '___NV_TILE_LAUNCH_META_DATA___', 'symbol_defined': True, 'target': '.nv.constant4', 'type': 2}], 'attempt': 'attempts/006-packed_output_cast', 'calls': 0, 'code_bytes': 768, 'conversion_IR': [{'attributes': '[("rounding_mode", Integer(0, Scalar(I32)))]', 'opcode': 'FToF', 'operand_types': ['Tile(TileType { shape: [128], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [128], element_type: Scalar(F16) })]'}, {'attributes': '[("rounding_mode", Integer(0, Scalar(I32)))]', 'opcode': 'FToF', 'operand_types': ['Tile(TileType { shape: [128], element_type: Scalar(F32) })'], 'result_types': '[Tile(TileType { shape: [128], element_type: Scalar(BF16) })]'}], 'cubin_sha256': '57687507f9cf8d7121ba7be44446c2c02b66d2102f119ddf5961b820e8bb022d', 'entry': 'packed_output_cast', 'floating_FMA_IR': [], 'floating_MMA_IR': [], 'frame_bytes': {'packed_output_cast': 0}, 'local_instructions': 0, 'local_sections': [], 'lowering': 'conversion_only', 'minimum_stack_bytes': {'packed_output_cast': 0}, 'production_direct_constant_qualifier': 'PASS', 'registers': 12, 'required_threads': [[128, 1, 1]], 'selected_machine_operations': {'F2FP.BF16.F32.PACK_AB': 2, 'F2FP.F16.F32.PACK_AB': 2}, 'source_sha256': 'f9966a469ac211fffcdb5462ddbe35ed508bd70e7a76ceb1c0cf1b86ddf17d10', 'static_shared_bytes': 0, 'tilebc_sha256': '4145a783292adb57b9398cc9a4e88b5de88930ee5fe70ab06b11daf611860bd4', 'undefined_symbols': 0}], 'observation_scope': 'read and rehashed external compiler artifacts; no compiler/build/GPU action here', 'pins': {'Rust': '1.98.1', 'bytecode': 'V13_4', 'cuTile_revision': '2f47ad0036a7c18ec11d460f70b297a5824f9788', 'optimization': 3, 'required_threads': [128, 1, 1], 'target': 'sm_89', 'tileiras': '13.4.92', 'worker_warps': 4}, 'report': 'sources/compiler-report.json'}, 'exceptional': {'FTZ': 'unqualified TC input/result/intermediate behavior is a qualification blocker for full subnormal claims; cannot repair with tolerance', 'Flex32': 'apply class rules AFTER operand RNE F16 conversion; finite F32 65536 ->Inf, then times0 ->NaN; minimum F32 subnormal ->F16 zero is intended range loss', 'Inf': 'propose correct sign for unambiguous single-sign infinite term with no invalid term and no prior graph-dependent overflow; finite overflow/cancellation separate diagnostic', 'NaN': 'propose NaN classification for live operand NaN or invalid 0*Inf or opposite signed infinite terms; ignore result NaN payload/sign; signaling quieting/flags not promised', 'empty': 'typed empty, zero logical elements, no numeric comparison or matmul/cast launch; validate metadata/errors first', 'isolated_subnormal': 'propose F16 2^-24*1 and BF16 2^-133*1 survival; normal BF16 2^-126*2^-7 separately probes F32 accumulator underflow; failure blocks claimed full proposed slice', 'nonempty_zero': 'zero magnitude/class required on certified all-order exact finite zero domain; sign diagnostic until selected-graph contract pinned; cancellation zero and underflow zero distinct', 'output_subnormal': 'separate RNE output-cast requirement; no evidence from source preservation or absence of HMMA FTZ suffix', 'source_subnormal': 'packed extraction must preserve exact source bits; this alone does not establish HMMA survival', 'zeroK': 'nonempty output typed +0 exact bits without input reads; linear then applies typed bias; zeroK gradients follow exact no-read identities'}, 'graphs': {'BF16': {'MMA_operand': 'BF16 bitcast', 'accumulator_scratch': 'F32, order unspecified', 'output': 'one whole-output RNE BF16 conversion', 'source_storage': 'packed BF16 bits'}, 'F16': {'MMA_operand': 'F16 bitcast', 'accumulator_scratch': 'F32, order unspecified', 'output': 'one whole-output RNE F16 conversion', 'source_storage': 'packed F16 bits'}, 'F32_I32': 'existing routes remain separately source-exact and regression-gated', 'F64': 'separate genuine F64 arithmetic and storage diagnostic; explicit F64 FMA schedule; no narrowed fallback', 'Flex32': {'MMA_operand': 'RNE F16 conversion inside load stage', 'accumulator_output': 'F32', 'output_metadata': 'Flex32, no output numeric conversion', 'source_storage': 'F32 exact source bits plus Flex32 metadata'}, 'linear': 'store matmul output in declared dtype, then separate typed RN/noFTZ bias add; preserve private output', 'linear_backward': 'dx via transposed linear without bias; dW via one flattened matmul; db via one typed axis0 sum over flattened rows', 'matmul_backward': 'dA=stored_matmul(dY,B^T), dB=stored_matmul(A^T,dY); enumerate differing batch axes ascending original index; serial RN F32 sum then storage RNE after EACH axis'}, 'qualification_blockers': ['HMMA internal faithful-rounding/error model and accumulation topology for nonexact bounds', 'TC subnormal input, product, accumulation and final output behavior', 'nonempty signed-zero contract', 'live NaN/Inf class behavior for selected graph including mask poison', 'production lowering identity must match qualified compiler graph', 'runtime layout/masks/packed slot handling, queue lifetime, ordinary autodiff and preservation regressions'], 'reduction_dependency': {'backward_axis_order': 'ascending ORIGINAL AXIS INDEX from broadcast_shape; backend default applies supplied sequence', 'global_sum_mean': 'separate stable original-length sorting graph, not used by backward reference', 'parent_pins': {'backend_float': 'ff0710e2746bab175fefb5c861ba4ee1221d65ac1be176bbadf842c4df4f5111', 'native_float_reduction': 'd5e40f48e9efbba7e046bb0ce9417125f2078c5c43be0bac31154de7d1895246', 'native_reduction': '32d5b4ae4854c453a99bb833fa4c72ccdbec643722816d2a859eb94ca7ce3c8e', 'reduction_kernel': '03ab53cbf39e7585ee82ad4a86084ba97a49de72435c449e0597c81183e77603', 'shared_oracle_not_used': '84021711460c5ae060eb774326cef0c6f69a44300b5ba4d778f066c6abaea249'}, 'public_float_sum_dims': 'caller-provided axis order', 'status': 'parent reports current53 all regressions and independent terminal review passed; no run read or replay by this task'}, 'schema_version': 1, 'scope': 'general public F16/BF16/Flex32 matmul, linear, ordinary equal-rank Autodiff; no production module authorization', 'status': 'provisional_design_not_public_API_acceptance', 'stored_graph_interval_propagation': 'map every valid scratch interval monotonically through destination RNE first; then typed bias add or pinned serial typed sum endpoints, storing after each axis. Nonfinite/zero signs handled separately.'}


def make_rows():
    rows = []
    for row in BUDGET['fixtures']:
        fn = {'geometry': geometry, 'gradient': grad, 'numeric': numeric,
              'f64_separate_audit': f64}.get(row['group'], planned)
        result = fn(row)
        if row['group'] in ('geometry', 'gradient'):
            consumers = {}
            for key in ('expected', 'forward', 'dA', 'dB', 'dx', 'dW', 'db', 'shared_source_gradient'):
                item = result.get(key)
                if not isinstance(item, dict) or 'bits' not in item:
                    continue
                fmt = storage(row['dtype'])
                t = Tensor(tuple(item['shape']), [decode(int(b, 16), fmt) for b in item['bits']], row['dtype'])
                consumer, proofs = sum_axis(t, len(t.shape) - 1)
                consumers[key] = {
                    'operation': 'typed sum along last original axis after final producer temporary drop',
                    'axis': len(t.shape) - 1, 'expected': expected_tensor(consumer, True),
                    'proof': proof_summary(proofs),
                    'dependency': 'pinned serial RN/noFTZ typed SUM, then storage RNE',
                    'runtime_status': 'pending; CPU expected values do not prove resource retirement'}
            result['last_drop_consumers'] = consumers
        rows.append(result)
    assert len(rows) == len({r['key'] for r in rows}) == 121
    assert dict(Counter(r['group'] for r in rows)) == BUDGET['row_counts']
    assert sum(len(r.get('cells', [])) for r in rows) == 65
    assert sum(len(r.get('last_drop_consumers', {})) for r in rows) == 141
    # This canonical serialization is the original reviewed artifact, unchanged.
    original = {'schema_version': 1, 'selected_rows': 121, 'reserved_rows': 3,
                'hard_cap_rows': 124, 'rows': rows,
                'unit': 'selected public operation row. Listed scalar cells are batched outputs, never hidden Cartesian axes.'}
    b = (json.dumps(original, indent=2, sort_keys=True) + '\n').encode()
    assert hashlib.sha256(b).hexdigest() == 'ea1a32874f02abfbe3ebffc7c9df6eed5419e500c37b5af093c3b0b45666a08f'
    return rows


def tensor_from(record):
    return Tensor(tuple(record['shape']), [decode(int(x, 16), storage(record['dtype'])) for x in record['bits']], record['dtype'])


def policies(record):
    fmt = storage(record['dtype'])
    result = []
    for i, bit in enumerate(record['bits']):
        v = decode(int(bit, 16), fmt)
        comparison = record.get('compare', [None] * len(record['bits']))[i]
        result.append('NaNClass' if v.kind == 'nan' else 'Infinity' if v.kind == 'inf'
                      else 'ZeroClass' if not v.q and comparison != 'exact_bits' else 'Exact')
    return result


def numeric_policy(c):
    ideal = c['ideal_one_F32_scratch_then_storage']
    if c['role'] == 'class_requirement':
        return {'nan': 'NaNClass', 'inf': 'Infinity', 'zero': 'ZeroClass'}.get(ideal['class'], 'Exact')
    if c['all_order_certificate']['proved'] and c['normal_or_zero_operands']:
        return 'ZeroClass' if ideal['class'] == 'zero' else 'Exact'
    if c['name'] in ('positive_well_conditioned_nonexact', 'F32_result_tie_even'):
        bounds = c['conditional_bounds'][0]['output_interval_bits']
        return {'ProjectInterval': list(map(lambda x: int(x, 16), bounds))}
    return 'Observation'


def runtime_case(row):
    group = {'geometry': 'Geometry', 'gradient': 'Gradient', 'numeric': 'Numeric',
             'errors_and_lifecycle': 'NativeExtension', 'preservation_regressions': 'Preservation',
             'f64_separate_audit': 'F64Diagnostic'}[row['group']]
    dt = row.get('dtype', 'F64' if group == 'F64Diagnostic' else None)
    matrices, expected, consumers, cells = [], [], [], []
    def matrix(name, record, physical=None):
        matrices.append({'name': name, 'values': record, 'physical': physical})
    def expect(name, record, stage='stored_destination', policy=None):
        expected.append({'name': name, 'values': record, 'stage': stage,
                         'policies': policies(record) if policy is None else policy})
    for key in ('lhs', 'rhs', 'input', 'weight', 'bias', 'upstream'):
        if key in row:
            matrix(key, row[key], row.get(key + '_physical'))
    for key in ('expected', 'forward', 'dA', 'dB', 'dx', 'dW', 'db', 'shared_source_gradient', 'stored_forward'):
        if key in row:
            expect(key, row[key], 'matmul_storage_before_bias' if key == 'stored_forward' else 'stored_destination')
    for key in ('lhs_backing_adjoint', 'rhs_backing_adjoint'):
        if key in row:
            expect(key, row[key], 'source_view_adjoint')
    for key, value in row.get('stages', {}).items():
        if key in ('dA_before_sum', 'dB_before_sum'):
            expect(key, value, 'matmul_storage_before_sum')
        elif key in ('dA_axes', 'dB_axes'):
            for s in value:
                expect(f'{key}.{s["axis"]}', s['output'], 'ascending_original_axis_stored_sum')
    for key, c in sorted(row.get('last_drop_consumers', {}).items()):
        consumers.append({'input': key, 'axis': c['axis'], 'expected': {
            'name': 'consumer.' + key, 'values': c['expected'],
            'policies': policies(c['expected']), 'stage': 'last_drop_typed_consumer'}})
    if group == 'Numeric':
        matrix('lhs', row['batched_lhs'])
        matrix('rhs', row['batched_rhs'])
        bits = [c['ideal_one_F32_scratch_then_storage']['bits'] for c in row['cells']]
        expect('expected', {'dtype': dt, 'shape': [len(bits), 1, 1], 'bits': bits},
               'whole_output_storage', [numeric_policy(c) for c in row['cells']])
        for c in row['cells']:
            bounds = c.get('conditional_bounds', [{}])[0]
            cells.append({'name': c['name'], 'ideal_bits': int(c['ideal_one_F32_scratch_then_storage']['bits'], 16),
                          'serial_bits': int(c['serial_F32_diagnostic']['bits'], 16),
                          'ideal_rational': c['exact_operand_dot']['rational'] or c['exact_operand_dot']['class'],
                          'sum_abs_products': bounds.get('sum_abs_products', ''),
                          'scratch_low': bounds.get('scratch_interval', ['', ''])[0],
                          'scratch_high': bounds.get('scratch_interval', ['', ''])[1]})
        if row['staging']:
            s = row['staging']['actual_broadcast_dA_fixture']
            for name, key in [('gradient.lhs', 'A'), ('gradient.rhs', 'B'), ('gradient.upstream', 'dY')]:
                matrix(name, s[key])
            for name, key in [('gradient.forward', 'forward'), ('gradient.dA_before_sum', 'dA_before_sum'), ('gradient.dA', 'dA')]:
                expect(name, s[key], 'stored_product_before_gradient_sum')
            h = pow2(-11 if dt == 'F16' else -8)
            x = Tensor((1, 2), [rounded(1, dt), rounded(h, dt)], dt)
            w = Tensor((2, 1), [rounded(1, dt)] * 2, dt)
            bias = Tensor((1,), [rounded(h, dt)], dt)
            y, stages = linear(x, w, bias)
            for name, t in [('linear.input', x), ('linear.weight', w), ('linear.bias', bias)]:
                matrix(name, t.export())
            expect('linear.stored', stages['matmul_stored'], 'matmul_storage_before_bias')
            expect('linear.output', y.export(), 'typed_bias_after_product_storage')
    if group == 'F64Diagnostic':
        if 'upstream' in row:
            y, _, _ = matmul(tensor_from(row['lhs']), tensor_from(row['rhs']))
            expect('forward', y.export(), 'double_only_not_mixed_support', ['Observation'] * len(y.values))
        if 'serial_F64_FMA' in row:
            expect('expected', {'dtype': 'F64', 'shape': [1, 1], 'bits': [row['serial_F64_FMA']['bits']]},
                   'explicit_double_schedule_observation', ['Observation'])
        for name in ('expected', 'dA', 'dB'):
            for e in expected:
                if e['name'] == name:
                    e['policies'] = ['Observation'] * len(e['values']['bits'])
        for i, z in enumerate(row.get('zero_or_empty_cases', [])):
            matrix(f'zero{i}.lhs', z['lhs']); matrix(f'zero{i}.rhs', z['rhs'])
            expect(f'zero{i}.expected', z['expected'], 'double_metadata_diagnostic', ['Observation'] * len(z['expected']['bits']))
    tracking = {'lhs_only': 'Lhs', 'rhs_only': 'Rhs', 'both': 'Both'}.get(row.get('tracked'), 'Both' if group == 'Gradient' else 'None')
    if group == 'F64Diagnostic' and 'upstream' in row:
        tracking = 'Both'
    if 'shared_source_gradient' in row:
        tracking = 'Shared'
        # A distinct repeated-source transpose route stays inside the same row.
        a = tensor_from(row['lhs']); g = tensor_from(row['upstream']); b = a.transpose()
        y, _, _ = matmul(a, b); da, db, _ = backward(a, b, g)
        joined = Tensor(a.shape, [rounded(add(x, y), storage(dt)) for x, y in zip(da.values, db.transpose().values, strict=True)], dt)
        expect('shared_transpose.forward', expected_tensor(y), 'repeated_source_transpose')
        expect('shared_transpose.gradient', expected_tensor(joined), 'typed_join_of_transpose_adjoints')
    raw_matrices, raw_expected = [], []
    if group in ('Geometry', 'Numeric'):
        a = tensor_from(row['batched_lhs'] if group == 'Numeric' else row['lhs'])
        b = tensor_from(row['batched_rhs'] if group == 'Numeric' else row['rhs'])
        # Scratch bypasses the final destination cast. Flex32 still converts
        # operands to F16. The independent exact model is used only where its
        # certificate/class policy supports the frozen comparison.
        scratch, _, _ = matmul(a, b, require_proof=False)
        bs = broadcast(a.shape[:-2], b.shape[:-2])
        scratch_bits = []
        for coord in coordinates(bs + (a.shape[-2], b.shape[-1])):
            batch, i, j = coord[:-2], coord[-2], coord[-1]
            av = [a.at(batch_coord(batch, a.shape[:-2]) + (i, k)) for k in range(a.shape[-1])]
            bv = [b.at(batch_coord(batch, b.shape[:-2]) + (k, j)) for k in range(b.shape[-2])]
            scratch_bits.append(json_value(rounded(exact_dot(av, bv, dt), 'F32'), 'F32')['bits'])
        ps = [numeric_policy(c) for c in row['cells']] if group == 'Numeric' else policies(row['expected'])
        ps = ['Observation' if isinstance(p, dict) or p == 'Observation' else
              'NaNClass' if decode(int(bit, 16), 'F32').kind == 'nan' else
              'Infinity' if decode(int(bit, 16), 'F32').kind == 'inf' else
              'ZeroClass' if decode(int(bit, 16), 'F32').q == 0 and p == 'ZeroClass' else 'Exact'
              for p, bit in zip(ps, scratch_bits, strict=True)]
        raw_expected.append({'name': 'scratch', 'values': {'dtype': 'F32', 'shape': list(scratch.shape), 'bits': scratch_bits},
                             'policies': ps, 'stage': 'F32_scratch_before_destination_RNE'})
    if group == 'Geometry' and row['key'].endswith('_dense_odd'):
        a, b = small_tensor((17, 33), dt, 2), small_tensor((33, 35), dt, 7)
        y, proof, _ = matmul(a, b)
        assert all(p['proved'] for p in proof)
        raw_matrices = [{'name': 'tile.lhs', 'values': a.export(), 'physical': layout(a, 'dense', True)},
                        {'name': 'tile.rhs', 'values': b.export(), 'physical': layout(b, 'dense', True)}]
        record = expected_tensor(y)
        raw_expected.append({'name': 'tile.output', 'values': record, 'policies': policies(record), 'stage': 'paired_tile_boundary_destination'})
        scratch = {'dtype': 'F32', 'shape': list(y.shape), 'bits': [json_value(v, 'F32')['bits'] for v in y.values]}
        raw_expected.append({'name': 'tile.scratch', 'values': scratch, 'policies': policies(record), 'stage': 'paired_tile_boundary_F32_scratch'})
    return {'id': row['key'], 'group': group, 'dtype': dt,
            'operation': 'Linear' if 'input' in row else 'Matmul' if matrices else 'ExternalGate',
            'tracking': tracking, 'matrices': matrices, 'expected': expected,
            'numeric': cells, 'consumers': consumers, 'raw_matrices': raw_matrices, 'raw_expected': raw_expected}


def encode_artifacts(rows):
    runtime = [runtime_case(row) for row in rows]
    blob = bytearray()
    dedup = {}
    def quote(s):
        return json.dumps(s, ensure_ascii=True)
    def array(xs):
        return '&[' + ','.join(map(str, xs)) + ']'
    def values(record):
        dt = record['dtype']; size = (sum(FORMATS[storage(dt)]) + 1) // 8
        data = b''.join(int(x, 16).to_bytes(size, 'little') for x in record['bits'])
        if data not in dedup:
            dedup[data] = len(blob); blob.extend(data)
        return f'Values{{dtype:DType::{dt},shape:{array(record["shape"])},offset:{dedup[data]},len:{len(record["bits"])}}}'
    def policy(p):
        if isinstance(p, dict):
            lo, hi = p['ProjectInterval']
            return f'Policy::ProjectInterval{{low:{lo},high:{hi}}}'
        return 'Policy::' + p
    def expectation(e):
        return f'Expected{{name:{quote(e["name"])},values:{values(e["values"])},policies:&[{",".join(map(policy, e["policies"]))}],stage:{quote(e["stage"])}}}'
    lines = ['// Generated by tools/nvidia/generate-mixed-matmul-fixtures.py. Data only.',
             '#[rustfmt::skip]', 'pub static CASES: &[Case] = &[']
    def matrix_descriptor(m):
        v = 'None'
        if m['physical']:
            p = m['physical']
            root = {'dtype': m['values']['dtype'], 'shape': [p['storage_elements']], 'bits': p['storage_bits']}
            v = f'Some(View{{strides:{array(p["strides_elements"])},base:{p["base_elements"]},root:{values(root)}}})'
        return f'Matrix{{name:{quote(m["name"])},values:{values(m["values"])},view:{v}}}'
    for c in runtime:
        ms = []
        for m in c['matrices']:
            v = 'None'
            if m['physical']:
                p = m['physical']
                root = {'dtype': m['values']['dtype'], 'shape': [p['storage_elements']], 'bits': p['storage_bits']}
                v = f'Some(View{{strides:{array(p["strides_elements"])},base:{p["base_elements"]},root:{values(root)}}})'
            ms.append(f'Matrix{{name:{quote(m["name"])},values:{values(m["values"])},view:{v}}}')
        ns = []
        for n in c['numeric']:
            ns.append('NumericCell{' + ','.join(k + ':' + (str(v) if isinstance(v, int) else quote(v)) for k, v in n.items()) + '}')
        cs = [f'Consumer{{input:{quote(x["input"])},axis:{x["axis"]},expected:{expectation(x["expected"])}}}' for x in c['consumers']]
        dt = 'None' if c['dtype'] is None else 'Some(DType::' + c['dtype'] + ')'
        lines.append('Case{' + f'id:{quote(c["id"])},group:Group::{c["group"]},dtype:{dt},operation:Operation::{c["operation"]},tracking:Tracking::{c["tracking"]},matrices:&[{",".join(ms)}],expected:&[{",".join(map(expectation, c["expected"]))}],numeric:&[{",".join(ns)}],consumers:&[{",".join(cs)}],raw_matrices:&[{",".join(map(matrix_descriptor,c["raw_matrices"]))}],raw_expected:&[{",".join(map(expectation,c["raw_expected"]))}]' + '},')
    lines.extend(['];', 'const FIXTURE_BYTES: &[u8] = include_bytes!("mixed_matmul_fixtures.bin");', ''])
    index = '\n'.join(lines).encode()
    counts = Counter()
    for c in runtime:
        counts['rows.' + c['group']] += 1
        counts['raw_expected_values'] += sum(len(e['values']['bits']) for e in c['raw_expected'])
        counts['raw_refinement_rows'] += bool(c['raw_matrices'])
        counts['numeric_cells'] += len(c['numeric'])
        counts['last_drop_consumers'] += len(c['consumers'])
        for e in c['expected']:
            counts['expected_tensors'] += 1
            counts['expected_values'] += len(e['values']['bits'])
            for p in e['policies']:
                counts['policy.' + ('ProjectInterval' if isinstance(p, dict) else p)] += 1
    manifest = {
        'schema_version': 1, 'status': 'proposed_freeze_requires_ROOT_approval',
        'numerical_policy': 'exact_normal_bits_class_gates_and_four_project_empirical_conditional_intervals',
        'conditional_model_label': 'project_empirical_conditional_normal_tree_not_vendor_guarantee',
        'vendor_order_or_rounding_guarantee': False, 'device_observations': [],
        'selected_rows': 121, 'reserved_rows': 3, 'hard_cap_rows': 124,
        'counts': dict(sorted(counts.items())), 'row_groups': BUDGET['row_counts'],
        'provenance': {
            'model_sha256': 'bec224241888c7259a316ac29aa307bd1dd3cb3b3cae5cd9d4e95eec9558abdb',
            'fixtures_sha256': 'ea1a32874f02abfbe3ebffc7c9df6eed5419e500c37b5af093c3b0b45666a08f',
            'policy_sha256': 'e17e389ded361ae999a5747bd6aed4acbaa53d338316c6c2293eebe810874efe',
            'independent_review_sha256': '2f6f09f37d60da5ef3cd83a410ccaacd29a8c67e78ff94030e427627ce2a73c1'},
        'generated': {'binary_bytes': len(blob), 'binary_sha256': hashlib.sha256(blob).hexdigest(),
                      'index_sha256': hashlib.sha256(index).hexdigest()},
        'external_required_gates': {
            'NativeExtension': 'raw owner: invalid metadata/recovery, physical guards and resident lifetime',
            'Preservation': 'eight existing regression routes; never new three-dtype acceptance',
            'F64Diagnostic': 'genuine double or unsupported; never narrowed fallback or mixed acceptance'},
        'frontend_diagnostic': {'id': 'public_linear_rank1_zero_frontend_unsqueeze',
            'label': 'frontend_source_derived_not_backend_result',
            'detail': 'Tensor<1>::unsqueeze::<2> zero reshape panics before backend; isolate CLI; Native raw primitive bridge is distinct'},
        'policy': ORIGINAL_POLICY, 'reviewed_fixtures': rows, 'runtime': runtime,
    }
    return bytes(blob), index, (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()


def self_test(rows):
    # Independent host encoding checks do not produce fixture values.
    counts = {}
    for fmt in ('F16', 'BF16'):
        n = 0
        for bits in range(65536):
            value = decode(bits, fmt)
            if value.kind == 'finite':
                assert encode(value, fmt) == bits
                n += 1
        counts[fmt] = n
    for low in range(0x7bff):
        mid = (decode(low, 'F16').q + decode(low + 1, 'F16').q) / 2
        for q in (mid, -mid):
            assert encode(q, 'F16') == int.from_bytes(struct.pack('>e', float(q)), 'big')
    from itertools import permutations
    terms = [finite(Q(x, 256)) for x in (1, -1, 1, 1, -1, 1)]
    def trees(xs):
        if len(xs) == 1:
            return {xs[0]}
        return {rounded(add(a, b), 'F32') for i in range(1, len(xs))
                for a in trees(xs[:i]) for b in trees(xs[i:])}
    assert all_order_certificate(terms)['proved']
    for order in permutations(terms):
        assert {v.q for v in trees(order)} == {Q(2, 256)}
        for mode in ('rn', 'rz', 'rp', 'rm'):
            acc = finite(0)
            for term in order:
                acc = rounded(add(acc, term), 'F32', mode)
            assert acc.q == Q(2, 256)
    scatter_rows = 0
    for row in rows:
        if row['group'] == 'gradient' and 'lhs' in row:
            a, b, g = map(tensor_from, (row['lhs'], row['rhs'], row['upstream']))
            da, db = adjoint_scatter(a, b, g)
            assert da == [v.q for v in tensor_from(row['dA']).values]
            assert db == [v.q for v in tensor_from(row['dB']).values]
            scatter_rows += 1
    return {'finite_roundtrips': counts, 'independent_half_midpoints': 63486,
            'all_order_permutations': 720, 'trees_per_permutation': 42,
            'independent_index_scatter_rows': scatter_rows, 'passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    rows = make_rows()
    binary, index, manifest = encode_artifacts(rows)
    root = Path(__file__).resolve().parents[2]
    outputs = {
        root / 'os/library/ai/src/validation/mixed_matmul_fixtures.bin': binary,
        root / 'os/library/ai/src/validation/mixed_matmul_index.rs': index,
        root / 'tools/nvidia/mixed-matmul-manifest.json': manifest}
    for path, data in outputs.items():
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit(f'reproduction mismatch: {path}')
        else:
            path.write_bytes(data)
    result = {'check': args.check, 'rows': 121, 'reserves': 3, 'numeric_cells': 65,
              'consumers': 141, 'binary_bytes': len(binary),
              'manifest_sha256': hashlib.sha256(manifest).hexdigest()}
    if args.self_test:
        result['self_test'] = self_test(rows)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
