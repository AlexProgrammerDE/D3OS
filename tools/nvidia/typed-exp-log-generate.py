#!/usr/bin/env python3
"""Generate independent Exp/Log fixtures with directed MPFR intervals.

Requires Python 3 and MPFR 4.2.2. No production arithmetic or GPU is used.
Use --check for read-only reproduction, or --output-root for explicit writes.
"""
import ctypes as C
import ctypes.util
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import struct
from collections import Counter

LIBNAME = ctypes.util.find_library('mpfr')
M = C.CDLL(LIBNAME)
class MP(C.Structure):
    _fields_ = [('prec', C.c_long), ('sign', C.c_int), ('exp', C.c_long), ('limbs', C.POINTER(C.c_ulong))]
P = C.POINTER(MP)
def bind(name, args, result=C.c_int):
    f = getattr(M, 'mpfr_' + name)
    f.argtypes, f.restype = args, result
    return f
init = bind('init2', [P,C.c_long], None)
clear = bind('clear', [P], None)
setstr = bind('set_str', [P,C.c_char_p,C.c_int,C.c_int])
mul2 = bind('mul_2si', [P,P,C.c_long,C.c_int])
exp = bind('exp', [P,P,C.c_int])
log = bind('log', [P,P,C.c_int])
getstr = bind('get_str', [C.c_void_p,C.POINTER(C.c_long),C.c_int,C.c_size_t,P,C.c_int], C.c_void_p)
free = bind('free_str', [C.c_void_p], None)
version = bind('get_version', [], C.c_char_p)
F = {'f64':(11,52), 'f32':(8,23), 'flex32':(8,23), 'f16':(5,10), 'bf16':(8,7)}
def info(fmt):
    e,f=F[fmt]
    return e,f,(1 << (e-1))-1,1 << (e+f),((1 << e)-1) << f

def decode(bits,fmt):
    e,f,bias,sign,inf=info(fmt)
    field=(bits >> f)&((1 << e)-1)
    frac=bits&((1 << f)-1)
    if field==(1 << e)-1:return ('nan' if frac else 'inf', -1 if bits&sign else 1)
    n=frac if field==0 else (1 << f)+frac
    power=(1-bias if field==0 else field-bias)-f
    q=Q(n << max(power,0),1 << max(-power,0))
    return ('finite',-q if bits&sign else q)

def rne_int(q):
    n,r=divmod(q.numerator,q.denominator)
    t=r*2-q.denominator
    return n+int(t>0 or (t==0 and n&1))

def encode(q,fmt,negative_zero=False):
    e,f,bias,sign,inf=info(fmt)
    s=sign if q<0 or (not q and negative_zero) else 0
    q=abs(q)
    if not q:return s
    k=q.numerator.bit_length()-q.denominator.bit_length()
    if q < Q(1 << max(k,0),1 << max(-k,0)):k-=1
    step=max(k-f,1-bias-f)
    n=rne_int(q / Q(1 << max(step,0),1 << max(-step,0)))
    if not n:return s
    if n>=(1 << (f+1)):
        n>>=1
        step+=1
    if step+f>bias:return s|inf
    if n<(1 << f):return s|n
    return s|((step+f+bias)<<f)|(n-(1 << f))

def as_q(x):
    exponent=C.c_long()
    ptr=getstr(None,C.byref(exponent),2,0,C.byref(x),0)
    try:digits=C.string_at(ptr).decode()
    finally:free(ptr)
    negative=digits.startswith('-')
    digits=digits.lstrip('-')
    n=int(digits,2)
    power=exponent.value-len(digits)
    q=Q(n << max(power,0),1 << max(-power,0))
    return -q if negative else q

def interval(q,op,precision):
    values=[MP(),MP(),MP()]
    for x in values:init(C.byref(x),precision)
    x,lo,hi=values
    try:
        # Every input is dyadic. Setting its integer numerator is exact at this precision.
        assert q.denominator&(q.denominator-1)==0
        n=abs(q.numerator)
        significant_bits=(n.bit_length() - ((n & -n).bit_length()-1)) if n else 0
        assert significant_bits<=precision
        assert setstr(C.byref(x),str(q.numerator).encode(),10,0)==0
        assert mul2(C.byref(x),C.byref(x),-(q.denominator.bit_length()-1),0)==0
        fn={'exp':exp,'log':log}[op]
        fn(C.byref(lo),C.byref(x),3) # RNDD: lower enclosure
        fn(C.byref(hi),C.byref(x),2) # RNDU: upper enclosure
        return as_q(lo),as_q(hi)
    finally:
        for v in values:clear(C.byref(v))

def confirmed(q,op,fmt,staged=False,start=96):
    p=start
    while p<=12288:
        lo,hi=interval(q,op,p)
        a,b=encode(lo,fmt),encode(hi,fmt)
        c,d=encode(lo,'f32'),encode(hi,'f32')
        if a==b and (not staged or c==d):
            stage=None
            if staged:
                kind,v=decode(c,'f32')
                stage=encode(v,fmt) if kind=='finite' else (info(fmt)[4] | (info(fmt)[3] if v<0 else 0))
            return a,stage,p,c if staged else None
        p*=2
    raise RuntimeError(('unresolved interval',fmt,op,str(q)))

def reference(bits,fmt,op,start=96):
    kind,q=decode(bits,fmt)
    _,_,_,sign,inf=info(fmt)
    if kind=='nan':return None,None,0,None
    if op=='log':
        if kind=='inf':return (inf if q>0 else None),None,0,None
        if q==0:return sign|inf,None,0,None
        if q<0:return None,None,0,None
        if q==1:return 0,0 if fmt in ('f16','bf16') else None,0,0
    else:
        if kind=='inf':return (inf if q>0 else 0),None,0,None
        if not q:return encode(Q(1),fmt),encode(Q(1),fmt) if fmt in ('f16','bf16') else None,0,None
        # e^16384 > 2^16384 and e^-16384 < 2^-16384. Both exceed all target limits.
        if q>16384:return inf,inf if fmt in ('f16','bf16') else None,0,None
        if q< -16384:return 0,0 if fmt in ('f16','bf16') else None,0,None
    return confirmed(q,op,fmt,fmt in ('f16','bf16'),start)

def hexcode(bits,fmt):
    return None if bits is None else f'{bits:0{(sum(F[fmt])+1)//4}x}'

def cases(fmt):
    e,f,bias,sign,inf=info(fmt)
    groups={}
    def add(b,label):
        if 0<=b<sign*2:groups.setdefault(b,set()).add(label)
    for s in (0,sign):
        for b in (0,1,2,3,(1<<f)-1,1<<f,(1<<f)+1,inf-2,inf-1,inf,inf+1,inf+(1<<(f-1)),inf+(1<<(f-1))+0x15,inf+(1<<f)-1):add(s|b,'special-storage')
        for exponent in range(1<<e):
            for fraction in (0,1,(1<<(f-1)),(1<<f)-1):add(s|(exponent<<f)|fraction,'exponent-strata')
    one=bias<<f
    for delta in range(-32,33):add(one+delta,'log-near-one')
    for n in (-1000,-800,-746,-745,-744,-710,-709,-104,-103,-100,-90,-88,-20,-17,-16,-1,1,10,11,12,80,88,89,100,700,709,710,800,1000):
        b=encode(Q(n),fmt)
        for delta in range(-2,3):add(b+delta,'range-and-f64-separation')
    for exponent in (-1074,-1022,-150,-149,-133,-126,-53,-52,-25,-24,-23,-14,-11,-10,-8,-7,0,24,53,100,128,500,1023):
        q=Q(1<<max(exponent,0),1<<max(-exponent,0))
        for s in (1,-1):
            b=encode(s*q,fmt)
            for d in range(-2,3):add(b+d,'dyadic-neighborhood')
    # Exact output thresholds: log maps these to exp transition inputs.
    minsub=decode(1,fmt)[1]
    minnormal=decode(1<<f,fmt)[1]
    maxfinite=decode(inf-1,fmt)[1]
    ulpmax=maxfinite-decode(inf-2,fmt)[1]
    thresholds={'zero-midpoint':minsub/2,'first-subnormal-midpoint':minsub*Q(3,2),
      'min-subnormal-value':minsub,'normal-midpoint':minnormal-minsub/2,
      'min-normal-value':minnormal,'max-finite-value':maxfinite,'overflow-midpoint':maxfinite+ulpmax/2}
    boundaries=[]
    for name,q in thresholds.items():
        b,_,p,_=confirmed(q,'log',fmt,start=192)
        boundaries.append({'name':name,'output_threshold_numerator':str(q.numerator),'output_threshold_denominator':str(q.denominator),'nearest_input':hexcode(b,fmt),'precision':p})
        for d in range(-4,5):add(b+d,'exp-'+name)
    seed_format='f32' if fmt=='flex32' else fmt
    state=hashlib.sha256((seed_format+' exact stored exp/log v1').encode()).digest()
    for _ in range(256):
        state=hashlib.sha256(state).digest()
        add(int.from_bytes(state,'little')%(sign*2),'sha256-strata')
    return groups,boundaries

def self_check():
    # All half representations round-trip, including signed zero and finite subnormals.
    checked=0
    for fmt in ('f16','bf16'):
        for b in range(65536):
            kind,q=decode(b,fmt)
            if kind=='finite':
                assert encode(q,fmt,bool(b&info(fmt)[3]))==b
                if fmt=='f16':
                    host=struct.unpack('<e', b.to_bytes(2,'little'))[0]
                    assert Q(host)==q
                    assert int.from_bytes(struct.pack('<e',host),'little')==b
                else:
                    host=struct.unpack('<f',(b<<16).to_bytes(4,'little'))[0]
                    assert Q(host)==q
            checked+=1
    # Tie parity, gradual underflow, overflow, sign symmetry for every format.
    for fmt in F:
        e,f,bias,sign,inf=info(fmt)
        m=decode(1,fmt)[1]
        assert encode(m/2,fmt)==0 and encode(3*m/2,fmt)==2
        assert encode(-m/2,fmt)==sign
        maximum=decode(inf-1,fmt)[1]
        ulp=maximum-decode(inf-2,fmt)[1]
        assert encode(maximum+ulp/2,fmt)==inf
        one=bias<<f
        assert encode((decode(one,fmt)[1]+decode(one+1,fmt)[1])/2,fmt)==one
        assert reference(0,fmt,'exp')[0]==one
        assert reference(sign,fmt,'log')[0]==sign|inf
        assert reference(one|sign,fmt,'log')[0] is None
        for b in (1,one-1,one+1,encode(Q(-1),fmt)):
            for op in ('exp','log'):
                a=reference(b,fmt,op,96)
                z=reference(b,fmt,op,384)
                assert a[:2]==z[:2],(fmt,b,op,a,z)
    return {'half_roundtrips':checked,'independent_struct_decode_and_f16_encode':True,'ties_ranges_signs_and_96_vs_384':True}

POLICY_TEXT = r'''{
  "classification_diagnostics": [
    "expected_normal_actual_zero",
    "expected_subnormal_actual_zero",
    "expected_zero_actual_nonzero",
    "expected_nan_actual_non_nan",
    "expected_inf_actual_other",
    "expected_finite_actual_inf",
    "zero_sign_mismatch",
    "infinity_sign_mismatch",
    "finite_sign_mismatch",
    "subnormal_to_normal",
    "normal_to_subnormal",
    "nan_output_quiet_bit_and_payload_observation_only"
  ],
  "classification_qualification_target": {
    "infinity": "Expected infinity requires matching infinity sign; unexpected infinity is a separate qualification failure.",
    "nan": "Expected NaN requires actual NaN class; no payload/sign gate.",
    "nonzero": "Expected finite nonzero requires actual finite nonzero with matching sign.",
    "normal_subnormal_boundary": "Nonzero normal <-> subnormal transitions are separate diagnostics and remain subject to the numerical/set target.",
    "normative_status": "Explicit strict project target; boundary failures can occur within a vendor empirical error allowance and must be reported separately.",
    "zero": "Expected zero requires matching signed zero; unexpected zero is a separate qualification failure."
  },
  "comparator_order": [
    "validate structural invariants",
    "classify raw bits",
    "apply exact_special_policy",
    "evaluate separate classification_qualification_target and retain every transition",
    "apply finite numerical target or half neighbor-image target",
    "report distances from both half ideals and modeled-set membership",
    "report observed maxima/histograms/worst inputs separately from deterministic expected_summary",
    "report qualification pass only for the executed bounded cases, never as a vendor guarantee"
  ],
  "deterministic_case_counts": {
    "bf16": {
      "exp": {
        "all": 65536,
        "expected_inf": 15567,
        "expected_nan": 254,
        "expected_normal": 34145,
        "expected_subnormal": 11,
        "expected_zero": 15559,
        "finite": 49714
      },
      "log": {
        "all": 65536,
        "expected_inf": 3,
        "expected_nan": 32894,
        "expected_normal": 32638,
        "expected_subnormal": 0,
        "expected_zero": 1,
        "finite": 32639
      }
    },
    "f16": {
      "exp": {
        "all": 65536,
        "expected_inf": 12917,
        "expected_nan": 2046,
        "expected_normal": 37479,
        "expected_subnormal": 891,
        "expected_zero": 12203,
        "finite": 50572
      },
      "log": {
        "all": 65536,
        "expected_inf": 3,
        "expected_nan": 33790,
        "expected_normal": 31742,
        "expected_subnormal": 0,
        "expected_zero": 1,
        "finite": 31743
      }
    },
    "f32": {
      "exp": {
        "all": 2621,
        "expected_inf": 609,
        "expected_nan": 14,
        "expected_normal": 1360,
        "expected_subnormal": 49,
        "expected_zero": 589,
        "finite": 1997
      },
      "log": {
        "all": 2621,
        "expected_inf": 3,
        "expected_nan": 1296,
        "expected_normal": 1321,
        "expected_subnormal": 0,
        "expected_zero": 1,
        "finite": 1322
      }
    },
    "f64": {
      "exp": {
        "all": 16972,
        "expected_inf": 4148,
        "expected_nan": 8,
        "expected_normal": 8615,
        "expected_subnormal": 46,
        "expected_zero": 4155,
        "finite": 12815
      },
      "log": {
        "all": 16972,
        "expected_inf": 3,
        "expected_nan": 8486,
        "expected_normal": 8482,
        "expected_subnormal": 0,
        "expected_zero": 1,
        "finite": 8483
      }
    },
    "flex32": {
      "exp": {
        "all": 2621,
        "expected_inf": 609,
        "expected_nan": 14,
        "expected_normal": 1360,
        "expected_subnormal": 49,
        "expected_zero": 589,
        "finite": 1997
      },
      "log": {
        "all": 2621,
        "expected_inf": 3,
        "expected_nan": 1296,
        "expected_normal": 1321,
        "expected_subnormal": 0,
        "expected_zero": 1,
        "finite": 1322
      }
    }
  },
  "equality_rule": "expected_summary equality compares deterministic fields only. Qualification targets and corpus identity are frozen by hashes. Observed metrics and pass/fail results never participate.",
  "exact_special_policy": {
    "exp": {
      "nan": "nan_class_only",
      "negative_infinity": "positive_zero",
      "negative_zero": "one",
      "positive_infinity": "positive_infinity",
      "positive_zero": "one"
    },
    "log": {
      "nan": "nan_class_only",
      "negative_finite_nonzero": "nan_class_only",
      "negative_infinity": "nan_class_only",
      "negative_zero": "negative_infinity",
      "one": "positive_zero",
      "positive_infinity": "positive_infinity",
      "positive_zero": "negative_infinity"
    },
    "provenance": "mathematical policy with CUDA Math API 13.4 special-value support; TileIR end-to-end GPU conformance remains unmeasured"
  },
  "expected_summary_fields": [
    "schema_version",
    "policy_sha256",
    "corpus_manifest_sha256",
    "options",
    "selected_case_count",
    "selected_dtype_operation_counts",
    "layout_case_count",
    "structural_check_counts"
  ],
  "finite_metric": {
    "binade_boundary": "integer representable-step distance, not real-error divided by a single chosen binade spacing",
    "distance": "abs(rank(actual)-rank(reference)); only finite nonzero pairs",
    "nan_infinity_zero": "no finite ULP metric",
    "rank": "for width w: positive b -> b | (1<<(w-1)); negative b -> (~b)&((1<<w)-1)"
  },
  "half_acceptance_set": {
    "algorithm": [
      "Obtain c = interval-confirmed ideal RN32(f(exact stored half input)).",
      "Let b = 2 for exp, 1 for log.",
      "Enumerate all F32 codes with ordered representable-step rank from rank(c)-b through rank(c)+b, restricted to valid non-NaN values.",
      "Apply exact destination RNE to each enumerated F32 value, including zero, subnormals and infinities.",
      "Deduplicate destination encodings. Actual destination bits must belong to this set.",
      "Apply the independent classification and nonzero rules even when set membership succeeds."
    ],
    "direct_reference": "Always retain ideal destination RN(f(x)) as a separate diagnostic. It never replaces the declared staged schedule.",
    "exact_special_inputs": "Use exact_special_policy first. Do not widen acceptance for exp signed zero, log one, infinities or NaNs.",
    "finite_input_f32_overflow_or_underflow": "If ideal RN32 is infinity or zero, retain class diagnostics and compare final destination class with the staged ideal. No finite ULP distance is defined there.",
    "meaning": "A derived project qualification envelope for the declared F32 operation followed by one destination RNE; not proof of hidden intermediate GPU bits.",
    "necessary_not_sufficient": "Destination set membership cannot prove which F32 intermediate the GPU actually computed."
  },
  "mandatory_structural_checks": [
    "shape",
    "preserved_dtype",
    "output_byte_count",
    "input_alias_bytes_unchanged_including_nan_payloads",
    "guards_unchanged",
    "empty_has_zero_launches"
  ],
  "no_ftz_policy": {
    "initial_status": "strict project qualification target, not a stated vendor guarantee",
    "minimum_subnormal": "Expected code 1 vs actual code 0 is a classification failure regardless of distance 1.",
    "requirement": "Expected nonzero -> actual zero fails the separate project classification target, even if an ULP limit or half acceptance set permits it."
  },
  "numerical_criteria": [
    {
      "cuda_math_observed_ulp": 1,
      "dtype": "f64",
      "f32_intermediate_step_limit": null,
      "finite_ulp_gate": 1,
      "half_api_zero_ulp_not_applicable": false,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "exp",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": 1,
      "dtype": "f64",
      "f32_intermediate_step_limit": null,
      "finite_ulp_gate": 1,
      "half_api_zero_ulp_not_applicable": false,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "log",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": 2,
      "dtype": "f32",
      "f32_intermediate_step_limit": null,
      "finite_ulp_gate": 2,
      "half_api_zero_ulp_not_applicable": false,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "exp",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": 1,
      "dtype": "f32",
      "f32_intermediate_step_limit": null,
      "finite_ulp_gate": 1,
      "half_api_zero_ulp_not_applicable": false,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "log",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": 2,
      "dtype": "flex32",
      "f32_intermediate_step_limit": null,
      "finite_ulp_gate": 2,
      "half_api_zero_ulp_not_applicable": false,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "exp",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": 1,
      "dtype": "flex32",
      "f32_intermediate_step_limit": null,
      "finite_ulp_gate": 1,
      "half_api_zero_ulp_not_applicable": false,
      "mode": "finite_nonzero_representable_step_limit",
      "operation": "log",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": null,
      "dtype": "f16",
      "f32_intermediate_step_limit": 2,
      "finite_ulp_gate": null,
      "half_api_zero_ulp_not_applicable": true,
      "mode": "f32_neighbor_destination_image",
      "operation": "exp",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": null,
      "dtype": "f16",
      "f32_intermediate_step_limit": 1,
      "finite_ulp_gate": null,
      "half_api_zero_ulp_not_applicable": true,
      "mode": "f32_neighbor_destination_image",
      "operation": "log",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": null,
      "dtype": "bf16",
      "f32_intermediate_step_limit": 2,
      "finite_ulp_gate": null,
      "half_api_zero_ulp_not_applicable": true,
      "mode": "f32_neighbor_destination_image",
      "operation": "exp",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    },
    {
      "cuda_math_observed_ulp": null,
      "dtype": "bf16",
      "f32_intermediate_step_limit": 1,
      "finite_ulp_gate": null,
      "half_api_zero_ulp_not_applicable": true,
      "mode": "f32_neighbor_destination_image",
      "operation": "log",
      "project_qualification_target": true,
      "provenance": "CUDA Programming Guide v13.4.2 observed error table; TileIR Exp Full explicitly refers to CUDA Math API; TileIR Log transfer is an explicit project target",
      "vendor_guaranteed": false
    }
  ],
  "observed_only_fields": [
    "max_observed_ulp",
    "ulp_histogram",
    "worst_input_bits",
    "classification_transition_counts",
    "expected_nonzero_actual_zero_counts",
    "subnormal_preservation_counts",
    "signed_zero_mismatch_counts",
    "nan_payload_observations",
    "gpu_metadata",
    "compiler_metadata",
    "linux_backend_selection",
    "unsupported_case_errors",
    "accuracy_qualification_status",
    "qualification_pass",
    "classification_qualification_failures",
    "numerical_qualification_failures",
    "half_envelope_membership_failures"
  ],
  "options": {
    "dtype": "optional one of scope.formats, otherwise all",
    "finite": "retain exactly finite input AND finite ideal output, including signed zeros/subnormals; negative log inputs and overflowing exp inputs excluded",
    "no_flex32": "exclude only Flex32 cases",
    "operation": "optional exp/15 or log/16, otherwise both",
    "skip_rank_zero": "exclude only rank-0 layouts; rank-1 [1] remains"
  },
  "policy_id": "typed-exp-log-reference-v2-qualification-targets",
  "references": {
    "bf16": "primary staged RNB16(RN32(f(exactB16))); also report ideal RNB16(f(exactB16))",
    "f16": "primary staged RN16(RN32(f(exact16))); also report ideal RN16(f(exact16))",
    "f32": "RN32(f(exact32))",
    "f64": "RN64(f(exact64))",
    "flex32": "RN32(f(exact32)); preserve Flex32 metadata",
    "inputs": "exact little-endian stored encodings",
    "nan": "null ideal bits means class NaN; output payload/sign not specified"
  },
  "schema_version": 2,
  "scope": {
    "existing_f32_math_validator": "separate, unchanged by this policy",
    "exp": 15,
    "formats": [
      "f64",
      "f32",
      "flex32",
      "f16",
      "bf16"
    ],
    "log": 16
  },
  "source_urls": {
    "cuda_guide": "https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#exponential-functions",
    "cuda_guide_version_observed": "13.4.2, local captured HTML version banner",
    "cuda_math_version": "13.4",
    "mpfr_version": "4.2.2",
    "tileir": "https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html#cuda-tile-exp",
    "vendor_compiler_version": "13.4.92"
  },
  "status": "frozen proposed project qualification contract; not a vendor guarantee; no GPU measurements",
  "supersedes_for_qualification": {
    "v1_file": "typed-exp-log-reference-policy.json",
    "v1_preserved": true,
    "v1_sha256": "dc6cb240bb4c9912279125baf1a0086e2307dc8b1904ed0ab1867112a4c90dcf"
  }
}
'''
PRIMARY_SOURCES = {'cuda-guide-13.2.0-math.html': {'bytes': 244769, 'resolved_url': 'https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html', 'retrieved_utc': '2026-10-25T23:14:12.512435+00:00', 'sha256': '49aa98a4f3cece0ab1f4ef470f8ed1edf922ea4920b240acc330f9059e6075b9', 'url': 'https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html'}, 'cuda-guide-current-math.html': {'bytes': 245348, 'resolved_url': 'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html', 'retrieved_utc': '2026-10-25T23:14:12.346408+00:00', 'sha256': '4277615c7d34b1f24fa9dd7908b7f71e63cc87d516740cc89bf56f05b5be7f77', 'url': 'https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html'}, 'cuda-math-13.4-double.html': {'bytes': 245449, 'resolved_url': 'https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html', 'retrieved_utc': '2026-10-25T23:14:12.217697+00:00', 'sha256': '33fedd606826502cf91036bc5f054dc3ef266d3228930ca667d30e5fa1fa7855', 'url': 'https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html'}, 'cuda-math-13.4-single.html': {'bytes': 243054, 'resolved_url': 'https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html', 'retrieved_utc': '2026-10-25T23:14:12.060736+00:00', 'sha256': 'de1b072026dae5d016001e21574daa1cb566ed21c424dccc29b069bad21b4195', 'url': 'https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html'}, 'mpfr-4.2.2-manual.html': {'bytes': 562991, 'resolved_url': 'https://www.mpfr.org/mpfr-4.2.2/mpfr.html', 'retrieved_utc': '2026-10-25T23:14:12.902433+00:00', 'sha256': 'ee996f3dcc11d1f88393da162bdd90c3fa29191ca658c0fc594ffd38a22638bc', 'url': 'https://www.mpfr.org/mpfr-4.2.2/mpfr.html'}, 'tileir-13.4-operations.html': {'bytes': 1177348, 'resolved_url': 'https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html', 'retrieved_utc': '2026-10-25T23:14:11.874698+00:00', 'sha256': 'b05512d7cc3b1571adacb29ab40e209ade6e826fc8d38cb0b760881e2b281773', 'url': 'https://docs.nvidia.com/cuda/tile-ir/13.4/sections/operations.html'}}
SOURCE_DEPENDENCIES = {'Ops.td': {'bytes': 203041, 'path': '/home/alex/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/2f47ad0/cutile-rs/cuda-tile-rs/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Ops.td', 'sha256': '1db40da82ff66ac7011e3ca31767a05dcac7ce8281f234b0c142f200021c7232'}, 'Types.td': {'bytes': 18558, 'path': '/home/alex/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/2f47ad0/cutile-rs/cuda-tile-rs/cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/Types.td', 'sha256': '5cceb7b5d26774cf6f934a98b70f3b2a9b66738156b475885913f08cecf8812f'}, 'cutile-core.rs': {'bytes': 162664, 'path': '/home/alex/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/2f47ad0/cutile-rs/cutile/src/_core.rs', 'sha256': 'a22300b5eecc329e69c6b17ae8d60de8b16b8771fd6fc040d5b4cece6a912aa9'}, 'burn-backend-0.22.1-cargo_vcs_info.json': {'bytes': 113, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-backend-0.22.1/.cargo_vcs_info.json', 'sha256': '14f661b0ae563e2f0fd176f411f1eed236d54e443b75df0f65656851496291c4'}, 'burn-cubecl-0.22.1-cargo_vcs_info.json': {'bytes': 112, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-cubecl-0.22.1/.cargo_vcs_info.json', 'sha256': '36d089884436b76ad90a9808c4376bfb36dcd12b8ebe20713b565dc5ea206eae'}, 'cubecl-cpp-0.11.3-cargo_vcs_info.json': {'bytes': 111, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cpp-0.11.3/.cargo_vcs_info.json', 'sha256': '09ab1cfa02b656ebf0739d44b0b30bb91d521548a1ad5eb24a314151eb75d1d0'}, 'cubecl-cuda-0.11.3-cargo_vcs_info.json': {'bytes': 112, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-cuda-0.11.3/.cargo_vcs_info.json', 'sha256': '61097cdf4279207321ce123c8407df762952252dd55fec158a4ad11d55b6aa04'}, 'cubecl-llvm-0.11.3-cargo_vcs_info.json': {'bytes': 112, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-llvm-0.11.3/.cargo_vcs_info.json', 'sha256': '82e10d06b9ff96ca780b4526fd1b8f84a4813ed3fa06170a7282dd698a7416d5'}, 'cubecl-libdevice.rs': {'bytes': 2371, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-llvm-0.11.3/src/nvptx/libdevice.rs', 'sha256': '39807f7f63cfc551e91dad7e907f4bec8242ada3c96435093876ee27fc80b083'}, 'cubecl-llvm-math.rs': {'bytes': 19486, 'path': '/home/alex/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-llvm-0.11.3/src/shared/to_llvm/math.rs', 'sha256': 'e843529cae0f6acce1d5d0c716d8fe0977615ec3d934a69927d2c1016cf19fa9'}, 'cuda-math-functions.h': {'bytes': 243685, 'path': '/home/alex/.t3/worktrees/D3OS/t3-41013c62/target/nvidia/cutile-toolkit/packages/nvidia/cu13/include/crt/math_functions.h', 'sha256': 'c6446ed6276ffa2fc56852355c795d6a24c6aee125393ad8a68a3eb5dabc123d'}, 'compiler-qualification.txt': {'bytes': 13143, 'path': '/home/alex/.cache/d3os/typed-exp-log-compiler-qualification.txt', 'sha256': 'cb2f2e8c51858028245f4afe3e0a5150c4c231918a7ce41c92b6e224c8cedb05'}}
PREPARATION_CORPUS_SHA256 = '63b98afcb862fcf934de753554c28e85e55ac45566f33cc60ed9f39e6a9c4884'


def canonical_nan(fmt):
    return info(fmt)[4] | (1 << (F[fmt][1] - 1))


def rank32(bits):
    return (~bits & 0xffffffff) if bits & 0x80000000 else bits | 0x80000000


def unrank32(rank):
    return rank & 0x7fffffff if rank & 0x80000000 else ~rank & 0xffffffff


def convert32(bits, fmt):
    kind, value = decode(bits, 'f32')
    if kind == 'nan':
        return canonical_nan(fmt)
    if kind == 'inf':
        return info(fmt)[4] | (info(fmt)[3] if value < 0 else 0)
    return encode(value, fmt, bool(bits & 0x80000000))


def exact_special(bits, fmt, op):
    kind, value = decode(bits, fmt)
    return (kind != 'finite' or value == 0 or
            (op == 'log' and (value < 0 or value == 1)))


def fixture_row(bits, fmt, op):
    ideal, stage, precision, mid = reference(bits, fmt, op)
    ideal = canonical_nan(fmt) if ideal is None else ideal
    if fmt not in ('f16', 'bf16'):
        return (bits, ideal, ideal, ideal, ideal, 0), precision
    if stage is None:
        stage = ideal
    if mid is None:
        kind, value = decode(stage, fmt)
        mid = (canonical_nan('f32') if kind == 'nan' else
               (info('f32')[4] | (info('f32')[3] if value < 0 else 0)) if kind == 'inf' else
               encode(value, 'f32', bool(stage & info(fmt)[3])))
    allowed = {stage}
    kind, value = decode(mid, 'f32')
    if not exact_special(bits, fmt, op) and kind == 'finite' and value:
        budget = 2 if op == 'exp' else 1
        allowed = {convert32(unrank32(r), fmt)
                   for r in range(rank32(mid) - budget, rank32(mid) + budget + 1)
                   if decode(unrank32(r), 'f32')[0] != 'nan'}
    # At most five consecutive F32 values map to at most two adjacent half codes.
    # This assertion makes the inclusive endpoints an exact set representation.
    sign = info(fmt)[3]
    mask = sign * 2 - 1
    rank = lambda b: ~b & mask if b & sign else b | sign
    codes = sorted(allowed, key=rank)
    assert all(rank(b) == rank(a) + 1 for a, b in zip(codes, codes[1:]))
    return (bits, ideal, stage, codes[0], codes[-1], mid), precision


def generate(reference_cache=None):
    assert version().decode() == '4.2.2', 'Use MPFR 4.2.2 for pinned reproduction'
    checks = self_check()
    blob = bytearray()
    ranges, counts = [], {}
    for fmt in F:
        groups, boundaries = cases(fmt)
        inputs = list(range(65536)) if fmt in ('f16', 'bf16') else sorted(groups)
        width = (sum(F[fmt]) + 1) // 8
        per_op, fmt_counts = [], {}
        for op in ('exp', 'log'):
            per_op.append((len(blob), len(inputs)))
            classes, precisions = Counter(), Counter()
            finite_count, differences = 0, 0
            for bits in inputs:
                words, precision = fixture_row(bits, fmt, op)
                _, ideal, staged, low, high, mid = words
                for word in words[:5]:
                    blob.extend(word.to_bytes(width, 'little'))
                blob.extend(mid.to_bytes(4, 'little'))
                kind, value = decode(staged, fmt)
                magnitude = staged & (info(fmt)[3] - 1)
                result_class = kind if kind != 'finite' else ('zero' if not magnitude else
                    'subnormal' if magnitude < 1 << F[fmt][1] else 'normal')
                classes[result_class] += 1
                precisions[str(precision)] += 1
                finite_count += decode(bits, fmt)[0] == 'finite' and decode(ideal, fmt)[0] == 'finite'
                differences += ideal != staged
            fmt_counts[op] = {'all': len(inputs), 'finite': finite_count,
                'classes': dict(sorted(classes.items())), 'precision_counts': dict(sorted(precisions.items())),
                'direct_staged_differences': differences}
            print(fmt, op, len(inputs), 'interval-confirmed references', flush=True)
        ranges.append(per_op)
        counts[fmt] = fmt_counts
    if reference_cache:
        # Independent saved preparation rows are checked after fresh evaluation.
        for fi, fmt in enumerate(F):
            width = (sum(F[fmt]) + 1) // 8
            stride = width * 5 + 4
            seen = [0, 0]
            for line in (reference_cache / (fmt + '.jsonl')).read_text().splitlines():
                row = json.loads(line)
                oi = ('exp', 'log').index(row['op'])
                at = ranges[fi][oi][0] + seen[oi] * stride
                words = [int.from_bytes(blob[at + i * width:at + (i+1) * width], 'little') for i in range(3)]
                ideal = canonical_nan(fmt) if row['ideal'] is None else int(row['ideal'], 16)
                stage = row.get('staged_ideal_f32')
                stage = ideal if stage is None else int(stage, 16)
                assert words == [int(row['input'], 16), ideal, stage], (fmt, row)
                seen[oi] += 1
            assert seen == [counts[fmt]['exp']['all'], counts[fmt]['log']['all']]
    policy_hash = hashlib.sha256(POLICY_TEXT.encode()).hexdigest()
    fixture_hash = hashlib.sha256(blob).hexdigest()
    rust = '// Generated by tools/nvidia/typed-exp-log-generate.py. Do not edit.\n'
    rust += f'pub const POLICY_SHA256: &str = "{policy_hash}";\n'
    rust += f'pub const FIXTURE_SHA256: &str = "{fixture_hash}";\n'
    rust += 'const FIXTURE_BYTES: &[u8] = include_bytes!("typed_exp_log_fixtures.bin");\n'
    rust += 'const RANGES: [[(usize, usize); 2]; 5] = ' + str(ranges) + ';\n'
    manifest = {'version': 1, 'corpus_version': 1, 'schema_version': 2,
        'policy_sha256': policy_hash, 'policy': json.loads(POLICY_TEXT),
        'policy_source_bytes': POLICY_TEXT,
        'fixture_sha256': fixture_hash, 'fixture_bytes': len(blob),
        'fixture_rust_sha256': hashlib.sha256(rust.encode()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'preparation_model_sha256': 'd249067c7972f545061aa6492ecb2d29772a04705a401cdaa19a3c9b1d77ba18',
        'primary_sources': PRIMARY_SOURCES, 'source_dependencies': SOURCE_DEPENDENCIES,
        'preparation_corpus_manifest_sha256': PREPARATION_CORPUS_SHA256, 'mpfr_version': version().decode(),
        'reference': 'adaptive directed MPFR intervals; exact rational destination RNE; no production arithmetic or observed GPU expectations',
        'binary_encoding': 'dtype/op order f64,f32,flex32,f16,bf16 / exp,log; sorted inputs; five little-endian dtype-width words input,ideal,staged,allowed_low,allowed_high then u32 intermediate_f32; quiet canonical expected NaNs classify only',
        'counts': counts, 'self_checks': checks,
        'operation_cases': sum(c[op]['all'] for c in counts.values() for op in ('exp', 'log')),
        'finite_operation_cases': sum(c[op]['finite'] for c in counts.values() for op in ('exp', 'log')),
        'qualification_status': 'predeclared project targets, not vendor guarantees; GPU unmeasured',
        'half_envelope': 'exact RNE image of neighboring ideal F32 codes; exact specials first; independent strict class/sign/nonzero gate',
        'shared_scope': 'primitive/public exp/log; ranks 0..8; empty/tails/views; exact retained sources/aliases/poison; materialized known-reference chains; launch counters and physical padding require raw harness'}
    return bytes(blob), rust.encode(), (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--reference-cache', type=Path)
    args = parser.parse_args()
    if not args.check and args.output_root is None:
        parser.error('Writes require an explicit --output-root')
    root = args.output_root or Path(__file__).resolve().parents[2]
    directory = root / 'os/library/ai/src/validation'
    targets = [directory / 'typed_exp_log_fixtures.bin', directory / 'typed_exp_log_fixtures.rs',
               root / 'tools/nvidia/typed-exp-log-manifest.json']
    for target, payload in zip(targets, generate(args.reference_cache)):
        if args.check:
            assert target.read_bytes() == payload, f'stale fixture: {target}'
        else:
            target.write_bytes(payload)
    print('typed exp/log fixtures verified' if args.check else 'typed exp/log fixtures generated')


if __name__ == '__main__':
    main()
