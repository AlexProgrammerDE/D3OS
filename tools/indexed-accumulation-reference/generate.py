"""API: generate.py writes fixtures/validation; --check recomputes and only reads."""
import sys
sys.dont_write_bytecode=True
import json,hashlib,resource,time,os
from pathlib import Path
from collections import Counter
from copy import deepcopy
from core import *
from main_cases import ordinary,public_case
from boundaries import build as build_boundaries
P=Path(__file__).parent


def serialize(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()


def verify_case(case):
    if case['family']=='public_tracked_gradient':return
    out,mapping,norm=direct(case)
    assert norm==case['normalized_index_expected']
    assert out==interpreter(case)
    if case['expected']['kind']=='error':assert out is None and norm['error']==case['expected']['semantic_error']
    else:
        assert out is not None
        assert [hx(b,case['dtype']) for b in out]==case['expected']['bits']
        assert mapping==case['logical_selection']['update_to_destination_flat']
        assert pack(out,case['dtype'])==case['expected']['packed']


def mutations(main,boundaries):
    caught=[]
    def rejects(label,bad):
        try:verify_case(bad)
        except (AssertionError,IndexError):caught.append(label)
        else:raise AssertionError('mutation survived: '+label)
    base=next(r for r in main if r['family']=='typed_arithmetic' and r['operation']=='Add' and r['dtype']=='I8')
    bad=deepcopy(base);bad['operation']='Mul';rejects('arithmetic_Add_replaced_by_Mul',bad)
    bad=deepcopy(next(r for r in main if r['family']=='typed_arithmetic' and r['view_profile']=='reversed-values' and r['operation']=='Mul'))
    t=bad['sources']['values'];t['strides_elements']=[abs(x) for x in t['strides_elements']];t['offset_elements']=0;rejects('ignored_negative_value_stride',bad)
    bad=deepcopy(next(r for r in main if r['family']=='typed_arithmetic' and r['view_profile']=='reversed-indices'))
    t=bad['sources']['indices'];t['strides_elements']=[abs(x) for x in t['strides_elements']];t['offset_elements']=0;rejects('ignored_negative_index_stride',bad)
    bad=deepcopy(next(r for r in main if r['id']=='R179'));t=bad['sources']['indices'];t['storage_bits']=[hx(int(b,16)&0xffffffff,t['dtype']) for b in t['storage_bits']];rejects('U64_index_narrowed_before_range_check',bad)
    pattern=next(x for b in boundaries for x in b['attachments'] if x.get('label')=='both_nan_base_payload' and x.get('operation')=='Min' and x.get('dtype')=='F16')
    wrong=Format(5,10).canonical;assert wrong!=int(pattern['expected_bits'],16);caught.append('MinMax_wrong_global_NaN_canonicalization')
    pattern=next(x for b in boundaries for x in b['attachments'] if x.get('label')=='every_update_store' and x.get('dtype')=='BF16' and x.get('operation')=='Mul')
    assert pattern['expected_bits']!=pattern['promoted_compute_once_final_bits'];caught.append('BF16_Mul_missing_each_update_store')
    index=next(b for b in boundaries if b['id']=='B06')['attachments'][0];assert index['wrong_narrowed_coordinate']==1 and index['expected']['error'];caught.append('wide_2pow32plus1_index_alias')
    return caught


def build():
    spec=json.loads((P/'specs.json').read_bytes());policy=json.loads((P/'policy.json').read_bytes())
    assert policy['FROZEN'] is False and policy['status']=='READY_FOR_REVIEW'
    assert hashlib.sha256((P/'ieee.py').read_bytes()).hexdigest()==policy['model_provenance']['owned_model_sha256']
    main=[public_case(r) if r['family']=='public_tracked_gradient' else ordinary(r) for r in spec['rows']]
    assert [r['id'] for r in main]==[f'R{i:03}' for i in range(1,193)]
    boundaries=build_boundaries(spec)
    for case in main:verify_case(case)
    families=dict(sorted(Counter(r['family'] for r in main).items()))
    assert families=={'accepted_assign_guard':13,'exact_index_guard':8,'legacy_ND_guard':3,'legacy_guard':16,'public_tracked_gradient':20,'typed_arithmetic':132}
    counts=dict(main_rows=192,symbolic_templates=24,families=families,by_dtype=dict(sorted(Counter(r['dtype'] for r in main).items())),by_route=dict(sorted(Counter(r['route'] for r in main).items())),by_operation=dict(sorted(Counter(r['operation'] for r in main).items())),by_index_dtype=dict(sorted(Counter(r['index_dtype'] for r in main).items())),by_view_profile=dict(sorted(Counter(r['view_profile'] for r in main).items())),by_expected_kind=dict(sorted(Counter(r['expected']['kind'] for r in main).items())),by_correctness_classification=dict(sorted(Counter(r['correctness_classification'] for r in main).items())),C3_class_patterns=sum(x.get('class_pattern',False) for b in boundaries for x in b['attachments']),boundary_attachments=sum(len(b['attachments']) for b in boundaries),boundary_attachment_kinds=dict(sorted(Counter(x['kind'] for b in boundaries for x in b['attachments']).items())),new_ids=0,blocked_rows=0)
    counts['format_route_expected_kind']={d:{r:{k:sum(c['dtype']==d and c['route']==r and c['expected']['kind']==k for c in main) for k in ('value','error','gradient')} for r in sorted({c['route'] for c in main if c['dtype']==d})} for d in counts['by_dtype']}
    validation=dict(status='REFERENCE_CHECKS_PASSED',FROZEN=False,counts=counts,detected_intentional_mutations=mutations(main,boundaries),coordinate_direct_vs_nested_rows=172,public_forward_vs_adjoint_rows=20,public_exact_central_difference_rows=20,exact_subset_checks=sum(c.get('certificate',{}).get('exact_reduction_subset_checks',0) for c in main),all_order_permutation_folds=sum(c.get('certificate',{}).get('permutation_folds',0) for c in main),dual_arithmetic_models='Fraction encoding search plus signed dyadic integer lattice; integer modulo versus byte-limb math and sign-key order.',device_rows_executed=0,scope='Reference candidate validation only. No public graph, compiler, physical ownership, residency, actual tails or Linux observation tested.')
    fixtures=dict(schema_version=1,status='READY_FOR_REVIEW',FROZEN=False,encoding_policy='policy.json#/encoding',main=main,symbolic=boundaries)
    return {'fixtures.json':serialize(fixtures),'validation.json':serialize(validation)}


def main():
    started=time.monotonic();check=sys.argv[1:]==['--check'];assert not sys.argv[1:] or check,'Only --check is accepted'
    result=build()
    if check:
        for name,b in result.items():assert (P/name).read_bytes()==b,'reproducibility mismatch: '+name
    else:
        for name,b in result.items():(P/name).write_bytes(b)
    u=resource.getrusage(resource.RUSAGE_SELF)
    print(json.dumps(dict(mode='read_only_check' if check else 'generate',status='READY_FOR_REVIEW',FROZEN=False,sha256={name:hashlib.sha256(b).hexdigest() for name,b in result.items()},resources=dict(affinity=sorted(os.sched_getaffinity(0)),cpu_seconds=u.ru_utime+u.ru_stime,wall_seconds=time.monotonic()-started,max_rss_kib=u.ru_maxrss))))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912));resource.setrlimit(resource.RLIMIT_CPU,(90,90));main()
