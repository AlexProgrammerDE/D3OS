"""Existing B01..B24 attachments only. Descriptor algebra never allocates huge storage."""
from fractions import Fraction as Q
from itertools import product
from copy import deepcopy
from core import *
from ieee import fold,final_store,convert_a,convert_b,arithmetic_a,arithmetic_b,F32,pow2


def num(label,dtype,op,base,updates,class_pattern=False):
    f=Format(*FLOATS[dtype]) if dtype in FLOATS else None
    actual=base;independent=base;trace=[];compute=[]
    for b in updates:
        if f and op in ('Add','Mul'):
            cf=F32 if dtype in ('F16','BF16') else f
            aw,bw=convert_a(actual,f,cf),convert_a(b,f,cf)
            c=arithmetic_a(aw,bw,op,cf)
            assert aw==convert_b(actual,f,cf) and bw==convert_b(b,f,cf) and c==arithmetic_b(aw,bw,op,cf)
            compute.append(dict(dtype='F32' if dtype in ('F16','BF16','Flex32') else 'F64',operand_bits=[hx(aw,'F32' if cf.fb==23 else 'F64'),hx(bw,'F32' if cf.fb==23 else 'F64')],result_bits=hx(c,'F32' if cf.fb==23 else 'F64')))
        actual=apply(actual,b,op,dtype);independent=apply(independent,b,op,dtype,True);assert actual==independent
        trace.append(hx(actual,dtype))
    return dict(label=label,kind='numeric_reference',device_validity='candidate_native_serial_fold_only_not_executed',geometry=dict(route='scatter',destination_shape=[1],values_shape=[len(updates)],index_dtype='I32',index_bits=['0x00000000']*len(updates),update_order='ascending logical order'),dtype=dtype,operation=op,base_bits=hx(base,dtype),update_bits=[hx(b,dtype) for b in updates],expected_bits=hx(actual,dtype),stored_trace=trace,compute_trace=compute,class_pattern=class_pattern,linux_actual='UNKNOWN',scope='selected_Native_C3' if dtype in FORMATS else 'integer_declared_width_reference')


def check_index(v,length):
    return 'recoverable_index_range' if v<0 or v>0x7fffffff else 'recoverable_coordinate_bounds' if v>=length else None


def validator(f):
    p=f['parameters'];rank=p[0]&255;fmt=(p[0]>>8)&255;op=(p[0]>>16)&255
    if rank not in range(1,9) or p[0]>>24 or fmt>12 or op>3:return False
    dims=p[8:8+rank]
    if p[1]>=rank or any(x<1 or x>0x7fffffff for x in dims) or size(dims)!=p[2]:return False
    if p[8+rank:16]!=[1]*(8-rank) or any(p[16+rank:24]) or any(p[24+rank:32]):return False
    ds=dims[:];ds[p[1]]=p[7]
    if p[4] or p[16:16+rank]!=strides(ds) or size(ds)!=p[3]:return False
    if any(x<1 or x>0x7fffffff for x in (p[2],p[3],p[5],p[7])):return False
    lo=hi=p[6]
    for n,s in zip(dims,p[24:24+rank]):
        s=s-(1<<32) if s>>31 else s;delta=(n-1)*s;lo+=min(delta,0);hi+=max(delta,0)
    if not 0<=lo<=hi<p[5]:return False
    n=[64,32,32,16,16,64,32,16,8,64,32,16,8][fmt]
    needs=[(p[2]*n+31)//32,p[5],1,(p[3]*n+31)//32]
    if any(a<b for a,b in zip(f['buffer_words'],needs)):return False
    starts=f['addresses'];ends=[a+b*4 for a,b in zip(starts,needs)]
    if any(e>(1<<64)-1 for e in ends):return False
    return not any(starts[a]<ends[b] and starts[b]<ends[a] for a in (2,3) for b in range(4) if a!=b)


def build(spec):
    out=[]
    for b in spec['boundaries']:
        out.append(dict(id=b['id'],name=b['name'],design_contract=b['contract'],scope='REFERENCE_ONLY',classification='descriptor_algebra_and_small_reference_attachments',attachments=[],linux_actual='UNKNOWN',runtime_obligations_executed=False))
    by={b['id']:b for b in out}
    def add(b,x):by[b]['attachments'].append(x)
    add('B01',dict(label='rank_zero_and_scalar_nd',kind='metadata_error_and_value',axis_cases=[dict(rank=0,route=r,axis=0,expected='reject_axis') for r in ('scatter','select_update','gather')],nd_scalar=dict(source_shape=[4],source_bits=[hx(encode(x,'F16'),'F16') for x in (1,2,3,4)],index_dtype='I32',index_shape=[1],index_bits=['0x00000002'],expected_shape=[],expected_bits=['0x4200'])))
    add('B02',dict(label='rank8_valid_rank9_reject',kind='descriptor',valid_shape=[1]*7+[4],valid_strides=strides([1]*7+[4]),valid_rank=8,reject_rank=9,no_large_storage=True))
    add('B03',dict(label='I32_count_limits',kind='descriptor_algebra',valid_logical_max=0x7fffffff,reject_logical=0x80000000,physical_64_words=2*0x7fffffff,no_storage_allocated=True))
    points=[]
    for p in spec['pointer']['symbolic_points']:
        j=p['logical_index'];pair=[j*2,j*2+1];assert pair==[int(p['low_word_hex'],16),int(p['high_word_hex'],16)]
        positive=[int.from_bytes(x.to_bytes(4,'little')+bytes(4),'little')*4 for x in pair]
        assert positive==[p['low_byte_offset'],p['high_byte_offset']]
        points.append(dict(logical_N=p['logical_N'],logical_index=j,u32_word_offsets=[hx(x,'U32') for x in pair],zero_extended_u64_offsets=[hx(x,'U64') for x in pair],positive_byte_offsets=positive))
    masked=[]
    for x in spec['pointer']['masked_invalid_indices']:
        for active in (False,True):
            valid=active and 0<=x<0x7fffffff;safe=x if valid else 0;assert [2*safe,2*safe+1]==[0,1]
            masked.append(dict(raw_logical=x,active=active,safe_pair=[0,1]))
    add('B04',dict(label='wide_pointer_pairs',kind='descriptor_algebra',points=points,masked=masked,wrong_signed_extension_bytes=-8589934592,correct_high_word_bytes=8589934592,no_storage_allocated=True,actual_ir_required=spec['pointer']['ir_acceptance']))
    for dtype in ('I8','I16','I32','I64'):
        t=raw_tensor(dtype,[1],[encode(-1,dtype)],'negative_index');assert normalized(t)['error']=='recoverable_index_range'
        add('B05',dict(label=dtype+'_negative',kind='index_error',source=t,expected=normalized(t)))
    for dtype in ('I64','U64'):
        x=(1<<32)+1;t=raw_tensor(dtype,[1],[x],'wide_index');assert normalized(t)['error'] and (x&0xffffffff)==1
        add('B06',dict(label=dtype+'_no_wrap_to_one',kind='index_error',source=t,expected=normalized(t),wrong_narrowed_coordinate=1))
    for length,index,error in [(4,3,None),(4,4,'recoverable_coordinate_bounds'),(0,0,'recoverable_coordinate_bounds')]:
        assert check_index(index,length)==error;add('B07',dict(label='upper_bound',kind='index_bounds',axis_length=length,index=index,expected_error=error))
    for dtype in ('I8','U8'):
        maxvalue=127 if dtype=='I8' else 255;assert check_index(maxvalue,256) is None
        add('B08',dict(label=dtype+'_wide_bound',kind='descriptor_index',dtype=dtype,index_bits=hx(maxvalue,dtype),axis_length=256,expected_normalized=maxvalue,expected_error=None,forbidden_bound_cast=hx(256,dtype)))
    assert 1*3-1==2
    add('B09',dict(label='reject_before_linearization',kind='index_error',destination_shape=[2,3],index_dtype='I64',index_shape=[1,2],index_bits=['0x0000000000000001','0xffffffffffffffff'],expected_error='recoverable_index_range',wrong_flat_alias=2))
    add('B10',dict(label='shape_product_overflow',kind='descriptor_error',shape=[65536,65536],exact_product=1<<32,expected='reject_before_I32_linearization',no_storage_allocated=True))
    add('B11',dict(label='validated_empty_axis_update',kind='empty_reference',dtype='F64',destination_shape=[2],base_bits=['0xfff0000000000003','0x8000000000000000'],update_shape=[0],expected_bits=['0xfff0000000000003','0x8000000000000000'],all_dtype_empty_allocations=[dict(dtype=d,logical_elements=0,unused_u32_words=1,element_dispatches=0) for d in list(FLOATS)+['I8','I16','I32','I64','U8','U16','U32','U64']],validation_before_empty=['dtype_match','numeric_dtype_allowed','shape_axis_rank'],nonempty_into_empty_axis='recoverable_coordinate_bounds'))
    add('B12',dict(label='ND_public_empty_rejections',kind='metadata_error',cases=[dict(index_shape=[0,1],expected='reject_empty_indices'),dict(index_shape=[3,0],expected='reject_K0_empty_indices'),dict(index_shape=[],expected='reject_index_rank0')]))
    add('B13',dict(label='ND_zero_trailing_slice',kind='empty_reference',destination_shape=[4,0],index_dtype='I32',index_shape=[3,1],index_bits=['0x00000002','0x00000000','0x00000009'],values_shape=[3,0],expected_output_bits=[],current_Native_coordinate_reads=0,qualification='Reference of selected current empty-payload shortcut; this is not an observed backend result.'))
    spans=[]
    for shape,ss,offset,extent,valid in [([2],[-2147483648],0,4,False),([2],[2147483647],0,4,False),([3],[-1],2,3,True)]:
        lo=offset+sum(min(0,(n-1)*s) for n,s in zip(shape,ss));hi=offset+sum(max(0,(n-1)*s) for n,s in zip(shape,ss));assert (0<=lo<=hi<extent)==valid
        spans.append(dict(shape=shape,strides_elements=ss,offset=offset,storage_extent=extent,span=[lo,hi],expected_valid=valid))
    add('B14',dict(label='signed_span_algebra',kind='descriptor_algebra',spans=spans))
    for dtype in ('F16','BF16'):
        f=Format(*FLOATS[dtype]);one=f.bias<<f.fb;snan=f.sign|f.inf|3;base=[snan,one,f.sign];result=base[:];result[1]=apply(one,one,'Add',dtype)
        before=pack(base,dtype);after=pack(result,dtype)
        assert int(before['u32_words'][-1],16)>>16==int(after['u32_words'][-1],16)>>16==0xa5a5
        assert base[0]==result[0] and base[2]==result[2]
        add('B15',dict(label=dtype+'_odd_field_and_tail',kind='packed_algebra',dtype=dtype,base_bits=[hx(x,dtype) for x in base],logical_field_updated=1,expected_bits=[hx(x,dtype) for x in result],before=before,after=after,untouched_signaling_nan=hx(snan,dtype),physical_runtime_evidence=False))
    add('B15',dict(label='stale_whole_word_CAS_counterexample',kind='counterexample_algebra',initial='0x00020001',low_update_stale_replacement='0x00020002',high_update_committed='0x00030001',stale_plain_store_wrong='0x00020002',CAS_expected_mismatch=True,recomputed_replacement='0x00030002',scope='CAS feasibility reference only; selected baseline is exclusive word ownership.'))
    for dtype in ('I8','I16','I32','I64','U8','U16','U32','U64'):
        mask=(1<<width(dtype))-1
        for op in ('Add','Mul'):add('B16',num('modulo_intermediates',dtype,op,mask-1,[mask,3,(1<<(width(dtype)-1))+1]))
    for dtype,a,b in [('I64',1<<63,(1<<63)-1),('U64',(1<<63)+1,(1<<63)+2),('U64',1<<63,(1<<63)-1)]:
        for op in ('Min','Max'):add('B16',num('full_width_comparator',dtype,op,a,[b]))
    for dtype in FORMATS:
        f=Format(*FORMATS[dtype]);one=f.bias<<f.fb
        add('B17',num('all_negative_base_included',dtype,'Max',encode(-9,dtype),[encode(-8,dtype),encode(-7,dtype)]))
        for op in ('Add','Mul','Min','Max'):add('B18',num('no_match_signaling_nan',dtype,op,f.sign|f.inf|3,[]))
        # Exactly 33 selected class patterns per format, 132 total; attachments only.
        for op,sa,sb in product(('Add','Mul','Min','Max'),(0,1),(0,1)):
            add('B19',num('same_or_mixed_zero',dtype,op,sa*f.sign,[sb*f.sign],True))
        for op in ('Add','Mul','Min','Max'):
            for label,a,b in [('single_left_snan',f.sign|f.inf|3,one),('single_right_qnan',one,f.inf|f.quiet|5),('both_nan_base_payload',f.sign|f.inf|3,f.inf|f.quiet|5)]:add('B20',num(label,dtype,op,a,[b],True))
            add('B20',num('nan_then_numeric_then_nan',dtype,op,f.sign|f.inf|3,[one,f.inf|5]))
        for label,op,a,b in [('inf_cancellation','Add',f.inf,f.sign|f.inf),('zero_times_inf','Mul',0,f.sign|f.inf),('same_sign_inf_add','Add',f.sign|f.inf,f.sign|f.inf),('inf_sign_xor','Mul',f.sign|f.inf,f.sign|one),('finite_vs_inf','Min',one,f.inf)]:add('B21',num(label,dtype,op,a,[b],True))
        for label,op,a,b in [('subnormal_add','Add',1,1),('normal_seam','Add',(1<<f.fb)-1,1),('half_min_subnormal','Mul',1,(f.bias-1)<<f.fb),('odd_subnormal_half','Mul',3,(f.bias-1)<<f.fb),('negative_underflow_zero','Mul',f.sign|1,(f.bias-1)<<f.fb),('overflow','Mul',f.inf-1,(f.bias+1)<<f.fb),('nonzero_tie','Min',one,one),('nonzero_tie','Max',one,one)]:add('B21' if label!='nonzero_tie' else 'B19',num(label,dtype,op,a,[b]))
        big=(f.bias+f.fb+1)<<f.fb
        for op,base,updates in [('Add',big,[one,big|f.sign]),('Mul',one|(1<<(f.fb-1)),[one+1,one+1])]:
            x=num('every_update_store',dtype,op,base,updates);exact=f.value(base)
            for b in updates:exact=exact+f.value(b) if op=='Add' else exact*f.value(b)
            wrong=f.encode(exact);assert int(x['expected_bits'],16)!=wrong
            x.update(exact_once_final_bits=hx(wrong,dtype),promoted_compute_once_final_bits=hx(final_store(base,updates,op,f),dtype));add('B22',x)
        x=num('ascending_logical_order',dtype,'Add',0,[big,one,big|f.sign]);y=num('different_order_diagnostic',dtype,'Add',0,[big,big|f.sign,one]);assert x['expected_bits']!=y['expected_bits'];x['alternate_order_expected_bits']=y['expected_bits'];add('B22',x)
    f=Format(8,7)
    for b,expected_compute,expected_store in [(0x3780,1,0),(0x3700,0,0),(0x3f80,0x10000,1)]:
        x=num('BF16_compute_FTZ_visibility','BF16','Mul',1,[b]);assert x['compute_trace'][0]['result_bits']==hx(expected_compute,'F32') and x['expected_bits']==hx(expected_store,'BF16');x['final_value_exposes_compute_FTZ']=bool(expected_store);add('B21',x)
    x=Q(1)+pow2(-8)+pow2(-25);direct=f.encode(x);staged=convert_a(F32.encode(x),F32,f);assert direct==0x3f81 and staged==0x3f80
    add('B21',dict(label='synthetic_converter_double_rounding',kind='converter_only',exact_fraction=str(x),direct_BF16='0x3f81',F32_then_BF16='0x3f80',reachable_stored_BF16_Add_Mul_counterexample_claimed=False))
    f=Format(11,52);add('B16',num('true_F64_ulp','F64','Add',f.bias<<f.fb,[encode(pow2(-52),'F64')]))
    add('B21',num('true_F64_beyond_F32','F64','Add',encode(pow2(200),'F64'),[encode(pow2(200),'F64')]))
    f=Format(8,23);add('B21',num('Flex32_no_operand_truncation','Flex32','Mul',(f.bias<<f.fb)+1,[f.bias<<f.fb]))
    for op in ('Min','Max'):
        add('B23',dict(label='inclusive_unique_gradient_tie',kind='analytic_gradient',operation=op,dtype='F16',base_bits='0x4000',update_bits=['0x4000'],index=0,upstream_bits='0x4200',expected_base_gradient_bits='0x4200',expected_update_gradient_bits=['0x4200'],rule='Both inclusive comparisons receive full upstream; duplicate gradients excluded. Exact central differences do not define this chosen tie subgradient.'))
    assert check_index(-1,4) and check_index(2,4) is None
    add('B24',dict(label='semantic_failure_then_valid_reference',kind='error_sequence_model',first_index=-1,first_expected='recoverable_index_range',second_index=2,second_expected_normalized=2,runtime_context_recovery_observed=False,Raw_future_obligations=['independent status buffers','recoverable error then valid public graph','guarded source and output padding','early-drop queued lifetimes','resident consumers without host payload fallback','actual physical ownership and tail preservation']))
    # Keep existing validator fixture labels within their existing B templates.
    for fixture in spec['validator_fixtures']:
        assert validator(fixture)==(fixture['expected']=='accept'),fixture['id']
        b='B24' if 'overlap' in fixture['id'] or 'range' in fixture['id'] or 'overflow' in fixture['id'] else 'B16' if 'high-word' in fixture['id'] else 'B14' if 'span' in fixture['id'] or 'reversed' in fixture['id'] or 'dense-output' in fixture['id'] else 'B02'
        x=deepcopy(fixture);x['label']=x.pop('id');x['kind']='proposed_validator_model';x['model_accepts']=validator(fixture);x['production_validator_executed']=False;add(b,x)
    assert sum(x.get('class_pattern',False) for b in out for x in b['attachments'])==132
    assert [b['id'] for b in out]==[f'B{i:02}' for i in range(1,25)]
    assert all(b['attachments'] for b in out)
    return out
