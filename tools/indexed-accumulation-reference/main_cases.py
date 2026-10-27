"""Materialize fixed main-row predicates; keep explicit public graphs intact."""
from copy import deepcopy
from itertools import product
from fractions import Fraction as Q
from core import *


def labels(row):
    dtype=row['value_dtype'];family=row['family']
    if family in ('legacy_guard','legacy_ND_guard'):classification='legacy_preservation_exact_finite_domain_only'
    elif family=='accepted_assign_guard':classification='accepted_assign_rawbits_equal_duplicate_values'
    elif family=='exact_index_guard':classification='recoverable_index_error_reference'
    elif family=='public_tracked_gradient':classification='public_graph_metadata_and_exact_dense_adjoint_reference'
    else:classification='new_native_selected_extension_exact_small_finite'
    return dict(id=row['id'],dtype=dtype,index_dtype=row['index_dtype'],operation=row['operation'],route=row['route'],family=family,design_rank=row['rank'],view_profile=row['view_profile'],design_predicate=deepcopy({k:v for k,v in row.items() if k!='public_autodiff'}),scope='REFERENCE_ONLY',device_validity='checked_negative_case_expected_error' if family=='exact_index_guard' else 'schema_valid_Native_candidate_not_executed',correctness_classification=classification,linux=dict(actual_observation='UNKNOWN',public_support='source_declined_F64' if dtype=='F64' else 'not_qualified_here',duplicate_nonadd='public_ND_undefined' if row['route']=='scatter_nd' and row['operation']!='Add' else 'no_new_portable_order_claim'))


def invalid_number(s):
    if '^' not in s:return int(s)
    power,_,tail=s.partition('+');a,b=power.split('^');return int(a)**int(b)+(int(tail) if tail else 0)


def ordinary(row):
    case=labels(row);rank=row['rank'];dtype=row['value_dtype'];profile=row['view_profile'];route=row['route'];op=row['operation'];assign=op=='Assign'
    axis=0 if rank==1 else 1
    shape=[5 if assign else 4] if rank==1 else [3 if assign else 2,5 if assign else 4]+[1]*(rank-2)
    diag=route=='scatter_nd' and profile=='broadcast-indices' and rank>=2
    if diag:shape=[4,4]+[1]*(rank-2)
    unique=row.get('collision') in ('unique','unique_ND_coordinates')
    # Broadcast ND non-Add deliberately uses distinct diagonal tuples.
    targets=[2,0,1] if unique else [2,2,0]
    if row['family']=='exact_index_guard':targets[1]=invalid_number(row['invalid_index'])
    if op=='Max' and (dtype in FLOATS or dtype[0]=='I'):basevalue=-9;updates=[-8,-7,-6]
    elif op=='Min':basevalue=9;updates=[8,7,6]
    elif op=='Mul':basevalue=2;updates=[2,3,4]
    else:basevalue=2;updates=[1,2,3]
    if assign:basevalue=1;updates=[7,7,7]
    dm={'nonzero-offset':'offset','permuted-destination':'permuted','overlapping-destination-snapshot':'overlap'}.get(profile,'dense')
    basevals=[basevalue+c[axis]%2+(c[0] if rank>1 and dm!='overlap' else 0) for c in coords(shape)]
    dest=tensor(dtype,shape,basevals,'destination_original',dm)
    if route in ('scatter','select_update'):
        vs=shape[:];vs[axis]=3;vals=[updates[c[axis]]+(c[0] if rank>1 and profile=='overlapping-destination-snapshot' else 0) for c in coords(vs)]
        if route=='scatter':
            ivals=[targets[c[axis]] for c in coords(vs)]
            idx=tensor(row['index_dtype'],vs,ivals,'indices_original','reversed' if profile=='reversed-indices' else 'offset' if profile=='nonzero-offset' else 'dense',axis)
            if profile=='broadcast-indices':
                idx=raw_tensor(row['index_dtype'],vs,[encode(x,row['index_dtype']) for x in targets],'indices_original',[1 if i==axis else 0 for i in range(rank)])
        else:idx=tensor(row['index_dtype'],[3],targets,'indices_original','reversed' if profile=='reversed-indices' else 'offset' if profile=='nonzero-offset' else 'dense',0)
    else:
        k=2 if diag else rank
        tuples=[([t,t] if diag else [t if i==axis else 0 for i in range(k)]) for t in targets]
        idx=tensor(row['index_dtype'],[3,k],[x for tup in tuples for x in tup],'indices_original','reversed' if profile=='reversed-indices' else 'offset' if profile=='nonzero-offset' else 'dense',0)
        if diag:idx=raw_tensor(row['index_dtype'],[3,2],[encode(t,row['index_dtype']) for t in targets],'indices_original',[1,0])
        vs=[3]+shape[k:];vals=[updates[c[0]] for c in coords(vs)]
    vm={'reversed-values':'reversed','strided-values':'strided','nonzero-offset':'offset'}.get(profile,'dense')
    values=tensor(dtype,vs,vals,'values_original',vm,axis if route!='scatter_nd' else 0)
    if assign:
        if dtype in FLOATS:
            f=Format(*FLOATS[dtype]);replacement=f.sign|f.inf|3;basebits=f.sign
        elif dtype in ('I64','U64'):replacement=(1<<63)+1;basebits=(1<<64)-3
        else:replacement=encode(7,dtype);basebits=encode(1,dtype)
        for t,b in [(dest,basebits),(values,replacement)]:
            for at in set(t['logical_to_storage']):t['storage_bits'][at]=hx(b,dtype)
    sources=dict(destination=dest,values=values,indices=idx)
    if profile=='overlapping-destination-snapshot':
        alias=deepcopy(dest);alias['shape']=[len(dest['storage_bits'])];alias['strides_elements']=[1];alias['offset_elements']=0;alias['logical_to_storage']=list(range(len(dest['storage_bits'])));sources['destination_alias_observer']=alias
    case.update(axis=axis if route!='scatter_nd' else None,sources=sources,preservation=preservation(sources),logical_selection=dict(destination_shape=shape,update_shape=vs,index_tuple_width=idx['shape'][-1] if route=='scatter_nd' else None))
    case['logical_selection']['destination_base_bits']=[hx(b,dtype) for b in logical(dest)]
    case['logical_selection']['update_value_bits']=[hx(b,dtype) for b in logical(values)]
    out,mapping,norm=direct(case);other=interpreter(case);assert out==other,row['id']
    case['normalized_index_expected']=norm
    if route=='select_update':
        case['resident_normalized_broadcast_geometry']=dict(shape=vs,strides_elements=[1 if i==axis else 0 for i in range(rank)],offset_elements=0,logical_i32=[targets[c[axis]] for c in coords(vs)] if not norm['error'] else None,source='normalized logical index vector, not an added tensor download')
    if out is None:
        assert row['family']=='exact_index_guard';case['expected']=dict(kind='error',semantic_error=norm['error'],destination_not_published=True,source_preservation_required=True)
    else:
        assert out==direct(case,True)[0]
        case['logical_selection']['update_to_destination_flat']=mapping
        case['logical_selection']['destination_base_bits']=[hx(b,dtype) for b in logical(dest)]
        case['logical_selection']['update_value_bits']=[hx(b,dtype) for b in logical(values)]
        case['expected']=dict(kind='value',shape=shape,strides_elements=strides(shape),offset_elements=0,storage_id='private_output',bits=[hx(b,dtype) for b in out],packed=pack(out,dtype),no_match_flat=[i for i in range(len(out)) if i not in mapping])
        case['certificate']=certificate(case,mapping)
        if unique:assert len(mapping)==len(set(mapping))
        for at in case['expected']['no_match_flat']:assert out[at]==logical(dest)[at]
        if row['family']=='typed_arithmetic':
            assert all(Format(*FLOATS[dtype]).kind(b) in ('finite','zero') for b in out+logical(dest)+logical(values)) if dtype in FLOATS else True
    assert all(size(t['shape'])<=64 for t in sources.values())
    return case


def from_metadata(t,raw=None,storage_id=None):
    vals=t.get('storage_values_flat',raw);assert vals is not None
    return raw_tensor(t['dtype'],t['shape'],[encode(x,t['dtype']) for x in vals],storage_id or t.get('backing_storage','independent'),t['strides_elements'],t['offset_elements'])


def public_forward(row,raw_values=None):
    m=row['public_autodiff'];a=m['operand'];ix=m['indices'];r=row['route'];raw=m['original_leaf']['storage_values_flat'] if raw_values is None else raw_values
    data=from_metadata(a,raw,'original_tracked_leaf');indices=from_metadata(ix,storage_id='indices_original');mapping=[];out=[]
    for c in coords(m['output']['shape']):
        if r=='embedding_backward':sc=(int(value(read(indices,c[:2]),ix['dtype'])),c[2])
        elif r in ('gather_backward','select_backward'):
            sc=list(c);sc[m['axis']]=int(value(read(indices,c if r=='gather_backward' else (c[m['axis']],)),ix['dtype']))
        else:
            batch=len(ix['shape'])-1;k=ix['shape'][-1];sc=tuple(int(value(read(indices,c[:batch]+(j,)),ix['dtype'])) for j in range(k))+c[batch:]
        at=address(data,sc);mapping.append(at);out.append(raw[at])
    return out,mapping


def independent_public(row,raw):
    m=row['public_autodiff'];a=m['operand'];idx=m['indices'];route=row['route']
    def materialize(shape,ss,offset,values):
        if not shape:return values[offset]
        return [materialize(shape[1:],ss[1:],offset+i*ss[0],values) for i in range(shape[0])]
    data=materialize(a['shape'],a['strides_elements'],a['offset_elements'],raw)
    indices=materialize(idx['shape'],idx['strides_elements'],idx['offset_elements'],idx['storage_values_flat'])
    out=[]
    for c in coords(m['output']['shape']):
        if route=='embedding_backward':v=data[indices[c[0]][c[1]]][c[2]]
        elif route=='select_backward':sc=list(c);sc[m['axis']]=indices[c[m['axis']]];v=get(data,sc)
        elif route=='gather_backward':sc=list(c);sc[m['axis']]=get(indices,c);v=get(data,sc)
        else:batch=len(idx['shape'])-1;sc=get(indices,c[:batch])+list(c[batch:]);v=get(data,sc)
        out.append(v)
    return out


def public_case(row):
    case=labels(row);m=row['public_autodiff'];dtype=row['value_dtype'];raw=m['original_leaf']['storage_values_flat'];out,mapping=public_forward(row)
    assert out==independent_public(row,raw)==m['output']['storage_values_flat']
    up=m['logical_upstream_at_output'];grad=[0]*len(raw)
    for at,g in zip(mapping,up,strict=True):grad[at]+=g
    destinationwise=[sum(g for at,g in zip(mapping,up,strict=True) if at==i) for i in range(len(raw))]
    assert grad==destinationwise==m['expected_gradient']['storage_values_flat']
    coefficient=m['upstream_coefficient']['storage_values_flat'];branches=m['graph']['branches'];root=branches*sum(x*y for x,y in zip(out,coefficient,strict=True))
    assert [root]==m['graph']['root']['expected_values_flat']
    def loss(vals):return branches*sum(x*y for x,y in zip(independent_public(row,vals),coefficient,strict=True))
    central=[]
    for i in range(len(raw)):
        plus=raw[:];minus=raw[:];plus[i]+=1;minus[i]-=1;central.append(Q(loss(plus)-loss(minus),2))
    assert central==grad
    # Every reduction subset, plus every destination's gradient subset, stays exact.
    checks=0
    for terms in [[x*y for x,y in zip(out,coefficient,strict=True)]]+[[g for at,g in zip(mapping,up,strict=True) if at==i] for i in range(len(raw))]:
        for mask in range(1<<len(terms)):
            total=sum(x for i,x in enumerate(terms) if mask>>i&1);assert value(encode(total,dtype),dtype)==total;checks+=1
    assert value(encode(root,dtype),dtype)==root
    sources={role:from_metadata(m[role],raw if role=='operand' else None,'original_tracked_leaf' if role in ('original_leaf','operand') else role+'_original') for role in ('original_leaf','operand','indices','upstream_coefficient')}
    # Keep the exact reviewed public graph role metadata alongside concrete bits.
    case.update(public_metadata=deepcopy(m),sources=sources,preservation=preservation(sources),normalized_index_expected=normalized(sources['indices']),logical_selection=dict(output_to_original_leaf_flat=mapping,operand_to_original_leaf_flat=m['operand'].get('view_to_original_leaf_flat')),expected=dict(kind='gradient',forward=raw_tensor(dtype,m['output']['shape'],[encode(x,dtype) for x in out],'private_forward'),dense_original_leaf_gradient=raw_tensor(dtype,m['original_leaf']['shape'],[encode(x,dtype) for x in grad],'private_leaf_gradient'),root=dict(dtype=dtype,shape=[1],bits=[hx(encode(root,dtype),dtype)]),seed=dict(dtype=dtype,shape=[1],bits=[hx(encode(1,dtype),dtype)]),upstream_bits=[hx(encode(x,dtype),dtype) for x in up]),certificate=dict(source_scatter_equals_destination_adjoint=True,exact_central_difference=[str(x) for x in central],exact_reduction_subset_checks=checks,root_exact=True,public_backend_executed=False))
    return case
