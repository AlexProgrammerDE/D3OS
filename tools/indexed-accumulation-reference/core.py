"""Small concrete tensor geometry and two independent reference traversals."""
from fractions import Fraction as Q
from itertools import product, permutations
from copy import deepcopy
import hashlib,json
from ieee import Format,FORMATS,step
FLOATS={**FORMATS,'F32':(8,23)}


def width(dtype):
    return 16 if dtype in ('F16','BF16') else 32 if dtype in ('Flex32','F32') else 64 if dtype=='F64' else int(dtype[1:])


def hx(n,dtype):
    return '0x'+format(n&((1<<width(dtype))-1),'0'+str(width(dtype)//4)+'x')


def encode(v,dtype):
    if dtype in FLOATS:return Format(*FLOATS[dtype]).encode(Q(v))
    return int(v)&((1<<width(dtype))-1)


def value(b,dtype):
    if isinstance(b,str):b=int(b,16)
    if dtype in FLOATS:return Format(*FLOATS[dtype]).value(b)
    n=width(dtype)
    return b-(1<<n) if dtype[0]=='I' and b>>(n-1) else b


def apply(a,b,op,dtype,secondary=False):
    if op=='Assign':return b
    if dtype in FLOATS:return step(a,b,op,Format(*FLOATS[dtype]),secondary)
    n=width(dtype);mask=(1<<n)-1
    if op in ('Min','Max'):
        if secondary:
            aa=a^(1<<(n-1)) if dtype[0]=='I' else a
            bb=b^(1<<(n-1)) if dtype[0]=='I' else b
        else:aa,bb=value(a,dtype),value(b,dtype)
        return (a if aa<=bb else b) if op=='Min' else (a if aa>=bb else b)
    if not secondary:return ((a+b) if op=='Add' else (a*b))&mask
    # Independent byte/limb schoolbook arithmetic; no wide host-float conversions.
    limbs=n//8;aa=list(a.to_bytes(limbs,'little'));bb=list(b.to_bytes(limbs,'little'));out=[0]*limbs
    if op=='Add':
        carry=0
        for i in range(limbs):total=aa[i]+bb[i]+carry;out[i]=total&255;carry=total>>8
    else:
        for i in range(limbs):
            carry=0
            for j in range(limbs-i):k=i+j;total=out[k]+aa[i]*bb[j]+carry;out[k]=total&255;carry=total>>8
    return int.from_bytes(bytes(out),'little')


def coords(shape):return product(*(range(n) for n in shape))


def size(shape):
    n=1
    for d in shape:n*=d
    return n


def strides(shape):
    out=[];n=1
    for d in reversed(shape):out.append(n);n*=d
    return list(reversed(out))


def linear(c,shape):
    result=0
    for x,d in zip(c,shape,strict=True):result=result*d+x
    return result


def address(t,c):return t['offset_elements']+sum(x*s for x,s in zip(c,t['strides_elements'],strict=True))


def read(t,c):return int(t['storage_bits'][address(t,c)],16)


def logical(t):return [read(t,c) for c in coords(t['shape'])]


def tensor(dtype,shape,vals,storage_id,mode='dense',reverse_axis=None):
    shape=list(shape);ss=strides(shape);offset=0
    if mode=='offset':offset=2
    if mode=='permuted':
        assert len(shape)>=2
        ss[0]=1;ss[1]=shape[0]
    if mode=='overlap':ss[0]=0
    if mode=='strided':ss=[s*2 for s in ss];offset=1
    if mode=='reversed':
        axis=reverse_axis if reverse_axis is not None else next((i for i in range(len(shape)-1,-1,-1) if shape[i]>1),0)
        offset+=(shape[axis]-1)*ss[axis];ss[axis]=-ss[axis]
    addresses=[offset+sum(c*s for c,s in zip(q,ss)) for q in coords(shape)]
    count=max(addresses,default=0)+1
    backing=[encode(11,dtype)]*count;assigned={}
    for at,v in zip(addresses,vals,strict=True):
        v=encode(v,dtype)
        if at in assigned:assert assigned[at]==v,'conflicting overlap materialization'
        backing[at]=v;assigned[at]=v
    return dict(dtype=dtype,shape=shape,storage_shape=[count],storage_id=storage_id,storage_bits=[hx(b,dtype) for b in backing],strides_elements=ss,offset_elements=offset,logical_to_storage=addresses)


def raw_tensor(dtype,shape,bits,storage_id,ss=None,offset=0):
    return dict(dtype=dtype,shape=shape,storage_shape=[len(bits)],storage_id=storage_id,storage_bits=[hx(b,dtype) for b in bits],strides_elements=ss if ss is not None else strides(shape),offset_elements=offset,logical_to_storage=[offset+sum(x*s for x,s in zip(c,ss if ss is not None else strides(shape))) for c in coords(shape)])


def pack(bits,dtype,tail_word=0xa5a5a5a5):
    n=width(dtype);word_count=max((len(bits)*n+31)//32,1);total=tail_word*sum(1<<(32*i) for i in range(word_count))
    for i,b in enumerate(bits):mask=((1<<n)-1)<<(i*n);total=(total&~mask)|(b<<(i*n))
    words=['0x'+format((total>>(32*i))&0xffffffff,'08x') for i in range(word_count)]
    # Independent byte assembly validates little-endian packing and all widths.
    byte_count=n//8;raw=b''.join(int(b).to_bytes(byte_count,'little') for b in bits)
    fill=tail_word.to_bytes(4,'little')*word_count;other=raw+fill[len(raw):]
    assert words==['0x'+format(int.from_bytes(other[i:i+4],'little'),'08x') for i in range(0,len(other),4)]
    unpacked=[(total>>(i*n))&((1<<n)-1) for i in range(len(bits))]
    assert unpacked==bits
    return dict(u32_words=words,tail_fill_word='0xa5a5a5a5',logical_elements=len(bits),unused_tail_bits=word_count*32-len(bits)*n)


def normalized(t):
    vals=[value(b,t['dtype']) for b in logical(t)]
    if any(x<0 or x>0x7fffffff for x in vals):return dict(error='recoverable_index_range',logical_values=[str(x) for x in vals],normalized_i32=None)
    return dict(error=None,logical_values=[str(x) for x in vals],normalized_i32=vals,normalized_shape=t['shape'],normalized_strides_elements=strides(t['shape']),normalized_offset_elements=0)


def target(c,route,axis,indices,destination_shape):
    if route=='scatter':
        out=list(c);out[axis]=int(value(read(indices,c),indices['dtype']));return tuple(out)
    if route=='select_update':
        out=list(c);out[axis]=int(value(read(indices,(c[axis],)),indices['dtype']));return tuple(out)
    k=indices['shape'][-1];row=c[0];prefix=tuple(int(value(read(indices,(row,j)),indices['dtype'])) for j in range(k))
    return prefix+tuple(c[1:])


def direct(case,secondary=False):
    src=case['sources'];d,v,idx=src['destination'],src['values'],src['indices'];norm=normalized(idx)
    if norm['error']:return None,[],norm
    out=logical(d);mapping=[]
    for c in coords(v['shape']):
        tc=target(c,case['route'],case['axis'],idx,d['shape'])
        if not all(0<=x<n for x,n in zip(tc,d['shape'],strict=True)):return None,[],dict(norm,error='recoverable_coordinate_bounds')
        at=linear(tc,d['shape']);mapping.append(at);out[at]=apply(out[at],read(v,c),case['operation'],case['dtype'],secondary)
    return out,mapping,norm


def nested(t):
    def build(dim,offset):
        if dim==len(t['shape']):return int(t['storage_bits'][offset],16)
        return [build(dim+1,offset+i*t['strides_elements'][dim]) for i in range(t['shape'][dim])]
    return build(0,t['offset_elements'])


def get(v,c):
    for x in c:v=v[x]
    return v


def interpreter(case):
    # Destination-wise nested lookup, distinct from source-wise flat direct().
    d,v,idx=(case['sources'][k] for k in ('destination','values','indices'));dn,vn,ix=nested(d),nested(v),nested(idx)
    nv=[value(b,idx['dtype']) for b in logical(idx)]
    if min(nv,default=0)<0 or max(nv,default=0)>0x7fffffff:return None
    outputs=[]
    for dc in coords(d['shape']):
        accumulator=get(dn,dc)
        for uc in coords(v['shape']):
            if case['route']=='scatter':
                dest=list(uc);dest[case['axis']]=int(value(get(ix,uc),idx['dtype']))
            elif case['route']=='select_update':
                dest=list(uc);dest[case['axis']]=int(value(ix[uc[case['axis']]],idx['dtype']))
            else:
                dest=[int(value(x,idx['dtype'])) for x in ix[uc[0]]]+list(uc[1:])
            if not all(0<=x<n for x,n in zip(dest,d['shape'],strict=True)):return None
            if tuple(dest)==dc:accumulator=apply(accumulator,get(vn,uc),case['operation'],case['dtype'],True)
        outputs.append(accumulator)
    return outputs


def certificate(case,mapping):
    dtype=case['dtype'];op=case['operation'];groups={}
    for at,b in zip(mapping,logical(case['sources']['values']),strict=True):groups.setdefault(at,[]).append(b)
    base=logical(case['sources']['destination']);checks=0;permutation_checks=0
    for at,updates in groups.items():
        results=set()
        for seq in set(permutations(updates)):
            a=base[at]
            for b in seq:a=apply(a,b,op,dtype)
            results.add(a);permutation_checks+=1
        # Assign intentionally preserves Native serial last-match meaning only.
        if op!='Assign':assert len(results)==1
        if dtype in FLOATS and op!='Assign':
            operands=[value(base[at],dtype)]+[value(b,dtype) for b in updates]
            if op in ('Add','Mul'):
                for mask in range(1<<len(operands)):
                    exact=Q(0 if op=='Add' else 1)
                    for i,x in enumerate(operands):
                        if mask>>i&1:exact=exact+x if op=='Add' else exact*x
                    bits=encode(exact,dtype);assert value(bits,dtype)==exact;checks+=1
            else:assert all(Format(*FLOATS[dtype]).kind(b) in ('finite','zero') for b in [base[at]]+updates)
    return dict(classification='native_serial_assign_only' if op=='Assign' else 'exact_small_finite_all_orders',permutation_folds=permutation_checks,exact_reduction_subset_checks=checks,linux_collision_observation='UNKNOWN')


def preservation(sources):
    groups={}
    for role,t in sources.items():groups.setdefault(t['storage_id'],[]).append(role)
    return dict(all_source_storage_unchanged={role:dict(storage_id=t['storage_id'],expected_bits=t['storage_bits']) for role,t in sources.items()},source_alias_groups=list(groups.values()),output_storage='private_dense_new_allocation',runtime_preservation_observed=False)
