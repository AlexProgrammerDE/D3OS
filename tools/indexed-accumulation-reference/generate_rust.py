"""Emit immutable Shared fixtures from the owned, byte-pinned reference closure."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
from collections import Counter
from itertools import product
import generate

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent.parent / 'os/library/ai/src/validation/indexed_accumulation/fixtures.rs'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def string(value):
    # JSON's ASCII string spelling is also valid Rust for these ASCII inputs.
    assert value.isascii()
    return json.dumps(value)


def array(values):
    return '&[' + ','.join(str(v) for v in values) + ']'


def bits(values):
    return '&[' + ','.join(v + 'u64' for v in values) + ']'


def dtype(value):
    return 'DType::' + value


def dense(shape):
    result = []
    stride = 1
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return list(reversed(result))


def count(shape):
    value = 1
    for size in shape:
        value *= size
    return value


def recipe(source):
    shape, strides, offset = source['shape'], source['strides_elements'], source['offset_elements']
    axes = sorted(range(len(shape)), key=lambda a: -abs(strides[a]))
    base_shape = [shape[a] if strides[a] else 1 for a in axes]
    steps = [abs(strides[a]) for a in axes if shape[a] > 1 and strides[a]]
    unit = min(steps, default=1)
    for a, expected in zip(axes, dense(base_shape), strict=True):
        if shape[a] > 1 and strides[a]:
            assert abs(strides[a]) == expected * unit, source
    low = offset + sum(min(0, (n - 1) * s) for n, s in zip(shape, strides, strict=True))
    elements = count(base_shape)
    end = low + max(0, elements - 1) * unit + int(elements > 0)
    assert 0 <= low <= end <= len(source['storage_bits'])
    # Independently simulate the public recipe's affine map, including broadcasts.
    inverse = [axes.index(a) for a in range(len(shape))]
    base_strides = dense(base_shape)
    actual = []
    for coordinate in product(*(range(n) for n in shape)):
        physical = low
        for a, x in enumerate(coordinate):
            x = 0 if strides[a] == 0 else shape[a] - 1 - x if strides[a] < 0 else x
            physical += x * base_strides[inverse[a]] * unit
        actual.append(physical)
    assert actual == source['logical_to_storage']
    operations = ['ViewStep::Reshape(' + array([len(source['storage_bits'])]) + ')']
    operations.append(f'ViewStep::Slice {{ start:{low}, end:{end}, step:{unit} }}')
    operations.append('ViewStep::Reshape(' + array(base_shape) + ')')
    operations.append('ViewStep::Permute(' + array(inverse) + ')')
    operations.append('ViewStep::Flip(' + array([a for a, s in enumerate(strides) if s < 0]) + ')')
    operations.append('ViewStep::Expand(' + array(shape) + ')')
    return '&[' + ','.join(operations) + ']'


def source(role, value):
    return 'SourceSpec{' + ','.join([
        'role:' + string(role), 'storage_id:' + string(value['storage_id']),
        'dtype:' + dtype(value['dtype']), 'storage_shape:' + array(value['storage_shape']),
        'storage_bits:' + bits(value['storage_bits']), 'shape:' + array(value['shape']),
        'strides_elements:' + array(value['strides_elements']),
        'offset_elements:' + str(value['offset_elements']),
        'logical_to_storage:' + array(value['logical_to_storage']), 'construction:' + recipe(value),
    ]) + '}'


def values(value, dt, raw_key='bits'):
    return 'ValueSpec{dtype:' + dtype(value.get('dtype', dt)) + ',shape:' + array(value['shape']) + ',bits:' + bits(value[raw_key]) + '}'


def node(value):
    if value is None:
        return 'Node::Null'
    if isinstance(value, bool):
        return 'Node::Bool(' + str(value).lower() + ')'
    if isinstance(value, int):
        return 'Node::Integer(' + str(value) + 'i128)'
    if isinstance(value, str):
        return 'Node::String(' + string(value) + ')'
    if isinstance(value, list):
        return 'Node::Array(&[' + ','.join(node(v) for v in value) + '])'
    assert isinstance(value, dict)
    return 'Node::Object(&[' + ','.join('(' + string(k) + ',' + node(v) + ')' for k, v in sorted(value.items())) + '])'


def policy(row):
    family = row['family']
    if family in ('legacy_guard', 'legacy_ND_guard'):
        return 'LegacyExactFiniteGuard'
    if family == 'accepted_assign_guard':
        return 'AcceptedAssignBitwiseGuard'
    if family == 'public_tracked_gradient':
        return 'PublicLinearAdjointExact'
    if family == 'exact_index_guard':
        return 'NativeRecoverableSemanticError'
    return 'NativeSerialC3' if row['dtype'] in ('F16', 'BF16', 'Flex32', 'F64') else 'DeclaredWidthInteger'


def case(row):
    expected = row['expected']
    dt = row['dtype']
    if expected['kind'] == 'value':
        result = 'Expected::Value(' + values(expected, dt) + ')'
        graph = 'None'
    elif expected['kind'] == 'error':
        result = 'Expected::Error(SemanticError::' + {'recoverable_index_range': 'IndexRange', 'recoverable_coordinate_bounds': 'CoordinateBounds'}[expected['semantic_error']] + ')'
        graph = 'None'
    else:
        m = row['public_metadata']
        result = 'Expected::Gradient(&GradientSpec{forward:' + values(expected['forward'], dt, 'storage_bits') + ',leaf_gradient:' + values(expected['dense_original_leaf_gradient'], dt, 'storage_bits') + ',root:' + values(expected['root'], dt) + ',seed:' + values(expected['seed'], dt) + ',upstream:' + bits(expected['upstream_bits']) + '})'
        output = m['output']['storage_values_flat']
        coefficient = m['upstream_coefficient']['storage_values_flat']
        # Reference generation only. All stored stage products/sums are exact small integers.
        encoded = [generate.hx(generate.encode(a * b, dt), dt) for a, b in zip(output, coefficient, strict=True)]
        branch_sum = generate.hx(generate.encode(sum(a * b for a, b in zip(output, coefficient, strict=True)), dt), dt)
        graph = 'Some(PublicGraph{branches:' + str(m['graph']['branches']) + ',product:ValueSpec{dtype:' + dtype(dt) + ',shape:' + array(m['output']['shape']) + ',bits:' + bits(encoded) + '},branch_sum:ValueSpec{dtype:' + dtype(dt) + ',shape:&[1],bits:' + bits([branch_sum]) + '}})'
    normalized = row['normalized_index_expected'].get('normalized_i32')
    norm = 'None' if normalized is None else 'Some(' + array(normalized) + ')'
    return 'Case{' + ','.join([
        'id:' + string(row['id']), 'dtype:' + dtype(dt), 'index_dtype:' + dtype(row['index_dtype']),
        'operation:Operation::' + row['operation'], 'route:Route::' + ''.join(w.title() for w in row['route'].split('_')),
        'family:' + string(row['family']), 'design_rank:' + str(row['design_rank']), 'view_profile:' + string(row['view_profile']),
        'axis:' + ('None' if row.get('axis', row.get('public_metadata', {}).get('axis')) is None else 'Some(' + str(row.get('axis', row.get('public_metadata', {}).get('axis'))) + ')'),
        'sources:&[' + ','.join(source(role, value) for role, value in sorted(row['sources'].items())) + ']',
        'normalized_indices:' + norm, 'expected:' + result, 'public_graph:' + graph,
        'policy:Policy::' + policy(row), 'frozen_json:' + string(canonical(row)), 'row_sha256:' + string(digest(canonical(row).encode())),
    ]) + '}'


def render():
    pins = json.loads((HERE / 'provenance.json').read_bytes())
    for name, expected in pins['baseline_sha256'].items():
        assert '/' not in name and digest((HERE / name).read_bytes()) == expected, name
    built = generate.build()
    for name, content in built.items():
        assert (HERE / name).read_bytes() == content, name
    fixture = json.loads(built['fixtures.json'])
    assert [r['id'] for r in fixture['main']] == [f'R{i:03}' for i in range(1, 193)]
    assert Counter(r['expected']['kind'] for r in fixture['main']) == {'value': 164, 'error': 8, 'gradient': 20}
    assert [b['id'] for b in fixture['symbolic']] == [f'B{i:02}' for i in range(1, 25)]
    pairs = [(b['id'], digest(canonical(a).encode())) for b in fixture['symbolic'] for a in b['attachments']]
    assert len(pairs) == len(set(pairs)) == 283
    assert sum(a.get('class_pattern', False) for b in fixture['symbolic'] for a in b['attachments']) == 132
    parts = ['// Generated by tools/indexed-accumulation-reference/generate_rust.py.\n// Historical baseline flags remain unchanged. Production source binding is not approved.\nuse super::*;\n']
    parts.append('#[rustfmt::skip]\npub(super) const POLICY_JSON: &str = ' + string(canonical(json.loads((HERE / 'policy.json').read_bytes()))) + ';\n')
    parts.append('#[rustfmt::skip]\npub(super) const FIXTURES_SHA256: &str = ' + string(digest(built['fixtures.json'])) + ';\n')
    parts.append('#[rustfmt::skip]\npub(super) static CASES: &[Case] = &[\n' + ',\n'.join(case(r) for r in fixture['main']) + '\n];\n')
    boundaries = []
    for boundary in fixture['symbolic']:
        attachments = []
        for attachment in boundary['attachments']:
            payload = canonical(attachment)
            attachments.append('ModelAttachment{key:AttachmentKey{template:' + string(boundary['id']) + ',content_sha256:' + string(digest(payload.encode())) + '},label:' + string(attachment['label']) + ',kind:' + string(attachment['kind']) + ',dtype:' + ('Some(' + dtype(attachment['dtype']) + ')' if 'dtype' in attachment else 'None') + ',class_pattern:' + str(attachment.get('class_pattern', False)).lower() + ',frozen_json:' + string(payload) + ',data:' + node(attachment) + '}')
        boundaries.append('SymbolicTemplate{id:' + string(boundary['id']) + ',name:' + string(boundary['name']) + ',attachments:&[' + ','.join(attachments) + ']}')
    parts.append('#[rustfmt::skip]\npub(super) static SYMBOLIC: &[SymbolicTemplate] = &[\n' + ',\n'.join(boundaries) + '\n];\n')
    return ''.join(parts).encode('ascii')


def main():
    assert sys.argv[1:] in ([], ['--check']), 'Only --check is accepted'
    output = render()
    if sys.argv[1:]:
        assert OUTPUT.read_bytes() == output, 'generated Rust differs'
    else:
        OUTPUT.write_bytes(output)
    print(json.dumps({'mode': 'check' if sys.argv[1:] else 'generate', 'FROZEN': False, 'source_binding_approved': False, 'rust_sha256': digest(output), 'bytes': len(output), 'main': 192, 'symbolic_templates': 24, 'attachments': 283, 'device_attempts': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
