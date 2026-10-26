"""Validate frozen forward-Erf evidence on the host, without executing a backend.

Public plans use schema 3 and five dtype-major Erf cells. Raw plans use the
separately audited raw inventory. Neither expected plan contains observed
maxima. A terminal record is evidence of runtime checks, not a substitute for
an execution provenance or binary/source certificate.
"""

import copy
import hashlib
import itertools
import json
import math
import re
from collections import Counter, deque
from pathlib import Path
from typing import NamedTuple

DTYPES = ("f64", "f32", "flex32", "f16", "bf16")
WIDTHS = (8, 4, 4, 2, 2)
FRACTIONS = (52, 23, 23, 10, 7)
EXPONENTS = (0x7FF0000000000000, 0x7F800000, 0x7F800000, 0x7C00, 0x7F80)
ONES = (0x3FF0000000000000, 0x3F800000, 0x3F800000, 0x3C00, 0x3F80)
ROW_COUNTS = (48362, 25592, 25592, 65536, 65536)
POLICY_SHA256 = "5d67c74ea3d9404264eacaac819693bad43e2fa0a953b56bfdbdd0bd310af11c"
FIXTURE_SHA256 = "661fec5a11dabfd9969f356dbbd38d8025b3704793f39d42e49f91eecab5a742"
MANIFEST_SHA256 = "37250692a9045d6035bc63158a8d354e2e60c7210364d2c457fccec30db15074"
RAW_CORE_SHA256 = "a4f0d57189712269c49e9e1362fd256130f38cc3fb3d92ba3a4144a076bf5a2f"
LENGTHS = (0, 1, 2, 3, 127, 128, 129, 255, 256, 257)
CLASSES = ("zero", "subnormal", "normal", "infinity", "nan")
RAW_CORE_KEYS = ("counts", "row_counts", "cell_values", "comparisons", "staged_classes", "ideal_classes", "oddness_checks", "full_finite_pairs", "route_counts", "policy_sha256", "fixture_sha256")
ROUTES = ("raw", "fresh", "Native.float_unary", "Native.unary", "Native.wrapper", "Dispatch", "Tensor", "queued.Native", "queued.raw", "queued.public")


class Reference(NamedTuple):
    input: int
    ideal: int
    staged: int
    low: int
    high: int
    intermediate_f32: int


class FeasibleImage(NamedTuple):
    pairs: frozenset
    staged: frozenset
    ideal: frozenset
    states: frozenset


class ReferencePlan(NamedTuple):
    """Opaque expected context. Frozen rows resolve it during validation."""
    kind: str
    expected: dict


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise RuntimeError("invalid Erf JSON value") from error


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate Erf JSON field")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("nonfinite Erf JSON number")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite Erf JSON exponent")
    return result


DECODER = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_nonfinite, parse_float=_float)


def strict_json(text):
    try:
        return DECODER.decode(text)
    except (ValueError, TypeError, RecursionError) as error:
        raise RuntimeError("invalid strict Erf JSON") from error


def _decode(text):
    try:
        return DECODER.raw_decode(text)
    except (ValueError, TypeError, RecursionError) as error:
        raise RuntimeError("invalid embedded Erf JSON") from error


def vector(value, length):
    return isinstance(value, list) and len(value) == length and all(type(n) is int and n >= 0 for n in value)


def matrix(value, rows, columns):
    return isinstance(value, list) and len(value) == rows and all(vector(row, columns) for row in value)


def sign(fi):
    return 1 << (8 * WIDTHS[fi] - 1)


def classify(fi, bits):
    magnitude = bits & (sign(fi) - 1)
    if not magnitude:
        return 0
    if magnitude < EXPONENTS[fi] & -EXPONENTS[fi]:
        return 1
    if magnitude < EXPONENTS[fi]:
        return 2
    return 3 if magnitude == EXPONENTS[fi] else 4


def rank(fi, bits):
    s = sign(fi)
    return (~bits & (2 * s - 1)) if bits & s else bits | s


def unrank(fi, value):
    s = sign(fi)
    return value & (s - 1) if value & s else ~value & (2 * s - 1)


def distance(fi, actual, expected):
    return abs(rank(fi, actual) - rank(fi, expected)) if classify(fi, actual) in (1, 2) and classify(fi, expected) in (1, 2) else None


def bucket(value):
    return next((i for i, upper in enumerate((0, 1, 2, 4, 8, 16)) if value <= upper), 6)


def convert(bits, source, destination):
    """Integer-only IEEE RNE conversion, including gradual underflow."""
    sf, df = FRACTIONS[source], FRACTIONS[destination]
    source_bias = (EXPONENTS[source] >> sf) // 2
    destination_bias = (EXPONENTS[destination] >> df) // 2
    s = sign(destination) if bits & sign(source) else 0
    field = (bits & EXPONENTS[source]) >> sf
    fraction = bits & ((1 << sf) - 1)
    if classify(source, bits) >= 3:
        return s | EXPONENTS[destination] | ((1 << (df - 1)) if fraction else 0)
    n = fraction if not field else (1 << sf) | fraction
    if not n:
        return s
    exponent = (1 - source_bias if not field else field - source_bias) - sf
    k = n.bit_length() - 1 + exponent
    step = max(k - df, 1 - destination_bias - df)
    shift = step - exponent
    if shift > 0:
        q, remainder = divmod(n, 1 << shift)
        halfway = 1 << (shift - 1)
        n = q + int(remainder > halfway or (remainder == halfway and q & 1))
    else:
        n <<= -shift
    if n >= 1 << (df + 1):
        n >>= 1
        step += 1
    if step + df > destination_bias:
        return s | EXPONENTS[destination]
    if n < 1 << df:
        return s | n
    return s | ((step + df + destination_bias) << df) | (n - (1 << df))


def passing(fi, row, actual):
    if type(actual) is not int or not 0 <= actual < 2 * sign(fi):
        return False
    wanted, observed = classify(fi, row.staged), classify(fi, actual)
    if wanted == 4:
        return observed == 4
    if observed >= 3 or actual & (sign(fi) - 1) > ONES[fi]:
        return False
    if wanted == 0 or classify(fi, row.input) in (0, 3, 4):
        return actual == row.staged
    if observed not in (1, 2) or (actual ^ row.staged) & sign(fi):
        return False
    return distance(fi, actual, row.staged) <= 2 if fi < 3 else rank(fi, row.low) <= rank(fi, actual) <= rank(fi, row.high)


def successors(fi, row):
    if classify(fi, row.input) in (0, 3, 4):
        return (row.staged,)
    if fi >= 3:
        return tuple(unrank(fi, r) for r in range(rank(fi, row.low), rank(fi, row.high) + 1))
    center = rank(fi, row.staged)
    return tuple(v for r in range(center - 2, center + 3) if passing(fi, row, v := unrank(fi, r)))


def _row_image(fi, row):
    # Preserve correlations: distance pair, both reference classes, output class.
    states = frozenset((distance(fi, actual, row.staged), distance(fi, actual, row.ideal),
                        classify(fi, row.staged), classify(fi, row.ideal), classify(fi, actual))
                       for actual in successors(fi, row))
    return FeasibleImage(frozenset((a, b) for a, b, sc, ic, ac in states),
                         frozenset((sc, ac) for a, b, sc, ic, ac in states),
                         frozenset((ic, ac) for a, b, sc, ic, ac in states), states)


def _capacity_counters(groups):
    result = {"image_groups": Counter(), "distance_pairs": Counter(), "transition_caps": {"staged": Counter(), "ideal": Counter()}}
    for image, count in groups.items():
        require(type(count) is int and count >= 0, "Erf image multiplicity is invalid")
        if not count:
            continue
        result["image_groups"][image] = count
        finite = frozenset((a, b) for a, b in image.pairs if a is not None and b is not None)
        if finite:
            require(len(finite) == len(image.pairs), "Erf image mixes finite-pair eligibility")
            result["distance_pairs"][finite] += count
        for name in ("staged", "ideal"):
            for edge in getattr(image, name):
                result["transition_caps"][name][edge] += count
    return result


def _image_capacities(limit, context):
    """One slot per static observation or chain stage, never per branch."""
    groups = Counter()
    for source, count in context["row_weights"].items():
        groups[limit["feasible_rows"][source]] += count
    for sources in context["chain_slots"]:
        images = [limit["feasible_rows"][source] for source in sources]
        image = FeasibleImage(*(frozenset().union(*(getattr(item, name) for item in images))
                                for name in ("pairs", "staged", "ideal", "states")))
        groups[image] += 1
    context.update(_capacity_counters(groups))


def _maximum_flow(edges, source, sink):
    """Small exact integer flow; reverse edges permit reassignment."""
    residual = {}
    for a, b, count in edges:
        require(type(count) is int and count >= 0, "Erf flow capacity is invalid")
        residual.setdefault(a, {})[b] = residual.get(a, {}).get(b, 0) + count
        residual.setdefault(b, {}).setdefault(a, 0)
    total = 0
    while True:
        parent = {source: None}
        pending = deque([source])
        while pending and sink not in parent:
            node = pending.popleft()
            for target, count in residual[node].items():
                if count and target not in parent:
                    parent[target] = node
                    pending.append(target)
        if sink not in parent:
            return total
        node, amount = sink, None
        while parent[node] is not None:
            previous = parent[node]
            amount = residual[previous][node] if amount is None else min(amount, residual[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= amount
            residual[node][previous] += amount
            node = previous
        total += amount


def _validate_transition_distance_allocation(fi, cell, context):
    for matrix_key, hist_key, metric in (("transitions", "histogram", 0), ("ideal_transitions", "ideal_histogram", 1)):
        crossing_groups = Counter()
        for image, count in context["image_groups"].items():
            classes = {state[2 + metric] for state in image.states}
            require(len(classes) == 1, "Erf image has mixed reference classes")
            source_class = next(iter(classes))
            same = {bucket(state[metric]) for state in image.states if state[metric] is not None and state[4] == source_class}
            crossing = {(source_class, state[4], bucket(state[metric])) for state in image.states
                        if source_class in (1, 2) and state[4] in (1, 2) and state[4] != source_class}
            if fi < 3 and source_class in (1, 2):
                # For this frozen wide contract, all ordinary images permit
                # every same-class distance 0/1/2. Exact specials permit only 0.
                # Thus cross-flow plus residual histogram capacity is sound:
                # unspent ordinary slots can fill any remaining same-class bin.
                require(same == {0, 1, 2} or (same == {0} and not crossing), "Erf wide same-class image proof differs")
            directions = {(r, c) for r, c, d in crossing}
            require(len(directions) <= 1, "Erf image crossing direction is ambiguous")
            if crossing:
                direction = next(iter(directions))
                eligible = frozenset(d for r, c, d in crossing)
                require(0 not in eligible, "Erf zero-distance image changes class")
                crossing_groups[direction, eligible] += count
        source, sink = ("source",), ("sink",)
        m, histogram = cell[matrix_key], cell[hist_key]
        demands = {(1, 2): m[1][2], (2, 1): m[2][1]}
        edges = [(source, ("transition", r, c), demand) for (r, c), demand in demands.items()]
        for index, ((direction, eligible), count) in enumerate(crossing_groups.items()):
            node = ("image", index)
            edges.append((("transition", *direction), node, count))
            edges.extend((node, ("distance", d), count) for d in eligible)
        # Both directions share these supplies; no bin can be spent twice.
        edges.extend((("distance", d), sink, n) for d, n in enumerate(histogram))
        require(_maximum_flow(edges, source, sink) == sum(demands.values()), "Erf transitions cannot share the per-image distance-bin supply")


def _reserve_witness_events(fi, cell, context, limit, witnesses):
    """Reserve each distinct (input, actual) exactly once before feasibility.

    Every eligible frozen source has static full-corpus clones. Reserving a
    static clone is sound even if a witness originally came from a chain:
    exchange its output with that clone in the relaxed chain-image model.
    The identical source admits both outputs and consumes the same marginals.
    """
    remaining = copy.deepcopy(cell)
    budget = context["row_weights"].copy()
    groups = context["image_groups"].copy()
    for source, actual in sorted(witnesses):
        row = limit["rows"][source]
        image = limit["feasible_rows"][source]
        state = (distance(fi, actual, row.staged), distance(fi, actual, row.ideal),
                 classify(fi, row.staged), classify(fi, row.ideal), classify(fi, actual))
        require(state in image.states and budget[source] > 0 and groups[image] > 0, "Erf witness has no remaining immutable source clone")
        budget[source] -= 1
        groups[image] -= 1
        remaining["values"] -= 1
        remaining["observed_classes"][state[4]] -= 1
        for mkey, hkey, pair_key, reference in (("transitions", "histogram", "finite_pairs", row.staged), ("ideal_transitions", "ideal_histogram", "ideal_finite_pairs", row.ideal)):
            r, c = classify(fi, reference), classify(fi, actual)
            require(remaining[mkey][r][c] > 0, "Erf witnesses overdraw a class population")
            remaining[mkey][r][c] -= 1
            d = distance(fi, actual, reference)
            if d is not None:
                require(remaining[pair_key] > 0 and remaining[hkey][bucket(d)] > 0, "Erf witnesses overdraw a metric population")
                remaining[pair_key] -= 1
                remaining[hkey][bucket(d)] -= 1
        if state[0] is not None:
            remaining["passing_finite_pairs"] -= 1
            remaining["passing_histogram"][bucket(state[0])] -= 1
        if state[2] == state[4] == 1:
            remaining["subnormal_preserved"] -= 1
    return remaining, _capacity_counters(groups)


def _validate_image_capacities(fi, cell, context):
    for matrix_key, hist_key, name in (("transitions", "histogram", "staged"), ("ideal_transitions", "ideal_histogram", "ideal")):
        m = cell[matrix_key]
        require(m[1][2] + m[2][1] <= sum(cell[hist_key][1:]), "Erf class changes exceed this metric's nonzero-distance population")
        require(all(count <= context["transition_caps"][name][r, c] for r, row in enumerate(m) for c, count in enumerate(row)), "Erf class transitions exceed immutable image capacity")
    groups = context["distance_pairs"]
    population = cell["finite_pairs"]
    require(sum(groups.values()) == population == cell["ideal_finite_pairs"], "Erf image-slot finite population differs")
    if fi < 3:
        # Identical wide references admit diagonal distance pairs only. Hall's
        # subset inequalities are the exact capacitated assignment criterion.
        require(all(a == b and a in (0, 1, 2) for choices in groups for a, b in choices), "Erf wide image distance proof differs")
        for mask in range(1, 8):
            selected = {i for i in range(3) if mask & (1 << i)}
            demand = sum(cell["histogram"][i] for i in selected)
            capacity = sum(n for choices, n in groups.items() if any(a in selected for a, _ in choices))
            require(demand <= capacity, "Erf wide histogram exceeds immutable image capacity")
        return
    # The frozen half images have distances 0 or 1 for both references.
    # A 2x2 joint table with these marginals has one integer variable x=p11:
    # p00=N-S-I+x, p01=I-x, p10=S-x, p11=x. For every subset of pair
    # categories, Hall bounds give a linear inequality in x. Intersecting
    # all integer bounds proves that the two histograms can share the slots.
    categories = ((0, 0), (0, 1), (1, 0), (1, 1))
    require(all(pair in categories for choices in groups for pair in choices), "Erf half image distance proof differs")
    staged_one, ideal_one = cell["histogram"][1], cell["ideal_histogram"][1]
    low, high = max(0, staged_one + ideal_one - population), min(staged_one, ideal_one)
    intercepts = (population - staged_one - ideal_one, ideal_one, staged_one, 0)
    slopes = (1, -1, -1, 1)
    for mask in range(1, 16):
        selected = {pair for i, pair in enumerate(categories) if mask & (1 << i)}
        capacity = sum(n for choices, n in groups.items() if choices & selected)
        intercept = sum(intercepts[i] for i in range(4) if mask & (1 << i))
        slope = sum(slopes[i] for i in range(4) if mask & (1 << i))
        if slope > 0:
            high = min(high, (capacity - intercept) // slope)
        elif slope < 0:
            low = max(low, -((capacity - intercept) // -slope))
        else:
            require(intercept <= capacity, "Erf joint half histogram exceeds immutable image capacity")
    require(low <= high, "Erf ideal/staged histograms have no joint immutable-image assignment")


def _load_bounds(manifest_path):
    """Verify the frozen identity and derive bounds from five words plus u32."""
    path = Path(manifest_path)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA256, "Erf manifest is not the frozen manifest")
    manifest = strict_json(raw.decode())
    require(manifest.get("schema_version") == 3 and manifest.get("corpus_version") == 1, "Erf manifest schema differs")
    require(manifest.get("policy_sha256") == POLICY_SHA256 and manifest.get("fixture_sha256") == FIXTURE_SHA256, "Erf numerical identity differs")
    policy_text = manifest.get("policy_source_bytes")
    require(isinstance(policy_text, str) and hashlib.sha256(policy_text.encode()).hexdigest() == POLICY_SHA256, "Erf policy bytes differ")
    require(canonical(strict_json(policy_text)) == canonical(manifest.get("policy")), "Erf policy content differs")
    root = path.parents[2]
    index = root / "os/library/ai/src/validation/typed_erf_fixtures.rs"
    generator = root / "tools/nvidia/typed-erf-generate.py"
    require(hashlib.sha256(index.read_bytes()).hexdigest() == manifest["fixture_rust_sha256"], "Erf fixture index differs")
    require(hashlib.sha256(generator.read_bytes()).hexdigest() == manifest["generator_sha256"], "Erf generator differs")
    data = (root / "os/library/ai/src/validation/typed_erf_fixtures.bin").read_bytes()
    expected_size = sum(n * (5 * width + 4) for n, width in zip(ROW_COUNTS, WIDTHS))
    require(type(manifest.get("fixture_bytes")) is int and len(data) == manifest["fixture_bytes"] == expected_size, "Erf fixture byte size differs")
    require(hashlib.sha256(data).hexdigest() == FIXTURE_SHA256, "Erf fixture hash differs")
    bounds, offset = [], 0
    for fi, dtype in enumerate(DTYPES):
        count = manifest["counts"][dtype]["all"]
        require(type(count) is int and count == ROW_COUNTS[fi], "Erf row inventory differs")
        width = WIDTHS[fi]
        limit = {"rows": {}, "staged_classes": [0] * 5, "ideal_classes": [0] * 5,
                     "staged": 2 if fi < 3 else 0, "ideal": 2 if fi < 3 else 0,
                     "class_changes": False, "finite_changes": False,
                     "chain_plans": copy.deepcopy(manifest["chain_plans"][dtype])}
        previous = -1
        for _ in range(count):
            words = [int.from_bytes(data[offset + k * width:offset + (k + 1) * width], "little") for k in range(5)]
            intermediate = int.from_bytes(data[offset + 5 * width:offset + 5 * width + 4], "little")
            offset += 5 * width + 4
            row = Reference(*words, intermediate)
            require(row.input > previous, "Erf input order or uniqueness differs")
            previous = row.input
            sc, ic = classify(fi, row.staged), classify(fi, row.ideal)
            source_class = classify(fi, row.input)
            limit["rows"][row.input] = row
            limit["staged_classes"][sc] += 1
            limit["ideal_classes"][ic] += 1
            limit["class_changes"] |= sc != ic
            limit["finite_changes"] |= (sc in (1, 2)) != (ic in (1, 2))
            for result in (row.ideal, row.staged):
                if source_class == 4:
                    require(classify(fi, result) == 4, "Erf NaN reference class differs")
                else:
                    require(classify(fi, result) <= 2 and result & (sign(fi) - 1) <= ONES[fi] and (result ^ row.input) & sign(fi) == 0, "Erf reference sign, class or codomain differs")
                    if source_class == 0:
                        require(result == row.input, "Erf zero reference differs")
                    elif source_class == 3:
                        require(result == ONES[fi] | (row.input & sign(fi)), "Erf infinity reference differs")
                    else:
                        require(classify(fi, result) in (1, 2), "Erf nonzero reference flushed")
            if fi < 3:
                require(row.ideal == row.staged and intermediate == 0 and row.low == row.high == row.staged, "Erf wide reference schedule differs")
            else:
                require(convert(intermediate, 1, fi) == row.staged, "Erf staged half conversion differs")
                if source_class in (0, 3, 4):
                    require(row.low == row.high == row.staged, "Erf special image differs")
                else:
                    magnitude, s = intermediate & 0x7FFFFFFF, intermediate & 0x80000000
                    image = sorted({convert(s | b, 1, fi) for b in range(max(1, magnitude - 2), min(ONES[1], magnitude + 2) + 1)}, key=lambda b: rank(fi, b))
                    require(image and image[0] == row.low and image[-1] == row.high and all(rank(fi, b) == rank(fi, a) + 1 for a, b in itertools.pairwise(image)), "Erf half neighbor image differs")
                    for actual in image:
                        require(passing(fi, row, actual), "Erf image violates semantic gates")
                        limit["staged"] = max(limit["staged"], distance(fi, actual, row.staged))
                        limit["ideal"] = max(limit["ideal"], distance(fi, actual, row.ideal))
        require(limit["staged_classes"] == [manifest["counts"][dtype]["staged_classes"].get(c, 0) for c in CLASSES], "Erf staged class census differs")
        require(limit["ideal_classes"] == [manifest["counts"][dtype]["ideal_classes"].get(c, 0) for c in CLASSES], "Erf ideal class census differs")
        image_pool = {}
        limit["feasible_rows"] = {}
        for source, row in limit["rows"].items():
            image = _row_image(fi, row)
            limit["feasible_rows"][source] = image_pool.setdefault(image, image)
        bounds.append(limit)
    require(offset == len(data), "Erf fixture has trailing bytes")
    # Rebuild each closure from admitted stored outputs, never ideal-only chains.
    for fi, limit in enumerate(bounds):
        require(len(limit["chain_plans"]) == 14, "Erf chain count differs")
        for plan in limit["chain_plans"]:
            level, finite, populations, deltas = [plan["seed"]], True, [], []
            levels = [level]
            for _ in range(3):
                require(all(b in limit["rows"] for b in level), "Erf actual-input closure is incomplete")
                finite &= all(classify(fi, b) <= 2 for b in level)
                classes = {(classify(fi, limit["rows"][b].staged), classify(fi, limit["rows"][b].ideal)) for b in level}
                require(len(classes) == 1, "Erf chain classes need a new population proof")
                populations.append(next(iter(classes)))
                values = {distance(fi, limit["rows"][b].ideal, limit["rows"][b].staged) for b in level}
                require(len(values) == 1, "Erf chain value-reference coupling needs a new proof")
                deltas.append(next(iter(values)))
                level = sorted({v for b in level for v in successors(fi, limit["rows"][b])})
                levels.append(level)
            require(plan["levels"] == levels and type(plan["finite"]) is bool and plan["finite"] == finite, "Erf frozen closure differs")
            plan["proved_classes"] = populations
            plan["proved_deltas"] = deltas
    return manifest, bounds

def load_bounds(manifest_path):
    try:
        return _load_bounds(manifest_path)
    except (OSError, UnicodeError, IndexError, KeyError, TypeError) as error:
        raise RuntimeError("Erf frozen reference input cannot be loaded") from error


SUMMARY_KEYS = {
    "schema_version", "policy_sha256", "fixture_sha256", "options", "dtypes", "operations", "cases",
    "numerical_cases", "layout_cases", "public_cases", "chain_cases", "chain_steps", "tensor_rows",
    "selected_dtype_operation_counts", "expected_value_counts", "values", "expected_staged_class_counts",
    "expected_ideal_class_counts", "chain_value_counts", "expected_oddness_checks", "source_checks",
    "alias_checks", "poison_checks", "metadata_checks", "empty_cases",
}


def validate_plan(expected):
    require(isinstance(expected, dict) and set(expected) == SUMMARY_KEYS, "Erf expected summary keys differ or contain observations")
    require(type(expected["schema_version"]) is int and expected["schema_version"] == 3, "Erf summary schema differs")
    require(expected["policy_sha256"] == POLICY_SHA256 and expected["fixture_sha256"] == FIXTURE_SHA256, "Erf summary numerical identity differs")
    options = expected["options"]
    require(isinstance(options, dict) and set(options) == {"no_flex32", "skip_rank_zero", "finite", "dtype", "op"}, "Erf selection keys differ")
    require(all(type(options[k]) is bool for k in ("no_flex32", "skip_rank_zero", "finite")) and options["dtype"] in (*DTYPES, "all") and options["op"] in ("all", "erf"), "Erf selection differs")
    integers = SUMMARY_KEYS - {"policy_sha256", "fixture_sha256", "options", "selected_dtype_operation_counts", "expected_value_counts", "expected_staged_class_counts", "expected_ideal_class_counts", "chain_value_counts", "expected_oddness_checks"}
    require(all(type(expected[k]) is int and expected[k] >= 0 for k in integers), "Erf summary counters differ")
    for key in ("selected_dtype_operation_counts", "expected_value_counts", "chain_value_counts", "expected_oddness_checks"):
        require(matrix(expected[key], 5, 1), "Erf per-cell work plan differs")
    for key in ("expected_staged_class_counts", "expected_ideal_class_counts"):
        require(isinstance(expected[key], list) and len(expected[key]) == 5 and all(matrix(row, 1, 5) for row in expected[key]), "Erf class plan dimensions differ")
        for fi, row in enumerate(expected[key]):
            require(sum(row[0]) + expected["chain_value_counts"][fi][0] == expected["expected_value_counts"][fi][0], "Erf mandatory class plan differs")
    require(sum(row[0] for row in expected["expected_value_counts"]) == expected["values"], "Erf total work differs")
    require(sum(row[0] for row in expected["chain_value_counts"]) == expected["chain_steps"] == 3 * expected["chain_cases"], "Erf materialized chain count differs")
    return expected["expected_value_counts"]


def summary_reference_plan(expected):
    """Return expected context; validation narrows chains using frozen closure."""
    validate_plan(expected)
    return ReferencePlan("shared", copy.deepcopy(expected))


def raw_reference_plan(expected):
    """Bind the separate, independently enumerated raw plan without shared slack."""
    require(isinstance(expected, dict) and all(k in expected for k in RAW_CORE_KEYS), "Erf raw plan is incomplete")
    core = {k: expected[k] for k in RAW_CORE_KEYS}
    require(hashlib.sha256(canonical(core).encode()).hexdigest() == RAW_CORE_SHA256, "Erf raw plan differs from its independent frozen inventory")
    require(expected.get("gpu_execution", False) is False and expected.get("observed_maxima") is None, "Erf raw expected plan contains observed results")
    return ReferencePlan("raw", copy.deepcopy(expected))


def _selected(fi, options):
    return not (options["no_flex32"] and fi == 2) and options["dtype"] in ("all", DTYPES[fi])


def _shared_context(expected, bounds):
    validate_plan(expected)
    options = expected["options"]
    independent = {k: 0 for k in SUMMARY_KEYS - {"policy_sha256", "fixture_sha256", "options", "selected_dtype_operation_counts", "expected_value_counts", "expected_staged_class_counts", "expected_ideal_class_counts", "chain_value_counts", "expected_oddness_checks"}}
    independent.update(schema_version=3, policy_sha256=POLICY_SHA256, fixture_sha256=FIXTURE_SHA256, options=options)
    for key in ("selected_dtype_operation_counts", "expected_value_counts", "chain_value_counts", "expected_oddness_checks"):
        independent[key] = [[0] for _ in DTYPES]
    for key in ("expected_staged_class_counts", "expected_ideal_class_counts"):
        independent[key] = [[[0] * 5] for _ in DTYPES]
    result = []
    for fi, limit in enumerate(bounds):
        context = {"staged": [0] * 5, "ideal": [0] * 5, "selected_inputs": set(), "reference_deltas": Counter(), "row_weights": Counter(), "chain_slots": []}
        result.append(context)
        if not _selected(fi, options):
            continue
        rows = [r for r in limit["rows"].values() if not options["finite"] or (classify(fi, r.input) <= 2 and classify(fi, r.ideal) <= 2)]
        require(rows, "Erf selected cell has no rows")
        inputs = [r.input for r in rows]
        context["selected_inputs"].update(inputs)
        value = lambda i, inputs=inputs: inputs[(i * 4051 + i // 7) % len(inputs)]
        layouts = []
        for length in LENGTHS:
            layouts.extend([
                [value(i) for i in range(length)],
                [value(i + 1) for i in range(length)],
                [value(i) for i in reversed(range(length))],
                [value(1 + 2 * i) for i in reversed(range(length))],
            ])
        if not options["skip_rank_zero"]:
            layouts.append([value(0)])
        for _ in range(2, 9):
            layouts.extend([[value(i) for i in range(21)], []])
        layouts.extend([[value(i) for i in range(7) for _ in range(4)], [], [value(i) for i in range(7) for _ in range(4)], [], [value(i) for i in range(12) for _ in range(2)]])
        cases = [inputs, *layouts]
        independent["dtypes"] += 1
        independent["operations"] += 1
        independent["tensor_rows"] += len(rows)
        independent["selected_dtype_operation_counts"][fi][0] = len(rows)
        independent["cases"] += 2 * len(cases)
        independent["numerical_cases"] += 2
        independent["layout_cases"] += 2 * len(layouts)
        independent["public_cases"] += len(cases)
        independent["empty_cases"] += 2 * sum(not case for case in cases)
        independent["source_checks"] += 2 * len(cases)
        independent["alias_checks"] += 2 * len(cases)
        independent["poison_checks"] += 2 * len(cases)
        independent["metadata_checks"] += 8 * len(cases)
        for case in cases:
            independent["values"] += 2 * len(case)
            independent["expected_value_counts"][fi][0] += 2 * len(case)
            unique = set(case)
            pairs = sum(b & sign(fi) == 0 and classify(fi, b) <= 2 and b | sign(fi) in unique for b in unique)
            independent["expected_oddness_checks"][fi][0] += 2 * pairs
            for b, repeats in Counter(case).items():
                row = limit["rows"][b]
                context["row_weights"][b] += 2 * repeats
                delta = distance(fi, row.ideal, row.staged)
                if delta is not None:
                    context["reference_deltas"][delta] += 2 * repeats
                independent["expected_staged_class_counts"][fi][0][classify(fi, row.staged)] += 2 * repeats
                independent["expected_ideal_class_counts"][fi][0][classify(fi, row.ideal)] += 2 * repeats
        context["staged"] = independent["expected_staged_class_counts"][fi][0].copy()
        context["ideal"] = independent["expected_ideal_class_counts"][fi][0].copy()
        for plan in limit["chain_plans"]:
            if options["finite"] and not plan["finite"]:
                continue
            independent["cases"] += 1
            independent["chain_cases"] += 1
            independent["chain_steps"] += 3
            independent["values"] += 3
            independent["expected_value_counts"][fi][0] += 3
            independent["chain_value_counts"][fi][0] += 3
            independent["source_checks"] += 1
            independent["alias_checks"] += 4
            independent["poison_checks"] += 1
            independent["metadata_checks"] += 9
            for sc, ic in plan["proved_classes"]:
                context["staged"][sc] += 1
                context["ideal"][ic] += 1
            for delta in plan["proved_deltas"]:
                if delta is not None:
                    context["reference_deltas"][delta] += 1
            for level in plan["levels"][:3]:
                context["selected_inputs"].update(level)
                context["chain_slots"].append(tuple(level))
        _image_capacities(limit, context)
    # Excluded cells still have explicit empty capacities.
    for limit, context in zip(bounds, result):
        if "distance_pairs" not in context:
            _image_capacities(limit, context)
    require(canonical(independent) == canonical(expected), "Erf expected summary differs from independent frozen geometry and selection")
    return result


def _resolve(reference_plan, bounds, expected_cells, total, oddness_expected):
    require(isinstance(reference_plan, ReferencePlan), "Erf reference context must come from a plan helper")
    if reference_plan.kind == "shared":
        expected = reference_plan.expected
        require(validate_plan(expected) == expected_cells and expected["values"] == total, "Erf shared cell plan differs")
        require(canonical(expected["expected_oddness_checks"]) == canonical(oddness_expected), "Erf expected oddness plan differs")
        return _shared_context(expected, bounds)
    require(reference_plan.kind == "raw", "Erf reference context kind differs")
    raw = raw_reference_plan(reference_plan.expected).expected
    require(canonical(raw["cell_values"]) == canonical(expected_cells) and raw["comparisons"] == total and canonical(raw["oddness_checks"]) == canonical(oddness_expected), "Erf raw observation plan differs")
    return _raw_context(raw, bounds)

def _raw_context(expected, bounds):
    """Independent bit-only raw geometry, including exact reference deltas."""
    result = []
    for fi, limit in enumerate(bounds):
        rows = list(limit["rows"].values())
        weights = Counter({row.input: 4 for row in rows})
        pool = []
        for cls in (0, 4, 3, 1, 2):
            members = [row.input for row in rows if classify(fi, row.input) == cls]
            pool.extend(members[:3])
            pool.extend(reversed(members[-3:]))
        pool.extend(row.input for row in rows[::max(len(rows) // 31, 1)])
        def values(n, pool=pool):
            return [pool[i % len(pool)] for i in range(n)]
        geometry = []
        for n in LENGTHS[1:]:
            storage = values(2 * n + 11)
            geometry.extend((values(n), [storage[1 + 2 * i] for i in range(n)], [storage[2 * n - 1 - 2 * i] for i in range(n)]))
        geometry.append(values(11)[1:6])
        storage = values(21)
        geometry.append([storage[r * 7 + c] for c in range(6, -1, -1) for r in range(2, -1, -1)])
        storage = values(45)
        geometry.append([storage[i] for i in (10, 12, 14, 28, 30, 32)])
        geometry.extend((values(1), values(24)))
        geometry.extend(values(3) for _ in range(3, 8))
        geometry.append([values(3)[1]] * 6)
        geometry.extend((values(7) * 4, [b for b in values(7) for _ in range(4)]))
        require(len(geometry) == 40 and sum(map(len, geometry)) == 3608, "Erf raw geometry enumeration differs")
        def pairs(items, fi=fi):
            unique = set(items)
            return sum(b < sign(fi) and classify(fi, b) <= 2 and b | sign(fi) in unique for b in unique)
        oddness = 4 * pairs(limit["rows"])
        for case in geometry:
            for b, n in Counter(case).items():
                weights[b] += 9 * n
            oddness += 9 * pairs(case)
        queued = [row.input for row in rows[::23][:17]]
        for b in queued:
            weights[b] += 4
        oddness += 4 * pairs(queued)
        context = {"staged": [0] * 5, "ideal": [0] * 5, "selected_inputs": set(weights), "reference_deltas": Counter(), "row_weights": weights, "chain_slots": []}
        for b, n in weights.items():
            row = limit["rows"][b]
            context["staged"][classify(fi, row.staged)] += n
            context["ideal"][classify(fi, row.ideal)] += n
            delta = distance(fi, row.ideal, row.staged)
            if delta is not None:
                context["reference_deltas"][delta] += n
        require(context["staged"] == expected["staged_classes"][fi][0] and context["ideal"] == expected["ideal_classes"][fi][0] and sum(weights.values()) == expected["cell_values"][fi][0] and oddness == expected["oddness_checks"][fi][0], "Erf raw independent populations or exact oddness differ")
        _image_capacities(limit, context)
        result.append(context)
    return result


FAILURES = (
    "failures", "classification_failures", "numerical_failures", "exact_special_failures",
    "half_envelope_failures", "codomain_failures", "unreferenced_results", "sign_mismatches",
    "nonzero_to_zero", "oddness_failures",
)
COUNTERS = (*FAILURES, "values", "oddness_checks", "passing_finite_pairs", "finite_pairs", "ideal_finite_pairs", "subnormal_preserved", "nan_payload_differences", "quiet_nan_outputs", "negative_nan_outputs", "reference_class_disagreements", "reference_nonzero_finite_disagreements")
ARRAYS = ("observed_classes", "unreferenced_observed_classes", "histogram", "ideal_histogram", "passing_histogram", "transitions", "ideal_transitions")
MAXIMA = ("max_ulp", "max_ideal_ulp", "passing_max_ulp")
WITNESSES = ("worst_input_bits", "worst_actual_bits", "worst_ideal_input_bits", "worst_ideal_actual_bits")
CELL_KEYS = {"dtype", "op", *COUNTERS, *ARRAYS, *MAXIMA, *WITNESSES}


def validate_accuracy(accuracy, expected_cells, total, bounds, *, reference_plan, oddness_expected):
    """Accept only a complete passing observation bound to its expected context."""
    require(matrix(expected_cells, 5, 1) and type(total) is int and total >= 0 and sum(row[0] for row in expected_cells) == total, "Erf observed work plan differs")
    require(matrix(oddness_expected, 5, 1) and isinstance(bounds, list) and len(bounds) == 5, "Erf oddness or bounds plan differs")
    contexts = _resolve(reference_plan, bounds, expected_cells, total, oddness_expected)
    require(isinstance(accuracy, dict) and set(accuracy) == {"failures", "cells"} and type(accuracy["failures"]) is int and accuracy["failures"] == 0, "Erf reports aggregate qualification failures")
    cells = accuracy["cells"]
    require(isinstance(cells, list) and len(cells) == 5, "Erf requires exactly five accuracy cells")
    for fi, (cell, limit, context) in enumerate(zip(cells, bounds, contexts)):
        require(isinstance(cell, dict) and set(cell) == CELL_KEYS and cell["dtype"] == DTYPES[fi] and cell["op"] == "erf", "Erf accuracy cell identity or keys differ")
        n = cell["values"]
        require(type(n) is int and n == expected_cells[fi][0] and all(type(cell[k]) is int and 0 <= cell[k] <= n for k in COUNTERS), "Erf per-cell counters differ")
        require(not any(cell[k] for k in FAILURES), "Erf cell reports numerical, semantic or oddness failures")
        require(cell["oddness_checks"] == oddness_expected[fi][0] and cell["oddness_checks"] <= n // 2, "Erf observed oddness population differs")
        require(vector(cell["observed_classes"], 5) and sum(cell["observed_classes"]) == n, "Erf observed class population differs")
        require(vector(cell["unreferenced_observed_classes"], 5) and not any(cell["unreferenced_observed_classes"]), "Erf includes unreferenced results")
        for key, population, pair_key in (("transitions", context["staged"], "finite_pairs"), ("ideal_transitions", context["ideal"], "ideal_finite_pairs")):
            m = cell[key]
            require(matrix(m, 5, 5) and [sum(row[c] for row in m) for c in range(5)] == cell["observed_classes"], "Erf class matrix columns differ")
            require([sum(row) for row in m] == population and sum(population) == n, "Erf frozen reference class populations differ")
            require(sum(m[r][c] for r in (1, 2) for c in (1, 2)) == cell[pair_key], "Erf finite-pair population differs")
        staged, ideal = cell["transitions"], cell["ideal_transitions"]
        permitted = ({0}, {1, 2}, {1, 2}, {3}, {4})
        require(not any(count for row, allowed in zip(staged, permitted) for c, count in enumerate(row) if c not in allowed), "Erf class transitions contradict a pass")
        require(cell["observed_classes"][3] == 0, "Erf infinity output violates codomain")
        changes, finite_changes = cell["reference_class_disagreements"], cell["reference_nonzero_finite_disagreements"]
        require(finite_changes <= changes and sum(abs(a - b) for x, y in zip(staged, ideal) for a, b in zip(x, y)) <= 2 * changes and abs(cell["finite_pairs"] - cell["ideal_finite_pairs"]) <= finite_changes, "Erf dual-reference accounting differs")
        require((limit["class_changes"] or (changes == 0 and staged == ideal)) and (limit["finite_changes"] or finite_changes == 0), "Erf reference class disagreements have no frozen witness")
        require(cell["subnormal_preserved"] == staged[1][1] and cell["passing_finite_pairs"] == cell["finite_pairs"], "Erf passing population differs")
        require(all(cell[k] <= cell["observed_classes"][4] for k in ("nan_payload_differences", "quiet_nan_outputs", "negative_nan_outputs")), "Erf NaN output diagnostics differ")
        for hist_key, count_key, max_key, cap in (("histogram", "finite_pairs", "max_ulp", limit["staged"]), ("passing_histogram", "passing_finite_pairs", "passing_max_ulp", limit["staged"]), ("ideal_histogram", "ideal_finite_pairs", "max_ideal_ulp", limit["ideal"])):
            histogram, count, maximum = cell[hist_key], cell[count_key], cell[max_key]
            require(vector(histogram, 7) and sum(histogram) == count, "Erf histogram count differs")
            if not count:
                require(maximum is None, "Erf empty histogram has a maximum")
            else:
                require(type(maximum) is int and 0 <= maximum <= cap and histogram[bucket(maximum)] > 0 and not any(histogram[bucket(maximum) + 1:]), "Erf maximum contradicts its target or histogram")
        require(cell["histogram"] == cell["passing_histogram"] and cell["max_ulp"] == cell["passing_max_ulp"], "Erf passing metrics contradict observations")
        if fi < 3:
            require(cell["ideal_histogram"] == cell["histogram"] and cell["max_ideal_ulp"] == cell["max_ulp"] and cell["worst_input_bits"] == cell["worst_ideal_input_bits"] and cell["worst_actual_bits"] == cell["worst_ideal_actual_bits"], "Erf identical wide references have different metrics or witnesses")
        witnesses = set()
        for prefix, pair_key, max_key in (("worst", "finite_pairs", "max_ulp"), ("worst_ideal", "ideal_finite_pairs", "max_ideal_ulp")):
            source_text, actual_text = cell[prefix + "_input_bits"], cell[prefix + "_actual_bits"]
            if not cell[pair_key]:
                require(source_text is None and actual_text is None, "Erf empty metric has a witness")
                continue
            require(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{16}", v) and int(v, 16) < 2 * sign(fi) for v in (source_text, actual_text)), "Erf witness encoding differs")
            source, actual = int(source_text, 16), int(actual_text, 16)
            row = limit["rows"].get(source)
            require(row is not None and source in context["selected_inputs"], "Erf witness is outside the selected immutable input domain")
            require(passing(fi, row, actual), "Erf witness fails its special, sign, nonzero, codomain or numerical gate")
            sc, ic, ac = classify(fi, row.staged), classify(fi, row.ideal), classify(fi, actual)
            require(staged[sc][ac] > 0 and ideal[ic][ac] > 0, "Erf witness has no population in both class matrices")
            for reference, count_key, maximum_key, histogram_key in ((row.staged, "finite_pairs", "max_ulp", "histogram"), (row.ideal, "ideal_finite_pairs", "max_ideal_ulp", "ideal_histogram")):
                steps = distance(fi, actual, reference)
                if steps is not None:
                    require(cell[count_key] > 0 and cell[maximum_key] is not None and steps <= cell[maximum_key] and cell[histogram_key][bucket(steps)] > 0, "Erf witness contradicts the other metric maximum or histogram")
            reference = row.ideal if prefix == "worst_ideal" else row.staged
            require(distance(fi, actual, reference) == cell[max_key], "Erf witness does not demonstrate its own maximum")
            witnesses.add((source, actual))
        remaining, capacities = _reserve_witness_events(fi, cell, context, limit, witnesses)
        _validate_image_capacities(fi, remaining, capacities)
        _validate_transition_distance_allocation(fi, remaining, capacities)
    return accuracy


def option_scope(expected):
    validate_plan(expected)
    options = expected["options"]
    return " ".join(f"{k}={str(options[k]).lower()}" for k in ("no_flex32", "skip_rank_zero", "finite", "dtype", "op"))


def _records(text, marker, execution_os):
    require(isinstance(text, str), "Erf serial evidence is not text")
    result = {"OK": [], "ACCURACY": []}
    for original in text.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", original)
        body = re.sub(r"^(?:\[[^\]\r\n]*\]\s*)+", "", line) if execution_os == "d3os" else line
        if body.startswith(("BURN_TYPED_ERF_FAILURE", "NVIDIA_BURN_TYPED_ERF_FAILURE")):
            raise RuntimeError("Erf serial contains a failure record")
        if not body.startswith(marker):
            require(re.search(r"(?<![A-Za-z0-9_])" + re.escape(marker), line) is None, "Erf marker is unanchored or has an unsupported log prefix")
            continue
        require(len(body) < 4_000_000, "Erf terminal record is too large")
        kind = body[len(marker):].split(" ", 1)[0]
        require(kind in result, "Erf terminal marker differs")
        result[kind].append(body)
    require(len(result["OK"]) == 1 and len(result["ACCURACY"]) <= 1, "Erf needs one unambiguous terminal marker")
    return result


def _manifest_identity(manifest):
    try:
        data = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise RuntimeError("Erf loaded manifest cannot be serialized") from error
    require(hashlib.sha256(data).hexdigest() == MANIFEST_SHA256, "Erf loaded manifest identity differs")


def validate(serial, expected, manifest, bounds, *, execution_os, backend, fusion, scope=None):
    """Validate one shared accuracy record and its matching success footer."""
    require((execution_os, backend, fusion) in (("linux", "flex_cpu", "disabled"), ("linux", "cuda", "enabled_materialized_chain_steps"), ("d3os", "native", "disabled")), "Erf platform or fusion identity differs")
    _manifest_identity(manifest)
    cells = validate_plan(expected)
    selected_scope = option_scope(expected)
    require(scope is None or scope == selected_scope, "Erf caller scope contradicts selection")
    records = _records(serial, "BURN_TYPED_ERF_", execution_os)
    require(len(records["ACCURACY"]) == 1, "Erf public evidence needs one accuracy record")
    common = f"os={execution_os} backend={backend} fusion={fusion} corpus=1"
    prefix = f"BURN_TYPED_ERF_OK {common} totals="
    line = records["OK"][0]
    require(line.startswith(prefix), "Erf terminal platform identity differs")
    totals, end = _decode(line[len(prefix):])
    remaining = line[len(prefix) + end:]
    require(canonical(totals) == canonical(expected) and remaining.startswith(" accuracy="), "Erf terminal totals differ from expected work")
    accuracy, end = _decode(remaining[len(" accuracy="):])
    suffix = remaining[len(" accuracy=") + end:]
    footer = f" policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} numerical_policy=project_qualification_targets staged_half=f32_neighbor_envelope no_ftz=strict preserved_inputs=exact_bytes chain=materialized_known_refs poison=retained_allocation scope=forward_erf_only {selected_scope}"
    require(suffix == footer, "Erf policy, fixtures, source preservation or exact scope footer differs")
    if records["ACCURACY"]:
        line = records["ACCURACY"][0]
        prefix = f"BURN_TYPED_ERF_ACCURACY {common} policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} qualification_passed=true totals="
        require(line.startswith(prefix), "Erf accuracy line reports failure or a different platform")
        other_totals, end = _decode(line[len(prefix):])
        remaining = line[len(prefix) + end:]
        require(canonical(other_totals) == canonical(expected) and remaining.startswith(" accuracy="), "Erf accuracy line totals differ")
        other_accuracy, end = _decode(remaining[len(" accuracy="):])
        require(remaining[len(" accuracy=") + end:] == "" and canonical(other_accuracy) == canonical(accuracy), "Erf accuracy and terminal records disagree")
    return validate_accuracy(accuracy, cells, expected["values"], bounds, reference_plan=summary_reference_plan(expected), oddness_expected=expected["expected_oddness_checks"])


RAW_COUNT_KEYS = ("corpus_rows", "corpus_batches", "geometry", "raw", "fresh", "native", "public", "rejected", "empty", "queued", "f32_routes", "typed_routes")


def _raw_counts(expected):
    return "Counts { " + ", ".join(f"{k}: {expected['counts'][k]}" for k in RAW_COUNT_KEYS) + " }"


def validate_raw(serial, expected, manifest, bounds, *, execution_os="d3os", backend="nvidia", fusion="disabled", scope="forward_erf_only", protocol=50, profiles=82):
    """Validate raw inventory/route records using the independent raw plan."""
    require((execution_os, backend, fusion, scope) == ("d3os", "nvidia", "disabled", "forward_erf_only"), "Erf raw platform or scope differs")
    require(type(protocol) is int and type(profiles) is int and (protocol, profiles) in ((50, 82), (51, 83), (52, 84), (53, 85), (54, 88)), "Erf raw unqualified protocol/profile pair")
    _manifest_identity(manifest)
    plan = raw_reference_plan(expected)
    records = _records(serial, "NVIDIA_BURN_TYPED_ERF_", execution_os)
    require(len(records["ACCURACY"]) == 1, "Erf raw needs one accuracy record")
    counts = _raw_counts(expected)
    routes = json.dumps(list(ROUTES))
    route_counts = json.dumps(expected["route_counts"])
    prefix = f"NVIDIA_BURN_TYPED_ERF_ACCURACY policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} frozen=true qualification_passed=true coherent=true comparisons={expected['comparisons']} counts={counts} route_names={routes} route_counts={route_counts} accuracy="
    line = records["ACCURACY"][0]
    require(line.startswith(prefix), "Erf raw accuracy identity, failures, counts or routes differ")
    accuracy, end = _decode(line[len(prefix):])
    require(line[len(prefix) + end:] == "", "Erf raw accuracy contains trailing data")
    c = expected["counts"]
    footer = f"NVIDIA_BURN_TYPED_ERF_OK operations=1 formats=5 corpus_rows={c['corpus_rows']} corpus_batches={c['corpus_batches']} chunk=4096 raw_poison_variants=2 comparisons={expected['comparisons']} rows={json.dumps(expected['row_counts'])} counts={counts} route_names={routes} route_counts={route_counts} oddness_checks={json.dumps(expected['oddness_checks'])} policy_sha256={POLICY_SHA256} fixture_sha256={FIXTURE_SHA256} corpus_version=1 sdk_operations=25 protocol={protocol} profiles={profiles} kernel=81 guards=ok tails=zero raw_poison_independence=exact_bytes source_nan_payloads=preserved resident=true full_writer=verified raw_f32=FloatArithmetic native_f32=TensorElementwise queued_last_owner=verified errors=recoverable qualification=project_targets steps=2 half=f32_neighbor_image no_ftz=strict oddness=exact_sign_xor scope=forward_erf_only linux_parity=not_claimed backend=nvidia"
    require(records["OK"][0] == footer, "Erf raw exact footer, counts, preservation or SDK identity differs")
    return validate_accuracy(accuracy, expected["cell_values"], expected["comparisons"], bounds, reference_plan=plan, oddness_expected=expected["oddness_checks"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("typed-erf-manifest.json"))
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--execution-os", default="linux")
    parser.add_argument("--backend", default="flex_cpu")
    parser.add_argument("--fusion", default="disabled")
    args = parser.parse_args()
    manifest, bounds = load_bounds(args.manifest)
    expected = strict_json(args.expected.read_text())
    if args.raw:
        validate_raw(args.serial.read_text(), expected, manifest, bounds, execution_os=args.execution_os, backend=args.backend, fusion=args.fusion)
    else:
        validate(args.serial.read_text(), expected, manifest, bounds, execution_os=args.execution_os, backend=args.backend, fusion=args.fusion)
    print(json.dumps({"status": "validated_host_only", "gpu_executed": False}))
