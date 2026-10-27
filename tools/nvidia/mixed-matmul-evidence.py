"""Check frozen mixed-matmul telemetry against trusted, device-free exports.

Callers supply the independently generated summary, population and descriptions,
not fields taken from the observed report. This checks a logical record only.
Source/image/compiler provenance and execution provenance remain caller duties.
"""

import ast
import hashlib
import json
import math
import re
from collections import Counter

MODEL = "project_empirical_conditional_normal_tree_not_vendor_guarantee"
FORMATS = {
    "F16": (16, 5, 10),
    "BF16": (16, 8, 7),
    "Flex32": (32, 8, 23),
    "F32": (32, 8, 23),
    "F64": (64, 11, 52),
}
GROUPS = {
    "Geometry": "geometry",
    "Numeric": "numeric",
    "Gradient": "gradient",
    "NativeExtension": "native_extensions",
    "Preservation": "preservation",
    "F64Diagnostic": "f64_diagnostics",
}
SOURCE_ROUTES = {
    "source_before",
    "source_after",
    "alias_after",
    "autodiff_source_after",
    "autodiff_backing_source_after",
}
GRADIENT_ROUTES = {
    "autodiff",
    "autodiff_backing",
    "autodiff_transpose",
    "autodiff_staging",
}
ROUTES = (
    SOURCE_ROUTES
    | GRADIENT_ROUTES
    | {"primitive", "public", "consumer", "public_linear_staging"}
)
REPORT_FIELDS = "expected attempted_routes passed_required_routes failed required_checked diagnostic_observed model_checked checked_required_routes observed_diagnostic_routes unobserved_routes external_unobserved tracking_failures logical_success finalized failures observations worst_numeric_witness numeric_measurements diagnostic_measurements full_public_qualification physical_proof"

SUMMARY_FIELDS = (
    "version corpus_rows reserves numeric_cells corpus_last_drop_consumers selected_rows "
    "required_routes diagnostic_routes required_values diagnostic_values model_values "
    "source_values gradient_values consumer_routes external_gates groups dtype_values "
    "conditional_model vendor_rounding_guarantee device_observations physical_proof"
)

SOURCE_PINS = {
    "shared": "5d99a747cbcbdeed09ba5c5d01bc444ff607c8e4a9c6a4d3f4431a3ac4e3c4eb",
    "descriptor": "c4082b796538a782364e860658a816f547809aae0f6f9497961ea8a494c2526d",
    "fixture_blob": "c45e6e6eac8a1ee445a470318ed43576fb4652ed256bd4385d3b60b3a5e9266d",
    "generated_manifest": "9dc3fcfa1fedeae0e7489e53c5138a8bb18fb277f7521af93766f13487089c8a",
    "reference_policy": "e17e389ded361ae999a5747bd6aed4acbaa53d338316c6c2293eebe810874efe",
}
RAW_PLAN_SHA256 = "23ad56fa3b426352d4d2eb13e74074269386b26c5bbdf5cbf4cc7146ae349113"
RAW_PLAN_CANONICAL_SHA256 = {
    (54, 88): "9c4aa55385a50de6cd7ca29d9f0460f102f78b0001bec25d09b9f6291a52fd6c",
    (55, 89): "4d51d5ce554b1c19b44627cd8d390b7e6164f1c8a5a27a6ae5f0e2f3cc48972b",
}
RAW_COUNTS = dict(
    zip(
        (
            "raw_cases",
            "guarded_scratch",
            "fresh_scratch",
            "guarded_cast",
            "fresh_cast",
            "native_outputs",
            "mandatory_values",
            "diagnostic_values",
            "source_words",
            "rejections",
            "empty_aliases",
            "zero_k",
            "raw_zero_k",
            "deferred",
            "queued",
            "autodiff",
        ),
        (33, 66, 66, 44, 44, 342, 72414, 71, 45116, 83, 9, 3, 3, 15, 3, 21),
        strict=True,
    )
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _float(value):
    result = float(value)
    _require(math.isfinite(result), "nonfinite JSON exponent")
    return result


def strict_json(text):
    """Decode one JSON value; reject duplicates, NaN/Infinity and float overflow."""
    _require(type(text) is str, "JSON must be text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique,
            parse_constant=_constant,
            parse_float=_float,
        )
    except (TypeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error


def _same(actual, expected, label):
    """Structural equality without Python's bool/int or int/float aliases."""
    _require(type(actual) is type(expected), f"{label}: wrong type")
    if type(expected) is dict:
        _require(actual.keys() == expected.keys(), f"{label}: wrong fields")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif type(expected) is list:
        _require(len(actual) == len(expected), f"{label}: wrong length")
        for a, e in zip(actual, expected, strict=True):
            _same(a, e, label)
    else:
        _require(actual == expected, f"{label}: mismatch")


def _fields(value, keys, label):
    _require(
        type(value) is dict and set(value) == set(keys.split()),
        f"{label}: wrong fields",
    )


def _integer(value, label):
    _require(
        type(value) is int and 0 <= value <= 0xFFFFFFFFFFFFFFFF,
        f"{label}: expected u64",
    )
    return value


def _shape(value):
    _require(type(value) is list, "shape must be a list")
    for n in value:
        _integer(n, "shape dimension")
    return math.prod(value)


def _bits(dtype, value):
    _require(type(dtype) is str and dtype in FORMATS, "unknown dtype")
    _integer(value, "storage bits")
    _require(value < 1 << FORMATS[dtype][0], "bits exceed dtype width")
    return value


def _vector(dtype, values, length):
    _require(type(dtype) is str and dtype in FORMATS, "unknown dtype")
    _require(type(values) is list and len(values) == length, "storage length mismatch")
    for value in values:
        _bits(dtype, value)


def _key(value):
    _fields(value, "case_id route output", "route key")
    _require(all(type(x) is str and x for x in value.values()), "invalid route key")
    return value["case_id"], value["route"], value["output"]


def _class(dtype, bits):
    _, eb, fb = FORMATS[dtype]
    exponent, fraction = (bits >> fb) & ((1 << eb) - 1), bits & ((1 << fb) - 1)
    if exponent == (1 << eb) - 1:
        return "NaN" if fraction else "Infinity"
    if exponent == 0:
        return "Subnormal" if fraction else "Zero"
    return "Finite"


def _ordered(dtype, bits):
    width = FORMATS[dtype][0]
    sign = 1 << (width - 1)
    return (~bits) & ((1 << width) - 1) if bits & sign else bits | sign


def _number(dtype, bits):
    if _class(dtype, bits) in {"NaN", "Infinity"}:
        return None
    width, eb, fb = FORMATS[dtype]
    exponent = (bits >> fb) & ((1 << eb) - 1)
    fraction = bits & ((1 << fb) - 1)
    bias = (1 << (eb - 1)) - 1
    mantissa = fraction if exponent == 0 else (1 << fb) | fraction
    power = (1 - bias if exponent == 0 else exponent - bias) - fb
    return math.copysign(
        math.ldexp(float(mantissa), power), -1.0 if bits >> (width - 1) else 1.0
    )


def _rational(value):
    # Match the source's f64 numerator/denominator approximation, including its
    # inability to provide a finite reference for a nonfinite descriptor.
    _require(type(value) is str, "rational descriptor must be text")
    if value in {"nan", "inf", "-inf"}:
        return None
    _require(
        re.fullmatch(r"-?[0-9]+(?:/[1-9][0-9]*)?", value) is not None,
        "unknown rational reference",
    )
    try:
        numerator, separator, denominator = value.partition("/")
        result = float(numerator) / float(denominator if separator else "1")
        return result if math.isfinite(result) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _errors(dtype, actual, ideal):
    actual = _number(dtype, actual)
    absolute = abs(actual - ideal) if actual is not None and ideal is not None else None
    relative = absolute / abs(ideal) if absolute is not None and ideal != 0.0 else None
    return tuple(
        x if x is not None and math.isfinite(x) else None for x in (absolute, relative)
    )


def _metric(actual, expected, label):
    if expected is None:
        _require(actual is None, f"{label}: metric must be null")
        return
    _require(type(actual) in (int, float), f"{label}: invalid metric type")
    try:
        value = float(actual)
    except OverflowError as error:
        raise ValueError(f"{label}: metric exceeds f64") from error
    _require(math.isfinite(value) and value >= 0, f"{label}: invalid finite metric")
    # These are approximate f64 diagnostics, never acceptance tolerances.
    _require(
        math.isclose(value, expected, rel_tol=8 * 2**-52, abs_tol=0.0),
        f"{label}: contradictory metric",
    )


def _policy(dtype, expected, policy):
    _require(type(policy) is dict and type(policy.get("kind")) is str, "invalid policy")
    kind = policy["kind"]
    _require(
        kind
        in {
            "exact",
            "zero_class",
            "nan_class",
            "infinity",
            "project_interval",
            "observation",
        },
        "unknown policy",
    )
    _fields(policy, "kind low high" if kind == "project_interval" else "kind", "policy")
    if kind == "project_interval":
        for end in (policy["low"], policy["high"]):
            _bits(dtype, end)
            _require(_number(dtype, end) is not None, "nonfinite interval endpoint")
        _require(
            _ordered(dtype, policy["low"]) <= _ordered(dtype, policy["high"]),
            "reversed interval",
        )
        _require(
            _number(dtype, expected) is not None
            and _ordered(dtype, policy["low"])
            <= _ordered(dtype, expected)
            <= _ordered(dtype, policy["high"]),
            "interval reference is not finite and inside its bounds",
        )
    if kind in {"zero_class", "nan_class", "infinity"}:
        _require(
            _class(dtype, expected)
            == {"zero_class": "Zero", "nan_class": "NaN", "infinity": "Infinity"}[kind],
            "policy/reference class mismatch",
        )


def _accepts(dtype, expected, actual, policy):
    kind, cls = policy["kind"], _class(dtype, actual)
    if kind == "exact":
        return actual == expected
    if kind == "zero_class":
        return cls == "Zero"
    if kind == "nan_class":
        return cls == "NaN"
    if kind == "infinity":
        return cls == "Infinity" and actual == expected
    if kind == "project_interval":
        return cls not in {"NaN", "Infinity"} and (
            _ordered(dtype, policy["low"])
            <= _ordered(dtype, actual)
            <= _ordered(dtype, policy["high"])
        )
    return kind == "observation"


def _policy_text(policy):
    if policy["kind"] == "project_interval":
        return f"ProjectInterval {{ low: {policy['low']}, high: {policy['high']} }}"
    return {
        "exact": "Exact",
        "zero_class": "ZeroClass",
        "nan_class": "NaNClass",
        "infinity": "Infinity",
        "observation": "Observation",
    }[policy["kind"]]


def _descriptions(descriptions):
    if type(descriptions) is dict:
        _fields(
            descriptions,
            "conditional_model device_initialized expected_population frontend_diagnostic full_public_qualification rows",
            "descriptions",
        )
        _same(
            descriptions["device_initialized"], False, "inventory device initialization"
        )
        _same(
            descriptions["full_public_qualification"], False, "inventory qualification"
        )
        _same(descriptions["conditional_model"], MODEL, "inventory model")
        descriptions = descriptions["rows"]
    _require(type(descriptions) is list, "descriptions must contain rows")
    rows = {}
    for row in descriptions:
        _fields(
            row,
            "id group dtype operation raw_matrices raw_expected matrices expected numeric status",
            "row description",
        )
        _require(
            type(row["id"]) is str and row["id"] and row["id"] not in rows,
            "duplicate/invalid described row",
        )
        _require(
            type(row["group"]) is str and row["group"] in GROUPS,
            "unknown described group",
        )
        _require(
            row["dtype"] is None
            or (type(row["dtype"]) is str and row["dtype"] in FORMATS),
            "unknown described dtype",
        )
        _require(
            type(row["operation"]) is str
            and row["operation"] in {"Matmul", "Linear", "ExternalGate"},
            "unknown operation",
        )
        _same(row["status"], "not_device_observed", "descriptor status")
        for field in ("raw_matrices", "raw_expected"):
            _integer(row[field], field)
        for field, bits_field in (
            ("matrices", "logical_bits"),
            ("expected", "expected_bits"),
        ):
            _require(type(row[field]) is list, "invalid matrix descriptors")
            names = set()
            for matrix in row[field]:
                _fields(
                    matrix,
                    f"name dtype shape {bits_field}"
                    + (" stored_stage" if field == "expected" else ""),
                    "matrix",
                )
                _require(
                    type(matrix["name"]) is str and matrix["name"] not in names,
                    "duplicate matrix name",
                )
                names.add(matrix["name"])
                _vector(matrix["dtype"], matrix[bits_field], _shape(matrix["shape"]))
                if field == "expected":
                    _require(
                        type(matrix["stored_stage"]) is str, "invalid stored stage"
                    )
        _require(type(row["numeric"]) is list, "invalid numeric descriptors")
        for cell in row["numeric"]:
            _fields(
                cell,
                "name ideal_bits serial_bits ideal_rational sum_abs_products scratch_low scratch_high",
                "numeric descriptor",
            )
            _bits(row["dtype"], cell["ideal_bits"])
            _bits(row["dtype"], cell["serial_bits"])
            _require(
                all(
                    type(cell[k]) is str
                    for k in (
                        "name",
                        "ideal_rational",
                        "sum_abs_products",
                        "scratch_low",
                        "scratch_high",
                    )
                ),
                "invalid numeric descriptor text",
            )
        rows[row["id"]] = row
    return rows


def _broadcast(left, right):
    size = max(len(left), len(right))
    left, right = [1] * (size - len(left)) + left, [1] * (size - len(right)) + right
    result = []
    for a, b in zip(left, right, strict=True):
        _require(a == b or a == 1 or b == 1, "stage broadcast mismatch")
        result.append(b if a == 1 else a)
    return result


def _stage_witness(witness, row, dtype, output_shape, depth=0):
    _require(depth < 16, "stage graph is too deep")
    _fields(
        witness,
        "operation operands axis result_shape output_shape source_view bits_origin",
        "stage witness",
    )
    _same(
        witness["bits_origin"],
        "immutable_fixture_not_device_capture",
        "stage bits origin",
    )
    _same(witness["output_shape"], output_shape, "stage output shape")
    _shape(witness["result_shape"])
    _require(type(witness["operands"]) is list, "invalid stage operands")
    for operand in witness["operands"]:
        _fields(
            operand, "origin source dtype shape bits views producer", "stage operand"
        )
        _same(operand["dtype"], dtype, "stage operand dtype")
        _shape(operand["shape"])
        origin, source = operand["origin"], operand["source"]
        _require(type(source) is str and type(origin) is str, "invalid stage source")
        if origin == "derived_stage":
            _require(
                source in {"shared_transpose.dA", "shared_transpose.dB"},
                "unknown derived stage",
            )
            _require(type(operand["producer"]) is dict, "missing stage producer")
            shape = operand["producer"].get("output_shape")
            _shape(shape)
            _stage_witness(operand["producer"], row, dtype, shape, depth + 1)
            bits = None
        else:
            _require(origin in {"matrix", "expected_stage"}, "unknown operand origin")
            field, bits_field = (
                ("matrices", "logical_bits")
                if origin == "matrix"
                else ("expected", "expected_bits")
            )
            candidates = [m for m in row[field] if m["name"] == source]
            _require(len(candidates) == 1, "unknown stage matrix/reference")
            reference = candidates[0]
            _same(reference["dtype"], dtype, "stage reference dtype")
            _same(operand["producer"], None, "frozen operand producer")
            shape, bits = list(reference["shape"]), list(reference[bits_field])
        _require(type(operand["views"]) is list, "invalid stage views")
        for view in operand["views"]:
            _require(
                type(view) is dict and type(view.get("kind")) is str,
                "invalid operand transform",
            )
            if view["kind"] == "reshape":
                _fields(view, "kind shape", "reshape")
                _require(
                    _shape(view["shape"]) == math.prod(shape),
                    "reshape changes operand population",
                )
                shape = list(view["shape"])
            else:
                _require(
                    view["kind"] == "transpose" and len(shape) >= 2,
                    "unknown operand transform",
                )
                _fields(view, "kind first second", "transpose")
                _same(view["first"], len(shape) - 2, "transpose first axis")
                _same(view["second"], len(shape) - 1, "transpose second axis")
                rows, columns = shape[-2:]
                if bits is not None:
                    bits = [
                        bits[(batch * rows + r) * columns + c]
                        for batch in range(math.prod(shape[:-2]))
                        for c in range(columns)
                        for r in range(rows)
                    ]
                shape[-2], shape[-1] = columns, rows
        _same(operand["shape"], shape, "transformed operand shape")
        _same(operand["bits"], bits, "transformed operand bits")
    operation, operands = witness["operation"], witness["operands"]
    _require(
        type(operation) is str
        and operation in {"matmul", "typed_sum", "typed_add", "view_adjoint"},
        "unknown stage operation",
    )
    _require(
        len(operands) == (2 if operation in {"matmul", "typed_add"} else 1),
        "stage operand count mismatch",
    )
    if operation == "matmul":
        lhs, rhs = (o["shape"] for o in operands)
        _require(
            len(lhs) >= 2 and len(rhs) >= 2 and lhs[-1] == rhs[-2],
            "stage contraction mismatch",
        )
        result = _broadcast(lhs[:-2], rhs[:-2]) + [lhs[-2], rhs[-1]]
    elif operation == "typed_add":
        result = _broadcast(operands[0]["shape"], operands[1]["shape"])
    elif operation == "typed_sum":
        axis = _integer(witness["axis"], "sum axis")
        result = list(operands[0]["shape"])
        _require(axis < len(result), "sum axis out of bounds")
        result[axis] = 1
    else:
        view = witness["source_view"]
        _fields(view, "strides base root_shape", "source view")
        _integer(view["base"], "source view base")
        _shape(view["root_shape"])
        _require(
            type(view["strides"]) is list
            and len(view["strides"]) == len(operands[0]["shape"]),
            "source view rank mismatch",
        )
        _require(
            all(type(n) is int and -(1 << 63) <= n < 1 << 63 for n in view["strides"]),
            "invalid signed source strides",
        )
        result = view["root_shape"]
    if operation != "typed_sum":
        _same(witness["axis"], None, "stage axis")
    if operation != "view_adjoint":
        _same(witness["source_view"], None, "stage source view")
    _same(witness["result_shape"], result, "stage result shape")
    _require(
        math.prod(result) == math.prod(output_shape),
        "stage output reshape population mismatch",
    )


def _context(expected_summary, population, descriptions):
    rows = _descriptions(descriptions)
    _require(type(population) is list, "population must be a list")
    plans, selected, totals = {}, set(), Counter()
    dtypes = dict.fromkeys(("F16", "BF16", "Flex32", "F64"), 0)
    for plan in population:
        _fields(
            plan,
            "key dtype shape values required_values diagnostic_values model_values expected_bits policies stored_stage source stage_witness",
            "population row",
        )
        key = _key(plan["key"])
        _require(key not in plans, "duplicate population key")
        _require(key[0] in rows and key[1] in ROUTES, "unknown population row/route")
        row = rows[key[0]]
        _require(
            row["group"] in {"Geometry", "Numeric", "Gradient"},
            "unsupported selection: only Scope::Shared qualifies",
        )
        _require(
            type(plan["source"]) is bool
            and plan["source"] == (key[1] in SOURCE_ROUTES),
            "source route identity mismatch",
        )
        _require(
            type(plan["stored_stage"]) is str and plan["stored_stage"],
            "invalid population stage",
        )
        n = _shape(plan["shape"])
        if plan["source"]:
            _same(plan["stage_witness"], None, "source stage witness")
        else:
            _stage_witness(plan["stage_witness"], row, plan["dtype"], plan["shape"])
        _same(plan["values"], n, "planned shape/value count")
        _vector(plan["dtype"], plan["expected_bits"], n)
        _require(
            plan["dtype"] in dtypes and plan["dtype"] != "F64",
            "unsupported Shared dtype",
        )
        _require(
            type(plan["policies"]) is list and len(plan["policies"]) == n,
            "policy count mismatch",
        )
        counts = Counter()
        for expected, policy in zip(
            plan["expected_bits"], plan["policies"], strict=True
        ):
            _policy(plan["dtype"], expected, policy)
            if plan["source"]:
                _require(
                    policy["kind"] == "exact", "source identity must use exact bits"
                )
            counts[
                "diagnostic_values"
                if policy["kind"] == "observation"
                else "required_values"
            ] += 1
            counts["model_values"] += policy["kind"] == "project_interval"
        for field in ("required_values", "diagnostic_values", "model_values"):
            _same(plan[field], counts[field], f"planned {field}")
            totals[field] += counts[field]
        totals["required_routes"] += counts["required_values"] > 0 or n == 0
        totals["diagnostic_routes"] += counts["diagnostic_values"] > 0
        totals["source_values"] += n if plan["source"] else 0
        totals["consumer_routes"] += key[1] == "consumer"
        totals["gradient_values"] += (
            n if key[1] in GRADIENT_ROUTES and not key[2].endswith("forward") else 0
        )
        dtypes[plan["dtype"]] += n
        plans[key] = plan
        selected.add(key[0])
    _require(selected and plans, "unsupported selection: zero selected cases")
    groups = dict.fromkeys(GROUPS.values(), 0)
    for case_id in selected:
        groups[GROUPS[rows[case_id]["group"]]] += 1
    summary = {
        "version": 1,
        "corpus_rows": 121,
        "reserves": 3,
        "numeric_cells": 65,
        "corpus_last_drop_consumers": 141,
        "selected_rows": len(selected),
        **{
            field: totals[field]
            for field in (
                "required_routes",
                "diagnostic_routes",
                "required_values",
                "diagnostic_values",
                "model_values",
                "source_values",
                "gradient_values",
                "consumer_routes",
            )
        },
        "external_gates": 0,
        "groups": groups,
        "dtype_values": dtypes,
        "conditional_model": MODEL,
        "vendor_rounding_guarantee": False,
        "device_observations": 0,
        "physical_proof": False,
    }
    _same(expected_summary, summary, "independent summary/population")
    return plans, rows, summary


def _measurement(observation, plan, row, i):
    cell = row["numeric"][i]
    dtype, actual = observation["dtype"], observation["values"][i]
    policy = plan["policies"][i]
    absolute, relative = _errors(dtype, actual, _rational(cell["ideal_rational"]))
    ulp = (
        None
        if _number(dtype, actual) is None or _number(dtype, cell["ideal_bits"]) is None
        else abs(_ordered(dtype, actual) - _ordered(dtype, cell["ideal_bits"]))
    )
    return {
        "key": observation["key"],
        "cell": i,
        "name": cell["name"],
        "policy": _policy_text(policy),
        "required": policy["kind"] != "observation",
        "actual_class": _class(dtype, actual),
        "actual_bits": actual,
        "ideal_stored_bits": cell["ideal_bits"],
        "serial_diagnostic_bits": cell["serial_bits"],
        "exact_operand_dot": cell["ideal_rational"],
        "sum_abs_products": cell["sum_abs_products"],
        "conditional_scratch_interval": [cell["scratch_low"], cell["scratch_high"]],
        "approximate_absolute_error": absolute,
        "approximate_relative_error": relative,
        "storage_ULP_distance": ulp,
        "metrics_are_acceptance": False,
    }


def _worst(observation, plan, row, i, ulp):
    dtype, actual = observation["dtype"], observation["values"][i]
    expected = plan["expected_bits"][i]
    cell = (
        row["numeric"][i]
        if observation["key"]["output"] == "expected" and i < len(row["numeric"])
        else None
    )
    rational = cell["ideal_rational"] if cell else ""
    reference = _rational(rational) if rational else _number(dtype, expected)
    absolute, relative = _errors(dtype, actual, reference)
    stage = plan["stage_witness"]
    lhs = stage["operands"][0] if stage["operands"] else None
    rhs = stage["operands"][1] if len(stage["operands"]) > 1 else None
    return {
        "key": observation["key"],
        "operation": stage["operation"],
        "cell": i,
        "stored_stage": plan["stored_stage"],
        "dtype": dtype,
        "actual": actual,
        "expected": expected,
        "storage_ULP_distance": ulp,
        "approximate_absolute_error": absolute,
        "approximate_relative_error": relative,
        "exact_ideal_rational": rational,
        "sum_abs_products": cell["sum_abs_products"] if cell else "",
        "lhs_shape": lhs["shape"] if lhs else None,
        "rhs_shape": rhs["shape"] if rhs else None,
        "lhs_bits": lhs["bits"] if lhs else None,
        "rhs_bits": rhs["bits"] if rhs else None,
        "stage_witness": stage,
        "metrics_are_acceptance": False,
    }


def _measurement_equal(actual, expected, label):
    _require(
        type(actual) is dict and actual.keys() == expected.keys(),
        f"{label}: wrong fields",
    )
    for key, value in expected.items():
        if key in {"approximate_absolute_error", "approximate_relative_error"}:
            _metric(actual[key], value, f"{label}.{key}")
        else:
            _same(actual[key], value, f"{label}.{key}")


def validate_report(report, expected_summary, population, descriptions):
    """Validate all observations and diagnostics; raise ValueError on rejection.

    The returned counters are derived from observations. No claimed success bit
    or claimed count can make a missing, duplicate or invalid observation pass.
    Only a nonempty Shared selection qualifies, including filtered CPU records.
    """
    plans, rows, summary = _context(expected_summary, population, descriptions)
    _fields(
        report,
        REPORT_FIELDS,
        "report",
    )
    _same(report["expected"], summary, "report expected summary")
    for label in (
        "failed",
        "unobserved_routes",
        "external_unobserved",
        "tracking_failures",
    ):
        _same(report[label], 0, label)
    _same(report["failures"], [], "failure examples")
    _same(report["finalized"], True, "report must be finalized")
    for label, value in (
        ("logical_success", True),
        ("full_public_qualification", False),
        ("physical_proof", False),
    ):
        _same(report[label], value, label)
    _require(
        type(report["observations"]) is list and report["observations"],
        "no device observations",
    )
    seen, counts, numeric, diagnostic, worst = set(), Counter(), [], [], None
    for observation in report["observations"]:
        _fields(
            observation,
            "key dtype shape values required_checked diagnostic_observed model_checked zero_sign_differences",
            "observation",
        )
        key = _key(observation["key"])
        _require(key in plans, "unexpected observed route")
        _require(key not in seen, "duplicate observed route")
        seen.add(key)
        plan, row = plans[key], rows[key[0]]
        dtype = plan["dtype"]
        _same(observation["dtype"], dtype, "observed dtype")
        _same(observation["shape"], plan["shape"], "observed shape")
        _vector(dtype, observation["values"], plan["values"])
        zero_differences = 0
        for i, (actual, expected, policy) in enumerate(
            zip(
                observation["values"],
                plan["expected_bits"],
                plan["policies"],
                strict=True,
            )
        ):
            _require(
                _accepts(dtype, expected, actual, policy),
                f"value policy failure: {key} cell={i}",
            )
            zero_differences += (
                _class(dtype, actual) == _class(dtype, expected) == "Zero"
                and actual != expected
            )
            if (
                not plan["source"]
                and _number(dtype, actual) is not None
                and _number(dtype, expected) is not None
            ):
                ulp = abs(_ordered(dtype, actual) - _ordered(dtype, expected))
                if worst is None or ulp > worst["storage_ULP_distance"]:
                    worst = _worst(observation, plan, row, i, ulp)
        for actual_field, expected_field in (
            ("required_checked", "required_values"),
            ("diagnostic_observed", "diagnostic_values"),
            ("model_checked", "model_values"),
        ):
            _same(
                observation[actual_field],
                plan[expected_field],
                f"observation {actual_field}",
            )
            counts[actual_field] += plan[expected_field]
        _same(
            observation["zero_sign_differences"],
            zero_differences,
            "zero sign differences",
        )
        required_route = plan["required_values"] > 0 or plan["values"] == 0
        counts["checked_required_routes"] += required_route
        counts["passed_required_routes"] += required_route
        counts["observed_diagnostic_routes"] += plan["diagnostic_values"] > 0
        if key[2] == "expected" and row["numeric"]:
            _require(
                len(row["numeric"]) == plan["values"],
                "numeric descriptor/value count mismatch",
            )
            for i in range(plan["values"]):
                measure = _measurement(observation, plan, row, i)
                numeric.append(measure)
                if not measure["required"]:
                    diagnostic.append(measure)
    _require(
        seen == plans.keys(),
        "missing route observation (including diagnostics and empty routes)",
    )
    counts["attempted_routes"] = len(seen)
    for field, value in counts.items():
        _same(report[field], value, field)
    # Counter keys with a derived zero must also be checked.
    for field in (
        "required_checked",
        "diagnostic_observed",
        "model_checked",
        "checked_required_routes",
        "passed_required_routes",
        "observed_diagnostic_routes",
    ):
        _same(report[field], counts[field], field)
    for field, expected in (
        ("numeric_measurements", numeric),
        ("diagnostic_measurements", diagnostic),
    ):
        _require(
            type(report[field]) is list and len(report[field]) == len(expected),
            f"{field}: missing/extra metrics",
        )
        for actual, derived in zip(report[field], expected, strict=True):
            _measurement_equal(actual, derived, field)
    if worst is None:
        _same(report["worst_numeric_witness"], None, "worst numeric witness")
    else:
        _measurement_equal(
            report["worst_numeric_witness"], worst, "worst numeric witness"
        )
    return {
        "logical_success": True,
        "finalized": True,
        "scope": "Shared",
        "expected": summary,
        **counts,
        "observed_routes": len(seen),
        "full_public_qualification": False,
        "physical_proof": False,
        "vendor_rounding_guarantee": False,
        "conditional_model": MODEL,
    }


# infer::report logs INFO at infer.rs:50. The userspace logger adds its
# [infer.rs@050] field; sys_log forwards it at sys_logger.rs:20. Kernel log.rs
# adds seconds.milliseconds, INF and its source field, with the exact SGR
# colors from graphic::ansi. Permit this production prefix, its SGR-free form,
# or bare CLI records. Never strip arbitrary leading text or payload escapes.
_TELEMETRY_MARKERS = frozenset(
    {
        "BURN_MIXED_MATMUL_REPORT",
        "BURN_MIXED_MATMUL_LOGICAL_OK",
        "NVIDIA_BURN_MIXED_MATMUL_PLAN",
        "NVIDIA_BURN_MIXED_MATMUL_SHARED",
        "NVIDIA_BURN_MIXED_MATMUL_SEPARATE",
        "NVIDIA_BURN_MIXED_MATMUL_OBSERVED",
        "NVIDIA_BURN_MIXED_MATMUL_STRICT_OK",
        "NVIDIA_BURN_MIXED_MATMUL_RAW_NUMERIC",
    }
)
_LOG_TIME = r"\[(?:0|[1-9][0-9]*)\.[0-9]{3}\]"
_D3OS_LOG_RECORD = re.compile(
    r"(?:" + _LOG_TIME + r"\[INF\]\[sys_logger\.rs@020\]"
    r"|\x1b\[36m"
    + _LOG_TIME
    + r"\x1b\[94m\[INF\]\x1b\[35m\[sys_logger\.rs@020\]\x1b\[39m)"
    r" \[infer\.rs@050\] (?P<record>.*)"
)


def _has_marker(line, marker):
    return line.startswith(marker)


def _serial_lines(serial, *, execution_os):
    _require(type(serial) is str and "\x00" not in serial, "invalid serial text")
    raw_lines = serial.split("\n")
    lines = []
    for i, line in enumerate(raw_lines):
        # CR is a terminator only as part of a physical CRLF line ending.
        if i < len(raw_lines) - 1 and line.endswith("\r"):
            line = line[:-1]
        if "BURN_MIXED_MATMUL_" in line:
            logged = _D3OS_LOG_RECORD.fullmatch(line)
            if logged:
                _require(execution_os == "d3os", "D3OS logger prefix in Linux run")
                line = logged["record"]
            token, separator, payload = line.partition(" ")
            _require(
                token in _TELEMETRY_MARKERS and separator and payload,
                "unknown or contaminated mixed-matmul record marker/prefix",
            )
            _require(
                (execution_os == "linux" and not token.startswith("NVIDIA_"))
                or (execution_os == "d3os" and token != "BURN_MIXED_MATMUL_REPORT"),
                "mixed-matmul record marker belongs to another OS",
            )
            _require(
                re.search(r"[\x00-\x1f\x7f]", line) is None,
                "control sequence in mixed-matmul record payload",
            )
        # Only a complete separate-scope record can contain honest backend
        # failure diagnostics without failing the Shared run.
        if not _has_marker(line, "NVIDIA_BURN_MIXED_MATMUL_SEPARATE "):
            _require(
                not re.search(
                    r"MIXED_MATMUL_FAILURE|UNFROZEN|mixed matmul strict failure|mixed matrix logical conformance failed|\bpanicked\b",
                    line,
                    re.IGNORECASE,
                ),
                "failure marker in serial",
            )
        lines.append(line)
    return lines


def _one_line(lines, marker):
    matches = [
        (i, line[len(marker) :])
        for i, line in enumerate(lines)
        if _has_marker(line, marker)
    ]
    _require(len(matches) == 1, f"missing/duplicate {marker}")
    return matches[0]


def _footer(summary, execution_os, backend, fusion, scope):
    _require(scope == "Shared", "unsupported selection: only Scope::Shared qualifies")
    for label in (execution_os, backend, fusion):
        _require(
            type(label) is str and re.fullmatch(r"[A-Za-z0-9_]+", label),
            "invalid runtime label",
        )
    return (
        f"BURN_MIXED_MATMUL_LOGICAL_OK version={summary['version']} scope={scope} "
        f"rows={summary['selected_rows']} required_checked={summary['required_values']} "
        f"diagnostic_observed={summary['diagnostic_values']} model_checked={summary['model_values']} "
        f"physical_proof=false full_public_qualification=false conditional_model={MODEL} "
        f"os={execution_os} backend={backend} fusion={fusion}"
    )


def validate(
    serial,
    expected_summary,
    population,
    descriptions,
    *,
    execution_os,
    backend,
    fusion,
    scope="Shared",
):
    """Validate a complete Shared report followed by its exact runtime footer."""
    _require(scope == "Shared", "unsupported selection: only Scope::Shared qualifies")
    _require(
        type(execution_os) is str and execution_os in {"linux", "d3os"},
        "unsupported execution OS",
    )
    lines = _serial_lines(serial, execution_os=execution_os)
    marker = (
        "BURN_MIXED_MATMUL_REPORT "
        if execution_os == "linux"
        else "NVIDIA_BURN_MIXED_MATMUL_SHARED "
    )
    other = (
        "NVIDIA_BURN_MIXED_MATMUL_SHARED "
        if execution_os == "linux"
        else "BURN_MIXED_MATMUL_REPORT "
    )
    _require(
        not any(_has_marker(line, other) for line in lines),
        "report prefix does not match execution OS",
    )
    report_i, report_text = _one_line(lines, marker)
    footer_i, _ = _one_line(lines, "BURN_MIXED_MATMUL_LOGICAL_OK ")
    _require(report_i < footer_i, "stale/out-of-order success footer")
    result = validate_report(
        strict_json(report_text), expected_summary, population, descriptions
    )
    _same(
        lines[footer_i],
        _footer(expected_summary, execution_os, backend, fusion, scope),
        "runtime success footer",
    )
    if any(_has_marker(line, "NVIDIA_BURN_MIXED_MATMUL_SEPARATE ") for line in lines):
        _require(execution_os == "d3os", "D3OS separate descriptors in Linux run")
        _separate_reports(lines, report_i, footer_i)
    return result | {"execution_os": execution_os, "backend": backend, "fusion": fusion}


def _debug_counts(text):
    match = re.fullmatch(r"Counts \{ (.*) \}", text)
    _require(match is not None, "invalid Rust Counts record")
    counts = {}
    for field in match[1].split(", "):
        item = re.fullmatch(r"([a-z_]+): (0|[1-9][0-9]*)", field)
        _require(
            item is not None and item[1] not in counts, "invalid/duplicate Counts field"
        )
        counts[item[1]] = _integer(int(item[2]), "raw count")
    _same(counts, RAW_COUNTS, "frozen Raw counts")
    return counts


def _summary_shape(summary):
    _fields(summary, SUMMARY_FIELDS, "separate summary")
    for field in set(SUMMARY_FIELDS.split()) - {
        "groups",
        "dtype_values",
        "conditional_model",
        "vendor_rounding_guarantee",
        "physical_proof",
    }:
        _integer(summary[field], f"separate summary {field}")
    for field, value in (
        ("version", 1),
        ("corpus_rows", 121),
        ("reserves", 3),
        ("numeric_cells", 65),
        ("corpus_last_drop_consumers", 141),
        ("device_observations", 0),
        ("vendor_rounding_guarantee", False),
        ("physical_proof", False),
        ("conditional_model", MODEL),
    ):
        _same(summary[field], value, f"separate summary {field}")
    for field, keys in (
        ("groups", " ".join(GROUPS.values())),
        ("dtype_values", "F16 BF16 Flex32 F64"),
    ):
        _fields(summary[field], keys, f"separate summary {field}")
        for value in summary[field].values():
            _integer(value, f"separate summary {field} count")
    _require(
        sum(summary["groups"].values()) == summary["selected_rows"],
        "separate row population mismatch",
    )
    _require(
        sum(summary["dtype_values"].values())
        == summary["required_values"] + summary["diagnostic_values"],
        "separate dtype population mismatch",
    )
    _require(
        summary["model_values"] <= summary["required_values"],
        "separate model is not a required subset",
    )


def _separate_reports(lines, after, before):
    separate, indices = [], []
    for scope, rows in (
        ("NativeExtensions", 12),
        ("Preservation", 8),
        ("F64Diagnostics", 8),
    ):
        index, payload = _one_line(
            lines, f"NVIDIA_BURN_MIXED_MATMUL_SEPARATE scope={scope} "
        )
        _require(
            after < index < before and payload.startswith("qualification=false "),
            "missing separate diagnostic label/order",
        )
        descriptor = strict_json(payload[len("qualification=false ") :])
        _fields(descriptor, REPORT_FIELDS, "separate report")
        for field in (
            "attempted_routes",
            "passed_required_routes",
            "failed",
            "required_checked",
            "diagnostic_observed",
            "model_checked",
            "checked_required_routes",
            "observed_diagnostic_routes",
            "unobserved_routes",
            "external_unobserved",
            "tracking_failures",
        ):
            _integer(descriptor[field], f"separate {field}")
        for field in (
            "observations",
            "numeric_measurements",
            "diagnostic_measurements",
            "failures",
        ):
            _require(type(descriptor[field]) is list, f"separate {field}: wrong type")
        _require(
            descriptor["worst_numeric_witness"] is None
            or type(descriptor["worst_numeric_witness"]) is dict,
            "separate worst witness: wrong type",
        )
        _require(
            type(descriptor) is dict and type(descriptor.get("expected")) is dict,
            "invalid separate report",
        )
        _summary_shape(descriptor["expected"])
        _same(
            descriptor["expected"].get("selected_rows"),
            rows,
            "separate descriptor population",
        )
        _same(descriptor.get("finalized"), True, "separate report must be finalized")
        for field in ("physical_proof", "full_public_qualification", "logical_success"):
            _same(descriptor.get(field), False, f"separate {field}")
        external = {"NativeExtensions": 12, "Preservation": 8, "F64Diagnostics": 1}[
            scope
        ]
        _same(
            descriptor["expected"].get("external_gates"),
            external,
            "separate external population",
        )
        _same(
            descriptor.get("external_unobserved"),
            external,
            "separate external gates must remain unobserved",
        )
        separate.append(descriptor)
        indices.append(index)
    _require(
        sum(_has_marker(line, "NVIDIA_BURN_MIXED_MATMUL_SEPARATE ") for line in lines)
        == 3,
        "unexpected separate scope",
    )
    _require(indices == sorted(set(indices)), "out-of-order separate scopes")
    return separate


def validate_raw(
    serial, expected_counts, *, protocol=54, profiles=88, source_pins=None
):
    """Check the frozen Raw marker contract, not unreported per-word values.

    source_pins, when supplied, is the five-label SOURCE_PINS dictionary.
    The parent must independently call validate_report on returned shared_report,
    and bind the run to frozen source, image and compiler provenance. Matching
    counters cannot establish physical guards, queued residency or GPU behavior.
    """
    _same(expected_counts, RAW_COUNTS, "independent frozen Raw counts")
    _require(
        type(protocol) is int
        and type(profiles) is int
        and (protocol, profiles) in RAW_PLAN_CANONICAL_SHA256,
        "Raw unqualified protocol/profile pair",
    )
    pins = SOURCE_PINS if source_pins is None else source_pins
    _same(pins, SOURCE_PINS, "independent frozen source pins")
    lines = _serial_lines(serial, execution_os="d3os")
    _require(
        not any(_has_marker(line, "BURN_MIXED_MATMUL_REPORT ") for line in lines),
        "Linux report in Raw D3OS run",
    )
    markers = (
        "NVIDIA_BURN_MIXED_MATMUL_PLAN ",
        "NVIDIA_BURN_MIXED_MATMUL_SHARED ",
        "NVIDIA_BURN_MIXED_MATMUL_OBSERVED ",
        "BURN_MIXED_MATMUL_LOGICAL_OK ",
        "NVIDIA_BURN_MIXED_MATMUL_STRICT_OK ",
    )
    records = [_one_line(lines, marker) for marker in markers]
    indices = [record[0] for record in records]
    _require(indices == sorted(set(indices)), "out-of-order Raw markers")
    plan = re.fullmatch(
        rf"frozen=true protocol={protocol} kernels={profiles} source_pins=(\[.*?\]) raw=(Counts \{{[^{{}}]*\}}) "
        r"shared=(\{.*\}) population=(\{.*\}) conditional_model="
        + MODEL
        + r" full_public_qualification=false",
        records[0][1],
    )
    _require(plan is not None, "missing frozen Raw PLAN fields")
    try:
        pairs = ast.literal_eval(plan[1])
    except (ValueError, SyntaxError, RecursionError) as error:
        raise ValueError("invalid source pins") from error
    _require(
        type(pairs) is list
        and all(
            type(p) is tuple and len(p) == 2 and all(type(v) is str for v in p)
            for p in pairs
        ),
        "invalid source pin pairs",
    )
    _require(
        len(pairs) == len(pins) and len({p[0] for p in pairs}) == len(pairs),
        "duplicate/missing source pin",
    )
    _same(dict(pairs), pins, "Raw runtime source pins")
    _debug_counts(plan[2])
    shared_summary, raw_population = strict_json(plan[3]), strict_json(plan[4])
    # Protocol 54 binds the approved plan.json SHA256 above. Protocol 55 adds
    # only one zero counter per kernel array for the appended indexed entry.
    # It binds every recipe, shape, counter and Shared summary without trusting
    # telemetry as its own expected plan. JSON field order is immaterial.
    canonical = json.dumps(
        raw_population, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    _require(
        hashlib.sha256(canonical.encode()).hexdigest()
        == RAW_PLAN_CANONICAL_SHA256[protocol, profiles],
        "Raw population differs from frozen plan",
    )
    _same(shared_summary, raw_population["shared"], "Raw Shared summary")
    shared = strict_json(records[1][1])
    _fields(shared, REPORT_FIELDS, "Raw Shared report")
    _same(shared.get("expected"), shared_summary, "Raw Shared report population")
    for field in (
        "failed",
        "tracking_failures",
        "unobserved_routes",
        "external_unobserved",
    ):
        _same(shared.get(field), 0, f"Raw Shared {field}")
    _same(shared.get("failures"), [], "Raw Shared failures")
    _same(shared.get("logical_success"), True, "Raw Shared claimed success")
    _same(shared.get("finalized"), True, "Raw Shared must be finalized")
    for field in ("physical_proof", "full_public_qualification"):
        _same(shared.get(field), False, f"Raw Shared {field}")
    for actual, expected in (
        ("required_checked", "required_values"),
        ("diagnostic_observed", "diagnostic_values"),
        ("model_checked", "model_values"),
        ("checked_required_routes", "required_routes"),
        ("passed_required_routes", "required_routes"),
        ("observed_diagnostic_routes", "diagnostic_routes"),
    ):
        _same(shared.get(actual), shared_summary[expected], f"Raw Shared {actual}")
    _require(
        type(shared.get("observations")) is list and shared["observations"],
        "Raw Shared observations absent",
    )
    separate = _separate_reports(lines, indices[1], indices[2])
    observed = re.fullmatch(
        r"accepted=true expected=(Counts \{[^{}]*\}) observed=(Counts \{[^{}]*\}) failed=0 examples=\[\] "
        r"external_extension_evidence=raw_separate preservation=separate f64=diagnostic_only full_public_qualification=false",
        records[2][1],
    )
    _require(
        observed is not None, "Raw observations did not pass every source-bound gate"
    )
    _debug_counts(observed[1])
    actual_counts = _debug_counts(observed[2])
    _same(
        lines[records[3][0]],
        _footer(shared_summary, "d3os", "native", "disabled", "Shared"),
        "Raw runtime footer",
    )
    strict = re.fullmatch(
        rf"scope=shared93_plus_raw protocol={protocol} kernels={profiles} observed=(Counts \{{[^{{}}]*\}}) "
        r"original_allocation_tail=checked full_public_qualification=false",
        records[4][1],
    )
    _require(strict is not None, "missing Raw final strict gate")
    _debug_counts(strict[1])
    return {
        "raw_marker_consistency": True,
        "counts": actual_counts,
        "protocol": protocol,
        "profiles": profiles,
        "source_pins": dict(pins),
        "shared_report": shared,
        "separate_reports": separate,
        "requires_independent_shared_validation": True,
        "requires_execution_provenance": True,
        "requires_source_image_compiler_provenance": True,
        "raw_per_word_values_revalidated": False,
        "raw_numeric_metrics_revalidated": False,
        "physical_proof": False,
        "queued_residency_proven": False,
        "full_public_qualification": False,
        "vendor_rounding_guarantee": False,
    }
