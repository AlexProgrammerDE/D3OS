"""Check Power qualification records against immutable stored-input references."""

import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple

DTYPES = ("f64", "f32", "flex32", "f16", "bf16")
SCHEDULES = (
    "mathematical",
    "ones",
    "identity",
    "square",
    "reciprocal",
    "square_reciprocal",
)
ROUTES = (
    "tensor",
    "scalar_impl",
    "powi_scalar_impl",
    "powf_scalar",
    "powi_scalar",
    "public_powf_scalar",
    "public_powi_scalar",
    "tensor_powi",
    "public_tensor",
)
WIDTHS = (8, 4, 4, 2, 2)
EXPONENTS = (0x7FF0000000000000, 0x7F800000, 0x7F800000, 0x7C00, 0x7F80)


class Reference(NamedTuple):
    lhs: int
    rhs: int
    ideal: int
    staged: int
    low: int
    high: int
    exact: bool


def vector(value, length):
    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(n) is int and n >= 0 for n in value)
    )


def bucket(value):
    for index, upper in enumerate((0, 1, 2, 4, 8, 16)):
        if value <= upper:
            return index
    return 6


def classify(fi, bits):
    magnitude = bits & ((1 << (8 * WIDTHS[fi] - 1)) - 1)
    exponent = EXPONENTS[fi]
    if magnitude == 0:
        return 0
    if magnitude < exponent & -exponent:
        return 1
    if magnitude < exponent:
        return 2
    return 3 if magnitude == exponent else 4


def rank(fi, bits):
    sign = 1 << (8 * WIDTHS[fi] - 1)
    return (~bits & (2 * sign - 1)) if bits & sign else bits | sign


def distance(fi, a, b):
    return (
        abs(rank(fi, a) - rank(fi, b))
        if classify(fi, a) in (1, 2) and classify(fi, b) in (1, 2)
        else None
    )


def passing(fi, schedule, row, actual):
    wanted, observed = classify(fi, row.staged), classify(fi, actual)
    if schedule == "identity":
        return actual == row.staged
    if wanted == 4:
        return observed == 4
    if wanted in (0, 3) or schedule != "mathematical" or row.exact:
        return actual == row.staged
    if observed not in (1, 2) or (actual ^ row.staged) >> (8 * WIDTHS[fi] - 1):
        return False
    if fi >= 3:
        return rank(fi, row.low) <= rank(fi, actual) <= rank(fi, row.high)
    return distance(fi, actual, row.staged) <= (2 if fi == 0 else 4)


def load_references(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    policy = manifest["policy"]
    policy_digest = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        manifest.get("corpus_version") != 1
        or manifest.get("policy_sha256") != policy_digest
        or type(policy.get("f64_steps")) is not int
        or policy["f64_steps"] != 2
        or type(policy.get("f32_flex32_steps")) is not int
        or policy["f32_flex32_steps"] != 4
        or policy.get("fast_math") is not False
        or policy.get("half") != "destination RNE image of +/-4 F32 neighbors"
    ):
        raise RuntimeError("Power reference policy differs from qualification targets")
    data = (
        manifest_path.parents[2]
        / "os/library/ai/src/validation/typed_power_fixtures.bin"
    ).read_bytes()
    if (
        len(data) != manifest["fixture_bytes"]
        or hashlib.sha256(data).hexdigest() != manifest["fixture_sha256"]
        or manifest["dtype_order"] != list(DTYPES)
        or manifest["schedule_order"]
        != [
            "Mathematical",
            "Ones",
            "Identity",
            "Square",
            "Reciprocal",
            "SquareReciprocal",
        ]
    ):
        raise RuntimeError("Power reference inventory differs")
    references, limits, offset = {}, {}, 0
    for fi, dtype in enumerate(DTYPES):
        width = WIDTHS[fi]
        record_bytes = 6 * width + 6
        metadata = manifest["formats"][dtype]
        if (
            metadata["offset_bytes"] != offset
            or metadata["record_bytes"] != record_bytes
        ):
            raise RuntimeError("Power reference layout differs")
        previous = None
        for _ in range(metadata["rows"]):
            end = offset + record_bytes
            if end > len(data):
                raise RuntimeError("Power reference row is truncated")
            words = [
                int.from_bytes(
                    data[offset + i * width : offset + (i + 1) * width], "little"
                )
                for i in range(6)
            ]
            schedule_id, exact = data[offset + 6 * width + 4 : end]
            if schedule_id >= len(SCHEDULES) or exact not in (0, 1):
                raise RuntimeError("Power reference selector is invalid")
            schedule = SCHEDULES[schedule_id]
            row = Reference(*words, bool(exact))
            key = (schedule_id, row.lhs, row.rhs)
            if previous is not None and previous >= key:
                raise RuntimeError("Power reference keys are not unique and ordered")
            previous = key
            if schedule_id and not row.exact:
                raise RuntimeError(
                    "Power shortcut reference does not require stored rounding"
                )
            references[dtype, schedule, row.lhs, row.rhs] = row
            bound = limits.setdefault((dtype, schedule), [0, 0])
            if schedule == "mathematical" and fi < 3:
                if row.ideal != row.staged:
                    raise RuntimeError("Power wide mathematical references differ")
                bound[:] = [2 if fi == 0 else 4] * 2
            else:
                for actual in (
                    (row.low, row.high) if schedule_id == 0 else (row.staged,)
                ):
                    if passing(fi, schedule, row, actual):
                        for i, ref in enumerate((row.staged, row.ideal)):
                            steps = distance(fi, actual, ref)
                            if steps is not None:
                                bound[i] = max(bound[i], steps)
            offset = end
    if offset != len(data):
        raise RuntimeError("Power reference byte count differs")
    return manifest, references, limits


def cell_key(cell):
    if (
        not isinstance(cell, dict)
        or cell.get("dtype") not in DTYPES
        or cell.get("route") not in ROUTES
        or cell.get("schedule") not in SCHEDULES
        or type(cell.get("chain")) is not bool
    ):
        raise RuntimeError("Power cell identity is invalid")
    return cell["dtype"], cell["route"], cell["schedule"], cell["chain"]


def planned_cells(summary):
    if (
        not isinstance(summary, dict)
        or any(type(v) is not int or v < 0 for k, v in summary.items() if k != "cells")
        or not isinstance(summary.get("cells"), list)
    ):
        raise RuntimeError("Power work plan is invalid")
    result = {}
    for cell in summary["cells"]:
        key = cell_key(cell)
        if (
            key in result
            or type(cell.get("values")) is not int
            or cell["values"] <= 0
            or any(
                not vector(cell.get(k), 5) or sum(cell[k]) != cell["values"]
                for k in ("reference_classes", "ideal_classes")
            )
        ):
            raise RuntimeError("Power reference population plan is invalid")
        result[key] = cell
    if (
        sum(c["values"] for c in result.values()) != summary.get("values")
        or sum(c["values"] for key, c in result.items() if key[3])
        != summary.get("chain_values")
        or summary.get("nonchain_values", -1) + summary.get("chain_values", -1)
        != summary.get("values")
    ):
        raise RuntimeError("Power value plan totals differ")
    return result


def validate_accuracy(accuracy, summary, references, limits):
    plan = planned_cells(summary)
    if (
        not isinstance(accuracy, dict)
        or type(accuracy.get("failures")) is not int
        or accuracy["failures"] != 0
        or not isinstance(accuracy.get("cells"), list)
    ):
        raise RuntimeError("Power accuracy reports failures")
    seen = set()
    counters = (
        "values",
        "failures",
        "classification_failures",
        "numerical_failures",
        "exact_special_failures",
        "unreferenced_results",
        "finite_pairs",
        "ideal_finite_pairs",
        "normal_subnormal_transitions",
        "sign_mismatches",
    )
    for cell in accuracy["cells"]:
        key = cell_key(cell)
        if key in seen or key not in plan:
            raise RuntimeError("Power observed cell is duplicate or unplanned")
        seen.add(key)
        dtype, _, schedule, _chain = key
        fi, expected = DTYPES.index(dtype), plan[key]
        c = cell.get("accuracy")
        if (
            not isinstance(c, dict)
            or type(c.get("values")) is not int
            or c["values"] != expected["values"]
            or any(
                type(c.get(k)) is not int or not 0 <= c[k] <= c["values"]
                for k in counters
            )
        ):
            raise RuntimeError("Power observation counts differ")
        if any(
            c[k]
            for k in (
                "failures",
                "classification_failures",
                "numerical_failures",
                "exact_special_failures",
                "unreferenced_results",
                "sign_mismatches",
            )
        ):
            raise RuntimeError("Power observed cell reports failures")
        if (
            not vector(c.get("observed_classes"), 5)
            or sum(c["observed_classes"]) != c["values"]
            or not vector(c.get("unreferenced_observed_classes"), 5)
            or any(c["unreferenced_observed_classes"])
        ):
            raise RuntimeError("Power observed classes differ")
        for matrix_key, reference_key, pairs_key in (
            ("transitions", "reference_classes", "finite_pairs"),
            ("ideal_transitions", "ideal_classes", "ideal_finite_pairs"),
        ):
            matrix = c.get(matrix_key)
            if (
                not isinstance(matrix, list)
                or len(matrix) != 5
                or any(not vector(row, 5) for row in matrix)
            ):
                raise RuntimeError("Power classification matrix is invalid")
            if [sum(row[i] for row in matrix) for i in range(5)] != c[
                "observed_classes"
            ]:
                raise RuntimeError("Power classification columns differ")
            # Every permitted closure of the frozen Power chains retains its
            # staged and ideal reference classes, so chains bind these too.
            if [sum(row) for row in matrix] != expected[reference_key]:
                raise RuntimeError(
                    "Power reference populations differ from the frozen plan"
                )
            finite_pairs = sum(matrix[r][v] for r in (1, 2) for v in (1, 2))
            if finite_pairs != c[pairs_key]:
                raise RuntimeError("Power finite-pair observations differ")
        matrix = c["transitions"]
        allowed = ({0}, {1, 2}, {1, 2}, {3}, {4})
        if any(
            n
            for row, columns in zip(matrix, allowed)
            for actual, n in enumerate(row)
            if actual not in columns
        ):
            raise RuntimeError("Power classification contradicts qualification")
        if c["normal_subnormal_transitions"] != matrix[1][2] + matrix[2][1]:
            raise RuntimeError("Power boundary transition diagnostics differ")
        if (
            schedule == "mathematical"
            and fi < 3
            and any(
                c[staged] != c[ideal]
                for staged, ideal in (
                    ("transitions", "ideal_transitions"),
                    ("finite_pairs", "ideal_finite_pairs"),
                    ("histogram", "ideal_histogram"),
                    ("max_ulp", "max_ideal_ulp"),
                )
            )
        ):
            raise RuntimeError("Power identical references have contradictory metrics")
        for index, (hist_key, count_key, max_key, witness_key) in enumerate(
            (
                ("histogram", "finite_pairs", "max_ulp", "worst"),
                (
                    "ideal_histogram",
                    "ideal_finite_pairs",
                    "max_ideal_ulp",
                    "worst_ideal",
                ),
            )
        ):
            counts, maximum, count = c.get(hist_key), c.get(max_key), c[count_key]
            if not vector(counts, 7) or sum(counts) != count:
                raise RuntimeError("Power histogram observations differ")
            witness = c.get(witness_key)
            if count == 0:
                if maximum is not None or witness is not None:
                    raise RuntimeError(
                        "Power empty metric contains a maximum or witness"
                    )
                continue
            if (
                type(maximum) is not int
                or not 0 <= maximum <= limits[dtype, schedule][index]
                or not counts[bucket(maximum)]
                or any(counts[bucket(maximum) + 1 :])
            ):
                raise RuntimeError(
                    "Power maximum contradicts qualification or histogram"
                )
            if not isinstance(witness, dict) or witness.get("schedule") != schedule:
                raise RuntimeError("Power witness schedule differs")
            bits = {}
            for name in ("lhs", "rhs", "actual", "staged", "ideal"):
                value = witness.get(name)
                if (
                    not isinstance(value, str)
                    or not re.fullmatch(r"0x[0-9a-f]{16}", value)
                    or int(value, 16) >= 1 << (8 * WIDTHS[fi])
                ):
                    raise RuntimeError("Power witness bits are invalid")
                bits[name] = int(value, 16)
            row = references.get((dtype, schedule, bits["lhs"], bits["rhs"]))
            if (
                row is None
                or (bits["staged"], bits["ideal"]) != (row.staged, row.ideal)
                or not passing(fi, schedule, row, bits["actual"])
            ):
                raise RuntimeError("Power witness has no passing immutable reference")
            actual_class = classify(fi, bits["actual"])
            if (
                not c["transitions"][classify(fi, row.staged)][actual_class]
                or not c["ideal_transitions"][classify(fi, row.ideal)][actual_class]
            ):
                raise RuntimeError(
                    "Power witness does not occur in its classification matrices"
                )
            for reference, pairs, maximum_key, histogram_key in (
                (row.staged, "finite_pairs", "max_ulp", "histogram"),
                (row.ideal, "ideal_finite_pairs", "max_ideal_ulp", "ideal_histogram"),
            ):
                witness_distance = distance(fi, bits["actual"], reference)
                if witness_distance is not None and (
                    not c[pairs]
                    or c[maximum_key] is None
                    or witness_distance > c[maximum_key]
                    or not c[histogram_key][bucket(witness_distance)]
                ):
                    raise RuntimeError("Power witness contradicts an error metric")
            ref = row.staged if index == 0 else row.ideal
            if distance(fi, bits["actual"], ref) != maximum:
                raise RuntimeError("Power witness does not demonstrate its maximum")
    if seen != set(plan):
        raise RuntimeError("Power accuracy omits planned cells")
    return accuracy


def validate(
    text,
    expected,
    manifest,
    references,
    limits,
    *,
    execution_os,
    backend,
    fusion,
    scope,
):
    prefix = (
        f"BURN_TYPED_POWER_OK os={execution_os} backend={backend} fusion={fusion} corpus={manifest['corpus_version']} "
        f"policy_sha256={manifest['policy_sha256']} fixture_sha256={manifest['fixture_sha256']} qualification_passed=true totals="
    )
    pattern = re.compile(
        (r"(?:\[[^\]\r\n]*\]\s*)*" if execution_os == "d3os" else "")
        + "("
        + re.escape(prefix)
        + ".*)"
    )
    lines = [
        m.group(1)
        for line in text.splitlines()
        if (m := pattern.fullmatch(re.sub(r"\x1b\[[0-9;]*m", "", line)))
    ]
    if len(lines) != 1:
        raise RuntimeError("Power requires one terminal qualification marker")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate Power qualification JSON field")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite Power qualification JSON constant: {value}")

    decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid)
    totals, end = decoder.raw_decode(lines[0][len(prefix) :])
    remaining = lines[0][len(prefix) + end :]
    if json.dumps(totals, sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ) or not remaining.startswith(" accuracy="):
        raise RuntimeError("Power work plan differs from the frozen corpus")
    accuracy, end = decoder.raw_decode(remaining[len(" accuracy=") :])
    remaining = remaining[len(" accuracy=") + end :]
    footer = (
        " reference=directed_mpfr_4.2.2_stored_inputs_f64_2steps_f32_4steps_half_f32_neighbors4"
        " wrapper=stored_rne_schedule scalar=original_variants_public_float_f64 tensor_powi=int_into_float_then_pow"
        " nan=class_identity_alias_exact_bytes chain=materialized_each_primitive "
        + scope
    )
    if remaining != footer:
        raise RuntimeError("Power numerical policy or scope differs")
    return validate_accuracy(accuracy, expected, references, limits)
