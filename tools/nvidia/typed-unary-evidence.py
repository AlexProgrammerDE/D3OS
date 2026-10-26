"""Validate observed unary math qualification independently of GPU execution."""

import hashlib
import json
import re
from pathlib import Path

DTYPES = ("f64", "f32", "flex32", "f16", "bf16")
OPERATIONS = ("log1p", "sinh", "cosh", "tan")
BUDGETS = ((1, 2, 1, 2), (1, 3, 2, 4), (1, 3, 2, 4))


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


def classification(bits, sign, exponent):
    magnitude = bits & (sign - 1)
    if magnitude == 0:
        return 0
    if magnitude < exponent & -exponent:
        return 1
    if magnitude < exponent:
        return 2
    return 3 if magnitude == exponent else 4


def ordered(bits, sign):
    return sign - (bits & (sign - 1)) if bits & sign else sign + bits


def load_bounds(manifest_path):
    """Derive necessary aggregate limits from the exact immutable row bytes."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    binary = (
        manifest_path.parents[2]
        / "os/library/ai/src/validation/typed_unary_fixtures.bin"
    )
    data = binary.read_bytes()
    if (
        len(data) != manifest["fixture_bytes"]
        or hashlib.sha256(data).hexdigest() != manifest["fixture_sha256"]
    ):
        raise RuntimeError("typed unary reference bytes differ from the manifest")
    bounds, offset = [], 0
    for fi, dtype in enumerate(DTYPES):
        width = (8, 4, 4, 2, 2)[fi]
        sign = 1 << (8 * width - 1)
        exponent = (0x7FF0000000000000, 0x7F800000, 0x7F800000, 0x7C00, 0x7F80)[fi]

        for oi, op in enumerate(OPERATIONS):
            limit = {
                "staged": 0,
                "ideal": 0,
                "class_changes": False,
                "finite_changes": False,
                "rows": {},
                "staged_classes": [0] * 5,
                "ideal_classes": [0] * 5,
            }
            count = manifest["counts"][dtype][op]["all"]
            for _ in range(count):
                end = offset + 5 * width + 4
                if end > len(data):
                    raise RuntimeError("truncated typed unary reference row")
                input_bits, ideal, staged, low, high = [
                    int.from_bytes(
                        data[offset + k * width : offset + (k + 1) * width], "little"
                    )
                    for k in range(5)
                ]
                offset = end
                sc, ic = classification(staged, sign, exponent), classification(
                    ideal, sign, exponent
                )
                limit["rows"][input_bits] = (ideal, staged, low, high)
                limit["staged_classes"][sc] += 1
                limit["ideal_classes"][ic] += 1
                limit["class_changes"] |= sc != ic
                limit["finite_changes"] |= (sc in (1, 2)) != (ic in (1, 2))
                if fi >= 3:
                    for actual in (low, high):
                        if classification(actual, sign, exponent) in (1, 2):
                            for key, reference, kind in (
                                ("staged", staged, sc),
                                ("ideal", ideal, ic),
                            ):
                                if kind in (1, 2):
                                    limit[key] = max(
                                        limit[key],
                                        abs(
                                            ordered(reference, sign)
                                            - ordered(actual, sign)
                                        ),
                                    )
            if fi < 3:
                limit.update(staged=BUDGETS[fi][oi], ideal=BUDGETS[fi][oi])
            bounds.append(limit)
    if offset != len(data):
        raise RuntimeError("typed unary reference row inventory differs")
    return manifest, bounds


def validate_plan(expected):
    values = expected.get("values")
    cells = expected.get("expected_value_counts")
    if (
        type(expected.get("schema_version")) is not int
        or expected["schema_version"] != 3
        or type(values) is not int
        or values < 0
        or not isinstance(cells, list)
        or len(cells) != 5
        or any(not vector(row, 4) for row in cells)
        or sum(map(sum, cells)) != values
    ):
        raise RuntimeError("typed unary per-operation work plan is invalid")
    return cells


def summary_reference_plan(expected):
    cells = validate_plan(expected)
    for key in ("expected_staged_class_counts", "expected_ideal_class_counts"):
        values = expected.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 5
            or any(
                not isinstance(row, list)
                or len(row) != 4
                or any(not vector(cell, 5) for cell in row)
                for row in values
            )
        ):
            raise RuntimeError("typed unary reference class plan shape differs")
    chains = expected.get("chain_value_counts")
    if (
        not isinstance(chains, list)
        or len(chains) != 5
        or any(not vector(row, 4) for row in chains)
    ):
        raise RuntimeError("typed unary chain plan shape differs")
    result = []
    for fi in range(5):
        for oi in range(4):
            try:
                staged = expected["expected_staged_class_counts"][fi][oi]
                ideal = expected["expected_ideal_class_counts"][fi][oi]
                slack = expected["chain_value_counts"][fi][oi]
            except (KeyError, IndexError, TypeError) as error:
                raise RuntimeError(
                    "typed unary reference population plan is missing"
                ) from error
            if (
                not vector(staged, 5)
                or not vector(ideal, 5)
                or type(slack) is not int
                or slack < 0
                or sum(staged) + slack != cells[fi][oi]
                or sum(ideal) + slack != cells[fi][oi]
            ):
                raise RuntimeError("typed unary reference population plan differs")
            result.append({"staged": staged, "ideal": ideal, "slack": slack})
    return result


def raw_reference_plan(bounds, expected_cells):
    # Four complete immutable-corpus passes plus the separately audited raw,
    # fresh, Native and public geometry routes. Geometry is bounded here;
    # the raw harness checks every one of its references individually.
    result = []
    for index, limit in enumerate(bounds):
        fi, oi = divmod(index, 4)
        minimum = 4 * len(limit["rows"])
        if expected_cells[fi][oi] != minimum + 32036:
            raise RuntimeError("typed unary raw geometry inventory differs")
        result.append(
            {
                "staged": [4 * n for n in limit["staged_classes"]],
                "ideal": [4 * n for n in limit["ideal_classes"]],
                "slack": 32036,
            }
        )
    return result


def validate_accuracy(
    accuracy, expected_cells, expected_values, bounds, *, reference_plan
):
    if (
        type(expected_values) is not int
        or expected_values < 0
        or not isinstance(expected_cells, list)
        or len(expected_cells) != 5
        or any(not vector(row, 4) for row in expected_cells)
        or sum(map(sum, expected_cells)) != expected_values
        or len(bounds) != 20
        or len(reference_plan) != 20
    ):
        raise RuntimeError("typed unary observation plan is invalid")
    if (
        not isinstance(accuracy, dict)
        or type(accuracy.get("failures")) is not int
        or accuracy["failures"] != 0
    ):
        raise RuntimeError("typed unary reports qualification failures")
    cells = accuracy.get("cells")
    if not isinstance(cells, list) or len(cells) != 20:
        raise RuntimeError("typed unary requires twenty accuracy cells")
    failed = (
        "failures",
        "classification_failures",
        "numerical_failures",
        "exact_special_failures",
        "half_envelope_failures",
        "codomain_failures",
        "unreferenced_results",
        "sign_mismatches",
        "nonzero_to_zero",
    )
    counters = (
        *failed,
        "values",
        "passing_finite_pairs",
        "finite_pairs",
        "ideal_finite_pairs",
        "subnormal_preserved",
        "nan_payload_differences",
        "quiet_nan_outputs",
        "negative_nan_outputs",
        "reference_class_disagreements",
        "reference_nonzero_finite_disagreements",
    )
    for index, cell in enumerate(cells):
        fi, oi = divmod(index, 4)
        if not isinstance(cell, dict) or (cell.get("dtype"), cell.get("op")) != (
            DTYPES[fi],
            OPERATIONS[oi],
        ):
            raise RuntimeError("typed unary accuracy cell identity differs")
        n = cell.get("values")
        if (
            type(n) is not int
            or n != expected_cells[fi][oi]
            or any(
                type(cell.get(k)) is not int or not 0 <= cell[k] <= n for k in counters
            )
        ):
            raise RuntimeError("typed unary accuracy counts differ")
        if any(cell[k] for k in failed):
            raise RuntimeError("typed unary cell reports qualification failures")
        if (
            not vector(cell.get("observed_classes"), 5)
            or sum(cell["observed_classes"]) != n
        ):
            raise RuntimeError("typed unary observed classes differ")
        if not vector(cell.get("unreferenced_observed_classes"), 5) or any(
            cell["unreferenced_observed_classes"]
        ):
            raise RuntimeError("typed unary contains unreferenced observations")
        for key, pair_key in (
            ("transitions", "finite_pairs"),
            ("ideal_transitions", "ideal_finite_pairs"),
        ):
            matrix = cell.get(key)
            if (
                not isinstance(matrix, list)
                or len(matrix) != 5
                or any(not vector(row, 5) for row in matrix)
            ):
                raise RuntimeError("typed unary classification matrix is invalid")
            if [sum(row[c] for row in matrix) for c in range(5)] != cell[
                "observed_classes"
            ]:
                raise RuntimeError("typed unary classification columns differ")
            finite = sum(matrix[r][c] for r in (1, 2) for c in (1, 2))
            if cell[pair_key] != finite:
                raise RuntimeError("typed unary finite-pair observations differ")
        population = reference_plan[index]
        for key, minimum_key in (
            ("transitions", "staged"),
            ("ideal_transitions", "ideal"),
        ):
            matrix = cell[key]
            totals = [sum(row) for row in matrix]
            minimum = population[minimum_key]
            slack = population["slack"]
            if (
                not vector(minimum, 5)
                or type(slack) is not int
                or slack < 0
                or sum(minimum) + slack != n
                or any(
                    not lower <= actual <= lower + slack
                    for actual, lower in zip(totals, minimum)
                )
            ):
                raise RuntimeError(
                    "typed unary reference populations contradict the frozen work plan"
                )
        staged, ideal = cell["transitions"], cell["ideal_transitions"]
        allowed = ({0}, {1, 2}, {1, 2}, {3}, {4})
        if any(
            count
            for row, columns in zip(staged, allowed)
            for c, count in enumerate(row)
            if c not in columns
        ):
            raise RuntimeError(
                "typed unary classification contradicts passing evidence"
            )
        limit = bounds[index]
        changes, finite_changes = (
            cell["reference_class_disagreements"],
            cell["reference_nonzero_finite_disagreements"],
        )
        matrix_difference = sum(
            abs(a - b)
            for row_a, row_b in zip(staged, ideal)
            for a, b in zip(row_a, row_b)
        )
        if (
            finite_changes > changes
            or matrix_difference > 2 * changes
            or abs(cell["finite_pairs"] - cell["ideal_finite_pairs"]) > finite_changes
            or (not limit["class_changes"] and (changes or staged != ideal))
            or (not limit["finite_changes"] and finite_changes)
        ):
            raise RuntimeError("typed unary ideal/staged reference accounting differs")
        if (
            cell["subnormal_preserved"] != staged[1][1]
            or cell["passing_finite_pairs"] != cell["finite_pairs"]
        ):
            raise RuntimeError("typed unary passing counters differ")
        if any(
            cell[k] > cell["observed_classes"][4]
            for k in (
                "nan_payload_differences",
                "quiet_nan_outputs",
                "negative_nan_outputs",
            )
        ):
            raise RuntimeError("typed unary NaN diagnostics differ")
        for key, count_key, maximum_key, maximum_limit in (
            ("histogram", "finite_pairs", "max_ulp", limit["staged"]),
            (
                "passing_histogram",
                "passing_finite_pairs",
                "passing_max_ulp",
                limit["staged"],
            ),
            ("ideal_histogram", "ideal_finite_pairs", "max_ideal_ulp", limit["ideal"]),
        ):
            hist, count, maximum = cell.get(key), cell[count_key], cell.get(maximum_key)
            if not vector(hist, 7) or sum(hist) != count:
                raise RuntimeError("typed unary histogram counts differ")
            if count == 0:
                if maximum is not None:
                    raise RuntimeError("typed unary empty histogram has a maximum")
            elif (
                type(maximum) is not int
                or not 0 <= maximum <= maximum_limit
                or not hist[bucket(maximum)]
                or any(hist[bucket(maximum) + 1 :])
            ):
                raise RuntimeError(
                    "typed unary observed maximum contradicts qualification"
                )
        if (
            cell["histogram"] != cell["passing_histogram"]
            or cell["max_ulp"] != cell["passing_max_ulp"]
        ):
            raise RuntimeError("typed unary passing metrics contradict observations")
        if fi < 3 and (
            cell["ideal_histogram"] != cell["histogram"]
            or cell["max_ideal_ulp"] != cell["max_ulp"]
        ):
            raise RuntimeError("typed unary wide-format reference metrics differ")
        width = (64, 32, 32, 16, 16)[fi]
        exponent = (0x7FF0000000000000, 0x7F800000, 0x7F800000, 0x7C00, 0x7F80)[fi]
        for prefix, count_key in (
            ("worst", "finite_pairs"),
            ("worst_ideal", "ideal_finite_pairs"),
        ):
            for suffix in ("input_bits", "actual_bits"):
                bits = cell.get(f"{prefix}_{suffix}")
                if cell[count_key] == 0:
                    if bits is not None:
                        raise RuntimeError(
                            "typed unary empty metrics contain a witness"
                        )
                elif (
                    not isinstance(bits, str)
                    or not re.fullmatch(r"[0-9a-f]{16}", bits)
                    or int(bits, 16) >= 1 << width
                ):
                    raise RuntimeError("typed unary witness bits are invalid")
            if cell[count_key]:
                actual = int(cell[f"{prefix}_actual_bits"], 16)
                if not 0 < actual & ((1 << (width - 1)) - 1) < exponent:
                    raise RuntimeError(
                        "typed unary finite witness is not finite and nonzero"
                    )
                if oi == 2 and (
                    actual >> (width - 1)
                    or actual
                    < (0x3FF0000000000000, 0x3F800000, 0x3F800000, 0x3C00, 0x3F80)[fi]
                ):
                    raise RuntimeError("typed unary cosh witness violates its codomain")
                source = int(cell[f"{prefix}_input_bits"], 16)
                row = limit["rows"].get(source)
                if row is None:
                    raise RuntimeError(
                        "typed unary worst witness has no immutable reference"
                    )
                direct, wanted, low, high = row
                sign = 1 << (width - 1)
                reference = direct if prefix == "worst_ideal" else wanted
                maximum = cell[
                    "max_ideal_ulp" if prefix == "worst_ideal" else "max_ulp"
                ]
                if (
                    classification(reference, sign, exponent) not in (1, 2)
                    or abs(ordered(actual, sign) - ordered(reference, sign)) != maximum
                ):
                    raise RuntimeError(
                        "typed unary witness does not demonstrate its reported maximum"
                    )
                one = (0x3FF0000000000000, 0x3F800000, 0x3F800000, 0x3C00, 0x3F80)[fi]
                special = classification(source, sign, exponent) in (0, 3, 4) or (
                    oi == 0 and source & sign and source & (sign - 1) >= one
                )
                staged_steps = abs(ordered(actual, sign) - ordered(wanted, sign))
                if (
                    classification(wanted, sign, exponent) not in (1, 2)
                    or (actual ^ wanted) & sign
                    or (special and actual != wanted)
                    or (fi < 3 and staged_steps > BUDGETS[fi][oi])
                    or (
                        fi >= 3
                        and not ordered(low, sign)
                        <= ordered(actual, sign)
                        <= ordered(high, sign)
                    )
                ):
                    raise RuntimeError(
                        "typed unary worst witness fails the numerical qualification"
                    )
    return accuracy


def validate(text, expected, fixtures, bounds, *, execution_os, backend, fusion, scope):
    cells = validate_plan(expected)
    prefix = f"BURN_TYPED_UNARY_OK os={execution_os} backend={backend} fusion={fusion} corpus={fixtures['version']} totals="
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
        raise RuntimeError("typed unary requires one terminal qualification marker")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate typed unary JSON field")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite typed unary JSON constant: {value}")

    decoder = json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid)
    totals, end = decoder.raw_decode(lines[0][len(prefix) :])
    remaining = lines[0][len(prefix) + end :]
    if json.dumps(totals, sort_keys=True) != json.dumps(
        expected, sort_keys=True
    ) or not remaining.startswith(" accuracy="):
        raise RuntimeError("typed unary totals differ from the frozen plan")
    accuracy, end = decoder.raw_decode(remaining[len(" accuracy=") :])
    remaining = remaining[len(" accuracy=") + end :]
    footer = (
        f" policy_sha256={fixtures['policy_sha256']} fixture_sha256={fixtures['fixture_sha256']}"
        " numerical_policy=project_qualification_targets staged_half=f32_neighbor_envelope"
        " no_ftz=strict preserved_inputs=exact_bytes chain=materialized_known_refs"
        " poison=retained_allocation scope=strict_native " + scope
    )
    if remaining != footer:
        raise RuntimeError("typed unary policy, fixture, or scope differs")
    return validate_accuracy(
        accuracy,
        cells,
        expected["values"],
        bounds,
        reference_plan=summary_reference_plan(expected),
    )
