"""Strict logical indexed-accumulation evidence checks, without device execution.

Trusted inputs come from device-free CLI inventory, bound again to the immutable
reference bytes here. Runtime assertions about retained physical storage still
need parent-approved source/execution provenance and separate Raw proof.
"""

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tools/indexed-accumulation-reference"
LINUX_BACKEND = "dispatch<fusion<cubecl<cuda>>>"
NATIVE_BACKEND = "dispatch<d3os-native-nvidia>"
DTYPES = (
    "F64",
    "F32",
    "Flex32",
    "F16",
    "BF16",
    "I64",
    "I32",
    "I16",
    "I8",
    "U64",
    "U32",
    "U16",
    "U8",
)
CHECKS = (
    "InputStorage",
    "InputView",
    "RetainedStorage",
    "RetainedView",
    "Output",
    "PublicStage",
    "SemanticError",
    "NoPublishedOutput",
    "Recovery",
)
SOURCE_PINS = {
    "os/library/ai/src/validation/indexed_accumulation.rs": "f67acd617825ac9532512e86036825da655bfb5a4e63940a0f10a905a667b0bd",
    "os/library/ai/src/validation/indexed_accumulation/fixtures.rs": "732da3045f38a42cada3c419b7034b6553d839c19d54b2ddd393c107c515cc0f",
    "os/library/ai/tests/indexed-accumulation-reference.rs": "94441005ccc2c0af36099c1aa16b9711197ecac0856f3c02f48f22d33a40380b",
    "tools/indexed-accumulation-reference/boundaries.py": "28929a6446e724e0484642f5eabae544c2c3a4fbad6a0a253d14154206b5b573",
    "tools/indexed-accumulation-reference/core.py": "cbc9ec46e374788e9598692224be006f4cacc58c4ce5569a4a5e7a777b40a1de",
    "tools/indexed-accumulation-reference/fixtures.json": "bb5d4a2bbc47f5b014424557dd2a5a15cb4371f9fd2ae52cf0230e76ccf7c588",
    "tools/indexed-accumulation-reference/generate.py": "972f0f4e304499466468ce161cf6600446c14b866e78e4bb8acc7bd3fa13f9dc",
    "tools/indexed-accumulation-reference/generate_rust.py": "ef1c07d59283254e47e363caca47fa0043ab73cb51411e083dbe019a9d154f09",
    "tools/indexed-accumulation-reference/ieee.py": "4bfc73465bed27477f03acdd0d122370a254e039ee74a58882eba41b84862b12",
    "tools/indexed-accumulation-reference/main_cases.py": "96d3b37ab5c90157aaf7e4fb810487635654c496ef3366d42b418292164ed82e",
    "tools/indexed-accumulation-reference/policy.json": "d02e3c5b7f86251e5c60361be4be87e2ac9e31b37353a4b306a3cb6818ac4a9d",
    "tools/indexed-accumulation-reference/provenance.json": "ddcc28397f3271550c8ecd23a6f70cbce46ccb4ff9d906f062ffaccbf64d035d",
    "tools/indexed-accumulation-reference/specs.json": "481ce345ce47aef2cf830385ad466a5980cad8679352109c61949cd33b894c9d",
    "tools/indexed-accumulation-reference/validation.json": "890213bde9054330b1dd642e56424abc78e2cc940e72b0cb2d84745715b9cf62",
    "tools/mnist-benchmark/src/bin/indexed_accumulation.rs": "c6958321835a5815c088800639a5e694d7b4627302b7c3a232fc0ab4e7c32578",
}
# Primary sources for the exact Linux name: Dispatch -> Fusion<CubeBackend>
# -> Client::name -> CUDA ServerUtilities("cuda"). Native also passes through
# Dispatch::name and wraps Native::name. No Device query is needed.
BACKEND_SOURCE_PINS = {
    "os/library/burn-native/src/backend/mod.rs": "0e0074107aad7d0ecd28b46804ddc0016a031678cf6a5d7ca31af5ceeff9af77",
    "third_party/burn-dispatch/src/backend.rs": "39ece66164c53c9c85b17bc45aea101c8a6ce70df8a2bba39fb9574255d4dedf",
    "tools/mnist-benchmark/Cargo.toml": "d7386d83c3ab526c6596fbd61d0586b9a3fb153535c44a03e64a5e89178476b4",
}
BACKEND_LIBRARY_PINS = {
    "burn-cubecl-0.22.1/src/backend.rs": "e596edf60beecb0c3ccdbabe39ac9b5a2442f1f998d97157f598f4f3900a15b5",
    "burn-cubecl-0.22.1/src/lib.rs": "deffbba923c74452d767abfe0a727c242295240d52eeab0d3cc404f8e75bad3c",
    "burn-fusion-0.22.1/src/backend.rs": "adf2c767b962be7ba67563c3c51502ddfbbd0ed3ecee5425338802b5c6260e64",
    "cubecl-runtime-0.11.3/src/client.rs": "515b1a10abe323db4aca8d37051a63e9c7786a29123c6aa971f1f17c32e8b781",
    "cubecl-cuda-0.11.3/src/runtime.rs": "25917ebc45fed27a81127e4feb6027e4ea0b1b2ffdc4b458629d50f846c52de3",
    "burn-std-0.22.1/src/device_settings.rs": "cbccc9a43eaf1b297dfb278e25d7c68a4ee6f4150564b915157f2996a68f9ad6",
}


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
    number = float(value)
    _require(math.isfinite(number), "nonfinite JSON exponent")
    return number


def strict_json(text):
    """Decode one complete JSON value with duplicate/nonfinite rejection."""
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
    _require(type(actual) is type(expected), f"{label}: wrong type")
    if type(expected) is dict:
        _require(actual.keys() == expected.keys(), f"{label}: wrong fields")
        for key, value in expected.items():
            _same(actual[key], value, f"{label}.{key}")
    elif type(expected) is list:
        _require(len(actual) == len(expected), f"{label}: wrong count")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            _same(a, b, f"{label}[{index}]")
    else:
        _require(actual == expected, f"{label}: differs from immutable reference")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def source_paths():
    """Return the reviewed Shared closure and primary backend-name sources."""
    paths = {name: ROOT / name for name in SOURCE_PINS | BACKEND_SOURCE_PINS}
    registry = Path.home() / ".cargo/registry/src"
    for name in BACKEND_LIBRARY_PINS:
        matches = sorted(registry.glob(f"*/{name}"))
        _require(len(matches) == 1, f"missing/ambiguous backend primary source: {name}")
        paths[name] = matches[0]
    return paths


def validate_backend_sources():
    """Bind exact Linux/Native names to their unchanged primary source bytes."""
    paths = source_paths()
    pins = BACKEND_SOURCE_PINS | BACKEND_LIBRARY_PINS
    for name, digest in pins.items():
        _same(_sha(paths[name].read_bytes()), digest, f"backend source {name}")
    return {name: paths[name] for name in pins}


def validate_sources():
    """Check exact reviewed bytes; execution/OS provenance is still external."""
    paths = source_paths()
    for name, digest in (
        SOURCE_PINS | BACKEND_SOURCE_PINS | BACKEND_LIBRARY_PINS
    ).items():
        _require(
            _sha(paths[name].read_bytes()) == digest, f"source pin differs: {name}"
        )
    return paths


def _references():
    # Recheck bytes on every public validation, including after a previous pass.
    provenance_bytes = (REFERENCE / "provenance.json").read_bytes()
    _require(
        _sha(provenance_bytes)
        == SOURCE_PINS["tools/indexed-accumulation-reference/provenance.json"],
        "provenance pin differs",
    )
    provenance = strict_json(provenance_bytes.decode())
    for name, digest in provenance["baseline_sha256"].items():
        _require(
            "/" not in name and _sha((REFERENCE / name).read_bytes()) == digest,
            f"reference pin differs: {name}",
        )
    return (
        strict_json((REFERENCE / "fixtures.json").read_text()),
        strict_json((REFERENCE / "policy.json").read_text()),
        provenance,
    )


def _policy(row):
    family = row["family"]
    if family in ("legacy_guard", "legacy_ND_guard"):
        return "LegacyExactFiniteGuard"
    if family == "accepted_assign_guard":
        return "AcceptedAssignBitwiseGuard"
    if family == "public_tracked_gradient":
        return "PublicLinearAdjointExact"
    if family == "exact_index_guard":
        return "NativeRecoverableSemanticError"
    return (
        "NativeSerialC3"
        if row["dtype"] in ("F16", "BF16", "Flex32", "F64")
        else "DeclaredWidthInteger"
    )


def _key(identifier, check, subject):
    return {"id": identifier, "check": check, "subject": subject}


def _key_order(key):
    return key["id"], CHECKS.index(key["check"]), key["subject"]


def _keys(row):
    identifier = row["id"]
    keys = []
    for role, source in row["sources"].items():
        for check, subject in (
            ("InputStorage", source["storage_id"]),
            ("RetainedStorage", source["storage_id"]),
            ("InputView", role),
            ("RetainedView", role),
        ):
            key = _key(identifier, check, subject)
            if key not in keys:
                keys.append(key)
    kind = row["expected"]["kind"]
    if kind == "value":
        keys.append(_key(identifier, "Output", "result"))
    elif kind == "error":
        keys.extend(
            _key(identifier, check, subject)
            for check, subject in (
                ("SemanticError", "result"),
                ("NoPublishedOutput", "result"),
                ("Recovery", "same_context_valid_operation"),
            )
        )
    else:
        names = ["forward", "root", "original_leaf_gradient"]
        for branch in range(1, row["public_metadata"]["graph"]["branches"] + 1):
            names.extend((f"branch_{branch}_product", f"branch_{branch}_sum"))
        keys.extend(_key(identifier, "PublicStage", name) for name in names)
    return sorted(keys, key=_key_order)


def reference_selection(scope, dtype=None):
    """Derive pure expected inventory independently of the observed CLI report."""
    fixtures, policies, provenance = _references()
    _require(
        scope in ("LinuxCommon", "NativeValues", "NativeErrors", "ModelReference"),
        "unknown scope",
    )
    _require(dtype is None or type(dtype) is str and dtype in DTYPES, "unknown dtype")
    rows = [
        r
        for r in fixtures["main"]
        if (dtype is None or r["dtype"] == dtype)
        and (
            scope == "LinuxCommon"
            and r["dtype"] != "F64"
            and r["expected"]["kind"] != "error"
            or scope == "NativeValues"
            and r["expected"]["kind"] != "error"
            or scope == "NativeErrors"
            and r["expected"]["kind"] == "error"
        )
    ]
    models = [
        {
            "key": {
                "template": template["id"],
                "content_sha256": _sha(_canonical(attachment)),
            },
            "evidence": "MODEL",
            "attachment": attachment,
        }
        for template in fixtures["symbolic"]
        for attachment in template["attachments"]
        if scope == "ModelReference"
        and (dtype is None or attachment.get("dtype") == dtype)
    ]
    _require(rows or models, "empty selection")
    keys = [key for row in rows for key in _keys(row)]
    selected = {r["id"] for r in rows}
    excluded = [
        {
            "id": r["id"],
            "reason": "Native_error_scope_only"
            if r["expected"]["kind"] == "error"
            else "public_CUDA_F64_source_declined"
            if scope == "LinuxCommon" and r["dtype"] == "F64"
            else "outside_selected_scope_or_dtype",
        }
        for r in fixtures["main"]
        if r["id"] not in selected
    ]
    population = {
        "corpus_version": 1,
        "scope": scope,
        "dtype_filter": dtype,
        "fixtures_sha256": provenance["baseline_sha256"]["fixtures.json"],
        "numerical_freeze_sha256": provenance["numerical_freeze_sha256"],
        "FROZEN": False,
        "source_binding_approved": False,
        "observation": "NOT_EXECUTED",
        "structured_policies": policies,
        "cases": [
            {
                "frozen_row": r,
                "row_sha256": _sha(_canonical(r)),
                "execution_policy": _policy(r),
            }
            for r in rows
        ],
        "models": models,
        "excluded": excluded,
        "required_observations": keys,
    }
    summary = {
        "scope": scope,
        "cases": len(rows),
        "values": sum(r["expected"]["kind"] == "value" for r in rows),
        "errors": sum(r["expected"]["kind"] == "error" for r in rows),
        "public_gradients": sum(r["expected"]["kind"] == "gradient" for r in rows),
        "templates": len({m["key"]["template"] for m in models}),
        "attachments": len(models),
        "class_patterns_MODEL": sum(
            m["attachment"].get("class_pattern", False) for m in models
        ),
        "source_roles": sum(len(r["sources"]) for r in rows),
        "source_storages": sum(k["check"] == "InputStorage" for k in keys),
        "observation_keys": len(keys),
        "device_initialized": False,
        "FROZEN": False,
        "source_binding_approved": False,
    }
    return summary, population


def validate_selection(expected_summary, population):
    """Bind both inventory exports to immutable selected membership and policies."""
    _require(type(population) is dict, "population must be an object")
    expected, independent = reference_selection(
        population.get("scope"), population.get("dtype_filter")
    )
    _same(expected_summary, expected, "summary")
    _same(population, independent, "population")
    return independent


def _width(dtype):
    return (
        16
        if dtype in ("F16", "BF16")
        else 32
        if dtype in ("F32", "Flex32")
        else int(dtype[1:])
    )


def _integer_bits(value, dtype):
    """Encode exact small integer metadata, without host floating arithmetic."""
    _require(type(value) is int, "public metadata value must be an integer")
    width = _width(dtype)
    if dtype[0] in "IU":
        bits = value % (1 << width)
    elif value == 0:
        bits = 0
    else:
        exponent_bits, fraction_bits = {
            "F16": (5, 10),
            "BF16": (8, 7),
            "F32": (8, 23),
            "Flex32": (8, 23),
            "F64": (11, 52),
        }[dtype]
        magnitude = abs(value)
        exponent = magnitude.bit_length() - 1
        shift = fraction_bits - exponent
        _require(
            shift >= 0 or magnitude % (1 << -shift) == 0,
            "metadata integer is not exactly representable",
        )
        mantissa = magnitude << shift if shift >= 0 else magnitude >> -shift
        bits = (
            (int(value < 0) << (width - 1))
            | ((exponent + (1 << (exponent_bits - 1)) - 1) << fraction_bits)
            | (mantissa - (1 << fraction_bits))
        )
    return f"0x{bits:0{width // 4}x}"


def _snapshot(dtype, shape, bits):
    _require(dtype in DTYPES, "unknown snapshot dtype")
    _require(
        type(shape) is list
        and all(type(n) is int and 0 <= n <= 2147483647 for n in shape)
        and len(shape) <= 8,
        "invalid snapshot shape",
    )
    _require(
        type(bits) is list and math.prod(shape) == len(bits),
        "snapshot shape/count differs",
    )
    _require(
        all(
            type(b) is str
            and re.fullmatch(r"0x[0-9a-f]{" + str(_width(dtype) // 4) + r"}", b)
            for b in bits
        ),
        "invalid snapshot bit width/encoding",
    )
    return {"dtype": dtype, "shape": shape, "bits": bits}


def _value_snapshot(value, dtype, bits="bits"):
    return _snapshot(value.get("dtype", dtype), value["shape"], value[bits])


def _source_snapshot(source):
    # Check affine original source geometry, rather than trusting an emitted view.
    from itertools import product

    addresses = [
        source["offset_elements"]
        + sum(
            x * stride
            for x, stride in zip(coordinate, source["strides_elements"], strict=True)
        )
        for coordinate in product(*(range(n) for n in source["shape"]))
    ]
    _same(addresses, source["logical_to_storage"], "original source view")
    return _snapshot(
        source["dtype"],
        source["shape"],
        [source["storage_bits"][address] for address in addresses],
    )


def _stage_plan(row):
    """Independent expected snapshots and exact operand identities in execution order."""
    identifier, dtype, expected = row["id"], row["dtype"], row["expected"]
    inputs = {
        role: (_key(identifier, "InputView", role), _source_snapshot(source))
        for role, source in row["sources"].items()
    }
    stages = []

    def add(check, name, snapshot, operands):
        key = _key(identifier, check, name)
        actual_operands = [
            {"key": k, "actual": s, "unavailable": None} for k, s in operands
        ]
        stages.append(
            {
                "key": key,
                "expected": snapshot,
                "actual": snapshot,
                "read_error": None,
                "actual_operands": actual_operands,
            }
        )
        return key, snapshot

    if expected["kind"] == "value":
        add(
            "Output",
            "result",
            _value_snapshot(expected, dtype),
            [inputs[n] for n in ("destination", "values", "indices")],
        )
    elif expected["kind"] == "error":
        add(
            "Recovery",
            "same_context_valid_operation",
            _snapshot(dtype, [2], [_integer_bits(n, dtype) for n in (2, 1)]),
            [],
        )
    else:
        metadata = row["public_metadata"]
        output, coefficient = metadata["output"], metadata["upstream_coefficient"]
        forward = _snapshot(
            dtype,
            output["shape"],
            [_integer_bits(n, dtype) for n in output["storage_values_flat"]],
        )
        _same(
            forward,
            _value_snapshot(expected["forward"], dtype, "storage_bits"),
            "reference forward",
        )
        _same(
            inputs["upstream_coefficient"][1],
            _snapshot(
                dtype,
                coefficient["shape"],
                [_integer_bits(n, dtype) for n in coefficient["storage_values_flat"]],
            ),
            "reference coefficient",
        )
        forward_operand = add(
            "PublicStage",
            "forward",
            forward,
            [inputs[n] for n in ("operand", "indices")],
        )
        products = [
            a * b
            for a, b in zip(
                output["storage_values_flat"],
                coefficient["storage_values_flat"],
                strict=True,
            )
        ]
        sums = []
        for branch in range(1, metadata["graph"]["branches"] + 1):
            product_operand = add(
                "PublicStage",
                f"branch_{branch}_product",
                _snapshot(
                    dtype, output["shape"], [_integer_bits(n, dtype) for n in products]
                ),
                [forward_operand, inputs["upstream_coefficient"]],
            )
            sums.append(
                add(
                    "PublicStage",
                    f"branch_{branch}_sum",
                    _snapshot(dtype, [1], [_integer_bits(sum(products), dtype)]),
                    [product_operand],
                )
            )
        root_snapshot = _snapshot(
            dtype, [1], [_integer_bits(sum(products) * len(sums), dtype)]
        )
        _same(root_snapshot, _value_snapshot(expected["root"], dtype), "reference root")
        root = add("PublicStage", "root", root_snapshot, sums)
        gradient = metadata["expected_gradient"]
        gradient_snapshot = _snapshot(
            dtype,
            gradient["shape"],
            [_integer_bits(n, dtype) for n in gradient["storage_values_flat"]],
        )
        _same(
            gradient_snapshot,
            _value_snapshot(
                expected["dense_original_leaf_gradient"], dtype, "storage_bits"
            ),
            "reference gradient",
        )
        add(
            "PublicStage",
            "original_leaf_gradient",
            gradient_snapshot,
            [inputs[n] for n in ("operand", "indices", "upstream_coefficient")]
            + [root],
        )
    return stages


def _semantic_event(row):
    expected = {
        "recoverable_index_range": "IndexRange",
        "recoverable_coordinate_bounds": "CoordinateBounds",
    }[row["expected"]["semantic_error"]]
    operation = "float_scatter" if row["dtype"][0] not in "IU" else "int_scatter"
    return {
        "id": row["id"],
        "expected": expected,
        "actual_error": f"An error happened during execution\nCaused by:\n  D3OS native Burn {operation}: invalid operation parameter",
        "output_metadata_valid": True,
        "accepted": True,
    }


def _expected_report(summary, population, backend):
    rows = [c["frozen_row"] for c in population["cases"]]
    model = summary["scope"] == "ModelReference"
    return {
        "corpus_version": 1,
        "scope": summary["scope"],
        "dtype_filter": population["dtype_filter"],
        "backend": backend,
        "completion": "Finished",
        "finalized": True,
        "ok": True,
        "FROZEN": False,
        "source_binding_approved": False,
        "numerical_freeze_sha256": population["numerical_freeze_sha256"],
        "fixtures_sha256": population["fixtures_sha256"],
        "device_attempted": summary["cases"],
        "device_passed": summary["cases"],
        "device_failed": 0,
        "model_attempted": summary["attachments"],
        "model_passed": summary["attachments"],
        "successful_observations": population["required_observations"],
        "failed_observations": [],
        "missing_observations": [],
        "failure_count": 0,
        "omitted_failure_details": 0,
        "failures": [],
        "stage_observations": [s for row in rows for s in _stage_plan(row)],
        "seeds": [
            {
                "id": row["id"],
                "status": "expected_only_source_backed",
                "expected": _value_snapshot(row["expected"]["seed"], row["dtype"]),
                "actual": None,
            }
            for row in rows
            if row["expected"]["kind"] == "gradient"
        ],
        "model_results": [
            {"key": m["key"], "passed": True, "evidence": "MODEL"}
            for m in sorted(
                population["models"],
                key=lambda m: (m["key"]["template"], m["key"]["content_sha256"]),
            )
        ],
        "synchronized": not model,
        "device_poisoned": False,
        "source_observation": "retained_clone_lineage_logical_reads_after_whole_graph",
        "physical_tails_ownership_residency_proven": False,
        "full_parity": False,
        "structured_policies": population["structured_policies"],
        "semantic_events": [
            _semantic_event(row) for row in rows if row["expected"]["kind"] == "error"
        ],
        "selected_rows": [
            {
                "id": c["frozen_row"]["id"],
                "row_sha256": c["row_sha256"],
                "policy": c["execution_policy"],
            }
            for c in population["cases"]
        ],
        "attempted_ids": sorted(row["id"] for row in rows),
    }


def validate_report(report, expected_summary, population, *, execution_os, backend):
    """Validate a closed finalized report against independent frozen references."""
    _require(
        type(report) is dict and type(expected_summary) is dict,
        "report and summary must be objects",
    )
    # Reject incomplete keyed observations before the more expensive reference bind.
    # Successful acceptance still compares the complete independent population.
    observations = report.get("successful_observations")
    _require(
        type(observations) is list
        and type(expected_summary.get("observation_keys")) is int
        and len(observations) == expected_summary["observation_keys"],
        "incomplete observation keys",
    )
    independent = validate_selection(expected_summary, population)
    validate_backend_sources()
    scope = independent["scope"]
    _require(execution_os in ("linux", "d3os"), "unknown execution OS")
    wanted = (
        "host_model_no_device"
        if scope == "ModelReference"
        else LINUX_BACKEND
        if execution_os == "linux"
        else NATIVE_BACKEND
    )
    _same(backend, wanted, "backend selection")
    _require(
        scope == "ModelReference"
        or (execution_os == "linux" and scope == "LinuxCommon")
        or (execution_os == "d3os" and scope in ("NativeValues", "NativeErrors")),
        "scope belongs to another OS",
    )
    _same(report, _expected_report(expected_summary, independent, wanted), "report")
    return {
        "scope": scope,
        "backend": wanted,
        "device_attempted": expected_summary["cases"],
        "device_passed": expected_summary["cases"],
        "model_passed": expected_summary["attachments"],
        "observations": expected_summary["observation_keys"],
        "logical_snapshots_independently_revalidated": scope != "ModelReference",
        "all_physical_source_words_independently_revalidated": False,
        "requires_parent_source_bound_runtime_assertions": True,
        "physical_tails_ownership_residency_proven": False,
        "full_parity": False,
        "vendor_parity": False,
        "performance": False,
    }


_LOG_TIME = r"\[(?:0|[1-9][0-9]*)\.[0-9]{3}\]"
_D3OS_LOG_RECORD = re.compile(
    r"(?:" + _LOG_TIME + r"\[INF\]\[sys_logger\.rs@020\]"
    r"|\x1b\[36m"
    + _LOG_TIME
    + r"\x1b\[94m\[INF\]\x1b\[35m\[sys_logger\.rs@020\]\x1b\[39m)"
    r" \[infer\.rs@050\] (?P<record>.*)"
)
MARKERS = frozenset(
    {
        "BURN_INDEXED_ACCUMULATION_REPORT",
        "BURN_INDEXED_ACCUMULATION_OK",
        "BURN_INDEXED_ACCUMULATION_MODEL_OK",
        "NVIDIA_BURN_INDEXED_ACCUMULATION_SHARED",
        "NVIDIA_BURN_INDEXED_ACCUMULATION_PLAN",
        "NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_C3",
        "NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_PASS",
        "NVIDIA_BURN_INDEXED_ACCUMULATION_STRICT_OK",
    }
)


def _records(serial, execution_os, *, allow_inherited=False):
    _require(execution_os in ("linux", "d3os"), "unknown execution OS")
    _require(
        type(allow_inherited) is bool
        and (not allow_inherited or execution_os == "d3os"),
        "inherited capture requires Native Raw scope",
    )
    _require(type(serial) is str and "\x00" not in serial, "invalid serial text")
    result = []
    lines = serial.split("\n")
    for index, line in enumerate(lines):
        if index < len(lines) - 1 and line.endswith("\r"):
            line = line[:-1]
        # Reserve our namespace before allowing any unrelated inherited lines.
        # Case, separator, prefix, suffix and diagnostic aliases must not escape
        # strict marker validation by looking like another suite.
        if re.search(r"indexed[\s_.-]*accumulation", line, re.IGNORECASE):
            logged = _D3OS_LOG_RECORD.fullmatch(line)
            if logged:
                _require(execution_os == "d3os", "D3OS logger prefix in Linux run")
                line = logged["record"]
            token, separator, payload = line.partition(" ")
            _require(
                token in MARKERS and separator and payload,
                "unknown/contaminated indexed accumulation marker",
            )
            _require(
                execution_os == "d3os" or not token.startswith("NVIDIA_"),
                "foreign OS marker",
            )
            _require(
                execution_os != "d3os" or token != "BURN_INDEXED_ACCUMULATION_REPORT",
                "CLI report in Native run",
            )
            _require(
                re.search(r"[\x00-\x1f\x7f]", line) is None,
                "control sequence in record",
            )
            result.append((token, payload))
        elif not allow_inherited:
            _require(
                re.search(
                    r"\bpanicked\b|INDEXED.ACCUMULATION.FAILED|\b(?:NVIDIA_)?BURN_",
                    line,
                    re.IGNORECASE,
                )
                is None,
                "failure diagnostic in serial",
            )
    return result


def _footer(summary):
    if summary["scope"] == "ModelReference":
        return f"evidence=MODEL templates={summary['templates']} attachments={summary['attachments']} device_attempted=0 device_passed=0"
    return (
        f"scope={summary['scope']} attempted={summary['cases']} passed={summary['cases']} "
        f"source_roles={summary['source_roles']} source_storages={summary['source_storages']} "
        f"observations={summary['observation_keys']} logical_only=true full_parity=false"
    )


def validate(serial, expected_summary, population, *, execution_os, backend):
    """Require exactly one OS-correct report followed by its exact success footer."""
    validate_selection(expected_summary, population)
    records = _records(serial, execution_os)
    report_marker = (
        "BURN_INDEXED_ACCUMULATION_REPORT"
        if execution_os == "linux"
        else "NVIDIA_BURN_INDEXED_ACCUMULATION_SHARED"
    )
    footer_marker = (
        "BURN_INDEXED_ACCUMULATION_MODEL_OK"
        if expected_summary.get("scope") == "ModelReference"
        else "BURN_INDEXED_ACCUMULATION_OK"
    )
    _require(
        len(records) == 2 and records[0][0] == report_marker,
        "missing/duplicate/unexpected report",
    )
    _same(records[1], (footer_marker, _footer(expected_summary)), "terminal footer")
    return validate_report(
        strict_json(records[0][1]),
        expected_summary,
        population,
        execution_os=execution_os,
        backend=backend,
    )


RAW_SOURCE = "os/application/infer/src/gpu_tests/burn/indexed_accumulation.rs"
RAW_COUNTS = {
    "valid_trials": 196,
    "invalid_trials": 16,
    "recoveries": 16,
    "c3_original": 132,
    "c3_changed_base": 132,
    "native_physical": 99,
    "native_errors": 21,
    "empty_aliases": 33,
    "queued": 11,
    "public_lifetimes": 5,
    "native_metadata_rejects": 26,
}
RAW_HOST_PLAN = {
    "FROZEN": True,
    "raw_recipes": 138,
    "valid_GPU_recipes": 98,
    "invalid_status_GPU_recipes": 8,
    "host_metadata_rejects": 32,
    "raw_launches": 228,
    "C3_vectors": 16,
    "C3_unique_keys": 132,
    "MODEL_device_passes": 0,
    "main_rows_added": 0,
    "device_observation": "NOT_EXECUTED",
}
RAW_FINAL_FOOTER = (
    "protocol=55 profiles=89 raw_launches=228 shared_values=184 shared_errors=8 "
    "MODEL_device_passes=0 os=d3os backend=native qualification=project_contract full_parity=false"
)


def _pinned_json(pin, label):
    _require(
        type(pin) is dict and set(pin) == {"path", "sha256"},
        f"{label}: file pin required",
    )
    _require(
        type(pin["path"]) is str
        and type(pin["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]),
        f"{label}: invalid file pin",
    )
    data = Path(pin["path"]).read_bytes()
    _same(_sha(data), pin["sha256"], label + " SHA256")
    return strict_json(data.decode("utf-8"))


def _raw_sources(source_pins):
    _require(
        type(source_pins) is dict
        and set(source_pins) == {"approved", "final", "sources", "raw_plan", "recipes"},
        "Raw final source/plan/recipe pins required",
    )
    _same(source_pins["approved"], True, "Raw parent approval")
    _same(source_pins["final"], True, "Raw final freeze")
    sources = source_pins["sources"]
    _require(
        type(sources) is dict
        and set(SOURCE_PINS).union({RAW_SOURCE}) <= sources.keys(),
        "Raw/Shared reviewed closure missing",
    )
    for name, digest in sources.items():
        _require(
            type(name) is str
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest),
            "invalid Raw source pin",
        )
        _same(_sha((ROOT / name).read_bytes()), digest, f"Raw source {name}")
    for name, digest in SOURCE_PINS.items():
        _same(sources[name], digest, "Shared source binding")
    _require(
        "const FROZEN: bool = true;" in (ROOT / RAW_SOURCE).read_text(),
        "Raw source is not frozen",
    )


def _raw_plan(raw_plan, source_pins):
    sealed = _pinned_json(source_pins["raw_plan"], "Raw closed plan")
    _same(raw_plan, sealed, "Raw supplied plan")
    recipes = _pinned_json(source_pins["recipes"], "Raw recipes")
    _, models = reference_selection("ModelReference")
    class_models = {
        (m["key"]["template"], m["key"]["content_sha256"]): m["attachment"]
        for m in models["models"]
        if m["attachment"].get("class_pattern", False)
    }
    c3_keys = [
        {"template": template, "sha256": digest} for template, digest in class_models
    ]
    # The reviewed archived plan keeps its historical false status. The separate
    # live host_plan() and external parent source approval establish the freeze.
    wanted = {
        "C3_keybound_changed_base_trials": 132,
        "C3_keybound_original_base_trials": 132,
        "C3_keys": c3_keys,
        "C3_original_unique_keys": 132,
        "C3_vectors": 16,
        "F32_internal_companion_vectors": 4,
        "FROZEN": False,
        "MODEL_attachments_GPU_count": 0,
        "approval": "Root ceiling approved, exact plan/numerical/source/count freeze pending",
        "category_counts": {
            "C3_keybound": 16,
            "F32_internal_companion": 4,
            "baseline": 52,
            "broadcast_signed_stride": 13,
            "invalid_status": 8,
            "signed_stride": 13,
        },
        "device_attempts_observed": 0,
        "device_passes_observed": 0,
        "host_metadata_rejects": 32,
        "initialized_base_trials_per_GPU_recipe": 2,
        "invalid_status_GPU_recipes": 8,
        "main_rows_added": 0,
        "raw_invalid_status_launches": 16,
        "raw_recovery_launches": 16,
        "raw_total_launches": 228,
        "raw_valid_launches": 196,
        "recipes": 138,
        "recipes_sha256": source_pins["recipes"]["sha256"],
        "reference_core_sha256": SOURCE_PINS[
            "tools/indexed-accumulation-reference/core.py"
        ],
        "reference_ieee_sha256": SOURCE_PINS[
            "tools/indexed-accumulation-reference/ieee.py"
        ],
        "schema_version": 1,
        "valid_GPU_recipes": 98,
    }
    _same(raw_plan, wanted, "Raw independent closed plan")
    _require(
        type(recipes) is list and len(recipes) == 106, "Raw GPU recipe count differs"
    )
    _require(len({r["name"] for r in recipes}) == 106, "Raw duplicate recipe names")
    _require(
        sum(r["invalid"] is True for r in recipes) == 8
        and sum(r["invalid"] is False for r in recipes) == 98,
        "Raw valid/invalid recipes differ",
    )
    c3_recipes = [r for r in recipes if r["keys"]]
    _require(len(c3_recipes) == 16, "Raw C3 vector count differs")
    observed_keys = []
    records = []
    for recipe in c3_recipes:
        _require(
            type(recipe["name"]) is str
            and re.fullmatch(
                r"C3_(?:F64|Flex32|F16|BF16)_(?:Add|Mul|Min|Max)", recipe["name"]
            ),
            "Raw C3 vector identity differs",
        )
        for key in recipe["keys"]:
            identity = key["template"], key["sha256"]
            _require(
                identity in class_models, "Raw C3 key is not an immutable class pattern"
            )
            attachment = class_models[identity]
            _same(recipe["dtype"], attachment["dtype"], "Raw C3 dtype")
            _same(recipe["op"], attachment["operation"], "Raw C3 operation")
            _same(
                key["expected"],
                int(attachment["expected_bits"], 16),
                "Raw C3 original expected bits",
            )
            lane = key["lane"]
            _require(
                type(lane) is int and 0 <= lane < recipe["parameters"][3],
                "Raw C3 field is outside the initialized output",
            )
            _same(
                recipe["trials"][0]["expected_bits"][lane],
                key["expected"],
                "Raw C3 original field reference",
            )
            observed_keys.append(identity)
        for trial in (0, 1):
            for key in recipe["keys"]:
                records.append(
                    (
                        "NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_C3",
                        (
                            f"evidence=RAW_HARDWARE template={key['template']} contentsha={key['sha256']} "
                            f"vector={recipe['name']} field={key['lane']} initialized_base_trial={trial} matched=true MODEL_device_passes=0"
                        ),
                    )
                )
    _require(
        len(observed_keys) == len(set(observed_keys)) == 132
        and set(observed_keys) == set(class_models),
        "Raw canonical C3 membership differs",
    )
    return records


def validate_raw(serial, raw_plan, *, protocol=55, profiles=89, source_pins):
    """Check the final parent Native wrapper, Raw assertions, and both Shared scopes.

    raw_plan is the immutable archived closed-plan.json, not live host_plan().
    source_pins must be supplied by a trusted caller after parent approval:
    {approved: true, final: true, sources: {repo_path: sha256, ...},
     raw_plan: {path: ..., sha256: ...}, recipes: {path: ..., sha256: ...}}.
    sources includes Raw and all fifteen Shared sources. No runtime-emitted source
    hash is invented. Image/compiler/OS/execution binding remains external.
    This API consumes the whole Native capture. Unrelated inherited success and
    diagnostic lines remain outside its scope and require parent validation.
    Raw markers attest source-bound internal checks, never host per-word reads.
    """
    _same(protocol, 55, "Raw protocol")
    _same(profiles, 89, "Raw profiles")
    _raw_sources(source_pins)
    c3_records = _raw_plan(raw_plan, source_pins)
    records = _records(serial, "d3os", allow_inherited=True)
    markers = (
        ["NVIDIA_BURN_INDEXED_ACCUMULATION_PLAN"]
        + ["NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_C3"] * 264
        + ["NVIDIA_BURN_INDEXED_ACCUMULATION_RAW_PASS"]
        + ["NVIDIA_BURN_INDEXED_ACCUMULATION_SHARED"] * 2
        + ["BURN_INDEXED_ACCUMULATION_OK"] * 2
        + ["NVIDIA_BURN_INDEXED_ACCUMULATION_STRICT_OK"]
    )
    _same([marker for marker, _ in records], markers, "Native complete marker order")
    plan = re.fullmatch(
        r"protocol=55 profiles=89 services=14 population=(.*)", records[0][1]
    )
    _require(plan is not None, "Native PLAN protocol/profile/service fields differ")
    _same(strict_json(plan[1]), RAW_HOST_PLAN, "Native live host plan")
    _same(records[1:265], c3_records, "Raw exact key/vector/field/trial records")
    counts = ", ".join(f"{name}: {value}" for name, value in RAW_COUNTS.items())
    _same(
        records[265][1],
        "raw_recipes=138 raw_launches=228 counts=Counts { "
        + counts
        + " } main_rows_added=0 MODEL_device_passes=0 synchronous_status_wait=true",
        "Raw observed counts and status wait",
    )
    shared_receipts = []
    shared_reports = []
    for index, scope in enumerate(("NativeValues", "NativeErrors")):
        summary, population = reference_selection(scope)
        report = strict_json(records[266 + index][1])
        shared_receipts.append(
            validate_report(
                report, summary, population, execution_os="d3os", backend=NATIVE_BACKEND
            )
        )
        _same(
            records[268 + index][1], _footer(summary), f"{scope} exact logical footer"
        )
        shared_reports.append(report)
    _same(records[270][1], RAW_FINAL_FOOTER, "Native final strict footer")
    return {
        "protocol": 55,
        "profiles": 89,
        "services": 14,
        "raw_recipes": 138,
        "raw_launches": 228,
        "C3_items": 264,
        "counts": dict(RAW_COUNTS),
        "shared": shared_receipts,
        "shared_reports": shared_reports,
        "shared_reports_independently_validated": True,
        "source_bound_runtime_assertions": True,
        "requires_parent_execution_source_image_compiler_binding": True,
        "requires_parent_inherited_suite_validation": True,
        "unrelated_lines_validated": False,
        "native_control_read_bytes_allowed": 4,
        "host_per_word_revalidation": False,
        "physical_tails_ownership_residency_proven": False,
        "full_parity": False,
        "vendor_parity": False,
        "performance": False,
    }
