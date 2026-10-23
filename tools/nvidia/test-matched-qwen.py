#!/usr/bin/env python3
"""Check the matched benchmark audit against private, corrupted model views."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def reject(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("the auditor accepted corrupted benchmark data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-directory", type=Path, default=ROOT / "target/qwen")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--d3os-log", type=Path, required=True)
    parser.add_argument("--linux-log", type=Path, required=True)
    parser.add_argument(
        "--inputs", type=Path, default=ROOT / "target/nvidia/matched-qwen"
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/source-b11827",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_source / "gguf-py"))
    import gguf
    import numpy as np

    spec = importlib.util.spec_from_file_location(
        "matched_qwen", ROOT / "tools/nvidia/prepare-linux-qwen.py"
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    archive = args.qwen_directory / "qwen3.5-0.8b.d3t"
    records, settings = auditor.audit_weights(archive, args.gguf)
    assert len(records) == 320 and len(settings) == 19
    # Copy-on-write mappings preserve both original model files.
    reader = gguf.GGUFReader(args.gguf, mode="c")
    with patch.object(gguf, "GGUFReader", return_value=reader):
        for tensor in [
            reader.tensors[0],
            next(t for t in reader.tensors if t.name == "output_norm.weight"),
        ]:
            data = tensor.data.view(np.uint8).ravel()
            original = int(data[0])
            try:
                data[0] = original ^ 1
                reject(lambda: auditor.audit_weights(archive, args.gguf))
            finally:
                data[0] = original
        tensor = reader.tensors[0]
        original = tensor.shape.copy()
        try:
            tensor.shape[:] = original[::-1]
            reject(lambda: auditor.audit_weights(archive, args.gguf))
        finally:
            tensor.shape[:] = original
        with patch.dict(
            reader.fields,
            {
                "qwen35.attention.layer_norm_rms_epsilon": SimpleNamespace(
                    contents=lambda: 1e-5
                )
            },
        ):
            reject(lambda: auditor.audit_weights(archive, args.gguf))
        reader.tensors.append(reader.tensors[0])
        try:
            reject(lambda: auditor.audit_weights(archive, args.gguf))
        finally:
            reader.tensors.pop()

    text = args.d3os_log.read_text(errors="replace")
    prompt, decode = auditor.trace_tokens(text)
    assert len(prompt) == 22 and len(decode) == 64
    reject(
        lambda: auditor.trace_tokens(
            text.replace("phase=decode run=2", "phase=decode run=1")
        )
    )
    reject(
        lambda: auditor.trace_tokens(
            text.replace("session_capacity=256", "session_capacity=128")
        )
    )
    changed = decode.copy()
    changed[0] = (changed[0] + 1) % 248320
    before = f"phase=decode run=2 ids={json.dumps(decode)}"
    after = f"phase=decode run=2 ids={json.dumps(changed)}"
    assert before in text
    reject(lambda: auditor.trace_tokens(text.replace(before, after)))
    print("MATCHED_QWEN_AUDIT_TEST_OK weight_corruptions=5 trace_corruptions=3")
    spec = importlib.util.spec_from_file_location(
        "linux_reference", ROOT / "tools/nvidia/linux-reference.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    text = args.linux_log.read_text()
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    reference.validate_matched_qwen(text, manifest, "auto")
    reject(
        lambda: reference.validate_matched_qwen(
            text.replace("selection_mismatches=0", "selection_mismatches=1"),
            manifest,
            "auto",
        )
    )
    reject(
        lambda: reference.validate_matched_qwen(
            text.replace("n_ctx=256", "n_ctx=128"), manifest, "auto"
        )
    )
    reject(
        lambda: reference.validate_matched_qwen(
            text.replace("K (f32):", "K (f16):"), manifest, "auto"
        )
    )
    print("MATCHED_QWEN_REFERENCE_TEST_OK corruptions=3")


if __name__ == "__main__":
    main()
