#!/usr/bin/env python3
"""Audit Qwen weights and export identical tokens for the Linux reference."""

import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import re
import shutil
import struct
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b11827"
SOURCE_SHA256 = "988d6834bd8b57ba973922ac0bda27f282a50f17c42fcdf30817915434f06c3a"
BINARY_NAME = "llama-b11827-bin-ubuntu-cuda-12.8-x64.tar.gz"
BINARY_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/b11827/{BINARY_NAME}"
)
BINARY_SHA256 = "c1f538781281ec07841badf0f3972ed88e4f4b60b5aa7598139cadfa3d51a484"


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def source_tree(directory):
    archive = directory.with_suffix(".tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or digest(archive) != SOURCE_SHA256:
        temporary = archive.with_suffix(".download")
        with (
            urllib.request.urlopen(SOURCE_URL, timeout=120) as source,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(source, output)
        if digest(temporary) != SOURCE_SHA256:
            raise ValueError("llama.cpp source checksum mismatch")
        temporary.replace(archive)
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source:
            parts = Path(member.name).parts
            if parts[1:2] in (("include",), ("gguf-py",)) or parts[1:3] == (
                "ggml",
                "include",
            ):
                member.name = str(Path(*parts[1:]))
                source.extract(member, directory, filter="data")
    return directory


def binary_tree(directory):
    archive = directory.parent / BINARY_NAME
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or digest(archive) != BINARY_SHA256:
        temporary = archive.with_suffix(".download")
        with (
            urllib.request.urlopen(BINARY_URL, timeout=120) as source,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(source, output)
        if digest(temporary) != BINARY_SHA256:
            raise ValueError("llama.cpp binary checksum mismatch")
        temporary.replace(archive)
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        for member in source:
            parts = Path(member.name).parts
            if parts[:1] != ("llama-b11827",):
                raise ValueError("unexpected release archive path")
            if len(parts) > 1:
                member.name = str(Path(*parts[1:]))
                source.extract(member, directory, filter="data")
    return directory


def archive_records(mapped):
    magic, version, count = struct.unpack_from("<8sII", mapped)
    if (magic, version, count) != (b"D3TENSR\0", 2, 321):
        raise ValueError("expected the pinned version-2 Qwen text archive")
    position = 16
    seen = set()
    for _ in range(count):
        length, rank, elements, dtype = struct.unpack_from("<HHQH", mapped, position)
        position += 14
        if not 1 <= length <= 256 or not 1 <= rank <= 8 or dtype not in (0, 1):
            raise ValueError("invalid archive record")
        name = mapped[position : position + length].decode()
        position += length
        shape = struct.unpack_from("<" + "Q" * rank, mapped, position)
        position += rank * 8
        end = position + elements * (4 if dtype == 0 else 2)
        if name in seen or not all(shape) or math.prod(shape) != elements:
            raise ValueError("invalid archive name or shape")
        if end > len(mapped):
            raise ValueError("truncated archive data")
        seen.add(name)
        yield name, shape, dtype, position, elements
        position = end
    if position != len(mapped):
        raise ValueError("trailing archive data")


def audit_weights(archive, model):
    import gguf
    import numpy as np

    reader = gguf.GGUFReader(model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(reader.tensors) != 320 or len(tensors) != 320:
        raise ValueError("require exactly 320 unique GGUF text tensors")
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, 24)
    seen = set()
    records = []
    configuration = None
    with archive.open("rb") as file:
        # NumPy views retain the mapping during exception reporting too.
        data = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        for name, shape, dtype, offset, count in archive_records(data):
            if name == "qwen3_5.config":
                if dtype != 0 or shape != (16,):
                    raise ValueError("invalid native Qwen configuration")
                configuration = np.frombuffer(
                    data, dtype="<f4", count=16, offset=offset
                ).copy()
                continue
            mapped = mapping.get_name(
                "model." + name.replace(".dt_bias", ".dt_proj.bias"),
                try_suffixes=(".weight", ".bias"),
            )
            tensor = tensors.get(mapped)
            if tensor is None or mapped in seen:
                raise ValueError(f"missing or duplicate GGUF tensor: {name}")
            expected_shape = tuple(reversed(tuple(int(n) for n in tensor.shape)))
            if name.endswith("conv1d.weight"):
                shape = (shape[0], shape[2])
            if shape != expected_shape:
                raise ValueError(f"shape mismatch: {name}")
            source = np.frombuffer(
                data,
                dtype="<f4" if dtype == 0 else "<u2",
                count=count,
                offset=offset,
            )
            conversion = "identity"
            max_ulp = 0
            if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                if dtype != 1 or not np.array_equal(
                    source, tensor.data.view("<u2").ravel()
                ):
                    raise ValueError(f"BF16 weight mismatch: {name}")
            elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                value = (
                    source
                    if dtype == 0
                    else (source.astype(np.uint32) << 16).view(np.float32)
                )
                if name.endswith("A_log"):
                    value = -np.exp(value)
                    conversion = "negative_exp"
                elif name.endswith("norm.weight") and not name.endswith(
                    "linear_attn.norm.weight"
                ):
                    value = value + np.float32(1)
                    conversion = "norm_offset"
                else:
                    conversion = "f32_expansion"
                target = tensor.data.ravel()
                if not np.isfinite(value).all() or not np.isfinite(target).all():
                    raise ValueError(f"non-finite weight: {name}")
                ulp = np.abs(
                    value.view(np.uint32).astype(np.int64)
                    - target.view(np.uint32).astype(np.int64)
                )
                max_ulp = int(ulp.max())
                # GGUF stores -exp(A_log), rather than the checkpoint's A_log.
                # NumPy and the converter's exp implementations can round differently.
                if max_ulp > (2 if conversion == "negative_exp" else 0):
                    raise ValueError(
                        f"converted weight mismatch: {name}, ULP={max_ulp}"
                    )
                del value, target, ulp
            else:
                raise ValueError(f"unexpected quantized tensor: {name}")
            del source
            seen.add(mapped)
            records.append(
                {
                    "source": name,
                    "gguf": mapped,
                    "conversion": conversion,
                    "max_ulp": max_ulp,
                }
            )
    data.close()
    if seen != tensors.keys() or len(seen) != 320:
        raise ValueError("the audit did not cover every text tensor")
    if configuration is None:
        raise ValueError("missing native model configuration")
    c = configuration.tolist()
    settings = {
        "general.architecture": "qwen35",
        "qwen35.embedding_length": c[0],
        "qwen35.feed_forward_length": c[1],
        "qwen35.block_count": c[2],
        "qwen35.attention.head_count": c[4],
        "qwen35.attention.head_count_kv": c[5],
        "qwen35.attention.key_length": c[6],
        "qwen35.attention.value_length": c[6],
        "qwen35.rope.dimension_count": c[7],
        "qwen35.ssm.group_count": c[8],
        "qwen35.ssm.state_size": c[9],
        "qwen35.ssm.inner_size": c[8] * c[10],
        "qwen35.ssm.time_step_rank": c[8],
        "qwen35.ssm.conv_kernel": c[11],
        "qwen35.full_attention_interval": c[12],
        "qwen35.context_length": c[13],
        "qwen35.rope.freq_base": c[14],
        "qwen35.attention.layer_norm_rms_epsilon": c[15],
        "qwen35.rope.dimension_sections": [11, 11, 10, 0],
    }
    for name, expected in settings.items():
        field = reader.fields.get(name)
        if field is None or field.contents() != expected:
            raise ValueError(f"model configuration mismatch: {name}")
    return records, settings


def trace_tokens(text, decode_tokens=64):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    prompt = re.findall(r"INFERENCE_BENCH_TOKENS phase=prompt ids=(\[.*?\])", text)
    decode = re.findall(
        r"INFERENCE_BENCH_TOKENS phase=decode run=(\d+) ids=(\[.*?\])", text
    )
    if len(prompt) != 1 or len(decode) != 3 or "INFERENCE_BENCH_OK qwen " not in text:
        raise ValueError(
            "require one complete verified D3OS benchmark with three decode traces"
        )
    prompt = json.loads(prompt[0])
    if [int(run) for run, _ in decode] != [0, 1, 2]:
        raise ValueError("require three distinct D3OS benchmark runs in order")
    decoded = [json.loads(value) for _, value in decode]
    if (
        any(value != decoded[0] for value in decoded)
        or len(decoded[0]) != decode_tokens
    ):
        raise ValueError("D3OS decode traces disagree or have the wrong length")
    if not 1 <= len(prompt) <= 32 or any(
        type(n) is not int or not 0 <= n < 248320 for n in prompt + decoded[0]
    ):
        raise ValueError("invalid token IDs or prompt length")
    checksum = 0
    for token in decoded[0]:
        checksum = (checksum * 31 + token) % (1 << 64)
    benches = re.findall(r"INFERENCE_BENCH qwen ([^\r\n]*)", text)
    if len(benches) != 3:
        raise ValueError("missing D3OS benchmark measurements")
    for run, line in enumerate(benches):
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        expected = {
            "run": run,
            "session_capacity": ((len(prompt) + decode_tokens + 255) // 256) * 256,
            "prompt_tokens": len(prompt),
            "decode_tokens": decode_tokens,
            "checksum": checksum,
        }
        if any(fields.get(key) != str(value) for key, value in expected.items()):
            raise ValueError(
                "D3OS measurements do not match the exported token workload"
            )
    return prompt, decoded[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-directory", type=Path, default=ROOT / "target/qwen")
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--d3os-log", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, choices=(64, 4096), default=64)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "target/nvidia/matched-qwen"
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/source-b11827",
    )
    parser.add_argument(
        "--llama-directory",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/llama-b11827",
    )
    args = parser.parse_args()
    source = source_tree(args.llama_source.resolve())
    binaries = binary_tree(args.llama_directory.resolve())
    sys.path.insert(0, str(source / "gguf-py"))
    from tokenizers import Tokenizer

    spec = importlib.util.spec_from_file_location(
        "prepare_qwen", ROOT / "tools/inference/prepare-qwen.py"
    )
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    manifest = json.loads((args.qwen_directory / "manifest.json").read_text())
    if (
        manifest["revision"] != exporter.REVISION
        or manifest["weights_sha256"] != exporter.WEIGHTS_SHA
    ):
        raise ValueError("unexpected D3OS checkpoint provenance")
    archive = args.qwen_directory / "qwen3.5-0.8b.d3t"
    archive_sha = digest(archive)
    if archive_sha != manifest["artifacts"][archive.name]:
        raise ValueError("D3OS archive checksum mismatch")
    tokenizer_path = args.tokenizer_json or args.qwen_directory / "cache/tokenizer.json"
    if args.tokenizer_json is None and not tokenizer_path.exists():
        tokenizer_path = args.qwen_directory / "tokenizer.json"
    if digest(tokenizer_path) != exporter.TOKENIZER_SHA:
        raise ValueError("upstream tokenizer checksum mismatch")
    prompt, decode = trace_tokens(
        args.d3os_log.read_text(errors="replace"), args.decode_tokens
    )
    capacity = ((len(prompt) + len(decode) + 255) // 256) * 256
    user = (ROOT / "tools/inference/qwen-benchmark.prompt").read_text().strip()
    chat = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if (
        Tokenizer.from_file(str(tokenizer_path))
        .encode(chat, add_special_tokens=False)
        .ids
        != prompt
    ):
        raise ValueError("D3OS prompt does not match the upstream tokenizer")
    records, settings = audit_weights(archive, args.gguf)
    args.output.mkdir(parents=True, exist_ok=True)
    contract = b"QWBENCH\0" + struct.pack(
        "<IIII", 2, len(prompt), len(decode), capacity
    )
    contract += struct.pack("<" + "I" * (len(prompt) + len(decode)), *prompt, *decode)
    (args.output / "tokens.bin").write_bytes(contract)
    evidence = {
        "model": exporter.MODEL,
        "revision": exporter.REVISION,
        "source_weights_sha256": exporter.WEIGHTS_SHA,
        "d3os_archive_sha256": archive_sha,
        "gguf_sha256": digest(args.gguf),
        "llama_source_sha256": SOURCE_SHA256,
        "llama_binary_archive_sha256": BINARY_SHA256,
        "source_headers": {
            str(path.relative_to(source)): digest(path)
            for directory in (source / "include", source / "ggml/include")
            for path in sorted(directory.glob("*.h"))
        },
        "libraries": {
            path.name: digest(path) for path in sorted(binaries.glob("*.so*"))
        },
        "d3os_log_sha256": digest(args.d3os_log),
        "tokens_sha256": hashlib.sha256(contract).hexdigest(),
        "prompt": user,
        "prompt_ids": prompt,
        "decode_ids": decode,
        "cache_type": "f32",
        "batch_size": 32,
        "context_capacity": capacity,
        "model_settings": settings,
        "tensors": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(
        f"MATCHED_QWEN_INPUT_OK tensors={len(records)} prompt_tokens={len(prompt)} decode_tokens={len(decode)}"
    )


if __name__ == "__main__":
    main()
