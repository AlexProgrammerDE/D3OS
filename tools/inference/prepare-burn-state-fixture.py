#!/usr/bin/env python3
"""Create independent binary64 references for Burn's stateful Qwen operations."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def generated(count, seed, denominator=71):
    return [f32(((i * 37 + seed * 17) % 127 - 63) / denominator) for i in range(count)]


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def norm(row, weight, epsilon, offset):
    scale = 1 / math.sqrt(dot(row, row) / len(row) + epsilon)
    return [x * scale * (offset + w) for x, w in zip(row, weight)]


def rotate(row, position, rotary, theta):
    result = row.copy()
    half = rotary // 2
    for i in range(half):
        angle = position / theta ** (2 * i / rotary)
        sin, cos = math.sin(angle), math.cos(angle)
        result[i] = row[i] * cos - row[i + half] * sin
        result[i + half] = row[i + half] * cos + row[i] * sin
    return result


def attention(records, case, heads, kv_heads, dim, rotary, prefix, rows):
    base = f"attention.{case}"
    capacity = prefix + rows + 2
    epsilon = f32(1e-6)
    theta = f32(10000)
    qwidth, kvwidth = heads * dim, kv_heads * dim
    qg = generated(rows * 2 * qwidth, 7)
    key, value = generated(rows * kvwidth, 11), generated(rows * kvwidth, 17)
    qnorm, knorm = generated(dim, 3), generated(dim, 5)
    keys, values = generated(capacity * kvwidth, 41), generated(capacity * kvwidth, 43)
    records[f"{base}.config"] = (
        [8],
        [heads, kv_heads, dim, rotary, theta, epsilon, prefix, capacity],
    )
    for name, shape, data in [
        ("qg", [rows, 2 * qwidth], qg),
        ("key", [rows, kvwidth], key),
        ("value", [rows, kvwidth], value),
        ("qnorm", [dim], qnorm),
        ("knorm", [dim], knorm),
        ("keys", [capacity, kv_heads, dim], keys),
        ("values", [capacity, kv_heads, dim], values),
    ]:
        records[f"{base}.{name}"] = (shape, data.copy())
    outputs = []
    for row in range(rows):
        position = prefix + row
        for head in range(kv_heads):
            start = row * kvwidth + head * dim
            prepared = rotate(
                norm(key[start : start + dim], knorm, epsilon, 1),
                position,
                rotary,
                theta,
            )
            keys[
                position * kvwidth + head * dim : position * kvwidth + (head + 1) * dim
            ] = list(map(f32, prepared))
        values[position * kvwidth : (position + 1) * kvwidth] = value[
            row * kvwidth : (row + 1) * kvwidth
        ]
        for head in range(heads):
            start = row * 2 * qwidth + head * 2 * dim
            query = rotate(
                norm(qg[start : start + dim], qnorm, epsilon, 1),
                position,
                rotary,
                theta,
            )
            kv_head = head // (heads // kv_heads)
            scores = [
                dot(
                    query,
                    keys[
                        p * kvwidth + kv_head * dim : p * kvwidth + (kv_head + 1) * dim
                    ],
                )
                / math.sqrt(dim)
                for p in range(position + 1)
            ]
            maximum = max(scores)
            scores = [math.exp(score - maximum) for score in scores]
            total = math.fsum(scores)
            outputs.extend(
                math.fsum(
                    scores[p] / total * values[p * kvwidth + kv_head * dim + d]
                    for p in range(position + 1)
                )
                * sigmoid(qg[start + dim + d])
                for d in range(dim)
            )
    records[f"{base}.output"] = ([rows, qwidth], outputs)
    records[f"{base}.final_keys"] = ([capacity, kv_heads, dim], keys)
    records[f"{base}.final_values"] = ([capacity, kv_heads, dim], values)


def delta(
    records, case, key_heads, heads, key_dim, value_dim, kernel, rows, projected=False
):
    base = f"delta.{case}"
    epsilon = f32(1e-6)
    keywidth, vwidth = key_heads * key_dim, heads * value_dim
    channels = keywidth * 2 + vwidth
    convolution, state = (
        generated(channels * kernel, 51, 251),
        generated(heads * key_dim * value_dim, 53, 1001),
    )
    weights = generated(channels * kernel, 3)
    decay, bias, rms = (
        generated(heads, 5),
        generated(heads, 7),
        generated(value_dim, 11),
    )
    records[f"{base}.config"] = (
        [7],
        [key_heads, heads, key_dim, value_dim, kernel, epsilon, rows],
    )
    if projected:
        width = 6
        hidden = generated(rows * width, 13, 128)
        records[f"{base}.hidden"] = ([rows, width], hidden)
        projections = []
        for seed, (name, size) in enumerate(
            [("qkv", channels), ("gate", vwidth), ("beta", heads), ("alpha", heads)]
        ):
            matrix = [((i * 7 + seed * 11) % 31 - 15) / 32 for i in range(size * width)]
            records[f"{base}.weight_{name}"] = ([size, width], matrix)
            projections.append(
                [
                    f32(
                        dot(
                            hidden[row * width : (row + 1) * width],
                            matrix[out * width : (out + 1) * width],
                        )
                    )
                    for row in range(rows)
                    for out in range(size)
                ]
            )
        qkv, gate, beta, alpha = projections
    else:
        qkv, gate = generated(rows * channels, 13), generated(rows * vwidth, 17)
        beta, alpha = generated(rows * heads, 19), generated(rows * heads, 23)
    for name, shape, data in [
        ("qkv", [rows, channels], qkv),
        ("gate", [rows, vwidth], gate),
        ("beta", [rows, heads], beta),
        ("alpha", [rows, heads], alpha),
        ("weights", [channels, 1, kernel], weights),
        ("decay", [heads], decay),
        ("bias", [heads], bias),
        ("rms", [value_dim], rms),
        ("convolution", [channels, kernel], convolution),
        ("recurrent", [heads, key_dim, value_dim], state),
    ]:
        records[f"{base}.{name}"] = (shape, data.copy())
    outputs = []
    for row in range(rows):
        mixed = []
        for channel in range(channels):
            start = channel * kernel
            history = convolution[start + 1 : start + kernel] + [
                qkv[row * channels + channel]
            ]
            convolution[start : start + kernel] = history
            value = dot(history, weights[start : start + kernel])
            mixed.append(value * sigmoid(value))
        for head in range(heads):
            key_head = head // (heads // key_heads)
            query = mixed[key_head * key_dim : (key_head + 1) * key_dim]
            key = mixed[
                keywidth + key_head * key_dim : keywidth + (key_head + 1) * key_dim
            ]
            value = mixed[
                keywidth * 2 + head * value_dim : keywidth * 2 + (head + 1) * value_dim
            ]
            qscale = 1 / math.sqrt(dot(query, query) + 1e-6) / math.sqrt(key_dim)
            kscale = 1 / math.sqrt(dot(key, key) + 1e-6)
            query, key = [q * qscale for q in query], [k * kscale for k in key]
            rate = sigmoid(beta[row * heads + head])
            alpha_head = alpha[row * heads + head] + bias[head]
            softplus = max(alpha_head, 0) + math.log1p(math.exp(-abs(alpha_head)))
            discount = math.exp(-math.exp(decay[head]) * softplus)
            prediction = [0.0] * value_dim
            for k in range(key_dim):
                for d in range(value_dim):
                    index = (head * key_dim + k) * value_dim + d
                    state[index] *= discount
                    prediction[d] += key[k] * state[index]
            update = [(v - p) * rate for v, p in zip(value, prediction)]
            out = [0.0] * value_dim
            for k in range(key_dim):
                for d in range(value_dim):
                    index = (head * key_dim + k) * value_dim + d
                    state[index] = f32(state[index] + key[k] * update[d])
                    out[d] += query[k] * state[index]
            outputs.extend(
                v * (g * sigmoid(g))
                for v, g in zip(
                    norm(out, rms, epsilon, 0),
                    gate[
                        row * vwidth + head * value_dim : row * vwidth
                        + (head + 1) * value_dim
                    ],
                )
            )
    records[f"{base}.output"] = ([rows, vwidth], outputs)
    records[f"{base}.final_convolution"] = ([channels, kernel], convolution)
    records[f"{base}.final_recurrent"] = ([heads, key_dim, value_dim], state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("os/library/ai/tests/fixtures/qwen-state.d3t"),
    )
    args = parser.parse_args()
    records = {}
    attention_cases = [
        (4, 2, 4, 2, 0, 7),
        (8, 2, 256, 64, 3, 5),
        (2, 1, 258, 64, 2, 4),
        (4, 2, 16, 8, 511, 3),
    ]
    delta_cases = [
        (2, 2, 5, 7, 3, 7),
        (1, 3, 5, 7, 3, 7),
        (2, 4, 128, 128, 4, 3),
        (1, 1, 129, 131, 9, 3),
        (4, 4, 5, 8, 4, 7),
    ]
    for case, config in enumerate(attention_cases):
        attention(records, case, *config)
    for case, config in enumerate(delta_cases):
        delta(records, case, *config, projected=case == 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(b"D3TENSR\0" + struct.pack("<II", 1, len(records)))
        for name, (shape, values) in records.items():
            encoded = name.encode()
            assert math.prod(shape) == len(values)
            assert all(math.isfinite(value) for value in values)
            output.write(struct.pack("<HHQ", len(encoded), len(shape), len(values)))
            output.write(encoded + struct.pack("<" + "Q" * len(shape), *shape))
            output.write(struct.pack("<" + "f" * len(values), *values))
    manifest = {
        "reference": "Python binary64 math; rounded FP32 inputs, cache writes, and projection outputs",
        "attention_cases": attention_cases,
        "delta_cases": delta_cases,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "records": len(records),
                "sha256": manifest["sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
