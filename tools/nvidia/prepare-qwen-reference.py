#!/usr/bin/env python3
"""Export independent FP32 Qwen references one Transformers layer at a time."""

import argparse
import gc
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
}


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("linux_results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--gguf-python",
        type=Path,
        default=ROOT / "target/nvidia/linux-llama/source-b11827/gguf-py",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.gguf_python.resolve()))
    import gguf
    import numpy as np
    import torch
    import transformers
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5TextRotaryEmbedding,
    )

    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    baseline = json.loads(args.linux_results.read_text())
    identity = digest(args.model)
    if identity != baseline["model_sha256"]:
        raise ValueError("reference input targets a different model")
    prompt, decode = baseline["prompt_ids"], baseline["decode_ids"]
    if not 1 <= len(prompt) <= 256 or len(decode) != 128:
        raise ValueError("unsupported benchmark token counts")
    tokens = prompt + decode
    config = Qwen3_5TextConfig(**json.loads(args.config.read_text())["text_config"])
    config._attn_implementation = "eager"
    config.dtype = "float32"
    reader = gguf.GGUFReader(str(args.model))
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    def matrix(name, start=0, end=None):
        tensor = tensors[name]
        shape = tuple(int(dim) for dim in tensor.shape[::-1])
        if len(shape) == 1:
            return torch.from_numpy(
                gguf.dequantize(tensor.data, tensor.tensor_type).reshape(shape).copy()
            )
        rows, width = shape
        end = rows if end is None else end
        row_bytes = tensor.n_bytes // rows
        data = tensor.data.view(np.uint8).reshape(-1)[
            start * row_bytes : end * row_bytes
        ]
        return torch.from_numpy(
            gguf.dequantize(data, tensor.tensor_type).reshape(end - start, width).copy()
        )

    keys, heads = config.linear_num_key_heads, config.linear_num_value_heads
    repeats, dim = heads // keys, config.linear_value_head_dim

    def grouped_rows(value, dimension):
        return (
            value.reshape(repeats, keys, dimension, *value.shape[1:])
            .transpose(0, 1)
            .reshape(value.shape)
            .contiguous()
        )

    def parameter(name, layer, shape):
        value = matrix(f"blk.{layer}.{NAMES[name]}")
        if repeats > 1 and name.startswith("linear_attn."):
            if name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
                split = 2 * keys * config.linear_key_head_dim
                value = torch.cat((value[:split], grouped_rows(value[split:], dim)))
            elif name.endswith("in_proj_z.weight"):
                value = grouped_rows(value, dim)
            elif name.endswith(
                ("in_proj_a.weight", "in_proj_b.weight", "A_log", "dt_bias")
            ):
                value = grouped_rows(value, 1)
            elif name.endswith("out_proj.weight"):
                value = (
                    value.reshape(value.shape[0], repeats, keys, dim)
                    .transpose(1, 2)
                    .reshape(value.shape)
                    .contiguous()
                )
        if name.endswith("A_log"):
            if not torch.all(value < 0):
                raise ValueError("invalid GGUF decay parameter")
            value = (-value).log()
        elif name.endswith("norm.weight") and name != "linear_attn.norm.weight":
            value = value - 1
        return value.reshape(shape)

    spec = importlib.util.spec_from_file_location(
        "qwen_export", ROOT / "tools/inference/prepare-qwen.py"
    )
    archive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    args.output.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output / "fixture.d3t"
    with torch.inference_mode(), fixture_path.open("wb") as fixture:
        archive.header(fixture, config.num_hidden_layers + 2)
        token_array = np.array(tokens, dtype="<f4")
        archive.record(
            fixture, "tokens", token_array.shape, token_array.tobytes(), dtype=0
        )
        hidden = torch.stack(
            [matrix("token_embd.weight", token, token + 1)[0] for token in tokens]
        ).unsqueeze(0)
        positions = torch.arange(len(tokens)).view(1, -1)
        rotary = Qwen3_5TextRotaryEmbedding(config)
        embeddings = rotary(hidden, positions.unsqueeze(0).expand(3, -1, -1))
        mask = torch.full((1, 1, len(tokens), len(tokens)), float("-inf")).triu(1)
        for index in range(config.num_hidden_layers):
            with torch.device("meta"):
                layer = Qwen3_5DecoderLayer(config, index)
            state = {
                name: parameter(name, index, value.shape)
                for name, value in layer.state_dict().items()
            }
            layer.load_state_dict(state, strict=True, assign=True)
            layer.eval()
            hidden = layer(
                hidden,
                position_embeddings=embeddings,
                position_ids=positions,
                attention_mask=mask
                if config.layer_types[index] == "full_attention"
                else None,
                past_key_values=None,
                use_cache=False,
            )
            data = hidden[0].numpy().astype("<f4", copy=False)
            archive.record(
                fixture, f"layer.{index}", data.shape, data.tobytes(), dtype=0
            )
            print(
                f"QWEN_FP32_LAYER_OK os=linux backend=torch_cpu layer={index} tokens={len(tokens)}",
                flush=True,
            )
            del state, layer
            gc.collect()
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(
            matrix("output_norm.weight") - 1, requires_grad=False
        )
        hidden = norm(hidden)[0]
        logits = np.empty((len(tokens), config.vocab_size), dtype="<f4")
        output_name = (
            "output.weight" if "output.weight" in tensors else "token_embd.weight"
        )
        for start in range(0, config.vocab_size, 2048):
            end = min(config.vocab_size, start + 2048)
            logits[:, start:end] = (hidden @ matrix(output_name, start, end).T).numpy()
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite FP32 reference logits")
        archive.record(fixture, "logits", logits.shape, logits.tobytes(), dtype=0)
    logit_path = args.output / "logits.bin"
    with logit_path.open("wb") as file:
        file.write(
            b"D3QLOG\0\0"
            + struct.pack(
                "<6I", 1, config.vocab_size, len(prompt), len(decode), 2048, 32
            )
        )
        file.write(bytes.fromhex(identity))
        file.write(struct.pack(f"<{len(tokens)}I", *tokens))
        file.write(logits[len(prompt) - 1 :].tobytes())
    manifest = {
        "os": "Linux host",
        "backend": "PyTorch CPU FP32",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gguf_model_sha256": identity,
        "config_sha256": digest(args.config),
        "input_json_sha256": digest(args.linux_results),
        "source_sha256": digest(Path(__file__)),
        "tokens": len(tokens),
        "layers": config.num_hidden_layers,
        "fixture_sha256": digest(fixture_path),
        "logits_sha256": digest(logit_path),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"QWEN_FP32_REFERENCE_OK os=linux backend=torch_cpu layers={config.num_hidden_layers} tokens={len(tokens)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
