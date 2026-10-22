#!/usr/bin/env python3
"""Export pinned Qwen3.5-0.8B text weights and Transformers reference outputs."""

import argparse
import hashlib
import json
import mmap
import shutil
import struct
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WEIGHTS = "model.safetensors-00001-of-00001.safetensors"
WEIGHTS_SHA = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
TOKENIZER_SHA = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
CONFIG_SHA = "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"
TOKENIZER_CONFIG_SHA = (
    "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
)
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
CONFIG = [
    1024,
    3584,
    24,
    248320,
    8,
    2,
    256,
    64,
    16,
    128,
    128,
    4,
    4,
    262144,
    10000000,
    1e-6,
]
DEFAULT_PROMPT = "What is 2 + 2? Answer briefly."


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download(cache, name, expected):
    path = cache / name
    if path.exists() and digest(path) == expected:
        return path
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading {name}", flush=True)
    with (
        urllib.request.urlopen(
            f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120
        ) as source,
        temporary.open("wb") as dest,
    ):
        shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
    if digest(temporary) != expected:
        raise ValueError(f"checksum mismatch for {name}")
    temporary.replace(path)
    return path


def header(file, count, version=2):
    file.write(b"D3TENSR\0" + struct.pack("<II", version, count))


def record(file, name, shape, data, dtype=0):
    name = name.encode()
    count = 1
    for dim in shape:
        count *= dim
    width = {0: 4, 1: 2}[dtype]
    if len(data) != count * width:
        raise ValueError("tensor byte count does not match its shape")
    file.write(struct.pack("<HHQH", len(name), len(shape), count, dtype))
    file.write(name)
    file.write(struct.pack("<" + "Q" * len(shape), *shape))
    file.write(data)


def export_model(source, output):
    with (
        source.open("rb") as file,
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        size = struct.unpack_from("<Q", mapped)[0]
        metadata = json.loads(mapped[8 : 8 + size])
        prefix = "model.language_model."
        tensors = {
            key.removeprefix(prefix): value
            for key, value in metadata.items()
            if key.startswith(prefix)
        }
        if len(tensors) != 320:
            raise ValueError(f"unexpected text tensor count: {len(tensors)}")
        temporary = output.with_suffix(".tmp")
        with temporary.open("wb") as dest:
            header(dest, len(tensors) + 1)
            record(dest, "qwen3_5.config", [16], struct.pack("<16f", *CONFIG))
            for name, tensor in sorted(tensors.items()):
                dtype = {"F32": 0, "BF16": 1}[tensor["dtype"]]
                start, end = tensor["data_offsets"]
                view = memoryview(mapped)[8 + size + start : 8 + size + end]
                record(dest, name, tensor["shape"], view, dtype)
                view.release()
        temporary.replace(output)
    print(
        f"Exported {len(tensors)} text tensors ({output.stat().st_size} bytes)",
        flush=True,
    )


def export_tokenizer(source, output):
    data = json.loads(source.read_text())
    if (
        data["normalizer"] != {"type": "NFC"}
        or data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] != PATTERN
    ):
        raise ValueError("unsupported tokenizer normalization or split expression")
    model = data["model"]
    if (
        model["type"] != "BPE"
        or model["byte_fallback"]
        or model["ignore_merges"]
        or model["dropout"] is not None
    ):
        raise ValueError("unsupported BPE options")
    # The GPT-2 byte alphabet assigns missing bytes to code points starting at 256.
    inverse = {
        chr(byte): byte
        for byte in list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    }
    extra = 0
    for byte in range(256):
        if byte not in inverse.values():
            inverse[chr(256 + extra)] = byte
            extra += 1
    vocab = model["vocab"]
    tokens = {
        index: bytes(inverse[char] for char in token) for token, index in vocab.items()
    }
    special = []
    for added in data["added_tokens"]:
        if (
            added.get("normalized", False)
            or added.get("lstrip", False)
            or added.get("rstrip", False)
            or added.get("single_word", False)
        ):
            raise ValueError("unsupported added-token options")
        tokens[added["id"]] = added["content"].encode()
        special.append((added["id"], int(added["special"])))
    if sorted(tokens) != list(range(len(tokens))):
        raise ValueError("token IDs must be contiguous")
    with output.open("wb") as file:
        file.write(
            b"D3BPE\0\0\0"
            + struct.pack("<IIII", 1, len(tokens), len(model["merges"]), len(special))
        )
        for index in range(len(tokens)):
            file.write(struct.pack("<I", len(tokens[index])))
            file.write(tokens[index])
        for merge in model["merges"]:
            left, right = merge.split(" ") if isinstance(merge, str) else merge
            file.write(
                struct.pack("<III", vocab[left], vocab[right], vocab[left + right])
            )
        for token, is_special in special:
            file.write(struct.pack("<II", token, is_special))
    print(f"Exported tokenizer ({len(tokens)} tokens)", flush=True)


def reference(
    weights, tokenizer_path, tokenizer_config, config_path, output, prompt, steps
):
    import numpy as np
    import torch
    import transformers
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    upstream_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        chat_template=json.loads(tokenizer_config.read_text())["chat_template"],
    )
    chat = upstream_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(chat, add_special_tokens=False).ids
    config = Qwen3_5TextConfig(**json.loads(config_path.read_text())["text_config"])
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    with safe_open(weights, framework="pt", device="cpu") as source:
        names = source.keys()
        state = {
            "model." + key.removeprefix("model.language_model."): source.get_tensor(
                key
            ).float()
            for key in names
            if key.startswith("model.language_model.")
        }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    model.load_state_dict(state, strict=True, assign=True)
    # Nonpersistent RoPE buffers are absent from the checkpoint. Initialize them on CPU.
    model.model.rotary_emb = type(model.model.rotary_emb)(config)
    model.eval()
    hidden = [[] for _ in range(config.num_hidden_layers)]
    handles = []
    for index, layer in enumerate(model.model.layers):

        def capture(module, args, value, index=index):
            if isinstance(value, tuple):
                value = value[0]
            hidden[index].append(value[0].detach().numpy().copy())

        handles.append(layer.register_forward_hook(capture))
    with torch.inference_mode():
        result = model(torch.tensor([prompt_ids]), use_cache=True)
        logits = [result.logits[0].numpy().copy()]
        tokens = prompt_ids.copy()
        generated = []
        for _ in range(steps):
            token = int(result.logits[0, -1].argmax())
            generated.append(token)
            tokens.append(token)
            result = model(
                torch.tensor([[token]]),
                past_key_values=result.past_key_values,
                use_cache=True,
            )
            logits.append(result.logits[0].numpy().copy())
    for handle in handles:
        handle.remove()
    tensors = {
        "tokens": np.array(tokens, dtype="<f4"),
        "logits": np.concatenate(logits),
        **{
            f"layer.{index}": np.concatenate(values)
            for index, values in enumerate(hidden)
        },
    }
    with output.open("wb") as file:
        header(file, len(tensors))
        for name, values in tensors.items():
            values = np.asarray(values, dtype="<f4", order="C")
            record(file, name, values.shape, values.tobytes())
    # Include diverse tokenizer cases, independently encoded by the upstream library.
    samples = [
        prompt,
        chat,
        "cafe\u0301 café 中文 🦀",
        "I'm WE'RE you've he'll I'd 'ſaa",
        "  abc   def\n\n  ghi\r\n",
        "123٤٥Ⅷ² 你好！\n",
        "a" * 513,
        "!?" * 129,
        "<|im_start|>user\nhello<|im_end|>",
    ]
    import random

    rng = random.Random(715)
    alphabet = "abAZ 0123\n\r\t'!?é中\u0301🦀\u2003٤²"
    samples += [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 80)))
        for _ in range(96)
    ]
    with output.with_name("tokenizer-fixture.bin").open("wb") as file:
        file.write(struct.pack("<I", len(samples)))
        for index, text in enumerate(samples):
            chat_case = index == 1
            raw = (prompt if chat_case else text).encode()
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            file.write(
                struct.pack("<III", len(raw), len(ids), int(chat_case))
                + raw
                + struct.pack("<" + "I" * len(ids), *ids)
            )
    stop = next(
        (index for index, token in enumerate(generated) if token in (248044, 248046)),
        len(generated),
    )
    return {
        "reference_continuation_tokens": generated,
        "stopped_on_eos": stop < len(generated),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "prompt": prompt,
        "prompt_tokens": prompt_ids,
        "generated_tokens": generated[:stop],
        "continuation": tokenizer.decode(generated[:stop]),
        "verified_positions": len(tokens),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/qwen")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="export assets without installing Torch or Transformers",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 64:
        parser.error("--steps must be from 1 to 64")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or args.output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    weights = download(cache, WEIGHTS, WEIGHTS_SHA)
    tokenizer = download(cache, "tokenizer.json", TOKENIZER_SHA)
    config = download(cache, "config.json", CONFIG_SHA)
    tokenizer_config = download(cache, "tokenizer_config.json", TOKENIZER_CONFIG_SHA)
    export_model(weights, args.output / "qwen3.5-0.8b.d3t")
    export_tokenizer(tokenizer, args.output / "qwen3.5.bpe")
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "scope": "text-only; BF16 weights, FP32 arithmetic",
    }
    if not args.skip_reference:
        manifest.update(
            reference(
                weights,
                tokenizer,
                tokenizer_config,
                config,
                args.output / "qwen-fixture.d3t",
                args.prompt,
                args.steps,
            )
        )
    manifest["artifacts"] = {
        path.name: digest(path)
        for path in args.output.iterdir()
        if path.suffix in (".d3t", ".bpe", ".bin")
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
