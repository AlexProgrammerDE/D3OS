#!/usr/bin/env python3
"""Export official AlexNet weights and an independent NumPy reference fixture."""

import argparse
import collections
import hashlib
import io
import json
import pickle
import struct
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

WEIGHTS_URL = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
WEIGHTS_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
DOG_URL = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
DOG_SHA256 = "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a"


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download(url, destination, expected=None):
    if not destination.exists() or (expected and sha256(destination) != expected):
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if expected and sha256(temporary) != expected:
            raise ValueError(f"checksum mismatch: {url}")
        temporary.replace(destination)
    return destination


class CheckpointReader(pickle.Unpickler):
    """Only reconstruct float tensors from the checksum-pinned official checkpoint."""

    def __init__(self, archive):
        self.archive = archive
        super().__init__(io.BytesIO(archive.read("archive/data.pkl")))

    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): "float32",
            ("torch._utils", "_rebuild_tensor_v2"): self.tensor,
            ("torch._utils", "_rebuild_parameter"): lambda data, requires_grad, hooks: (
                data
            ),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                f"unsupported checkpoint type: {module}.{name}"
            )
        return allowed[module, name]

    def persistent_load(self, identifier):
        kind, dtype, key, location, size = identifier
        if kind != "storage" or dtype != "float32" or location != "cpu":
            raise pickle.UnpicklingError("unsupported tensor storage")
        data = self.archive.read(f"archive/data/{key}")
        if len(data) != size * 4:
            raise pickle.UnpicklingError("invalid tensor storage length")
        return data

    @staticmethod
    def tensor(storage, offset, shape, strides, requires_grad, hooks):
        return np.ndarray(
            shape,
            dtype="<f4",
            buffer=storage,
            offset=offset * 4,
            strides=tuple(s * 4 for s in strides),
        )


def write_archive(destination, tensors):
    temporary = destination.with_suffix(".partial")
    with temporary.open("wb") as output:
        output.write(b"D3TENSR\0" + struct.pack("<II", 1, len(tensors)))
        for name, tensor in tensors.items():
            values = np.asarray(tensor, dtype="<f4", order="C")
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite tensor: {name}")
            encoded = name.encode("utf-8")
            output.write(struct.pack("<HHQ", len(encoded), values.ndim, values.size))
            output.write(encoded)
            output.write(struct.pack("<" + "Q" * values.ndim, *values.shape))
            output.write(values.tobytes())
    temporary.replace(destination)


def conv2d(image, weight, bias, stride, padding):
    kh, kw = weight.shape[2:]
    image = np.pad(image, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(image, (kh, kw), axis=(2, 3))[
        :, :, ::stride, ::stride
    ]
    batch, _, height, width, _, _ = windows.shape
    columns = windows.transpose(0, 1, 4, 5, 2, 3).reshape(batch, -1, height * width)
    result = weight.reshape(weight.shape[0], -1) @ columns
    return result.reshape(batch, -1, height, width) + bias[None, :, None, None]


def max_pool(image):
    windows = np.lib.stride_tricks.sliding_window_view(image, (3, 3), axis=(2, 3))[
        :, :, ::2, ::2
    ]
    return windows.max(axis=(-2, -1))


def reference(weights, image):
    observed = collections.OrderedDict(input=image)
    for layer, stride, padding, pool in [
        (0, 4, 2, True),
        (3, 1, 2, True),
        (6, 1, 1, False),
        (8, 1, 1, False),
        (10, 1, 1, True),
    ]:
        image = np.maximum(
            conv2d(
                image,
                weights[f"features.{layer}.weight"],
                weights[f"features.{layer}.bias"],
                stride,
                padding,
            ),
            0,
        )
        observed[f"features.{layer}"] = image
        if pool:
            image = max_pool(image)
    image = image.reshape(1, 9216)
    for layer in (1, 4, 6):
        image = (
            image @ weights[f"classifier.{layer}.weight"].T
            + weights[f"classifier.{layer}.bias"]
        )
        if layer != 6:
            image = np.maximum(image, 0)
        observed["logits" if layer == 6 else f"classifier.{layer}"] = image
    return observed


def preprocess(path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    resized = (
        (int(width * 256 / height), 256)
        if width > height
        else (256, int(height * 256 / width))
    )
    image = image.resize(resized, Image.Resampling.BILINEAR)
    left, top = round((image.width - 224) / 2), round((image.height - 224) / 2)
    image = np.asarray(
        image.crop((left, top, left + 224, top + 224)), dtype=np.float32
    ).transpose(2, 0, 1) / np.float32(255)
    image = image - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    image /= np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return image[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("target/inference"))
    parser.add_argument(
        "--image",
        type=Path,
        help="use a local image instead of the PyTorch dog photograph",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = download(
        WEIGHTS_URL, args.output / "alexnet-owt-7be5be79.pth", WEIGHTS_SHA256
    )
    with zipfile.ZipFile(checkpoint) as archive:
        weights = CheckpointReader(archive).load()
    image_path = args.image or download(DOG_URL, args.output / "dog.jpg", DOG_SHA256)
    fixture = reference(weights, preprocess(image_path))
    model_path, fixture_path = args.output / "alexnet.d3t", args.output / "dog.d3t"
    write_archive(model_path, weights)
    write_archive(fixture_path, fixture)
    logits = fixture["logits"][0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    classes = np.argsort(-probabilities)[:5]
    manifest = {
        "architecture": "torchvision-alexnet-eval-224",
        "weights_url": WEIGHTS_URL,
        "weights_sha256": WEIGHTS_SHA256,
        "image_sha256": sha256(image_path),
        "model_sha256": sha256(model_path),
        "fixture_sha256": sha256(fixture_path),
        "reference": "independent NumPy float32 operators",
        "numpy_version": np.__version__,
        "pillow_version": pillow_version,
        "top5": [
            {"class": int(c), "probability": float(probabilities[c])} for c in classes
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
