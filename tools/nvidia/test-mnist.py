#!/usr/bin/env python3
"""Check guest artifact framing, corruption detection, and interrupted exports."""

import binascii
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_mnist", Path(__file__).with_name("run-mnist.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def artifact(payload):
    data = (
        struct.pack("<8sIIQQ", b"D3MNIST1", 1, 36, len(payload), 0)
        + bytes(36)
        + payload
    )
    return data + struct.pack("<I", binascii.crc32(data))


def frames(data):
    name = "mnist.checkpoint"
    result = [
        f"MNIST_ARTIFACT_BEGIN name={name} bytes={len(data)} crc32={binascii.crc32(data):08x}"
    ]
    for offset in range(0, len(data), 256):
        result.append(
            f"MNIST_ARTIFACT_DATA name={name} offset={offset} hex={data[offset : offset + 256].hex()}"
        )
    result.append(f"MNIST_ARTIFACT_END name={name}")
    return result


class ArtifactTransport(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.data = artifact(bytes(i % 256 for i in range(600)))

    def test_complete_export_preserves_every_byte(self):
        completed, partial = runner.extract_artifacts(
            "\n".join(frames(self.data)), self.output
        )
        self.assertEqual((self.output / "mnist.checkpoint").read_bytes(), self.data)
        self.assertEqual(completed["mnist.checkpoint"]["bytes"], len(self.data))
        self.assertEqual(partial, [])

    def test_missing_repeated_reordered_and_corrupt_chunks_fail(self):
        original = frames(self.data)
        invalid = [
            original[:2] + original[3:],
            original[:2] + original[1:],
            [original[0], original[2], original[1], *original[3:]],
        ]
        corrupted = artifact(bytes(i % 256 for i in range(600)))
        corrupt_frames = frames(corrupted)
        corrupt_frames[1] = corrupt_frames[1][:-2] + (
            "00" if corrupt_frames[1][-2:] != "00" else "01"
        )
        invalid.append(corrupt_frames)
        for transcript in invalid:
            with self.subTest(transcript=transcript[:1]):
                with self.assertRaises(RuntimeError):
                    runner.extract_artifacts("\n".join(transcript), self.output)
                self.assertFalse((self.output / "mnist.checkpoint").exists())

    def test_interrupted_new_checkpoint_preserves_the_last_complete_one(self):
        old = frames(self.data)
        replacement = frames(artifact(bytes([17]) * 700))
        completed, partial = runner.extract_artifacts(
            "\n".join(old + replacement[:2]), self.output
        )
        self.assertEqual((self.output / "mnist.checkpoint").read_bytes(), self.data)
        self.assertEqual(completed["mnist.checkpoint"]["bytes"], len(self.data))
        self.assertEqual(len(partial), 1)


if __name__ == "__main__":
    unittest.main()
