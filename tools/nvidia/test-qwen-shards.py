#!/usr/bin/env python3
"""Verify bounded model preparation, repair, and filesystem-copy fallback."""

import errno
import hashlib
import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "qwen_shards", Path(__file__).with_name("prepare-qwen-shards.py")
)
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)
SHARDS.PART_BYTES = 1 << 20


class ShardsTest(unittest.TestCase):
    def test_prepare_repairs_corrupt_parts_and_preserves_complete_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = bytes(range(256)) * 8192 + b"last partial part"
            source = root / "source.gguf"
            source.write_bytes(data)
            expected = hashlib.sha256(data).hexdigest()
            output = root / "parts"
            manifest = SHARDS.prepare(source, output, expected)
            self.assertEqual(manifest["sha256"], expected)
            self.assertEqual(len(manifest["parts"]), 3)
            index = (output / "model.index").read_bytes()
            self.assertEqual(
                struct.unpack_from("<IIQI", index, 8), (1, 1 << 20, len(data), 3)
            )
            self.assertEqual(index[28:], bytes.fromhex(expected))
            (output / "model.001").write_bytes(b"corrupt")
            SHARDS.prepare(source, output, expected)
            restored = b"".join(
                (output / part["name"]).read_bytes() for part in manifest["parts"]
            )
            self.assertEqual(restored, data)
            self.assertEqual(source.read_bytes(), data)
            with self.assertRaises(ValueError):
                SHARDS.prepare(source, output, "00" * 32)

    def test_copy_fallback_obeys_offset_length_and_rejects_early_eof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.write_bytes(bytes(range(256)) * 32)
            with (
                source.open("rb") as file,
                patch.object(
                    SHARDS.fcntl,
                    "ioctl",
                    side_effect=OSError(errno.EOPNOTSUPP, "unsupported"),
                ),
            ):
                SHARDS.copy_part(file, root / "part", 103, 1027)
                self.assertEqual(
                    (root / "part").read_bytes(), source.read_bytes()[103:1130]
                )
                with self.assertRaises(ValueError):
                    SHARDS.copy_part(file, root / "truncated", 8190, 4)


if __name__ == "__main__":
    unittest.main()
