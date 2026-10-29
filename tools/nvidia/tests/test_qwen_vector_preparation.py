"""Filesystem publication controls; these tests make no compiler or GPU claim."""

import importlib.util
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load("core_preparation", "prepare-kernels.py")
vector = load("vector_preparation", "prepare-qwen-vector.py")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.staging = self.root / "staging"
        self.output = self.root / "output"
        self.staging.mkdir()
        self.manifest = {"kernels": {"qwen-vector": {"input": "qwen-vector.ptx"}}}
        (self.staging / "manifest.json").write_text(json.dumps(self.manifest))
        (self.staging / "qwen-vector.ptx").write_bytes(bytes(range(16)))

    def test_new_bundle_preserves_nested_manifests_and_publishes_all_bytes(self):
        nested = self.staging / "assets"
        nested.mkdir()
        (nested / "manifest.json").write_bytes(bytes(range(31)))
        vector.publish_bundle(self.staging, self.output)
        for source in self.staging.rglob("*"):
            if source.is_file():
                self.assertEqual(
                    source.read_bytes(),
                    (self.output / source.relative_to(self.staging)).read_bytes(),
                )

    def test_existing_bundle_is_untouched_when_output_claim_fails(self):
        self.output.mkdir()
        previous = bytes(range(19))
        (self.output / "manifest.json").write_bytes(previous)
        with self.assertRaises(FileExistsError):
            vector.publish_bundle(self.staging, self.output)
        self.assertEqual((self.output / "manifest.json").read_bytes(), previous)

    def test_publishers_reject_recursive_outputs_before_writing(self):
        nested = self.staging / "nested"
        with self.assertRaises(ValueError):
            vector.publish_bundle(self.staging, nested)
        with self.assertRaises(ValueError):
            core.publish_bundle(self.staging, nested, self.manifest, audit=True)
        self.assertFalse(nested.exists())

    def test_interrupted_new_bundle_has_no_success_manifest(self):
        with (
            patch.object(vector.shutil, "copytree", side_effect=OSError),
            self.assertRaises(OSError),
        ):
            vector.publish_bundle(self.staging, self.output)
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(
            json.loads((self.staging / "manifest.json").read_text()), self.manifest
        )

    def test_declared_ptx_survives_and_machine_code_stays_in_host_audit(self):
        binary = bytes(range(41))
        (self.staging / "inference.cubin").write_bytes(binary)
        (self.staging / "inference.sass").write_bytes(binary[::-1])
        (self.staging / "obsolete.ptx").write_bytes(binary)
        (self.staging / "manifest.json").chmod(0o640)
        core.publish_bundle(self.staging, self.output, self.manifest, audit=True)
        self.assertEqual(
            (self.output / "qwen-vector.ptx").read_bytes(), bytes(range(16))
        )
        self.assertEqual(
            (self.output / "host-audit/inference.cubin").read_bytes(), binary
        )
        self.assertFalse((self.output / "inference.cubin").exists())
        self.assertFalse((self.output / "obsolete.ptx").exists())
        self.assertEqual(
            json.loads((self.output / "manifest.json").read_text()), self.manifest
        )
        self.assertEqual(
            stat.S_IMODE((self.output / "manifest.json").stat().st_mode), 0o640
        )

    def test_core_only_removes_stale_vector_inputs_and_audits(self):
        audit = self.output / "host-audit"
        audit.mkdir(parents=True)
        for name in ("qwen-vector.cubin", "qwen-vector.sass"):
            (audit / name).write_bytes(bytes(range(7)))
        (self.output / "qwen-vector.ptx").write_bytes(bytes(range(11)))
        (self.staging / "qwen-vector.ptx").unlink()
        core.publish_bundle(self.staging, self.output, {"kernels": {}}, audit=False)
        self.assertFalse((self.output / "qwen-vector.ptx").exists())
        self.assertEqual(list(audit.iterdir()), [])

    def test_failed_core_copy_keeps_the_previous_manifest(self):
        self.output.mkdir()
        previous = bytes(range(23))
        (self.output / "manifest.json").write_bytes(previous)
        with (
            patch.object(core.shutil, "copyfile", side_effect=OSError),
            self.assertRaises(OSError),
        ):
            core.publish_bundle(self.staging, self.output, self.manifest, audit=True)
        self.assertEqual((self.output / "manifest.json").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
