"""Exercise guest staging copies without opening a device or launching a guest."""

import errno
import importlib.util
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "linux_reference_copy", Path(__file__).resolve().parents[1] / "linux-reference.py"
)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class GuestCopyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=os.environ.get("D3OS_COPY_TEST_DIRECTORY")
        )
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source"
        self.payload = bytes(range(256)) * 128
        self.source.write_bytes(self.payload)
        self.source.chmod(0o751)
        os.utime(self.source, ns=(1_700_000_000_000_000_000,) * 2)
        self.root = self.directory / "guest"

    def check_copy(self):
        reference.copy_file(self.root, self.source, "/model.gguf")
        target = self.root / "model.gguf"
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o751)
        self.assertEqual(target.stat().st_mtime_ns, self.source.stat().st_mtime_ns)
        self.assertNotEqual(target.stat().st_ino, self.source.stat().st_ino)
        with target.open("r+b") as output:
            output.write(b"\xff" * 256)
        self.assertEqual(self.source.read_bytes(), self.payload)
        return target

    def test_copy_and_replacement_preserve_metadata_and_source(self):
        target = self.check_copy()
        target.chmod(0o600)
        self.check_copy()

    def test_library_symlink_copies_referenced_bytes(self):
        link = self.directory / "library.so"
        link.symlink_to(self.source)
        reference.copy_file(self.root, link, "/lib64/library.so")
        target = self.root / "lib64/library.so"
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o751)

    def test_cross_filesystem_and_unsupported_clone_fall_back(self):
        for code in (errno.EXDEV, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOTTY):
            with (
                self.subTest(errno=code),
                mock.patch.object(
                    reference.fcntl, "ioctl", side_effect=OSError(code, "clone")
                ),
            ):
                self.check_copy()

    def test_fallback_discards_partial_clone_contents(self):
        def partial_clone(output, _request, _source):
            os.write(output, b"\x9a" * (len(self.payload) * 2))
            raise OSError(errno.EOPNOTSUPP, "clone")

        with mock.patch.object(reference.fcntl, "ioctl", side_effect=partial_clone):
            self.check_copy()

    def test_other_clone_errors_propagate(self):
        with (
            mock.patch.object(
                reference.fcntl, "ioctl", side_effect=OSError(errno.EIO, "clone")
            ),
            self.assertRaises(OSError) as error,
        ):
            reference.copy_file(self.root, self.source, "/model.gguf")
        self.assertEqual(error.exception.errno, errno.EIO)
        self.assertEqual(self.source.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
