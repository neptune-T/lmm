import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("hf_sync", Path(__file__).parents[1] / "scripts/hf_sync.py")
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class BackupIntegrity(unittest.TestCase):
    def test_same_size_stale_lfs_is_rejected(self):
        remote = SimpleNamespace(size=8, lfs=SimpleNamespace(sha256="old"))
        self.assertFalse(sync.verify_file(remote, 8, "new", "unused"))

    def test_same_size_stale_git_file_is_rejected(self):
        remote = SimpleNamespace(size=8, lfs=None, blob_id="old")
        self.assertFalse(sync.verify_file(remote, 8, "unused", "new"))

    def test_content_hashes_pass(self):
        self.assertTrue(sync.verify_file(SimpleNamespace(size=8, lfs=SimpleNamespace(sha256="sha")), 8, "sha", "blob"))
        self.assertTrue(sync.verify_file(SimpleNamespace(size=8, lfs=None, blob_id="blob"), 8, "sha", "blob"))
        self.assertFalse(sync.verify_file(None, 8, "sha", "blob"))

    def test_missing_input_cannot_be_successful_empty_backup(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                sync.files_of(Path(d) / "missing")

    def test_git_blob_hash_matches_git(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data"
            p.write_bytes(b"content\n")
            expected = subprocess.check_output(["git", "hash-object", str(p)], text=True).strip()
            self.assertEqual(sync.git_blob(p), expected)


if __name__ == "__main__":
    unittest.main()
