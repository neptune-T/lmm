import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pack_hssd_backup_lj import groups, verify_tar


class ArchiveIntegrity(unittest.TestCase):
    def test_groups_preserve_all_files(self):
        records = [{"path": str(i), "size": 3} for i in range(5)]
        result = list(groups(records, 7))
        self.assertEqual([len(x) for x in result], [2, 2, 1])
        self.assertEqual([r for batch in result for r in batch], records)

    def test_member_hash_missing_and_unexpected_files(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.tar"
            with tarfile.open(p, "w:") as archive:
                member = tarfile.TarInfo("objects/a")
                member.size = 4
                archive.addfile(member, io.BytesIO(b"good"))
            record = {"path": "objects/a", "size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
            verify_tar(p, [record])
            with self.assertRaisesRegex(RuntimeError, "content mismatch"):
                verify_tar(p, [{**record, "sha256": hashlib.sha256(b"evil").hexdigest()}])
            with self.assertRaisesRegex(RuntimeError, "missing"):
                verify_tar(p, [record, {**record, "path": "objects/missing"}])
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                verify_tar(p, [])


if __name__ == "__main__":
    unittest.main()
