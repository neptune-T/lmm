"""Recovery boundaries: failed gates and missing backups must never become completion."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "rebuild_collect_lj", Path(__file__).parents[1] / "scripts/rebuild_collect_lj.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class RecoveryBoundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        (self.out / "receipts").mkdir()
        self.p = patch.multiple(M, OUT=self.out, COMMIT="fixture", create=True)
        self.p.start()
        self.addCleanup(self.p.stop)

    def verified(self, d):
        M.dump(self.out / "receipts" / f"{d.name}.json",
               {"repo": "Neptune-T/lmm-alphabot2-ovmm-runs", "revision": "fixture",
                "files": 2, "verified": 2, "mismatch": []})

    def test_failed_gate_is_backed_up_but_never_skipped_as_pass(self):
        def fail(d):
            (d / "evidence.txt").write_text("failed check")
            raise RuntimeError("gate failed")
        with patch.object(M, "backup", side_effect=self.verified) as backup:
            with self.assertRaisesRegex(RuntimeError, "gate failed"):
                M.stage("one", fail)
            backup.assert_called_once()
            self.assertFalse(json.loads((self.out / "stages/one/result.json").read_text())["pass"])
            with self.assertRaisesRegex(RuntimeError, "needs diagnosis"):
                M.stage("one", lambda _: self.fail("Failed work must not retry automatically"))

    def test_backup_failure_retries_only_backup_without_reexecuting_work(self):
        with patch.object(M, "backup", side_effect=RuntimeError("HF unavailable")):
            with self.assertRaisesRegex(RuntimeError, "HF unavailable"):
                M.stage("one", lambda d: (d / "result.txt").write_text("completed"))
        with patch.object(M, "backup", side_effect=self.verified) as backup:
            M.stage("one", lambda _: self.fail("Completed work must not execute twice"))
            backup.assert_called_once()

    def test_mismatched_receipt_cannot_skip_stage(self):
        d = self.out / "stages/one"
        M.dump(d / "result.json", {"pass": True})
        self.verified(d)
        p = self.out / "receipts/one.json"
        r = json.loads(p.read_text())
        r["mismatch"] = ["wrong-content"]
        M.dump(p, r)
        with self.assertRaisesRegex(RuntimeError, "Invalid verified receipt"):
            M.stage("one", lambda _: self.fail("Invalid receipt must block"))

    def test_geometry_order_is_stable_before_outcomes(self):
        rs = [{"episode_id": str(i), "scene": str(i % 3)} for i in range(30)]
        a = M.scene_order(rs)
        self.assertEqual(a, M.scene_order(list(reversed(rs))))
        self.assertEqual(len({r["scene"] for r in a[:3]}), 3)


if __name__ == "__main__":
    unittest.main()
