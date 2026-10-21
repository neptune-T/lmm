import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import rebuild_train_lj as driver


class PipelineTests(unittest.TestCase):
    def test_user_hold_blocks_start_and_commands(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, "RUN", Path(tmp)), \
                patch.object(driver.subprocess, "Popen") as popen:
            (Path(tmp) / "TRAINING_PAUSED_BY_USER.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "Training paused by user"):
                driver.main()
            with self.assertRaisesRegex(RuntimeError, "Training paused by user"):
                driver.command(Path(tmp), "train", ["never-run"])
            popen.assert_not_called()

    def manifest(self):
        m = {"dataset_version": "lj_recollect_20261020-test", "excluded_episodes": []}
        for split, n in (("train", 2000), ("val", 100)):
            m[split] = [f"{split}-{i}" for i in range(n)]
            m[f"n_{split}"] = n
            m[f"{split}_detail"] = [
                {"unit": u, "episode": u, "code_commit": "frozen"} for u in m[split]]
        return m

    def test_rejects_leakage_and_old_manifest(self):
        m = self.manifest()
        driver.validate_manifest(m, {"target_reached": True})
        bad = copy.deepcopy(m)
        bad["val_detail"][0]["episode"] = bad["train_detail"][0]["episode"]
        with self.assertRaises(RuntimeError):
            driver.validate_manifest(bad, {"target_reached": True})
        bad = copy.deepcopy(m)
        bad["dataset_version"] = "v2-9a54b0ff11a4"
        with self.assertRaises(RuntimeError):
            driver.validate_manifest(bad, {"target_reached": True})

    def test_rejects_incomplete_pool_and_qualification_leak(self):
        m = self.manifest()
        with self.assertRaises(RuntimeError):
            driver.validate_manifest(m, {"target_reached": False})
        m["excluded_episodes"] = [m["train_detail"][0]["episode"]]
        with self.assertRaises(RuntimeError):
            driver.validate_manifest(m, {"target_reached": True})

    def test_backup_failure_does_not_rerun_completed_work(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, "OUT", Path(tmp)), \
                patch.object(driver, "event"), patch.object(driver, "receipt_ok", return_value=False):
            calls = []
            def work(d):
                calls.append(d)
            with patch.object(driver, "backup", side_effect=RuntimeError("offline")):
                with self.assertRaises(RuntimeError):
                    driver.stage("closed", work)
            with patch.object(driver, "backup") as backup:
                driver.stage("closed", work)
                backup.assert_called_once()
            self.assertEqual(len(calls), 1)

    def test_failed_stage_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, "OUT", Path(tmp)), \
                patch.object(driver, "event"), patch.object(driver, "backup"):
            def fail(d):
                (d / "evidence").write_text("keep")
                raise RuntimeError("bad data")
            with self.assertRaises(RuntimeError):
                driver.stage("bad", fail)
            with self.assertRaisesRegex(RuntimeError, "Preserved incomplete"):
                driver.stage("bad", lambda d: self.fail("must not retry"))
            self.assertEqual((Path(tmp) / "stages/bad/evidence").read_text(), "keep")

    def test_gpu_shortage_waits_without_launch(self):
        resources = {"ram_available": 200 * 2**30, "persistent_free": 1000 * 2**30,
                     "free_mib": 99 * 1024}
        with patch.object(driver.collection, "resources", return_value=resources):
            self.assertFalse(driver.require_capacity(True)[0])
            self.assertTrue(driver.require_capacity(False)[0])


if __name__ == "__main__":
    unittest.main()
