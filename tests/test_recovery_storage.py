"""Regression checks for disk-full backup and storage-only strict resume."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/alphabot2_pi05'))
from backup_storage import checkpoint_environment, check_snapshot_capacity
from storage_relocation import relocate


class RecoveryStorageTests(unittest.TestCase):
    def test_large_snapshot_rejects_root_before_copy(self):
        with tempfile.TemporaryDirectory() as root:
            if Path(root).stat().st_dev != Path('/').stat().st_dev:
                self.skipTest('tmp is a separate mount')
            with self.assertRaisesRegex(RuntimeError, 'root filesystem'):
                check_snapshot_capacity(root, 42 << 30)

    def test_storage_space_guard(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('backup_storage.shutil.disk_usage') as usage:
                usage.return_value.free = 1
                with self.assertRaisesRegex(RuntimeError, 'Insufficient'):
                    check_snapshot_capacity(root, 20)

    def test_worker_sets_stage_even_without_parent_environment(self):
        with tempfile.TemporaryDirectory() as root:
            original_stat = Path.stat
            def stat(path, *a, **kw):
                s = original_stat(path, *a, **kw)
                if str(path).endswith('backups/hf_stage'):
                    from types import SimpleNamespace
                    return SimpleNamespace(st_dev=s.st_dev + 1)
                return s
            with patch.dict(os.environ, {}, clear=True), patch.object(Path, 'stat', stat):
                env = checkpoint_environment(root)
            self.assertEqual(env['HF_SYNC_STAGE'], str(Path(root).resolve()/'backups/hf_stage'))
            self.assertTrue(env['HF_XET_CACHE'].startswith(root))

    def test_relocation_cannot_change_recipe_or_source(self):
        c = {'out':'/old/run','train_data':'/old/data','seed':1000,'steps':10000}
        self.assertEqual(relocate(c, {'out':{'from':'/old/run','to':'/new/run'}})['out'], '/new/run')
        self.assertEqual(c['out'], '/old/run')
        for mapping in [{'seed':{'from':1000,'to':1}},
                        {'out':{'from':'/wrong','to':'/new'}},
                        {'out':{'from':'/old/run','to':'relative'}}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                relocate(c, mapping)


if __name__ == '__main__': unittest.main()
