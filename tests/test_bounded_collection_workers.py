import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from bounded_collection_workers import worker_budget, canonical_rows

class WorkerTests(unittest.TestCase):
    def test_caps_gpu_ram_and_worker_count(self):
        self.assertEqual(worker_budget(23799, 1000 * 2**30, 4), 4)
        self.assertEqual(worker_budget(12000, 1000 * 2**30, 4), 1)
        self.assertEqual(worker_budget(23799, 64 * 2**30, 4), 2)
        self.assertEqual(worker_budget(200000, 1000 * 2**30, 40), 4)
    def test_preserves_frozen_order(self):
        rows = [{'episode_id': str(i)} for i in [3, 1, 2]]
        self.assertEqual([r['episode_id'] for r in canonical_rows(rows, ['1','2','3'])], ['1','2','3'])
    def test_incomplete_duplicate_extra_fail(self):
        for rows in [[{'episode_id':'1'}], [{'episode_id':'1'}, {'episode_id':'1'}],
                     [{'episode_id':'1'}, {'episode_id':'3'}]]:
            with self.assertRaises(RuntimeError):
                canonical_rows(rows,['1','2'])
