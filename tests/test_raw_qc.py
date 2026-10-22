import unittest,tempfile,json
from pathlib import Path
from lmm.embodiments.alphabot2.raw_qc import KEYS,semantic_reject,quarantine
class RawQC(unittest.TestCase):
 def test_only_semantic_failure_is_rejectable(self):
  chain={k:{'pass':True} for k in KEYS}
  self.assertFalse(semantic_reject(chain))
  chain['fk_next_state_matches_label']['pass']=False
  self.assertTrue(semantic_reject(chain))
  for key in KEYS:
   if key=='fk_next_state_matches_label':continue
   bad={k:dict(v) for k,v in chain.items()};bad[key]['pass']=False
   self.assertFalse(semantic_reject(bad))
 def test_preserves_original_and_excludes_only_rejected_unit(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);p=root/'results_w00.jsonl'
   original=''.join(json.dumps({'episode_id':s,'accepted':True})+'\n' for s in ['bad','good'])
   p.write_text(original); raw=root/'images.npz';raw.write_bytes(b'unchanged')
   quarantine(root,{'bad':{'reason':'cartesian_target_not_executed'}})
   rows=[json.loads(l) for l in p.read_text().splitlines()]
   self.assertFalse(rows[0]['accepted']);self.assertTrue(rows[0]['replay_accepted_before_raw_qc']);self.assertTrue(rows[1]['accepted'])
   saved=list((root/'raw_qc_provenance').iterdir());self.assertEqual(saved[0].read_text(),original)
   quarantine(root,{'bad':{'reason':'cartesian_target_not_executed'}})
   self.assertEqual(len(list((root/'raw_qc_provenance').iterdir())),1)
   self.assertEqual(raw.read_bytes(),b'unchanged')
if __name__=='__main__':unittest.main()
