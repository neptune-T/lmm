"""Keep semantic-IK rejects out of training without hiding infrastructure faults."""
import hashlib
import json
from pathlib import Path

KEYS = ('lowering', 'joint_tracking', 'fk_next_state_matches_label', 'gripper', 'base')

def semantic_reject(chain):
    failed = [k for k in KEYS if not chain[k]['pass']]
    return failed == ['fk_next_state_matches_label']

def quarantine(root, rejected):
    """Preserve original result bytes, then exclude entire failed trajectories atomically.

    Raw observation/action files and original replay acceptance are never modified.
    """
    root = Path(root)
    provenance = root / 'raw_qc_provenance'
    provenance.mkdir(exist_ok=True)
    for path in sorted(root.glob('results*.jsonl')):
        original = path.read_bytes()
        rows = [json.loads(l) for l in original.splitlines() if l]
        changed = False
        for row in rows:
            if row['episode_id'] in rejected and row.get('accepted'):
                row['replay_accepted_before_raw_qc'] = True
                row['accepted'] = False
                row['raw_qc_rejection'] = rejected[row['episode_id']]
                changed = True
        if changed:
            digest = hashlib.sha256(original).hexdigest()
            saved = provenance / (path.name + '.' + digest + '.original')
            if not saved.exists():
                saved.write_bytes(original)
            tmp = path.with_suffix('.jsonl.tmp')
            tmp.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            tmp.replace(path)
