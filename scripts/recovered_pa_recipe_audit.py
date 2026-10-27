"""Compare the live reconstructed SFT contract with the historical v2-B log.

Freezes the subsequent PA-OG evaluation protocol. Does not execute rollouts,
claim new success rates, change training, or add M-SFT/RL.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--old-log',required=True)
    a=ap.parse_args(); root=Path(a.run); code=Path(__file__).resolve().parents[1]
    oldpath=Path(a.old_log)
    old=next(json.loads(line[5:]) for line in oldpath.read_text().splitlines() if line.startswith('INFO '))
    current=json.loads((root/'sft/v2B/run_info_step000000.json').read_text())
    keys=['policy_io','history','state_dim','image_features','loss_cfg','global_batch',
          'train_episodes','train_frames','val_frames','trainable_params','delta_timestamps']
    checks={key:current[key]==old[key] for key in keys}
    for key in ['effective_mass_by_stage','nav_omega_class_mass']:
        x,y=current['sampling'][key],old['sampling'][key]
        checks['sampling/'+key]=x.keys()==y.keys() and all(abs(x[k]-y[k])<1e-9 for k in x)
    source=code/'docs/v2/results/pa_report_data_2026-10-19.json'
    report=json.loads(source.read_text())
    baseline=[p for p in report['passes'] if p['dir']=='pa_v2B']
    assert sum(p['eval_count'] for p in baseline)==60 and sum(p['funnel']['full'] for p in baseline)==15
    units={}
    for group in ['pa_v2B','pa_v2B_heldout','pa_v2B_val']:
        rows=[p for p in report['passes'] if p['dir']==group]
        # Historical workers finish in different orders; compare unit identities,
        # retaining every unit and rejecting duplicates before canonical sorting.
        assert rows and all(len(r['units'])==len(set(r['units'])) for r in rows)
        assert all(set(r['units'])==set(rows[0]['units']) for r in rows)
        units[group]=dict(units=sorted(rows[0]['units']),seeds=[r['seed'] for r in rows],split=rows[0]['split'])
    protocol={'label':'PA-OG + Base Lock','policy_selection':'lowest fixed SFT-VAL loss best checkpoint',
              'BASE_LOCK':1,'MANIP1_TRIES':1,'MANIP2_TRIES':1,'MANIP1_MAX':600,'MANIP2_MAX':500,
              'SETTLE_STEPS':120,'HOLD_GAP':0.05,'TUCK_RAD':0.30,'HOLD_K':3,
              'dtype':'float32','max_batch':1,'EXEC_STEPS':5,'history':current['history'],
              'phase_switches':'planner arrival; observable holding and tucked state, never expert stage labels',
              'units_from_historical_report':units,'auto_add_msft_or_rl':False,
              'acceptance':'non-zero full successes must be replay-verified; report held-out and OVMM-val separately',
              'execution_status':'not_run_on_recovered_checkpoint'}
    record={'checks':checks,'passed':all(checks.values()),'old_log_sha256':hashlib.sha256(oldpath.read_bytes()).hexdigest(),
            'historical_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'historical_only_success':'15/60 training-unit PA-OG + base lock; not a current-run result',
            'intentional_execution_differences':{'old_world':old.get('world'),'new_world':current['world'],
                'old_accum':old.get('grad_accum'),'new_accum':current['grad_accum'],
                'note':'same effective batch and recipe; different DDP sampling stream, not bitwise reproduction'},
            'evaluation_protocol':protocol}
    (root/'checks/pa_recipe_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    (root/'pa_eval_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'passed':record['passed'],'checks':checks}),flush=True)
    if not record['passed']:raise SystemExit(1)


if __name__=='__main__':main()
