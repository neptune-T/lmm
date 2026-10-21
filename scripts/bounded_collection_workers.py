"""Opt-in GPU7-only closed-shard parallelism; no scheduler, training or GPU adoption.

Keeps the original worker partition code, and restores serial row order before gates.
Only applies AFTER the pilot gate and its HF receipt pass. Running pilot jobs are untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker_budget(free_mib, ram_bytes, requested):
    return max(1, min(4, int(requested), max(0, (free_mib - 8192) // 2048),
                      max(0, (ram_bytes - 32 * 2**30) // (16 * 2**30))))


def canonical_rows(rows, order):
    indexed = {str(r['episode_id']): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(order):
        raise RuntimeError('Parallel shard has missing, extra or duplicate episode rows')
    return [indexed[k] for k in order]


def launch_if_configured(script, kind):
    if os.environ.get('LMM_PARALLEL_CHILD') == '1' or not os.environ.get('OUT'):
        return
    out = Path(os.environ['OUT']).resolve()
    stage, collection = out.parent, out.parent.parent.parent
    if stage.parent.name != 'stages' or collection.name != 'collection':
        return
    if not ((kind == 'collect' and stage.name.startswith('collect_')) or
            (kind == 'screen' and stage.name.startswith('screen_'))):
        return
    config = collection / 'worker_config.json'
    receipt = collection / 'worker_config_verified.json'
    if not config.exists() or not receipt.exists():
        return
    import rebuild_collect_lj as driver
    if collection != driver.OUT or os.environ.get('GPU') != '7':
        raise RuntimeError('Parallel collection is restricted to this reconstruction on GPU7')
    if not driver.receipt_ok(receipt):
        raise RuntimeError('Worker configuration backup not verified')
    cfg = json.loads(config.read_text())
    if cfg.get('gpu') != 7 or cfg.get('training_authorized') is not False:
        raise RuntimeError('Invalid data-only worker configuration')
    ready = collection / 'stages/pilot_gate/result.json'
    if not ready.exists() or not json.loads(ready.read_text()).get('pass'):
        return
    if not driver.receipt_ok(collection / 'receipts/pilot_gate.json'):
        return
    if os.environ.get('CLAIM_DIR') or os.environ.get('MAX_OK'):
        raise RuntimeError('Parallel closed shards cannot inherit claims or MAX_OK')
    if os.environ.get('WORKER', '0') != '0' or os.environ.get('NWORKERS', '1') != '1':
        raise RuntimeError('Refusing nested worker partition')
    before = driver.resources()
    workers = worker_budget(before['free_mib'], before['ram_available'], cfg['max_workers'])
    if workers < 2:
        return
    if kind == 'collect':
        order = os.environ['EPISODES'].split(',')
        pattern = 'results_w*.jsonl'
    else:
        geometry = json.loads(Path(os.environ['SCREEN']).read_text())
        order = [str(r['episode_id']) for r in sorted(geometry, key=lambda r: (r['scene'], int(r['episode_id']))) if r['feasible']]
        pattern = 'emb_w*.jsonl'
    record = {'workers': workers, 'gpu': 7, 'kind': kind, 'training_authorized': False,
              'code_commit': subprocess.check_output(['git', '-C', str(driver.CODE), 'rev-parse', 'HEAD'], text=True).strip(),
              'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'resources_before': before,
              'started': time.time(), 'unit_order': order, 'children': []}
    driver.dump(stage / 'worker_execution.json', record)
    children = []
    for i in range(workers):
        while True:
            r = driver.resources()
            if r['free_mib'] >= 10240 and r['ram_available'] >= 48 * 2**30:
                break
            print('WAIT_WORKER_HEADROOM', i, json.dumps(r), flush=True)
            time.sleep(30)
        env = {**os.environ, 'LMM_PARALLEL_CHILD': '1', 'WORKER': str(i), 'NWORKERS': str(workers)}
        with (stage / f'{kind}_worker_{i:02d}.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(script)], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        children.append(child)
        record['children'].append({'worker': i, 'pid': child.pid})
        driver.dump(stage / 'worker_execution.json', record)
    with (stage / 'worker_memory.jsonl').open('w') as memory:
        while any(p.poll() is None for p in children):
            rss = {}
            for p in children:
                status = Path(f'/proc/{p.pid}/status')
                try:
                    rss[p.pid] = {line.split(':')[0]: line.split(':', 1)[1].strip()
                                  for line in status.read_text().splitlines() if line.startswith(('VmRSS:', 'VmHWM:'))}
                except FileNotFoundError:
                    pass
            memory.write(json.dumps({'time': time.time(), 'workers': rss, **driver.resources()}) + '\n')
            memory.flush()
            time.sleep(10)
    record['exit_codes'] = [p.wait() for p in children]
    record['finished'] = time.time()
    driver.dump(stage / 'worker_execution.json', record)
    if any(record['exit_codes']):
        raise RuntimeError('Collection worker failed; preserve all outputs for diagnosis')
    files = sorted(out.glob(pattern))
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines() if line]
    ordered = canonical_rows(rows, order)
    preserved = out / 'worker_results'
    preserved.mkdir()
    for f in files:
        f.rename(preserved / f.name)
    (out / ('results_w00.jsonl' if kind == 'collect' else 'emb_w00.jsonl')).write_text(
        ''.join(json.dumps(row) + '\n' for row in ordered))
    print('PARALLEL_SHARD_COMPLETE', json.dumps(record), flush=True)
    raise SystemExit(0)
