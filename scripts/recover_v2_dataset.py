"""CPU-only recovery audit of the existing, previously qualified v2 conversion.

This does NOT claim to repeat raw-simulator source alignment if raw NPZs are lost.
It copies bytes with SHA256 verification, decodes every video, checks episode
timing, labels, base-episode separation and historical gate evidence. Original
sources are never modified. All progress and per-video results are resumable.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)


def episodes(root):
    return pa.concat_tables([pq.read_table(p) for p in sorted((root / 'meta/episodes').rglob('*.parquet'))]).to_pylist()


def decode(job):
    import av
    path, windows, fps, sha, cache = job
    cache = Path(cache)
    if cache.exists():
        old = json.loads(cache.read_text())
        if old.get('sha256') == sha and old.get('pass'):
            return old
    with av.open(path) as c:
        stream = c.streams.video[0]
        stream.thread_count = 2
        times = np.asarray([float(f.time) for f in c.decode(stream)])
    errors = []
    for ep, lo, hi, length in windows:
        sel = times[(times >= lo - .5 / fps) & (times < hi - .5 / fps)]
        if len(sel) != length or (len(sel) > 1 and np.max(np.abs(np.diff(sel) - 1 / fps)) >= .25 / fps):
            errors.append(ep)
    expected = sum(x[3] for x in windows)
    result = dict(path=path, sha256=sha, decoded=len(times), expected=expected,
                  episodes=len(windows), bad_episodes=errors,
                  passed_at=time.time(), **{'pass': len(times) == expected and not errors})
    write(cache, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--staged', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    source, staged, out = map(Path, (a.source, a.staged, a.out))
    out.mkdir(parents=True, exist_ok=True)
    staged.mkdir(parents=True, exist_ok=True)
    (out / 'videos').mkdir(exist_ok=True)
    report = {'kind': 'recovered_existing_conversion', 'started': time.time(), 'datasets': {},
              'raw_source_alignment_revalidated': False,
              'scope': 'Existing v2 conversion; historical raw alignment retained as historical evidence, not re-run.'}
    manifest, bases, jobs = {}, {}, []
    for split, n in [('train', 2099), ('val', 102)]:
        name = 'alphabot2_ovmm_v2_' + split
        src, dst = source / 'datasets' / name, staged / name
        for f in sorted(src.rglob('*')):
            if not f.is_file():
                continue
            assert not f.is_symlink(), f
            rel = str(f.relative_to(src))
            sha = digest(f)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or digest(target) != sha:
                tmp = target.with_suffix(target.suffix + '.part')
                shutil.copyfile(f, tmp)
                assert digest(tmp) == sha, target
                tmp.replace(target)
            manifest[name + '/' + rel] = {'sha256': sha, 'bytes': f.stat().st_size}
        write(out / 'file_manifest.json', manifest)
        print('COPIED_VERIFIED', split, flush=True)
        info = json.loads((dst / 'meta/info.json').read_text())
        idx = json.loads((dst / 'ovmm_episode_index.json').read_text())
        assert idx['dataset_version'] == 'v2-9a54b0ff11a4'
        assert len(idx['episodes']) == info['total_episodes'] == n
        es = sorted(episodes(dst), key=lambda e: e['episode_index'])
        assert len(es) == n
        bases[split] = {str(e['episode']) for e in idx['episodes']}
        table = pa.concat_tables([pq.read_table(f) for f in sorted((dst / 'data').rglob('*.parquet'))])
        assert len(table) == info['total_frames'] == sum(e['frames'] for e in idx['episodes'])
        assert np.array_equal(table['index'].to_numpy(), np.arange(len(table)))
        for col, width in [('observation.state', 16), ('action', 18)]:
            values = table[col].combine_chunks()
            assert np.all(np.diff(values.offsets.to_numpy()) == width)
            assert np.isfinite(values.values.to_numpy()).all()
        fm = np.load(dst / 'frame_meta.npz')
        assert np.array_equal(fm['episode_index'], table['episode_index'].to_numpy())
        by_video = {}
        for i, (e, raw) in enumerate(zip(es, idx['episodes'])):
            start, end = e['dataset_from_index'], e['dataset_to_index']
            assert e['episode_index'] == i and end - start == raw['frames'] == e['length']
            if i: assert es[i-1]['dataset_to_index'] == start
            sl = table.slice(start, end - start)
            assert np.all(sl['episode_index'].to_numpy() == i)
            assert np.array_equal(sl['frame_index'].to_numpy(), np.arange(end - start))
            assert np.allclose(sl['timestamp'].to_numpy(), np.arange(end-start) / info['fps'], atol=1e-4)
            for key in ['observation.images.camera_top', 'observation.images.camera_left', 'observation.images.camera_right']:
                rel = info['video_path'].format(video_key=key, chunk_index=e[f'videos/{key}/chunk_index'], file_index=e[f'videos/{key}/file_index'])
                by_video.setdefault(rel, []).append((i, e[f'videos/{key}/from_timestamp'], e[f'videos/{key}/to_timestamp'], e['length']))
        for rel, windows in by_video.items():
            sha = manifest[name + '/' + rel]['sha256']
            jobs.append((str(dst / rel), windows, info['fps'], sha, str(out / 'videos' / (hashlib.sha256((name+rel).encode()).hexdigest() + '.json'))))
        report['datasets'][split] = dict(episodes=n, frames=len(table), videos=len(by_video),
                                         missing_raw_sources=sum(not Path(e['source']).exists() for e in idx['episodes']),
                                         sha256_copy_verified=True, finite_labels=True, indices_and_timestamps=True)
        print('LABELS_PASS', split, report['datasets'][split], flush=True)
        del table, fm
    assert not bases['train'].intersection(bases['val']), 'base episode leakage'
    report['base_episode_leakage'] = 0
    logs = sorted((source / 'logs').glob('v2_convert_*.log'))
    assert len(logs) == 26, len(logs)
    for f in logs:
        text = f.read_text(errors='replace')
        assert '###CONVERT###' in text and not re.search(r'dropped.*frame', text), str(f)
    report['conversion_logs'] = {'checked': len(logs), 'dropped': 0}
    historical = {}
    for fn, marker in [('v2_verify_videos.log', '###VIDEO_VERIFY###'), ('v2_gate_B.log', '###GATE###')]:
        text = (source / 'logs' / fn).read_text(errors='replace')
        records = [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]
        assert records and records[-1]['pass'] is True, fn
        historical[fn] = {'sha256': digest(source / 'logs' / fn), 'record': records[-1]}
    report['historical_not_rerun'] = historical
    write(out / 'audit_in_progress.json', report)
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(decode, jobs):
            results.append(result)
            print('VIDEO', len(results), len(jobs), result['decoded'], result['pass'], flush=True)
    report['video_files'] = len(results)
    report['decoded_frames_all_cameras'] = sum(r['decoded'] for r in results)
    report['bad_videos'] = [r for r in results if not r['pass']]
    report['pass'] = not report['bad_videos']
    report['finished'] = time.time()
    write(out / 'recovered_integrity.json', report)
    print('RECOVERED_INTEGRITY', report['pass'], flush=True)
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
