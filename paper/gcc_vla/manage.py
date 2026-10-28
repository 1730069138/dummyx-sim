#!/usr/bin/env python3
"""Paper code setup and explicit service commands. Never starts experiments on import."""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BATCH = {
    'mechanism72': 'gcc_vla_revision_20261028_132637',
    'round2': 'gcc_vla_experiments_round2_20261028_190400',
    'round3': 'gcc_vla_round3_20261028_214716',
}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    rows = json.loads((HERE / 'provenance/frozen_sources.json').read_text())
    for r in rows:
        assert sha(ROOT / r['path']) == r['sha256'], ('source changed', r['path'])
    return len(rows)

def prepare(target):
    """Install exact archived scripts into historical paths; never overwrite different files."""
    n = verify()
    target = target.resolve()
    mapping = json.loads((HERE / 'config/install_map.json').read_text())
    planned = []
    # Preflight everything before any copy; an existing result package is not a fresh run.
    for name in BATCH.values():
        p = target / 'outputs' / name / 'experiments'
        if p.exists() and any(p.rglob('result.json')):
            raise ValueError('Existing experimental results: do not install over them: ' + str(p))
    for r in mapping:
        src, dst = ROOT / r['archive'], target / r['destination']
        if dst.exists() and sha(src) != sha(dst):
            raise ValueError('Different installed code; use a clean checkout: ' + str(dst))
        planned.append((src, dst))
    for src, dst in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
    for name in BATCH.values():
        b = target / 'outputs' / name
        for d in ['audit', 'experiments', 'services', 'scripts']:
            (b / d).mkdir(parents=True, exist_ok=True)
    # Scheduler-only migration: replace a historical data-index read by the same
    # reserved seed ranges. No task/control/statistical parameter is changed.
    p = target / 'outputs' / BATCH['round2'] / 'scripts/schedule_round2.py'
    old = "old=json.loads((R/'outputs/pace_latest_evidence_20261027_1400/逐回合汇总.json').read_text());oldseeds={int(x['seed']) for x in old}"
    new = "oldseeds=set(range(610000,610050))|set(range(620000,620050))  # reserved original600 protocol seeds; no raw-data dependency"
    text = p.read_text()
    assert text.count(old) == 1
    original_sha = sha(p)
    p.write_text(text.replace(old, new))
    changes = [dict(path=str(p), archived_sha256=original_sha, installed_sha256=sha(p),
                    change='Only seed-disjointness configuration read; 600 and 72 reserved seed ranges unchanged')]
    b = target / 'outputs' / BATCH['round2']
    plan = b / 'statistics_plan.json'
    (b / 'statistics_plan.sha256.json').write_text(json.dumps(dict(path=str(plan), sha256=sha(plan)), indent=2))
    b = target / 'outputs' / BATCH['round3']
    (b / 'statistics_plan.sha256').write_text(sha(b / 'statistics_plan.json'))
    receipt = target / 'outputs' / ('gcc_vla_setup_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    receipt.mkdir()
    (receipt / 'installed_code.json').write_text(json.dumps(dict(verified_frozen_files=n,
        copied=[dict(source=str(a),destination=str(b),sha256=sha(b)) for a,b in planned],
        orchestration_changes=changes, no_results_created=True), ensure_ascii=False, indent=2))
    if target == Path('/home/jun/dummyx-sim'):
        pointer = Path('/tmp/gcc_revision_path')
        value = str(target / 'outputs' / BATCH['mechanism72'])
        if pointer.exists() and pointer.read_text().strip() != value:
            raise ValueError('Existing /tmp/gcc_revision_path belongs to another run; inspect before replacing')
        pointer.write_text(value)
    print(json.dumps(dict(verified=n, installed=len(planned), receipt=str(receipt)), ensure_ascii=False))

def entry_module():
    p = ROOT / 'scripts/deploy/gcc_vla_paper/run_experiment.py'
    spec = importlib.util.spec_from_file_location('paper_main_entry', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only authentication file location, as confirmed by the author; no key is bundled.
    source = inspect.getsource(module.check_services).replace('/home/jun/1', '/home/jun/ssh-keys/1')
    exec(source, module.__dict__)
    return module

def service_check(batch):
    verify()
    module = entry_module()
    remote, services = module.check_services(list(range(6)))
    d = ROOT / 'outputs' / BATCH[batch] / 'services/identity'
    d.mkdir(parents=True, exist_ok=False)
    (d / 'remote.json').write_text(json.dumps(remote, indent=2))
    (d / 'metadata.json').write_text(json.dumps(services, indent=2))
    print('Verified model/source/seed protocol; no inference; identity:', d)

def admit72():
    import numpy as np
    verify()
    b = ROOT / 'outputs' / BATCH['mechanism72']
    run = b / 'experiments/pilot'
    rows = []
    for p in sorted(run.glob('worker_gpu*/*/result.json')):
        r = json.loads(p.read_text())
        assert r['error'] is None and not r['metrics'].get('metric_error')
        assert r['result'] is not None
        reason = r['result']['reason']
        assert reason in ['success','step_limit','tool_floor_contact','reference_infeasible'] or reason.startswith('safety_abort:')
        f = json.loads((p.parent / 'factory_receipt.json').read_text())
        assert f and all(x['current_guard'] == (r['arm'] != 'B') for x in f)
        data = Path(r['metrics']['data_path'])
        with np.load(data, allow_pickle=False) as z:
            assert np.isfinite(z['episode_peak_joint_torque']).all()
        with np.load(data.with_name('paper_truth_ep_001.npz'), allow_pickle=False) as z:
            assert len(z['time_s']) and np.isfinite(z['all_sum_force_N']).all()
        if r['arm'] == 'A':
            with np.load(data.with_name('contact_ep_001.npz'), allow_pickle=False) as z:
                assert np.all(z['force_delta_m_s'] == 0)
        rows.append(dict(path=str(p),arm=r['arm'],scene=r['scene_id'],passed=True))
    assert len(rows) == 6 and len({(r['arm'],r['scene']) for r in rows}) == 6
    out = run / 'regression_receipt.json'
    with out.open('x') as f:
        json.dump(dict(passed=True, checks=rows, admission='New-machine smoke schema/switch/source admission; task success is not a gate; not the historical saved-state regression'), f, indent=2)
    print('Six pilot records admitted:', out)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('verify')
    q = sub.add_parser('prepare'); q.add_argument('--target', type=Path, default=ROOT)
    q = sub.add_parser('check-services'); q.add_argument('--batch', choices=list(BATCH), required=True)
    q = sub.add_parser('service'); q.add_argument('--gpu', type=int, choices=range(6), required=True); q.add_argument('--execute', action='store_true')
    sub.add_parser('admit72')
    a = p.parse_args()
    if a.action == 'verify': print('Frozen source SHA256 checks passed:', verify())
    elif a.action == 'prepare': prepare(a.target)
    elif a.action == 'check-services': service_check(a.batch)
    elif a.action == 'admit72': admit72()
    elif a.action == 'service':
        config = json.loads((HERE / 'config/service_commands.json').read_text())
        item = next(x for x in config['commands'] if x['gpu'] == a.gpu)
        cmd = ['ssh','-tt','-i','/home/jun/ssh-keys/1','jun@100.64.142.55','bash -lc ' + shlex.quote(item['remote_command'])]
        print(shlex.join(cmd), flush=True)
        if a.execute: subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
