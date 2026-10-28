#!/usr/bin/env python3
"""Read-only summary of newly generated main/72 runs, from original NPZ endpoints.

No controller, simulator, policy, paper values, or historical batch is imported.
Missing unified contact arrays are missing, never replaced by legacy metrics.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import numpy as np

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def number(z, key):
    return z[key].item()

def extract(data, ident):
    row = dict(ident)
    with np.load(data, allow_pickle=False) as z:
        aligned = 'task_success' in z.files and 'clean_success' in z.files
        if aligned:
            sr = bool(number(z, 'task_success'))
            pcr = bool(number(z, 'pillar_contact'))
            floor = bool(number(z, 'tool_floor_contact'))
            tilt = bool(number(z, 'pillar_tilt'))
        else:
            sr = bool(number(z, 'episode_success'))
            pcr = bool(number(z, 'episode_obstacle_contact'))
            floor = bool(number(z, 'episode_tool_floor_contact'))
            tilt = bool(number(z, 'episode_knockdown'))
        stsr = sr and not (pcr or floor or tilt)
        if aligned:
            assert stsr == bool(number(z, 'clean_success'))
        row.update(SR=sr, STSR=stsr, PCR=pcr, tool_floor=floor, pillar_tilt=tilt,
                   termination_reason=str(number(z, 'termination_reason')) if 'termination_reason' in z.files else None)
    original = data.with_name('data_ep_001.npz')
    if not original.exists():
        original = data
    with np.load(original, allow_pickle=False) as z:
        row['T_s'] = float(number(z, 'episode_duration_s')) if 'episode_duration_s' in z.files else None
        row['tau_peak_Nm'] = float(number(z, 'episode_peak_joint_torque')) if 'episode_peak_joint_torque' in z.files else None
        row['touch_8Nm'] = row['tau_peak_Nm'] >= 8 - 1e-10 if row['tau_peak_Nm'] is not None else None
        if row['termination_reason'] is None and 'termination_reason' in z.files:
            row['termination_reason'] = str(number(z, 'termination_reason'))
    truth = data.with_name('paper_truth_ep_001.npz')
    row.update(J_Ns=None, J4_Ns=None, Fpeak_N=None, unified_load_source='missing')
    if truth.exists():
        with np.load(truth, allow_pickle=False) as z:
            f, t = np.asarray(z['all_sum_force_N']), np.asarray(z['time_s'])
            assert f.ndim == t.ndim == 1 and len(f) == len(t) and len(f) > 0
            assert np.isfinite(f).all() and np.isfinite(t).all()
            assert len(t) < 2 or np.allclose(np.diff(t), .001, atol=1e-10, rtol=0)
            row.update(J_Ns=float(f.sum()*.001),
                       J4_Ns=float(f[:4000].sum()*.001) if len(f) >= 4000 else None,
                       Fpeak_N=float(f.max()), unified_load_source=str(truth),
                       T_s=len(f)*.001)
    row.update(data_path=str(data), data_sha256=sha(data))
    return row

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    assert not a.output.exists(), 'Use a new output directory'
    rows, technical, identities = [], [], set()
    for run in a.run:
        assert run.is_dir(), run
        for rp in sorted(run.glob('worker_gpu*/*/result.json')):
            r = json.loads(rp.read_text())
            ident = dict(method=r.get('method_id'), config=r.get('arm',r.get('method_id')),
                         scene=r['scene_id'], seed=r['seed'], gpu=r['gpu'])
            reason = str((r.get('result') or {}).get('reason', ''))
            if r.get('error') is not None or r.get('metrics',{}).get('metric_error') or reason.startswith(('policy_error:','simulation_error:','perception_failed:')):
                technical.append(dict(path=str(rp),reason=reason,error=r.get('error'))); continue
            data = list(rp.parent.rglob('data_ep_001.npz'))
            assert len(data) == 1, ('Missing/ambiguous original NPZ; do not infer endpoints', rp)
            key = tuple(ident[k] for k in ['method','config','scene','seed'])
            assert key not in identities, ('Duplicate allocation; do not select by outcome',key)
            identities.add(key); rows.append(extract(data[0],ident))
        # The native aligned VLSA uses records and statuses rather than worker result.json.
        if (run/'plan.json').exists():
            plan = json.loads((run/'plan.json').read_text())
            if plan.get('method_id') == 'vlsa_native':
                for job in plan['jobs']:
                    paths = list((run/'records'/job['id']).glob('*/aligned_ep_001.npz'))
                    status = run/'logs'/(job['id']+'.status.json')
                    if len(paths)!=1 or not status.exists() or not json.loads(status.read_text())['completed_episode']:
                        technical.append(dict(path=str(status),reason='missing or incomplete aligned VLSA record')); continue
                    ident = dict(method='vlsa_native',config='vlsa_native',scene=job['scene_id'],seed=job['seed'],gpu=job['gpu'])
                    key = tuple(ident[k] for k in ['method','config','scene','seed'])
                    assert key not in identities, ('Duplicate allocation',key)
                    identities.add(key); rows.append(extract(paths[0],ident))
    assert rows, 'No valid original episodes found'
    summary=[]
    for key in sorted({(r['method'],r['config'],r['scene']) for r in rows}):
        rr=[r for r in rows if (r['method'],r['config'],r['scene'])==key]
        s=dict(method=key[0],config=key[1],scene=key[2],n=len(rr))
        for k in ['SR','STSR','PCR','tool_floor','pillar_tilt','touch_8Nm']:
            valid=[r[k] for r in rr if r[k] is not None]
            s[k+'_count']=sum(valid);s[k+'_n']=len(valid)
        for k in ['J_Ns','J4_Ns','Fpeak_N','T_s','tau_peak_Nm']:
            values=np.array([r[k] for r in rr if r[k] is not None],float)
            s[k+'_n']=len(values);s[k+'_mean']=float(values.mean()) if len(values) else None
            s[k+'_sampleSD']=float(values.std(ddof=1)) if len(values)>1 else None
        summary.append(s)
    a.output.mkdir(parents=True)
    write_csv(a.output/'per_episode.csv', rows);write_csv(a.output/'summary.csv',summary)
    (a.output/'technical_or_missing.json').write_text(json.dumps(technical,ensure_ascii=False,indent=2))
    print(json.dumps(dict(episodes=len(rows),technical_or_missing=len(technical),groups=len(summary),output=str(a.output)),ensure_ascii=False))

if __name__ == '__main__':
    main()
