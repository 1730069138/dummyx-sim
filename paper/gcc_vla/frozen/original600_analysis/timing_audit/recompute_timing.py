#!/usr/bin/env python3
"""Read frozen latest-600 logs. New outputs only; no simulation or inference."""
import csv, hashlib, json, math, os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
OUT = Path(__file__).resolve().parent
INDEX = ROOT / '逐回合汇总.csv'
FROZEN = ROOT / '脚本与身份'
SOURCE_FILES = [INDEX,
 FROZEN/'共享运行时/deploy_screwdriver_client_old.py',
 FROZEN/'共享运行时/tabletop_pace_preview.py',
 FROZEN/'共享运行时/agv_pace.py',
 FROZEN/'外部适配/oscbf_independent.py',
 FROZEN/'NEO/neo_tracking_parent.py',
 FROZEN/'NEO/neo_candidate_mesh.py',
 FROZEN/'NEO/neo_solver.py',
 FROZEN/'VLSA/deploy_vlsa_aligned_v6.py']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def identity(path):
    path=Path(path)
    return dict(absolute_path=str(path),bytes=path.stat().st_size,sha256=sha(path))

def finite(a):
    a=np.asarray(a,dtype=float).reshape(-1)
    return a[np.isfinite(a)]

def stats(a):
    a=finite(a)
    if not len(a):return dict(n=0,mean_ms=None,sample_sd_ms=None,p50_ms=None,p95_ms=None,over50_count=None)
    return dict(n=len(a),mean_ms=float(a.mean()),sample_sd_ms=float(a.std(ddof=1)) if len(a)>1 else None,
                p50_ms=float(np.percentile(a,50,method='linear')),p95_ms=float(np.percentile(a,95,method='linear')),
                over50_count=int(np.count_nonzero(a>50)))

def write_json(name,value):
    with open(OUT/name,'x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def write_csv(name,rows):
    with open(OUT/name,'x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

with INDEX.open(encoding='utf8') as f: rows=list(csv.DictReader(f))
assert len(rows)==600
assert len({(r['method'],r['scene'],r['seed']) for r in rows})==600
buckets=defaultdict(list);per=[];identities=[]
for r in rows:
    data=Path(r['package_data'])
    if r['method']=='vlsa_native':data=data.with_name('data_ep_001.npz')
    with np.load(data,allow_pickle=False) as z:
        logged=np.asarray(z['safety_layer_time_ms'],dtype=float).reshape(-1)
        t=finite(logged)
        assert len(logged)>0
        if r['method']=='raw_pi0':assert len(t)==0
        else:assert len(t)==len(logged)
        components={name:finite(z[name]) if name in z.files else np.array([]) for name in ('apf_time_ms','contact_controller_time_ms','qp_time_ms')}
        if r['method'] not in ('raw_pi0','vlsa_native'):
            assert np.allclose(logged,np.asarray(z['apf_time_ms'])+np.asarray(z['contact_controller_time_ms']),rtol=1e-10,atol=1e-10)
    s=stats(t)
    e={k:r[k] for k in ('method','scene','seed','assignment_id','task_success','clean_success','pillar_contact','reason','backend','run_id')}
    e.update(logged_action_count=len(logged),safety_applicable=r['method']!='raw_pi0',safety_finite_count=s['n'],
             safety_mean_ms=s['mean_ms'],safety_sample_sd_ms=s['sample_sd_ms'],safety_p50_ms=s['p50_ms'],safety_p95_ms=s['p95_ms'],safety_over50_count=s['over50_count'])
    for name,v in components.items():
        cs=stats(v);prefix=name.replace('_time_ms','').replace('_ms','')
        e.update({prefix+'_sample_count':cs['n'],prefix+'_mean_ms':cs['mean_ms'],prefix+'_p50_ms':cs['p50_ms'],prefix+'_p95_ms':cs['p95_ms']})
    # For raw pi0 only, this identifies the common timed substep mapping/observer
    # segment. For methods with filters that same raw field additionally includes filters.
    e['raw_common_execution_observer_mean_ms']=stats(components['contact_controller_time_ms'])['mean_ms'] if r['method']=='raw_pi0' else None
    e.update(indexed_endpoint_file=str(Path(r['package_data'])),timing_data_file=str(data),timing_data_sha256=sha(data))
    identities.append(dict(method=r['method'],scene=r['scene'],seed=r['seed'],indexed_endpoint_file=r['package_data'],**identity(data)))
    per.append(e);buckets[(r['method'],r['scene'])].append(dict(row=e,t=t,components=components))

groups=[]
for (m,s),es in sorted(buckets.items()):
    assert len(es)==50
    t=np.concatenate([e['t'] for e in es]);ps=stats(t)
    em=finite([e['row']['safety_mean_ms'] if e['row']['safety_mean_ms'] is not None else np.nan for e in es]);ems=stats(em)
    g=dict(method=m,scene=s,episode_count=50,task_success_count=sum(e['row']['task_success']=='True' for e in es),
           backend_counts=dict(Counter(e['row']['backend'] for e in es)),safety_applicable=m!='raw_pi0',
           timing_finite_episode_count=sum(len(e['t'])>0 for e in es),timing_complete_episode_count=sum(len(e['t'])==e['row']['logged_action_count'] for e in es),
           logged_action_count=sum(e['row']['logged_action_count'] for e in es),pooled_action_sample_count=ps['n'],
           pooled_action_mean_ms=ps['mean_ms'],pooled_action_sample_sd_ms=ps['sample_sd_ms'],pooled_action_p50_ms=ps['p50_ms'],pooled_action_p95_ms=ps['p95_ms'],
           pooled_action_over50_count=ps['over50_count'],pooled_action_over50_fraction=float(ps['over50_count']/ps['n']) if ps['n'] else None,
           episode_mean_finite_count=len(em),episode_mean_mean_ms=ems['mean_ms'],episode_mean_sample_sd_ms=ems['sample_sd_ms'],
           episode_mean_p50_ms=ems['p50_ms'],episode_mean_p95_ms=ems['p95_ms'])
    for name in ('apf_time_ms','contact_controller_time_ms','qp_time_ms'):
        v=np.concatenate([e['components'][name] for e in es]);cs=stats(v);prefix=name.replace('_time_ms','').replace('_ms','')
        for k,v in cs.items():g[prefix+'_pooled_'+k]=v
    groups.append(g)

notes={
 'source_cohort':'latest600 only: 6 methods x 2 fixed scenes x 50 completed episodes; all failures retained',
 'unit':'milliseconds per logged nominal 50ms action; last partially executed action remains included if logged',
 'pooled_definition':'Concatenate all finite per-action safety_layer_time_ms values within each group. mean/P50/P95 are computed on that concatenated vector, not averaged episode percentiles.',
 'episode_mean_definition':'First calculate arithmetic mean of finite action timings within each episode; group episode_mean_mean_ms and sample SD (ddof=1) describe these 50 episode means equally weighted.',
 'percentile_method':'numpy.percentile(method=linear)',
 'pure_pi0':'No external safety algorithm: safety_layer_time_ms is all NaN and exported as null/blank, not zero. contact_controller_time_ms still records common target mapping plus torque observer/housekeeping and is separately retained.',
 'shared_runtime_boundary':'For G/G+C/OSCBF/NEO: action-level geometry process plus accumulated substep C/final geometry guard/target mapping, torque observer, and some in-memory debug copies. Geometry final guard and observer execute up to 50 times per action.',
 'vlsa_boundary':'VLSA native data_ep_001.npz safety_layer_time_ms: one action-level adapt_joint_target_through_vlsa call, including kinematics/CBF construction/QP/DLS and additional CBF diagnostics; no shared 50-step torque-observer loop in this clock. Attached-geometry updating before this function is excluded.',
 'excluded':'Policy inference/network, initial perception fitting, mujoco.mj_step, rendering/video and most logging/evaluation are excluded; the clock is not total control-cycle latency.',
 'comparability':'Recorded deployment execution segments have different boundaries and update frequencies; report them as in-situ safety-execution-related compute, not a rigorously identical-boundary algorithm-only speed ranking. Current logs cannot isolate a common pure-algorithm wall clock for all methods.',
 'over50':'Observed recorded segment above 50ms; a below-50ms segment does not establish an end-to-end deadline guarantee or hard real-time control.',
 'runtime_conditions':'Wall clocks under parallel processes. OPENBLAS/OMP/MKL constrained to one thread by launch configuration, but CPU scheduling and shared load remain. All non-VLSA rows are OSMesa; VLSA main18 EGL+32 OSMesa, X40 12 EGL+38 OSMesa; backend retained per episode, no post-hoc exclusion.',
 'solver_fields':'OSCBF/NEO solver_compute_ms spans full project_current_target including geometry/Jacobians/constraints/output; VLSA qp_time_ms wraps CVXPY problem.solve, not OSQP iterations alone.',
 'no_reexecution':'Only saved arrays were read. No simulation, inference, GPU run, parameter adjustment, or source changes.'
}
write_csv('per_episode.csv',per)
flat=[]
for g in groups:
    item=dict(g);item['backend_counts']=json.dumps(item['backend_counts'],ensure_ascii=False,sort_keys=True);flat.append(item)
write_csv('groups.csv',flat)
write_json('groups.json',dict(definitions=notes,groups=groups))
write_json('sources.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),index_and_frozen_code=[identity(p) for p in SOURCE_FILES],episode_timing_data=identities,analysis_script=identity(__file__)))
with (OUT/'README.md').open('x',encoding='utf8') as f:
    f.write('# 最新600回合原位耗时离线审计\n\n')
    for k,v in notes.items():f.write('**'+k+'**: '+v+'\n\n')
    f.write('文件：per_episode.csv为600条完整回合；groups.csv与groups.json为12组。sources.json记录全部实际计时文件和冻结源码的绝对路径、字节数与SHA256。空安全耗时表示未应用外部安全控制，不以零替代。所有失败回合仍在数据中。\n')
manifest=[identity(p) for p in sorted(OUT.iterdir()) if p.is_file()]
write_json('output_sha256.json',manifest)
print(json.dumps(dict(output_directory=str(OUT),episodes=len(per),groups=len(groups),outputs=[p.name for p in sorted(OUT.iterdir())]),ensure_ascii=False))
