#!/usr/bin/env python3
"""Isolated six-worker dispatcher. plan is read-only; init and run require explicit action.

Every original scene/seed assignment is run once. Interrupted work is a dispatch
state, not an experiment outcome; no automatic retry or replacement is made.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import sqlite3
import subprocess
import sys
import threading
from unittest.mock import patch

ROOT=Path('/home/jun/dummyx-sim')
PARENT=ROOT/'outputs/pace_native_aligned_20261027/align_20261027_1125'
OUT=PARENT/'scheduler_v3'
OLD=PARENT/'scheduler_v1'
PREVIOUS=PARENT/'scheduler_v2'
HERE=Path(__file__).resolve()
AUDITOR=HERE.parent/'analysis_v3.py'
ENTRY=HERE.parent/'deploy_vlsa_aligned_v6.py'
OLD_ENTRY=HERE.parent/'deploy_vlsa_aligned_v4.py'
PYTHON='/home/jun/anaconda3/envs/vlsa_env/bin/python'
BENCH=ROOT/'outputs/pace_external50/benchmark_20261027_062049/experiments'
PROTOCOLS={'main':BENCH/'formal_main_n50/protocol.json','pillar_x_plus40mm':BENCH/'formal_pillar_x_plus40mm_n50/protocol.json'}
CACHE=BENCH/'model_cache'
MAIN=CACHE/'a99947328ad49d5af48b97f19dc81ffe096ef3b0ec06754a42f838fc2a7ef1a3.mjb'
PRESSURE=ROOT/'outputs/pace_gc_directional/research_20261026_214604/analysis/xplus40_sharedfilter_preflight_v1/pillar_x_plus40mm'
PORTS={0:8001,1:8002,2:8000,3:8003,4:8004,5:8005}
PROMPT='Pick up the screwdriver and place it into the box.'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_new(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def jobs():
    result=[]
    for scene,path in PROTOCOLS.items():
        protocol=json.loads(path.read_text())
        expected=list(range(610000 if scene=='main' else 620000,610050 if scene=='main' else 620050))
        assert protocol['seeds']==expected and protocol['n_per_method_scene']==50
        assignments={int(j['seed']):j for w in protocol['workers'].values() for j in w if j['method_id']=='pace_gc'}
        assert set(assignments)==set(expected)
        for seed in expected:
            a=assignments[seed]; gpu=int(a['gpu']); port=int(a['port']); assert PORTS[gpu]==port
            jobid=f'{scene}_{seed}'; record=OUT/'records'/jobid
            cmd=[PYTHON,'-B','-u',str(HERE),'exec-native','--entry',str(ENTRY),'--',
                '--host','100.64.142.55','--port',str(port),'--num_episodes','1','--max_steps','300',
                '--mode','vlsa','--fixed_eval','--initial_state_protocol','full_v1','--settle_before_start',
                '--vlsa_action_space','cartesian','--task_prompt',PROMPT,'--eef_ellipsoid_scale','0.8',
                '--max_linear_correction','0.08','--max_angular_correction','0.08','--servo_response','0.4',
                '--obstacle_source','perception','--obstacle_text','red pillar.','--device','cpu',
                '--groundingdino_config','/home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                '--groundingdino_checkpoint','/home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth',
                '--record-root',str(record),'--episode-seed',str(seed),'--aligned-model-mjb',str(MAIN if scene=='main' else PRESSURE/'model.mjb')]
            if scene!='main':cmd += ['--aligned-scene-folder',str(PRESSURE),'--aligned-trim-z']
            result.append(dict(id=jobid,scene=scene,seed=seed,gpu=gpu,port=port,record_root=str(record),
                               benchmark_protocol=str(path),benchmark_assignment=a['id'],command=cmd))
    assert len(result)==100 and len({j['id'] for j in result})==100
    # Interleave the fixed scene pairs in advance; outcomes never alter the queue.
    result.sort(key=lambda j: (j['seed'] % 10000, j['scene'] != 'main'))
    return result

def audit_previous():
    """Require evidence every prior dispatched attempt stopped before inference."""
    prev=json.loads((OLD/'plan.json').read_text())
    old_jobs={j['id']:j for j in prev['jobs']}
    new_jobs={j['id']:j for j in jobs()}
    assert set(old_jobs)==set(new_jobs)==set(j['id'] for j in jobs()) and len(old_jobs)==100
    for key in old_jobs:
        a,b=old_jobs[key],new_jobs[key]
        assert all(a[k]==b[k] for k in ('scene','seed','gpu','port')), key
    text=OLD_ENTRY.read_text(); assert sha(OLD_ENTRY)==prev['source_sha256'][str(OLD_ENTRY)]
    assert text.index("json.dump(dict(model_path=") < text.index('result = policy.infer(request)')
    assert not any(x.strip() in ('import json','from json import loads') for x in text.splitlines())
    con=sqlite3.connect('file:'+str(OLD/'dispatch.sqlite3')+'?mode=ro',uri=True)
    rows=con.execute('SELECT id,status,pid,exit_code FROM dispatch').fetchall();con.close()
    assert len(rows)==100 and all(x[1] in ('pending','running','technical_error') for x in rows)
    evidence=[]
    for jobid,status,pid,rc in rows:
        log=OLD/'logs'/f'{jobid}.log';record=OLD/'records'/jobid
        if status=='pending':
            assert not log.exists() and not record.exists(), jobid
            continue
        assert log.is_file(),jobid
        if status=='running' and pid is not None:
            assert not Path(f'/proc/{pid}').exists(),f'{jobid}: previous process may still be alive'
        raw=log.read_text(errors='replace')
        chunks=list(record.rglob('policy_chunk_*.npz')) if record.exists() else []
        lineage=list(record.rglob('policy_lineage.jsonl')) if record.exists() else []
        sidecars=list(record.rglob('aligned_ep_*.npz')) if record.exists() else []
        assert not chunks and not lineage and not sidecars, f'{jobid}: inference evidence exists; no retry'
        assert not any(mark in raw for mark in ('eval_rng','policy_chunk_','request_index','Policy inference','policy.infer(')),f'{jobid}: possible inference in log'
        if status=='technical_error':
            assert rc==1 and "NameError: name 'json' is not defined" in raw, f'{jobid}: unexpected prior error'
        else:
            assert not record.exists() and 'Loading GroundingDINO checkpoint:' in raw, f'{jobid}: incomplete attempt not proved pre-init'
        evidence.append(dict(id=jobid,status=status,log=str(log),log_sha256=sha(log),record_exists=record.exists(),no_policy_chunks=True,no_policy_lineage=True))
    return dict(old_plan=str(OLD/'plan.json'),old_plan_sha256=sha(OLD/'plan.json'),old_ledger=str(OLD/'dispatch.sqlite3'),old_ledger_sha256=sha(OLD/'dispatch.sqlite3'),old_source=str(OLD_ENTRY),old_source_sha256=sha(OLD_ENTRY),prior_dispatched=len(evidence),prior_attempts=evidence,policy_requests_proved_absent_by='v4 raises missing json before first policy.infer, or was stopped while DINO was loading before record initialization; no saved chunks or inference log markers',retries_are_initialized_attempts_only=True,completed_episodes=0)

def classify_previous():
    """Keep all finished EGL outcomes, including reaped-by-parent-pending zombies."""
    import importlib.util
    import numpy as np
    spec=importlib.util.spec_from_file_location('aligned_saved_result_audit',AUDITOR)
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    plan=json.loads((PREVIOUS/'plan.json').read_text())
    previous={j['id']:j for j in plan['jobs']};fresh={j['id']:j for j in jobs()}
    assert len(previous)==len(fresh)==100 and previous.keys()==fresh.keys()
    for k in fresh:
        assert all(previous[k][f]==fresh[k][f] for f in ('scene','seed','gpu','port','benchmark_assignment'))
    con=sqlite3.connect('file:'+str(PREVIOUS/'dispatch.sqlite3')+'?mode=ro',uri=True)
    rows=con.execute('SELECT id,status,pid,exit_code FROM dispatch').fetchall();con.close()
    assert len(rows)==100
    receipt_path=PREVIOUS/'pause_completed_workers.json'
    receipt=json.loads(receipt_path.read_text())
    prior_workers={x['id']:x for x in receipt['workers']}
    retained=[];pending=[]
    for jobid,status,pid,exitcode in rows:
        j=previous[jobid]; record=Path(j['record_root']);log=PREVIOUS/'logs'/f'{jobid}.log'
        if status=='pending':
            assert not log.exists() and not record.exists(), jobid+': pending with artifacts is ambiguous'
            pending.append(jobid);continue
        assert status in ('completed','running'), jobid+': unresolved prior technical error or unexpected status'
        if status=='running':
            item=prior_workers.get(jobid)
            assert item and item['pid']==pid and item['proc_state']=='Z' and item['exit_code_raw']==0, jobid+': missing zero-exit pause receipt'
            assert sha(item['sidecar'])==item['sidecar_sha256'] and sha(item['summary'])==item['summary_sha256'], jobid+': pause receipt artifact changed'
        else: assert exitcode==0, jobid+': completed without zero exit'
        side=list(record.glob('*/aligned_ep_001.npz')); data=list(record.glob('*/data_ep_001.npz'))
        assert len(side)==len(data)==1 and side[0].parent==data[0].parent, jobid+': missing outcome data'
        summary=record/'summary.txt';lineage=record/'policy_lineage.jsonl';initial=record/'initial_pre_action_state.npz'
        assert all(p.is_file() for p in (summary,lineage,initial,log))
        assert 'episodes: 1' in summary.read_text(),jobid+': summary incomplete'
        checked=checker.audit(side[0])
        with np.load(data[0],allow_pickle=False) as z:
            assert bool(z['episode_success'])==checked['task_success'],jobid+': original and aligned success differ'
        calls=[json.loads(x) for x in lineage.read_text().splitlines() if x.strip()]
        assert calls and all(x['request_index']==i and x['episode_seed']==j['seed'] and x['eval_rng']==dict(episode_seed=j['seed'],request_index=i) for i,x in enumerate(calls)),jobid+': incomplete policy lineage'
        chunks=sorted(record.glob('policy_chunk_*.npz'))
        assert len(chunks)==len(calls),jobid+': incomplete policy chunks'
        artifacts=[side[0],data[0],summary,lineage,initial,log,*chunks]
        retained.append(dict(**j,backend='egl',prior_dispatch_status=status,verified_exit_code=0,
            sidecar_path=str(side[0]),sidecar_sha256=sha(side[0]),data_path=str(data[0]),data_sha256=sha(data[0]),
            artifact_sha256={str(x):sha(x) for x in artifacts},audit=checked))
    assert len(retained)+len(pending)==100 and len({x['id'] for x in retained})==len(retained)
    assert {x['id'] for x in retained}.isdisjoint(pending)
    return dict(retained_completed=retained,pending_ids=pending,prior_ledger=str(PREVIOUS/'dispatch.sqlite3'),
                prior_ledger_sha256=sha(PREVIOUS/'dispatch.sqlite3'),prior_plan=str(PREVIOUS/'plan.json'),
                prior_plan_sha256=sha(PREVIOUS/'plan.json'),pause_receipt=str(receipt_path),pause_receipt_sha256=sha(receipt_path),auditor=str(AUDITOR),auditor_sha256=sha(AUDITOR),
                backend_change='EGL to OSMesa; match original benchmark camera rendering without a visible window',
                no_completed_episodes_discarded=True,no_repeat_of_finished_seeds=True)

def osmesa_selftest():
    source=PREVIOUS/'records/main_610000/initial_pre_action_state.npz'
    benchmark=BENCH/'formal_main_n50/worker_gpu0/pace_gc_main_r000/policy_lineage.jsonl'
    expected=json.loads(benchmark.read_text().splitlines()[0])['observation_hash']
    code="""import numpy as np,mujoco,hashlib,json,sys
z=np.load(sys.argv[1]);m=mujoco.MjModel.from_binary_path(sys.argv[2]);d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['state'],int(z['spec']));mujoco.mj_forward(m,d);r=mujoco.Renderer(m,height=256,width=256);o={}
for cam,key in [('rear_cam','observation/image'),('wrist_cam','observation/wrist_image')]:
 r.update_scene(d,camera=cam);o[key]=hashlib.sha256(r.render().tobytes()).hexdigest()
r.close();print(json.dumps(o))"""
    env=os.environ.copy();env.update(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.pop('DISPLAY',None)
    observed=json.loads(subprocess.check_output([PYTHON,'-B','-c',code,str(source),str(MAIN)],env=env,text=True,timeout=60))
    assert all(observed[k]==expected[k] for k in observed),'OSMesa synthetic saved-state cameras differ from benchmark'
    return dict(backend='osmesa',saved_state=str(source),saved_state_sha256=sha(source),benchmark_lineage=str(benchmark),benchmark_lineage_sha256=sha(benchmark),image_hashes=observed,match_benchmark=True,policy_calls=0,physics_steps=0)

def init():
    assert PARENT.is_dir(), 'Parent output root must exist; scheduler never creates it'
    assert not OUT.exists(), 'Never overwrite an existing dispatch directory'
    sources=[HERE,ENTRY,MAIN,PRESSURE/'model.mjb',PRESSURE/'initial_state.npz',PRESSURE/'preflight_result.json',*PROTOCOLS.values()]
    hashes={str(p):sha(p) for p in sources}
    assert hashes[str(MAIN)]==MAIN.stem
    assert hashes[str(PRESSURE/'model.mjb')]=='5e7ff3065c96b66cef8c1bce04703e9c0e36d54b09d5f5be10547348651ff17a'
    prior=audit_previous()
    resume=classify_previous()
    render=osmesa_selftest()
    alljobs=[j for j in jobs() if j['id'] in set(resume['pending_ids'])]
    for j in alljobs:j['backend']='osmesa'
    assert len(alljobs)+len(resume['retained_completed'])==100
    OUT.mkdir(); (OUT/'records').mkdir();(OUT/'logs').mkdir()
    write_new(OUT/'prior_preinference_audit.json',prior)
    write_new(OUT/'resume_evidence.json',resume)
    write_new(OUT/'osmesa_selftest.json',render)
    write_new(OUT/'plan.json',dict(created_utc=now(),jobs=alljobs,retained_completed=resume['retained_completed'],prior_ledger_sha256=resume['prior_ledger_sha256'],backend_change=resume['backend_change'],source_sha256=hashes,
        description='dispatch only; same nominal seeds do not prove paired initial observations or inference chunks',
        no_policy_calls_at_registration=True,per_group=50,real_robot=False))
    db=sqlite3.connect(OUT/'dispatch.sqlite3')
    db.execute('CREATE TABLE dispatch (id TEXT PRIMARY KEY, scene TEXT NOT NULL, seed INTEGER NOT NULL, gpu INTEGER NOT NULL, port INTEGER NOT NULL, status TEXT NOT NULL, started_utc TEXT, finished_utc TEXT, pid INTEGER, exit_code INTEGER, note TEXT)')
    db.executemany('INSERT INTO dispatch(id,scene,seed,gpu,port,status) VALUES(:id,:scene,:seed,:gpu,:port,\'pending\')',alljobs)
    db.commit();db.close()
    print(json.dumps(dict(output=str(OUT),assigned=100,retained=len(resume['retained_completed']),pending=len(alljobs),status='registered',policy_calls=0)))

def run_native(entry, args):
    # Rendering only: mimic the native passive viewer while retaining EGL cameras.
    os.environ['MUJOCO_GL']='osmesa';os.environ['PYOPENGL_PLATFORM']='osmesa';os.environ.pop('DISPLAY',None)
    import mujoco
    import mujoco.viewer
    class Viewer:
        def __init__(self,model,data,*a,**kw):self._running=True;self.user_scn=mujoco.MjvScene(model,maxgeom=32)
        def is_running(self):return self._running
        def sync(self):pass
        def close(self):self._running=False
        def __enter__(self):return self
        def __exit__(self,*a):self.close()
    path=Path(entry).resolve(strict=True)
    sys.argv=[str(path),*args];sys.path.insert(0,str(ROOT/'scripts/deploy'));sys.path.insert(0,str(path.parent))
    with patch.object(mujoco.viewer,'launch_passive',new=Viewer):runpy.run_path(str(path),run_name='__main__')

def run():
    manifest=json.loads((OUT/'plan.json').read_text())
    for path,digest in manifest['source_sha256'].items():
        if sha(path)!=digest:raise RuntimeError('Frozen source changed: '+path)
    lock=(OUT/'dispatcher.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    db=sqlite3.connect(OUT/'dispatch.sqlite3')
    active=db.execute("SELECT id,pid FROM dispatch WHERE status='running'").fetchall()
    if active:
        db.close();raise RuntimeError('Existing running assignments need explicit inspection; no silent retry: '+repr(active))
    errors=db.execute("SELECT id,note FROM dispatch WHERE status='technical_error'").fetchall()
    if errors:
        db.close();raise RuntimeError('Technical failure remains unresolved; dispatcher paused: '+repr(errors))
    db.close()
    env=os.environ.copy();env.update(MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NO_PROXY='100.64.142.55,localhost,127.0.0.1',no_proxy='100.64.142.55,localhost,127.0.0.1');env.pop('DISPLAY',None)
    halted=threading.Event()
    dispatch_lock=threading.Lock()
    def worker(gpu):
        conn=sqlite3.connect(OUT/'dispatch.sqlite3',timeout=60)
        for j in (x for x in manifest['jobs'] if x['gpu']==gpu):
            if halted.is_set(): break
            row=conn.execute('SELECT status FROM dispatch WHERE id=?',(j['id'],)).fetchone()
            if row[0]!='pending':continue
            log=OUT/'logs'/f"{j['id']}.log"
            if log.exists() or Path(j['record_root']).exists():
                conn.execute("UPDATE dispatch SET status='technical_error',note=? WHERE id=?",('Output exists before execution; no retry',j['id']));conn.commit();halted.set();break
            status='technical_error'
            try:
                with dispatch_lock:
                    if halted.is_set():break
                    conn.execute("UPDATE dispatch SET status='running',started_utc=? WHERE id=?",(now(),j['id']));conn.commit()
                    with log.open('xb',buffering=0) as f:
                        process=subprocess.Popen(j['command'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                        conn.execute('UPDATE dispatch SET pid=? WHERE id=?',(process.pid,j['id']));conn.commit()
                rc=process.wait()
                sidecars=list(Path(j['record_root']).glob('*/aligned_ep_001.npz'))
                status='completed' if rc==0 and len(sidecars)==1 else 'technical_error'
                note=None if status=='completed' else f'exit={rc}; sidecars={len(sidecars)}; inspect original log and record directory'
                with dispatch_lock:
                    if status=='technical_error':halted.set()
                    conn.execute('UPDATE dispatch SET status=?,finished_utc=?,exit_code=?,note=? WHERE id=?',(status,now(),rc,note,j['id']));conn.commit()
            except Exception as e:
                with dispatch_lock:
                    halted.set()
                    conn.execute("UPDATE dispatch SET status='technical_error',finished_utc=?,note=? WHERE id=?",(now(),repr(e),j['id']));conn.commit()
            print(json.dumps(dict(id=j['id'],status=status),ensure_ascii=False),flush=True)
            if halted.is_set():break
        conn.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(worker,gpu) for gpu in PORTS]
        for f in futures:f.result()
    conn=sqlite3.connect(OUT/'dispatch.sqlite3')
    print(json.dumps(dict(counts=conn.execute('SELECT status,COUNT(*) FROM dispatch GROUP BY status').fetchall(), technical_failure_paused=halted.is_set(), note='Previously launched tasks allowed to finish; pending tasks remain unstarted' if halted.is_set() else None),ensure_ascii=False),flush=True);conn.close()

def main():
    if len(sys.argv)>1 and sys.argv[1]=='exec-native':
        q=argparse.ArgumentParser();q.add_argument('--entry',required=True);q.add_argument('args',nargs=argparse.REMAINDER);a=q.parse_args(sys.argv[2:]);run_native(a.entry,a.args[1:] if a.args and a.args[0]=='--' else a.args);return
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','audit-previous','assess-previous','osmesa-selftest','init','run']);a=p.parse_args()
    if a.action=='plan':print(json.dumps(dict(output=str(OUT),jobs=jobs(),policy_calls=0),ensure_ascii=False,indent=2))
    elif a.action=='audit-previous':print(json.dumps(audit_previous(),ensure_ascii=False,indent=2))
    elif a.action=='assess-previous':print(json.dumps(classify_previous(),ensure_ascii=False,indent=2))
    elif a.action=='osmesa-selftest':print(json.dumps(osmesa_selftest(),ensure_ascii=False,indent=2))
    elif a.action=='init':init()
    else:run()
if __name__=='__main__':main()
