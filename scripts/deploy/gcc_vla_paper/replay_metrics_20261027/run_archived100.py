import os,sys,csv,json,subprocess,concurrent.futures
from pathlib import Path
P=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
O=Path('/home/jun/dummyx-sim/outputs/gcc_vla_load_latency_20261027/replay_exact')
S=Path(__file__).with_name('replay_vlsa_exact.py')
rows=[r for r in csv.DictReader((P/'逐回合汇总.csv').open()) if r['method']=='vlsa_native']
remaining=[]
for r in rows:
 p=O/r['scene']/r['assignment_id']/'replay_validation.json'
 if p.exists():
  assert json.loads(p.read_text())['admitted']
 else:remaining.append(r['assignment_id'])
assert len(rows)==100 and len(remaining)==94
parts=[remaining[i::4] for i in range(4)]
def run(i,ids):
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 log=O/('worker_'+str(i)+'.log')
 with log.open('x') as f:
  c=subprocess.run(['/home/jun/anaconda3/envs/vlsa_env/bin/python',str(S),'--output',str(O),'--ids',*ids],env=env,stdout=f,stderr=subprocess.STDOUT)
 return dict(worker=i,count=len(ids),exit_code=c.returncode,log=str(log))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(lambda x:run(*x),enumerate(parts)):print(json.dumps(result),flush=True)
reports=[json.loads((O/r['scene']/r['assignment_id']/'replay_validation.json').read_text()) for r in rows]
assert len(reports)==100 and all(r['admitted'] for r in reports)
receipt=dict(episodes=len(reports),groups=2,n_per_group=50,new_inference_calls=sum(r['new_inference_calls'] for r in reports),new_qp_calls=0,admitted=sum(r['admitted'] for r in reports),physical_substeps=sum(r['physical_substeps'] for r in reports),initialization_substeps=sum(r['initialization_steps'] for r in reports),max_errors={k:max(r['errors'][k] for r in reports) for k in reports[0]['errors']},event_mismatches={k:sum(r['event_mismatches'][k] for r in reports) for k in reports[0]['event_mismatches']},max_legacy_impulse_error_Ns=max(r['legacy_impulse_abs_error_Ns'] for r in reports),all_endpoints_match=all(r['endpoints_match'] for r in reports))
(O/'batch_validation.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
