"""Offline admission only: original smoke arrays, no engine or policy calls."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[1];S=B/'experiments/smoke';sys.path.insert(0,str(B/'scripts'))
from round2_trace import validate_trace_npz
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
results=[]
for rp in sorted(S.glob('worker_gpu*/*/result.json')):
 r=json.loads(rp.read_text());tp=Path(r['trace_path']);dp=Path(r['metrics']['data_path']);fp=dp.with_name('paper_truth_ep_001.npz');item=dict(id=r['id'],seed=r['seed'],gpu=r['gpu'],config=r['config_name'],scene=r['scene_id'],task_success=r['result']['success'],termination_reason=r['result']['reason'],trace_path=str(tp),data_path=str(dp),paper_truth_path=str(fp))
 try:
  assert r['error'] is None and not r['metrics'].get('recording_error',False)
  assert not any(x in r['result']['reason'] for x in ['policy_error','simulation_error','connection','timeout'])
  item['schema_validation']=validate_trace_npz(tp)
  with np.load(tp,allow_pickle=False) as z,np.load(dp,allow_pickle=False) as d,np.load(fp,allow_pickle=False) as f:
   done=z['step_completed'].astype(bool);n=int(done.sum());F=z['sum_external_contact_force_N'][done];time=z['time_after_s'][done]
   assert n==len(f['time_s'])
   tferr=float(np.max(np.abs(F-f['all_sum_force_N']))) if n else 0.;terr=float(np.max(np.abs(time-f['time_s']))) if n else 0.
   assert tferr<1e-10 and terr<1e-12
   commands=d['executed_ctrl'].reshape(-1,d['executed_ctrl'].shape[-1]);commands=commands[np.isfinite(commands[:,0])]
   assert commands.shape==z['ctrl_written'][done].shape
   cerr=float(np.max(np.abs(commands-z['ctrl_written'][done]))) if n else 0.;assert cerr<1e-12
   tau=float(np.max(np.abs(z['qfrc_actuator_arm'][done]))) if n else 0.;tauerr=abs(tau-float(d['episode_peak_joint_torque']));assert tauerr<1e-10
   item.update(samples=n,max_Fsum_error_N=tferr,max_time_error_s=terr,max_applied_ctrl_error=cerr,tau_peak_scalar_error_Nm=tauerr,trace_sha256=sha(tp),paper_truth_sha256=sha(fp),passed=True)
  print(item['id'],'PASS',n,'Fsum error',tferr,flush=True)
 except Exception as exc:
  item.update(passed=False,error=type(exc).__name__+': '+str(exc));print(item['id'],'FAIL',item['error'],flush=True)
 results.append(item)
invalid_fields={'strict_valid':'hook does not supply this field; default False is UNOBSERVED, not measured false','pillar_tilt_episode':'hook does not supply this field; default False is UNOBSERVED, not measured false','tool_floor_contact':'snapshot before current-step floor predicate update; do not use for exact first floor-contact time','strict_frames':'existing counter before current action-end update; do not use as current-step strict-valid predicate'}
res=dict(passed=len(results)==6 and all(r['passed'] for r in results),status='PASS_WITH_EXPLICIT_UNUSED_FIELD_EXCLUSIONS' if len(results)==6 and all(r['passed'] for r in results) else 'FAIL',episodes=len(results),task_successes=sum(bool(r['task_success']) for r in results),task_failures=sum(not r['task_success'] for r in results),technical_errors=sum(not r['passed'] for r in results),physics_steps=0,policy_calls=0,readonly=True,mandatory_exclusions=invalid_fields,admission_condition='Analysis MUST NOT interpret default False strict_valid/pillar_tilt_episode as measurements. Official task/STSR/tilt/floor endpoints come from original result. No logger mutation or retroactive signal fabrication.',checks=results)
(B/'audit/smoke_admission.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print('FINAL',res['status'],flush=True)
