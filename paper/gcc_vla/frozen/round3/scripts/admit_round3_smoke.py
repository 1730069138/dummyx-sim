"""Read-only admission of the two excluded smoke allocations."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
O=Path(__file__).resolve().parents[1];sys.path.insert(0,str(O/'scripts'))
from round2_trace import validate_trace_npz
from analyze_round2 import validate_trace
S=Path(sys.argv[1]) if len(sys.argv)>1 else O/'experiments/smoke';assert S.resolve().is_relative_to(O);results=[];first=[]
plan=json.loads((O/'statistics_plan.json').read_text());threshold=np.asarray(plan['jump_limiter']['threshold_rad'])
for rp in sorted(S.glob('worker_gpu*/*/result.json')):
 r=json.loads(rp.read_text());tp=Path(r['trace_path']);dp=Path(r['metrics']['data_path']);fp=dp.with_name('paper_truth_ep_001.npz');item={'id':r['id'],'seed':r['seed'],'config':r['config_name'],'gpu':r['gpu'],'path':str(rp)}
 assert r['stage']=='smoke' and r['seed']==890300 and r['gpu']==2
 assert r['error'] is None and not r['metrics'].get('recording_error',False)
 assert not any(s in r['result']['reason'] for s in ['policy_error','simulation_error','connection','timeout'])
 item['schema_validation']=validate_trace_npz(tp)
 with np.load(tp,allow_pickle=False) as z,np.load(dp,allow_pickle=False) as d,np.load(fp,allow_pickle=False) as truth:
  meta,done=validate_trace(z);n=int(done.sum());F=z['sum_external_contact_force_N'][done];ts=z['time_after_s'][done]
  assert n==len(truth['time_s'])
  assert np.allclose(F,truth['all_sum_force_N'],atol=1e-10,rtol=0)
  assert np.allclose(ts,truth['time_s'],atol=1e-12,rtol=0)
  commands=d['executed_ctrl'].reshape(-1,d['executed_ctrl'].shape[-1]);commands=commands[np.isfinite(commands[:,0])]
  assert np.allclose(commands,z['ctrl_written'][done],atol=1e-12,rtol=0)
  assert abs(float(np.max(np.abs(z['qfrc_actuator_arm'][done])))-float(d['episode_peak_joint_torque']))<1e-10
  if r['config_name']=='F_limited':
   valid=z['previous_applied_valid'].astype(bool)&done
   assert np.all(np.abs(z['delta_u_from_previous'][valid])<=threshold+1e-12)
   assert meta['target_jump_limit_rad']==plan['jump_limiter']['threshold_rad']
   item['limited_substeps']=int(np.any(np.abs(z['limiter_delta'][done])>1e-12,axis=1).sum())
  else:assert np.all(z['limiter_delta'][done]==0)
  item.update(samples=n,passed=True)
 c=np.load(rp.parent/'policy_chunk_000.npz',allow_pickle=False);lineage=json.loads((rp.parent/'policy_lineage.jsonl').read_text().splitlines()[0]);first.append((c['state'],c['actions'],lineage['observation_hash']))
 results.append(item)
assert len(results)==2
same={'state_equal':np.array_equal(first[0][0],first[1][0]),'first_action_equal':np.array_equal(first[0][1],first[1][1]),'observation_hash_equal':first[0][2]==first[1][2]};assert all(same.values())
receipt={'passed':True,'smoke_episodes':2,'formal_episodes':0,'physics_steps_added_by_this_analysis':0,'policy_calls_by_this_analysis':0,'first_input_comparison':same,'checks':results,'mandatory_exclusions':{'strict_valid':'defaultFalse is unobserved; do not use','pillar_tilt_episode':'defaultFalse is unobserved; use original NPZ outcome','tool_floor_contact':'snapshot before current update; not exact first-event timing','strict_frames':'cached action-level counter,not semantic phase truth'},'threshold_recalibrated':False}
with (O/'audit/smoke_admission.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print('SMOKE ADMITTED',same,flush=True)
