from pathlib import Path
import numpy as np
import json,csv,hashlib,collections
D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
O=Path('/home/jun/dummyx-sim/outputs/gcc_vla_round2_20261028_153421')
def load(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def difference(a,b):
 assert a.shape==b.shape,(a.shape,b.shape)
 assert np.isfinite(a).all() and np.isfinite(b).all()
 return float(np.max(np.abs(a-b))) if a.size else 0.
index=load(D/'逐回合汇总.json');ii={(r['method'],r['scene'],int(r['seed'])):r for r in index};pairs=[];fail=[];source=[]
for scene in ['main','pillar_x_plus40mm']:
 seeds=sorted(s for m,c,s in ii if m=='raw_pi0' and c==scene);assert len(seeds)==50
 for seed in seeds:
  rr=ii['raw_pi0',scene,seed];gg=ii['pace_g',scene,seed];rp=Path(rr['package_data']);gp=Path(gg['package_data']);rj=Path(rr['package_job']);gj=Path(gg['package_job']);jobr=load(rj/'job.json');jobg=load(gj/'job.json');mr=load(rp.parent.parent/'manifest.json');mg=load(gp.parent.parent/'manifest.json')
  pair=dict(scene=scene,seed=seed,raw_id=rr['assignment_id'],G_id=gg['assignment_id'],raw_gpu=jobr['gpu'],G_gpu=jobg['gpu'],raw_port=jobr['port'],G_port=jobg['port'],same_gpu=jobr['gpu']==jobg['gpu'],same_model_sha256=mr['model_sha256']==mg['model_sha256'],raw_path=str(rp),G_path=str(gp))
  assert jobr['seed']==jobg['seed']==seed
  with np.load(rp) as r,np.load(gp) as g,np.load(rp.parent/'paper_truth_ep_001.npz') as rt,np.load(gp.parent/'paper_truth_ep_001.npz') as gt,np.load(gp.parent/'preview_ep_001.npz') as pre,np.load(gp.parent/'guard_ep_001.npz') as gu:
   start=float(r['episode_start_time']);gs=float(g['episode_start_time']);assert abs(start-gs)<1e-12
   for x in [r,g]:
    assert np.all(x['valid_substeps'][:80]==50);assert abs(float(x['end_sim_time'][79])-start-4)<1e-8
   pair.update(episode_start_time_s=start,actions_compared=80,force_substeps_compared=4000,raw_duration_s=float(r['episode_duration_s']),G_duration_s=float(g['episode_duration_s']))
   for k in ['raw_action','q_pos_vla','q_pos','q_vel','q_acc','q_tau','replay_state','sim_time','end_sim_time']:
    pair[k+'_max_abs_diff']=difference(r[k][:80],g[k][:80])
   pair['executed_ctrl_4000_max_abs_diff']=difference(r['executed_ctrl'][:80].reshape(4000,8),g['executed_ctrl'][:80].reshape(4000,8))
   pair['Fsum_4000_max_abs_diff_N']=difference(rt['all_sum_force_N'][:4000],gt['all_sum_force_N'][:4000]);pair['truth_time_4000_max_abs_diff_s']=difference(rt['time_s'][:4000],gt['time_s'][:4000])
   pair['raw_J4_Ns']=float(rt['all_sum_force_N'][:4000].sum()*.001);pair['G_J4_Ns']=float(gt['all_sum_force_N'][:4000].sum()*.001);pair['J4_difference_G_minus_raw_Ns']=pair['G_J4_Ns']-pair['raw_J4_Ns']
   dd=np.max(np.abs(r['executed_ctrl'][:80].reshape(4000,8)-g['executed_ctrl'][:80].reshape(4000,8)),axis=1); di=np.flatnonzero(dd>0); first=int(di[0]) if len(di) else 4000
   nz=np.flatnonzero(rt['all_sum_force_N'][:4000]>0);last=int(nz[-1]) if len(nz) else None
   pair['first_executed_target_difference_substep']=first if first<4000 else None
   pair['first_executed_target_difference_relative_s']=first*.001 if first<4000 else None
   pair['first_executed_target_difference_absolute_s']=start+first*.001 if first<4000 else None
   pair['last_nonzero_load_relative_s']=(last+1)*.001 if last is not None else None
   pair['last_nonzero_load_absolute_s']=float(rt['time_s'][last]) if last is not None else None
   pair['raw_Fsum_peak_after_control_divergence_N']=float(np.max(rt['all_sum_force_N'][first:4000])) if first<4000 else 0.
   pair['G_Fsum_peak_after_control_divergence_N']=float(np.max(gt['all_sum_force_N'][first:4000])) if first<4000 else 0.
   pair['control_equal_through_nonzero_loads']=not np.any(dd[rt['all_sum_force_N'][:4000]>0])
   pair['initial_replay_state_max_abs_diff']=difference(r['replay_state'][0],g['replay_state'][0])
   pair['raw_force_samples_in_first4']=int(np.count_nonzero(rt['time_s']<=start+4+1e-8));pair['G_force_samples_in_first4']=int(np.count_nonzero(gt['time_s']<=start+4+1e-8));assert pair['raw_force_samples_in_first4']==pair['G_force_samples_in_first4']==4000
   active=pre['geometry_active'].astype(bool);changed=np.linalg.norm(pre['joint_correction_rad'],axis=1)>1e-12;ga=gu['reference_guard_active'].astype(bool);gc=np.linalg.norm(gu['reference_guard_joint_delta_rad'],axis=1)>1e-12;mask=gu['time_s']<start+4-1e-8
   assert mask.sum()==4000
   pair.update(G_geometry_active_first4=int(active[:80].sum()),G_geometry_changed_first4=int(changed[:80].sum()),G_guard_active_first4=int(ga[mask].sum()),G_guard_changed_first4=int(gc[mask].sum()),G_geometry_active_full=int(active.sum()),G_guard_active_full=int(ga.sum()))
   for name,flag,times in [('geometry_active',active,g['sim_time']),('geometry_changed',changed,g['sim_time']),('guard_active',ga,gu['time_s']),('guard_changed',gc,gu['time_s'])]:
    where=np.flatnonzero(flag);first=float(times[where[0]]) if len(where) else None;pair['first_'+name+'_absolute_s']=first;pair['first_'+name+'_relative_s']=first-start if first is not None else None
   for ep,root,label in [(r,rj,'raw'),(g,gj,'G')]:
    chunks=[]
    for k in [0,1]:
     pp=root/f'policy_chunk_{k:03d}.npz'
     with np.load(pp) as z:chunks.append(z['actions'])
    acts=np.concatenate(chunks,axis=0)[:80];pair[label+'_saved_policy_vs_rawaction_max_abs_diff']=difference(acts,ep['raw_action'][:80])
   with np.load(rj/'policy_chunk_000.npz') as pr,np.load(gj/'policy_chunk_000.npz') as pg:pair['policy_chunk0_max_abs_diff']=difference(pr['actions'],pg['actions'])
   with np.load(rj/'policy_chunk_001.npz') as pr,np.load(gj/'policy_chunk_001.npz') as pg:pair['policy_chunk1_consumed30_max_abs_diff']=difference(pr['actions'][:30],pg['actions'][:30])
  lr=[json.loads(s) for s in (rj/'policy_lineage.jsonl').read_text().splitlines()];lg=[json.loads(s) for s in (gj/'policy_lineage.jsonl').read_text().splitlines()]
  assert len(lr)>=2 and len(lg)>=2
  pair['lineage_first2_observations_equal']=all(lr[k]['observation_hash']==lg[k]['observation_hash'] for k in [0,1]);pair['lineage_first2_rng_equal']=all(lr[k]['eval_rng']==lg[k]['eval_rng']==dict(episode_seed=seed,request_index=k) for k in [0,1]);pair['lineage_gpu_consistent']=all(lr[k]['gpu']==jobr['gpu'] and lg[k]['gpu']==jobg['gpu'] for k in [0,1])
  for root in [rj,gj]:
   for path in [root/'job.json',root/'policy_lineage.jsonl']:source.append(dict(path=str(path),sha256=digest(path)))
  for path in [rp,gp,rp.parent/'paper_truth_ep_001.npz',gp.parent/'paper_truth_ep_001.npz',gp.parent/'preview_ep_001.npz',gp.parent/'guard_ep_001.npz']:source.append(dict(path=str(path),sha256=digest(path)))
  # The gate tests source validity and an explained identical Fsum integral, not an invented expectation of identical free-space trajectories.
  numeric=['raw_action_max_abs_diff','q_pos_vla_max_abs_diff','sim_time_max_abs_diff','end_sim_time_max_abs_diff','Fsum_4000_max_abs_diff_N','truth_time_4000_max_abs_diff_s','J4_difference_G_minus_raw_Ns','raw_saved_policy_vs_rawaction_max_abs_diff','G_saved_policy_vs_rawaction_max_abs_diff','policy_chunk0_max_abs_diff','policy_chunk1_consumed30_max_abs_diff','raw_Fsum_peak_after_control_divergence_N','G_Fsum_peak_after_control_divergence_N','initial_replay_state_max_abs_diff']
  booleans=['same_gpu','same_model_sha256','lineage_first2_observations_equal','lineage_first2_rng_equal','lineage_gpu_consistent','control_equal_through_nonzero_loads']
  pair['pass']=all(pair[k]==0 for k in numeric) and all(pair[k] for k in booleans)
  if not pair['pass']:fail.append(pair['raw_id']);print('UNEXPLAINED',pair['raw_id'],flush=True)
  pairs.append(pair)
rows=[]
for scene in ['main','pillar_x_plus40mm']:
 p=[x for x in pairs if x['scene']==scene];z=dict(scene=scene,n=len(p),pairs_passed=sum(x['pass'] for x in p),same_gpu_pairs=sum(x['same_gpu'] for x in p),gpu_pair_counts=dict(collections.Counter(str(x['raw_gpu']) for x in p)))
 for field in ['raw_J4_Ns','G_J4_Ns']:
  a=np.array([x[field] for x in p]);z[field]=dict(mean=float(a.mean()),sd=float(a.std(ddof=1)))
 for field in ['first_geometry_active_relative_s','first_guard_active_relative_s','first_executed_target_difference_relative_s','last_nonzero_load_relative_s']:
  a=[x[field] for x in p if x[field] is not None];z[field]=dict(n=len(a),min=min(a) if a else None,max=max(a) if a else None)
 rows.append(z)
result=dict(gate='PASS' if not fail and len(pairs)==100 else 'UNEXPLAINED',pairs=100,failed_pair_ids=fail,scope='existing raw_pi0 and pace_g, matched scene+seed; first80 actions and first4000 substeps only; no physics/inference',summary=rows,per_pair=pairs,coverage=dict(nominal_actions='80 x 7 dimensions per episode, two saved policy requests; 50+30 consumed actions',actual_joint_targets='4000 x 8 actuator controls at1ms; 6 arm and 2 fingers',joint_trajectory='80 action-end q_pos/q_vel/q_acc/q_tau plus80 saved action-start integration states at50ms',force='all_sum_force_N and timestamps4000 per episode, dt1ms',missing='No directly saved 1ms qpos/qvel trajectory; not reconstructed or inferred as observed. This limitation does not prevent direct comparison of actual1ms controls and measured archived Fsum.'),sources=source,script_sha256=digest(__file__),gate_interpretation='PASS verifies 100pair force-history identity is fully explained: same model/state/GPU/nominal actions; all nonzero counted external loads occur while actual controls equal, and after actual control divergence both histories remain exactly zero through4s. PASS does not mean all first4s actual controls or trajectories equal.')
(O/'audit/J4逐回合核查.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with (O/'audit/J4逐回合核查.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
print(json.dumps(dict(gate=result['gate'],summary=rows),ensure_ascii=False,indent=2))
