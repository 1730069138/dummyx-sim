from pathlib import Path
import json,csv,collections,hashlib
import numpy as np
D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400');OLD=Path('/home/jun/dummyx-sim/outputs/gcc_vla_revision_20261028_132637');O=Path('/home/jun/dummyx-sim/outputs/gcc_vla_round2_20261028_153421')
def read(p):return json.loads(Path(p).read_text())
def maxrun(a):
 best=cur=0
 for v in a:cur=cur+1 if v else 0;best=max(best,cur)
 return best
idx=read(D/'逐回合汇总.json');peaks=[];gcj4=[];torque=[]
for r in idx:
 p=Path(r['package_data']);dp=p.with_name('data_ep_001.npz') if r['method']=='vlsa_native' else p
 with np.load(dp) as z:
  torque.append(dict(cohort='original600',method=r['method'],scene=r['scene'],seed=r['seed'],data=str(dp),actuator_torque_history_field=[k for k in z.files if k in ['qfrc_actuator','actuator_force','actuator_torque','tau_actuator','qfrc_actuator_history']],action_end_q_tau_shape=list(z['q_tau'].shape) if 'q_tau' in z else None,ctrl_shape=list(z['executed_ctrl'].shape) if 'executed_ctrl' in z else None,tau_external_shape=list(z['tau_external'].shape) if 'tau_external' in z else None,episode_peak_joint_torque_Nm=float(z['episode_peak_joint_torque']) if 'episode_peak_joint_torque' in z else None))
 if r['method'] not in ['raw_pi0','pace_gc']:continue
 job=Path(r['package_job'])
 with np.load(dp) as z,np.load(p.with_name('paper_truth_ep_001.npz')) as t:
  start=float(z['episode_start_time']);F=t['all_sum_force_N'];pk=int(np.argmax(F));peak_abs=float(t['time_s'][pk]);ctrl=z['executed_ctrl'].reshape(-1,8);a=dict(method=r['method'],scene=r['scene'],seed=r['seed'],peak_sample=pk,peak_absolute_s=peak_abs,peak_relative_s=peak_abs-start,Fsum_peak_N=float(F[pk]),largest_individual_norm_at_Fsum_peak_N=float(t['all_peak_individual_N'][pk]),gripper_target_total_at_peak_m=float(ctrl[pk,6:].sum()),protected_external_contact_count_at_peak=int(t['all_contacts'][pk]),maximum_contribution_contact_pair='UNVERIFIED: no per-contact force-to-pair linkage saved',force_pair_attribution_available=False)
  widths=ctrl[:,6:].sum(axis=1);close=np.flatnonzero(widths<widths[0]-.001);a['first_closing_command_relative_s']=float(close[0]*.001) if len(close) else None;a['peak_before_first_closing_command']=a['first_closing_command_relative_s'] is not None and a['peak_relative_s']<a['first_closing_command_relative_s']
  first_two=None;atpeak=None
  with (job/'contact_object_samples.jsonl').open() as stream:
   for line in stream:
    item=json.loads(line)
    if item['finger_contacts']==2 and first_two is None:first_two=float(item['time_s'])
    if abs(float(item['time_s'])-peak_abs)<1e-9:atpeak=item
    if first_two is not None and atpeak is not None:break
  a['first_two_finger_contact_relative_s']=first_two-start if first_two is not None else None;a['peak_before_first_two_finger_contact']=first_two is not None and peak_abs<first_two;a['finger_contacts_at_peak']=int(atpeak['finger_contacts']) if atpeak else None;a['pillar_contact_at_peak']=bool(atpeak['pillar_contact']) if atpeak else None;a['peak_contact_log_found']=atpeak is not None;a['peak_recorded_pairs_without_force']=(atpeak or {}).get('pairs')
  if r['method']=='pace_gc':
   with np.load(p.with_name('contact_ep_001.npz')) as c,np.load(p.with_name('preview_ep_001.npz')) as pre,np.load(p.with_name('guard_ep_001.npz')) as gu:
    assert len(c['time_s'])==len(F)==len(gu['time_s']);assert abs(float(c['time_s'][pk])-peak_abs)<1e-9
    a.update(contact_gate_at_peak=bool(c['contact_gate'][pk]),height_compensation_norm_at_peak_m_s=float(np.linalg.norm(c['height_delta_m_s'][pk])),force_compensation_norm_at_peak_m_s=float(np.linalg.norm(c['force_delta_m_s'][pk])),support_reference_valid_at_peak=bool(c['support_before_valid'][pk]),feedback_force_norm_at_peak_N=float(np.linalg.norm(c['feedback_force_N'][pk])))
    geom=np.repeat(pre['geometry_active'].astype(bool),z['valid_substeps'].astype(int));geomchange=np.repeat(np.linalg.norm(pre['joint_correction_rad'],axis=1)>1e-12,z['valid_substeps'].astype(int));guard=gu['reference_guard_active'].astype(bool);guardchange=np.linalg.norm(gu['reference_guard_joint_delta_rad'],axis=1)>1e-12
    assert len(geom)==len(F)
    q=dict(scene=r['scene'],seed=r['seed'],geometry_active_first4=int(geom[:4000].sum()),geometry_changed_first4=int(geomchange[:4000].sum()),guard_active_first4=int(guard[:4000].sum()),guard_changed_first4=int(guardchange[:4000].sum()),geometry_changed_positive_load_overlap=int((geomchange[:4000]&(F[:4000]>0)).sum()),guard_changed_positive_load_overlap=int((guardchange[:4000]&(F[:4000]>0)).sum()),first_geometry_change_relative_s=float(np.flatnonzero(geomchange)[0]*.001) if geomchange.any() else None,first_guard_change_relative_s=float(np.flatnonzero(guardchange)[0]*.001) if guardchange.any() else None)
    for key,mask in [('geometry',geomchange[:4000]),('guard',guardchange[:4000])]:q[key+'_active_first4_load_integral_Ns']=float(F[:4000][mask].sum()*.001)
    gcj4.append(q)
  peaks.append(a)
print('original200 peaks and600 torque coverage done',flush=True)
new=[];technical=[];phase=[];failure=[]
for stage in ['formal','formal_resume_connection','formal_resume_policy_timeout']:
 for rp in (OLD/'experiments'/stage).glob('worker_gpu*/*/result.json'):
  r=read(rp);reason=str((r.get('result') or {}).get('reason',''))
  if r['error'] is not None or reason.startswith(('policy_error:','simulation_error:','perception_error:')):technical.append(dict(id=r['id'],path=str(rp),reason=reason,error=r['error']));continue
  p=next(rp.parent.rglob('data_ep_001.npz'))
  with np.load(p) as z:
   torque.append(dict(cohort='new72',method=r['arm'],scene=r['scene_id'],seed=r['seed'],data=str(p),actuator_torque_history_field=[k for k in z.files if k in ['qfrc_actuator','actuator_force','actuator_torque','tau_actuator','qfrc_actuator_history']],action_end_q_tau_shape=list(z['q_tau'].shape) if 'q_tau' in z else None,ctrl_shape=list(z['executed_ctrl'].shape) if 'executed_ctrl' in z else None,tau_external_shape=list(z['tau_external'].shape) if 'tau_external' in z else None,episode_peak_joint_torque_Nm=float(z['episode_peak_joint_torque']) if 'episode_peak_joint_torque' in z else None))
   if r['arm'] not in ['F','B']:continue
   with np.load(p.with_name('preview_ep_001.npz')) as pre,np.load(p.with_name('paper_quality_ep_001.npz')) as q:
    n=len(z['valid_substeps']);assert len(pre['pickup_gate'])==len(q['placement_valid'])==n;valid=z['valid_substeps'].astype(int);pickup=pre['pickup_gate'].astype(bool);place=q['placement_valid'].astype(bool);where=np.flatnonzero(place);firstplace=int(where[0]) if len(where) else None;labels=np.where(pickup,1,0);labels[np.arange(n)>(firstplace if firstplace is not None else n)]=2;ss=np.repeat(labels,valid);total=len(ss)
    if r['arm']=='F':
     with np.load(p.with_name('guard_ep_001.npz')) as gu:active=gu['reference_guard_active'].astype(bool);changed=np.linalg.norm(gu['reference_guard_joint_delta_rad'],axis=1)>1e-12;assert len(active)==total
    else:
     assert not p.with_name('guard_ep_001.npz').exists();assert read(p.with_name('metadata_ep_001.json'))['preview']['current_reference_guard'] is False;active=np.zeros(total,bool);changed=active.copy()
    for k,label in enumerate(['pre_pickup15mm_gate','post_pickup_gate_before_first_valid_placement','after_first_saved_valid_placement']):
     mask=ss==k;den=int(mask.sum());num=int((mask&active).sum());phase.append(dict(id=r['id'],arm=r['arm'],scene=r['scene_id'],seed=r['seed'],phase_proxy=label,valid_substeps=den,guard_active_substeps=num,guard_changed_substeps=int((mask&changed).sum()),guard_active_fraction=num/den if den else None,source='saved pickup_gate, saved action-end placement_valid; NOT actual controller phase label; B zero by disabled code/config'))
    if r['arm']=='F' and r['scene_id']=='pillar_x_plus40mm' and not bool(z['episode_success']):
     a=dict(id=r['id'],seed=r['seed'],reason=str(z['termination_reason']),pillar_contact=bool(z['episode_obstacle_contact']),tool_floor_contact=bool(z['episode_tool_floor_contact']),pillar_tilt=bool(z['episode_knockdown']),duration_s=float(z['episode_duration_s']),pickup_gate_ever=bool(pickup.any()),tool_lift_now_ever=bool(pre['tool_lift_now'].any()),placement_valid_action_count=int(place.sum()),placement_valid_max_consecutive_full_actions=maxrun(place&(valid==50)),fully_inside_actions=int(q['fully_inside'].sum()),final_tool_pos_m=z['screwdriver_pos'][-1].tolist(),max_saved_action_end_tool_origin_height_m=float(z['screwdriver_pos'][:,2].max()),final_tool_center_error_m=float(q['tool_center_error_m'][-1]),geometry_intervention_actions=int(pre['geometry_active'].sum()),guard_active_substeps=int(active.sum()),source=str(p))
     a['classification']='tool_floor_contact' if a['tool_floor_contact'] else ('never_saved_valid_placement' if not place.any() else ('valid_placement_transient_below10fullactions' if a['placement_valid_max_consecutive_full_actions']<10 else 'other_strict_condition_not_met; exact all1ms grasp_lift_latches_unavailable'))
     failure.append(a)
assert len(failure)==11
pg=[]
for scene in ['main','pillar_x_plus40mm']:
 for arm in ['F','B']:
  for k in ['pre_pickup15mm_gate','post_pickup_gate_before_first_valid_placement','after_first_saved_valid_placement']:
   rr=[r for r in phase if r['scene']==scene and r['arm']==arm and r['phase_proxy']==k];assert len(rr)==12;den=sum(r['valid_substeps'] for r in rr);num=sum(r['guard_active_substeps'] for r in rr);pg.append(dict(scene=scene,arm=arm,phase_proxy=k,n_episodes=12,valid_substeps=den,guard_active_substeps=num,guard_active_fraction=num/den if den else None))
res=dict(peak200=peaks,GC_J4_geometry_load_overlap=gcj4,new72_X40_F11_failures=failure,new72_phase_proxy_per_episode=phase,new72_phase_proxy_groups=pg,technical_attempts_retained=technical,torque_history_coverage=torque,phase_limit='No saved exact lift/transport/place controller phase; proxies explicitly defined from existing15mm pickup_gate and action-end placement_valid. No1ms complete grasp/lift success latches saved; failure category is observed necessary-condition diagnostic, not causal attribution.',force_pair_limit='No per-contact force-to-identity arrays; maxima contribution pair unverified. contact_object_samples selects pillar-related pairs and stores geometry without force; cannot infer the missing maximum contributor.',ctrl_limit='executed_ctrl are position targets, not actuator torques; tau_external is observer residual, not actuator torque. q_tau is action-end sampled; episode_peak_joint_torque scalar does not give saturation duration.',physics_steps=0,policy_calls=0)
(O/'audit/峰值与72失败诊断.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
for name,rows in [('峰值200回合诊断.csv',peaks),('GC前4秒几何载荷重叠.csv',gcj4),('X40_F11失败分类.csv',failure),('F_B阶段代理guard比例.csv',pg),('逐子步力矩字段覆盖.csv',torque)]:
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (O/'audit'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('PEAK',[(m,s,sum(p['peak_before_first_two_finger_contact'] for p in peaks if p['method']==m and p['scene']==s),sum(p['peak_before_first_closing_command'] for p in peaks if p['method']==m and p['scene']==s)) for m in ['raw_pi0','pace_gc'] for s in ['main','pillar_x_plus40mm']]);print('GCoverlap',sum(r['geometry_changed_positive_load_overlap'] for r in gcj4),sum(r['guard_changed_positive_load_overlap'] for r in gcj4));print('Ffail',collections.Counter(r['classification'] for r in failure));print('phase',json.dumps(pg,ensure_ascii=False))
