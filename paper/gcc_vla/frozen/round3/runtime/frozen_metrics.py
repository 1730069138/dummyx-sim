from pathlib import Path
import json,numpy as np

def longest_run(mask,period=.05):
 longest=current=0
 for value in mask:
  current=current+1 if value else 0;longest=max(longest,current)
 return longest*period

def extract(job_path,result):
 out=dict(success=bool(result and result['success']),reason=result['reason'] if result else 'technical_error',recording_error=result is None)
 if not result:return out
 out.update(pillar_contact=result.get('obstacle_contact',False),floor_contact=result.get('tool_floor_contact',False),pillar_tilt=result.get('pillar_tilted',False),duration_s=result['duration_s'],control_steps=result['control_steps'],restricted_completion_s=result['duration_s'] if result['success'] else 15.,clean_success=bool(result['success'] and not result['obstacle_contact'] and not result['tool_floor_contact'] and not result['pillar_tilted']))
 files=list(Path(job_path).rglob('data_ep_001.npz'))
 if not files:return out
 src=files[0];out['data_path']=str(src)
 with np.load(src) as z:
  out.update(mean_safety_ms=float(z['safety_layer_time_ms'].mean()) if len(z['safety_layer_time_ms']) else None,p95_safety_ms=float(np.percentile(z['safety_layer_time_ms'],95)) if len(z['safety_layer_time_ms']) else None,deadline_misses=int(sum(z['safety_layer_time_ms']>50)),policy_requests=len(z['policy_inference_step_ms']),policy_total_latency_ms=float(z['policy_inference_step_ms'][:,1].sum()),tcp_path_m=float(np.linalg.norm(np.diff(z['tcp_pos'],axis=0),axis=1).sum()) if len(z['tcp_pos'])>1 else 0.,maximum_joint_torque_Nm=float(z['episode_peak_joint_torque']),legacy_task_impulse_Ns=float(z['episode_contact_impulse']))
  lineage=Path(job_path)/'policy_lineage.jsonl'
  if lineage.exists():out['first_observation_hash']=json.loads(lineage.read_text().splitlines()[0])['observation_hash']
  terminal_z=float(z['screwdriver_pos'][-1,2]) if len(z['screwdriver_pos']) else None;out['terminal_tool_origin_z_m']=terminal_z
 f=src.with_name('paper_truth_ep_001.npz')
 if f.exists():
  with np.load(f) as z:
   out['force_samples']=len(z['time_s'])
   for domain in ['all','task']:
    force=z[f'{domain}_sum_force_N'];peak=z[f'{domain}_peak_individual_N'];rolling=np.convolve(force,np.ones(10)/10,'valid') if len(force)>=10 else force
    out.update({f'{domain}_impulse_Ns':float(force.sum()*.001),f'{domain}_sum_peak_N':float(force.max()) if len(force) else None,f'{domain}_individual_peak_N':float(peak.max()) if len(peak) else None,f'{domain}_peak10ms_N':float(rolling.max()) if len(rolling) else None})
 f=src.with_name('paper_quality_ep_001.npz')
 if f.exists():
  with np.load(f) as z:
   valid=z['valid_substeps']==50;inside=longest_run(z['fully_inside']&valid);nonbox=longest_run(z['nonbox_support']&valid)
   out.update(full_containment=bool(inside>=.5-1e-9),full_containment_max_s=inside,stable_nonbox_support=bool(nonbox>=.5-1e-9),terminal_tool_center_error_m=float(z['tool_center_error_m'][-1]),terminal_home_error_rad=float(z['home_error_rad'][-1]),observed_home_tolerance=bool(np.any(z['home_error_rad']<=.1)))
 f=src.with_name('attraction_ep_001.npz')
 if f.exists():
  with np.load(f) as z:
   phase=z['phase'];out.update(attraction_active_steps=int(sum(z['active'])),transport_to_lift_reversions=int(sum((phase[:-1]=='transport')&(phase[1:]=='lift'))),orientation_projection_rejections=int(sum(z['orientation_rejected'])),attraction_correction_sum_rad=float(np.linalg.norm(z['joint_correction_rad'],axis=1).sum()))
 f=src.with_name('preview_ep_001.npz')
 if f.exists():
  with np.load(f) as z:
   if 'preview_lift_requested' in z:out['predictive_lift_requests']=int(sum(z['preview_lift_requested']))
 return out
