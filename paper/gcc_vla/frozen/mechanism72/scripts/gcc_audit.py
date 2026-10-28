from pathlib import Path
import json,hashlib,csv,collections
import numpy as np
O=Path('/tmp/gcc_revision_path').read_text();O=Path(O);D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400'); S=D/'脚本与身份/共享运行时'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
idx=json.loads((D/'逐回合汇总.json').read_text()); rows=[]; sources=[];ident=[]
for r in idx:
 p=Path(r['package_data']); assert p.exists(),p
 with np.load(p,allow_pickle=False) as d:
  f={'task_success':'task_success','clean_success':'clean_success','pillar_contact':'pillar_contact'} if r['method']=='vlsa_native' else {'task_success':'episode_success','pillar_contact':'episode_obstacle_contact','tool_floor_contact':'episode_tool_floor_contact','pillar_tilt':'episode_knockdown'}
  for k,v in f.items():assert bool(d[v].item())==r[k],(p,k)
  if r['method'] not in ('pace_g','pace_gc'):continue
  data={k:d[k] for k in ['f_estimated','estimated_contact','f_control','valid_substeps','episode_duration_s']}
 root=p.parent; manifest=json.loads((root.parent/'manifest.json').read_text())
 mm=[]
 for n,h in manifest['modules_sha256'].items():
  q=S/n; mm.append(q.exists() and sha(q)==h)
 ident.append({'assignment_id':r['assignment_id'],'matching_modules':sum(mm),'declared_modules':len(mm),'all_match':all(mm)})
 rec={k:r[k] for k in ['method','scene','seed','assignment_id','task_success','clean_success','pillar_contact']}
 sources.append({'path':str(p),'sha256':sha(p)})
 with np.load(root/'preview_ep_001.npz') as g, np.load(root/'guard_ep_001.npz') as b, np.load(root/'paper_truth_ep_001.npz') as t:
  active=g['geometry_active'].astype(bool);gd=np.linalg.norm(g['joint_correction_rad'],axis=1);ga=b['reference_guard_active'].astype(bool);bn=np.linalg.norm(b['reference_guard_joint_delta_rad'],axis=1)
  rec.update(actions=len(active),geom_active=int(active.sum()),geom_changed=int((gd>1e-10).sum()),geom_delta_max_rad=float(gd.max()),geom_clearance_min_m=float(g['current_clearance_m'].min()),guard_samples=len(ga),guard_active=int(ga.sum()),guard_changed=int((bn>1e-10).sum()),guard_max_rad=float(bn.max()),guard_before_deficit_max_m=float(np.maximum(b['reference_guard_bound_m']-b['reference_guard_before_m'],0).max()),guard_infeasible=int((~b['reference_guard_feasible'].astype(bool)).sum()),guard_at_negative_clearance=int((ga&(b['reference_guard_clearance_m']<0)).sum()),guard_at_above_margin=int((ga&(b['reference_guard_clearance_m']>=.01)).sum()))
  for w,mask in [('full',np.ones(len(t['time_s']),bool)),('fixed4',np.arange(len(t['time_s']))<4000)]:
   for domain in ['all','task']:
    f=t[domain+'_sum_force_N'][mask]; rec[f'{domain}_{w}_J']=float(f.sum()*.001);rec[f'{domain}_{w}_peak']=float(f.max());rec[f'{domain}_{w}_exposure_s']=float((t[domain+'_contacts'][mask]>0).sum()*.001)
  if r['method']=='pace_gc':
   with np.load(root/'contact_ep_001.npz') as c:
    assert len(c['time_s'])==len(ga)==len(t['time_s'])
    v=c['nominal_velocity_m_s'];hv=c['height_delta_m_s'];fv=c['force_delta_m_s'];mv=c['mapped_velocity_m_s'];fg=np.linalg.norm(fv,axis=1)>1e-9;hg=np.linalg.norm(hv,axis=1)>1e-9;gate=c['contact_gate'].astype(bool)
    rec.update(contact_gate=int(gate.sum()),height_active=int(hg.sum()),force_active=int(fg.sum()),force_v_max=float(np.linalg.norm(fv,axis=1).max()),force_up=int((fv[:,2]>1e-9).sum()),force_down=int((fv[:,2]<-1e-9).sum()),height_s=float(hg.sum()*.001),force_s=float(fg.sum()*.001),guard_force_overlap=int((ga&fg).sum()),guard_height_overlap=int((ga&hg).sum()),force_feedback_mean_N=float(np.linalg.norm(c['feedback_force_N'][gate],axis=1).mean()),nominal_down_mean=float(np.maximum(-v[:,2],0).mean()),height_down_mean=float(np.maximum(-(v+hv)[:,2],0).mean()),preDLS_down_mean=float(np.maximum(-(v+hv+fv)[:,2],0).mean()),mapped_down_mean=float(np.maximum(-mv[:,2],0).mean()),anchor_confirmed=bool(c['support_after_valid'].any()))
 rows.append(rec)
groups=[]
for key in sorted({(r['method'],r['scene']) for r in rows}):
 rr=[r for r in rows if (r['method'],r['scene'])==key];g=dict(method=key[0],scene=key[1],n=len(rr))
 for k in rr[0]:
  if isinstance(rr[0][k],(int,float,bool)) and k!='seed':g[k]={'sum':float(sum(r[k] for r in rr)),'mean':float(np.mean([r[k] for r in rr])),'sd':float(np.std([r[k] for r in rr],ddof=1))}
 groups.append(g)
for name,obj in [('mechanism_per_episode.json',rows),('mechanism_groups.json',groups),('runtime_identity_200.json',ident),('diagnostic_sources.json',sources)]: (O/'audit'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
with (O/'audit'/'mechanism_per_episode.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
print('600 endpoint checks passed; 200 GCC rows; identities',sum(r['all_match'] for r in ident),len(ident))
for g in groups:
 print(g['method'],g['scene'],{k:g[k] for k in ['geom_active','actions','guard_active','guard_samples','guard_max_rad','guard_at_negative_clearance','guard_at_above_margin','all_full_J','all_fixed4_J','task_full_exposure_s','task_fixed4_exposure_s']})
 if g['method']=='pace_gc':print('CONTACT', {k:g[k] for k in ['contact_gate','height_active','force_active','force_up','force_down','guard_force_overlap','anchor_confirmed','nominal_down_mean','height_down_mean','preDLS_down_mean','mapped_down_mean']})
