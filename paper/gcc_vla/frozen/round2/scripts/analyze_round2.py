#!/usr/bin/env python3
"""Read-only round2 analysis. Never imports MuJoCo, clients, worker, or controller.
Use a fresh --output-dir on every run. All source episodes and failed attempts stay intact.
"""
from pathlib import Path
import argparse, collections, csv, datetime, gzip, hashlib, json, math, re
import numpy as np
from scipy import stats

DEFAULT_ROOT=Path('/home/jun/dummyx-sim/outputs/gcc_vla_experiments_round2_20261028_190400')
PHASE={0:'approach_observable',1:'closing_observable',2:'lift_transport_observable',3:'bin_contact_or_valid_placement_observable'}
ROLE={0:'other',1:'robot',2:'tool',3:'pillar',4:'table',5:'bin',6:'floor'}

def readj(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x

def writej(p,x):
 with Path(p).open('x') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False)
def writecsv(p,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple,np.ndarray)) else clean(v) for k,v in row.items()})

def stat(values):
 a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
 return dict(n=len(a),mean=float(a.mean()) if len(a) else None,sd=float(a.std(ddof=1)) if len(a)>1 else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)
def wilson(k,n):
 if not n:return [None,None]
 z=1.959963984540054;den=1+z*z/n;c=(k/n+z*z/2/n)/den;e=z/den*math.sqrt(k/n*(1-k/n)/n+z*z/4/n/n);return [100*(c-e),100*(c+e)]
def holm(rows,m):
 assert len(rows)==m
 order=sorted(range(m),key=lambda i:rows[i]['p_raw'] if rows[i]['p_raw'] is not None else 1.)
 prev=0.
 for j,i in enumerate(order):
  p=rows[i]['p_raw'];prev=max(prev,min(1.,(m-j)*(p if p is not None else 1.)))
  rows[i]['p_holm']=prev if p is not None else None
  rows[i]['holm_family_m']=m

def find_data(jobdir,result):
 hints=[result.get('data_path'),(result.get('metrics') or {}).get('data_path')]
 for value in hints:
  if value and Path(value).exists():return Path(value)
 paths=list(jobdir.rglob('data_ep_001.npz'))
 if len(paths)>1:raise ValueError('multiple data files within one attempt: '+str(jobdir))
 return paths[0] if paths else None

def classify_identity(path,result,job,trace_meta,plan):
 seed=int(result.get('seed',job.get('seed',trace_meta.get('seed',-1))))
 scene=result.get('scene_id',job.get('scene_id',trace_meta.get('scene_id',trace_meta.get('scene'))))
 if scene in ['x40','X40','X+40','pillar_x40']:scene='pillar_x_plus40mm'
 text=' '.join(str(x) for x in [path,result.get('experiment'),job.get('experiment'),trace_meta.get('experiment')])
 experiment=next((e for e in ['exp1','exp2','exp3'] if re.search(r'(?<![a-z0-9])'+e+r'(?![0-9])',text,re.I)),None)
 if experiment is None:
  for e,p in plan['experiments'].items():
   for s,seeds in p['seeds'].items():
    if seed in seeds:experiment=e;scene=scene or s
 stage=str(result.get('stage',job.get('stage',''))).lower()
 if not stage:stage='smoke' if 'smoke' in str(path).lower() or 'pilot' in str(path).lower() else 'formal'
 cfg=job.get('config') or {};cfg=cfg if isinstance(cfg,dict) else {'label':cfg}
 value=next((v for v in [result.get('config_name'),result.get('arm'),job.get('config_name'),job.get('arm'),trace_meta.get('config_name'),trace_meta.get('config'),cfg.get('label')] if isinstance(v,str) and v),None)
 values=[value or '',str(job.get('group','')),str(cfg.get('mechanism_ablation','')),str(cfg.get('direction_mode','')),str(cfg.get('geometry_direction','')),str(cfg.get('round2_direction',''))]
 words=' '.join(values).lower();method=str(result.get('method_id',job.get('method_id',cfg.get('method_id','')))).lower()
 config=None
 if 'jump' in words or 'limited' in words:config='F_jump_limited'
 elif experiment=='exp2':config='normal' if 'normal' in words else 'fixed' if 'fixed' in words else None
 else:
  for v in values:
   v=v.lower().strip()
   if v in ['raw','raw_pi0','pi0']:config='raw';break
   if v in ['g','pace_g','geometry']:config='G';break
   if v in ['f','pace_gc','full','complete']:config='F';break
   if v in ['b','no_guard','nofinal']:config='B';break
   if v in ['oscbf','oscbf_adapted']:config='oscbf';break
  if config is None:
   if cfg.get('mechanism_ablation')=='B' or re.search(r'(^|_)b(_|$)',words):config='B'
   elif 'oscbf' in method or 'oscbf' in words:config='oscbf'
   elif method=='raw_pi0' or 'raw_pi0' in words:config='raw'
   elif method=='pace_g':config='G'
   elif method=='pace_gc':config='F'
 if experiment=='exp2' and config is None:
  direction=cfg.get('direction',trace_meta.get('direction'))
  if direction in ['fixed','normal']:config=direction
 if experiment is None or scene not in ['main','pillar_x_plus40mm'] or config is None:raise ValueError('unrecognized experiment/scene/config '+repr((experiment,scene,config,seed,values)))
 return dict(experiment=experiment,stage=stage,config=config,scene=scene,seed=seed,gpu=result.get('gpu',job.get('gpu',trace_meta.get('gpu'))))

def validate_trace(z):
 meta=json.loads(str(z['metadata_json']));done=z['step_completed'].astype(bool);n=len(done);off=z['contact_offsets']
 assert off.shape==(n+1,) and off[0]==0 and np.all(np.diff(off)>=0)
 assert off[-1]==len(z['contact_geom1'])
 assert np.allclose((z['time_after_s']-z['time_before_s'])[done],.001,atol=1e-8,rtol=0)
 for a,b,c in [('target_geometry_uG','target_nominal_a','delta_geometry'),('target_contact_ubar','target_geometry_uG','delta_contact_clipped'),('u_post_guard','target_contact_ubar','delta_guard'),('u_applied','u_post_guard','limiter_delta')]:assert np.allclose(z[a]-z[b],z[c],atol=1e-10,equal_nan=True)
 assert np.allclose(z['ctrl_written'][:,meta['arm_actuator_indices']],z['u_applied'],atol=1e-10,equal_nan=True)
 owners=np.repeat(np.arange(n),np.diff(off));norm=np.linalg.norm(z['contact_wrench_world_on_geom2'][:,:3],axis=1)
 reconstructed=np.bincount(owners,weights=norm*z['contact_protected_external'],minlength=n)
 assert np.allclose(reconstructed[done],z['sum_external_contact_force_N'][done],atol=1e-9,rtol=1e-11)
 ranges=np.asarray(meta['actuator_forcerange'])[meta['arm_actuator_indices']];limited=np.asarray(meta['actuator_forcelimited'],bool)[meta['arm_actuator_indices']]
 force=z['actuator_force_arm'];computed=limited[None,:]&((force<=ranges[None,:,0]+1e-8)|(force>=ranges[None,:,1]-1e-8))
 assert np.array_equal(computed[done],z['actuator_at_limit_arm'][done])
 assert np.isfinite(force[done]).all()
 return meta,done

def contacts_at(z,meta,k):
 a,b=map(int,z['contact_offsets'][k:k+2]);out=[]
 for j in range(a,b):
  norm=float(np.linalg.norm(z['contact_wrench_world_on_geom2'][j,:3]));g1=int(z['contact_geom1'][j]);g2=int(z['contact_geom2'][j]);r1=int(z['contact_role1'][j]);r2=int(z['contact_role2'][j])
  out.append(dict(geom1=g1,geom2=g2,geom1_name=meta['geom_names'][g1],geom2_name=meta['geom_names'][g2],roles=[ROLE.get(r1,str(r1)),ROLE.get(r2,str(r2))],force_norm_N=norm,protected_external=bool(z['contact_protected_external'][j]),protected_internal=bool(z['contact_protected_internal'][j])))
 return sorted(out,key=lambda r:r['force_norm_N'],reverse=True)

def trace_summary(z,meta,done,ident,event_writer=None):
 completed=np.flatnonzero(done);N=len(completed);a=dict(trace_attempted_steps=len(done),physical_samples=N,trace_uncompleted_steps=int((~done).sum()))
 force=z['sum_external_contact_force_N'][done];times=z['time_after_s'][done]
 if N:
  contiguous=np.allclose(np.diff(times),.001,atol=1e-8,rtol=0)
  assert contiguous,'noncontiguous completed physical steps'
  start=float(z['time_before_s'][completed[0]]);pk=int(completed[np.argmax(force)])
  a.update(J=float(force.sum()*.001),J4=float(force[:4000].sum()*.001) if N>=4000 else None,Fpk=float(force.max()),T=float(N*.001),fixed4_complete=N>=4000,trace_start_s=start,peak_relative_s=float(z['time_after_s'][pk]-start),peak_contacts=contacts_at(z,meta,pk))
  external=[r for r in a['peak_contacts'] if r['protected_external']]
  a['peak_max_contribution_external_pair']=external[0] if external else None
 else:a.update(J=None,J4=None,Fpk=None,T=0.,fixed4_complete=False,trace_start_s=None,peak_relative_s=None,peak_contacts=[],peak_max_contribution_external_pair=None)
 sat=z['actuator_at_limit_arm'].astype(bool)&done[:,None];anysat=sat.any(axis=1);where=np.flatnonzero(anysat);first=int(where[0]) if len(where) else None
 a.update(tau_limit_episode=bool(anysat.any()),tau_limit_substeps=int(anysat.sum()),tau_limit_joint_substeps=int(sat.sum()),tau_limit_per_joint=sat.sum(axis=0).tolist(),tau_limit_fraction=float(anysat.sum()/N) if N else None,first_limit_relative_s=float(z['time_before_s'][first]-a['trace_start_s']) if first is not None else None,first_limit_joint_indices=(np.flatnonzero(sat[first])+1).tolist() if first is not None else [],first_limit_phase=PHASE.get(int(z['phase_pre'][first]),'unknown') if first is not None and z['phase_metric_valid_pre'][first] else ('unknown_phase_predicates' if first is not None else None),first_limit_contacts=contacts_at(z,meta,first) if first is not None else [])
 names=['geometry','contact_clipped','OSCBF_QP_projection' if ident['config']=='oscbf' else 'GCC_final_guard'];contrib=np.stack([abs(z['delta_geometry']),abs(z['delta_contact_clipped']),abs(z['delta_guard'])],axis=2);dom=collections.Counter();phasecounts=collections.Counter();contactcounts=collections.Counter()
 if ident['experiment']!='exp1':
  joint_values=contrib[sat]
  if len(joint_values):
   maxima=joint_values.max(axis=1);ties=(abs(joint_values-maxima[:,None])<=1e-12).sum(axis=1);wins=joint_values.argmax(axis=1);dom['none_3_modules']=int((maxima<=1e-12).sum());dom['tie_3_modules']=int(((maxima>1e-12)&(ties>1)).sum())
   for j,name in enumerate(names):dom[name]=int(((maxima>1e-12)&(ties==1)&(wins==j)).sum())
  phasevalues=z['phase_pre'][where].copy();phasevalid=z['phase_metric_valid_pre'][where]
  for value,count in zip(*np.unique(np.where(phasevalid,phasevalues,-1),return_counts=True)):phasecounts[PHASE.get(int(value),'unknown_phase_predicates')]+=int(count)
 for k in (where if ident['experiment']=='exp1' else []):
  phase=PHASE.get(int(z['phase_pre'][k]),'unknown') if z['phase_metric_valid_pre'][k] else 'unknown_phase_predicates';phasecounts[phase]+=1
  pairs=contacts_at(z,meta,int(k));present={'/'.join(sorted(p['roles']))+('/internal' if p['protected_internal'] else '/external' if p['protected_external'] else '/unprotected') for p in pairs if p['force_norm_N']>0}
  contactcounts.update(present)
  for j in np.flatnonzero(sat[k]):
   v=contrib[k,j];maxv=float(v.max());w=np.flatnonzero(abs(v-maxv)<=1e-12);label='none_3_modules' if maxv<=1e-12 else names[int(w[0])] if len(w)==1 else 'tie_3_modules';dom[label]+=1
  if event_writer is not None:event_writer.writerow(dict(**ident,substep=int(k),relative_s=float(z['time_before_s'][k]-a['trace_start_s']),phase=phase,saturated_joint_indices=json.dumps((np.flatnonzero(sat[k])+1).tolist()),actual_torque_Nm=json.dumps(z['actuator_force_arm'][k].tolist()),geometry_delta_rad=json.dumps(z['delta_geometry'][k].tolist()),contact_delta_rad=json.dumps(z['delta_contact_clipped'][k].tolist()),guard_delta_rad=json.dumps(z['delta_guard'][k].tolist()),target_jump_rad=json.dumps(clean(z['delta_u_from_previous'][k])),position_error_rad=json.dumps(z['position_error_before'][k].tolist()),contacts=json.dumps(pairs,ensure_ascii=False)))
 a.update(saturated_joint_dominant_association_counts=dict(dom),saturated_substep_phase_counts=dict(phasecounts),saturated_substep_contact_category_counts=dict(contactcounts),tau_peak_per_joint_Nm=np.max(abs(z['actuator_force_arm'][done]),axis=0).tolist() if N else [None]*6,limited_demand_exceeds_joint_substeps=int((z['actuator_demand_exceeds_arm']&sat).sum()))
 return a

def phase_window(z,done,data_path):
 # Primary prewritten definition: original saved action-level cached pickup gate.
 # Sensitivity: uniform independent1ms diagnostic latch; never label as actual B guard.
 placement=z['placement_valid_post'].astype(bool)&z['phase_metric_valid_post']&done
 def window(pickup):
  picked=np.flatnonzero(done&pickup)
  if not len(picked):return dict(substeps=0,active=0,fraction=None,first_pickup_s=None,first_direct_placement_predicate_s=None)
  first_pickup=float(z['time_before_s'][picked[0]])
  placed=np.flatnonzero(placement&(z['time_after_s']>=first_pickup-1e-9))
  first_valid=float(z['time_after_s'][placed[0]]) if len(placed) else math.inf
  mask=done&pickup&(z['time_after_s']<=first_valid+1e-9);den=int(mask.sum());num=int((mask&z['guard_active']).sum())
  return dict(substeps=den,active=num,fraction=num/den if den else None,first_pickup_s=first_pickup,first_direct_placement_predicate_s=first_valid if np.isfinite(first_valid) else None)
 pickup=None;source='unavailable'
 if data_path is not None:
  p=data_path.with_name('preview_ep_001.npz')
  if p.exists():
   with np.load(p) as pre:
    action=z['action_index'].astype(int)
    if 'pickup_gate' in pre and len(pre['pickup_gate'])>int(action.max(initial=-1)):pickup=pre['pickup_gate'][action].astype(bool);source=str(p)+':pickup_gate[action_index]'
 base=window(pickup) if pickup is not None else dict(substeps=None,active=None,fraction=None,first_pickup_s=None,first_direct_placement_predicate_s=None)
 sensitivity=window(z['geometry_pickup_gate_diagnostic'].astype(bool))
 qualified=done&z['guard_debug_valid']&z['guard_qualified'];at=np.flatnonzero(qualified);first_qualified=float(z['time_before_s'][at[0]]) if len(at) else None
 result=dict(guard_window_available=pickup is not None,guard_window_source=source,guard_window_definition='primary saved original action-cached pickup gate; until first direct placement.valid predicate after pickup, not0.5s strictsuccess',ever_placement_valid=bool(placement.any()),first_actual_guard_qualification_s=first_qualified)
 result.update({'guard_window_'+k:v for k,v in base.items()})
 result.update({'diagnostic_lift_window_'+k:v for k,v in sensitivity.items()})
 result['diagnostic_lift_window_source']='independent1ms geometry_pickup_gate_diagnostic latch, uniform15mm; not actual B guard qualification'
 return result

def data_endpoints(data_path,result):
 if data_path is not None:
  with np.load(data_path) as d:
   keys=['episode_success','episode_obstacle_contact','episode_tool_floor_contact','episode_knockdown']
   if all(k in d for k in keys):
    sr,pcr,floor,tilt=[bool(d[k]) for k in keys];return dict(SR=sr,STSR=sr and not(pcr or floor or tilt),PCR=pcr,tool_floor=floor,pillar_tilt=tilt,termination_reason=str(d['termination_reason']) if 'termination_reason' in d else str((result.get('result') or {}).get('reason','')))
 m=result.get('metrics') or {};v=result.get('result') or {}
 if all(k in m for k in ['success','pillar_contact','floor_contact','pillar_tilt']):
  sr,pcr,floor,tilt=[bool(m[k]) for k in ['success','pillar_contact','floor_contact','pillar_tilt']];return dict(SR=sr,STSR=sr and not(pcr or floor or tilt),PCR=pcr,tool_floor=floor,pillar_tilt=tilt,termination_reason=str(m.get('reason',v.get('reason',''))))
 raise ValueError('No frozen endpoint fields; do not infer success from folder')

def failure_category(reason):
 if reason=='step_limit' or 'step_limit' in reason:return 'step_limit'
 if 'infeasible' in reason or 'safety_abort' in reason:return 'infeasible_abort'
 return 'other_task_termination'

def plot_preselected(rows,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 palette=['#000000','#0072B2','#D55E00','#009E73','#CC79A7','#7B6A00'];styles=['-','--','-.',':','-','--'];figs=[]
 for config in ['raw','G','F']:
  selected=[r for r in rows if r['experiment']=='exp1' and r['stage']=='formal' and r['scene']=='main' and r['seed']==810000 and r['config']==config]
  if not selected:figs.append(dict(config=config,status='preselected allocation missing; no effect-based replacement'));continue
  row=selected[0]
  with np.load(row['trace_path']) as zz:
   z={k:zz[k] for k in zz.files};meta=json.loads(str(z['metadata_json']));done=z['step_completed'].astype(bool);t=z['time_before_s'][done]-row['trace_start_s'];force=z['actuator_force_arm'][done];sat=z['actuator_at_limit_arm'][done];jump=z['delta_u_from_previous'][done].copy();jump[~z['previous_applied_valid'][done]]=np.nan
   with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'figure.facecolor':'white'}):
    fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True,layout='constrained')
    for j in range(6):
     axes[0].plot(t,force[:,j],color=palette[j],linestyle=styles[j],lw=.7,label=f'Joint {j+1}')
     axes[0].scatter(t[sat[:,j]],force[sat[:,j],j],color=palette[j],s=2,marker='x',linewidths=.4)
     axes[1].plot(t,jump[:,j],color=palette[j],linestyle=styles[j],lw=.7)
     axes[3].plot(t,z['position_error_before'][done,j],color=palette[j],linestyle=styles[j],lw=.7)
    ranges=np.asarray(meta['actuator_forcerange'])[meta['arm_actuator_indices']]
    for bound in sorted(set(ranges.reshape(-1))):axes[0].axhline(bound,color='0.4',lw=.6,linestyle=':')
    for key,label,color,style in [('delta_geometry','Geometry increment','#0072B2','-'),('delta_contact_clipped','Contact increment','#D55E00','--'),('delta_guard','Final guard increment','#009E73','-.')]:axes[2].plot(t,np.max(abs(z[key][done]),axis=1),label=label,color=color,linestyle=style,lw=.8)
    axes[0].set_ylabel('Actual actuator torque (N m)');axes[1].set_ylabel('Applied-target jump (rad)');axes[2].set_ylabel('Joint infinity norm (rad)');axes[3].set_ylabel('Position error (rad)');axes[3].set_xlabel('Episode time (s)');axes[0].legend(ncol=3,loc='best',fontsize=8);axes[2].legend(loc='best',fontsize=8)
    if row['first_limit_relative_s'] is not None:
     for ax in axes:ax.axvline(row['first_limit_relative_s'],color='0.2',lw=.7,linestyle='--')
    for ax in axes:ax.grid(alpha=.18);ax.set_xlim(0,row['T'])
    fig.suptitle(f'Preselected exp1/main seed 810000, {config}; markers = saturation; module curves = association')
    base=out/'figures'/('exp1_main_810000_'+config);fig.savefig(str(base)+'.png',dpi=300,facecolor='white');fig.savefig(str(base)+'.pdf',facecolor='white');plt.close(fig)
   figs.append(dict(config=config,status='generated; visual QA pending',png=str(base)+'.png',pdf=str(base)+'.pdf',trace=row['trace_path'],source_sha256=sha(row['trace_path']),transforms='all completed samples; no smoothing/downsampling; first jump missing retainedNaN; 3module curves per-stepjoint infinity norm; saturation flags markers',alt_text='Four aligned time panels show six actual actuator torques, six target jumps, three command contribution magnitudes, and six position errors.'))
 return figs

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--no-plots',action='store_true');p.add_argument('--only-smoke',action='store_true');p.add_argument('--freeze-jump-thresholds',action='store_true');args=p.parse_args();root=args.root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 assert not any(out.iterdir()),'Choose an empty new output directory; no overwrite'
 (out/'figures').mkdir();plan=readj(root/'statistics_plan.json');receipt=readj(root/'statistics_plan.sha256.json')
 if 'path' in receipt and 'sha256' in receipt:assert Path(receipt['path']).resolve()==(root/'statistics_plan.json').resolve() and sha(receipt['path'])==receipt['sha256'],'Frozen statistics plan drift'
 else:
  for name,h in receipt.items():assert sha(root/name)==h,'Frozen statistics plan drift'
 attempts=[];rows=[];audit=[];sources=[]
 fields=['experiment','stage','config','scene','seed','gpu','substep','relative_s','phase','saturated_joint_indices','actual_torque_Nm','geometry_delta_rad','contact_delta_rad','guard_delta_rad','target_jump_rad','position_error_rad','contacts']
 with gzip.open(out/'exp1_saturation_events.csv.gz','wt',newline='') as stream:
  ew=csv.DictWriter(stream,fieldnames=fields);ew.writeheader()
  for rp in sorted((root/'experiments').rglob('result.json')):
   jobdir=rp.parent;r=readj(rp);job=readj(jobdir/'job.json') if (jobdir/'job.json').exists() else {};tp=Path(r.get('trace_path') or jobdir/'substep_trace.npz');tm={}
   if tp.exists():
    with np.load(tp) as z:tm=json.loads(str(z['metadata_json']))
   try:ident=classify_identity(rp,r,job,tm,plan)
   except Exception as exc:audit.append(dict(path=str(rp),status='identity_unresolved',error=str(exc)));continue
   if args.only_smoke and ident['stage']!='smoke':continue
   reason=str((r.get('result') or {}).get('reason',(r.get('metrics') or {}).get('reason','')));technical=r.get('error') is not None or reason.startswith(('policy_error:','simulation_error:','perception_error:')) or bool((r.get('metrics') or {}).get('metric_error'))
   attempt=dict(**ident,result_path=str(rp),trace_path=str(tp) if tp.exists() else None,technical=technical,reason=reason,error=r.get('error'));attempts.append(attempt)
   if technical:continue
   if ident['stage']!='formal':
    if tp.exists():
     try:
      with np.load(tp) as zz:sm={k:zz[k] for k in zz.files};sm_meta,sm_done=validate_trace(sm)
      attempt['smoke_trace_validated']=True;attempt['smoke_completed_physical_steps']=int(sm_done.sum())
     except Exception as exc:attempt['smoke_trace_validated']=False;audit.append(dict(path=str(rp),status='smoke_trace_validation_failed',error=repr(exc)))
    continue
   expected=plan['experiments'][ident['experiment']]
   if ident['config']=='F_jump_limited':
    probe_plan=plan['jump_limiter_optional'];probe_valid=ident['scene']==probe_plan['scene'] and ident['seed'] in probe_plan['seeds'] and ident['experiment']=='exp1'
    if not probe_valid:audit.append(dict(path=str(rp),status='unplanned_optional_probe',identity=ident));continue
    frozen_probe=root/'audit/jump_limit_frozen.json'
    if not frozen_probe.exists():audit.append(dict(path=str(rp),status='optional_probe_without_preexisting_frozen_threshold'));continue
    assert readj(frozen_probe)['source_n']==10
    assert np.array_equal(np.asarray((job.get('config') or {}).get('target_jump_limit_rad')),np.asarray(readj(frozen_probe)['threshold_rad'])),'probe target threshold differs from frozen value'
   elif ident['seed'] not in expected['seeds'][ident['scene']] or ident['config'] not in expected['configurations']:audit.append(dict(path=str(rp),status='unplanned_formal_allocation',identity=ident));continue
   if not tp.exists():audit.append(dict(path=str(rp),status='missing_required_trace'));continue
   try:
    with np.load(tp) as zz:
     z={k:zz[k] for k in zz.files};meta,done=validate_trace(z);dp=find_data(jobdir,r);row=dict(**ident,id=r.get('id',job.get('id',jobdir.name)),result_path=str(rp),trace_path=str(tp),data_path=str(dp) if dp else None)
     row.update(data_endpoints(dp,r));row.update(trace_summary(z,meta,done,ident,ew if ident['experiment']=='exp1' else None));row.update(phase_window(z,done,dp));row['failure_category']=None if row['SR'] else failure_category(row['termination_reason'])
     if dp is not None:
      truth=dp.with_name('paper_truth_ep_001.npz')
      if truth.exists():
       with np.load(truth) as pp:assert len(pp['time_s'])==row['physical_samples'];assert np.allclose(pp['all_sum_force_N'],z['sum_external_contact_force_N'][done],atol=1e-9,rtol=1e-11),'trace versus frozen contact_evaluation force mismatch'
     duplicate=[v for v in rows if all(v[k]==row[k] for k in ['experiment','config','scene','seed'])]
     if duplicate:raise ValueError('multiple technically valid completions for one allocation; manual resolution required')
     rows.append(row)
     for path in [rp,tp]+([dp] if dp else []):sources.append(dict(path=str(path),sha256=sha(path)))
   except Exception as exc:audit.append(dict(path=str(rp),status='trace_or_endpoint_validation_failed',error=repr(exc)))
 # Infrastructure loss may leave no result.json. Preserve the frozen sidecar attempt
 # separately from its later same-allocation recovery, never as a task failure.
 journal=root/'audit/technical_interruptions.json'
 if journal.exists() and not args.only_smoke:
  records=readj(journal);assert isinstance(records,list),'technical interruption sidecar must be a list'
  represented={str(Path(a['result_path']).parent.resolve()):a for a in attempts if a.get('result_path')}
  for rec in records:
   original=Path(rec['original_path']).resolve();job=readj(original/'job.json') if (original/'job.json').exists() else {}
   normalized=dict(rec,config_name=rec.get('config_name',rec.get('config')),scene_id=rec.get('scene_id',rec.get('scene')))
   ident=classify_identity(original/'result.json',normalized,job,{},plan)
   previous=represented.get(str(original))
   if previous and previous['technical']:continue
   if previous:audit.append(dict(path=str(original),status='technical_journal_conflicts_with_non_technical_result',journal=str(journal)))
   attempts.append(dict(**ident,id=rec.get('id',job.get('id')),result_path=rec.get('result_path'),trace_path=rec.get('trace_path'),technical=True,reason=str(rec.get('reason','infrastructure_interruption_without_result')),error=rec.get('error'),original_path=str(original),source_journal=str(journal),source_journal_sha256=sha(journal),missing_result=rec.get('result_path') is None,recovery_run=rec.get('recovery_run'),endpoint_status='unknown technical interruption, not task failure, not extra allocation'))
  sources.append(dict(path=str(journal),sha256=sha(journal)))
 groups=[]
 for experiment,exp in plan['experiments'].items():
  for scene in ['main','pillar_x_plus40mm']:
   for config in exp['configurations']:
    rr=[r for r in rows if r['experiment']==experiment and r['scene']==scene and r['config']==config];g=dict(experiment=experiment,scene=scene,config=config,n_planned=exp['n_per_configuration_scene'],n_completed=len(rr),complete_cohort=len(rr)==exp['n_per_configuration_scene'])
    for key in ['SR','STSR','PCR','pillar_tilt','tau_limit_episode']:
     k=sum(r[key] for r in rr);g[key]=dict(count=k,n=len(rr),percent=100*k/len(rr) if rr else None,wilson95_percent=wilson(k,len(rr)))
    for key in ['J','J4','Fpk','T','tau_limit_substeps','first_limit_relative_s']:g[key]=stat([r[key] for r in rr])
    g['failure_categories']=dict(collections.Counter(r['failure_category'] for r in rr if not r['SR']));g['first_limit_phase_distribution']=dict(collections.Counter(r['first_limit_phase'] for r in rr if r['tau_limit_episode']));g['dominant_module_associations']=dict(sum((collections.Counter(r['saturated_joint_dominant_association_counts']) for r in rr),collections.Counter()))
    den=sum(r['guard_window_substeps'] or 0 for r in rr if not r['SR']);num=sum(r['guard_window_active'] or 0 for r in rr if not r['SR']);g['failed_postpickup_prefirstplacement_guard']=dict(available_failed_episodes=sum(r['guard_window_available'] and not r['SR'] for r in rr),valid_substeps=den,active_substeps=num,fraction=num/den if den else None)
    diagden=sum(r['diagnostic_lift_window_substeps'] or 0 for r in rr if not r['SR']);diagnum=sum(r['diagnostic_lift_window_active'] or 0 for r in rr if not r['SR']);g['failed_diagnostic_lift_sensitivity_guard']=dict(valid_substeps=diagden,active_substeps=diagnum,fraction=diagnum/diagden if diagden else None,source='uniform independent1ms15mm latch; not actual guard gate')
    groups.append(g)
 tests=[]
 for fam in plan['families']:
  family=[];cfgA,cfgB=('normal','fixed') if fam['experiment']=='exp2' else ('F','B')
  for scene in fam['scenes']:
   aa=[r for r in rows if r['experiment']==fam['experiment'] and r['scene']==scene and r['config']==cfgA];bb=[r for r in rows if r['experiment']==fam['experiment'] and r['scene']==scene and r['config']==cfgB];ready=len(aa)==len(bb)==50
   for key in fam['outcomes']:
    t=dict(family=fam['id'],experiment=fam['experiment'],scene=scene,outcome=key,config_A=cfgA,config_B=cfgB,test=fam['test'],planned_cohort_complete=ready,p_raw=None)
    if key in ['SR','STSR','PCR']:
     ka=sum(r[key] for r in aa);kb=sum(r[key] for r in bb);t.update(n_A=len(aa),n_B=len(bb),count_A=ka,count_B=kb,risk_difference_pp=100*(ka/len(aa)-kb/len(bb)) if aa and bb else None,odds_ratio=None)
     if ready:test=stats.fisher_exact([[ka,len(aa)-ka],[kb,len(bb)-kb]],alternative='two-sided');t.update(p_raw=float(test.pvalue),odds_ratio=float(test.statistic))
    else:
     va=np.array([r[key] for r in aa if r[key] is not None]);vb=np.array([r[key] for r in bb if r[key] is not None]);t.update(n_A=len(va),n_B=len(vb),U=None,cliffs_delta=None,mean_difference=None,coverage_note='J4 unavailable for early termination; no zero fill; observed-complete4s subset may be selected by termination')
     if len(va) and len(vb):t.update(cliffs_delta=float(np.sign(va[:,None]-vb[None,:]).mean()),mean_difference=float(va.mean()-vb.mean()))
     if ready and len(va) and len(vb):test=stats.mannwhitneyu(va,vb,alternative='two-sided',method='asymptotic',use_continuity=True);t.update(U=float(test.statistic),p_raw=float(test.pvalue))
    if not ready:t['reason']='planned full cohorts incomplete; no interim inferential claim'
    family.append(t)
  holm(family,fam['m']);tests.extend(family)
 primary_rows=[r for r in rows if r['config']!='F_jump_limited'];probe_rows=[r for r in rows if r['config']=='F_jump_limited']
 probe_groups=[]
 for config in ['F','F_jump_limited']:
  rr=[r for r in rows if r['experiment']=='exp1' and r['scene']=='pillar_x_plus40mm' and r['config']==config];g=dict(config=config,scene='pillar_x_plus40mm',n_planned=5,n_completed=len(rr),SR=sum(r['SR'] for r in rr),STSR=sum(r['STSR'] for r in rr),PCR=sum(r['PCR'] for r in rr),tau_limit_episodes=sum(r['tau_limit_episode'] for r in rr))
  for key in ['J','J4','Fpk','T','tau_limit_substeps']:g[key]=stat([r[key] for r in rr])
  probe_groups.append(g)
 calibration=None
 raw=[r for r in rows if r['experiment']=='exp1' and r['config']=='raw']
 if args.freeze_jump_thresholds:
  assert len(raw)==10 and all(g['complete_cohort'] for g in groups if g['experiment']=='exp1' and g['config']=='raw'),'all10raw allocations required before calibration'
  samples=[[] for j in range(6)];rawsources=[]
  for r in raw:
   with np.load(r['trace_path']) as z:
    mask=z['step_completed']&z['previous_applied_valid'];jump=abs(z['delta_u_from_previous'][mask])
    for j in range(6):samples[j].extend(jump[:,j][np.isfinite(jump[:,j])&(jump[:,j]>1e-12)].tolist())
   rawsources.append(dict(path=r['trace_path'],sha256=sha(r['trace_path'])))
  thresholds=[max(1e-4,float(np.percentile(v,99,method='linear'))) if v else 1e-4 for v in samples]
  calibration=dict(source_n=10,rule='per-joint99percentile of all10raw valid completed absolute jumps>1e-12rad; floor1e-4rad',nonzero_epsilon_rad=1e-12,threshold_rad=thresholds,nonzero_n=[len(v) for v in samples],sources=rawsources,statistics_plan_sha256=sha(root/'statistics_plan.json'),independent_recalculation=True)
  calibration_path=root/'audit/jump_limit_frozen.json'
  if calibration_path.exists():
   historical=readj(calibration_path);assert historical['threshold_rad']==thresholds and historical.get('source_n')==10,'frozen threshold drift; do not overwrite'
   expected_sources={v.get('trace_path',v.get('path')):v['sha256'] for v in historical['sources']};actual_sources={v['path']:v['sha256'] for v in rawsources};assert expected_sources==actual_sources,'raw10source hash drift'
   assert historical['nonzero_samples_per_joint']==[len(v) for v in samples],'calibration sample count mismatch'
   assert historical.get('frozen_before_probe') is True,'historical pre-probe freeze not established'
   calibration.update(historical_frozen_reference=str(calibration_path),historical_frozen_sha256=sha(calibration_path),historical_record_says_frozen_before_probe=True,all10source_sha256_match=True)
  else:
   assert not any(a['config']=='F_jump_limited' for a in attempts),'cannot create pre-probe freeze after probe attempts'
   calibration.update(frozen_before_probe=True,nonzero_samples_per_joint=[len(v) for v in samples]);writej(calibration_path,calibration)
  writej(out/'jump_thresholds_independent_recalculation.json',calibration)
 figures=[] if args.no_plots else plot_preselected(rows,out)
 result=dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),planned_formal=440,completed_formal=len(primary_rows),completed_optional_probe=len(probe_rows),attempts=attempts,validation_issues=audit,groups=groups,tests=tests,per_episode=primary_rows,optional_probe_per_episode=probe_rows,optional_probe_descriptive_groups=probe_groups,figures=figures,jump_calibration=calibration,sources=sources,script_sha256=sha(__file__),statistics_plan_sha256=sha(root/'statistics_plan.json'),software=dict(numpy=np.__version__,scipy=__import__('scipy').__version__),physics_steps=0,policy_calls=0,notes=['Saturation contribution labels are synchronous associations, not causal attribution. OSCBF uses the current_guard hook for its own QP; its delta_guard is OSCBF_QP_projection, not GCC final guard.','Fsum is scalar sum of external force norms, not directional net impulse.','Observed phase rules are not task-stage ground truth.','Additional layout study not run.'])
 flat_diagnostics=[]
 for g in groups:
  if g['experiment']!='exp1':continue
  rr=[r for r in primary_rows if r['experiment']=='exp1' and r['scene']==g['scene'] and r['config']==g['config']]
  flat=dict(experiment='exp1',scene=g['scene'],config=g['config'],n_completed=g['n_completed'],limit_episodes=g['tau_limit_episode']['count'],no_limit_episodes=g['n_completed']-g['tau_limit_episode']['count'],limit_steps_per_episode_mean=g['tau_limit_substeps']['mean'],limit_steps_per_episode_sd=g['tau_limit_substeps']['sd'],saturated_joint_times_substeps=sum(r['tau_limit_joint_substeps'] for r in rr))
  for phase in [*PHASE.values(),'unknown_phase_predicates']:flat['first_limit_episodes_'+phase]=g['first_limit_phase_distribution'].get(phase,0)
  for label in ['geometry','contact_clipped','GCC_final_guard','OSCBF_QP_projection','tie_3_modules','none_3_modules']:flat['dominant_joint_substeps_'+label]=g['dominant_module_associations'].get(label,0)
  flat_diagnostics.append(flat)
 writecsv(out/'exp1_diagnostics_flat.csv',flat_diagnostics)
 window_rows=[]
 for g in groups:
  if g['experiment']=='exp3':
   a=g['failed_postpickup_prefirstplacement_guard'];b=g['failed_diagnostic_lift_sensitivity_guard'];window_rows.append(dict(scene=g['scene'],config=g['config'],failed_episodes=g['n_completed']-g['SR']['count'],action_cached_pickup_window_substeps=a['valid_substeps'],action_cached_guard_active=a['active_substeps'],action_cached_guard_fraction=a['fraction'],diagnostic1ms_lift_window_substeps=b['valid_substeps'],diagnostic1ms_guard_active=b['active_substeps'],diagnostic1ms_guard_fraction=b['fraction'],placement_end='first direct placement.valid predicate after pickup, not0.5s strict success'))
 writecsv(out/'exp3_failure_guard_windows.csv',window_rows)
 writej(out/'analysis_round2.json',result);writecsv(out/'per_episode.csv',primary_rows);writecsv(out/'optional_probe_per_episode.csv',probe_rows);writecsv(out/'optional_probe_descriptive_groups.csv',probe_groups);writecsv(out/'groups.csv',groups);writecsv(out/'tests.csv',tests);writecsv(out/'attempts.csv',attempts);writecsv(out/'validation_issues.csv',audit)
 md=['# GCC-VLA第二轮新增实验结果','','只读统计输出；本目录保存本次分析快照，不修改原始记录或已冻结配置。',f"实际技术有效主要正式分配 {len(primary_rows)}/440；另有可选限幅诊断{len(probe_rows)}/5；存在{sum(a['technical'] for a in attempts)}次已保留技术异常尝试；日志/身份核验问题{len(audit)}项。冒烟回合单列，不并入正式分母。",'']
 for experiment,exp in plan['experiments'].items():
  md += [f'## {experiment}',f"种子范围：main{min(exp['seeds']['main'])}–{max(exp['seeds']['main'])}；X40{min(exp['seeds']['pillar_x_plus40mm'])}–{max(exp['seeds']['pillar_x_plus40mm'])}。",'', '|场景|配置|有效/计划|SR|STSR|PCR|J均值±SD N·s|J4均值±SD N·s（有效n）|FΣ峰值 N|触限回合|T s|','|---|---|---|---|---|---|---|---|---|---|---|']
  def fmt(v):return '缺失' if v['mean'] is None else f"{v['mean']:.3f}±{v['sd']:.3f}" if v['sd'] is not None else f"{v['mean']:.3f}（n=1）"
  def prop(v):
   lo,hi=v['wilson95_percent'];return f"{v['count']}/{v['n']}（{lo:.1f}–{hi:.1f}%）" if v['n'] else '未运行'
  for g in [g for g in groups if g['experiment']==experiment]:md.append(f"|{g['scene']}|{g['config']}|{g['n_completed']}/{g['n_planned']}|{prop(g['SR'])}|{prop(g['STSR'])}|{prop(g['PCR'])}|{fmt(g['J'])}|{fmt(g['J4'])}（n={g['J4']['n']}）|{fmt(g['Fpk'])}|{g['tau_limit_episode']['count']}/{g['n_completed']}|{fmt(g['T'])}|")
  md += ['','比例区间为Wilson95%；连续量先逐回合再报告样本SD。J4早终止缺失，不补零。任务失败分类、首次触限观察阶段及同期最大模块分布见groups.csv；物理帧不是独立实验样本。','']
  tt=[t for t in tests if t['experiment']==experiment]
  if tt:
   md+=['|族|场景|结果|双侧原始p|Holm p|效应量|','|---|---|---|---|---|---|']
   for t in tt:md.append(f"|{t['family']}|{t['scene']}|{t['outcome']}|{t['p_raw']}|{t['p_holm']}|{t.get('cliffs_delta',t.get('risk_difference_pp'))}|")
  if experiment=='exp1':
   md+=['','|场景|配置|触限回合|每回合触限子步均值±SD|首次触限观察阶段回合分布|饱和关节×子步最大贡献同刻计数（非因果）|','|---|---|---|---|---|---|']
   for g in [g for g in groups if g['experiment']=='exp1']:md.append(f"|{g['scene']}|{g['config']}|{g['tau_limit_episode']['count']}/{g['n_completed']}|{fmt(g['tau_limit_substeps'])}|{json.dumps(g['first_limit_phase_distribution'],ensure_ascii=False)}|{json.dumps(g['dominant_module_associations'],ensure_ascii=False)}|")
   md+=['','可读诊断汇总已拆平为exp1_diagnostics_flat.csv；首次阶段按触限回合计，主导贡献按饱和关节×子步计，二者分母不同。触限time_before为区间开始，接触力time_after为同一完成区间求解结果。预声明典型图为main810000的raw/G/F；完整路径与缺失状态见analysis_round2.json.figures。触限各时刻模块增量、跳变、位置误差和全部接触对存原trace；同步关联明细见exp1_saturation_events.csv.gz，不能改称模块因果归因。OSCBF的projection hook是其自身QP，不是GCC最终复核。']
 md += ['## 可选跳变限幅诊断（独立于440主分配）',f'完成{len(probe_rows)}/5；阈值来源及冻结状态见audit/jump_limit_frozen.json，逐回合见optional_probe_per_episode.csv，F基线与限幅5例的描述汇总见optional_probe_descriptive_groups.csv。', '', '## 异常、未完成与解释边界','','主要guard失败窗口按原动作级preview缓存pickup门控，到首次原placement.valid谓词为止（非0.5s严格成功）；另列统一1ms15mm诊断锁存敏感性，F另存真实guard资格起点。缓存门控与子步资格可能相差最多49ms，不能静默互换。所有技术尝试及恢复保留在attempts.csv；重复有效完成、身份不明、trace与原评价不一致单列validation_issues。未完成计划不作确认性优势表述。失败从不按有利结果筛除。目标跳变限幅只有在原10条raw阈值来源完成并冻结后才可运行，单列诊断。额外偏移布局加分项未运行。','', '本报告使用既定contact_evaluation外接触定义；三模块分解不能判定哪个模块因果导致触限。B的零guard来自关闭配置与实际缺调用，不能用更快完成掩盖真实柱接触或倾倒。图形尚需实际打开作最终视觉QA。']
 with (out/'results_round2.md').open('x') as f:f.write('\n'.join(md))
 print(json.dumps(dict(output=str(out),completed=len(primary_rows),optional_probe_completed=len(probe_rows),technical_attempts=sum(a['technical'] for a in attempts),validation_issues=len(audit),figures=len(figures)),ensure_ascii=False))
if __name__=='__main__':main()
