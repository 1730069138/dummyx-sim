#!/usr/bin/env python3
"""Read-only original-trajectory force audit. No MuJoCo import, step or inference.
Outputs are created exclusively; an existing output file is never overwritten.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
import numpy as np

BASE=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
METHODS={'raw_pi0','pace_g','pace_gc','oscbf','neo_mesh'}
EXPECTED_SCENES={'main','pillar_x_plus40mm'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)

def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def stat(values):
    x=np.asarray([v for v in values if v is not None],dtype=float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,sd=float(x.std(ddof=1)) if len(x)>1 else None,
                median=float(np.median(x)) if len(x) else None,min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args();dest=args.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
    require(not (dest/'per_episode.json').exists(),'Audit exists: choose an empty new output directory; no overwrite is allowed')
    source_index=BASE/'逐回合汇总.csv';rows=list(csv.DictReader(source_index.open(encoding='utf-8-sig')))
    selected=[r for r in rows if r['method'] in METHODS];require(len(selected)==500,'Expected exactly five methods x two scenes x 50')
    sources={};episodes=[];metric_names=[]
    def sourced(path,role):
        path=path.resolve(strict=True)
        if str(path) not in sources:sources[str(path)]=dict(absolute_path=str(path),role=role,size_bytes=path.stat().st_size,sha256=sha(path))
    sourced(source_index,'frozen_600_episode_index')
    for source in [BASE/'脚本与身份/共享运行时/tabletop_paper_ablation.py',BASE/'脚本与身份/共享运行时/deploy_screwdriver_client_old.py',BASE/'脚本与身份/共享运行时/joint_torque_observer.py',Path(__file__)]:sourced(source,'force_definition_or_audit_source')
    for r in selected:
        dp=Path(r['package_data']).resolve(strict=True);fp=dp.with_name('paper_truth_ep_001.npz');sourced(dp,'original_episode_npz');sourced(fp,'original_1ms_external_contact_arrays')
        with np.load(dp,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        with np.load(fp,allow_pickle=False) as z:f={k:z[k] for k in z.files}
        label=r['assignment_id'];dt=float(a['force_sample_period_s']);start=float(a['episode_start_time']);times=np.asarray(f['time_s'],float);n=len(times);valid=np.asarray(a['valid_substeps'],int)
        require(r['scene'] in EXPECTED_SCENES,label+': unknown scene');require(abs(dt-.001)<1e-12,label+': timestep not 1ms')
        require(n==int(valid.sum()) and n>0,label+': valid-substep coverage mismatch');require(np.all((valid>=0)&(valid<=50)),label+': invalid action sample counts')
        require(len(valid)==len(a['sim_time'])==len(a['end_sim_time'])==len(a['q_tau']),label+': action-array count mismatch')
        expected=start+dt*np.arange(1,n+1);time_error=float(np.max(np.abs(times-expected)));require(time_error<1e-7,label+': force timestamp gaps or offset')
        require(abs(float(a['episode_duration_s'])-n*dt)<1e-7,label+': duration mismatch');require(np.max(np.abs((a['end_sim_time']-a['sim_time'])-valid*dt))<1e-7,label+': action duration mismatch')
        require(len(a['force_time_s'])==n and np.max(np.abs(a['force_time_s']+dt-times))<1e-7,label+': observer/truth phase mismatch')
        require(n>=4000 and np.count_nonzero(times<=start+4.+1e-8)==4000,label+': incomplete fixed first4s window')
        require(bool(a['episode_success'])==(r['task_success']=='True'),label+': source/index success mismatch')
        require(str(a['termination_reason'])==r['reason'],label+': termination reason mismatch')
        row={k:r[k] for k in ['method','scene','seed','assignment_id','run_id','package_data','package_job']}
        row.update(task_success=bool(a['episode_success']),clean_success=r['clean_success']=='True',pillar_contact=r['pillar_contact']=='True',termination_reason=str(a['termination_reason']),physical_samples=n,action_samples=len(valid),episode_start_time_s=start,duration_s=n*dt,fixed4_complete=True,force_timestamp_max_error_s=time_error,force_sample_period_s=dt,paper_truth_path=str(fp))
        for domain in ['all','task']:
            force=np.asarray(f[domain+'_sum_force_N'],float);peak=np.asarray(f[domain+'_peak_individual_N'],float);count=np.asarray(f[domain+'_contacts'])
            require(len(force)==len(peak)==len(count)==n,label+': contact-array count mismatch')
            require(np.isfinite(force).all() and np.isfinite(peak).all() and np.all(force>=-1e-12) and np.all(peak>=-1e-12),label+': nonfinite or negative physical force')
            require(np.all(peak<=force+1e-8),label+': individual peak exceeds force sum')
            for window,ff,pp,cc in [('full',force,peak,count),('fixed4',force[:4000],peak[:4000],count[:4000])]:
                duration=len(ff)*dt;moving=np.convolve(ff,np.ones(10,dtype=float)/10,mode='valid')
                metrics={f'{domain}_{window}_impulse_Ns':float(ff.sum()*dt),f'{domain}_{window}_sum_peak_N':float(ff.max()),f'{domain}_{window}_individual_peak_N':float(pp.max()),f'{domain}_{window}_mean_force_N':float(ff.sum()*dt/duration),f'{domain}_{window}_moving10ms_peak_N':float(moving.max()),f'{domain}_{window}_contact_duration_s':float(np.count_nonzero(cc>0)*dt)}
                row.update(metrics)
                for k in metrics:
                    if k not in metric_names:metric_names.append(k)
        require(np.all(f['task_sum_force_N']<=f['all_sum_force_N']+1e-8),label+': task subset exceeds all force')
        torque=np.asarray(a['q_tau'],float);require(torque.shape==(len(valid),6) and np.isfinite(torque).all(),label+': invalid action-end torque array')
        row['joint_torque_action_end_peak_Nm']=float(np.abs(torque).max())
        row['joint_torque_original_1ms_peak_scalar_Nm']=float(a['episode_peak_joint_torque'])
        require(math.isfinite(row['joint_torque_original_1ms_peak_scalar_Nm']),label+': invalid recorded torque peak')
        row['joint_torque_original_1ms_peak_scalar_source']='data_ep_001.npz:episode_peak_joint_torque; original execution-loop reduction, not independently reconstructable from 1ms torque arrays'
        episodes.append(row)
    metric_names+=['duration_s','joint_torque_action_end_peak_Nm','joint_torque_original_1ms_peak_scalar_Nm']
    groups=[];flat=[]
    for method in sorted(METHODS):
        for scene in sorted(EXPECTED_SCENES):
            rr=[r for r in episodes if r['method']==method and r['scene']==scene];require(len(rr)==50,f'{method}/{scene}: wrong n')
            group=dict(method=method,scene=scene,n=50,task_success_n=sum(r['task_success'] for r in rr),fixed4_complete_n=sum(r['fixed4_complete'] for r in rr),subsets={})
            ff={k:v for k,v in group.items() if k!='subsets'}
            for subset,predicate in [('all_episodes',lambda r:True),('success_only',lambda r:r['task_success']),('failure_only',lambda r:not r['task_success'])]:
                ss=[r for r in rr if predicate(r)];group['subsets'][subset]=dict(n=len(ss),metrics={k:stat([r[k] for r in ss]) for k in metric_names})
                ff[subset+'__n']=len(ss)
                for key,values in group['subsets'][subset]['metrics'].items():
                    for statistic,value in values.items():ff[subset+'__'+key+'__'+statistic]=value
            groups.append(group);flat.append(ff)
    write_json(dest/'per_episode.json',episodes);write_csv(dest/'per_episode.csv',episodes);write_json(dest/'groups.json',groups);write_csv(dest/'groups.csv',flat)
    write_json(dest/'sources_sha256.json',list(sources.values()))
    receipt=dict(original_episode_count=len(episodes),groups=len(groups),fixed4_complete=sum(r['fixed4_complete'] for r in episodes),original_physics_samples=sum(r['physical_samples'] for r in episodes),maximum_force_timestamp_error_s=max(r['force_timestamp_max_error_s'] for r in episodes),new_policy_inference_calls=0,new_mujoco_physics_steps=0,original_files_modified=0,metric_source='original paper_truth_ep_001.npz, recomputed 1ms arrays',excluded_methods=dict(vlsa_native='not included: original all-external 1ms contact force arrays were not stored; separate validated trajectory replay is required'),audit_script_sha256=sha(Path(__file__)),outputs=[])
    for filename in ['per_episode.json','per_episode.csv','groups.json','groups.csv','sources_sha256.json']:
        p=dest/filename;receipt['outputs'].append(dict(absolute_path=str(p),size_bytes=p.stat().st_size,sha256=sha(p)))
    write_json(dest/'validation_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
