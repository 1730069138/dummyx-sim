#!/usr/bin/env python3
"""Merge original500 loads, validated replay100 loads and original600 timing.

This script performs file analysis only, never imports MuJoCo or policy clients.
It must only be run after all 100 exact replay validations are complete. Output
files use exclusive creation and are written only after the complete cohort is
validated in memory. No partial group or source modification is permitted.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math
import numpy as np

ROOT=Path('/home/jun/dummyx-sim/outputs/gcc_vla_load_latency_20261027')
PACKAGE=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
METHOD_ORDER=['raw_pi0','pace_g','pace_gc','oscbf','neo_mesh','vlsa_native']
LABELS={'raw_pi0':'纯 π0','pace_g':'GCC-VLA G','pace_gc':'GCC-VLA G+C','oscbf':'OSCBF 独立适配','neo_mesh':'NEO 网格独立适配','vlsa_native':'VLSA 原脚本协议对齐版'}
SCENES=['main','pillar_x_plus40mm']
SCENE_LABEL={'main':'原场景','pillar_x_plus40mm':'柱体 X+40 mm'}
SUBSETS=['all_episodes','success_only','failure_only']
LOAD_SUFFIXES=['impulse_Ns','sum_peak_N','individual_peak_N','mean_force_N','moving10ms_peak_N','contact_duration_s']
LOAD_METRICS=[f'{domain}_{window}_{suffix}' for domain in ['all','task'] for window in ['full','fixed4'] for suffix in LOAD_SUFFIXES]
TORQUE_METRICS=['joint_torque_original_1ms_peak_scalar_Nm','joint_torque_action_end_peak_Nm','joint_torque_replayed_1ms_peak_Nm']

def require(x,msg):
    if not x:raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def loadj(p):
    with Path(p).open(encoding='utf8') as f:return json.load(f)

def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def writej(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def writecsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def summary(v):
    a=np.asarray([x for x in v if x is not None],float)
    require(np.isfinite(a).all(),'Nonfinite grouped value')
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,sd=float(a.std(ddof=1)) if len(a)>1 else None,
                median=float(np.median(a)) if len(a) else None,p95=float(np.percentile(a,95,method='linear')) if len(a) else None,
                min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)

def clock(a):
    a=np.asarray(a,float).reshape(-1);a=a[np.isfinite(a)]
    return dict(n=len(a),mean_ms=float(a.mean()) if len(a) else None,p50_ms=float(np.percentile(a,50,method='linear')) if len(a) else None,
                p95_ms=float(np.percentile(a,95,method='linear')) if len(a) else None,over50_count=int(np.count_nonzero(a>50)) if len(a) else None)

def key(r):return r['method'],r['scene'],str(r['seed'])

def boolean(x):
    if isinstance(x,bool):return x
    require(x in ['True','False'],'Unexpected boolean '+str(x))
    return x=='True'

def number(x):return None if x in [None,''] else float(x)

def fstat(v,digits=2):
    if v['n']==0:return '—'
    if v['sd'] is None:return f"{v['mean']:.{digits}f}"
    return f"{v['mean']:.{digits}f} ± {v['sd']:.{digits}f}"

def markdown(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(r.get(k,'')) for k in headers)+' |\n' for r in rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent);args=ap.parse_args();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    names=['per_episode.json','per_episode.csv','groups.json','groups.csv','publication_main_table.csv','publication_main_table.md','publication_loads.csv','publication_loads.md','publication_all_metrics_long.csv','sources_sha256.json','README.md','validation_receipt.json']
    require(not any((out/n).exists() for n in names),'Some output already exists: choose a new output directory; never overwrite')
    idxp=PACKAGE/'逐回合汇总.csv';op=ROOT/'load_audit_original500/per_episode.json';tp=ROOT/'timing_audit/per_episode.csv';tgp=ROOT/'timing_audit/groups.json'
    index=csvrows(idxp);original=loadj(op);timings=csvrows(tp);tg=loadj(tgp)['groups']
    require(len(index)==len(timings)==600 and len(original)==500 and len(tg)==12,'Input count mismatch')
    ii={key(r):r for r in index};oo={key(r):r for r in original};tt={key(r):r for r in timings};gg={(r['method'],r['scene']):r for r in tg}
    require(len(ii)==len(tt)==600 and len(oo)==500 and set(tt)==set(ii),'Duplicate or missing input identities')
    require(set(oo)=={k for k in ii if k[0]!='vlsa_native'},'Original500 does not exactly match indexed five methods')
    require(all(sum(r['method']==m and r['scene']==s for r in index)==50 for m in METHOD_ORDER for s in SCENES),'Not six methods x two scenes x50')
    sources={}
    def sourced(p,role):
        p=Path(p).resolve(strict=True)
        if str(p) not in sources:sources[str(p)]=dict(absolute_path=str(p),role=role,size_bytes=p.stat().st_size,sha256=sha(p))
    for p,role in [(idxp,'original_complete600_index'),(op,'original500_load_metrics'),(tp,'original600_recorded_timing_metrics'),(tgp,'original600_recorded_timing_groups'),(Path(__file__),'offline_merge_source'),(ROOT/'load_audit_original500/sources_sha256.json','original500_source_hash_manifest'),(ROOT/'timing_audit/sources.json','timing_source_hash_manifest')]:sourced(p,role)
    # Validate prior audit's source manifest before accepting its derived500 metrics.
    for item in loadj(ROOT/'load_audit_original500/sources_sha256.json'):
        require(sha(item['absolute_path'])==item['sha256'],'Original500 source hash drift: '+item['absolute_path'])
        sourced(item['absolute_path'],'original500_manifest_verified_source')
    replay_records={};validation_errors=defaultdict(float);event_totals=Counter();maximum_nominal_error=0.;sum_replay_steps=0
    # Gate the WHOLE100 first; no metrics/output can be accepted from partial replay.
    replay_index=[r for r in index if r['method']=='vlsa_native'];require(len(replay_index)==100,'Expected100 VLSA indexed episodes')
    for r in replay_index:
        folder=ROOT/'replay_exact'/r['scene']/r['assignment_id'];vp=folder/'replay_validation.json';fp=folder/'replayed_external_loads.npz';v=loadj(vp)
        label=r['assignment_id']
        require(v.get('admitted') is True and v.get('endpoints_match') is True,label+': replay not admitted/endpoints mismatch')
        require(v.get('initialization_integration_exact') is True,label+': initial integration not exact')
        require(v['method']=='vlsa_native' and v['assignment']==label and v['scene']==r['scene'] and str(v['seed'])==r['seed'] and v['backend']==r['backend'],label+': replay identity mismatch')
        require(v['new_inference_calls']==0 and v['qp_calls']==0,label+': replay contains new inference or optimization')
        require(set(v['errors'])=={'q_start_rad','tcp_end_m','eef_end_m','tool_end_m','tool_z_substep_m','time_s','max_height_m','pillar_up'},label+': validation error schema changed')
        require(set(v['event_mismatches'])=={'pillar_contact','finger_count','floor_latch','placement_valid','grasp_latch','strict_valid','strict_frames'},label+': event schema changed')
        for name,val in v['errors'].items():
            require(math.isfinite(float(val)) and 0<=val<=(1e-12 if name=='q_start_rad' else 0.),label+': nonexact replay error '+name)
            validation_errors[name]=max(validation_errors[name],float(val))
        require(all(int(x)==0 for x in v['event_mismatches'].values()),label+': event mismatch')
        event_totals.update(v['event_mismatches'])
        require(v['legacy_impulse_abs_error_Ns']==0 and v['original_legacy_impulse_Ns']==v['replayed_legacy_impulse_Ns'] and v['original_legacy_peak_N']==v['replayed_legacy_peak_N'],label+': original scalar force history not exact')
        require(math.isfinite(v['nominal_twist_max_error']) and v['nominal_twist_max_error']<=1e-12,label+': nominal twist reconstruction error')
        maximum_nominal_error=max(maximum_nominal_error,v['nominal_twist_max_error'])
        for k in ['task_success','clean_success','pillar_contact','tool_floor_contact','pillar_tilt']:
            require(v['endpoints'][k]==boolean(r[k]),label+': endpoint/index mismatch '+k)
        for pathkey,hashkey in [('data_path','data_sha256'),('aligned_path','aligned_sha256'),('initial_state_path','initial_sha256'),('model_path','model_sha256')]:
            require(sha(v[pathkey])==v[hashkey],label+': input hash drift '+pathkey);sourced(v[pathkey],'validated_replay_original_input')
        require(Path(v['aligned_path']).resolve()==Path(r['package_data']).resolve(),label+': endpoint source path mismatch')
        sourced(vp,'exact_replay_validation');sourced(fp,'validated_replay100_external_contact_arrays')
        replay_records[key(r)]=(v,fp,vp);sum_replay_steps+=v['physical_substeps']
    require(len(replay_records)==100,'Not all100 replay records admitted')
    episodes=[];timing_arrays={}
    for r in index:
        k=key(r);tr=tt[k]
        require(tr['assignment_id']==r['assignment_id'] and tr['backend']==r['backend'],'Timing identity mismatch')
        if r['method']!='vlsa_native':
            row=dict(oo[k]);row.update(load_source_kind='original_recorded_1ms_external_contacts',load_original_independent_episode=True,replay_of_existing_episode=False,
                                    joint_torque_replayed_1ms_peak_Nm=None,replay_validation_path=None,replay_load_path=None)
            for name in ['task_success','clean_success','pillar_contact']:require(row[name]==boolean(r[name]),'Original500/index outcome mismatch')
        else:
            v,fp,vp=replay_records[k];side=Path(r['package_data']);dp=side.with_name('data_ep_001.npz')
            with np.load(side,allow_pickle=False) as z:record={n:z[n] for n in z.files}
            with np.load(dp,allow_pickle=False) as z:old={n:z[n] for n in z.files}
            with np.load(fp,allow_pickle=False) as z:forces={n:z[n] for n in z.files}
            actions=np.asarray(record['actions'],float);subs=np.asarray(record['substeps'],float);valid=actions[:,2].astype(int);n=len(subs);dt=.001
            require(n==int(valid.sum())==v['physical_substeps'] and len(actions)==v['action_count'] and np.all((valid>=0)&(valid<=50)),r['assignment_id']+': valid substep count mismatch')
            times=np.asarray(forces['time_s'],float);require(len(times)==n and n>=4000,'Insufficient replay sample coverage')
            start=float(subs[0,0]-dt);time_error=float(np.max(np.abs(times-(start+dt*np.arange(1,n+1)))))
            require(time_error<1e-7 and np.max(np.abs(times-subs[:,0]))==0.,'Replay force timestamp differs from original substeps')
            require(abs(v['simulation_duration_s']-n*dt)<1e-7 and np.count_nonzero(times<=start+4.+1e-8)==4000,'Replay duration/fixed4 coverage mismatch')
            row={n:r[n] for n in ['method','scene','seed','assignment_id','run_id','package_data','package_job']}
            row.update(task_success=boolean(r['task_success']),clean_success=boolean(r['clean_success']),pillar_contact=boolean(r['pillar_contact']),termination_reason=r['reason'],
                       physical_samples=n,action_samples=len(actions),episode_start_time_s=start,duration_s=n*dt,fixed4_complete=True,force_timestamp_max_error_s=time_error,force_sample_period_s=dt,
                       paper_truth_path=None,load_source_kind='validated_exact_replay_of_archived_VLSA_execution',load_original_independent_episode=True,replay_of_existing_episode=True,
                       replay_validation_path=str(vp),replay_load_path=str(fp))
            for domain in ['all','task']:
                force=np.asarray(forces[domain+'_sum_force_N'],float);peak=np.asarray(forces[domain+'_peak_individual_N'],float);count=np.asarray(forces[domain+'_contacts'])
                require(len(force)==len(peak)==len(count)==n and np.isfinite(force).all() and np.isfinite(peak).all() and np.all(force>=-1e-12) and np.all(peak>=-1e-12),'Invalid replay force')
                require(np.all(peak<=force+1e-8),'Individual contact force larger than summed force')
                for window,ff,pp,cc in [('full',force,peak,count),('fixed4',force[:4000],peak[:4000],count[:4000])]:
                    duration=len(ff)*dt;moving=np.convolve(ff,np.ones(10)/10,mode='valid')
                    row.update({f'{domain}_{window}_impulse_Ns':float(ff.sum()*dt),f'{domain}_{window}_sum_peak_N':float(ff.max()),
                                f'{domain}_{window}_individual_peak_N':float(pp.max()),f'{domain}_{window}_mean_force_N':float(ff.sum()*dt/duration),
                                f'{domain}_{window}_moving10ms_peak_N':float(moving.max()),f'{domain}_{window}_contact_duration_s':float(np.count_nonzero(cc>0)*dt)})
            require(np.all(forces['task_sum_force_N']<=forces['all_sum_force_N']+1e-8),'Task force subset exceeds all force')
            row['joint_torque_action_end_peak_Nm']=None
            require('q_tau' not in old,'VLSA torque schema changed: review action-end source')
            row['joint_torque_original_1ms_peak_scalar_Nm']=float(old['episode_peak_joint_torque'])
            row['joint_torque_original_1ms_peak_scalar_source']='original data_ep_001.npz:episode_peak_joint_torque; original-loop scalar preserved, distinct from replayed torque'
            row['joint_torque_replayed_1ms_peak_Nm']=float(v['peak_torque_Nm'])
        row.update(display_method=LABELS[r['method']],display_scene=SCENE_LABEL[r['scene']],backend=r['backend'],tool_floor_contact=boolean(r['tool_floor_contact']),pillar_tilt=boolean(r['pillar_tilt']))
        for name in ['safety_mean_ms','safety_sample_sd_ms','safety_p50_ms','safety_p95_ms','apf_mean_ms','contact_controller_mean_ms','qp_mean_ms','raw_common_execution_observer_mean_ms']:
            row[name]=number(tr.get(name))
        for name in ['logged_action_count','safety_finite_count']:
            row[name]=int(tr[name])
        row['safety_over50_count']=None if tr['safety_over50_count']=='' else int(tr['safety_over50_count'])
        row['safety_timing_path']=tr['timing_data_file'];row['safety_timing_source_kind']='original_execution_in_situ_wall_clock_not_replay'
        require(sha(tr['timing_data_file'])==tr['timing_data_sha256'],'Timing source hash changed')
        sourced(tr['timing_data_file'],'original_in_situ_timing_and_recorded_torque')
        with np.load(tr['timing_data_file'],allow_pickle=False) as z:t=np.asarray(z['safety_layer_time_ms'],float);timing_arrays[k]=t[np.isfinite(t)]
        require(len(timing_arrays[k])==row['safety_finite_count'],'Timing sample count changed')
        episodes.append(row)
    require(len(episodes)==600 and len({key(r) for r in episodes})==600,'Final episode count mismatch')
    metric_names=LOAD_METRICS+['duration_s']+TORQUE_METRICS+['safety_mean_ms','safety_p95_ms','contact_controller_mean_ms','qp_mean_ms']
    groups=[];flat=[];main_table=[];loads_table=[];long=[]
    for m in METHOD_ORDER:
        for s in SCENES:
            rr=[r for r in episodes if r['method']==m and r['scene']==s];require(len(rr)==50,'Incomplete final group')
            group=dict(method=m,display_method=LABELS[m],scene=s,display_scene=SCENE_LABEL[s],n=50,task_success_n=sum(r['task_success'] for r in rr),
                       clean_success_n=sum(r['clean_success'] for r in rr),pillar_contact_n=sum(r['pillar_contact'] for r in rr),tool_floor_contact_n=sum(r['tool_floor_contact'] for r in rr),pillar_tilt_n=sum(r['pillar_tilt'] for r in rr),
                       fixed4_complete_n=sum(r['fixed4_complete'] for r in rr),backend_counts=dict(Counter(r['backend'] for r in rr)),source_counts=dict(Counter(r['load_source_kind'] for r in rr)),subsets={})
            flatrow={k:v for k,v in group.items() if k!='subsets'}
            flatrow['backend_counts']=json.dumps(flatrow['backend_counts'],ensure_ascii=False);flatrow['source_counts']=json.dumps(flatrow['source_counts'],ensure_ascii=False)
            for subset in SUBSETS:
                ss=[r for r in rr if subset=='all_episodes' or (r['task_success'] if subset=='success_only' else not r['task_success'])]
                metrics={k:summary([r.get(k) for r in ss]) for k in metric_names};pooled=np.concatenate([timing_arrays[key(r)] for r in ss]) if ss else np.array([]);pc=clock(pooled)
                group['subsets'][subset]=dict(n=len(ss),metrics=metrics,runtime_pooled_original_actions=pc)
                flatrow[subset+'__n']=len(ss)
                for metric,st in metrics.items():
                    for stat,val in st.items():flatrow[subset+'__'+metric+'__'+stat]=val
                    long.append(dict(method=m,display_method=LABELS[m],scene=s,display_scene=SCENE_LABEL[s],subset=subset,subset_episode_n=len(ss),metric=metric,**st))
                for k,vv in pc.items():flatrow[subset+'__runtime_pooled__'+k]=vv
                for domain in ['all','task']:
                    for window in ['full','fixed4']:
                        pub=dict(方法=LABELS[m],场景=SCENE_LABEL[s],回合子集=subset,n=len(ss),接触域=domain,时间窗=window)
                        for suffix in LOAD_SUFFIXES:pub[suffix+' mean±SD']=fstat(metrics[f'{domain}_{window}_{suffix}'])
                        loads_table.append(pub)
            stored=gg[(m,s)];pc=group['subsets']['all_episodes']['runtime_pooled_original_actions'];ms=group['subsets']['all_episodes']['metrics']
            require(pc['n']==stored['pooled_action_sample_count'],'Pooled clock n mismatch')
            if pc['n']:
                require(abs(pc['mean_ms']-stored['pooled_action_mean_ms'])<1e-10 and abs(pc['p95_ms']-stored['pooled_action_p95_ms'])<1e-10,'Pooled clock metric changed')
            groups.append(group);flat.append(flatrow)
            pub=dict(方法=LABELS[m],场景=SCENE_LABEL[s],n=50,任务成功=f"{group['task_success_n']}/50",干净成功=f"{group['clean_success_n']}/50",柱接触=f"{group['pillar_contact_n']}/50",
                     全回合时长_s=fstat(ms['duration_s']),全外部冲量_Ns=fstat(ms['all_full_impulse_Ns']),前4s全外部冲量_Ns=fstat(ms['all_fixed4_impulse_Ns']),任务高度带全回合冲量_Ns=fstat(ms['task_full_impulse_Ns']),任务高度带前4s冲量_Ns=fstat(ms['task_fixed4_impulse_Ns']),
                     全外部总力峰值_N=fstat(ms['all_full_sum_peak_N']),全外部10ms滑均峰值_N=fstat(ms['all_full_moving10ms_peak_N']),原记录1ms力矩峰值_Nm=fstat(ms['joint_torque_original_1ms_peak_scalar_Nm']),
                     原位耗时回合均值_ms=fstat(ms['safety_mean_ms']),原位全动作合并P50_ms=f"{pc['p50_ms']:.2f}" if pc['n'] else '—',原位全动作合并P95_ms=f"{pc['p95_ms']:.2f}" if pc['n'] else '—',计时动作数=pc['n'],超过50ms动作数=pc['over50_count'] if pc['n'] else '—',载荷来源='原执行1ms记录' if m!='vlsa_native' else '原轨迹严格核验重播')
            main_table.append(pub)
    # All writing occurs after every500+100 input and every12 group passed.
    writej(out/'per_episode.json',episodes);writecsv(out/'per_episode.csv',episodes);writej(out/'groups.json',groups);writecsv(out/'groups.csv',flat)
    writecsv(out/'publication_main_table.csv',main_table);writecsv(out/'publication_loads.csv',loads_table);writecsv(out/'publication_all_metrics_long.csv',long)
    notes='各力学数值为回合指标的均值±样本标准差；不支持显著性结论。all是机器人+工具与外界逐接触三维力模之和，task仅是0.18<z<0.22m的任务高度带；均不等于柱体专属力。full受原终止时刻影响，fixed4是完整前4s（4000样本），无补零。冲量为模的积分，非向量净冲量。原执行500与严格核验重播100按来源分列，重播不增加回合或新独立样本，新增策略推理0。原记录力矩标量、动作末采样力矩、重播1ms力矩三者分开；VLSA动作末力矩缺失留空。原位安全相关计算耗时不含推理/物理积分/渲染，更新频率和计时边界并不完全一致；纯π0安全耗时不适用。P50/P95合并全部有限原执行动作统计，不是各回合P95均值。'
    with (out/'publication_main_table.md').open('x',encoding='utf8') as f:f.write('# 最新600回合完整主表\n\n'+markdown(list(main_table[0]),main_table)+'\n'+notes+'\n')
    with (out/'publication_loads.md').open('x',encoding='utf8') as f:
        f.write('# 全部/成功/失败子集的完整载荷表\n\n'+notes+'\n\n')
        for subset in SUBSETS:
            sub=[r for r in loads_table if r['回合子集']==subset];f.write('## '+subset+'\n\n'+markdown(list(sub[0]),sub)+'\n')
    with (out/'README.md').open('x',encoding='utf8') as f:
        f.write('# GCC-VLA 最新600回合载荷与耗时合并\n\n'+notes+'\n\n')
        f.write('输入：load_audit_original500/per_episode.json；timing_audit/per_episode.csv与groups.json；原latest600索引；replay_exact下100个replay_validation.json和replayed_external_loads.npz。\n\n')
        f.write('载荷统计完全沿用原500定义：dt=.001s，sumcontactnorm冲量/峰值/单接触峰值/均力/完整10点滑窗最大均力/接触记录数>0持续时间；full及fixed4分别计算；all与task分别计算。每组50全部保留，all_episodes/success_only/failure_only分别给n、有效n、mean、sample SD、median、P95、min/max。无成功组成功子集n=0，指标null。\n\n')
        f.write('全100重播准入必须：admitted、初始积分完全一致、endpoints全一致、7类事件零不一致、q_start最大舍入差≤1e-12，其余归档状态/时间差严格0、旧冲量和峰值完全一致、名义twist误差≤1e-12、新推理/QP调用为0。该准入比重播程序原阈值严格；任何失败均停止整600发布，不删回合。\n\n')
        f.write('groups.json是12个对象的列表；subsets下metrics为回合指标统计，runtime_pooled_original_actions为该子集全部原执行动作计时合并统计。metrics.safety_p95_ms是逐回合P95的分布，仅用于补充，不能替代runtime_pooled_original_actions.p95_ms。publication_main_table提供主指标，publication_loads提供三子集四域/窗口完整载荷，publication_all_metrics_long为所有回合指标统计长表。\n')
    writej(out/'sources_sha256.json',list(sources.values()))
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete_episode_count=600,groups=12,episodes_per_group=50,task_success_count=sum(r['task_success'] for r in episodes),failure_count=sum(not r['task_success'] for r in episodes),
                 physical_load_source_episode_counts=dict(Counter(r['load_source_kind'] for r in episodes)),validated_replay_count=100,original_trajectory_replays_are_new_independent_episodes=False,new_policy_inference_calls=0,merge_mujoco_physics_steps=0,
                 validated_prior_replay_physics_substeps=sum_replay_steps,original_files_modified=0,fixed4_complete_episode_count=sum(r['fixed4_complete'] for r in episodes),replay_maximum_errors=dict(validation_errors),replay_event_mismatch_totals=dict(event_totals),maximum_nominal_twist_error=maximum_nominal_error,
                 torque_source_separation='original recorded scalar used across600; action-end sample only original500; replay1ms scalar only replay100, never substituted into original scalar',script_sha256=sha(__file__),outputs=[])
    for name in names:
        p=out/name
        if name!='validation_receipt.json':receipt['outputs'].append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    writej(out/'validation_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
