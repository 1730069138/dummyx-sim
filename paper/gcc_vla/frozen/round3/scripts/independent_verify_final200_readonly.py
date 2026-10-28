#!/usr/bin/env python3
"""Run only AFTER all200 formal allocations finish. No primary-analyzer imports."""
from pathlib import Path
import argparse,json,hashlib,collections,math
import numpy as np
from scipy import stats

R=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,d):
    assert not p.exists(),p;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def holm(items):
    previous=0
    for k,i in enumerate(sorted(range(len(items)),key=lambda i:items[i]['p_raw'])):
        previous=max(previous,min(1.,(len(items)-k)*items[i]['p_raw']));items[i]['p_holm']=previous
def summarize(v):
    a=np.array([x for x in v if x is not None],float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'sd':float(a.std(ddof=1)) if len(a)>1 else None}
def wilson(k,n):
    z=1.959963984540054;den=1+z*z/n;c=(k/n+z*z/2/n)/den;h=z/den*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n));return [100*(c-h),100*(c+h)]
def close(a,b,tol=1e-12):
    return a is None and b is None or a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=tol,abs_tol=1e-25)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--primary-analysis',required=True,type=Path);p.add_argument('--attempt-dir',action='append',default=[],type=Path);p.add_argument('--output-prefix',default='independent_final200_verification');args=p.parse_args()
    assert '/' not in args.output_prefix
    planpath=R/'statistics_plan.json';plan=json.loads(planpath.read_text());assert sha(planpath)==(R/'statistics_plan.sha256').read_text().strip();planned={x['id']:x for x in plan['allocations']};assert len(planned)==200
    primary=json.loads(args.primary_analysis.read_text());assert primary['completed_formal']==200 and primary['complete_cohort'] and not primary['validation_issues'],'Wait for complete final200 before independent effect analysis'
    sources={};issues=[];candidates=collections.defaultdict(list);attempts=[];first={}
    def source(p):
        p=Path(p).resolve();assert p.is_relative_to(R),p
        if str(p) not in sources:sources[str(p)]=sha(p)
    source(planpath);source(args.primary_analysis)
    journalpath=R/'audit/technical_interruptions.json';journal=json.loads(journalpath.read_text()) if journalpath.exists() else []
    if journalpath.exists():source(journalpath)
    assert isinstance(journal,list)
    for directory in [R/'experiments/formal']+[x.resolve() for x in args.attempt_dir]:
        assert directory.is_relative_to(R)
        proto=directory/'protocol.json';manifest=directory/'allocations.json';source(proto);source(manifest);assert sha(proto)==(directory/'protocol.sha256').read_text().strip()
        protocol=json.loads(proto.read_text());assert protocol['statistics_plan_sha256']==sha(planpath)
        for job in json.loads(manifest.read_text()):
            identity=job['id'];e=planned[identity];assert all(job[k]==e[k] for k in ['id','seed','gpu','rep','config_name','port']) and job['scene_id']==e['scene']
            folder=directory/f"worker_gpu{job['gpu']}"/identity;rp=folder/'result.json';attempt={'id':identity,'attempt_dir':str(folder),'status':None};attempts.append(attempt)
            if not rp.exists():
                explained=any(x['id']==identity and Path(x['original_path']).resolve()==folder.resolve() for x in journal);attempt['status']='technical_from_explicit_journal' if explained else 'missing_unexplained'
                if not explained:issues.append({'id':identity,'issue':'unexplained_missing_attempt'})
                continue
            source(rp);result=json.loads(rp.read_text());reason=str((result.get('result') or {}).get('reason',(result.get('metrics') or {}).get('reason','')));m=result.get('metrics') or {}
            technical=result.get('error') is not None or m.get('metric_error') or m.get('recording_error') or reason.startswith(('perception_failed:','perception_error:','policy_error:','simulation_error:','user_interrupt'))
            if technical:attempt.update(status='technical_not_task_failure',reason=reason,error=result.get('error'));continue
            try:
                for k in ['id','seed','gpu','rep','config_name']:assert result[k]==job[k]
                assert result['scene_id']==e['scene'] and result['stage']=='formal' and result['experiment']=='jump_expansion'
                paths=list(folder.rglob('data_ep_001.npz'));assert len(paths)==1,'Missing/multiple original NPZ; metrics fallback forbidden';dp=paths[0];tp=Path(result['trace_path']).resolve();assert tp.is_relative_to(folder.resolve()) and tp.exists();truth=dp.with_name('paper_truth_ep_001.npz');assert truth.exists()
                for q in [dp,tp,truth,folder/'job.json',folder/'policy_lineage.jsonl',folder/'policy_chunk_000.npz']:source(q)
                with np.load(dp,allow_pickle=False) as z:
                    sr,pc,fl,tilt=[bool(z[k]) for k in ['episode_success','episode_obstacle_contact','episode_tool_floor_contact','episode_knockdown']];term=str(z['termination_reason']);original_taupk=float(z['episode_peak_joint_torque']);duration=float(z['episode_duration_s'])
                assert term==reason
                for k,v in [('success',sr),('pillar_contact',pc),('floor_contact',fl),('pillar_tilt',tilt)]:assert k in m and bool(m[k])==v,('NPZ versus result',k)
                with np.load(truth,allow_pickle=False) as z:f=np.array(z['all_sum_force_N'],float);times=np.array(z['time_s'],float)
                with np.load(tp,allow_pickle=False) as z:
                    done=np.array(z['step_completed'],bool);assert done.all() and done.any();meta=json.loads(str(z['metadata_json']));N=len(done);time=np.array(z['time_after_s'],float);before=np.array(z['time_before_s'],float);force=np.array(z['sum_external_contact_force_N'],float);act=np.array(z['actuator_force_arm'],float);ranges=np.array(meta['actuator_forcerange'])[meta['arm_actuator_indices']];assert np.array_equal(ranges,np.tile([-8.,8.],(6,1)))
                    assert len(f)==len(times)==N and np.isfinite(f).all() and np.isfinite(act).all()
                    timeerr=float(abs(time-times).max());ferr=float(abs(force-f).max());assert timeerr<1e-12 and np.allclose(force,f,rtol=1e-11,atol=1e-9)
                    assert np.allclose(time-before,.001,rtol=0,atol=1e-10) and np.allclose(np.diff(time),.001,rtol=0,atol=1e-10)
                    limited=(act<=ranges[None,:,0]+1e-8)|(act>=ranges[None,:,1]-1e-8);assert np.array_equal(limited,np.asarray(z['actuator_at_limit_arm'],bool));anylim=limited.any(axis=1)
                    assert abs(abs(act).max()-original_taupk)<1e-10 and abs(N*.001-duration)<1e-8
                    prev=np.array(z['previous_applied_valid'],bool);assert not prev[0] and prev[1:].all()
                    if job['config_name']=='F_limited':assert np.array_equal(np.asarray(meta['target_jump_limit_rad']),plan['jump_limiter']['threshold_rad']) and np.all(abs(z['delta_u_from_previous'][prev])<=np.array(plan['jump_limiter']['threshold_rad'])+1e-12)
                    else:assert meta.get('target_jump_limit_rad') is None and np.all(z['limiter_delta']==0)
                    jm=meta['job'];assert all(jm[k]==job[k] for k in ['id','seed','gpu','config_name','scene_id'])
                with np.load(folder/'policy_chunk_000.npz',allow_pickle=False) as z:polstate=z['state'].copy();actions=z['actions'].copy()
                lineage=json.loads((folder/'policy_lineage.jsonl').read_text().splitlines()[0]);assert lineage['episode_seed']==e['seed'] and lineage['gpu']==e['gpu'] and lineage['eval_rng']=={'episode_seed':e['seed'],'request_index':0}
                first[identity]=(polstate,actions,lineage['observation_hash'],lineage['gpu'])
                row={'id':identity,'scene':e['scene'],'config':e['config_name'],'seed':e['seed'],'gpu':e['gpu'],'SR':sr,'STSR':sr and not(pc or fl or tilt),'PCR':pc,'tool_floor':fl,'pillar_tilt':tilt,'termination_reason':term,'J':float(f.sum()*.001),'J4':float(f[:4000].sum()*.001) if N>=4000 else None,'Fpk':float(f.max()),'T':N*.001,'tau_limit_episode':bool(anylim.any()),'tau_limit_substeps':int(anylim.sum()),'data_path':str(dp),'trace_path':str(tp),'truth_time_max_abs_difference_s':timeerr,'truth_force_max_abs_difference_N':ferr,'endpoint_source':'mandatory_original_NPZ_no_fallback'}
                candidates[identity].append(row);attempt['status']='valid_original_NPZ_completion'
            except Exception as ex:attempt.update(status='validation_unknown_not_task_failure',error=repr(ex));issues.append({'id':identity,'issue':'independent_array_validation','error':repr(ex)})
    rows=[]
    for identity in planned:
        if len(candidates[identity])!=1:issues.append({'id':identity,'issue':'missing_or_duplicate_valid_completions','n':len(candidates[identity])})
        else:rows.append(candidates[identity][0])
    assert len(rows)==200 and not issues,issues
    firstpairs=[]
    for scene in ['main','pillar_x_plus40mm']:
        for seed in sorted({r['seed'] for r in rows if r['scene']==scene}):
            ids=[next(r['id'] for r in rows if (r['scene'],r['seed'],r['config'])==(scene,seed,c)) for c in ['F','F_limited']];a,b=[first[x] for x in ids]
            firstpairs.append({'scene':scene,'seed':seed,'same_GPU':a[3]==b[3],'first_state_equal':np.array_equal(a[0],b[0]),'first_observation_hash_equal':a[2]==b[2],'first_whole_action_block_equal':np.array_equal(a[1],b[1]),'first_block_max_abs_difference':float(abs(a[1]-b[1]).max())})
    groups=[]
    for scene in ['main','pillar_x_plus40mm']:
        for config in ['F','F_limited']:
            rr=[r for r in rows if (r['scene'],r['config'])==(scene,config)];assert len(rr)==50;g={'scene':scene,'config':config,'n':50}
            for k in ['SR','STSR','PCR','tau_limit_episode','pillar_tilt']:g[k]={'count':sum(r[k] for r in rr),'n':50};g[k]['wilson95_percent']=wilson(g[k]['count'],50)
            for k in ['J','J4','Fpk','T','tau_limit_substeps']:g[k]=summarize([r[k] for r in rr])
            groups.append(g)
    tests=[]
    for family,metrics in [('binary6',['SR','STSR','PCR']),('load4',['J','J4'])]:
        one=[]
        for scene in ['main','pillar_x_plus40mm']:
            aa=[r for r in rows if (r['scene'],r['config'])==(scene,'F_limited')];bb=[r for r in rows if (r['scene'],r['config'])==(scene,'F')]
            for metric in metrics:
                t={'family':family,'scene':scene,'outcome':metric}
                if family=='binary6':
                    ka,kb=sum(r[metric] for r in aa),sum(r[metric] for r in bb);q=stats.fisher_exact([[ka,50-ka],[kb,50-kb]],alternative='two-sided');t.update(p_raw=float(q.pvalue),count_limited=ka,count_F=kb,risk_difference_pp=(ka-kb)*2)
                else:
                    a=[r[metric] for r in aa if r[metric] is not None];b=[r[metric] for r in bb if r[metric] is not None];q=stats.mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True);t.update(p_raw=float(q.pvalue),U_limited=float(q.statistic),cliffs_delta_limited_minus_F=2*float(q.statistic)/(len(a)*len(b))-1,n_limited=len(a),n_F=len(b))
                one.append(t)
        holm(one);tests+=one
    discrepancies=[];primaryrows={r['id']:r for r in primary['per_episode']}
    for r in rows:
        q=primaryrows[r['id']]
        for k in ['SR','STSR','PCR','tool_floor','pillar_tilt','termination_reason','tau_limit_episode','tau_limit_substeps']:
            if r[k]!=q[k]:discrepancies.append({'id':r['id'],'field':k,'independent':r[k],'primary':q[k]})
        for k in ['J','J4','Fpk','T']:
            if not close(r[k],q[k]):discrepancies.append({'id':r['id'],'field':k,'independent':r[k],'primary':q[k]})
    for t in tests:
        q=next(q for q in primary['statistical_tests'] if all(t[k]==q[k] for k in ['family','scene','outcome']))
        for k in t:
            if k in ['family','scene','outcome']:continue
            if not close(t[k],q[k]):discrepancies.append({'test':[t['family'],t['scene'],t['outcome']],'field':k,'independent':t[k],'primary':q[k]})
    result={'status':'PASS' if not discrepancies else 'MISMATCH','n':200,'no_original_NPZ_fallback':True,'max_truth_timestamp_error_s':max(r['truth_time_max_abs_difference_s'] for r in rows),'max_truth_force_difference_N':max(r['truth_force_max_abs_difference_N'] for r in rows),'groups':groups,'tests':tests,'first_request_pair_summary':{k:sum(r[k] for r in firstpairs) for k in ['same_GPU','first_state_equal','first_observation_hash_equal','first_whole_action_block_equal']},'first_request_pairs':firstpairs,'strict_later_pairing_claim':False,'discrepancies':discrepancies,'validation_issues':issues,'per_episode':rows,'attempts':attempts,'sources':[{'path':k,'sha256':v} for k,v in sources.items()],'physics_steps':0,'policy_calls':0}
    dest=R/'task3'/args.output_prefix;write_new(dest.with_suffix('.json'),result);md=dest.with_suffix('.md');assert not md.exists();md.write_text('# 最终200回合独立原数组核验\n\n结论：**'+result['status']+'**。200个冻结身份全部使用原始NPZ，无metrics回退；真实原paper_truth时间戳逐子步与trace最大差'+str(result['max_truth_timestamp_error_s'])+' s。全部终点、J/J₄/峰值/T、真实力矩触限次数及6项Fisher/4项MWU独立重算，与主分析差异'+str(len(discrepancies))+'项。\n\n100个同场景同seed首请求对观察到：'+str(result['first_request_pair_summary'])+'。逐对state/首帧摘要/整个首动作块及GPU结果均在JSON内，仅说明首请求实际观察，不声称后续闭环严格配对。\n\n所有源路径及SHA、有效与技术尝试、逐回合数值和检验结果见同名JSON。技术或验证未知尝试不计任务失败；同身份有效完成必须唯一。该独立脚本未导入主分析器，无物理推进或策略调用。\n');print(json.dumps({'status':result['status'],'n':200,'differences':len(discrepancies),'first_pairs':result['first_request_pair_summary'],'path':str(dest.with_suffix('.json'))},ensure_ascii=False))

if __name__=='__main__':main()
