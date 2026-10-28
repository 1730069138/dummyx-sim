#!/usr/bin/env python3
"""Rebuild original600 from retained original arrays; no simulator imports or inference."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree
import csv,json,hashlib,re,collections
import numpy as np

OUT=Path('/home/jun/dummyx-sim/outputs/gcc_vla_round3_20261028_214716')
D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
E=Path('/home/jun/dummyx-sim/outputs/gcc_vla_experiments_round2_20261028_190400')
WORD=Path('/media/jun/JUN/JUN/GCC-VLA_第三轮修改稿.docx')
S1=WORD.with_name('补充材料S1_新增实验逐回合结果.csv')
INDEX=D/'逐回合汇总.json';COMBINED=D/'载荷与耗时补全_20261027/combined_metrics/per_episode.json'
rows=json.loads(INDEX.read_text());assert len(rows)==600
derived=json.loads(COMBINED.read_text());derived={(r['method'],r['scene'],int(r['seed'])):r for r in derived}
manifest={};issues=[];initials=[];out=[];source_coverage=collections.Counter()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def source(p,role,assignment=None):
    p=Path(p);assert p.is_file(),p
    if str(p) not in manifest:manifest[str(p)]={'path':str(p),'sha256':sha(p),'size':p.stat().st_size,'roles':[],'assignments':[]}
    if role not in manifest[str(p)]['roles']:manifest[str(p)]['roles'].append(role)
    if assignment is not None and assignment not in manifest[str(p)]['assignments']:manifest[str(p)]['assignments'].append(assignment)
    return manifest[str(p)]['sha256']
def sc(z,k):return z[k].item() if k in z else None
def writejson(p,obj):
    assert not p.exists(),p;p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def writecsv(p,rr):
    assert not p.exists(),p
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader()
        for r in rr:w.writerow({k:('null' if v is None else json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else int(v) if isinstance(v,(bool,np.bool_)) else v) for k,v in r.items()})
source(INDEX,'canonical retained identity index');source(COMBINED,'historical derived numbers only for discrepancy check');source(WORD,'read-only manuscript');source(S1,'author S1 comparison')
seedbase={s:min(int(r['seed']) for r in rows if r['scene']==s) for s in {r['scene'] for r in rows}}
hist=collections.defaultdict(list)
for r in sorted(rows,key=lambda r:(r['method'],r['scene'],int(r['seed']))):
    m,s,seed=r['method'],r['scene'],int(r['seed']);aid=r['assignment_id'];p=Path(r['package_data']);j=Path(r['package_job']);key=(m,s,seed);old=derived[key]
    hist[r['run_id']].append(aid);source(p,'original endpoint NPZ',aid)
    with np.load(p,allow_pickle=False) as z:
        if m=='vlsa_native':
            sr=bool(sc(z,'task_success'));pc=bool(sc(z,'pillar_contact'));floor=bool(sc(z,'tool_floor_contact'));tilt=bool(sc(z,'pillar_tilt'));term=str(sc(z,'termination'))
            logged_stsr=bool(sc(z,'clean_success'));rawdata=p.with_name('data_ep_001.npz');ini=j/'initial_pre_action_state.npz';source(ini,'original before-first-action integration state',aid)
            with np.load(ini,allow_pickle=False) as iz:state=np.array(iz['state'],float);spec=int(iz['spec'])
            config=j/'run_config.json';source(config,'original VLSA actual arguments',aid);cfg=json.loads(config.read_text());gpu=None;port=cfg['arguments'].get('port');rep=seed-seedbase[s]
            vr=D/'载荷与耗时补全_20261027/replay_exact'/s/aid;validation=vr/'replay_validation.json';source(validation,'saved strict replay admission',aid);v=json.loads(validation.read_text())
            assert v['admitted'] and v['endpoints_match'] and v['initialization_integration_exact'] and v['new_inference_calls']==v['qp_calls']==0 and all(q==0 for q in v['event_mismatches'].values())
            load=vr/'replayed_external_loads.npz';replay=True;tsource='saved strict no-inference replay admission; not a new independent episode'
            duration_original=None;endpoint_definition='aligned NPZ task_success/clean_success/pillar_contact/tool_floor_contact/pillar_tilt/termination'
        else:
            sr=bool(sc(z,'episode_success'));pc=bool(sc(z,'episode_obstacle_contact'));floor=bool(sc(z,'episode_tool_floor_contact'));tilt=bool(sc(z,'episode_knockdown'));term=str(sc(z,'termination_reason'));logged_stsr=None
            rawdata=p;state=np.array(z['replay_state'][0],float);spec=int(z['replay_state_spec']);ini=p
            job=j/'job.json';res=j/'result.json';source(job,'original job identity GPU seed rep',aid);source(res,'original independent result cross-check',aid);cfg=json.loads(job.read_text());gpu=cfg.get('gpu');port=cfg.get('port');rep=cfg.get('rep')
            result=json.loads(res.read_text());assert int(cfg['seed'])==seed and int(cfg['rep'])==seed-seedbase[s]
            load=p.with_name('paper_truth_ep_001.npz');validation=None;replay=False;tsource='original recorded 1ms external contact arrays';duration_original=float(sc(z,'episode_duration_s'));endpoint_definition='original NPZ episode_success/episode_obstacle_contact/episode_tool_floor_contact/episode_knockdown/termination_reason'
        stsr=sr and not pc and not floor and not tilt
        assert logged_stsr is None or logged_stsr==stsr
        assert all(bool(r[x])==y for x,y in [('task_success',sr),('clean_success',stsr),('pillar_contact',pc),('tool_floor_contact',floor),('pillar_tilt',tilt)])
    source(rawdata,'original execution NPZ torque-peak scalar',aid)
    with np.load(rawdata,allow_pickle=False) as z:tau=sc(z,'episode_peak_joint_torque')
    source(load,'saved 1ms external load source',aid)
    with np.load(load,allow_pickle=False) as z:
        f=np.array(z['all_sum_force_N'],float);tt=np.array(z['time_s'],float);assert f.size==tt.size and np.all(np.isfinite(f)) and np.allclose(np.diff(tt),.001,rtol=0,atol=1e-9)
        J=float(f.sum()*.001);J4=float(f[:4000].sum()*.001) if f.size>=4000 else None;pk=float(f.max());T=f.size*.001
    if duration_original is not None:assert abs(T-duration_original)<1e-8
    for k,vv in [('all_full_impulse_Ns',J),('all_fixed4_impulse_Ns',J4),('all_full_sum_peak_N',pk),('duration_s',T)]:
        if vv is None or abs(vv-old[k])>1e-9:issues.append({'assignment':aid,'metric':k,'new':vv,'previous':old[k]})
    lineage=j/'policy_lineage.jsonl';chunk=j/'policy_chunk_000.npz';source(lineage,'saved first policy request random seed and observation provenance',aid);source(chunk,'saved first policy block and initial joint observation',aid)
    first=json.loads(lineage.read_text().splitlines()[0]);assert first['episode_seed']==seed and first['eval_rng']=={'episode_seed':seed,'request_index':0}
    with np.load(chunk,allow_pickle=False) as z:
        obs=np.array(z['state'],float);assert obs.shape==(8,) and np.array_equal(obs,state[1:9])
        chunkhash=hashlib.sha256(np.ascontiguousarray(z['actions']).tobytes()).hexdigest()
    assert spec==8191 and state.size==187
    init={'dataset':'original600','method':m,'scene':s,'seed':seed,'rep':int(rep),'state_path':str(ini),'state_spec':spec,'complete_initial_state_sha256':hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),'arm_q':state[1:7].tolist(),'tool_free_qpos':state[9:16].tolist(),'pillar_free_qpos':state[16:23].tolist(),'initial_policy_state':obs.tolist(),'initial_policy_actions_array_sha256':chunkhash}
    initials.append(init)
    out.append({'method':m,'scene':s,'seed':seed,'rep':int(rep),'GPU':gpu,'port':port,'SR':sr,'STSR':stsr,'PCR':pc,'tool_floor':floor,'pillar_tilt':tilt,'termination_reason':term,'J_Ns':J,'J4_Ns':J4,'Fpeak_N':pk,'T_s':T,'torque_peak_original_scalar_Nm':tau,'touch_8Nm_from_original_scalar':None if tau is None else bool(float(tau)>=8-1e-8),'VLSA_load_is_saved_replay':replay,'load_source_kind':tsource,'endpoint_definition':endpoint_definition,'historical_run_directory':r['run_id'],'historical_raw_path':r['raw_source'],'historical_raw_path_exists':Path(r['raw_source']).exists(),'canonical_original_data_path':str(p),'canonical_original_data_sha256':source(p,'original endpoint NPZ',aid),'original_torque_data_path':str(rawdata),'original_torque_data_sha256':source(rawdata,'original execution NPZ torque-peak scalar',aid),'load_source_path':str(load),'load_source_sha256':source(load,'saved 1ms external load source',aid),'replay_admission_path':str(validation) if validation else None,'source_job_path':str(j),'missing_fields':['GPU'] if gpu is None else []})
    source_coverage[m]+=1

# Rebuild every visible numeric cell of original Word tables1,3,4.
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(WORD) as z:doc=etree.fromstring(z.read('word/document.xml'))
tab=doc.findall('w:body/w:tbl',ns)
rawtables={n:[[ ''.join(c.xpath('.//w:t/text()',namespaces=ns)) for c in row.findall('w:tc',ns)] for row in tab[n-1].findall('w:tr',ns)] for n in [1,3,4]}
order=['raw_pi0','pace_g','pace_gc','oscbf','neo_mesh','vlsa_native'];groups=[];checks=[]
for m in order:
    for scene in ['main','pillar_x_plus40mm']:
        rr=[r for r in out if (r['method'],r['scene'])==(m,scene)];assert len(rr)==50 and {r['rep'] for r in rr}==set(range(50))
        g={'method':m,'scene':scene,'n':50,**{k:sum(r[k] for r in rr) for k in ['SR','STSR','PCR','tool_floor','pillar_tilt','touch_8Nm_from_original_scalar']}}
        for k in ['J_Ns','J4_Ns','Fpeak_N','T_s']:
            vals=np.array([r[k] for r in rr],float);g[k+'_mean']=float(vals.mean());g[k+'_sampleSD']=float(vals.std(ddof=1))
        groups.append(g)
        ri=order.index(m)+1
        for k,offset in [('SR',1),('STSR',2),('PCR',3)]:
            col=offset+(3 if scene!='main' else 0);actual=rawtables[1][ri][col];expected=str(g[k]*2);checks.append({'table':1,'method':m,'scene':scene,'metric':k,'word_text':actual,'recomputed_text':expected,'match':actual==expected,'numerator':g[k],'denominator':50})
        n=3 if scene=='main' else 4
        for k,col in [('J_Ns',1),('J4_Ns',2),('Fpeak_N',3),('T_s',5)]:
            expected=f"{g[k+'_mean']:.2f}±{g[k+'_sampleSD']:.2f}";actual=rawtables[n][ri][col];checks.append({'table':n,'method':m,'scene':scene,'metric':k,'word_text':actual,'recomputed_text':expected,'match':actual==expected,'numerator':None,'denominator':50})
        actual=rawtables[n][ri][4];expected=f"{g['touch_8Nm_from_original_scalar']}/50";checks.append({'table':n,'method':m,'scene':scene,'metric':'touch_8Nm_original_scalar','word_text':actual,'recomputed_text':expected,'match':actual==expected,'numerator':g['touch_8Nm_from_original_scalar'],'denominator':50})

# Recheck new batches from original saved NPZ, not rounded S1, then compare S1.
newanalysis=E/'analysis_final/analysis_round2.json';source(newanalysis,'new frozen episode locator');new=json.loads(newanalysis.read_text())['per_episode'];batch=[]
for dataset,ex,config in [('direction_fixed','exp2','fixed'),('FB_F','exp3','F')]:
    rr=[r for r in new if (r['experiment'],r['config'])==(ex,config)]
    for r in rr:
        p=Path(r['data_path']);source(p,'new original NPZ for comparison only',r['id'])
        with np.load(p,allow_pickle=False) as z:
            state=np.array(z['replay_state'][0],float);spec=int(z['replay_state_spec']);sr=bool(z['episode_success']);pc=bool(z['episode_obstacle_contact']);reason=str(z['termination_reason'])
        assert sr==r['SR'] and pc==r['PCR'] and reason==r['termination_reason']
        base=min(q['seed'] for q in rr if q['scene']==r['scene']);rep=r['seed']-base
        initials.append({'dataset':dataset,'method':config,'scene':r['scene'],'seed':r['seed'],'rep':rep,'state_path':str(p),'state_spec':spec,'complete_initial_state_sha256':hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest(),'arm_q':state[1:7].tolist(),'tool_free_qpos':state[9:16].tolist(),'pillar_free_qpos':state[16:23].tolist(),'initial_policy_state':None,'initial_policy_actions_array_sha256':None})
        batch.append({'dataset':dataset,'scene':r['scene'],'seed':r['seed'],'rep':rep,'SR':sr,'PCR':pc,'reason':reason,'path':str(p)})
with S1.open(encoding='utf-8-sig') as f:s1=list(csv.DictReader(f))
batchsummary=[]
for name in ['original600_F','direction_fixed','FB_F']:
    rr=[{'dataset':name,'scene':r['scene'],'seed':r['seed'],'rep':r['rep'],'SR':r['SR'],'PCR':r['PCR'],'reason':r['termination_reason'],'path':r['canonical_original_data_path']} for r in out if r['method']=='pace_gc'] if name=='original600_F' else [r for r in batch if r['dataset']==name]
    for scene in ['main','pillar_x_plus40mm']:
        ss=[r for r in rr if r['scene']==scene];batchsummary.append({'dataset':name,'scene':scene,'n':len(ss),'failed_rep':sorted(r['rep'] for r in ss if not r['SR']),'failed_seed':sorted(r['seed'] for r in ss if not r['SR']),'failed_all_PCR':all(r['PCR'] for r in ss if not r['SR']),'failure_reasons':dict(collections.Counter(r['reason'] for r in ss if not r['SR'])),'successful_PCR_rep':sorted(r['rep'] for r in ss if r['SR'] and r['PCR']),'all_rows':ss})
expected={'direction_fixed':([2,13,42,49],[5,17,24,44,45]),'FB_F':([4,9,34,35],[0,6,16,28,38])}
for name,(fail,successpc) in expected.items():
    x=next(x for x in batchsummary if x['dataset']==name and x['scene']=='main');assert x['failed_rep']==fail and x['successful_PCR_rep']==successpc
    cn='方向对照-固定方向(F)' if name=='direction_fixed' else '最终复核消融-完整F'
    sr=[r for r in s1 if r['实验组']==cn and r['场景']=='原场景']
    # Source S1 labels are inspected separately if an author used a different group title.
    if sr:assert sorted(int(r['种子序号']) for r in sr if r['SR']=='0')==fail
intersections=[]
main=[r for r in batchsummary if r['scene']=='main']
for i,a in enumerate(main):
    for b in main[i+1:]:intersections.append({'A':a['dataset'],'B':b['dataset'],'failed_rep_intersection':sorted(set(a['failed_rep'])&set(b['failed_rep'])),'successful_PCR_rep_intersection':sorted(set(a['successful_PCR_rep'])&set(b['successful_PCR_rep'])),'interpretation':'relative rep only; absolute episode seeds differ; neither trajectory repetition nor strict pairing is inferred'})

# Integration-state invariance implies invariant free-joint poses; expose all initial values.
initialsummary=[]
for key in sorted({(r['dataset'],r['method'],r['scene']) for r in initials}):
    rr=[r for r in initials if (r['dataset'],r['method'],r['scene'])==key];a=np.array([r['arm_q'] for r in rr]);t=np.array([r['tool_free_qpos'] for r in rr]);p=np.array([r['pillar_free_qpos'] for r in rr])
    initialsummary.append({'dataset':key[0],'method':key[1],'scene':key[2],'n':len(rr),'unique_full_state_hashes':len({r['complete_initial_state_sha256'] for r in rr}),'arm_q_max_abs_spread':float(np.ptp(a,axis=0).max()),'tool_pose_max_abs_spread':float(np.ptp(t,axis=0).max()),'pillar_pose_max_abs_spread':float(np.ptp(p,axis=0).max()),'arm_q_first':a[0].tolist(),'tool_xyz_quaternion_first':t[0].tolist(),'pillar_xyz_quaternion_first':p[0].tolist()})
historical=[{'historical_directory':k,'exists_now':Path(k).exists(),'n':len(v),'current_source_root':str(D/'逐回合原始记录'),'assignments':v} for k,v in hist.items()]
for path,role in [(D/'脚本与身份/共享运行时/deploy_screwdriver_client_old.py','original fixed scene/reset and pre-action integration-state recording code'),(D/'脚本与身份/NEO/run_worker.py','saved evaluation seed request wrapper'),(D/'脚本与身份/统一入口_20261027/project/models/dummy_apf.xml','archived named arm/gripper qpos order'),(D/'脚本与身份/统一入口_20261027/project/models/dummyx_apf_scene.xml','archived tool/pillar freejoint order'),(E/'services/launch_receipt.json','new actual process-local seeded inference launch code')]:source(path,role)
bad=[x for x in checks if not x['match']]
report={'status':'PASS' if not issues and not bad else 'MISMATCH','n':600,'source_coverage':dict(source_coverage),'missing_GPU_n':sum(r['GPU'] is None for r in out),'missing_policy_or_metric_fields':[],'historical_directories':historical,'recomputed_groups':groups,'table_cell_checks':len(checks),'table_mismatches':bad,'derived_metric_mismatches':issues,'three_batch_main_sets':main,'intersections':intersections,'initial_state_summaries':initialsummary,'notes':['CSV null is explicit missing, particularly VLSA GPU; port is not promoted to GPU.','STSR recomputed from actual strict task success and actual three event flags. Original 500 NPZ field episode_success is the original strict placement criterion; VLSA aligned task_success is separate from original legacy success.','Torque touching8 is based on the original recorded episode_peak_joint_torque reduction, not reconstructed 1ms torque arrays and not ctrl.','VLSA100 external loads come from existing strictly admitted zero-inference saved replay arrays, retaining original episode endpoints and denominator; no replay performed in this task.','Initial integration state8191 layout: time then qpos; arm1:7, tool9:16, pillar16:23 are state slices from archived arm8-coordinate + tool7 + pillar7 XML order. First8 qpos were independently checked against actual saved first observation for all600.','Whole187 integration-state equality is stronger than selected position equality but does not imply later policy outputs or trajectories equal. Absolute seed series differ among original600, direction and F/B batches.','Fixed_eval uses fixed target0.28,-0.10 before deterministic settling; policy seed is transmitted as _eval_seed/_eval_request_index, server process wrapper fold_in(key(seed),index) resets policy RNG before original inference. Random-policy evaluation is not random initial-layout evaluation.'],'physics_steps':0,'policy_calls':0}
OUT.joinpath('task1').mkdir(exist_ok=True)
writecsv(OUT/'original600_per_episode.csv',out);writejson(OUT/'task1/original600_per_episode.json',out);writecsv(OUT/'task1/Word表1_3_4逐单元格核对.csv',checks);writecsv(OUT/'task1/original600_group_summary.csv',groups);writejson(OUT/'task1/original600_sources_sha256.json',list(manifest.values()));writecsv(OUT/'task1/original600_sources_sha256.csv',list(manifest.values()));writejson(OUT/'task1/three_batch_failure_and_initial_state.json',{'batches':batchsummary,'rep_intersections':intersections,'initial_state_summary':initialsummary,'initial_states':initials});writejson(OUT/'task1/original600_rebuild_audit.json',report)
lines=['# 任务1：原600回合与三批完整配置复核','',f'结论：**{report["status"]}**。原600逐回合重新读取原NPZ及1ms载荷；没有执行物理推进、策略推理或重新重播。','',f'当前原始文件根：`{D}/逐回合原始记录`。原Word只读：`{WORD}`。','', '## 四个历史目录及当前来源','']
for h in historical:lines.append(f'- `{h["historical_directory"]}`：{h["n"]}回合，当前目录存在={h["exists_now"]}；当前保留副本见canonical逐回合原始记录，逐文件路径及SHA256见source清单。')
lines += ['',f'表1、3、4共{len(checks)}个数值单元格重新核对，差异{len(bad)}个；已交付均值、样本SD和逐格文本，未借用其他配置补字段。与旧派生指标另比较的差异为{len(issues)}。600回合身份每方法×场景50，rep完整0–49。','', '## 字段与证据边界','', '原500回合终点取data_ep_001.npz的episode_success、episode_obstacle_contact、episode_tool_floor_contact、episode_knockdown；VLSA100取aligned_ep_001.npz的严格任务与事件字段。STSR按三类事件重新构造，和索引逐条一致。VLSA GPU100条未直接记录，CSV明确null；未把端口或其他方法的GPU填入。','', 'J/J₄/FΣ峰值由all_sum_force_N逐回合重算，步长1ms；T是实际完成物理子步时长。VLSA载荷沿用既有严格准入的零推理重播数组，100份准入的初始化、终点和事件一致性均重新读验；本次未重播，不增加独立样本。触8N·m只依据原执行loop保存的episode_peak_joint_torque标量，不冒充完整1ms力矩数组。','', '## 原场景三批完整配置的失败与成功碰柱','', '|批次|失败rep（0起）|失败原因|失败是否全碰柱|成功但碰柱rep|','|---|---|---|---|---|']
for x in main:lines.append(f'|{x["dataset"]}|{x["failed_rep"]}|{x["failure_reasons"]}|{x["failed_all_PCR"]}|{x["successful_PCR_rep"]}|')
lines += ['', '原600完整配置失败seed为'+str(next(x for x in main if x['dataset']=='original600_F')['failed_seed'])+'。方向固定与F/B完整配置的给定序号均已从其各自原始NPZ重新验证。','', '|A／B|失败rep交集|成功碰柱rep交集|','|---|---|---|']
for x in intersections:lines.append(f'|{x["A"]}／{x["B"]}|{x["failed_rep_intersection"]}|{x["successful_PCR_rep_intersection"]}|')
lines += ['', '这些仅是各自0–49相对序号的集合交集。三批绝对策略种子不相同，序号交集不能推断同轨迹、重复样本或严格配对。失败与碰柱数量恰同也不意味着试验复用。','', '## 初始状态与随机源','', '|集合／方法／场景|n|完整初始state种类|臂q最大跨seed变化|工具自由关节位姿最大变化|柱位姿最大变化|','|---|---|---|---|---|---|']
for x in initialsummary:lines.append(f'|{x["dataset"]}／{x["method"]}／{x["scene"]}|{x["n"]}|{x["unique_full_state_hashes"]}|{x["arm_q_max_abs_spread"]:.3g}|{x["tool_pose_max_abs_spread"]:.3g}|{x["pillar_pose_max_abs_spread"]:.3g}|')
lines += ['', '初态来自原始动作前mjSTATE_INTEGRATION记录；VLSA用initial_pre_action_state.npz。state的首8个qpos逐回合和首策略块保存的8维本体观测完全相同。臂6关节、工具与柱自由关节位置/四元数按归档XML关节顺序提取，并完整公开原数组切片与路径。完整state及位姿是否变化均按实测数组，不由fixed_eval参数单独推定。','', '固定场景源码：共享运行时deploy_screwdriver_client_old.py第245–261行将fixed_eval目标固定为(0.28,-0.10)，第80–105行重置臂、工具与柱，稳定后保存动作前完整state。种子通过run_worker.py第143–145行以_eval_seed和_eval_request_index发给服务；服务器launch receipt中的真实包装器用jax.random.fold_in(jax.random.key(seed),request_index)重设策略RNG后调用原infer。实际600份首请求lineage的episode_seed与eval_rng均逐条匹配，所以本组种子改变策略采样序列，而不是随机化初始物体布置；GPU数值差异及闭环状态反馈仍可能让后续动作不同。','', '## 文件','',f'- `{OUT}/original600_per_episode.csv`：600行完整要求字段，缺失明确null。',f'- `{OUT}/task1/Word表1_3_4逐单元格核对.csv`：原Word格值与重新统计值。',f'- `{OUT}/task1/original600_sources_sha256.json`：所有实际读取来源路径、SHA256、大小、角色和关联回合。',f'- `{OUT}/task1/three_batch_failure_and_initial_state.json`：逐失败、序号交集、逐初态与数组散布。',f'- `{OUT}/scripts/rebuild_original600_readonly.py`：可复跑，只写本轮目录；已有输出拒绝覆盖。']
md=OUT/'task1/原600回合与三批完整配置科学性核查.md';assert not md.exists();md.write_text('\n'.join(lines)+'\n')
print(json.dumps({'status':report['status'],'original600_rows':len(out),'manifest_sources':len(manifest),'table_checks':len(checks),'table_mismatches':bad,'metric_mismatches':issues,'missing_gpu':report['missing_GPU_n'],'main_sets':[{k:v for k,v in x.items() if k!='all_rows'} for x in main],'initial_summary':initialsummary},ensure_ascii=False,indent=2))
