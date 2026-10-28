#!/usr/bin/env python3
"""Additional S1 artifact and initial-state check; entirely read-only source access."""
from pathlib import Path
import json,csv,hashlib,collections
import numpy as np
R=Path('/home/jun/dummyx-sim/outputs/gcc_vla_round3_20261028_214716')
E=Path('/home/jun/dummyx-sim/outputs/gcc_vla_experiments_round2_20261028_190400')
S1=Path('/media/jun/JUN/JUN/补充材料S1_新增实验逐回合结果.csv')
analysis=E/'analysis_final/analysis_round2.json';a=json.loads(analysis.read_text());rr=[r for r in a['per_episode'] if r['experiment'] in ['exp2','exp3']]
with S1.open(encoding='utf-8-sig') as f:s1=list(csv.DictReader(f))
assert len(s1)==len(rr)==400
labels={('exp2','fixed'):'方向对照-固定方向(F)',('exp2','normal'):'方向对照-最近点法向',('exp3','F'):'复核消融-F',('exp3','B'):'复核消融-B'}
scenes={'main':'原场景','pillar_x_plus40mm':'X+40 mm'}
ss={(r['实验组'],r['场景'],int(r['种子'])):r for r in s1};assert len(ss)==400
issues=[];checks=[];sources={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source(p):
    p=Path(p)
    if str(p) not in sources:sources[str(p)]={'path':str(p),'sha256':sha(p),'size':p.stat().st_size}
source(S1);source(analysis)
for r in rr:
    row=ss[labels[r['experiment'],r['config']],scenes[r['scene']],r['seed']];p=Path(r['data_path']);t=Path(r['trace_path']);truth=p.with_name('paper_truth_ep_001.npz');source(p);source(t);source(truth)
    with np.load(p,allow_pickle=False) as z:
        sr=bool(z['episode_success']);pc=bool(z['episode_obstacle_contact']);fl=bool(z['episode_tool_floor_contact']);tilt=bool(z['episode_knockdown']);term=str(z['termination_reason'])
    with np.load(truth,allow_pickle=False) as z:
        f=np.array(z['all_sum_force_N'],float);J=float(f.sum()*.001);J4=float(f[:4000].sum()*.001);pk=float(f.max());T=len(f)*.001
    with np.load(t,allow_pickle=False) as z:
        comp=np.asarray(z['step_completed'],bool);limit=np.asarray(z['actuator_at_limit_arm'],bool);nlimit=int(np.any(limit[comp],axis=1).sum())
    seedbase=min(q['seed'] for q in rr if q['experiment']==r['experiment'] and q['scene']==r['scene'])
    exact={'种子':r['seed'],'种子序号':r['seed']-seedbase,'GPU':r['gpu'],'SR':int(sr),'STSR':int(sr and not(pc or fl or tilt)),'PCR':int(pc),'工具触地':int(fl),'柱体倾倒':int(tilt),'终止原因':term,'触限':int(nlimit>0),'触限子步数':nlimit}
    diffs=[]
    for k,v in exact.items():
        if str(v)!=row[k]:diffs.append({'field':k,'source':v,'S1':row[k]})
    for k,v in [('J (N·s)',J),('J4 (N·s)',J4),('FΣ峰值 (N)',pk),('T (s)',T)]:
        text=row[k];decimals=len(text.partition('.')[2]);expected=float(round(v,decimals))
        if expected!=float(text):diffs.append({'field':k,'source':v,'rounded_source':expected,'S1':text})
    checks.append({'id':r['id'],'seed':r['seed'],'config':r['config'],'experiment':r['experiment'],'scene':r['scene'],'S1_matches_original_arrays':not diffs,'differences':diffs})
    if diffs:issues.append(checks[-1])
initialfile=R/'task1/three_batch_failure_and_initial_state.json';source(initialfile);initial=json.loads(initialfile.read_text());cross=[]
for scene in ['main','pillar_x_plus40mm']:
    rows=[r for r in initial['initial_states'] if r['scene']==scene and (r['dataset'] in ['direction_fixed','FB_F'] or (r['dataset']=='original600' and r['method']=='pace_gc'))]
    assert len(rows)==150
    fields={k:len({json.dumps(r[k]) for r in rows}) for k in ['complete_initial_state_sha256','arm_q','tool_free_qpos','pillar_free_qpos']}
    assert all(v==1 for v in fields.values());cross.append({'scene':scene,'n_initial_states':150,'unique_values':fields,'all_three_batches_complete_initial_states_exact_equal':True})
result={'status':'PASS' if not issues else 'FAIL','S1_rows':len(s1),'S1_unique_identities':len(ss),'original_NPZ_truth_trace_checks':checks,'mismatch_n':len(issues),'cross_batch_initial_state_checks':cross,'notes':['The author S1 uses exact group label 复核消融-F (not 最终复核消融-完整F); this audit requires all400 exact identities, with no conditional skip.','Floating S1 numbers are checked at each stored decimal precision against original arrays; source values are retained in the original analysis and can be regenerated.','ctrl is not torque; saturation flags here are actual per-substep actuator_at_limit_arm restricted to completed steps.','These150-per-scene initial states are exactly equal across batches despite disjoint absolute policy seed ranges. This does not assert equal generated policy blocks or trajectories.'],'sources':list(sources.values()),'physics_steps':0,'policy_calls':0}
p=R/'task1/S1逐回合原数组与三批初态交叉核验.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
md=R/'task1/S1逐回合原数组与三批初态交叉核验.md';assert not md.exists();md.write_text('# S1逐回合原数组与三批初态交叉核验\n\n结论：**'+result['status']+'**。\n\n作者实际S1四个实验标签、400个唯一身份全部与对应的原始data NPZ、paper_truth 1ms载荷和trace实际触限标志逐字段核对，差异'+str(len(issues))+'项。计数、终止原因、GPU、绝对seed、0起rep、实际触限回合/子步均精确匹配；J/J₄/峰值/T按S1各自保存精度与原数组重算值一致。未以舍入后的S1反过来估计原统计。\n\n原600完整F、方向固定F、扩大F/B的F在每个场景合计150份完整初始state逐字节哈希相同；臂q、工具位置/四元数、柱位置/四元数亦分别完全相同。因此不仅批内跨seed固定，同一场景三批之间也有相同的实际初态。绝对策略seed不同，不能以rep相同或相同初态推断后续动作/轨迹重复。\n\n本补充核验弥补主脚本中S1标签条件检查的范围：此处严格要求真实标签“复核消融-F”，并没有跳过任何一组或一行。主脚本对两批给定失败/成功碰柱序号的原始NPZ验证本身不受该标签影响。\n\n全部来源路径及SHA256、400行核验状态和两场景交叉初态结果见同名JSON。原Word、CSV和实验文件未修改，无物理推进与策略调用。\n')
print(json.dumps({'status':result['status'],'S1_rows':len(s1),'mismatches':len(issues),'cross':cross,'source_files':len(sources),'audit':str(p)},ensure_ascii=False,indent=2))
