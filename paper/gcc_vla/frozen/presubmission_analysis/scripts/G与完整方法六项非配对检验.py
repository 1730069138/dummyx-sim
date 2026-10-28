from pathlib import Path
import json,csv,hashlib
import numpy as np
from scipy import stats
O=Path('/home/jun/dummyx-sim/outputs/gcc_vla_round2_20261028_153421');D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400')
idx=json.loads((D/'逐回合汇总.json').read_text());rows=[]
for r in idx:
 if r['method'] not in ['pace_g','pace_gc']:continue
 p=Path(r['package_data'])
 with np.load(p) as z,np.load(p.with_name('paper_truth_ep_001.npz')) as t:
  a=dict(scene=r['scene'],method=r['method'],seed=r['seed'],J_Ns=float(t['all_sum_force_N'].sum()*.001),J4_Ns=float(t['all_sum_force_N'][:4000].sum()*.001),Fpk_N=float(t['all_sum_force_N'].max()),STSR=bool(z['episode_success']) and not(bool(z['episode_obstacle_contact']) or bool(z['episode_tool_floor_contact']) or bool(z['episode_knockdown'])))
  assert len(t['time_s'])>=4000
  rows.append(a)
def holm(rows):
 ordered=sorted(enumerate(rows),key=lambda it:it[1]['p_raw']);last=0.
 for k,(i,r) in enumerate(ordered):last=max(last,min(1.,(len(rows)-k)*r['p_raw']));r['p_holm']=last
cts=[];binary=[]
for scene in ['main','pillar_x_plus40mm']:
 groups={m:sorted([r for r in rows if r['method']==m and r['scene']==scene],key=lambda x:x['seed']) for m in ['pace_g','pace_gc']};assert all(len(v)==50 for v in groups.values())
 for key in ['J_Ns','J4_Ns','Fpk_N']:
  a=np.array([r[key] for r in groups['pace_gc']]);b=np.array([r[key] for r in groups['pace_g']]);test=stats.mannwhitneyu(a,b,alternative='two-sided',method='asymptotic',use_continuity=True)
  cts.append(dict(scene=scene,outcome=key,n_complete=50,n_G=50,test='two-sided Mann-Whitney U; asymptotic tie correction and continuity correction',U=float(test.statistic),p_raw=float(test.pvalue),cliffs_delta_complete_minus_G=float(np.sign(a[:,None]-b[None,:]).mean()),complete_mean=float(a.mean()),complete_sd=float(a.std(ddof=1)),G_mean=float(b.mean()),G_sd=float(b.std(ddof=1)),mean_difference_complete_minus_G=float(a.mean()-b.mean()),holm_family='six continuous contrasts =3outcomes x2scenes'))
 ka=sum(r['STSR'] for r in groups['pace_gc']);kb=sum(r['STSR'] for r in groups['pace_g']);t=stats.fisher_exact([[ka,50-ka],[kb,50-kb]],alternative='two-sided')
 binary.append(dict(scene=scene,outcome='STSR',test='two-sided Fisher exact; independent descriptive fallback',complete_success=ka,G_success=kb,n_each=50,odds_ratio=float(t.statistic),risk_difference_percentage_points=2.*(ka-kb),p_raw=float(t.pvalue),holm_family='two STSR scene contrasts, separate from continuous6'))
holm(cts);holm(binary)
res=dict(selection='Per explicit two-block identity admission requested this round, G/GC second policy blocks differ all100pairs, so nonpaired MWU/Fisher fixed before testing; not selected by p-values. First block/initialstate equality and sameGPU diagnosis retained. Does not mean all methods randomized or independent hardware.',post_hoc=True,interpretation='Existing seed blocks share conditions; unpaired tests are requested fallback, not proof of randomized assignment. Descriptive group effects and current-scene evidence only; no generalization/causal isolation guarantee.',continuous_six=cts,STSR_two=binary,per_episode=rows,software=dict(numpy=np.__version__,scipy=__import__('scipy').__version__),policy_calls=0,physics_steps=0)
(O/'audit/G与完整方法_统计检验.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
for name,data in [('G与完整方法_六项检验.csv',cts),('G与完整方法_STSR检验.csv',binary),('G与完整方法_检验原始回合指标.csv',rows)]:
 with (O/'audit'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print(json.dumps(dict(continuous=cts,STSR=binary),ensure_ascii=False,indent=2))
