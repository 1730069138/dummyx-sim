from pathlib import Path
import csv,json,collections,hashlib
import numpy as np
D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400');O=Path('/home/jun/dummyx-sim/outputs/gcc_vla_round2_20261028_153421')
rows=list(csv.DictReader((O/'audit/原600种子与设备分配.csv').open()));assert len(rows)==600
idx=json.loads((D/'逐回合汇总.json').read_text()); ii={(r['method'],r['scene'],int(r['seed'])):r for r in idx};dev={(r['method'],r['scene'],int(r['seed'])):r for r in rows};counts=collections.Counter();blocks=[];devblocks=[]
for scene in ['main','pillar_x_plus40mm']:
 seeds=sorted(seed for m,s,seed in ii if m=='pace_g' and s==scene)
 for seed in seeds:
  dd=[r for r in rows if r['scene']==scene and int(r['seed'])==seed];assert len(dd)==6
  gpus={r['gpu'] for r in dd if r['gpu'] != ''};counts[scene,str(len(gpus))]+=1;devblocks.append(dict(scene=scene,seed=seed,known_five_methods_distinct_gpu_n=len(gpus),six_method_total_gpu_n=None,unknown_gpu_methods=[r['method'] for r in dd if r['gpu']==''],gpu_by_method={r['method']:r['gpu'] for r in dd}))
  a=ii['pace_g',scene,seed];b=ii['pace_gc',scene,seed];aj=Path(a['package_job']);bj=Path(b['package_job']);da=dev['pace_g',scene,seed];db=dev['pace_gc',scene,seed]
  z=dict(scene=scene,seed=seed,same_gpu=da['gpu']==db['gpu'],same_port=da['port']==db['port'])
  for k in [0,1]:
   with np.load(aj/f'policy_chunk_{k:03d}.npz') as x,np.load(bj/f'policy_chunk_{k:03d}.npz') as y:
    z[f'chunk{k}_equal']=np.array_equal(x['actions'],y['actions']);z[f'chunk{k}_max_abs_diff']=float(np.max(abs(x['actions']-y['actions'])))
  la=[json.loads(v) for v in (aj/'policy_lineage.jsonl').read_text().splitlines()];lb=[json.loads(v) for v in (bj/'policy_lineage.jsonl').read_text().splitlines()]
  for k in [0,1]:z[f'chunk{k}_observation_equal']=la[k]['observation_hash']==lb[k]['observation_hash'];z[f'chunk{k}_rng_equal']=la[k]['eval_rng']==lb[k]['eval_rng']
  with np.load(a['package_data']) as x,np.load(b['package_data']) as y:z['initial_state_equal']=np.array_equal(x['replay_state'][0],y['replay_state'][0])
  blocks.append(z)
summary=[]
for scene in ['main','pillar_x_plus40mm']:
 rr=[r for r in blocks if r['scene']==scene];summary.append(dict(scene=scene,n=50,same_gpu=sum(r['same_gpu'] for r in rr),same_port=sum(r['same_port'] for r in rr),initial_state_equal=sum(r['initial_state_equal'] for r in rr),**{key:sum(r[key] for r in rr) for key in ['chunk0_equal','chunk1_equal','chunk0_observation_equal','chunk1_observation_equal','chunk0_rng_equal','chunk1_rng_equal']},chunk0_max_abs_diff=max(r['chunk0_max_abs_diff'] for r in rr),chunk1_max_abs_diff=max(r['chunk1_max_abs_diff'] for r in rr)))
res=dict(known_five_method_seed_device_distribution=[dict(scene=s,known_distinct_gpu_n=int(n),seed_blocks=v,unknown_VLSA_gpu=1) for (s,n),v in counts.items()],summary=summary,per_pair=blocks,all_six_device_blocks=devblocks)
p=O/'audit/原600设备与G_GC首两块核查_修正.json';assert not p.exists();p.write_text(json.dumps(res,ensure_ascii=False,indent=2));print(json.dumps(dict(distribution=res['known_five_method_seed_device_distribution'],summary=summary),ensure_ascii=False,indent=2))
