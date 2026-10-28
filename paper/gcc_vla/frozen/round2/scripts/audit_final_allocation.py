"""Read-only final source, allocation and initial action block audit; no physics import."""
from pathlib import Path
import json,hashlib,collections,datetime,argparse
import numpy as np
O=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
def h(p):
 x=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):x.update(b)
 return x.hexdigest()
backups=[]
for r in json.loads((O/'backup_sha256.json').read_text()):
 sh,bh=h(r['source']),h(r['backup']);backups.append(dict(source=r['source'],backup=r['backup'],expected=r['sha256'],original=sh,backup_hash=bh,pass_=sh==bh==r['sha256']))
checks=[];starts=collections.defaultdict(list);allocations=[];issues=[]
for stage in ['smoke','exp1','jump','exp23','exp23_recovery']:
 d=O/'experiments'/stage
 if not d.exists():continue
 prot=json.loads((d/'protocol.json').read_text());assert h(d/'protocol.json')==(d/'protocol.sha256').read_text().strip()
 for i,(src,expected) in enumerate(prot['source_hashes'].items()):
  snapshot=d/'frozen_sources'/str(i)/Path(src).name
  checks.append(dict(stage=stage,path=src,current=h(src),frozen=h(snapshot),expected=expected,pass_=h(src)==h(snapshot)==expected))
 for job in json.loads((d/'allocations.json').read_text()):
  jp=d/f"worker_gpu{job['gpu']}"/job['id'];rp=jp/'result.json';row=dict(job,stage=stage,result_path=str(rp),exists=rp.exists());allocations.append(row)
  if not rp.exists():
   recovered=O/'experiments/exp23_recovery'/f"worker_gpu{job['gpu']}"/job['id']/'result.json'
   if stage=='exp23' and job['seed']==860049 and job['gpu']==4 and recovered.exists():
    row.update(resolved_by_technical_recovery=str(recovered),partial_removed_by_user=not jp.exists());continue
   issues.append(dict(stage=stage,id=job['id'],issue='missing_result'));continue
  r=json.loads(rp.read_text());assert all(r[k]==v for k,v in job.items());reason=(r.get('result') or {}).get('reason','');technical=bool(r.get('error') or (r.get('metrics') or {}).get('metric_error') or reason.startswith(('policy_error:','simulation_error:','perception_error:')));row.update(technical=technical,reason=reason)
  if technical:issues.append(dict(stage=stage,id=job['id'],issue='technical_attempt',reason=reason));continue
  assert Path(r['trace_path']).is_relative_to(O)
  chunk=jp/'policy_chunk_000.npz';lineage=jp/'policy_lineage.jsonl'
  if chunk.exists() and lineage.exists():
   with np.load(chunk) as z:c=z['actions'].copy();s=z['state'].copy()
   first=json.loads(lineage.read_text().splitlines()[0]);starts[('exp23' if stage=='exp23_recovery' else stage,job['experiment'],job['scene_id'],job['seed'])].append(dict(id=job['id'],gpu=job['gpu'],c=c,s=s,obs=first['observation_hash']))
  else:issues.append(dict(stage=stage,id=job['id'],issue='initial_block_missing'))
pairs=[]
for key,rows in starts.items():
 ref=rows[0]
 for x in rows[1:]:pairs.append(dict(stage=key[0],experiment=key[1],scene=key[2],seed=key[3],reference=ref['id'],candidate=x['id'],same_gpu=ref['gpu']==x['gpu'],state_equal=np.array_equal(ref['s'],x['s']),observation_hash_equal=ref['obs']==x['obs'],first_action_block_equal=np.array_equal(ref['c'],x['c']),first_action_block_max_abs_diff=float(np.max(abs(ref['c']-x['c'])))))
# Optional limiter shares current exp1 F seeds deliberately; compare its initial service request too.
for job in [x for x in allocations if x['stage']=='jump' and x['exists'] and not x.get('technical')]:
 ref=O/'experiments/exp1'/f"worker_gpu{job['gpu']}"/f"exp1_F_{job['scene_id']}_r{job['rep']:03d}"/'policy_chunk_000.npz';probe=Path(job['result_path']).parent/'policy_chunk_000.npz'
 with np.load(ref) as z,np.load(probe) as zz:pairs.append(dict(stage='jump_vs_exp1',experiment='exp1_jump',scene=job['scene_id'],seed=job['seed'],state_equal=np.array_equal(z['state'],zz['state']),first_action_block_equal=np.array_equal(z['actions'],zz['actions']),first_action_block_max_abs_diff=float(np.max(abs(z['actions']-zz['actions'])))))
result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True,backup_pass=all(x['pass_'] for x in backups),backup_count=len(backups),source_snapshot_checks=checks,source_all_pass=all(x['pass_'] for x in checks),allocation_count=len(allocations),by_stage=dict(collections.Counter(x['stage'] for x in allocations)),allocations=allocations,issues=issues,initial_block_contrasts=pairs,initial_all_equal=all(x['first_action_block_equal'] for x in pairs),notes=['Smoke used historical pre-amendment statistics plan; current hash mismatch for that one metadata-only amended file is expected, frozen snapshot still equals expected.','First block equality is an input/RNG audit, not subsequent closed-loop trajectory equivalence or a reason to change pre-registered unpaired tests.'])
# Smoke intentionally froze the pre-formal metadata plan; retain exact mismatch and explain.
for c in checks:
 if c['stage']=='smoke' and c['path']==str(O/'statistics_plan.json') and c['frozen']==c['expected'] and c['current']!=c['expected']:c['explained_preformal_metadata_amendment']=True
result['unexplained_source_mismatches']=[c for c in checks if not c['pass_'] and not c.get('explained_preformal_metadata_amendment')]
with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(path=str(a.output),backups=result['backup_pass'],unexplained_source_mismatches=len(result['unexplained_source_mismatches']),issues=len(issues),initial_pairs=len(pairs),first_blocks_all_equal=result['initial_all_equal'])))
