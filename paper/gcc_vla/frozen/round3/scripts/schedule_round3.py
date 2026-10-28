"""Only schedules two frozen round2 configurations; no controller changes."""
from pathlib import Path
import argparse,copy,json,hashlib,subprocess,importlib.util,datetime,shutil
O=Path(__file__).resolve().parents[1];R=Path('/home/jun/dummyx-sim')
s=importlib.util.spec_from_file_location('entry',R/'scripts/deploy/gcc_vla_paper/run_experiment.py');entry=importlib.util.module_from_spec(s);s.loader.exec_module(entry)
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','formal'],required=True);p.add_argument('--run-name');a=p.parse_args()
plan=json.loads((O/'statistics_plan.json').read_text());assert hashlib.sha256((O/'statistics_plan.json').read_bytes()).hexdigest()==(O/'statistics_plan.sha256').read_text().strip()
run_name=a.run_name or a.stage
assert run_name.replace('_','').isalnum()
out=O/'experiments'/run_name;out.mkdir(parents=True,exist_ok=False)
if a.stage=='formal':assert json.loads((O/'audit/smoke_admission.json').read_text())['passed']
profiles=json.loads((R/'scripts/deploy/gcc_vla_paper/config/profiles.json').read_text());groups=[];jobs=[]
for scene in (['pillar_x_plus40mm'] if a.stage=='smoke' else ['main','pillar_x_plus40mm']):
 for label in ['F','F_limited']:
  base=copy.deepcopy(profiles['pace_gc/'+scene]);cfg=base['config'];gid='jump_expansion_'+label+'_'+scene
  cfg.update(id=gid,phase=a.stage,experiment='jump_expansion',config_name=label,geometry_direction='fixed_up_base')
  if label=='F_limited':cfg['target_jump_limit_rad']=plan['jump_limiter']['threshold_rad']
  else:cfg.pop('target_jump_limit_rad',None)
  flags=base['flags'];f=out/'profiles'/f'{gid}.json';f.parent.mkdir(exist_ok=True);f.write_text(json.dumps(cfg,indent=2))
  for key,val in [('--host','100.64.142.55'),('--paper_experiment',f),('--output_dir',out/'unused'),('--model_cache',O/'experiments/model_cache'),('--pace_tracking_model',O/'runtime/tabletop_tracking_model.json'),('--groundingdino_config','/home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),('--groundingdino_checkpoint','/home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth')]:flags=entry.set_flag(flags,key,val)
  groups.append({'id':gid,'config':cfg,'flags':flags})
allocation=([{'id':f'jump_expansion_{label}_pillar_x_plus40mm_smoke','config_name':label,'scene':'pillar_x_plus40mm','seed':890300,'rep':0,'gpu':2,'port':entry.PORTS[2]} for label in ['F','F_limited']] if a.stage=='smoke' else plan['allocations'])
for job in allocation:
 label=job['config_name'];scene=job['scene'];gid='jump_expansion_'+label+'_'+scene
 jobs.append(dict(id=job['id'],group=gid,arm=label,config_name=label,experiment='jump_expansion',stage=a.stage,rep=job['rep'],seed=job['seed'],gpu=job['gpu'],port=job['port'],scene_id=scene,method_id='pace_gc'))
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted((O/'runtime').glob('*.*')),O/'scripts/run_round2_worker.py',O/'scripts/round2_trace.py',O/'scripts/ablation_support.py',Path(__file__),O/'statistics_plan.json',O/'audit/jump_limit_frozen.json',O/'services/identity/metadata.json'] if p.is_file()}
for rel,h in json.loads((R/'scripts/deploy/gcc_vla_paper/config/source_manifest.json').read_text())['files'].items():sources[str(R/rel)]=h
protocol=dict(run_id=O.name+'_'+a.stage,groups=groups,workers={str(g):[j for j in jobs if j['gpu']==g] for g in range(6)},factory='neo_factory:build_modules',service_metadata=str(O/'services/identity/metadata.json'),pressure_scenes={'pillar_x_plus40mm':{'folder':str(R/'models/gcc_vla_paper/x40')}},source_hashes=sources,first_batch_gate=False,statistics_plan=str(O/'statistics_plan.json'),statistics_plan_sha256=hashlib.sha256((O/'statistics_plan.json').read_bytes()).hexdigest(),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'protocol.sha256').write_text(hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest());(out/'allocations.json').write_text(json.dumps(jobs,indent=2))
freeze=out/'frozen_sources';freeze.mkdir()
for i,(name,h) in enumerate(sources.items()):
 src=Path(name);assert hashlib.sha256(src.read_bytes()).hexdigest()==h,(name,'source drift');dst=freeze/str(i)/src.name;dst.parent.mkdir();shutil.copy2(src,dst);assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
processes=[]
for gpu in range(6):
 if not protocol['workers'][str(gpu)]:continue
 cmd=[str(entry.DUMMY_PYTHON),'-B','-u',str(O/'scripts/run_round2_worker.py'),'--run',str(out),'--gpu',str(gpu),'--port',str(entry.PORTS[gpu]),'--factory','neo_factory:build_modules']
 env=entry.headless_environment();env.update(MPLCONFIGDIR=str(O/'audit/mpl_cache'),XDG_CACHE_HOME=str(O/'audit/local_cache'),HF_HOME=str(O/'audit/hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 log=(out/f'worker_gpu{gpu}.log').open('xb');process=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);log.close();processes.append((gpu,process))
(out/'scheduler_pids.json').write_text(json.dumps({str(g):p.pid for g,p in processes}));print('REGISTERED',protocol['run_id'],'n=',len(jobs),'gpus=',[g for g,p in processes],flush=True)
for gpu,p in processes:p.wait()
status={str(g):p.returncode for g,p in processes};(out/'exit_status.json').write_text(json.dumps(status));print('EXIT',status,flush=True)
