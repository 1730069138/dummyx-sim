from pathlib import Path
import argparse,copy,json,hashlib,subprocess,importlib.util,datetime
O=Path(__file__).resolve().parents[1];R=Path('/home/jun/dummyx-sim');s=importlib.util.spec_from_file_location('entry',R/'scripts/deploy/gcc_vla_paper/run_experiment.py');entry=importlib.util.module_from_spec(s);s.loader.exec_module(entry)
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','exp1','exp23','jump'],required=True);a=p.parse_args();out=O/'experiments'/a.stage;out.mkdir(exist_ok=False)
profiles=json.loads((R/'scripts/deploy/gcc_vla_paper/config/profiles.json').read_text());groups=[];jobs=[]
def group(exp,label,method,scene,direction='fixed_up_base',ablation=None,limit=None):
 gid=f'{exp}_{label}_{scene}';base=copy.deepcopy(profiles[method+'/'+scene]);cfg=base['config'];cfg.update(id=gid,phase='smoke' if a.stage=='smoke' else 'formal',experiment=exp,config_name=label,geometry_direction=direction)
 if ablation is not None:cfg['mechanism_ablation']=ablation
 if limit is not None:cfg['target_jump_limit_rad']=limit
 flags=base['flags'];f=out/'profiles'/f'{gid}.json';f.parent.mkdir(exist_ok=True);f.write_text(json.dumps(cfg,indent=2))
 for key,val in [('--host','100.64.142.55'),('--paper_experiment',f),('--output_dir',out/'unused'),('--model_cache',O/'experiments/model_cache'),('--pace_tracking_model',O/'runtime/tabletop_tracking_model.json'),('--groundingdino_config','/home/jun/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'),('--groundingdino_checkpoint','/home/jun/GroundingDINO/weights/groundingdino_swint_ogc.pth')]:flags=entry.set_flag(flags,key,val)
 groups.append(dict(id=gid,config=cfg,flags=flags));return gid

def job(gid,seed,rep,gpu):
 cfg=next(x['config'] for x in groups if x['id']==gid);jobs.append(dict(id=f'{gid}_r{rep:03d}',group=gid,arm=cfg['config_name'],config_name=cfg['config_name'],experiment=cfg['experiment'],stage='smoke' if a.stage=='smoke' else 'formal',rep=rep,seed=seed,gpu=gpu,port=entry.PORTS[gpu],scene_id=cfg['scene_id'],method_id=cfg['method_id']))
if a.stage=='smoke':
 for exp,label,method,scene,seed,gpu,direction,ab in [('exp1','raw','raw_pi0','main',890000,0,'fixed_up_base',None),('exp1','F','pace_gc','pillar_x_plus40mm',890001,1,'fixed_up_base',None),('exp2','fixed','pace_gc','pillar_x_plus40mm',890100,2,'fixed_up_base',None),('exp2','normal','pace_gc','pillar_x_plus40mm',890100,2,'closest_normal',None),('exp3','F','pace_gc','pillar_x_plus40mm',890200,3,'fixed_up_base',None),('exp3','B','pace_gc','pillar_x_plus40mm',890200,3,'fixed_up_base','B')]:job(group(exp,label,method,scene,direction,ab),seed,0,gpu)
elif a.stage=='exp1':
 assert (O/'audit/smoke_admission.json').is_file() and json.loads((O/'audit/smoke_admission.json').read_text())['passed']
 for si,scene in enumerate(['main','pillar_x_plus40mm']):
  ids=[group('exp1',label,method,scene) for label,method in [('raw','raw_pi0'),('G','pace_g'),('F','pace_gc'),('oscbf','oscbf')]]
  for rep in range(5):
   for k in range(4):job(ids[(k+rep)%4],810000+si*10000+rep,rep,(rep+3*si)%6)
elif a.stage=='exp23':
 assert json.loads((O/'audit/smoke_admission.json').read_text())['passed']
 for exp,seedbase in [('exp2',830000),('exp3',850000)]:
  for si,scene in enumerate(['main','pillar_x_plus40mm']):
   ids=([group(exp,'fixed','pace_gc',scene),group(exp,'normal','pace_gc',scene,'closest_normal')] if exp=='exp2' else [group(exp,'F','pace_gc',scene),group(exp,'B','pace_gc',scene,ablation='B')])
   for rep in range(50):
    for k in range(2):job(ids[(k+rep)%2],seedbase+si*10000+rep,rep,(rep+3*si)%6)
else:
 threshold=json.loads((O/'audit/jump_limit_frozen.json').read_text());assert threshold['source_n']==10
 gid=group('exp1_jump','F_limited','pace_gc','pillar_x_plus40mm',limit=threshold['threshold_rad'])
 for rep in range(5):job(gid,820000+rep,rep,(rep+3)%6)
# Same-seed use is intentional within each contrast; all ranges exclude the original600/72.
old=json.loads((R/'outputs/pace_latest_evidence_20261027_1400/逐回合汇总.json').read_text());oldseeds={int(x['seed']) for x in old};oldseeds|=set(range(730000,730012))|set(range(740000,740012))
assert all(x['seed'] not in oldseeds for x in jobs)
runid=O.name+'_'+a.stage
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*sorted((O/'runtime').glob('*.*')),O/'scripts/run_round2_worker.py',O/'scripts/round2_trace.py',O/'scripts/ablation_support.py',O/'scripts/schedule_round2.py',O/'statistics_plan.json',O/'services/identity/metadata.json'] if p.is_file()}
# Include unchanged historical control/model identities to detect drift before every episode.
for rel,h in json.loads((R/'scripts/deploy/gcc_vla_paper/config/source_manifest.json').read_text())['files'].items():sources[str(R/rel)]=h
protocol=dict(run_id=runid,groups=groups,workers={str(g):[j for j in jobs if j['gpu']==g] for g in range(6)},factory='neo_factory:build_modules',service_metadata=str(O/'services/identity/metadata.json'),pressure_scenes={'pillar_x_plus40mm':{'folder':str(R/'models/gcc_vla_paper/x40')}},source_hashes=sources,first_batch_gate=False,plan='Read MD as authorized; fixed 50ms/1ms/300step/placement; no outcome based stop',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));(out/'protocol.sha256').write_text(hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest());(out/'allocations.json').write_text(json.dumps(jobs,indent=2))
freeze=out/'frozen_sources';freeze.mkdir()
for i,(name,h) in enumerate(sources.items()):
 src=Path(name);assert hashlib.sha256(src.read_bytes()).hexdigest()==h,(name,'source drift');dst=freeze/str(i)/src.name;dst.parent.mkdir();shutil_copy=False
 import shutil;shutil.copy2(src,dst);assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
processes=[]
for gpu in range(6):
 if not protocol['workers'][str(gpu)]:continue
 cmd=[str(entry.DUMMY_PYTHON),'-B','-u',str(O/'scripts/run_round2_worker.py'),'--run',str(out),'--gpu',str(gpu),'--port',str(entry.PORTS[gpu]),'--factory','neo_factory:build_modules'];log=(out/f'worker_gpu{gpu}.log').open('xb');p=subprocess.Popen(cmd,cwd=R,env=entry.headless_environment(),stdout=log,stderr=subprocess.STDOUT);log.close();processes.append((gpu,p))
(out/'scheduler_pids.json').write_text(json.dumps({str(g):p.pid for g,p in processes}));print('REGISTERED',runid,'n=',len(jobs),'gpus=',[g for g,p in processes],flush=True)
for gpu,p in processes:p.wait()
status={str(g):p.returncode for g,p in processes};(out/'exit_status.json').write_text(json.dumps(status));print('EXIT',status,flush=True)
