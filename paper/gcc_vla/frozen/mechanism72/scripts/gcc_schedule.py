from pathlib import Path
import sys,os,json,hashlib,subprocess,importlib.util,inspect,copy,datetime,shutil,argparse
R=Path('/home/jun/dummyx-sim');O=Path(Path('/tmp/gcc_revision_path').read_text());S=O/'scripts';sys.path.insert(0,str(S))
from ablation_support import sha
spec=importlib.util.spec_from_file_location('gccentry',R/'scripts/deploy/gcc_vla_paper/run_experiment.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','formal'],required=True);a=p.parse_args();run=O/'experiments'/a.stage;run.mkdir(parents=True,exist_ok=False)
entry.verify_sources()
identity=O/'services/identity'
if not identity.exists():
 identity.mkdir();f=inspect.getsource(entry.check_services).replace('/home/jun/1','/home/jun/ssh-keys/1');ns=dict(entry.__dict__);exec(f,ns);remote,services=ns['check_services'](list(range(6)))
 (identity/'remote.json').write_text(json.dumps(remote,indent=2));(identity/'metadata.json').write_text(json.dumps(services,indent=2))
if a.stage=='formal':assert json.loads((O/'experiments/pilot/regression_receipt.json').read_text())['passed']
profiles=json.loads((R/'scripts/deploy/gcc_vla_paper/config/profiles.json').read_text());groups=[];jobs=[];n=1 if a.stage=='pilot' else 12
for scene in ['main','pillar_x_plus40mm']:
 for arm in ['F','A','B']:
  base=copy.deepcopy(profiles['pace_gc/'+scene]);config=base['config'];gid=arm+'_'+scene;config.update(id=gid,phase='mechanism_'+a.stage,mechanism_ablation=arm)
  flags=base['flags'];profile=run/'profiles'/f'{gid}.json'
  for k,v in [('--host','100.64.142.55'),('--paper_experiment',profile),('--output_dir',run/'unused'),('--model_cache',O/'experiments/model_cache')]:flags=entry.set_flag(flags,k,v)
  profile.parent.mkdir(exist_ok=True);profile.write_text(json.dumps(config,indent=2));groups.append(dict(id=gid,config=config,flags=flags))
# Interleave all arms and scenes. Rotate each condition across GPU/port.
for rep in range(n):
 for si,scene in enumerate(['main','pillar_x_plus40mm']):
  for ai,arm in enumerate(['F','A','B']):
   gpu=(si*3+ai+rep)%6;seed=(729000 if si==0 else 739000)+(0 if a.stage=='pilot' else 1000)+rep
   jobs.append(dict(id=f'{arm}_{scene}_r{rep:03d}',group=arm+'_'+scene,arm=arm,rep=rep,seed=seed,gpu=gpu,port=entry.PORTS[gpu],scene_id=scene,method_id='pace_gc'))
plan=dict(stage=a.stage,n_per_arm_scene=n,jobs=jobs,arms={'F':'complete frozen GCC-VLA','A':'vc suppressed before DLS; support estimator/height brake and final guard retained','B':'current_guard false after construction; action-level geometry and contact unchanged'},unit='episode seed block; distinct from historical600 and pilot',primary=['SR','STSR','PCR','all_full_J','all_fixed4_J'],statistics='Two F-A/F-B comparisons x two scenes x five outcomes =20 primary contrasts; paired seed differences only after exact initial-state+first-policy-block audit; Wilson 95% per proportion; paired bootstrap mean difference 10000 resamples seed20261028 and exact McNemar / paired sign permutation for loads, Holm over all20; otherwise unpaired descriptive only. n12 exploratory, no equivalence/noninferiority claim.',stopping='No task-outcome stop or sample selection. Technical error retains unknown and stops infrastructure; repair only after recording, no replacement by favourable sample.',fixed_window='0..4s requires all valid samples; no zero padding; if early termination shorter, report missing coverage and common observed window sensitivity separately',pilot_admission='All six complete; no technical/finite/schema errors; F baseline exact sources; A force_delta identically0 with feedback+anchor retained; B guard disabled; task success never a gate.',checkpoint='/home/jun/VLA/openpi/checkpoints/pi0_dummyx_lora/dummyx_lora_run0405/40000',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'statistics_plan.json').write_text(json.dumps(plan,indent=2));source_manifest=json.loads((R/'scripts/deploy/gcc_vla_paper/config/source_manifest.json').read_text())['files'];source_hashes={str(R/k):h for k,h in source_manifest.items()};source_hashes.update({str(p):sha(p) for p in S.glob('*.py')});source_hashes[str(run/'statistics_plan.json')]=sha(run/'statistics_plan.json')
protocol=dict(run_id='gcc_mechanism_20261028_'+a.stage,groups=groups,workers={str(g):[j for j in jobs if j['gpu']==g] for g in range(6)},factory='neo_factory:build_modules',service_metadata=str(identity/'metadata.json'),pressure_scenes={'pillar_x_plus40mm':{'folder':str(R/'models/gcc_vla_paper/x40')}},source_hashes=source_hashes,first_batch_gate=False)
(run/'protocol.json').write_text(json.dumps(protocol,indent=2));(run/'protocol.sha256').write_text(sha(run/'protocol.json'))
# Freeze independent source snapshots before any client executes.
freeze=run/'frozen_sources';freeze.mkdir()
for i,(path,h) in enumerate(source_hashes.items()):
 q=Path(path)
 if q.suffix in ['.py','.json']:
  dest=freeze/str(i)/q.name;dest.parent.mkdir();shutil.copy2(q,dest);assert sha(dest)==h
processes=[]
for gpu in range(6):
 cmd=[str(entry.DUMMY_PYTHON),'-B','-u',str(S/'run_ablation_worker.py'),'--run',str(run),'--gpu',str(gpu),'--port',str(entry.PORTS[gpu]),'--factory','neo_factory:build_modules']
 log=(run/f'worker_gpu{gpu}.log').open('xb');processes.append(subprocess.Popen(cmd,cwd=R,env=entry.headless_environment(),stdout=log,stderr=subprocess.STDOUT));log.close()
print('REGISTERED',run,'planned',len(jobs),flush=True)
for p in processes:p.wait()
status={str(i):p.returncode for i,p in enumerate(processes)};(run/'exit_status.json').write_text(json.dumps(status));print('COMPLETED',status,flush=True)
