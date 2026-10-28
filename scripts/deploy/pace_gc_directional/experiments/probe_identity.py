from pathlib import Path
import sys,json,subprocess,os
import datetime
WORK=Path('/home/jun/dummyx-sim/outputs/pace_gc_directional/research_20261026_214604')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_json(p,v):
 assert not p.exists(),str(p)
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
remote=r'''import json,subprocess,pathlib,hashlib,os
pidlist=[int(x) for x in subprocess.check_output(['pgrep','-f','/home/jun/VLA/openpi/.venv/bin/python3'],text=True).split()]
processes=[]
for pid in pidlist:
 try:
  cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'serve_policy.py' not in cmd:continue
  env=pathlib.Path(f'/proc/{pid}/environ').read_bytes().split(b'\0');selected={s.split(b'=',1)[0].decode():s.split(b'=',1)[1].decode() for s in env if s.startswith((b'CUDA_VISIBLE_DEVICES=',b'XLA_PYTHON_CLIENT_MEM_FRACTION='))}
  processes.append(dict(pid=pid,cmdline=cmd,environment=selected,start_ticks=pathlib.Path(f'/proc/{pid}/stat').read_text().split()[21]))
 except (FileNotFoundError,PermissionError):pass
root=pathlib.Path('/home/jun/VLA/openpi');checkpoint=root/'checkpoints/pi0_dummyx_lora/dummyx_lora_run0405/40000'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[]
for p in sorted(checkpoint.rglob('*')):
 if p.is_file():files.append(dict(path=str(p),size=p.stat().st_size,sha256=digest(p)))
source=[]
for p in [root/'scripts/serve_policy.py',root/'src/openpi/policies/policy.py',root/'src/openpi/policies/policy_config.py',root/'src/openpi/training/config.py',root/'src/openpi/policies/dummyx_policy.py']:
 if p.exists():source.append(dict(path=str(p),sha256=digest(p)))
print(json.dumps(dict(gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),compute_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],text=True),services=processes,checkpoint_files=files,source=source)))'''
r=subprocess.run(['ssh','-i','/home/jun/1','-o','BatchMode=yes','-o','ConnectTimeout=12','jun@100.64.142.55','python3 -'],input=remote,text=True,capture_output=True,timeout=180)
if r.returncode:raise RuntimeError(r.stderr)
data=json.loads(r.stdout);data['captured_utc']=now();write_json(WORK/'evidence/remote_identity_initial.json',data)
# Metadata only: no inference requests here.
os.environ['NO_PROXY']=os.environ['no_proxy']='100.64.142.55,localhost,127.0.0.1'
sys.path.insert(0,'/home/jun/dummyx-sim/scripts/deploy')
from openpi_client.websocket_client_policy import WebsocketClientPolicy
metadata=[]
for gpu,port in [(0,8001),(1,8002),(2,8000),(3,8003),(4,8004),(5,8005)]:
 c=WebsocketClientPolicy('100.64.142.55',port);m=c.get_server_metadata();c._ws.close();assert m.get('eval_seed_protocol'),(gpu,port,m)
 matches=[p for p in data['services'] if f'--port={port} ' in p['cmdline'] and p['environment'].get('CUDA_VISIBLE_DEVICES')==str(gpu)]
 assert len(matches)==1 and 'pi0_dummyx_lora' in matches[0]['cmdline'] and 'dummy_real' not in matches[0]['cmdline']
 metadata.append(dict(gpu=gpu,port=port,metadata=m,service=matches[0]))
write_json(WORK/'evidence/service_metadata_initial.json',metadata)
print(json.dumps({'services_verified':len(metadata),'checkpoint_files_hashed':len(data['checkpoint_files']),'policy_calls':0,'all_matching_old_tabletop':True}))
