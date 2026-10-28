from pathlib import Path
import inspect,importlib.util,json
O=Path(__file__).resolve().parents[1];p=Path('/home/jun/dummyx-sim/scripts/deploy/gcc_vla_paper/run_experiment.py');s=importlib.util.spec_from_file_location('verified_entry',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ns=dict(m.__dict__);source=inspect.getsource(m.check_services).replace('/home/jun/1','/home/jun/ssh-keys/1');exec(source,ns);remote,services=ns['check_services'](list(range(6)))
D=O/'services/identity';D.mkdir(exist_ok=False);(D/'remote.json').write_text(json.dumps(remote,indent=2));(D/'metadata.json').write_text(json.dumps(services,indent=2));print(json.dumps({'ready_gpus':[x['gpu'] for x in services],'ports':[x['port'] for x in services],'checkpoint_files_verified':len(remote['checkpoint_files']),'source_files_verified':len(remote['source'])}))
