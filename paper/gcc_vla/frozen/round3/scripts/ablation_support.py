from pathlib import Path
import json,hashlib
R=Path('/home/jun/dummyx-sim')
O=Path(__file__).resolve().parent.parent
class RootProxy:
 def __init__(self,root):self.root=root
 def __truediv__(self,suffix):
  if str(suffix)=='outputs/pace_external50/benchmark_20261027_062049/experiments/model_cache':return O/'experiments/model_cache'
  return self.root/suffix
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new_output(p):
 p=Path(p).resolve()
 if not p.is_relative_to(O):raise ValueError('Outside isolated package: '+str(p))
 return p
def verify_freeze(run):
 run=new_output(run);p=run/'protocol.json';assert sha(p)==(run/'protocol.sha256').read_text().strip()
 d=json.loads(p.read_text())
 for p,h in d['source_hashes'].items():assert sha(p)==h,('Frozen source drift',p)
 return d
