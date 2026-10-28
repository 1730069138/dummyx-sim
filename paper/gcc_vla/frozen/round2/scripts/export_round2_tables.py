"""Flatten validated analysis JSON for delivery; no recalculation or raw-file writes."""
from pathlib import Path
import argparse,csv,json,shutil
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();d=json.loads(a.analysis.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
def csvwrite(name,rows):
 if not rows:return
 out=a.output_dir/name;assert not out.exists();keys=list(dict.fromkeys(k for r in rows for k in r))
 with out.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
for exp in ['exp1','exp2','exp3']:
 rows=[]
 for g in d['groups']:
  if g['experiment']!=exp:continue
  r={k:g[k] for k in ['experiment','scene','config','n_planned','n_completed','complete_cohort']}
  for key in ['SR','STSR','PCR','pillar_tilt','tau_limit_episode']:
   v=g[key];r.update({key+'_count':v['count'],key+'_n':v['n'],key+'_percent':v['percent'],key+'_wilson95_lower_percent':v['wilson95_percent'][0],key+'_wilson95_upper_percent':v['wilson95_percent'][1]})
  for key in ['J','J4','Fpk','T','tau_limit_substeps','first_limit_relative_s']:
   for sk in ['n','mean','sd','min','max']:r[key+'_'+sk]=g[key][sk]
  for key in ['step_limit','infeasible_abort','other_task_termination']:r['fail_'+key]=g['failure_categories'].get(key,0)
  for k,v in g['failed_postpickup_prefirstplacement_guard'].items():r['failed_action_pickup_window_'+k]=v
  rows.append(r)
 csvwrite(exp+'_summary.csv',rows)
for source,target in [('per_episode.csv','all440_per_episode.csv'),('tests.csv','statistical_tests.csv'),('attempts.csv','all_attempts.csv'),('exp1_diagnostics_flat.csv','exp1_saturation_diagnostics.csv'),('exp3_failure_guard_windows.csv','exp3_failure_guard_windows.csv'),('optional_probe_per_episode.csv','optional_probe_per_episode.csv'),('optional_probe_descriptive_groups.csv','optional_probe_summary.csv')]:
 src=a.analysis.parent/source;dst=a.output_dir/target
 if not src.exists():continue
 assert not dst.exists();shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
print(json.dumps(dict(output_dir=str(a.output_dir.resolve()),completed=d['completed_formal'],optional=d['completed_optional_probe'])))
