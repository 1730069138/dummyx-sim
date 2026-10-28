from pathlib import Path
import sys,json,hashlib
import numpy as np
import mujoco
from types import SimpleNamespace
O=Path(Path('/tmp/gcc_revision_path').read_text());S=O/'backup/冻结运行时';sys.path.insert(0,str(S))
from tabletop_pace_runtime import TabletopArm
from tabletop_pace_preview import AGVF2Preview,AGVF2Config
from tabletop_tool_proxy import batch_capsule_clearances
D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400');idx=json.loads((D/'逐回合汇总.json').read_text());out=[];models={}
for rr in idx:
 if rr['method'] not in ['pace_g','pace_gc']:continue
 p=Path(rr['package_data']);ep=p.parent;run=ep.parent
 mf=json.loads((run/'manifest.json').read_text());mh=mf['model_sha256']
 if mh not in models:models[mh]=mujoco.MjModel.from_binary_path(str(run/'model.mjb'))
 m=models[mh];d=mujoco.MjData(m);z=np.load(p);g=np.load(ep/'preview_ep_001.npz');st=np.load(ep/'stages_ep_001.npz')
 spec=int(z['replay_state_spec']);mujoco.mj_setState(m,d,z['replay_state'][0],spec);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
 arm=TabletopArm(m,d);meta=json.loads((ep/'metadata_ep_001.json').read_text())['preview'];obs={k:np.asarray(meta['obstacle'][k]) for k in ['p1','p2','r','placement_goal']}
 helper=AGVF2Preview(arm,obs,float(d.xpos[arm.target_body,2]),AGVF2Config(max_correction_m_s=.15,lift_preference=0),distance_fn=lambda *a:None,pickup_mode='latched',joint_filter='off',preserve_orientation=True,current_guard=True,safety_only=True,preview_enabled=False,route_preference=True,staged_route=True,prediction_mode='execution_lift',proxy_mode='components')
 scratch=mujoco.MjData(m)
 for k in np.flatnonzero(g['geometry_active']):
  mujoco.mj_setState(m,d,z['replay_state'][k],spec);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
  tl=helper.carried_tool_local(d) if g['tool_capsule_attached'][k] else None
  caps=helper.capsules(d,tl);vals,norm,points=batch_capsule_clearances(caps,helper.obstacle);i=int(np.argmin(vals));n=norm[i];point=points[i]
  dc=float(vals[i]);assert abs(dc-float(g['current_clearance_m'][k]))<1e-10,(dc,g['current_clearance_m'][k])
  J=np.zeros((3,m.nv));Jr=J.copy();mujoco.mj_jacSite(m,d,J,Jr,helper.tcp_id)
  JP=np.zeros_like(J);body=helper.eef_capsules_local[i]['body_id'] if i<len(helper.eef_capsules_local) else helper.wrist_id
  mujoco.mj_jac(m,d,JP,None,point,body)
  dq=st['geometry_target'][k,:6]-st['before_geometry_target'][k,:6]
  pred=float(n@J[:,arm.dadr]@dq);actual=float(n@JP[:,arm.dadr]@dq)
  def clearance(q):
   scratch.qpos[:]=d.qpos;scratch.qpos[arm.qadr]=q;mujoco.mj_kinematics(m,scratch)
   return helper.clearance(helper.capsules(scratch,tl))[0]
  small=clearance(d.qpos[arm.qadr]+1e-4*dq)-dc
  before=clearance(st['before_geometry_target'][k,:6]);after=clearance(st['geometry_target'][k,:6])
  out.append(dict(method=rr['method'],scene=rr['scene'],seed=rr['seed'],action_index=int(k),current_clearance_m=dc,tcp_linear_delta_m=pred,closest_linear_delta_m=actual,small_step_delta_m=small,nominal_target_clearance_m=before,geometry_target_clearance_m=after,endpoint_gain_m=after-before,angular_added_norm=float(np.linalg.norm(Jr[:,arm.dadr]@dq))))
 z.close();g.close();st.close()
 if len(out)%1000<70:print('processed',rr['assignment_id'],len(out),flush=True)
(O/'audit/kinematic_local_counterfactual.json').write_text(json.dumps(out,ensure_ascii=False))
groups=[]
for key in sorted({(r['method'],r['scene']) for r in out}):
 a=[r for r in out if (r['method'],r['scene'])==key];g=dict(method=key[0],scene=key[1],active_states=len(a))
 for f in ['tcp_linear_delta_m','closest_linear_delta_m','small_step_delta_m','endpoint_gain_m']:
  v=np.array([r[f] for r in a]);g[f]=dict(positive=int((v>1e-12).sum()),negative=int((v< -1e-12).sum()),mean=float(v.mean()),min=float(v.min()),max=float(v.max()))
 g['angular_added_max']=max(r['angular_added_norm'] for r in a);groups.append(g)
(O/'audit/kinematic_summary.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2));print(json.dumps(groups,indent=2));print('policy calls=0, physics steps=0')
