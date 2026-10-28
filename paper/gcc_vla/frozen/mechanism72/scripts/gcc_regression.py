from pathlib import Path
import sys,json,hashlib
import numpy as np,mujoco
O=Path(Path('/tmp/gcc_revision_path').read_text());sys.path[:0]=[str(O/'scripts'),str(O/'backup/冻结运行时')]
from tabletop_pace_runtime import TabletopArm,TabletopAdmittance
from joint_torque_observer import ForceEstimate
from ablation_contact import NoForceVelocity
D=Path('/home/jun/dummyx-sim/outputs/pace_latest_evidence_20261027_1400');idx=json.loads((D/'逐回合汇总.json').read_text());receipts=[]
for scene in ['main','pillar_x_plus40mm']:
 rr=next(r for r in idx if r['method']=='pace_gc' and r['scene']==scene);p=Path(rr['package_data']);root=p.parent.parent;m=mujoco.MjModel.from_binary_path(str(root/'model.mjb'));d=mujoco.MjData(m);z=np.load(p);c=np.load(p.parent/'contact_ep_001.npz');guard=np.load(p.parent/'guard_ep_001.npz');per=json.loads((root/'perception_ep_001/perception.json').read_text())['geometry']
 arm=TabletopArm(m,d);base=TabletopAdmittance(arm,perception=per);A=NoForceVelocity(arm,perception=per)
 for k in range(min(len(z['replay_state']),100)):
  j=k*50;mujoco.mj_setState(m,d,z['replay_state'][k],int(z['replay_state_spec']));mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
  for ctrl in [base,A]:
   ctrl.support.z=float(c['support_before_m'][j]) if c['support_before_valid'][j] else None
   ctrl.latest=ForceEstimate(c['feedback_force_N'][j].copy(),np.zeros(6),bool(np.linalg.norm(c['feedback_force_N'][j])>0));ctrl.latest_time=float(c['feedback_sample_time_s'][j]) if c['feedback_sample_valid'][j] else None
   ctrl.compensation[:]=c['force_delta_m_s'][j-1] if j else 0.
  nominal=c['nominal_action_target'][j];b,bd=base.process(nominal);a,ad=A.process(nominal)
  err=float(np.max(abs(b-guard['post_contact_target'][j])));assert err<1e-11,(scene,k,err)
  assert np.array_equal(ad['feedback_force_N'],bd['feedback_force_N']);assert ad['support_before_valid']==bd['support_before_valid'];assert np.array_equal(ad['height_delta_m_s'],bd['height_delta_m_s']);assert np.linalg.norm(ad['force_delta_m_s'])==0
  tcp,J=A.geometry();expected=np.clip(nominal[:6]+np.clip(J.T@np.linalg.solve(J@J.T+.05**2*np.eye(3),ad['height_delta_m_s']*.05),-.05,.05),arm.limits[:,0],arm.limits[:,1]);assert np.max(abs(expected-a[:6]))<1e-13
  receipts.append(dict(scene=scene,action=k,baseline_historical_max_error_rad=err,height_active=bool(np.linalg.norm(ad['height_delta_m_s'])>1e-10),A_changed=bool(np.max(abs(a-b))>1e-10)))
assert any(r['height_active'] for r in receipts) and any(r['A_changed'] for r in receipts)
res=dict(states=len(receipts),passed=True,max_historical_error_rad=max(r['baseline_historical_max_error_rad'] for r in receipts),height_active_states=sum(r['height_active'] for r in receipts),A_changed_states=sum(r['A_changed'] for r in receipts),policy_calls=0,physics_steps=0,rows=receipts)
(O/'audit/ablation_A_component_regression.json').write_text(json.dumps(res,indent=2));print({k:v for k,v in res.items() if k!='rows'})
