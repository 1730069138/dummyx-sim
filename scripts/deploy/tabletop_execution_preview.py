"""Execution-aware kinematic forecast. No policy calls or physics integration.

Includes current-state geometry and final reference projection, then an empirical
joint response. Not a collision certificate; calibration is local to this arm.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import mujoco

class ExecutionReferencePredictor:
    def __init__(self,owner,model_path):
        from tabletop_pace_runtime import TabletopArm
        from tabletop_pace_preview import AGVF2Preview
        self.owner=owner;self.path=Path(model_path);self.model=json.loads(self.path.read_text())
        self.coef=np.asarray(self.model['coefficients'],float)
        if self.coef.shape!=(6,3) or not np.isfinite(self.coef).all():raise ValueError('Invalid empirical tracking model')
        if not np.isclose(self.model['dt'],owner.dt):raise ValueError('Tracking-model control period mismatch')
        self.data=mujoco.MjData(owner.arm.model);self.arm=TabletopArm(owner.arm.model,self.data,owner.arm.steps)
        self.shadow=AGVF2Preview(self.arm,owner.obstacle,owner.initial_tool_height,owner.config,
            distance_fn=owner.distance_fn,pickup_mode=owner.pickup_mode,joint_filter=owner.joint_filter,
            preserve_orientation=owner.preserve_orientation,current_guard=owner.current_guard,
            safety_only=owner.safety_only,six_dof=owner.six_dof,preview_enabled=False,
            route_preference=owner.route_preference,staged_route=owner.staged_route,proxy_mode=owner.proxy_mode)
        self.rows=[]
    def forecast(self,chunk,index,stop,tool_local,first_target=None,record=True):
        o=self.owner;a=self.arm;d=self.data;m=a.model
        d.qpos[:]=o.arm.data.qpos;d.qvel[:]=o.arm.data.qvel
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        wrist=d.body('link6_1');tool=d.body('real_screwdriver');R=wrist.xmat.reshape(3,3).copy()
        rel=R.T@(tool.xpos-wrist.xpos);relR=R.T@tool.xmat.reshape(3,3)
        tooladr=m.jnt_qposadr[m.body_jntadr[m.body('real_screwdriver').id]]
        def synchronize():
            mujoco.mj_kinematics(m,d)
            if tool_local is not None:
                wrist=d.body('link6_1');rotation=wrist.xmat.reshape(3,3)
                d.qpos[tooladr:tooladr+3]=wrist.xpos+rotation@rel
                quat=np.empty(4);mujoco.mju_mat2Quat(quat,(rotation@relR).ravel());d.qpos[tooladr+3:tooladr+7]=quat
                mujoco.mj_kinematics(m,d)
            mujoco.mj_comPos(m,d)
        out=[];self.shadow.pickup_ever=o.pickup_ever;self.shadow.route_phase=o.route_phase
        self.shadow.rows.clear();self.shadow.guard_failures.clear();self.shadow.guard_projection_events.clear()
        for j in range(index,stop):
            synchronize();q=d.qpos[a.qadr].copy();vel=d.qvel[a.dadr].copy()
            # Predict what the existing preview-OFF execution pipeline would request.
            if j==index and first_target is not None:
                target=np.asarray(first_target,float).copy()
            else:
                target,_=self.shadow.process(chunk,j,execution_stop=stop)
            reference,debug=self.shadow.project_current_target(target) if o.current_guard else (target,{'reference_guard_feasible':True})
            feasible=bool(debug['reference_guard_feasible'])
            if feasible:
                predicted=q+self.coef[:,0]*(reference[:6]-q)+self.coef[:,1]*o.dt*vel+self.coef[:,2]
                predicted=np.clip(predicted,a.limits[:,0],a.limits[:,1])
                d.qpos[a.qadr]=predicted;d.qvel[a.dadr]=(predicted-q)/o.dt
                d.qpos[a.gripper_qadr]=np.clip(reference[6]/2,0,.04)
                synchronize()
                clearance,normal=o.clearance(o.capsules(d,tool_local))
            else:
                # Forecasted reference rejection is explicitly logged, not a physical collision.
                clearance,normal=o.clearance(o.capsules(d,tool_local));clearance=min(float(clearance),0.)
            if record:
                self.rows.append(dict(action_index=len(o.rows),future_step=j-index+1,clearance_m=float(clearance),reference_feasible=feasible,predicted_q=d.qpos[a.qadr].copy(),protected_reference=reference.copy()))
            out.append((float(clearance),np.asarray(normal).copy(),j-index+1))
            if not feasible:break
        return out
    def arrays(self):
        return {key:np.asarray([r[key] for r in self.rows]) for key in self.rows[0]} if self.rows else {}
    def metadata(self):
        return dict(algorithm='protected_reference_empirical_tracking_v1',model_path=str(self.path),model_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),calibration=self.model,reference='same-state F2 without preview, then final reference guard',prediction='empirical joint response with rigid carried-tool proxy',limitations='Frozen future attraction/phase, empirical mean-reference approximation, no physics integration or continuous certificate')
