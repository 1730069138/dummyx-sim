# Adapted from agv_box_attraction.py; original AGV implementation is unchanged.
"""Explicit DINO box-goal preference, separate from autonomous safety claims."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass(frozen=True)
class BoxAttractionConfig:
    gain_s_inv: float = 1.
    maximum_speed_m_s: float = 1.2
    stop_distance_m: float = .035
    activation_range_m: float = .4
    minimum_lift_m: float = .05
    decay_power: float = 1.
    forbid_outward: bool = False
    staged_route: bool = False
    check_descent: bool = True
    commit_transport: bool = False
    preserve_orientation: bool = True

    def __post_init__(self):
        if not np.isfinite(self.decay_power) or self.decay_power < 0:
            raise ValueError("decay_power must be finite and nonnegative; zero is constant-speed ablation")

class AGVBoxAttraction:
    def __init__(self,arm,goal,initial_height,config=None):
        self.arm=arm;self.goal=np.asarray(goal,dtype=float).copy();self.initial_height=float(initial_height)
        if self.goal.shape!=(3,) or not np.isfinite(self.goal).all():raise ValueError('Invalid perceived box attraction point')
        self.config=config or BoxAttractionConfig();self.dt=arm.steps*arm.model.opt.timestep
        self.pose_scratch=arm.mj.MjData(arm.model);self.scratch=arm.mj.MjData(arm.model);self.rows=[];self.correction=np.zeros(6);self.phase="inactive";self.transport_committed=False
    def preserve_target_orientation(self, nominal_q, proposed_q):
        """Keep the added attraction from changing nominal TCP orientation (1e-6 rad)."""
        a=self.arm;d=self.pose_scratch
        d.qpos[:]=a.data.qpos;d.qpos[a.qadr]=nominal_q
        a.mj.mj_kinematics(a.model,d);a.mj.mj_comPos(a.model,d)
        reference=d.site('tcp_site').xmat.reshape(3,3).copy()
        nominal_center=(d.body('link6_1').xpos+d.site('tcp_site').xpos)/2.
        if self.config.forbid_outward:
            from tabletop_route_preference import near_base_direction
            near=np.asarray(self.near_direction)
        q=np.clip(proposed_q,a.limits[:,0],a.limits[:,1]).copy()
        for _ in range(12):
            d.qpos[a.qadr]=q;a.mj.mj_kinematics(a.model,d);a.mj.mj_comPos(a.model,d)
            error=Rotation.from_matrix(reference@d.site('tcp_site').xmat.reshape(3,3).T).as_rotvec()
            if np.linalg.norm(error)<=1e-6:
                if self.config.forbid_outward:
                    center=(d.body('link6_1').xpos+d.site('tcp_site').xpos)/2.
                    if near@(center-nominal_center)<-1e-9:return np.asarray(nominal_q).copy(),0.,True
                return q,float(np.linalg.norm(error)),False
            Jr=np.zeros((3,a.model.nv))
            a.mj.mj_jacSite(a.model,d,None,Jr,a.model.site('tcp_site').id)
            J=Jr[:,a.dadr]
            delta=J.T@np.linalg.solve(J@J.T+1e-8*np.eye(3),error)
            if not np.isfinite(delta).all():break
            q=np.clip(q+delta,a.limits[:,0],a.limits[:,1])
            if np.linalg.norm(q-nominal_q)>2*np.linalg.norm(proposed_q-nominal_q)+1e-9:break
        # No feasible verified orientation-preserving attraction: pass nominal through.
        return np.asarray(nominal_q).copy(),0.,True

    def process(self,policy_target,index):
        a=self.arm;c=self.config;nominal=np.asarray(policy_target,dtype=float).copy()
        if nominal.shape!=(7,) or not np.isfinite(nominal).all():raise ValueError('Expected seven finite targets')
        nominal[:6]=np.clip(nominal[:6],a.limits[:,0],a.limits[:,1]);nominal[6]=np.clip(nominal[6],0,.08)
        d=self.scratch;d.qpos[:]=a.data.qpos;a.mj.mj_kinematics(a.model,d);a.mj.mj_comPos(a.model,d)
        tcp=d.site('tcp_site').xpos.copy();tool=d.body('real_screwdriver').xpos
        center=(d.body('link6_1').xpos+tcp)/2.;error=self.goal-center;distance=float(np.linalg.norm(error))
        carry=bool(nominal[6]<=.04 and np.linalg.norm(tcp-tool)<.05 and tool[2]>self.initial_height+c.minimum_lift_m)
        active=bool(carry and c.stop_distance_m<distance<c.activation_range_m);self.correction=np.zeros(6);velocity=np.zeros(3);angular=np.zeros(3)
        waypoint=self.goal.copy();route_clearance=np.nan
        if not carry:self.transport_committed=False
        if active and c.staged_route:
            from tabletop_route_preference import staged_guidance
            wrist=d.body('link6_1').xpos.copy();axis=d.body('real_screwdriver').xmat.reshape(3,3)[:,1]
            if getattr(self,'proxy_mode','enclosing')=='components':
                from tabletop_tool_proxy import world_tool_capsules,world_body_capsules
                proxies=world_body_capsules(d,self.eef_capsules_local)+world_tool_capsules(d,a.target_body,self.tool_capsules_local)
                capsules=[(c['p1'],c['p2'],c['r']) for c in proxies]
            else:
                capsules=[(wrist,tcp,.07),(tool+self.tool_extent[0]*axis,tool+self.tool_extent[1]*axis,self.tool_radius)]
            proposed_phase,waypoint,route_clearance=staged_guidance(capsules,center,self.goal,self.obstacle,self.near_direction,self.phase,check_descent=c.check_descent)
            if c.commit_transport and (self.transport_committed or proposed_phase=='transport'):
                self.transport_committed=True;self.phase='transport'
                waypoint=self.goal.copy()
                if np.linalg.norm(center[:2]-self.goal[:2])>c.stop_distance_m:waypoint[2]=max(center[2],self.goal[2])
            else:self.phase=proposed_phase
        elif not active:
            self.phase='transport' if carry and c.commit_transport and self.transport_committed else 'inactive'
        else:
            self.phase='transport'
        if active:
            guidance_error=waypoint-center
            guidance_direction=guidance_error/max(np.linalg.norm(guidance_error),1e-12)
            # Bounded inverse distance: nearer targets receive stronger attraction.
            velocity=guidance_direction*(c.maximum_speed_m_s*(c.stop_distance_m/distance)**c.decay_power)
            J=np.zeros((3,a.model.nv));Jr=J.copy();a.mj.mj_jacSite(a.model,d,J,Jr,a.model.site('tcp_site').id)
            wrist_J=np.zeros_like(J);a.mj.mj_jac(a.model,d,wrist_J,None,d.body('link6_1').xpos,a.model.body('link6_1').id)
            J=((J+wrist_J)/2.)[:,a.dadr];Jr=Jr[:,a.dadr]
            _,sv,vectors=np.linalg.svd(Jr,full_matrices=True);basis=vectors[int((sv>1e-9).sum()):].T if c.preserve_orientation else np.eye(6);reduced=J@basis
            self.correction=basis@reduced.T@np.linalg.solve(reduced@reduced.T+.05**2*np.eye(3),velocity*self.dt)
            scale=1.
            for j,value in enumerate(self.correction):
                if value>0:scale=min(scale,(a.limits[j,1]-nominal[j])/value)
                elif value<0:scale=min(scale,(a.limits[j,0]-nominal[j])/value)
            self.correction*=max(0.,scale);angular=Jr@self.correction
        out=nominal.copy();out[:6]=np.clip(nominal[:6]+self.correction,a.limits[:,0],a.limits[:,1])
        orientation_error=0.;orientation_rejected=False
        if active and c.preserve_orientation:
            out[:6],orientation_error,orientation_rejected=self.preserve_target_orientation(nominal[:6],out[:6])
            self.correction=out[:6]-nominal[:6];angular=Jr@self.correction
        elif active:
            pose=self.pose_scratch;pose.qpos[:]=a.data.qpos;pose.qpos[a.qadr]=nominal[:6]
            a.mj.mj_kinematics(a.model,pose);reference=pose.site('tcp_site').xmat.reshape(3,3).copy()
            pose.qpos[a.qadr]=out[:6];a.mj.mj_kinematics(a.model,pose)
            orientation_error=float(np.linalg.norm(Rotation.from_matrix(reference.T@pose.site('tcp_site').xmat.reshape(3,3)).as_rotvec()))
        self.rows.append(dict(phase=self.phase,waypoint_m=waypoint.copy(),route_min_clearance_m=route_clearance,orientation_error_rad=orientation_error,orientation_rejected=orientation_rejected,action_index=int(index),active=active,carry_gate=carry,distance_m=distance,tcp_m=tcp,capsule_center_m=center.copy(),goal_m=self.goal.copy(),velocity_m_s=velocity,joint_correction_rad=self.correction.copy(),angular_correction_rad=angular,policy_target=nominal,attracted_target=out.copy()))
        return out
    def guide_chunk(self,chunk):
        guided=np.asarray(chunk,dtype=float).copy();guided[:,:6]=np.clip(guided[:,:6],self.arm.limits[:,0],self.arm.limits[:,1]);nominal=guided[:,:6].copy();guided[:,:6]=np.clip(nominal+self.correction,self.arm.limits[:,0],self.arm.limits[:,1])
        if np.any(self.correction) and self.config.preserve_orientation:
            for i in range(len(guided)):
                guided[i,:6],_,_=self.preserve_target_orientation(nominal[i],guided[i,:6])
        return guided
    def arrays(self):
        shapes=dict(phase=(),waypoint_m=(3,),route_min_clearance_m=(),orientation_error_rad=(),orientation_rejected=(),action_index=(),active=(),carry_gate=(),distance_m=(),tcp_m=(3,),capsule_center_m=(3,),goal_m=(3,),velocity_m_s=(3,),joint_correction_rad=(6,),angular_correction_rad=(3,),policy_target=(7,),attracted_target=(7,))
        return {k:np.asarray([r[k] for r in self.rows],dtype='U16' if k=='phase' else bool if k in ('active','carry_gate','orientation_rejected') else int if k=='action_index' else float).reshape((-1,*shape)) for k,shape in shapes.items()}
    def metadata(self):
        return dict(algorithm='tabletop_box_attraction_v6_staged_lift_transport' if self.config.staged_route else 'tabletop_box_attraction_v5_verified_nominal_orientation',staged_route=dict(enabled=self.config.staged_route,transition='full translated wrist/tool capsule across-then-descend path >=1.5cm to enter transport, >=1cm to retain',lift_direction='up 3 : near base 1',transport_direction='toward goal XY at flight height, then descend; no outward rejection',scope='sampled static proxy geometry; no global feasibility or grasp certificate'),orientation_preservation=dict(enabled=self.config.preserve_orientation,reference='nominal policy TCP target orientation; policy rotations pass through',tolerance_rad=1e-6,scope='attraction output and guided preview targets only; downstream F2/guard and physical tracking are separate',fallback='zero attraction correction if projection fails'),scope='explicit task-goal attraction in addition to pi0 and PACE; not autonomous-policy-only success',goal_m=self.goal.tolist(),config=asdict(self.config),distance_law='speed = maximum_speed * (stop_distance / distance) ** decay_power for stop_distance < distance < activation_range; zero in stop zone or outside range',active_actions=sum(r['active'] for r in self.rows),gripper='policy passthrough; no release decision',return_home='no guidance after policy opens gripper',mapping=('linear angular-nullspace DLS' if self.config.preserve_orientation else 'unconstrained translational DLS, orientation ablation')+' at midpoint of link6 origin and tcp_site; midpoint Jacobian is average of endpoint Jacobians; bounded nonlinear orientation projection about nominal policy target; no historical path',priority='attraction before F2 avoidance, admittance and final obstacle reference guard')
