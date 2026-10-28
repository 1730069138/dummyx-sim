"""Independent NEO control-core adaptation for a frozen joint-target VLA.

Official RTB example / Robot helpers: 9f282d8ff952a6ca95d340ae6a62b04e545a302f.
Preserves task slack, manipulability objective, joint and collision dampers.
Shared geometry only; never invokes a PACE process/project/C/guard method.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import osqp
from scipy import sparse
from tabletop_tool_proxy import batch_capsule_clearances

OFFICIAL_COMMIT = '9f282d8ff952a6ca95d340ae6a62b04e545a302f'

@dataclass(frozen=True)
class NEOAdapterConfig:
    horizon_s: float = .05
    joint_velocity_weight: float = .01
    joint_stop_rad: float = .05
    joint_influence_rad: float = .9
    damper_gain: float = 1.
    obstacle_stop_m: float = .05
    obstacle_influence_m: float = .3
    joint_speed_limit_rad_s: float = 5.
    slack_limit: float = 10.
    error_floor: float = 1e-6
    manipulability_difference_rad: float = 1e-5
    feasibility_tolerance: float = 2e-7

    def __post_init__(self):
        for key in ('horizon_s', 'joint_velocity_weight', 'damper_gain',
                    'joint_speed_limit_rad_s', 'slack_limit', 'error_floor',
                    'manipulability_difference_rad', 'feasibility_tolerance'):
            if not np.isfinite(getattr(self,key)) or getattr(self,key) <= 0:
                raise ValueError('Expected finite positive '+key)
        if not 0 <= self.joint_stop_rad < self.joint_influence_rad:
            raise ValueError('Invalid joint damper distances')
        if not 0 <= self.obstacle_stop_m < self.obstacle_influence_m:
            raise ValueError('Invalid obstacle damper distances')


def joint_damper(q, limits, stop=.05, influence=.9, gain=1.):
    """A qdot <= b; retain both limit rows if their influence zones overlap."""
    q, limits = np.asarray(q,float), np.asarray(limits,float)
    A, b = [], []
    for k in range(len(q)):
        dlo, dhi = q[k]-limits[k,0], limits[k,1]-q[k]
        for distance, direction in ((dlo,-1.),(dhi,1.)):
            if distance <= influence:
                row=np.zeros(len(q));row[k]=direction
                A.append(row);b.append(gain*(distance-stop)/(influence-stop))
    return np.asarray(A).reshape(-1,len(q)), np.asarray(b)


def obstacle_damper(gaps, distance_jacobians, stop=.05, influence=.3, gain=1.):
    """Stationary shared obstacle; gradients point toward increasing clearance."""
    selected=np.asarray(gaps) <= influence
    return (-np.asarray(distance_jacobians)[selected],
            gain*(np.asarray(gaps)[selected]-stop)/(influence-stop), selected)


def neo_qp(J, reference_twist, manipulability_gradient, task_error, A, b,
           lower, upper, config=None):
    """Solve official-form [qdot,delta] QP, J qdot + delta = twist."""
    c=config or NEOAdapterConfig();J=np.asarray(J,float);n=J.shape[1]
    Q=np.diag(np.r_[np.full(n,c.joint_velocity_weight),
                    np.full(6,1/max(c.error_floor,float(task_error)))])
    linear=np.r_[-np.asarray(manipulability_gradient,float),np.zeros(6)]
    eq=np.c_[J,np.eye(6)]
    inequality=np.c_[np.asarray(A,float).reshape(-1,n),np.zeros((len(b),6))]
    lower=np.r_[np.asarray(lower,float),np.full(6,-c.slack_limit)]
    upper=np.r_[np.asarray(upper,float),np.full(6,c.slack_limit)]
    matrix=sparse.csc_matrix(np.vstack([eq,inequality,np.eye(n+6)]))
    solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(Q),q=linear,A=matrix,
        l=np.r_[reference_twist,np.full(len(b),-np.inf),lower],
        u=np.r_[reference_twist,b,upper],verbose=False,eps_abs=1e-9,eps_rel=1e-9,
        max_iter=20000,polishing=True,warm_starting=False)
    result=solver.solve(raise_error=False)
    x=np.full(n+6,np.nan) if result.x is None else np.asarray(result.x,float)
    tol=c.feasibility_tolerance
    eqerr=float(np.max(np.abs(eq@x-reference_twist)))
    residual=float(np.min(np.asarray(b)-inequality@x)) if len(b) else float('inf')
    feasible=bool(result.info.status_val in (1,2) and np.isfinite(x).all()
        and eqerr<=tol and residual>=-tol and np.all(x>=lower-tol)
        and np.all(x<=upper+tol))
    info=dict(status=result.info.status,status_val=int(result.info.status_val),
        iterations=int(result.info.iter),equality_max_abs_error=eqerr,
        inequality_min_residual=residual,objective=float(.5*x@Q@x+linear@x),
        primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res))
    return x[:n],x[n:],feasible,info


from neo_solver import neo_qp

class NEOIndependent:
    def __init__(self, helper, config=None):
        self.helper=helper;self.arm=helper.arm;self.obstacle=helper.obstacle
        self.config=config or NEOAdapterConfig()
        self.current_guard=True;self.refresh_pending=False;self.execution_predictor=None
        self.rows=[];self.failures=[];self.calls=0;self.active_calls=0
        self.solver_status_counts={};self._scratch=self.arm.mj.MjData(self.arm.model)
        m=self.arm.model
        joint_ids=np.asarray(m.actuator_trnid[:6,0],int)
        if not np.array_equal(m.jnt_qposadr[joint_ids],self.arm.qadr):
            raise ValueError('Actuator/joint mapping mismatch')
        kp=np.asarray(m.actuator_gainprm[:6,0],float)
        bias=np.asarray(m.actuator_biasprm[:6,:3],float)
        gears=np.asarray(m.actuator_gear[:6],float)
        if (not np.allclose(gears[:,0],1) or not np.allclose(gears[:,1:],0)
            or not np.allclose(bias[:,0],0) or not np.allclose(bias[:,1],-kp)
            or np.any(kp<=0) or np.any(bias[:,2]>=0)):
            raise ValueError('Expected unit-gear linear position actuators with damping')
        self.velocity_servo_time_s=-bias[:,2]/kp

    def capsules(self,data,tool_local=None):
        return self.helper.capsules(data,tool_local)

    def _geometry(self,target):
        h=self.helper;h.prepare(h.current)
        attached=bool(target[6]<=.04 and np.linalg.norm(
            h.current.site_xpos[h.tcp_id]-h.current.xpos[h.tool_id])<.05)
        local=h.carried_tool_local(h.current) if attached else None
        gaps,normals,points=batch_capsule_clearances(h.capsules(h.current,local),h.obstacle)
        bodies=[int(c['body_id']) for c in h.eef_capsules_local]
        bodies += [h.wrist_id]*(len(gaps)-len(bodies))
        return gaps,normals,points,bodies,attached

    def _jacobian(self,data):
        a=self.arm;Jp=np.zeros((3,a.model.nv));Jr=np.zeros_like(Jp)
        a.mj.mj_jacSite(a.model,data,Jp,Jr,self.helper.tcp_id)
        return np.vstack([Jp[:,a.dadr],Jr[:,a.dadr]])

    def _manipulability_gradient(self):
        # Current official neo.py calls jacobm() with default axes='all'.
        a=self.arm;h=self.helper;eps=self.config.manipulability_difference_rad
        gradient=[]
        for address in a.qadr:
            values=[]
            for sign in (-1.,1.):
                h.prepare(self._scratch);self._scratch.qpos[address]+=sign*eps
                a.mj.mj_kinematics(a.model,self._scratch);a.mj.mj_comPos(a.model,self._scratch)
                values.append(float(np.prod(np.linalg.svd(self._jacobian(self._scratch),compute_uv=False))))
            gradient.append((values[1]-values[0])/(2*eps))
        return np.asarray(gradient)

    def process(self,chunk,index,**kwargs):
        target=np.asarray(chunk[index],float).copy()
        target[:6]=np.clip(target[:6],self.arm.limits[:,0],self.arm.limits[:,1])
        target[6]=np.clip(target[6],0.,.08)
        gaps,*_=self._geometry(target)
        row=dict(action_index=len(self.rows),current_clearance_m=float(np.min(gaps)),
            deficit_m=0.,geometry_active=False,geometry_target=target.copy())
        self.rows.append(row);return target,row

    def project_current_target(self,target):
        started=time.perf_counter_ns();a=self.arm;c=self.config;h=self.helper
        target=np.asarray(target,float).copy()
        if target.shape!=(7,) or not np.isfinite(target).all():raise ValueError('Expected finite 7D target')
        target[:6]=np.clip(target[:6],a.limits[:,0],a.limits[:,1])
        target[6]=np.clip(target[6],0.,.08)
        q=a.data.qpos[a.qadr].copy();gaps,normals,points,bodies,attached=self._geometry(target)
        J=self._jacobian(h.current);distance_J=[]
        for normal,point,body in zip(normals,points,bodies):
            jp=np.zeros((3,a.model.nv));a.mj.mj_jac(a.model,h.current,jp,None,point,body)
            distance_J.append(normal@jp[:,a.dadr])
        distance_J=np.asarray(distance_J)
        Aj,bj=joint_damper(q,a.limits,c.joint_stop_rad,c.joint_influence_rad,c.damper_gain)
        Ao,bo,selected=obstacle_damper(gaps,distance_J,c.obstacle_stop_m,c.obstacle_influence_m,c.damper_gain)
        A=np.vstack([Aj,Ao]);b=np.r_[bj,bo]
        nominal=(target[:6]-q)/c.horizon_s;reference=J@nominal
        gradient=self._manipulability_gradient()+c.joint_velocity_weight*nominal
        error=float(np.sum(np.abs(reference*c.horizon_s)))
        lower=np.maximum(-c.joint_speed_limit_rad_s,(a.limits[:,0]-q)/c.horizon_s)
        upper=np.minimum(c.joint_speed_limit_rad_s,(a.limits[:,1]-q)/c.horizon_s)
        velocity,slack,feasible,info=neo_qp(J,reference,gradient,error,A,b,lower,upper,c)
        output=target.copy()
        if feasible:
            # u=q+(kv/kp)*qdot gives actuator force kv*(qdot-qdot_actual)
            # before control/force saturation, with current model's existing kp/kv.
            output[:6]=np.clip(q+self.velocity_servo_time_s*velocity+a.data.qfrc_bias[a.dadr]/a.model.actuator_gainprm[:6,0],
                               a.limits[:,0],a.limits[:,1])
        active=bool(feasible and np.max(np.abs(output[:6]-target[:6]))>1e-9)
        self.calls+=1;self.active_calls+=int(active)
        self.solver_status_counts[info['status']]=self.solver_status_counts.get(info['status'],0)+1
        if not feasible:self.failures.append(dict(time_s=float(a.data.time),**info))
        nearest=int(np.argmin(gaps));bound=-c.damper_gain*(gaps[nearest]-c.obstacle_stop_m)/(c.obstacle_influence_m-c.obstacle_stop_m)
        debug=dict(post_contact_target=target.copy(),reference_guard_active=active,
            reference_guard_qualified=True,reference_guard_feasible=feasible,
            reference_guard_clearance_m=float(gaps[nearest]),reference_guard_bound_m=float(bound*c.horizon_s),
            reference_guard_before_m=float(distance_J[nearest]@(target[:6]-q)),
            reference_guard_after_m=float(distance_J[nearest]@(output[:6]-q)),
            reference_guard_joint_delta_rad=output[:6]-target[:6],
            route_preference_qualified=False,route_projection_active=False,route_near_endpoint_m=0.,
            protected_component_count=len(gaps),constrained_component_count=int(selected.sum()),
            attached_tool=attached,neo_task_slack=slack,neo_manipulability_gradient=gradient,
            neo_joint_damper_rows=len(bj),neo_obstacle_damper_rows=len(bo),
            neo_reference_twist=reference,neo_task_error=error,
            neo_joint_velocity_command=velocity.copy(),
            neo_velocity_servo_time_s=self.velocity_servo_time_s.copy(),
            minimum_constraint_residual_m=float(info['inequality_min_residual']*c.horizon_s),
            neo_qp=info,solver_compute_ms=(time.perf_counter_ns()-started)/1e6)
        return output,debug

    def arrays(self):
        return {k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]} if self.rows else {}

    def metadata(self):
        return dict(algorithm='NEO_pi0_nominal_centered_velocity_servo_bias_compensated',
            official_repository='https://github.com/petercorke/robotics-toolbox-python',
            official_commit=OFFICIAL_COMMIT,config=asdict(self.config),
            objective='Y=.01 (qdot-qdot_nom) norm + error-scaled task slack norm - full-pose Yoshikawa gradient dot qdot',
            reference='J(q)*(clipped current pi0 joint target-q)/horizon; no independent goal or future action',
            interface='qtarget=clip(q+(kv/kp)*qdot+qfrc_bias/kp); model-specific kv/kp, per-substep; subject to actuator and joint saturation',
            manipulability='current official example axes=all; numerical central difference on isolated MuJoCo scratch data',
            shared_geometry='DINO/RGB-D obstacle and PACE component capsules only; actual closest-point Jacobian; simulator-assisted rigid tool pose',
            independent=True,PACE_G_called=False,PACE_C_called=False,PACE_guard_called=False,
            bounds='explicit adapter speed cap, legal position horizon bounds, official slack +/-10',
            fallback='infeasible solve returns feasible=False; executor must abort before submission; no certified stop claim',
            differences=['MuJoCo replaces RTB kinematics/Hessian/collision geometry',
                'stationary perceived obstacle, no dynamic obstacle velocity input',
                'instantaneous pi0 reference replaces independent fixed pose p_servo goal',
                'joint velocity regularizer centered on current pi0 nominal velocity instead of zero; adapter change',
                'physical twist displacement L1 error with floor replaces example Euler-scaled error',
                'finite-difference manipulability replaces analytic RTB Hessian',
                'both joint limit damper rows retained in overlapping influence zones',
                'velocity servo force via existing position actuator; explicit 5 rad/s speed limit; force and position saturation can affect tracking',
                'verified infeasible reference abort replaces absence of a qualified native fallback'],
            limitations='not full paper benchmark reproduction or continuous-time safety; no free-tool control',
            calls=self.calls,active_calls=self.active_calls,solver_status_counts=self.solver_status_counts,
            failures=self.failures,geometry_helper_metadata=self.helper.metadata())
