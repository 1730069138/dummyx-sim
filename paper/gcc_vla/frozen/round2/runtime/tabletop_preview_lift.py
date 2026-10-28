"""Private nonlinear FK projection for a short, pose-preserving lift waypoint."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def lift_waypoint_target(controller,nominal,direction,step_m,*,hold_current_orientation=False):
    a=controller.arm;m=a.model;d=controller.scratch
    base=a.data.qpos.copy();nominal=np.asarray(nominal,float)
    def fk(q):
        d.qpos[:]=base;d.qpos[a.qadr]=q
        a.mj.mj_kinematics(m,d)
        center=(d.body('link6_1').xpos+d.site('tcp_site').xpos)/2.
        return center.copy(),d.site('tcp_site').xmat.reshape(3,3).copy()
    current,current_rotation=fk(base[a.qadr]);_,reference=fk(nominal[:6]);policy_reference=reference.copy();reference=current_rotation if hold_current_orientation else reference;waypoint=current+step_m*np.asarray(direction,float)
    def residual(q):
        center,R=fk(q)
        return np.r_[center-waypoint,Rotation.from_matrix(reference.T@R).as_rotvec()]
    result=minimize(lambda q:.5*float((q-nominal[:6])@(q-nominal[:6])),base[a.qadr].copy(),jac=lambda q:q-nominal[:6],bounds=list(map(tuple,a.limits)),constraints=[dict(type='eq',fun=residual)],method='SLSQP',options=dict(maxiter=35,ftol=1e-12))
    errors=residual(result.x);position_error=float(np.linalg.norm(errors[:3]));orientation_error=float(np.linalg.norm(errors[3:]));feasible=bool(np.isfinite(result.x).all() and position_error<=1e-5 and orientation_error<=1e-6)
    output=nominal.copy()
    if feasible:output[:6]=result.x
    return output,dict(feasible=feasible,position_error_m=position_error,orientation_error_rad=orientation_error,waypoint_m=waypoint,nominal_orientation_delta_rad=Rotation.from_matrix(policy_reference.T@fk(result.x)[1]).as_rotvec())
