"""Numerically reduced but mathematically equivalent NEO QP, with original-space validation."""
import numpy as np
import osqp
from scipy import sparse

def neo_qp(J, reference_twist, manipulability_gradient, task_error, A, b, lower, upper, config):
    c=config
    J=np.asarray(J,float); t=np.asarray(reference_twist,float); n=J.shape[1]
    g=np.asarray(manipulability_gradient,float); A=np.asarray(A,float).reshape(-1,n); b=np.asarray(b,float)
    lower=np.asarray(lower,float); upper=np.asarray(upper,float)
    w=1/max(c.error_floor,float(task_error))
    # delta = t - Jv; substitution into the original QP preserves objective and feasible set.
    P=c.joint_velocity_weight*np.eye(n)+w*(J.T@J)
    q=-g-w*(J.T@t)
    matrix=sparse.csc_matrix(np.vstack([A,np.eye(n),J]))
    lo=np.r_[np.full(len(b),-np.inf),lower,t-c.slack_limit]
    hi=np.r_[b,upper,t+c.slack_limit]
    solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(P),q=q,A=matrix,l=lo,u=hi,
        verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,polishing=True,warm_starting=False)
    result=solver.solve(raise_error=False)
    v=np.full(n,np.nan) if result.x is None else np.asarray(result.x,float)
    delta=t-J@v
    tol=c.feasibility_tolerance
    eqerr=float(np.max(np.abs(J@v+delta-t)))
    residual=float(np.min(b-A@v)) if len(b) else float('inf')
    feasible=bool(result.info.status_val in (1,2) and np.isfinite(v).all() and np.isfinite(delta).all()
        and eqerr<=tol and residual>=-tol and np.all(v>=lower-tol) and np.all(v<=upper+tol)
        and np.all(delta>=-c.slack_limit-tol) and np.all(delta<=c.slack_limit+tol))
    info=dict(status=result.info.status,status_val=int(result.info.status_val),iterations=int(result.info.iter),
      equality_max_abs_error=eqerr,inequality_min_residual=residual,
      objective=float(.5*c.joint_velocity_weight*(v@v)+.5*w*(delta@delta)-g@v),
      primal_residual=float(result.info.prim_res),dual_residual=float(result.info.dual_res),
      solver_form='eliminate_slack_equivalent',solver_eps_abs=1e-8,solver_eps_rel=1e-8)
    return v,delta,feasible,info
