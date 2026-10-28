"""Explicit local up/near-base preference, not a global route planner."""
import numpy as np

def near_base_direction(base, obstacle):
    direction=np.asarray(base,dtype=float)[:2]-np.asarray(obstacle,dtype=float)[:2]
    length=np.linalg.norm(direction)
    if length<1e-9:raise ValueError('Base/pillar XY cannot define a near side')
    return np.r_[direction/length,0.]

def up_near_direction(base, obstacle):
    near=near_base_direction(base,obstacle)
    direction=3.*np.array([0.,0.,1.])+near
    return direction/np.linalg.norm(direction)

def bounded_multi_projection(rows,angular,required,lower,upper):
    """Minimum-norm correction in angular nullspace with several inequalities."""
    from scipy.optimize import minimize
    rows=np.asarray(rows,float);required=np.asarray(required,float)
    _,s,V=np.linalg.svd(angular,full_matrices=True);N=V[(s>1e-9).sum():].T
    A=np.vstack([N,-N,rows@N]);b=np.r_[lower,-upper,required]
    result=minimize(lambda y:.5*float(y@y),np.zeros(N.shape[1]),jac=lambda y:y,
        constraints=[dict(type='ineq',fun=lambda y:A@y-b,jac=lambda y:A)],method='SLSQP',options=dict(maxiter=60,ftol=1e-12))
    correction=N@result.x
    feasible=bool(np.isfinite(correction).all() and np.all(A@result.x>=b-1e-9))
    return correction,feasible

def staged_guidance(capsules, center, goal, obstacle, near, previous_phase='lift',check_descent=True):
    """Check crossing; optionally include the full later descent.

    Crossing-only mode leaves descent to the current-state guard after transport.
    This translated frozen-pose geometry is not a continuous safety certificate.
    """
    from tabletop_pace_runtime import segment_to_segment_distance
    center=np.asarray(center,float);goal=np.asarray(goal,float)
    across=goal.copy();across[2]=max(center[2],goal[2])
    minimum=np.inf
    legs=((center,across),(across,goal)) if check_descent else ((center,across),)
    from tabletop_tool_proxy import batch_segment_distances
    p1=np.array([c[0] for c in capsules]);p2=np.array([c[1] for c in capsules]);radii=np.array([c[2] for c in capsules])
    for start,end in legs:
        count=max(2,int(np.ceil(np.linalg.norm(end-start)/.003))+1)
        shifts=start+np.linspace(0.,1.,count)[:,None]*(end-start)-center
        first=(p1[None,:,:]+shifts[:,None,:]).reshape(-1,3);second=(p2[None,:,:]+shifts[:,None,:]).reshape(-1,3)
        distances,_,_,_=batch_segment_distances(first,second,obstacle['p1'],obstacle['p2'])
        minimum=min(minimum,float(np.min(distances-np.tile(radii,count)-obstacle['r'])))
    threshold=.01 if previous_phase=='transport' else .015
    if minimum>=threshold:
        waypoint=goal if np.linalg.norm(center[:2]-goal[:2])<=.035 else across
        return 'transport',waypoint.copy(),minimum
    # Lift is the principal component; a smaller base-side retreat avoids pure vertical IK.
    return 'lift',center+.09*np.array([0.,0.,1.])+.03*np.asarray(near),minimum
