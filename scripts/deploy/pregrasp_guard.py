"""Opt-in pregrasp reference guard. Kinematics only; no safety certificate.

Four separate mesh-derived oriented boxes cover wrist, palm and fingers. Finger
boxes sweep the measured-to-commanded opening interval. A single arm-target bias
parameterizes the next <=8 reference steps; only the first target is returned.
This predicts reference kinematics, not physical tracking/contact dynamics.
"""
import time
import numpy as np
import mujoco

DEFAULTS = dict(pregrasp_guard='off', pregrasp_geometry='split',
                pregrasp_preserve='on', pregrasp_clearance=.005,
                pregrasp_tcp_tolerance=.005, pregrasp_axis_tolerance_deg=5.)


def add_arguments(parser):
    parser.add_argument('--pregrasp_guard', choices=('off', 'shadow', 'on'), default='off')
    parser.add_argument('--pregrasp_geometry', choices=('split', 'coarse'), default='split')
    parser.add_argument('--pregrasp_preserve', choices=('on', 'off'), default='on')
    parser.add_argument('--pregrasp_clearance', type=float, default=.005)
    parser.add_argument('--pregrasp_tcp_tolerance', type=float, default=.005)
    parser.add_argument('--pregrasp_axis_tolerance_deg', type=float, default=5.)


def validate_arguments(args):
    for key in ('pregrasp_clearance','pregrasp_tcp_tolerance','pregrasp_axis_tolerance_deg'):
        if not np.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            raise ValueError(key+' must be finite and positive')
    if args.pregrasp_guard == 'on':
        raise ValueError('Pregrasp actuation is not validated; only off/shadow are available at runtime')
    if args.pregrasp_guard != 'off':
        required = dict(case=8, geometry_method='pace_preview', contact_gate='phase',
                        experiment_protocol='controlled', initial_state_protocol='full_v1',
                        research_method='off', diagnostics=False, obstacle_source='groundingdino',
                        joint_filter='on',preview_stop=8,consistency_sched='off',dir_select='off',
                        requery_on_intervene='off',control_mode='admittance')
        if any(getattr(args,k) != value for k,value in required.items()):
            raise ValueError('Pregrasp prototype requires F2 Case8/pace_preview/phase/controlled/full_v1/GroundingDINO without research/diagnostics')


def suffix(args):
    if args.pregrasp_guard == 'off':
        return ''
    import hashlib, json
    values={key:getattr(args,key) for key in DEFAULTS}
    tag=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()[:10]
    return '_pg-'+args.pregrasp_guard+'-'+tag


def segment_box_clearance(p1, p2, center, rotation, half, radius):
    """Minimum box SDF along a capsule axis, less radius.

    Positive values are exact separation for this box/capsule pair. Negative
    values are a conservative overlap surrogate, not penetration depth. Analytic
    piecewise candidates include face crossings and inside max-affine crossings.
    """
    a=rotation.T@(np.asarray(p1)-center)
    v=rotation.T@(np.asarray(p2)-np.asarray(p1))
    half=np.asarray(half)
    cuts=[0.,1.]
    for j in range(3):
        if abs(v[j])>1e-14:
            cuts.extend(float(t) for t in ((-half[j]-a[j])/v[j],(half[j]-a[j])/v[j]) if 0<t<1)
    cuts=sorted(set(cuts)); candidates=list(cuts)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        mid=a+v*((lo+hi)/2)
        active=np.abs(mid)>half
        if np.any(active):
            signs=np.sign(mid[active]); slope=signs*v[active]; offset=signs*a[active]-half[active]
            denominator=slope@slope
            if denominator>1e-20:
                candidates.append(float(np.clip(-(slope@offset)/denominator,lo,hi)))
    slopes=np.r_[v,-v]; offsets=np.r_[a-half,-a-half]
    for i in range(6):
        for j in range(i):
            if abs(slopes[i]-slopes[j])>1e-14:
                t=(offsets[j]-offsets[i])/(slopes[i]-slopes[j])
                if 0<t<1: candidates.append(float(t))
    points=a[None,:]+np.asarray(candidates)[:,None]*v
    q=np.abs(points)-half
    sdf=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
    return float(sdf.min()-radius)


class HandGeometry:
    BODIES=('link6_1','link7','link8','link9')

    def __init__(self, model):
        self.model=model; self.parts=[]; seen=set()
        body_names={mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name):name for name in self.BODIES}
        for geom in range(model.ngeom):
            body=int(model.geom_bodyid[geom])
            if body not in body_names or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):continue
            if model.geom_type[geom] != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError('Prototype expects mesh hand collision geometry')
            signature=(body,int(model.geom_dataid[geom]),model.geom_pos[geom].tobytes(),model.geom_quat[geom].tobytes())
            if signature in seen:continue
            seen.add(signature)
            joint_name={'link8':'Joint8','link9':'Joint9'}.get(body_names[body])
            joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,joint_name) if joint_name else None
            self.parts.append(dict(geom=geom,body=body_names[body],joint=joint,
                                   center=model.geom_aabb[geom,:3].copy(),half=model.geom_aabb[geom,3:].copy()))
        if set(part['body'] for part in self.parts)!=set(self.BODIES):
            raise ValueError('Incomplete hand geometry')

    def boxes(self, data, opening_intervals=None):
        result=[]
        for part in self.parts:
            geom=part['geom'];rotation=data.geom_xmat[geom].reshape(3,3)
            center=data.geom_xpos[geom]+rotation@part['center'];half=part['half'].copy()
            if opening_intervals is not None and part['joint'] is not None:
                joint=part['joint'];lo,hi=opening_intervals[joint]
                actual=float(data.qpos[self.model.jnt_qposadr[joint]])
                axis=data.xaxis[joint]
                center=center+axis*((lo+hi)/2-actual)
                half=half+np.abs(rotation.T@axis)*(hi-lo)/2
            result.append(dict(body=part['body'],geom=geom,center=center,rotation=rotation.copy(),half=half))
        return result

    def clearances(self, data, obstacle, intervals=None):
        return np.array([segment_box_clearance(obstacle['p1'],obstacle['p2'],
                         box['center'],box['rotation'],box['half'],obstacle['r'])
                         for box in self.boxes(data,intervals)])


class PregraspGuard:
    def __init__(self, model, args):
        self.model,self.args=model,args
        self.hand=HandGeometry(model);self.scratch=mujoco.MjData(model)
        self.tcp=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'tcp_site')
        self.wrist=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'link6_1')
        self.jaw=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'Joint8')
        joints=model.actuator_trnid[:6,0]
        self.lower=np.maximum(model.jnt_range[joints,0],model.actuator_ctrlrange[:6,0])
        self.upper=np.minimum(model.jnt_range[joints,1],model.actuator_ctrlrange[:6,1])

    def inspect(self, qpos, obstacle, intervals=None):
        self.scratch.qpos[:]=qpos
        mujoco.mj_kinematics(self.model,self.scratch)
        if self.args.pregrasp_geometry=='split':
            distances=self.hand.clearances(self.scratch,obstacle,intervals)
        else:
            # Exact historical capsule distance, imported only on explicit use.
            from deploy_screwdriver_client import segment_to_segment_distance
            distance,*_=segment_to_segment_distance(self.scratch.xpos[self.wrist],self.scratch.site_xpos[self.tcp],
                                                    np.asarray(obstacle['p1']),np.asarray(obstacle['p2']))
            distances=np.array([distance-.07-obstacle['r']])
        return distances,self.scratch.site_xpos[self.tcp].copy(),self.scratch.xaxis[self.jaw].copy()

    def propose(self, data, command, chunk, index, filter_state, obstacle, *, phase_latched=False,
                opening_intervals=None):
        started=time.perf_counter_ns()
        if self.args.pregrasp_guard=='off':
            return command,dict(status='off',applied=False,compute_ms=0.)
        if phase_latched or obstacle is None:
            return command,dict(status='bypass_phase' if phase_latched else 'no_obstacle',applied=False,compute_ms=0.)
        if chunk is None or np.asarray(chunk).ndim!=2 or np.asarray(chunk).shape[1]<7:
            raise ValueError('Frozen action chunk with >=7 columns required')
        if not 0<=index<len(chunk):raise ValueError('Invalid chunk index')
        if np.asarray(command).shape!=(8,) or not np.isfinite(command).all() or not np.isfinite(chunk[index:min(len(chunk),8)]).all():
            raise ValueError('Nonfinite reference')
        if (np.asarray(obstacle['p1']).shape!=(3,) or np.asarray(obstacle['p2']).shape!=(3,)
                or not np.isfinite(np.r_[obstacle['p1'],obstacle['p2'],obstacle['r']]).all()
                or obstacle['r']<0):raise ValueError('Invalid perceived capsule')
        actions=np.asarray(chunk[index:min(len(chunk),8)],float)
        initial=data.qpos.copy();nominal=np.asarray(command[:6]).copy()
        if opening_intervals is None:
            targets=[0.04 if action[6]>.02 else 0. for action in actions]
            opening_intervals={}
            for joint_name in ('Joint8','Joint9'):
                joint=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_name)
                actual=float(initial[self.model.jnt_qposadr[joint]])
                opening_intervals[joint]=(min(actual,*targets),max(actual,*targets))
        if any(not np.isfinite([lo,hi]).all() or lo>hi for lo,hi in opening_intervals.values()):
            raise ValueError('Invalid opening interval')
        current_distances,_,_=self.inspect(initial,obstacle,opening_intervals)
        def rollout(u):
            bias=u-nominal; q=initial.copy();state=np.array(filter_state,copy=True)
            previous=initial[:6].copy();distances=[];positions=[];axes=[]
            for k,action in enumerate(actions):
                if k==0:target=u
                else:
                    state=.6*(action[:6]-previous)+.4*state
                    target=np.clip(previous+np.clip(state,-.25,.25)+bias,
                                   np.maximum(self.lower,previous-.25),np.minimum(self.upper,previous+.25))
                # Sample reference midpoint and endpoint; no continuous-time claim.
                for arm in ((previous+target)/2,target):
                    q[:6]=arm
                    ds,p,a=self.inspect(q,obstacle,opening_intervals)
                    distances.extend(ds);positions.append(p);axes.append(a)
                previous=target.copy()
            return np.asarray(distances),np.asarray(positions),np.asarray(axes)
        baseline=rollout(nominal)
        record=dict(status='clear',applied=False,current_min_m=float(current_distances.min()),
                    nominal_min_m=float(baseline[0].min()),parts=[p['body'] for p in self.hand.parts],
                    horizon=len(actions),opening_intervals={str(k):list(v) for k,v in opening_intervals.items()},
                    nominal_target=nominal.tolist(),
                    status_scope='infeasible means this bounded local solver did not return a verified feasible candidate; not a global infeasibility proof.',
                    scope='Sampled reference kinematics; neither physical prediction nor continuous swept-arm safety.')
        clearance=self.args.pregrasp_clearance
        lower=np.maximum(self.lower,initial[:6]-.25);upper=np.minimum(self.upper,initial[:6]+.25)
        if np.any(nominal<lower-1e-9) or np.any(nominal>upper+1e-9):
            record.update(status='invalid_nominal_bounds',feasible=False)
        elif baseline[0].min()>=clearance:
            record.update(feasible=True,candidate_min_m=record['nominal_min_m'],tcp_deviation_m=0.,axis_deviation_deg=0.)
        else:
            from scipy.optimize import minimize
            cache={}
            def evaluated(u):
                key=np.asarray(u).tobytes()
                if key not in cache:cache[key]=rollout(u)
                return cache[key]
            def constraints(u):
                ds,p,a=evaluated(u);values=[ds-clearance]
                if self.args.pregrasp_preserve=='on':
                    values.append(self.args.pregrasp_tcp_tolerance-np.linalg.norm(p-baseline[1],axis=1))
                    values.append(np.einsum('ij,ij->i',a,baseline[2])-np.cos(np.deg2rad(self.args.pregrasp_axis_tolerance_deg)))
                return np.concatenate(values)
            opt=minimize(lambda u:.5*np.sum((u-nominal)**2),nominal,
                         jac=lambda u:u-nominal,bounds=list(zip(lower,upper)),
                         constraints=[dict(type='ineq',fun=constraints)],method='SLSQP',
                         options=dict(maxiter=25,ftol=1e-10))
            candidate=np.asarray(opt.x);ds,p,a=evaluated(candidate)
            feasible=bool(np.isfinite(candidate).all() and np.all(candidate>=lower-1e-9)
                          and np.all(candidate<=upper+1e-9) and constraints(candidate).min()>=-1e-7)
            record.update(status='candidate' if feasible else 'infeasible',feasible=feasible,
                          candidate_min_m=float(ds.min()),candidate_target=candidate.tolist(),
                          joint_correction_norm=float(np.linalg.norm(candidate-nominal)),
                          tcp_deviation_m=float(np.linalg.norm(p-baseline[1],axis=1).max()),
                          axis_deviation_deg=float(np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',a,baseline[2]),-1,1))).max()),
                          solver_message=str(opt.message),iterations=int(opt.nit))
            record['solver_success']=bool(opt.success)
            record['clearance_shortfall_m']=float(max(0.,clearance-ds.min()))
            record['tcp_excess_m']=float(max(0.,record['tcp_deviation_m']-self.args.pregrasp_tcp_tolerance))
            record['axis_excess_deg']=float(max(0.,record['axis_deviation_deg']-self.args.pregrasp_axis_tolerance_deg))
        record['compute_ms']=(time.perf_counter_ns()-started)/1e6
        if self.args.pregrasp_guard=='on' and record['status']=='candidate':
            result=command.copy();result[:6]=record['candidate_target'];record['applied']=True
            return result,record
        # Off/shadow never changes the command. An infeasible on-mode reference
        # must be rejected by the caller, rather than silently actuated.
        return command,record
